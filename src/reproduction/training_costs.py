"""Reconstruct observed cost from exported original QHFlow2 training events."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def summarize_run(rows, gpu_count, effective_batch, log_interval=50, session_gap=10):
    """Remove duplicate API rows and discard steps superseded by a later resume."""
    unique = {}
    for row in rows:
        key = row["history_step"]
        if key in unique and unique[key] != row:
            raise ValueError("Conflicting duplicate W&B history row")
        unique[key] = row
    ordered = sorted(unique.values(), key=lambda r: r["history_step"])
    if not ordered or gpu_count <= 0 or effective_batch <= 0:
        raise ValueError("Missing events or invalid batch/device count")
    sessions = [[]]
    for row in ordered:
        if sessions[-1]:
            prev = sessions[-1][-1]
            offset_change = (row["timestamp"] - row["runtime"]) - (prev["timestamp"] - prev["runtime"])
            if row["optimizer_step"] <= prev["optimizer_step"] or abs(offset_change) > session_gap:
                sessions.append([])
        sessions[-1].append(row)
    retained, segments = [], []
    for i, session in enumerate(sessions):
        # Later resumes supersede the discarded tail of earlier attempts.
        ceiling = min((s[0]["optimizer_step"] for s in sessions[i + 1:]), default=float("inf"))
        keep = [r for r in session if r["optimizer_step"] < ceiling]
        if not keep:
            continue
        seconds = keep[-1]["timestamp"] - keep[0]["timestamp"]
        if seconds < 0:
            raise ValueError("Nonmonotonic training timestamps")
        retained.append(keep)
        segments.append(dict(
            session=i, first_step=keep[0]["optimizer_step"], last_step=keep[-1]["optimizer_step"],
            first_history_step=keep[0]["history_step"], last_history_step=keep[-1]["history_step"],
            first_timestamp=keep[0]["timestamp"], last_timestamp=keep[-1]["timestamp"],
            gpu_hours=seconds * gpu_count / 3600, retained_events=len(keep),
        ))
    steps = [r["optimizer_step"] for session in retained for r in session]
    if any(b - a != log_interval for a, b in zip(steps, steps[1:])):
        raise ValueError("Missing or duplicated optimizer steps in retained history")
    windows = []
    for a, b in zip(retained[-1], retained[-1][1:]):
        if a["epoch"] == b["epoch"] and b["optimizer_step"] - a["optimizer_step"] == log_interval:
            elapsed = b["timestamp"] - a["timestamp"]
            if elapsed <= 0:
                raise ValueError("Nonpositive training window")
            windows.append(elapsed)
    if not windows:
        raise ValueError("No complete final-session training windows")
    samples = len(windows) * log_interval * effective_batch
    return dict(
        train_gpu_hours=sum(s["gpu_hours"] for s in segments),
        samples_per_second=samples / sum(windows),
        throughput_gpu_count=gpu_count, effective_batch=effective_batch,
        throughput_windows=len(windows), throughput_samples=samples, throughput_seconds=sum(windows),
        raw_events=len(rows), retained_events=len(steps), duplicate_api_events=len(rows) - len(unique),
        discarded_events=len(unique) - sum(len(s) for s in retained),
        first_logged_step=retained[0][0]["optimizer_step"], last_logged_step=retained[-1][-1]["optimizer_step"],
        segments=segments,
    )


def build_summary(history, runs):
    grouped = defaultdict(list)
    with gzip.open(history, "rt", newline="") as f:
        for row in csv.DictReader(f):
            for key in ("history_step", "optimizer_step", "epoch"):
                row[key] = int(row[key])
            for key in ("timestamp", "runtime"):
                row[key] = float(row[key])
            grouped[row["model"]].append(row)
    manifest = json.loads(Path(runs).read_text())
    if set(grouped) != set(manifest["models"]):
        raise ValueError("History and model manifest differ")
    result = dict(
        schema=1, source="W&B original training histories; later evaluation events excluded",
        history_sha256=hashlib.sha256(Path(history).read_bytes()).hexdigest(),
        runs_sha256=hashlib.sha256(Path(runs).read_bytes()).hexdigest(),
        gpu_hours_definition="Sum of first-to-last training-event wall spans in retained resume sessions times training GPU count. Includes validation/checkpointing inside spans; excludes startup, restart gaps, overwritten training tails and later evaluation. Approximate observed cost, not total allocated GPU time.",
        samples_per_second_definition="Processed molecules / elapsed time over consecutive 50-optimizer-step windows in the same epoch in the final retained training session; excludes epoch-boundary validation and startup. Effective batch 32.",
        session_rule="Split on optimizer-step rollback or a >10-second change in timestamp-minus-runtime; identical API duplicate rows count once",
        boundary_limit="The first 49 optimizer steps and intervals spanning restarts are not timed; endpoints are observed log events",
        hardware_note="Training GPU counts and microbatch/accumulation come from the published checkpoint configs. Current W&B hardware metadata was overwritten by later evaluations and cannot establish the original GPU model. These are not matched-hardware speedup measurements.",
        models={},
    )
    for name, config in manifest["models"].items():
        if any(r["run_id"] != config["run_id"] for r in grouped[name]):
            raise ValueError("W&B run identity mismatch")
        metrics = summarize_run(grouped[name], config["training_gpu_count"], config["effective_batch"])
        if metrics["first_logged_step"] != 49 or metrics["last_logged_step"] != 259999 or metrics["retained_events"] != 5200:
            raise ValueError("Incomplete original training history")
        result["models"][name] = {**config, **metrics}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_summary(args.history, args.runs), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
