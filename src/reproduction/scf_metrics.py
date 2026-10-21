"""Recompute SCF cycle ratios and count-based ERIC from traceable paired records.

This aggregates existing results; it does not launch SCF or infer wall-time speedup.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path


def aggregate_cycles(model_cycles, minao_cycles, initialization_cycles=1):
    """Use a ratio of sums for Figure 4 and a mean of paired ratios for ERIC.

    The +1 convention accounts for an initial Hamiltonian/Fock build. It is
    explicit, not a measurement of Fock builds or model inference time.
    """
    model, minao = list(model_cycles), list(minao_cycles)
    if not model or len(model) != len(minao):
        raise ValueError("SCF counts must be nonempty and paired")
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0
           for n in [*model, *minao, initialization_cycles]):
        raise ValueError("SCF counts and initialization overhead must be nonnegative integers")
    if any(n == 0 for n in minao):
        raise ValueError("Every paired MinAO cycle count must be positive")
    ratio = 100 * sum(model) / sum(minao)
    return dict(
        count=len(model), model_cycles_total=sum(model), minao_cycles_total=sum(minao),
        cycle_ratio_percent=ratio, cycle_reduction_percent=100 - ratio,
        eric=sum((n + initialization_cycles) / b for n, b in zip(model, minao)) / len(model),
        initialization_cycles=initialization_cycles,
    )


def summarize_recorded(cohort_path, cycles_path):
    """Require every model to contain exactly the explicit, fully converged cohort."""
    cohort_path, cycles_path = Path(cohort_path), Path(cycles_path)
    with cohort_path.open() as f:
        cohort = list(csv.DictReader(f))
    identities = {(int(r["prediction_index"]), int(r["dataset_index"])) for r in cohort}
    if not identities or len(identities) != len(cohort):
        raise ValueError("Empty or duplicate molecule cohort")
    for column in ("prediction_index", "dataset_index"):
        if len({int(r[column]) for r in cohort}) != len(cohort):
            raise ValueError(f"Duplicate {column}")
    groups = defaultdict(list)
    with cycles_path.open() as f:
        for row in csv.DictReader(f):
            groups[row["model"]].append(row)
    if not groups:
        raise ValueError("No SCF records")
    models, baseline = {}, {}
    for label, rows in sorted(groups.items()):
        observed = {(int(r["prediction_index"]), int(r["dataset_index"])) for r in rows}
        if len(rows) != len(identities) or observed != identities:
            raise ValueError(f"{label}: missing, duplicate or mismatched molecules")
        if any(r[k] != "True" for r in rows for k in ("pred_init_converged", "scratch_converged")):
            raise ValueError(f"{label}: nonconverged pair; headline metrics withheld")
        for row in rows:
            key = int(row["dataset_index"])
            n = int(row["scratch_cycles"])
            if key in baseline and baseline[key] != n:
                raise ValueError(f"{label}: inconsistent shared MinAO baseline")
            baseline[key] = n
        models[label] = aggregate_cycles(
            [int(r["pred_init_cycles"]) for r in rows],
            [int(r["scratch_cycles"]) for r in rows],
        )
    return dict(
        schema=1, cohort_count=len(identities),
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cohort_path, cycles_path)},
        cycle_ratio_definition="100 * sum(N_model) / sum(N_MINAO); lower is better",
        cycle_reduction_definition="100 - cycle_ratio_percent; higher is better",
        eric_definition="mean((N_model + 1) / N_MINAO); lower is better",
        eric_scope="Post-hoc count-based metric with one initial Hamiltonian build; no inference timing or directly measured Fock-build counts",
        provenance_note="Cohort attribution and solver settings are recorded separately in provenance.json",
        models=models,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cycles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_recorded(args.cohort, args.cycles)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
