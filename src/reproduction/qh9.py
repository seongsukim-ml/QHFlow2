"""Prepare traceable QH9 inputs and evaluate SCF or one-shot energy/forces.

Run with PYTHONPATH=src python -m reproduction.qh9 --help from the repo root.
The preparation step reads trusted local prediction .pt files; the evaluation
step reads only hashed NPZ files (allow_pickle=False).
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from .physics import build_mf, host, one_shot, orbitals, solve_scf
from .scf_metrics import aggregate_cycles


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def source_identity():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(git_commit=commit, python_files={str(p.relative_to(root)): sha(p) for p in sorted((root / "src").rglob("*.py"))})


def read_cohort(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty cohort")
    for r in rows:
        for k in ("prediction_index", "dataset_index"):
            r[k] = int(r[k])
    for key in ("prediction_index", "dataset_index"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {key}")
    return rows


def prepare(args):
    import torch
    from dataset_module.qh9_stable_readonly import QH9StableReadonly
    from common.matrix_transforms import matrix_transform_single
    rows = read_cohort(args.cohort)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = QH9StableReadonly(args.processed_directory, args.split_file,
                               compute_q_tensor=False, include_cut_orbital_coefficients=False)
    test_ids = set(dataset.test_mask.tolist())
    records = []
    try:
        for row in rows:
            idx, pred_id = row["dataset_index"], row["prediction_index"]
            if idx not in test_ids:
                raise ValueError(f"{idx} is not in the supplied test split")
            p = args.pred_dir / f"pred_{pred_id}.pt"
            # Only use prediction files you produced or otherwise trust.
            d = torch.load(p, map_location="cpu", weights_only=False)
            gt = dataset.get_gt_outputs(idx)
            atoms = np.asarray(d["atoms"]).reshape(-1)
            pos = np.asarray(d["pos"], dtype=np.float64)
            unit = d.get("length_unit")
            if unit == "bohr":
                from pyscf.lib.param import BOHR
                pos = pos * BOHR
            elif unit != "angstrom":
                raise ValueError(f"{p}: explicit length_unit required")
            if not np.array_equal(atoms, gt["atoms"].numpy().reshape(-1)) or not np.allclose(pos, gt["pos"], atol=2e-6, rtol=0):
                raise ValueError(f"{p}: geometry does not match dataset index {idx}")
            h = d["pred_hamiltonian"].to(torch.float64).squeeze()
            s = d["overlap"].to(torch.float64).squeeze()
            fmt = d.get("format")
            if fmt in {"e3nn", "e3nn_def2svp"}:
                h = matrix_transform_single(h, torch.tensor(atoms), "e3nn_to_pyscf_def2svp")
                s = matrix_transform_single(s, torch.tensor(atoms), "e3nn_to_pyscf_def2svp")
            elif fmt != "pyscf_def2svp":
                raise ValueError(f"{p}: unsupported/ambiguous AO format {fmt!r}")
            if not np.allclose(s.numpy(), gt["overlap_matrix"], atol=2e-6, rtol=0):
                raise ValueError(f"{p}: AO overlap differs from the reference")
            reference = gt["hamiltonian"].numpy()
            reference_meta = {"reference_source": "raw_dataset_float64"}
            if args.gt_dir:
                gt_path = args.gt_dir / f"gt_{pred_id}.pt"
                saved = torch.load(gt_path, map_location="cpu", weights_only=False)
                if saved.get("format") != fmt or saved.get("length_unit") != unit:
                    raise ValueError(f"{gt_path}: reference conventions differ from prediction")
                if not np.array_equal(saved["atoms"], d["atoms"]) or not np.allclose(saved["pos"], d["pos"], atol=2e-6, rtol=0):
                    raise ValueError(f"{gt_path}: reference geometry differs from prediction")
                reference = saved["hamiltonian"].to(torch.float64).squeeze()
                if fmt in {"e3nn", "e3nn_def2svp"}:
                    reference = matrix_transform_single(reference, torch.tensor(atoms), "e3nn_to_pyscf_def2svp")
                reference = reference.numpy()
                reference_meta = dict(reference_source=str(gt_path.resolve()), reference_sha256=sha(gt_path))
            filename = f"sample_{idx:06d}.npz"
            np.savez(output / filename, atoms=atoms, pos_ang=pos, H_pred=h.numpy(),
                     H_reference=reference, S=s.numpy())
            records.append({**row, "file": filename, "sha256": sha(output / filename),
                            "prediction_file": str(p.resolve()), "prediction_sha256": sha(p), **reference_meta})
    finally:
        dataset.close()
    write_json(output / "manifest.json", dict(schema=1, dataset="QH9Stable-random",
        matrix_convention="native_pyscf", position_unit="Angstrom", matrix_unit="Hartree",
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
        prediction_checkpoint_attribution="supplied by operator; verify against inference metadata",
        split_file=str(args.split_file.resolve()), split_sha256=sha(args.split_file),
        cohort_file=str(args.cohort.resolve()), cohort_sha256=sha(args.cohort), records=records))


def select(args):
    """Create a new named replay cohort, never relabel it as the paper cohort."""
    split = json.loads(args.split_file.read_text())
    ids = split["test"]
    count = args.count or len(ids)
    if count < 1 or count > len(ids):
        raise ValueError("count must be within the test split size")
    positions = sorted(np.random.default_rng(args.seed).choice(len(ids), count, replace=False))
    with args.output.open("x") as f:
        writer = csv.DictWriter(f, fieldnames=["prediction_index", "dataset_index", "molecule_id"])
        writer.writeheader()
        for position in positions:
            writer.writerow(dict(prediction_index=int(position), dataset_index=int(ids[position]), molecule_id=f"QH9Stable:{ids[position]}"))
    write_json(args.output.with_suffix(".provenance.json"), dict(status="new_replay_cohort", seed=args.seed,
               count=count, split_sha256=sha(args.split_file), cohort_sha256=sha(args.output)))


def infer(args):
    """Export native-AO H/S with stable identities directly from a checkpoint."""
    import torch
    from omegaconf import OmegaConf
    from pl_module import get_pl_model
    from common.setup import setup_tensor_type_and_seed
    from dataset_module.qh9_stable_readonly import QH9StableReadonly
    from torch_geometric.loader import DataLoader
    rows = read_cohort(args.cohort)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=not args.trust_legacy_checkpoint)
    conf = OmegaConf.create(checkpoint["hyper_parameters"]["conf"])
    if not conf.qh9:
        raise ValueError("This exporter is for QH9 checkpoints")
    conf.seed = args.seed
    setup_tensor_type_and_seed(conf)
    module = get_pl_model(conf)(conf).to(args.device).eval()
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    dataset = QH9StableReadonly(args.processed_directory, args.split_file,
                               compute_q_tensor=False, include_cut_orbital_coefficients=False)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        ids = [r["dataset_index"] for r in rows]
        if not set(ids) <= set(dataset.test_mask.tolist()):
            raise ValueError("Cohort contains non-test molecules")
        # Fixed batch size 1 keeps cohort/order replay independent of available memory.
        loader = DataLoader(dataset[torch.tensor(ids)], batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for row, batch in zip(rows, loader):
                batch = module.post_processing(batch.to(args.device), module.default_type)
                prediction, _, _ = module.sample(batch.clone(), num_timesteps=args.ode_steps, return_trajectory=False)
                h = module.build_final_matrix(batch, prediction["hamiltonian_diagonal_blocks"],
                    prediction["hamiltonian_non_diagonal_blocks"], transform=True, convention="e3nn_to_pyscf_def2svp")[0]
                gt = dataset.get_gt_outputs(row["dataset_index"])
                filename = f"sample_{row['dataset_index']:06d}.npz"
                np.savez(output / filename, atoms=gt["atoms"].numpy().reshape(-1), pos_ang=gt["pos"].numpy(),
                         H_pred=h.detach().cpu().double().numpy(), H_reference=gt["hamiltonian"].numpy(), S=gt["overlap_matrix"].numpy())
                records.append({**row, "file": filename, "sha256": sha(output / filename)})
    finally:
        dataset.close()
    write_json(output / "manifest.json", dict(schema=1, dataset="QH9Stable-random", matrix_convention="native_pyscf",
        position_unit="Angstrom", matrix_unit="Hartree", checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha(args.checkpoint), split_sha256=sha(args.split_file), cohort_sha256=sha(args.cohort),
        seed=args.seed, ode_steps=args.ode_steps, batch_size=1, device=args.device,
        exporter_sha256=sha(__file__), source=source_identity(), torch_version=torch.__version__, records=records))


def summarize(records, mode):
    good = [r for r in records if r["status"] == "ok"]
    summary = dict(expected=len(records), completed=len(good), failures=[r["dataset_index"] for r in records if r["status"] != "ok"])
    if mode == "scf":
        paired = [r for r in good if all(s["converged"] for s in r["solves"].values())]
        summary.update(paired_converged=len(paired), nonconverged=[r["dataset_index"] for r in good if r not in paired])
        # No hidden removal of difficult molecules from a publication number.
        if len(paired) == len(records) and paired:
            metrics = {label: aggregate_cycles(
                [r["solves"][label]["cycles"] for r in paired],
                [r["solves"]["minao"]["cycles"] for r in paired],
                initialization_cycles=1 if label == "predicted" else 0,
            ) for label in ("predicted", "dataset_hamiltonian")}
            for field in ("cycle_ratio_percent", "cycle_reduction_percent"):
                summary[field] = {label: value[field] for label, value in metrics.items()}
            summary["eric"] = metrics["predicted"]["eric"]
            summary["eric_definition"] = "mean((N_predicted + 1) / N_MINAO); one initial Hamiltonian build, excluding inference time"
            summary["aggregation"] = "100 * sum(model cycles) / sum(MinAO cycles); no +1 correction"
    elif len(good) == len(records) and good:
        de = np.array([r["predicted"]["energy_Ha"] - r["reference"]["energy_Ha"] for r in good])
        df = np.concatenate([(np.asarray(r["predicted"]["forces_Ha_per_Bohr"]) - r["reference"]["forces_Ha_per_Bohr"]).reshape(-1) for r in good])
        summary.update(energy_mae_Ha=float(abs(de).mean()), energy_rmse_Ha=float(np.sqrt((de**2).mean())),
                       force_component_mae_Ha_per_Bohr=float(abs(df).mean()), force_component_rmse_Ha_per_Bohr=float(np.sqrt((df**2).mean())))
    summary["publication_ready"] = len(good) == len(records) and (mode != "scf" or len(paired) == len(records))
    return summary


def evaluate(args):
    from pyscf import lib
    config = yaml.safe_load(args.config.read_text())
    lib.num_threads(config["threads"])
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("matrix_convention") != "native_pyscf" or manifest.get("position_unit") != "Angstrom" or manifest.get("matrix_unit") != "Hartree":
        raise ValueError("Input manifest has unsupported conventions/units")
    ids = [r["dataset_index"] for r in manifest["records"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicate molecule IDs")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("numpy", "scipy", "torch", "pyscf", "gpu4pyscf", "cupy-cuda12x"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    source = {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}
    protocol = dict(config=config, manifest_sha256=sha(args.manifest), source_sha256=source, versions=versions)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing output belongs to different inputs/settings/code/runtime")
    else:
        write_json(protocol_path, protocol)
    records = []
    for row in manifest["records"]:
        path = args.manifest.parent / row["file"]
        if sha(path) != row["sha256"]:
            raise ValueError(f"Input checksum changed: {path}")
        target = output / f"sample_{row['dataset_index']:06d}.json"
        if target.exists():
            r = json.loads(target.read_text())
            if r["protocol_sha256"] != sha(protocol_path) or r["input_sha256"] != row["sha256"]:
                raise ValueError(f"Stale result: {target}")
            if r["status"] == "ok":
                records.append(r)
                continue
        r = dict(dataset_index=row["dataset_index"], prediction_index=row["prediction_index"],
                 molecule_id=row.get("molecule_id"), input_sha256=row["sha256"], protocol_sha256=sha(protocol_path))
        try:
            with np.load(path, allow_pickle=False) as z:
                a, p, s = z["atoms"], z["pos_ang"], z["S"]
                mf = build_mf(a, p, config["dft"])
                if not np.allclose(host(mf.get_ovlp()), s, atol=2e-6, rtol=0):
                    raise ValueError("Geometry/overlap convention mismatch")
                if config["mode"] == "scf":
                    densities = {"minao": None}
                    for name, key in (("predicted", "H_pred"), ("dataset_hamiltonian", "H_reference")):
                        densities[name] = orbitals(z[key], s, int(sum(a)), config["orbitals"])[3]
                    r["solves"] = {name: solve_scf(a, p, dm, config["dft"]) for name, dm in densities.items()}
                elif config["mode"] == "energy_force":
                    r["predicted"] = one_shot(a, p, z["H_pred"], s, config["dft"], config["orbitals"])
                    r["reference"] = one_shot(a, p, z["H_reference"], s, config["dft"], config["reference_orbitals"])
                else:
                    raise ValueError("Unknown evaluation mode")
            r["status"] = "ok"
            # Reject nonfinite outputs before writing a reusable success record.
            json.dumps(r, allow_nan=False)
        except Exception as exc:
            r = {k: v for k, v in r.items() if k not in {"solves", "predicted", "reference"}}
            r.update(status="error", error=f"{type(exc).__name__}: {exc}")
        write_json(target, r)
        records.append(r)
    summary = summarize(records, config["mode"])
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["publication_ready"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("select", help="Select a new replay cohort from the explicit test split")
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=300, help="0 selects the entire test split")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=select)
    p = commands.add_parser("infer", help="Generate hashed native-AO predictions from a checkpoint")
    for name in ("cohort", "checkpoint", "processed-directory", "split-file", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ode-steps", type=int, default=3)
    p.add_argument("--trust-legacy-checkpoint", action="store_true", help="Allow pickle objects in an original checkpoint you trust")
    p.set_defaults(func=infer)
    p = commands.add_parser("prepare", help="Join explicit molecule IDs with trusted local predictions")
    for name in ("cohort", "pred-dir", "checkpoint", "processed-directory", "split-file", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, help="Use saved gt_flat matrices, preserving historical reference precision")
    p.set_defaults(func=prepare)
    p = commands.add_parser("evaluate", help="Resume a hash-checked SCF or energy/force evaluation")
    for name in ("manifest", "config", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
