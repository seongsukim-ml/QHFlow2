# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
"""Small, pickle-free QH9 fixtures for local loading and checkpoint smoke tests."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_samples(manifest_path):
    """Verify fixture integrity without importing Torch or opening the full dataset."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "qhflow2-samples-v1":
        raise ValueError("Unsupported sample schema")
    conventions = dict(basis="def2-svp", position_unit="Angstrom", hamiltonian_unit="Hartree",
                       overlap_unit="dimensionless", reference_ao_convention="native_pyscf",
                       input_ao_convention="pyscf_def2svp_to_e3nn, padded 14-orbital blocks")
    if any(manifest.get(key) != value for key, value in conventions.items()):
        raise ValueError("Unsupported sample units or AO conventions")
    records = manifest["records"]
    ids = [row["dataset_index"] for row in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate sample IDs")
    samples = []
    for row in records:
        if Path(row["file"]).name != row["file"]:
            raise ValueError("Sample filename must be a basename")
        path = manifest_path.parent / row["file"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Checksum mismatch: {row['file']}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"Non-finite sample: {row['file']}")
        n, h = row["num_atoms"], row["num_orbitals"]
        if arrays["atoms"].shape != (n,) or arrays["pos_ang"].shape != (n, 3):
            raise ValueError("Invalid geometry shape")
        for key in ("H_reference", "S", "H_initial"):
            value = arrays[key]
            if value.shape != (h, h) or not np.allclose(value, value.T, atol=1e-10, rtol=0):
                raise ValueError(f"Invalid symmetric matrix: {key}")
        if np.linalg.eigvalsh(arrays["S"])[0] <= 0:
            raise ValueError("Overlap is not positive definite")
        expected = {"pos": (n, 3), "atoms": (n, 1),
                    "edge_index_full": (2, n * (n - 1)),
                    "idx": (), "h_dim": (), "num_nodes": ()}
        for prefix, count in (("diagonal", n), ("non_diagonal", n * (n - 1))):
            for suffix in ("hamiltonian", "hamiltonian_mask", "overlap", "init_ham"):
                expected[f"{prefix}_{suffix}"] = (count, 14, 14)
        for key, shape in expected.items():
            if arrays[f"input__{key}"].shape != shape:
                raise ValueError(f"Invalid model input shape: {key}")
        if (int(arrays["input__idx"]) != row["dataset_index"] or
                int(arrays["input__h_dim"]) != h or int(arrays["input__num_nodes"]) != n):
            raise ValueError("Sample identity or dimensions disagree")
        if not np.array_equal(arrays["atoms"], arrays["input__atoms"].reshape(-1)):
            raise ValueError("Atomic numbers disagree")
        if not np.array_equal(arrays["pos_ang"], arrays["input__pos"]):
            raise ValueError("Coordinates disagree")
        samples.append((row, arrays))
    return manifest, samples


def as_pyg(arrays):
    import torch
    from torch_geometric.data import Data

    values = {key.removeprefix("input__"): torch.from_numpy(value.copy())
              for key, value in arrays.items() if key.startswith("input__")}
    values["num_nodes"] = int(values["num_nodes"])
    values["idx"] = int(values["idx"])
    return Data(**values)


def export_samples(args):
    """Select the first three official test rows from trusted local QH9 shards."""
    import pickle
    import torch
    from dataset_module.qh9_stable_readonly import QH9StableReadonly

    dataset = QH9StableReadonly(args.processed_directory, args.split_file,
                               compute_q_tensor=False, include_cut_orbital_coefficients=False)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for position, idx in enumerate(dataset.test_mask.tolist()[:3]):
            _, payload = dataset._payload(idx)
            raw = pickle.loads(payload)  # The same trusted local records as the training reader.
            gt, data = dataset.get_gt_outputs(idx), dataset[idx]
            arrays = dict(atoms=gt["atoms"].numpy().reshape(-1), pos_ang=gt["pos"].numpy(),
                          H_reference=gt["hamiltonian"].numpy(), S=gt["overlap_matrix"].numpy(),
                          H_initial=gt["initial_hamiltonian"].numpy())
            arrays.update({"input__" + key: value.numpy() if torch.is_tensor(value) else np.asarray(value)
                           for key, value in data.to_dict().items()})
            path = output / f"sample_{idx:06d}.npz"
            np.savez_compressed(path, **arrays)
            records.append(dict(dataset_index=idx, molecule_id=f"QH9Stable:{idx}",
                test_position=position, source_record_id=int(raw["id"]),
                source_lmdb_index=list(dataset._index[idx]), source_record_sha256=hashlib.sha256(payload).hexdigest(),
                num_atoms=len(arrays["atoms"]), num_orbitals=int(gt["h_dim"]),
                file=path.name, bytes=path.stat().st_size, sha256=sha256(path)))
    finally:
        dataset.close()
    manifest = dict(schema="qhflow2-samples-v1", dataset="QH9Stable-random", purpose="smoke_test_only",
        selection="First three entries of the recorded official test split; no accuracy-based selection",
        split_sha256=sha256(args.split_file), source_index_sha256=sha256(args.processed_directory / "index.json"),
        loader_sha256=sha256(Path(__file__).parents[1] / "dataset_module/qh9_dataset.py"),
        basis="def2-svp", position_unit="Angstrom", hamiltonian_unit="Hartree", overlap_unit="dimensionless",
        reference_ao_convention="native_pyscf", input_ao_convention="pyscf_def2svp_to_e3nn, padded 14-orbital blocks",
        changes="Lossless NPZ subset and model input blocks from the existing QHFlow2 read-only loader; no rounding",
        license="CC-BY-NC-SA-4.0", source="https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9",
        records=records)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    read_samples(output / "manifest.json")
    return dict(samples=len(records), dataset_indices=[r["dataset_index"] for r in records],
                compressed_data_bytes=sum(r["bytes"] for r in records))


def infer_samples(args):
    import torch
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader
    from common.setup import setup_tensor_type_and_seed
    from pl_module import get_pl_model

    manifest, samples = read_samples(args.manifest)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    conf = OmegaConf.create(checkpoint["hyper_parameters"]["conf"])
    conf.seed = args.seed
    setup_tensor_type_and_seed(conf)
    module = get_pl_model(conf)(conf).to(args.device).eval()
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with torch.no_grad():
        loader = DataLoader([as_pyg(arrays) for _, arrays in samples], batch_size=1, shuffle=False)
        for (row, arrays), batch in zip(samples, loader):
            batch = module.post_processing(batch.to(args.device), module.default_type)
            prediction, _, _ = module.sample(batch.clone(), num_timesteps=args.ode_steps, return_trajectory=False)
            h = module.build_final_matrix(batch, prediction["hamiltonian_diagonal_blocks"],
                prediction["hamiltonian_non_diagonal_blocks"], transform=True,
                convention="e3nn_to_pyscf_def2svp")[0].cpu().double().numpy()
            if h.shape != arrays["H_reference"].shape or not np.isfinite(h).all():
                raise ValueError("Invalid predicted Hamiltonian")
            path = output / row["file"]
            np.savez_compressed(path, atoms=arrays["atoms"], pos_ang=arrays["pos_ang"], H_pred=h,
                                H_reference=arrays["H_reference"], S=arrays["S"])
            records.append(dict(dataset_index=row["dataset_index"], prediction_index=row["test_position"],
                molecule_id=row["molecule_id"], file=path.name, sha256=sha256(path),
                hamiltonian_mae_Ha=float(np.abs(h - arrays["H_reference"]).mean())))
    result = dict(schema=1, purpose="smoke_test_only", dataset=manifest["dataset"],
        matrix_convention="native_pyscf", position_unit="Angstrom", matrix_unit="Hartree",
        checkpoint_sha256=sha256(args.checkpoint), sample_manifest_sha256=sha256(args.manifest),
        seed=args.seed, ode_steps=args.ode_steps, device=args.device, batch_size=1, records=records)
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Extract three fixtures from trusted local shards")
    for name in ("processed-directory", "split-file", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    export.set_defaults(func=export_samples)
    check = commands.add_parser("check", help="Validate samples using NumPy only")
    check.add_argument("--manifest", type=Path, required=True)
    check.set_defaults(func=lambda args: dict(status="ok", dataset_indices=[r["dataset_index"]
                       for r, _ in read_samples(args.manifest)[1]]))
    infer = commands.add_parser("infer", help="Run a safe inference export without the full dataset")
    for name in ("manifest", "checkpoint", "output"):
        infer.add_argument("--" + name, type=Path, required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--ode-steps", type=int, choices=(1, 3), default=1)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--threads", type=int, default=2)
    infer.set_defaults(func=infer_samples)
    args = parser.parse_args()
    if getattr(args, "threads", 1) < 1:
        parser.error("--threads must be positive")
    print(json.dumps(args.func(args), indent=2))


if __name__ == "__main__":
    main()
