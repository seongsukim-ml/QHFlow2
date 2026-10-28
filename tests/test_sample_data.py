"""Check the public numerical fixtures against AO physics and graph batching."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from pyscf import gto, lib
from torch_geometric.loader import DataLoader

from common.matrix_transforms import matrix_transform_single
from dataset_module.qh9_dataset import BaseQH9Dataset, _cut_matrix_3d
from reproduction.sample_data import as_pyg, read_samples

MANIFEST = Path(__file__).resolve().parents[1] / "examples/qh9stable/manifest.json"


def test_sample_overlap_and_blocks_match_native_ao_physics():
    lib.num_threads(1)
    _, samples = read_samples(MANIFEST)
    for row, arrays in samples:
        molecule = gto.M(atom=list(zip(arrays["atoms"].tolist(), arrays["pos_ang"].tolist())),
                         basis="def2-svp", unit="Angstrom", verbose=0)
        np.testing.assert_allclose(arrays["S"], molecule.intor("int1e_ovlp"), atol=2e-12, rtol=0)
        atoms = torch.from_numpy(arrays["atoms"])
        matrices = torch.from_numpy(np.stack([arrays["H_reference"], arrays["S"], arrays["H_initial"]]))
        transformed = matrix_transform_single(matrices, atoms, "pyscf_def2svp_to_e3nn")
        diag, off, diag_mask, off_mask, edges = _cut_matrix_3d(
            transformed, atoms, BaseQH9Dataset.orbital_mask, BaseQH9Dataset.full_orbitals)
        for i, suffix in enumerate(("hamiltonian", "overlap", "init_ham")):
            np.testing.assert_array_equal(arrays[f"input__diagonal_{suffix}"], diag[:, i].numpy())
            np.testing.assert_array_equal(arrays[f"input__non_diagonal_{suffix}"], off[:, i].numpy())
        np.testing.assert_array_equal(arrays["input__diagonal_hamiltonian_mask"], diag_mask[:, 0])
        np.testing.assert_array_equal(arrays["input__non_diagonal_hamiltonian_mask"], off_mask[:, 0])
        np.testing.assert_array_equal(arrays["input__edge_index_full"], edges)
        assert int(arrays["input__idx"]) == row["dataset_index"]


def test_pyg_batch_keeps_edges_inside_each_molecule():
    _, samples = read_samples(MANIFEST)
    batch = next(iter(DataLoader([as_pyg(arrays) for _, arrays in samples], batch_size=3)))
    edges = batch.edge_index_full
    assert torch.equal(batch.batch[edges[0]], batch.batch[edges[1]])
    assert batch.idx.tolist() == [row["dataset_index"] for row, _ in samples]
    assert batch.num_nodes == sum(row["num_atoms"] for row, _ in samples)


def test_changed_sample_is_rejected_before_loading(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    manifest["records"] = manifest["records"][:1]
    target = tmp_path / manifest["records"][0]["file"]
    shutil.copyfile(MANIFEST.parent / target.name, target)
    with target.open("ab") as handle:
        handle.write(b"unexpected modification")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        read_samples(tmp_path / "manifest.json")


def test_incorrect_coordinate_unit_is_not_silently_accepted(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    manifest["position_unit"] = "Bohr"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="units or AO conventions"):
        read_samples(tmp_path / "manifest.json")
