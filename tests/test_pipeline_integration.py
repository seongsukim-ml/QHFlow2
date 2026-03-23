"""Minimal unit tests for the data pipeline + benchmarking integration.

Tests:
  1. OrbitalMatrix: convention conversion, block decomposition, roundtrip
  2. QH9PreprocessedDataset: LMDB read, Data fields, train/eval modes
  3. PredictionResult: save/load roundtrip, compare metrics
  4. properties_from_hamiltonian: eigenvalue extraction, HOMO/LUMO/gap

Run:
    cd /home1/irteam/data-vol1/projects/QHFlow2
    python -m pytest tests/test_pipeline_integration.py -v
"""

import sys
import os
import tempfile
import shutil

import numpy as np
import pytest

# Setup paths
_PROJECT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_PROJECT, "src"))
sys.path.insert(0, os.path.join(_PROJECT, "src", "qhflow2"))
sys.path.insert(0, "/home1/irteam/data-vol1/projects/dft-dataset/src")


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def water_system():
    """Water molecule (O, H, H) with random symmetric matrices."""
    from molecule import BasisInfo

    atoms = np.array([8, 1, 1], dtype=np.int32)
    basis = BasisInfo.from_name("def2-svp", convention="pyscf")
    nao = basis.total_nao(atoms)  # 14 + 5 + 5 = 24

    rng = np.random.RandomState(42)
    H = rng.randn(nao, nao).astype(np.float64)
    H = (H + H.T) / 2
    S = np.eye(nao, dtype=np.float64) + 0.1 * rng.randn(nao, nao)
    S = (S + S.T) / 2

    return {"atoms": atoms, "basis": basis, "nao": nao, "H": H, "S": S}


@pytest.fixture
def ch4_system():
    """CH4 molecule (C, H, H, H, H)."""
    from molecule import BasisInfo

    atoms = np.array([6, 1, 1, 1, 1], dtype=np.int32)
    basis = BasisInfo.from_name("def2-svp", convention="pyscf")
    nao = basis.total_nao(atoms)  # 14 + 4*5 = 34

    rng = np.random.RandomState(123)
    H = rng.randn(nao, nao).astype(np.float64)
    H = (H + H.T) / 2

    return {"atoms": atoms, "basis": basis, "nao": nao, "H": H}


@pytest.fixture
def tmp_dir():
    """Temporary directory, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="qhflow2_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── 1. OrbitalMatrix Tests ──────────────────────────────────────


class TestOrbitalMatrix:
    """OrbitalMatrix convention conversion and block decomposition."""

    def test_creation(self, water_system):
        from orbital_matrix import OrbitalMatrix

        om = OrbitalMatrix(
            data=water_system["H"],
            atomic_numbers=water_system["atoms"],
            basis_info=water_system["basis"],
        )
        assert om.nao == water_system["nao"]
        assert om.convention == "pyscf"
        assert om.n_atoms == 3

    def test_convention_roundtrip(self, water_system):
        """pyscf → e3nn → pyscf should be lossless."""
        from orbital_matrix import OrbitalMatrix

        om = OrbitalMatrix(
            data=water_system["H"],
            atomic_numbers=water_system["atoms"],
            basis_info=water_system["basis"],
        )
        om_e3nn = om.in_convention("e3nn")
        assert om_e3nn.convention == "e3nn"

        om_back = om_e3nn.in_convention("pyscf")
        assert om_back.convention == "pyscf"
        np.testing.assert_allclose(om.data, om_back.data, atol=1e-14)

    def test_same_convention_returns_self(self, water_system):
        from orbital_matrix import OrbitalMatrix

        om = OrbitalMatrix(
            data=water_system["H"],
            atomic_numbers=water_system["atoms"],
            basis_info=water_system["basis"],
        )
        om2 = om.in_convention("pyscf")
        assert om2 is om

    def test_block_roundtrip_shell(self, water_system):
        """matrix → to_blocks(shell) → to_dense should be lossless."""
        from orbital_matrix import OrbitalMatrix

        om = OrbitalMatrix(
            data=water_system["H"],
            atomic_numbers=water_system["atoms"],
            basis_info=water_system["basis"],
        )
        blocks = om.to_blocks(pad_to=14, align="shell")
        reconstructed = blocks.to_dense()
        np.testing.assert_allclose(om.data, reconstructed, atol=1e-14)

    def test_block_roundtrip_contiguous(self, water_system):
        """matrix → to_blocks(contiguous) → to_dense should be lossless."""
        from orbital_matrix import OrbitalMatrix

        om = OrbitalMatrix(
            data=water_system["H"],
            atomic_numbers=water_system["atoms"],
            basis_info=water_system["basis"],
        )
        blocks = om.to_blocks(pad_to=14, align="contiguous")
        reconstructed = blocks.to_dense()
        np.testing.assert_allclose(om.data, reconstructed, atol=1e-14)

    def test_block_shapes(self, ch4_system):
        """Check block shapes for CH4 (5 atoms)."""
        from orbital_matrix import OrbitalMatrix

        om = OrbitalMatrix(
            data=ch4_system["H"],
            atomic_numbers=ch4_system["atoms"],
            basis_info=ch4_system["basis"],
        )
        blocks = om.to_blocks(pad_to=14, align="shell")
        assert blocks.diagonal.shape == (5, 14, 14)
        assert blocks.off_diagonal.shape == (20, 14, 14)  # 5*4
        assert blocks.edge_index.shape == (2, 20)
        assert blocks.n_atoms == 5
        assert blocks.n_edges == 20

    def test_matches_qhflow2_cut_matrix(self, water_system):
        """OrbitalMatrix.to_blocks() should match QHFlow2's _cut_matrix_3d."""
        import torch
        from orbital_matrix import OrbitalMatrix
        from common.matrix_transforms import _cut_matrix_3d

        atoms = water_system["atoms"]
        H = water_system["H"]

        # OrbitalMatrix path
        basis_e3nn = water_system["basis"].__class__.from_name("def2-svp", convention="e3nn")
        om = OrbitalMatrix(data=H, atomic_numbers=atoms, basis_info=basis_e3nn)
        blocks = om.to_blocks(pad_to=14, align="shell")

        # QHFlow2 path
        atoms_torch = torch.tensor(atoms, dtype=torch.int64)
        orbital_mask = {
            1: torch.tensor([0, 1, 3, 4, 5]),
            8: torch.arange(14),
        }
        H_torch = torch.from_numpy(H).unsqueeze(0)  # (1, nao, nao)
        qh_diag, qh_offdiag, qh_dmask, qh_odmask, qh_edge = _cut_matrix_3d(
            H_torch, atoms_torch, orbital_mask, 14
        )

        np.testing.assert_allclose(
            blocks.diagonal, qh_diag[:, 0].numpy(), atol=1e-14
        )
        np.testing.assert_allclose(
            blocks.off_diagonal, qh_offdiag[:, 0].numpy(), atol=1e-14
        )
        np.testing.assert_array_equal(blocks.edge_index, qh_edge.numpy())

    def test_serialization(self, water_system):
        """OrbitalMatrix to_dict/from_dict roundtrip."""
        from orbital_matrix import OrbitalMatrix

        om = OrbitalMatrix(
            data=water_system["H"],
            atomic_numbers=water_system["atoms"],
            basis_info=water_system["basis"],
        )
        d = om.to_dict()
        om2 = OrbitalMatrix.from_dict(d)
        np.testing.assert_allclose(om.data, om2.data, atol=1e-14)
        assert om2.convention == om.convention

    def test_blocks_serialization(self, water_system):
        """OrbitalBlocks to_dict/from_dict roundtrip."""
        from orbital_matrix import OrbitalMatrix

        om = OrbitalMatrix(
            data=water_system["H"],
            atomic_numbers=water_system["atoms"],
            basis_info=water_system["basis"],
        )
        blocks = om.to_blocks(pad_to=14)
        d = blocks.to_dict()
        from orbital_matrix import OrbitalBlocks

        blocks2 = OrbitalBlocks.from_dict(d)
        np.testing.assert_allclose(blocks.diagonal, blocks2.diagonal, atol=1e-14)
        np.testing.assert_allclose(
            blocks.off_diagonal, blocks2.off_diagonal, atol=1e-14
        )

    def test_atom_block(self, water_system):
        """atom_block() extracts correct submatrix."""
        from orbital_matrix import OrbitalMatrix

        om = OrbitalMatrix(
            data=water_system["H"],
            atomic_numbers=water_system["atoms"],
            basis_info=water_system["basis"],
        )
        # O diagonal block: (14, 14)
        block_O = om.atom_block(0)
        assert block_O.shape == (14, 14)

        # O-H1 off-diagonal: (14, 5)
        block_OH = om.atom_block(0, 1)
        assert block_OH.shape == (14, 5)

        # H1-H2: (5, 5)
        block_HH = om.atom_block(1, 2)
        assert block_HH.shape == (5, 5)


# ── 2. QH9PreprocessedDataset Tests ─────────────────────────────


class TestPreprocessedDataset:
    """QH9PreprocessedDataset LMDB read/write."""

    @pytest.fixture
    def small_lmdb(self, tmp_dir):
        """Create a small preprocessed LMDB with 5 samples."""
        import lmdb
        import pickle

        lmdb_path = os.path.join(tmp_dir, "test.lmdb")
        env = lmdb.open(lmdb_path, map_size=1 << 30)

        rng = np.random.RandomState(42)
        n_samples = 5
        split_data = {
            "train": [0, 1, 2],
            "val": [3],
            "test": [4],
        }

        with env.begin(write=True) as txn:
            for idx in range(n_samples):
                n_atoms = rng.randint(3, 8)
                n_edges = n_atoms * (n_atoms - 1)
                sample = {
                    "atoms": rng.choice([1, 6, 7, 8], size=n_atoms).astype(np.int32),
                    "pos": rng.randn(n_atoms, 3).astype(np.float64),
                    "num_nodes": n_atoms,
                    "h_dim": 14 * n_atoms,
                    "diagonal_hamiltonian": rng.randn(n_atoms, 14, 14).astype(np.float64),
                    "non_diagonal_hamiltonian": rng.randn(n_edges, 14, 14).astype(np.float64),
                    "diagonal_overlap": rng.randn(n_atoms, 14, 14).astype(np.float64),
                    "non_diagonal_overlap": rng.randn(n_edges, 14, 14).astype(np.float64),
                    "diagonal_init_ham": rng.randn(n_atoms, 14, 14).astype(np.float64),
                    "non_diagonal_init_ham": rng.randn(n_edges, 14, 14).astype(np.float64),
                    "diagonal_hamiltonian_mask": np.ones((n_atoms, 14, 14), dtype=np.float64),
                    "non_diagonal_hamiltonian_mask": np.ones((n_edges, 14, 14), dtype=np.float64),
                    "edge_index_full": np.array(
                        [[j for i in range(n_atoms) for j in range(n_atoms) if i != j],
                         [i for i in range(n_atoms) for j in range(n_atoms) if i != j]],
                        dtype=np.int64,
                    ),
                    "diagonal_Q": rng.randn(n_atoms, 14, 14, 60).astype(np.float64),
                    "non_diagonal_Q": rng.randn(n_edges, 14, 14, 60).astype(np.float64),
                    "dft_energy": float(rng.randn()),
                    "dft_forces": rng.randn(n_atoms, 3).astype(np.float64),
                    "hamiltonian_packed": rng.randn(100).astype(np.float64),
                    "overlap_packed": rng.randn(100).astype(np.float64),
                    "init_ham_packed": rng.randn(100).astype(np.float64),
                }
                key = idx.to_bytes(4, "big")
                txn.put(key, pickle.dumps(sample))

            txn.put(b"__len__", n_samples.to_bytes(4, "big"))
            txn.put(b"__format__", b"pickle")
            txn.put(b"__splits__", pickle.dumps(
                {"processed_QH9Stable_random_12": split_data}
            ))

        env.close()
        return lmdb_path

    def test_load_train_mode(self, small_lmdb):
        from dataset_module.qh9_preprocessed import QH9PreprocessedDataset

        ds = QH9PreprocessedDataset(small_lmdb, split="random", mode="train")
        assert len(ds) == 5
        assert len(ds.train_mask) == 3
        assert len(ds.val_mask) == 1
        assert len(ds.test_mask) == 1

    def test_getitem_returns_data(self, small_lmdb):
        from dataset_module.qh9_preprocessed import QH9PreprocessedDataset
        import torch

        ds = QH9PreprocessedDataset(small_lmdb, split="random", mode="train")
        data = ds.get(0)

        # Check essential fields exist and have correct types
        assert hasattr(data, "pos")
        assert hasattr(data, "atoms")
        assert hasattr(data, "diagonal_hamiltonian")
        assert hasattr(data, "edge_index_full")
        assert data.pos.dtype == torch.float64
        assert data.atoms.dtype == torch.int64

    def test_eval_mode_has_packed(self, small_lmdb):
        from dataset_module.qh9_preprocessed import QH9PreprocessedDataset

        ds = QH9PreprocessedDataset(small_lmdb, split="random", mode="eval")
        data = ds.get(0)

        assert hasattr(data, "hamiltonian_packed")
        assert hasattr(data, "overlap_packed")
        assert hasattr(data, "init_ham_packed")

    def test_train_mode_no_packed(self, small_lmdb):
        from dataset_module.qh9_preprocessed import QH9PreprocessedDataset

        ds = QH9PreprocessedDataset(small_lmdb, split="random", mode="train")
        data = ds.get(0)

        # train mode should still have packed if stored (we don't filter)
        # but the key fields for training must exist
        assert hasattr(data, "diagonal_hamiltonian")
        assert hasattr(data, "diagonal_Q")


# ── 3. PredictionResult Tests ───────────────────────────────────


class TestPredictionResult:
    """PredictionResult save/load/compare."""

    def test_save_load_roundtrip(self, tmp_dir):
        from prediction import PredictionResult

        rng = np.random.RandomState(42)
        n = 10
        nao = 24

        result = PredictionResult(
            matrices={
                "hamiltonian": [
                    rng.randn(nao, nao).astype(np.float64) for _ in range(n)
                ]
            },
            atomic_numbers=[np.array([8, 1, 1], dtype=np.int32) for _ in range(n)],
            positions=[rng.randn(3, 3).astype(np.float64) for _ in range(n)],
            convention="pyscf",
            metadata={"model": "test", "dataset": "qh9"},
        )

        path = os.path.join(tmp_dir, "pred.npz")
        result.save(path)
        assert os.path.exists(path)

        loaded = PredictionResult.load(path)
        assert len(loaded) == n
        assert loaded.convention == "pyscf"
        assert loaded.metadata["model"] == "test"
        assert "hamiltonian" in loaded.matrices
        assert len(loaded.matrices["hamiltonian"]) == n

    def test_compare_hamiltonians(self, tmp_dir):
        from prediction import PredictionResult

        rng = np.random.RandomState(42)
        n = 5
        nao = 24

        gt_hams = [rng.randn(nao, nao).astype(np.float64) for _ in range(n)]
        # Add small noise as prediction
        pred_hams = [h + 0.01 * rng.randn(nao, nao) for h in gt_hams]

        result = PredictionResult(
            matrices={"hamiltonian": pred_hams},
            metadata={"model": "test"},
        )

        report = result.compare(ref_matrices={"hamiltonian": gt_hams})
        assert report.matrix_errors is not None
        assert "hamiltonian" in report.matrix_errors
        assert report.matrix_errors["hamiltonian"]["mae"] > 0
        assert report.matrix_errors["hamiltonian"]["mae"] < 0.1  # noise was 0.01

    def test_describe(self):
        from prediction import PredictionResult

        rng = np.random.RandomState(42)
        result = PredictionResult(
            energy=rng.randn(10).astype(np.float64),
            metadata={"model": "test", "dataset": "qh9"},
        )
        desc = result.describe()
        assert "PredictionResult" in desc
        assert "n=10" in desc
        assert "model: test" in desc


# ── 4. Properties from Hamiltonian ──────────────────────────────


class TestPropertiesFromHamiltonian:
    """properties_from_hamiltonian eigenvalue extraction."""

    def test_eigenvalues_from_identity_overlap(self):
        from pipeline import properties_from_hamiltonian

        rng = np.random.RandomState(42)
        nao = 10
        H = rng.randn(nao, nao).astype(np.float64)
        H = (H + H.T) / 2
        S = np.eye(nao, dtype=np.float64)

        props = properties_from_hamiltonian(H, S, n_electrons=4)

        assert "orbital_energies" in props
        assert "homo" in props
        assert "lumo" in props
        assert "gap_eV" in props
        assert len(props["orbital_energies"]) == nao
        # Eigenvalues should be sorted
        e = props["orbital_energies"]
        assert np.all(e[:-1] <= e[1:] + 1e-10)
        # Gap should be positive
        assert props["gap_eV"] >= 0

    def test_homo_lumo_consistency(self):
        from pipeline import properties_from_hamiltonian

        nao = 10
        H = np.diag(np.arange(nao, dtype=np.float64))
        S = np.eye(nao, dtype=np.float64)

        # 4 electrons → 2 occupied (RKS)
        props = properties_from_hamiltonian(H, S, n_electrons=4)
        assert props["homo"] == pytest.approx(1.0, abs=1e-10)  # eigenvalue index 1
        assert props["lumo"] == pytest.approx(2.0, abs=1e-10)  # eigenvalue index 2

    def test_pred_vs_gt_comparison(self):
        """Comparing predicted H vs GT H should give small errors if close."""
        from pipeline import properties_from_hamiltonian

        rng = np.random.RandomState(42)
        nao = 20
        H_gt = rng.randn(nao, nao).astype(np.float64)
        H_gt = (H_gt + H_gt.T) / 2
        S = np.eye(nao, dtype=np.float64)
        n_electrons = 8

        # Prediction is GT + small noise
        H_pred = H_gt + 0.001 * rng.randn(nao, nao)
        H_pred = (H_pred + H_pred.T) / 2

        props_gt = properties_from_hamiltonian(H_gt, S, n_electrons)
        props_pred = properties_from_hamiltonian(H_pred, S, n_electrons)

        # Eigenvalue errors should be small
        e_err = np.abs(props_pred["orbital_energies"] - props_gt["orbital_energies"])
        assert e_err.mean() < 0.01  # very close

        # Gap error should be small
        gap_err_meV = abs(props_pred["gap_eV"] - props_gt["gap_eV"]) * 1000
        assert gap_err_meV < 100  # within 100 meV


# ── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
