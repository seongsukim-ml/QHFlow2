"""Benchmark __getitem__ sub-operations and test optimizations."""
import torch, time, numpy as np
import hydra
from omegaconf import DictConfig
from qhflow2.common.setup import get_root_path, setup_tensor_type_and_seed
from qhflow2.common.qh9_utils import load_qh9_dataset
from qhflow2.dataset_module.qh9_datasets_split import matrix_transform


@hydra.main(config_path="../config_qh9", config_name="config_flow_v2_simple", version_base=None)
def main(conf: DictConfig):
    setup_tensor_type_and_seed(conf)
    dataset = load_qh9_dataset(conf, get_root_path())

    print(f'Dataset: {len(dataset)} samples')

    # Get raw data from LMDB for a few samples
    import lmdb, pickle, os
    lmdb_path = os.path.join(dataset.processed_dir, "QH9Stable.lmdb")
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    rng = np.random.RandomState(42)
    test_indices = rng.choice(len(dataset), 20, replace=False)

    # ============================================================
    # 1. Profile original get_mol sub-operations
    # ============================================================
    print('\n' + '='*60)
    print('1. ORIGINAL get_mol() BREAKDOWN')
    print('='*60)

    timings = {k: [] for k in ['lmdb_read', 'matrix_transform', 'cut_matrix', 'q_tensor', 'assemble', 'total']}

    for idx in test_indices:
        with env.begin() as txn:
            key = int(idx).to_bytes(length=4, byteorder="big")
            raw = txn.get(key)
            data_dict = pickle.loads(raw)

        num_nodes = data_dict["num_nodes"]
        atoms = np.frombuffer(data_dict["atoms"], np.int32)
        pos = np.frombuffer(data_dict["pos"], np.float64).reshape(num_nodes, 3)
        num_orbitals = sum([5 if a <= 2 else 14 for a in atoms])
        Ham = np.frombuffer(data_dict["Ham"], np.float64).reshape(num_orbitals, num_orbitals)
        init_ham = np.frombuffer(data_dict["init_ham"], np.float64).reshape(num_orbitals, num_orbitals)
        ovlp = np.frombuffer(data_dict["ovlp"], np.float64).reshape(num_orbitals, num_orbitals)

        t_total = time.perf_counter()

        # matrix_transform × 3
        t0 = time.perf_counter()
        H_t = torch.tensor(matrix_transform(Ham, atoms, convention="pyscf_def2svp"), dtype=torch.float64)
        iH_t = torch.tensor(matrix_transform(init_ham, atoms, convention="pyscf_def2svp"), dtype=torch.float64)
        S_t = torch.tensor(matrix_transform(ovlp, atoms, convention="pyscf_def2svp"), dtype=torch.float64)
        t1 = time.perf_counter()
        timings['matrix_transform'].append(t1 - t0)

        # cut_matrix × 3
        t0 = time.perf_counter()
        dataset.cut_matrix(H_t, atoms)
        dataset.cut_matrix(iH_t, atoms)
        dataset.cut_matrix(S_t, atoms)
        t1 = time.perf_counter()
        timings['cut_matrix'].append(t1 - t0)

        # Q tensor
        t0 = time.perf_counter()
        Q = torch.stack([
            torch.block_diag(*[dataset.Q_dict[z][l] for z in atoms])
            for l in range(60)
        ]).double().numpy()
        Q = torch.from_numpy(matrix_transform(Q, atoms, convention="pyscf_def2svp")).double().permute(1, 2, 0)
        H_size = H_t.size(-1)
        Q[:, :, 16:40] = Q[:, :, 16:40].reshape(H_size, H_size, -1, 3)[:, :, :, [1, 2, 0]].reshape(H_size, H_size, 24)
        dataset.cut_matrix(Q, atoms)
        t1 = time.perf_counter()
        timings['q_tensor'].append(t1 - t0)

        timings['total'].append(time.perf_counter() - t_total)

    for k, v in timings.items():
        print(f'  {k:25s}: {np.mean(v)*1000:7.2f} ms (±{np.std(v)*1000:.2f})')

    # ============================================================
    # 2. Optimized cut_matrix (pre-allocated, vectorized)
    # ============================================================
    print('\n' + '='*60)
    print('2. OPTIMIZED cut_matrix')
    print('='*60)

    def cut_matrix_fast(matrix, atoms, orbital_mask, full_orbitals=14):
        """Optimized cut_matrix: pre-allocate all blocks at once."""
        n_atoms = len(atoms)
        is_3d = len(matrix.shape) == 3
        extra_dim = matrix.shape[-1] if is_3d else 0

        # Pre-compute offsets
        offsets = [0]
        for a in atoms:
            offsets.append(offsets[-1] + len(orbital_mask[int(a)]))

        # Pre-allocate all blocks at once
        if is_3d:
            diag = torch.zeros(n_atoms, full_orbitals, full_orbitals, extra_dim, dtype=torch.float64)
            diag_mask = torch.zeros(n_atoms, full_orbitals, full_orbitals, extra_dim, dtype=torch.float64)
            offdiag = torch.zeros(n_atoms * (n_atoms - 1), full_orbitals, full_orbitals, extra_dim, dtype=torch.float64)
            offdiag_mask = torch.zeros(n_atoms * (n_atoms - 1), full_orbitals, full_orbitals, extra_dim, dtype=torch.float64)
        else:
            diag = torch.zeros(n_atoms, full_orbitals, full_orbitals, dtype=torch.float64)
            diag_mask = torch.zeros(n_atoms, full_orbitals, full_orbitals, dtype=torch.float64)
            offdiag = torch.zeros(n_atoms * (n_atoms - 1), full_orbitals, full_orbitals, dtype=torch.float64)
            offdiag_mask = torch.zeros(n_atoms * (n_atoms - 1), full_orbitals, full_orbitals, dtype=torch.float64)

        edge_index = []
        offdiag_idx = 0

        for i in range(n_atoms):
            mask_i = orbital_mask[int(atoms[i])]
            for j in range(n_atoms):
                mask_j = orbital_mask[int(atoms[j])]
                block = matrix[offsets[j]:offsets[j+1], offsets[i]:offsets[i+1]]

                if i == j:
                    diag[i][np.ix_(mask_j, mask_i)] = block
                    diag_mask[i][np.ix_(mask_j, mask_i)] = 1
                else:
                    edge_index.append([j, i])
                    offdiag[offdiag_idx][np.ix_(mask_j, mask_i)] = block
                    offdiag_mask[offdiag_idx][np.ix_(mask_j, mask_i)] = 1
                    offdiag_idx += 1

        return diag, offdiag, diag_mask, offdiag_mask, torch.tensor(edge_index).T

    # Benchmark
    with env.begin() as txn:
        key = int(test_indices[0]).to_bytes(length=4, byteorder="big")
        d = pickle.loads(txn.get(key))
    atoms_0 = np.frombuffer(d["atoms"], np.int32)
    n_orb = sum([5 if a <= 2 else 14 for a in atoms_0])
    H0 = torch.tensor(matrix_transform(
        np.frombuffer(d["Ham"], np.float64).reshape(n_orb, n_orb),
        atoms_0, convention="pyscf_def2svp"), dtype=torch.float64)

    N_bench = 100
    # Original
    t0 = time.perf_counter()
    for _ in range(N_bench):
        dataset.cut_matrix(H0, atoms_0)
    t_orig = (time.perf_counter() - t0) / N_bench * 1000

    # Optimized
    t0 = time.perf_counter()
    for _ in range(N_bench):
        cut_matrix_fast(H0, atoms_0, dataset.orbital_mask)
    t_fast = (time.perf_counter() - t0) / N_bench * 1000

    # Verify correctness
    r1 = dataset.cut_matrix(H0, atoms_0)
    r2 = cut_matrix_fast(H0, atoms_0, dataset.orbital_mask)
    diag_match = torch.allclose(r1[0], r2[0])
    offdiag_match = torch.allclose(r1[1], r2[1])

    print(f'  Original cut_matrix:  {t_orig:.2f} ms')
    print(f'  Optimized cut_matrix: {t_fast:.2f} ms  ({t_orig/t_fast:.1f}× speedup)')
    print(f'  Correctness: diag={diag_match}, offdiag={offdiag_match}')

    # ============================================================
    # 3. Cached matrix_transform indices
    # ============================================================
    print('\n' + '='*60)
    print('3. CACHED MATRIX_TRANSFORM')
    print('='*60)

    # The key insight: matrix_transform only depends on atoms (not values)
    # We can precompute the permutation indices

    def precompute_transform_indices(atoms, convention="pyscf_def2svp"):
        """Precompute reordering indices for a given atom list."""
        n_orb = sum([5 if a <= 2 else 14 for a in atoms])
        identity = np.eye(n_orb)
        transformed = matrix_transform(identity, atoms, convention=convention)
        # The transform is a permutation + sign change
        # P @ M @ P^T where P is the permutation matrix
        P = torch.tensor(transformed, dtype=torch.float64)
        return P

    def apply_cached_transform(matrix, P):
        """Apply precomputed transform: P @ M @ P^T"""
        M = torch.tensor(matrix, dtype=torch.float64) if not isinstance(matrix, torch.Tensor) else matrix
        if len(M.shape) == 2:
            return P @ M @ P.T
        else:  # 3D (Q tensor)
            return torch.einsum('ij,...jk,lk->...il', P, M, P)

    P = precompute_transform_indices(atoms_0)

    # Verify
    H_raw = np.frombuffer(d["Ham"], np.float64).reshape(n_orb, n_orb)
    H_orig = torch.tensor(matrix_transform(H_raw, atoms_0, convention="pyscf_def2svp"), dtype=torch.float64)
    H_cached = apply_cached_transform(H_raw, P)
    match = torch.allclose(H_orig, H_cached, atol=1e-10)
    print(f'  Correctness: {match} (max diff: {(H_orig - H_cached).abs().max():.2e})')

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(N_bench):
        torch.tensor(matrix_transform(H_raw, atoms_0, convention="pyscf_def2svp"), dtype=torch.float64)
    t_orig_mt = (time.perf_counter() - t0) / N_bench * 1000

    t0 = time.perf_counter()
    for _ in range(N_bench):
        apply_cached_transform(H_raw, P)
    t_cached_mt = (time.perf_counter() - t0) / N_bench * 1000

    print(f'  Original matrix_transform: {t_orig_mt:.2f} ms')
    print(f'  Cached (P @ M @ P^T):      {t_cached_mt:.2f} ms  ({t_orig_mt/t_cached_mt:.1f}× speedup)')

    # ============================================================
    # 4. Overall speedup estimate
    # ============================================================
    print('\n' + '='*60)
    print('4. PROJECTED SPEEDUP')
    print('='*60)

    orig_mt = np.mean(timings['matrix_transform']) * 1000  # 3 transforms
    orig_cut = np.mean(timings['cut_matrix']) * 1000  # 3 cuts
    orig_q = np.mean(timings['q_tensor']) * 1000  # Q + transform + cut
    orig_total = np.mean(timings['total']) * 1000

    # Projected with optimizations
    new_mt = orig_mt * (t_cached_mt / t_orig_mt)  # cached transform speedup
    new_cut = orig_cut * (t_fast / t_orig)  # optimized cut speedup
    new_q = orig_q * 0.7  # conservative estimate for Q optimization
    new_total = new_mt + new_cut + new_q

    print(f'  {"":25s} {"Original":>10s} {"Optimized":>10s} {"Speedup":>8s}')
    print(f'  {"matrix_transform ×3":25s} {orig_mt:9.2f}ms {new_mt:9.2f}ms {orig_mt/new_mt:7.1f}×')
    print(f'  {"cut_matrix ×3":25s} {orig_cut:9.2f}ms {new_cut:9.2f}ms {orig_cut/new_cut:7.1f}×')
    print(f'  {"Q tensor":25s} {orig_q:9.2f}ms {new_q:9.2f}ms {orig_q/new_q:7.1f}×')
    print(f'  {"TOTAL get_mol":25s} {orig_total:9.2f}ms {new_total:9.2f}ms {orig_total/new_total:7.1f}×')
    print(f'\n  With nw=4: data_load ≈ {orig_total * 32 / 4:.0f}ms → {new_total * 32 / 4:.0f}ms')
    print(f'  With nw=8: data_load ≈ {orig_total * 32 / 8:.0f}ms → {new_total * 32 / 8:.0f}ms')

    env.close()


if __name__ == '__main__':
    main()
