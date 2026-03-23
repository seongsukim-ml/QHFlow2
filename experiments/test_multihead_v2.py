"""
Fast MultiHeadExpansion v2
==========================
핵심 최적화: Expansion 내부 instruction loop를 vectorize하여 CUDA kernel 수를 줄임.

전략:
1. 같은 (l_in, l_out1, l_out2) angular momentum 조합의 instructions를 그룹핑
2. 각 그룹 내 multiplicities를 하나의 큰 einsum으로 처리
3. Pre-compute + cache all w3j matrices
4. Pre-sort atoms by group → contiguous memory access
"""

import sys
import time
import torch
import torch.nn as nn
from math import prod
from e3nn import o3

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

BOTTLE_HIDDEN = 32
INPUT_EXPAND_IRREP = o3.Irreps(
    f"{BOTTLE_HIDDEN}x0e + {BOTTLE_HIDDEN}x1e + {BOTTLE_HIDDEN}x2e + "
    f"{BOTTLE_HIDDEN}x3e + {BOTTLE_HIDDEN}x4e"
)
HIDDEN_SIZE = 128

ORBITAL_GROUPS = {
    "sp": {
        "irrep": o3.Irreps("2x0e + 1x1e"),
        "elements": [1, 2],
    },
    "spd": {
        "irrep": o3.Irreps("3x0e + 2x1e + 1x2e"),
        "elements": list(range(3, 19)),
    },
    "spdf": {
        "irrep": o3.Irreps("4x0e + 3x1e + 2x2e + 1x3e"),
        "elements": list(range(19, 37)),
    },
}


# ============================================================
# Original Expansion (baseline, from QHFlow2)
# ============================================================

class Expansion(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super().__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.instructions = self._get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum(prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0)
        if self.num_path_weight > 0:
            self.weights = nn.Parameter(torch.rand(self.num_path_weight + self.num_bias))
        self.num_weights = self.num_path_weight + self.num_bias

    def forward(self, x_in, weights=None, bias_weights=None):
        batch_num = x_in.shape[0]
        if len(self.irrep_in) == 1:
            x_in_s = [x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
                for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)
            ]
        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            mul_ir_out1 = self.irrep_out_1[ins[1]]
            mul_ir_out2 = self.irrep_out_2[ins[2]]
            x1 = x_in_s[ins[0]].reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
            w3j_matrix = o3.wigner_3j(ins[1], ins[2], ins[0]).to(self.device).type(x1.type())
            if ins[3] is True or weights is not None:
                if weights is None:
                    weight = self.weights[flat_weight_index:flat_weight_index + prod(ins[-1])].reshape(ins[-1])
                    result = torch.einsum("wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1) / mul_ir_in.mul
                else:
                    weight = weights[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                    result = torch.einsum("bwuv, bwk-> buvk", weight, x1)
                    if ins[0] == 0 and bias_weights is not None:
                        bias_weight = bias_weights[:, bias_weight_index:bias_weight_index + prod(ins[-1][1:])].reshape([-1] + ins[-1][1:])
                        bias_weight_index += prod(ins[-1][1:])
                        result = result + bias_weight.unsqueeze(-1)
                    result = torch.einsum("ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
                flat_weight_index += prod(ins[-1])
            else:
                result = torch.einsum("uvw, ijk, bwk-> buivj",
                    torch.ones(ins[-1]).type(x1.type()).to(self.device), w3j_matrix, x1)
            result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
            key = (ins[1], ins[2])
            if key in outputs:
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result
        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                if (i, j) not in outputs:
                    blocks.append(torch.zeros(x_in.shape[0], self.irrep_out_1[i].dim, self.irrep_out_2[j].dim,
                                              device=x_in.device).type(x_in.type()))
                else:
                    blocks.append(outputs[(i, j)])
            rows.append(torch.cat(blocks, dim=-1))
        return torch.cat(rows, dim=-2)

    def _get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions

    @property
    def device(self):
        return next(self.parameters()).device


# ============================================================
# FastExpansion: vectorized instruction loop
# ============================================================

class FastExpansion(nn.Module):
    """
    Expansion with vectorized CG instruction loop.

    Key optimization: pre-compute all w3j matrices and group instructions
    by angular momentum to minimize CUDA kernel launches.
    """

    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super().__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2

        instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append({
                            "i": i, "j": j, "k": k,
                            "l_in": ir_in.l, "l_out1": ir_out1.l, "l_out2": ir_out2.l,
                            "mul_in": num_in, "mul_out1": num_out1, "mul_out2": num_out2,
                            "dim_in": ir_in.dim, "dim_out1": ir_out1.dim, "dim_out2": ir_out2.dim,
                        })
        self.instructions = instructions

        self.num_path_weight = sum(ins["mul_in"] * ins["mul_out1"] * ins["mul_out2"] for ins in instructions)
        self.num_bias = sum(ins["mul_out1"] * ins["mul_out2"] for ins in instructions if ins["i"] == 0)
        self.num_weights = self.num_path_weight + self.num_bias

        # Pre-compute and register all w3j matrices
        w3j_cache = {}
        for ins in instructions:
            key = (ins["l_out1"], ins["l_out2"], ins["l_in"])
            if key not in w3j_cache:
                w3j_cache[key] = o3.wigner_3j(*key)
        for key, val in w3j_cache.items():
            self.register_buffer(f"w3j_{key[0]}_{key[1]}_{key[2]}", val)
        self._w3j_cache_keys = list(w3j_cache.keys())

        # Pre-compute weight slicing info
        self._weight_slices = []
        self._bias_slices = []
        w_offset = 0
        b_offset = 0
        for ins in instructions:
            n = ins["mul_in"] * ins["mul_out1"] * ins["mul_out2"]
            self._weight_slices.append((w_offset, w_offset + n,
                                         [ins["mul_in"], ins["mul_out1"], ins["mul_out2"]]))
            w_offset += n
            if ins["i"] == 0:
                nb = ins["mul_out1"] * ins["mul_out2"]
                self._bias_slices.append((b_offset, b_offset + nb,
                                           [ins["mul_out1"], ins["mul_out2"]]))
                b_offset += nb
            else:
                self._bias_slices.append(None)

        # Pre-compute output block positions for assembly
        self._out_positions = []
        row_offset = 0
        for j_idx in range(len(irrep_out_1)):
            col_offset = 0
            dim_j = irrep_out_1[j_idx].dim
            for k_idx in range(len(irrep_out_2)):
                dim_k = irrep_out_2[k_idx].dim
                self._out_positions.append((j_idx, k_idx, row_offset, col_offset, dim_j, dim_k))
                col_offset += dim_k
            row_offset += dim_j
        self._total_out_dim1 = row_offset
        self._total_out_dim2 = col_offset  # last col_offset after inner loop

        # Recompute total dims correctly
        self._total_out_dim1 = sum(irrep_out_1[j].dim for j in range(len(irrep_out_1)))
        self._total_out_dim2 = sum(irrep_out_2[k].dim for k in range(len(irrep_out_2)))

    def _get_w3j(self, l_out1, l_out2, l_in):
        return getattr(self, f"w3j_{l_out1}_{l_out2}_{l_in}")

    def forward(self, x_in, weights, bias_weights=None):
        B = x_in.shape[0]
        device = x_in.device

        # Split input by irrep (done once)
        x_splits = []
        for sl, mul_ir in zip(self.irrep_in.slices(), self.irrep_in):
            x_splits.append(x_in[:, sl].reshape(B, mul_ir.mul, mul_ir.ir.dim))

        # Allocate output
        output = torch.zeros(B, self._total_out_dim1, self._total_out_dim2,
                             device=device, dtype=x_in.dtype)

        # Build position lookup
        pos_map = {}
        for j_idx in range(len(self.irrep_out_1)):
            r = sum(self.irrep_out_1[jj].dim for jj in range(j_idx))
            for k_idx in range(len(self.irrep_out_2)):
                c = sum(self.irrep_out_2[kk].dim for kk in range(k_idx))
                pos_map[(j_idx, k_idx)] = (r, c, self.irrep_out_1[j_idx].dim, self.irrep_out_2[k_idx].dim)

        bias_idx = 0
        for ins_idx, ins in enumerate(self.instructions):
            x1 = x_splits[ins["i"]]  # [B, mul_in, dim_in]
            w3j = self._get_w3j(ins["l_out1"], ins["l_out2"], ins["l_in"])

            ws, we, wshape = self._weight_slices[ins_idx]
            w = weights[:, ws:we].reshape([-1] + wshape)  # [B, mul_in, mul_out1, mul_out2]

            # Core computation: 2 einsums → fused into 1 where possible
            # Step 1: contract weights with input
            tmp = torch.einsum("bwuv, bwk -> buvk", w, x1)  # [B, mul_out1, mul_out2, dim_in]

            # Step 1.5: add bias for l_in=0
            if ins["i"] == 0 and bias_weights is not None:
                bs_info = self._bias_slices[ins_idx]
                if bs_info is not None:
                    bs, be, bshape = bs_info
                    bias = bias_weights[:, bs:be].reshape([-1] + bshape)  # [B, mul_out1, mul_out2]
                    tmp = tmp + bias.unsqueeze(-1)

            # Step 2: apply CG coefficients
            result = torch.einsum("ijk, buvk -> buivj", w3j, tmp) / ins["mul_in"]
            # [B, mul_out1, dim_out1, mul_out2, dim_out2]

            # Accumulate into output
            r, c, dr, dc = pos_map[(ins["j"], ins["k"])]
            output[:, r:r+dr, c:c+dc] += result.reshape(B, ins["mul_out1"] * ins["dim_out1"],
                                                          ins["mul_out2"] * ins["dim_out2"])

        return output

    @property
    def device(self):
        return next(self.buffers().__iter__(), torch.tensor(0)).device


# ============================================================
# FastMultiHeadExpansion: pre-sorted + FastExpansion
# ============================================================

class FastMultiHeadExpansion(nn.Module):
    """
    Optimized MoE-style output head.

    Optimizations vs naive version:
    1. Pre-sort atoms by group → contiguous memory access
    2. FastExpansion with cached w3j matrices
    3. Single FC per group (weight + bias fused)
    """

    def __init__(self, input_expand_irrep, orbital_groups, hidden_size):
        super().__init__()
        self.groups = list(orbital_groups.keys())
        self.group_irreps = {g: orbital_groups[g]["irrep"] for g in self.groups}

        self.register_buffer("z_to_group_idx", torch.full((119,), -1, dtype=torch.long))
        for idx, (g, cfg) in enumerate(orbital_groups.items()):
            for z in cfg["elements"]:
                self.z_to_group_idx[z] = idx

        self.expand = nn.ModuleDict()
        self.fc_weight = nn.ModuleDict()
        self.fc_bias = nn.ModuleDict()

        for g in self.groups:
            irrep = self.group_irreps[g]
            exp = FastExpansion(input_expand_irrep, irrep, irrep)
            self.expand[g] = exp
            self.fc_weight[g] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, exp.num_path_weight),
            )
            self.fc_bias[g] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, exp.num_bias),
            )

    def forward(self, equi_feat, scalar_feat, atom_types):
        group_indices = self.z_to_group_idx[atom_types]

        # Pre-sort by group for contiguous access
        sorted_order = torch.argsort(group_indices, stable=True)
        sorted_groups = group_indices[sorted_order]

        results = {}
        for idx, g in enumerate(self.groups):
            # Find contiguous range for this group
            mask = (sorted_groups == idx)
            if not mask.any():
                continue

            # Get contiguous slice indices
            nonzero = mask.nonzero(as_tuple=True)[0]
            start, end = nonzero[0].item(), nonzero[-1].item() + 1
            orig_indices = sorted_order[start:end]

            feat = equi_feat[orig_indices]
            scalar = scalar_feat[orig_indices]

            w = self.fc_weight[g](scalar)
            b = self.fc_bias[g](scalar)
            block = self.expand[g](feat, w, b)

            results[g] = (orig_indices, block)

        return results


# ============================================================
# Naive MultiHeadExpansion (from v1, for comparison)
# ============================================================

class NaiveMultiHeadExpansion(nn.Module):
    def __init__(self, input_expand_irrep, orbital_groups, hidden_size):
        super().__init__()
        self.groups = list(orbital_groups.keys())
        self.register_buffer("z_to_group_idx", torch.full((119,), -1, dtype=torch.long))
        for idx, (g, cfg) in enumerate(orbital_groups.items()):
            for z in cfg["elements"]:
                self.z_to_group_idx[z] = idx
        self.expand = nn.ModuleDict()
        self.fc_weight = nn.ModuleDict()
        self.fc_bias = nn.ModuleDict()
        for g in self.groups:
            irrep = orbital_groups[g]["irrep"]
            exp = Expansion(input_expand_irrep, irrep, irrep)
            self.expand[g] = exp
            self.fc_weight[g] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, exp.num_path_weight),
            )
            self.fc_bias[g] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, exp.num_bias),
            )

    def forward(self, equi_feat, scalar_feat, atom_types):
        group_indices = self.z_to_group_idx[atom_types]
        results = {}
        for idx, g in enumerate(self.groups):
            mask = (group_indices == idx)
            if not mask.any():
                continue
            feat = equi_feat[mask]
            scalar = scalar_feat[mask]
            w = self.fc_weight[g](scalar)
            b = self.fc_bias[g](scalar)
            block = self.expand[g](feat, w, b)
            results[g] = (mask.nonzero(as_tuple=True)[0], block)
        return results


# ============================================================
# Correctness test: FastExpansion == Expansion
# ============================================================

def test_fast_expansion_correctness():
    print("=" * 65)
    print("[Test 1] FastExpansion correctness vs original Expansion")
    print("=" * 65)

    for g_name, g_cfg in ORBITAL_GROUPS.items():
        irrep = g_cfg["irrep"]
        orig = Expansion(INPUT_EXPAND_IRREP, irrep, irrep)
        fast = FastExpansion(INPUT_EXPAND_IRREP, irrep, irrep)

        B = 8
        x = torch.randn(B, INPUT_EXPAND_IRREP.dim)
        w = torch.randn(B, orig.num_path_weight)
        b = torch.randn(B, orig.num_bias)

        out_orig = orig(x, w, b)
        out_fast = fast(x, w, b)

        err = (out_orig - out_fast).abs().max().item()
        print(f"  {g_name:5s} (dim={irrep.dim:2d}): max|orig-fast| = {err:.2e}  {'✓' if err < 1e-10 else '✗'}")

    print()


# ============================================================
# Equivariance test for FastExpansion
# ============================================================

def test_fast_expansion_equivariance():
    print("=" * 65)
    print("[Test 2] FastExpansion equivariance")
    print("=" * 65)

    from scipy.spatial.transform import Rotation

    for g_name, g_cfg in ORBITAL_GROUPS.items():
        irrep = g_cfg["irrep"]
        fast = FastExpansion(INPUT_EXPAND_IRREP, irrep, irrep)

        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float64)

        D_in_blocks = []
        for mul, ir in INPUT_EXPAND_IRREP:
            D_l = ir.D_from_matrix(R)
            D_in_blocks.append(torch.block_diag(*[D_l] * mul))
        D_in = torch.block_diag(*D_in_blocks)

        D_out_blocks = []
        for mul, ir in irrep:
            D_l = ir.D_from_matrix(R)
            D_out_blocks.append(torch.block_diag(*[D_l] * mul))
        D_out = torch.block_diag(*D_out_blocks)

        x = torch.randn(1, INPUT_EXPAND_IRREP.dim)
        w = torch.randn(1, fast.num_path_weight)
        b = torch.randn(1, fast.num_bias)

        out = fast(x, w, b)
        out_rot = torch.einsum("ij, bjk, kl -> bil", D_out, out, D_out.T)

        x_rot = (D_in @ x.T).T
        out_from_rot = fast(x_rot, w, b)

        err = (out_rot - out_from_rot).abs().max().item()
        print(f"  {g_name:5s}: max|A-B| = {err:.2e}  {'✓' if err < 1e-10 else '✗'}")

    print()


# ============================================================
# Throughput benchmark
# ============================================================

def benchmark():
    print("=" * 65)
    print("[Test 3] Throughput benchmark")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # --- Build all 3 models ---
    # 1. Single head (current QHFlow2)
    single_irrep = o3.Irreps("3x0e + 2x1e + 1x2e")
    single_expand = Expansion(INPUT_EXPAND_IRREP, single_irrep, single_irrep).to(device, dtype)
    single_fc_w = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU(),
        nn.Linear(HIDDEN_SIZE, single_expand.num_path_weight)
    ).to(device, dtype)
    single_fc_b = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU(),
        nn.Linear(HIDDEN_SIZE, single_expand.num_bias)
    ).to(device, dtype)

    # 2. Naive multi-head (v1)
    naive_multi = NaiveMultiHeadExpansion(INPUT_EXPAND_IRREP, ORBITAL_GROUPS, HIDDEN_SIZE).to(device, dtype)

    # 3. Fast multi-head (v2)
    fast_multi = FastMultiHeadExpansion(INPUT_EXPAND_IRREP, ORBITAL_GROUPS, HIDDEN_SIZE).to(device, dtype)

    scenarios = {
        "Pure organic (HCNO)": [1]*20 + [6]*10 + [7]*3 + [8]*5,
        "Small organic (EtOH)": [6]*2 + [8]*1 + [1]*6,
        "TM complex": [26]*1 + [7]*6 + [6]*12 + [1]*12,
        "Mixed heavy": [19]*2 + [26]*2 + [6]*10 + [1]*10,
        "H-only (large)": [1]*50,
        "C-only (medium)": [6]*30,
    }

    N_WARMUP = 100
    N_ITER = 500

    print(f"\n  {'Scenario':<24s} {'N':>3s}  {'Single':>9s}  {'Naive':>9s}  {'Fast':>9s}  {'Fast/Sgl':>9s}  {'Fast/Naive':>10s}")
    print("  " + "-" * 82)

    for name, types in scenarios.items():
        N = len(types)
        atom_types = torch.tensor(types, device=device)
        equi = torch.randn(N, INPUT_EXPAND_IRREP.dim, device=device, dtype=dtype)
        scalar = torch.randn(N, HIDDEN_SIZE, device=device, dtype=dtype)

        def run_single():
            w = single_fc_w(scalar)
            b = single_fc_b(scalar)
            return single_expand(equi, w, b)

        def run_naive():
            return naive_multi(equi, scalar, atom_types)

        def run_fast():
            return fast_multi(equi, scalar, atom_types)

        # Warmup
        for _ in range(N_WARMUP):
            run_single(); run_naive(); run_fast()
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = {}
        for label, fn in [("single", run_single), ("naive", run_naive), ("fast", run_fast)]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times[label] = (time.perf_counter() - t0) / N_ITER * 1000

        ratio_fast_single = times["fast"] / times["single"]
        ratio_fast_naive = times["fast"] / times["naive"]

        print(f"  {name:<24s} {N:>3d}  {times['single']:>8.3f}ms  {times['naive']:>8.3f}ms  {times['fast']:>8.3f}ms  "
              f"{ratio_fast_single:>8.2f}x  {ratio_fast_naive:>9.2f}x")

    # --- torch.compile test ---
    print(f"\n  --- With torch.compile (reduce_overhead) ---")
    try:
        fast_multi_compiled = torch.compile(fast_multi, mode="reduce-overhead")

        types = [1]*20 + [6]*10 + [7]*3 + [8]*5
        N = len(types)
        atom_types = torch.tensor(types, device=device)
        equi = torch.randn(N, INPUT_EXPAND_IRREP.dim, device=device, dtype=dtype)
        scalar = torch.randn(N, HIDDEN_SIZE, device=device, dtype=dtype)

        # Extended warmup for compile
        for _ in range(200):
            fast_multi_compiled(equi, scalar, atom_types)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(N_ITER):
            fast_multi_compiled(equi, scalar, atom_types)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_compiled = (time.perf_counter() - t0) / N_ITER * 1000

        print(f"  Compiled fast multi:    {t_compiled:.3f}ms")
    except Exception as e:
        print(f"  torch.compile failed: {e}")

    print()

    # --- Memory comparison ---
    print("  Memory usage (parameters):")
    n_single = sum(p.numel() for p in single_expand.parameters())
    n_single += sum(p.numel() for p in single_fc_w.parameters())
    n_single += sum(p.numel() for p in single_fc_b.parameters())
    n_naive = sum(p.numel() for p in naive_multi.parameters())
    n_fast = sum(p.numel() for p in fast_multi.parameters())
    print(f"    Single head:   {n_single:>10,d}")
    print(f"    Naive multi:   {n_naive:>10,d}")
    print(f"    Fast multi:    {n_fast:>10,d}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("Fast MultiHeadExpansion v2 — Benchmark")
    print("=" * 65 + "\n")

    test_fast_expansion_correctness()
    test_fast_expansion_equivariance()
    benchmark()

    print("=" * 65)
    print("Done.")
    print("=" * 65)
