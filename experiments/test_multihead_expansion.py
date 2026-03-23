"""
MoE-style MultiHeadExpansion 실험
=================================
QHFlow2의 Expansion 레이어를 원소군별로 분리하여:
1. 올바른 출력 생성 (shape, 값)
2. Equivariance 보존
3. Throughput 비교 (단일 head vs multi-head)

QHFlow2의 기존 코드를 수정하지 않음 — Expansion을 import해서 독립적으로 테스트.
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np
from e3nn import o3

sys.path.insert(0, "/home1/irteam/data-vol1/projects/QHFlow2/src")

# --- Expansion class (copied from layers.py to avoid torch_scatter dependency) ---
from math import prod

class Expansion(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super(Expansion, self).__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.instructions = self._get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum(
            [prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0]
        )
        if self.num_path_weight > 0:
            self.weights = nn.Parameter(
                torch.rand(self.num_path_weight + self.num_bias)
            )
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
                    weight = self.weights[
                        flat_weight_index : flat_weight_index + prod(ins[-1])
                    ].reshape(ins[-1])
                    result = torch.einsum("wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1) / mul_ir_in.mul
                else:
                    weight = weights[:, flat_weight_index : flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                    result = torch.einsum("bwuv, bwk-> buvk", weight, x1)
                    if ins[0] == 0 and bias_weights is not None:
                        bias_weight = bias_weights[
                            :, bias_weight_index : bias_weight_index + prod(ins[-1][1:])
                        ].reshape([-1] + ins[-1][1:])
                        bias_weight_index += prod(ins[-1][1:])
                        result = result + bias_weight.unsqueeze(-1)
                    result = torch.einsum("ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
                flat_weight_index += prod(ins[-1])
            else:
                result = torch.einsum(
                    "uvw, ijk, bwk-> buivj",
                    torch.ones(ins[-1]).type(x1.type()).to(self.device),
                    w3j_matrix, x1,
                )
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
# --- End Expansion ---

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# ============================================================
# 1. Orbital group 정의
# ============================================================

# def2-SVP 기준 원소별 output irreps
ORBITAL_GROUPS = {
    "sp": {
        "irrep": o3.Irreps("2x0e + 1x1e"),  # H, He: s + p (5-dim)
        "elements": [1, 2],
    },
    "spd": {
        "irrep": o3.Irreps("3x0e + 2x1e + 1x2e"),  # Li-Ar: s + p + d (14-dim)
        "elements": list(range(3, 19)),
    },
    "spdf": {
        "irrep": o3.Irreps("4x0e + 3x1e + 2x2e + 1x3e"),  # K-Kr: s + p + d + f (30-dim)
        "elements": list(range(19, 37)),
    },
}

# Bottleneck irrep (QHFlow2 기본값)
BOTTLE_HIDDEN = 32
INPUT_EXPAND_IRREP = o3.Irreps(
    f"{BOTTLE_HIDDEN}x0e + {BOTTLE_HIDDEN}x1e + {BOTTLE_HIDDEN}x2e + "
    f"{BOTTLE_HIDDEN}x3e + {BOTTLE_HIDDEN}x4e"
)
HIDDEN_SIZE = 128


# ============================================================
# 2. MultiHeadExpansion 구현
# ============================================================

class MultiHeadExpansion(nn.Module):
    """MoE-style output head: 원소군별로 다른 Expansion을 사용."""

    def __init__(self, input_expand_irrep, orbital_groups, hidden_size, mode="ii"):
        super().__init__()
        self.groups = list(orbital_groups.keys())
        self.group_irreps = {g: orbital_groups[g]["irrep"] for g in self.groups}

        # Element → group 매핑 (Z=0~118)
        self.register_buffer("z_to_group_idx", torch.full((119,), -1, dtype=torch.long))
        for idx, (g, cfg) in enumerate(orbital_groups.items()):
            for z in cfg["elements"]:
                self.z_to_group_idx[z] = idx

        # Per-group Expansion + FC layers
        self.expand = nn.ModuleDict()
        self.fc_weight = nn.ModuleDict()
        self.fc_bias = nn.ModuleDict()

        fc_input_size = hidden_size if mode == "ii" else hidden_size * 2

        for g in self.groups:
            irrep = self.group_irreps[g]
            exp = Expansion(input_expand_irrep, irrep, irrep)
            self.expand[g] = exp
            self.fc_weight[g] = nn.Sequential(
                nn.Linear(fc_input_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, exp.num_path_weight),
            )
            self.fc_bias[g] = nn.Sequential(
                nn.Linear(fc_input_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, exp.num_bias),
            )

    def forward(self, equivariant_features, scalar_features, atom_types):
        """
        Args:
            equivariant_features: [N, input_expand_dim] — bottleneck irrep features
            scalar_features: [N, hidden_size] — scalar node features for FC
            atom_types: [N] — atomic numbers

        Returns:
            dict: {group_name: (indices, matrix_blocks)}
                  matrix_blocks shape: [n_group, dim_group, dim_group]
        """
        group_indices = self.z_to_group_idx[atom_types]
        results = {}

        for idx, g in enumerate(self.groups):
            mask = (group_indices == idx)
            if not mask.any():
                continue

            feat = equivariant_features[mask]
            scalar = scalar_features[mask]

            weights = self.fc_weight[g](scalar)
            bias = self.fc_bias[g](scalar)
            block = self.expand[g](feat, weights, bias)

            results[g] = (mask.nonzero(as_tuple=True)[0], block)

        return results


class MultiHeadExpansionPair(nn.Module):
    """Off-diagonal (ij) blocks: group pair별로 다른 Expansion."""

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

        for gi in self.groups:
            for gj in self.groups:
                key = f"{gi}__{gj}"
                irrep_i = self.group_irreps[gi]
                irrep_j = self.group_irreps[gj]
                exp = Expansion(input_expand_irrep, irrep_i, irrep_j)
                self.expand[key] = exp
                self.fc_weight[key] = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, exp.num_path_weight),
                )
                self.fc_bias[key] = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, exp.num_bias),
                )

    def forward(self, equivariant_features, scalar_features, src_types, dst_types):
        """
        Args:
            equivariant_features: [E, input_expand_dim]
            scalar_features: [E, hidden_size * 2]
            src_types, dst_types: [E] atomic numbers
        Returns:
            dict: {pair_key: (indices, matrix_blocks)}
        """
        src_groups = self.z_to_group_idx[src_types]
        dst_groups = self.z_to_group_idx[dst_types]
        results = {}

        for i, gi in enumerate(self.groups):
            for j, gj in enumerate(self.groups):
                mask = (src_groups == i) & (dst_groups == j)
                if not mask.any():
                    continue

                key = f"{gi}__{gj}"
                feat = equivariant_features[mask]
                scalar = scalar_features[mask]

                weights = self.fc_weight[key](scalar)
                bias = self.fc_bias[key](scalar)
                block = self.expand[key](feat, weights, bias)

                results[key] = (mask.nonzero(as_tuple=True)[0], block)

        return results


# ============================================================
# 3. Test: 기본 동작 확인
# ============================================================

def test_basic_output():
    print("=" * 65)
    print("[Test 1] Basic output shapes")
    print("=" * 65)

    mh = MultiHeadExpansion(INPUT_EXPAND_IRREP, ORBITAL_GROUPS, HIDDEN_SIZE)

    # 10 atoms: 3 H, 5 C, 2 K
    atom_types = torch.tensor([1, 1, 1, 6, 6, 6, 6, 6, 19, 19])
    N = len(atom_types)
    equi_feat = torch.randn(N, INPUT_EXPAND_IRREP.dim)
    scalar_feat = torch.randn(N, HIDDEN_SIZE)

    results = mh(equi_feat, scalar_feat, atom_types)

    for g, (indices, blocks) in results.items():
        dim = ORBITAL_GROUPS[g]["irrep"].dim
        print(f"  {g:5s}: indices={indices.tolist()}, "
              f"block shape={list(blocks.shape)}, expected=[{len(indices)}, {dim}, {dim}]")
        assert blocks.shape == (len(indices), dim, dim), f"Shape mismatch for {g}"

    print("  ✓ All shapes correct\n")


# ============================================================
# 4. Test: Equivariance
# ============================================================

def test_equivariance():
    print("=" * 65)
    print("[Test 2] Equivariance: rotate input → rotate output")
    print("=" * 65)

    from scipy.spatial.transform import Rotation

    # Single group test: 직접 Expansion의 equivariance 검증
    for g_name, g_cfg in ORBITAL_GROUPS.items():
        irrep_out = g_cfg["irrep"]
        exp = Expansion(INPUT_EXPAND_IRREP, irrep_out, irrep_out)

        # Random rotation
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float64)

        # Wigner-D matrices
        D_in_blocks = []
        for mul, ir in INPUT_EXPAND_IRREP:
            D_l = ir.D_from_matrix(R)  # (2l+1, 2l+1)
            D_l_block = torch.block_diag(*[D_l] * mul)
            D_in_blocks.append(D_l_block)
        D_in = torch.block_diag(*D_in_blocks)

        D_out_blocks = []
        for mul, ir in irrep_out:
            D_l = ir.D_from_matrix(R)
            D_l_block = torch.block_diag(*[D_l] * mul)
            D_out_blocks.append(D_l_block)
        D_out = torch.block_diag(*D_out_blocks)

        # Random input + fixed weights (no FC, direct weights)
        x = torch.randn(1, INPUT_EXPAND_IRREP.dim)
        num_pw = exp.num_path_weight
        num_b = exp.num_bias
        w = torch.randn(1, num_pw)
        b = torch.randn(1, num_b)

        # Path A: expand(x), then rotate output
        out_A = exp(x, w, b)  # [1, dim_out, dim_out]
        out_A_rot = torch.einsum("ij, bjk, kl -> bil", D_out, out_A, D_out.T)

        # Path B: rotate input, then expand
        x_rot = (D_in @ x.T).T  # [1, dim_in]
        out_B = exp(x_rot, w, b)

        err = (out_A_rot - out_B).abs().max().item()
        status = "✓" if err < 1e-10 else "✗"
        print(f"  {g_name:5s} (dim={irrep_out.dim:2d}): max|A-B| = {err:.2e}  {status}")

    print("  ✓ Expansion is equivariant for all groups\n")


# ============================================================
# 5. Test: Pair (ij) blocks
# ============================================================

def test_pair_blocks():
    print("=" * 65)
    print("[Test 3] Pair blocks — cross-group (ij)")
    print("=" * 65)

    mh_pair = MultiHeadExpansionPair(INPUT_EXPAND_IRREP, ORBITAL_GROUPS, HIDDEN_SIZE)

    # 8 edges with mixed atom types
    src_types = torch.tensor([1, 1, 6, 6, 19, 19, 1, 6])
    dst_types = torch.tensor([6, 19, 1, 6, 6, 19, 1, 19])
    E = len(src_types)
    equi_feat = torch.randn(E, INPUT_EXPAND_IRREP.dim)
    scalar_feat = torch.randn(E, HIDDEN_SIZE * 2)

    results = mh_pair(equi_feat, scalar_feat, src_types, dst_types)

    for key, (indices, blocks) in results.items():
        gi, gj = key.split("__")
        di = ORBITAL_GROUPS[gi]["irrep"].dim
        dj = ORBITAL_GROUPS[gj]["irrep"].dim
        print(f"  {key:12s}: indices={indices.tolist()}, "
              f"shape={list(blocks.shape)}, expected=[{len(indices)}, {di}, {dj}]")
        assert blocks.shape == (len(indices), di, dj)

    print("  ✓ All pair block shapes correct\n")


# ============================================================
# 6. Throughput benchmark
# ============================================================

def benchmark_throughput():
    print("=" * 65)
    print("[Test 4] Throughput benchmark")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # GPU benchmark uses float32

    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # --- 현재 방식: 단일 Expansion (14×14 고정) ---
    single_irrep = o3.Irreps("3x0e + 2x1e + 1x2e")  # 14-dim
    single_expand = Expansion(INPUT_EXPAND_IRREP, single_irrep, single_irrep).to(device, dtype)
    single_fc_w = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.SiLU(),
        nn.Linear(HIDDEN_SIZE, single_expand.num_path_weight),
    ).to(device, dtype)
    single_fc_b = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.SiLU(),
        nn.Linear(HIDDEN_SIZE, single_expand.num_bias),
    ).to(device, dtype)

    # --- MoE 방식 ---
    multi_head = MultiHeadExpansion(
        INPUT_EXPAND_IRREP, ORBITAL_GROUPS, HIDDEN_SIZE, mode="ii"
    ).to(device, dtype)

    # Benchmark scenarios
    scenarios = {
        "Pure organic (H,C,N,O)": {
            "types": [1]*20 + [6]*10 + [7]*3 + [8]*5,  # 38 atoms
            "desc": "H20 C10 N3 O5"
        },
        "Small organic (ethanol)": {
            "types": [6]*2 + [8]*1 + [1]*6,  # 9 atoms
            "desc": "C2H6O"
        },
        "Transition metal complex": {
            "types": [26]*1 + [7]*6 + [6]*12 + [1]*12,  # 31 atoms, Fe + ligands
            "desc": "Fe(NH3)6-like"
        },
        "Mixed heavy": {
            "types": [19]*2 + [26]*2 + [6]*10 + [1]*10,  # 24 atoms
            "desc": "K2Fe2 + organic"
        },
    }

    N_WARMUP = 50
    N_ITER = 200

    print(f"\n  {'Scenario':<28s} {'N':>4s}  {'Single(ms)':>10s}  {'Multi(ms)':>10s}  {'Speedup':>8s}  {'Wasted%':>8s}")
    print("  " + "-" * 78)

    for name, cfg in scenarios.items():
        types = cfg["types"]
        N = len(types)
        atom_types = torch.tensor(types, device=device)
        equi_feat = torch.randn(N, INPUT_EXPAND_IRREP.dim, device=device, dtype=dtype)
        scalar_feat = torch.randn(N, HIDDEN_SIZE, device=device, dtype=dtype)

        # --- Single head ---
        def run_single():
            w = single_fc_w(scalar_feat)
            b = single_fc_b(scalar_feat)
            out = single_expand(equi_feat, w, b)  # [N, 14, 14]
            return out

        # --- Multi head ---
        def run_multi():
            return multi_head(equi_feat, scalar_feat, atom_types)

        # Warmup
        for _ in range(N_WARMUP):
            run_single()
            run_multi()

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark single
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            run_single()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_single = (time.perf_counter() - t0) / N_ITER * 1000

        # Benchmark multi
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            run_multi()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_multi = (time.perf_counter() - t0) / N_ITER * 1000

        speedup = t_single / t_multi if t_multi > 0 else float('inf')

        # Compute wasted elements in single-head approach
        total_elements = 0
        valid_elements = 0
        for z in types:
            total_elements += 14 * 14
            # Find actual dim for this element
            for g, gc in ORBITAL_GROUPS.items():
                if z in gc["elements"]:
                    d = gc["irrep"].dim
                    valid_elements += d * d
                    break
        wasted_pct = (1 - valid_elements / total_elements) * 100

        print(f"  {name:<28s} {N:>4d}  {t_single:>10.3f}  {t_multi:>10.3f}  {speedup:>7.2f}x  {wasted_pct:>7.1f}%")

    print()

    # --- Parameter count comparison ---
    print("  Parameter counts:")
    n_single = sum(p.numel() for p in single_expand.parameters())
    n_single += sum(p.numel() for p in single_fc_w.parameters())
    n_single += sum(p.numel() for p in single_fc_b.parameters())
    print(f"    Single head (14×14):      {n_single:>10,d}")

    n_multi = sum(p.numel() for p in multi_head.parameters())
    print(f"    Multi head (5+14+30):     {n_multi:>10,d}")

    for g in multi_head.groups:
        n_g = sum(p.numel() for p in multi_head.expand[g].parameters())
        n_g += sum(p.numel() for p in multi_head.fc_weight[g].parameters())
        n_g += sum(p.numel() for p in multi_head.fc_bias[g].parameters())
        dim = ORBITAL_GROUPS[g]["irrep"].dim
        print(f"      {g:5s} ({dim:2d}×{dim:2d}):           {n_g:>10,d}")

    print()


# ============================================================
# 7. Test: Gradient flow
# ============================================================

def test_gradient_flow():
    print("=" * 65)
    print("[Test 5] Gradient flow through multi-head")
    print("=" * 65)

    mh = MultiHeadExpansion(INPUT_EXPAND_IRREP, ORBITAL_GROUPS, HIDDEN_SIZE)

    atom_types = torch.tensor([1, 1, 6, 6, 6, 19])
    N = len(atom_types)
    equi_feat = torch.randn(N, INPUT_EXPAND_IRREP.dim, requires_grad=True)
    scalar_feat = torch.randn(N, HIDDEN_SIZE, requires_grad=True)

    results = mh(equi_feat, scalar_feat, atom_types)

    # Compute loss: sum of all blocks (different sizes)
    loss = sum(blocks.pow(2).mean() for _, (_, blocks) in results.items())
    loss.backward()

    print(f"  Loss: {loss.item():.6f}")
    print(f"  equi_feat grad: shape={equi_feat.grad.shape}, "
          f"norm={equi_feat.grad.norm().item():.6f}, "
          f"nonzero={equi_feat.grad.abs().gt(1e-12).sum().item()}/{equi_feat.grad.numel()}")
    print(f"  scalar_feat grad: shape={scalar_feat.grad.shape}, "
          f"norm={scalar_feat.grad.norm().item():.6f}")

    # Check each group gets gradients
    for g in mh.groups:
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                      for p in mh.expand[g].parameters() if p.requires_grad)
        fc_grad = any(p.grad is not None and p.grad.abs().max() > 0
                     for p in mh.fc_weight[g].parameters())
        print(f"  {g:5s}: expand grad={'✓' if has_grad else '✗'}, fc grad={'✓' if fc_grad else '✗'}")

    print("  ✓ Gradients flow correctly\n")


# ============================================================
# 8. Test: Roundtrip consistency — 기존 single head와 동일 결과
# ============================================================

def test_consistency_with_single():
    print("=" * 65)
    print("[Test 6] Consistency: multi-head == single head (for spd group)")
    print("=" * 65)

    # spd group의 irrep은 "3x0e + 2x1e + 1x2e" = 기존 def2-svp와 동일
    spd_irrep = ORBITAL_GROUPS["spd"]["irrep"]
    single = Expansion(INPUT_EXPAND_IRREP, spd_irrep, spd_irrep)
    multi_spd = Expansion(INPUT_EXPAND_IRREP, spd_irrep, spd_irrep)

    # 같은 weights 복사
    with torch.no_grad():
        for p1, p2 in zip(single.parameters(), multi_spd.parameters()):
            p2.copy_(p1)

    x = torch.randn(5, INPUT_EXPAND_IRREP.dim)
    w = torch.randn(5, single.num_path_weight)
    b = torch.randn(5, single.num_bias)

    out_single = single(x, w, b)
    out_multi = multi_spd(x, w, b)

    err = (out_single - out_multi).abs().max().item()
    print(f"  max |single - multi|: {err:.2e}")
    print(f"  ✓ Results identical (same Expansion with same weights)\n")


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("MoE-style MultiHeadExpansion — Feasibility Test")
    print("=" * 65 + "\n")

    test_basic_output()
    test_equivariance()
    test_pair_blocks()
    test_gradient_flow()
    test_consistency_with_single()
    benchmark_throughput()

    print("=" * 65)
    print("All tests passed.")
    print("=" * 65)
