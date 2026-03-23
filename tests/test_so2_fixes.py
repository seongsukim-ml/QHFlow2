"""
Comprehensive test: three SO2Expansion fixes vs original vs CG baseline.

Option A: Complex SH intermediate (precomputed effective CG coefficients)
Option B: Z-axis alignment (Rodrigues → Z, delta_m works in real SH for Rz)
Option C: CG Expansion fallback (standard Wigner-3j, no edge direction)
"""

import sys, os, builtins, types, importlib.util
from math import prod
builtins.prod = prod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
modules_mod = types.ModuleType('models.modules')
modules_mod.__dict__['InnerProduct'] = None
sys.modules['models.modules'] = modules_mod
sys.modules['models'] = types.ModuleType('models')

spec = importlib.util.spec_from_file_location(
    'models.layers',
    os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'layers.py'),
    submodule_search_locations=[],
)
layers = importlib.util.module_from_spec(spec)
layers.__package__ = 'models'
sys.modules['models.layers'] = layers
spec.loader.exec_module(layers)

import torch
import torch.nn as nn
from e3nn import o3
import time

Expansion = layers.Expansion
SO2Expansion = layers.SO2Expansion

# ============================================================
# Helper: real-to-complex SH change-of-basis matrix
# ============================================================

def real_to_complex_sh(l):
    """U matrix: U @ x_real = x_complex, for angular momentum l.

    Computed empirically by diagonalizing e3nn's D_l(Rz) to find the
    eigenvectors that map e3nn's real SH basis to the complex SH basis
    where D(Rz) is diagonal with eigenvalues exp(-i*m*phi).

    This correctly handles e3nn's non-standard real SH ordering
    (e.g., l=1: x, y, z instead of the physics convention y, z, x).
    """
    import math
    dim = 2 * l + 1
    if l == 0:
        return torch.eye(1, dtype=torch.complex128)

    # Construct Rz(phi) for a non-special angle
    phi = 0.7  # arbitrary
    cp, sp = math.cos(phi), math.sin(phi)
    Rz = torch.tensor([[cp, -sp, 0.], [sp, cp, 0.], [0., 0., 1.]], dtype=torch.float64)

    # e3nn's real Wigner-D for this rotation
    D_real = o3.Irrep(l, 1).D_from_matrix(Rz).to(torch.float64)

    # Eigendecompose: eigenvalues should be exp(-i*m*phi), m = -l..l
    eigenvalues, eigenvectors = torch.linalg.eig(D_real.to(torch.complex128))

    # Match each eigenvalue to m quantum number
    U = torch.zeros(dim, dim, dtype=torch.complex128)
    used = set()
    for m in range(-l, l + 1):
        target = torch.exp(torch.tensor(-1j * m * phi, dtype=torch.complex128))
        best_idx, best_diff = -1, float('inf')
        for idx in range(dim):
            if idx in used:
                continue
            diff = abs(eigenvalues[idx] - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        assert best_diff < 1e-5, f"No matching eigenvalue for m={m}, l={l}, diff={best_diff:.2e}"
        used.add(best_idx)
        U[m + l, :] = eigenvectors[:, best_idx]

    # Normalize rows to ensure unitarity
    for i in range(dim):
        U[i] /= U[i].norm()

    # Fix global phase: make the largest-magnitude element of each row real & positive
    for i in range(dim):
        idx_max = U[i].abs().argmax()
        phase = U[i, idx_max] / U[i, idx_max].abs()
        U[i] /= phase

    return U


def _verify_U_matrix(l):
    """Verify U by checking:
    1. U is unitary
    2. U @ D_real(Rz) @ U† is diagonal (exp(-i*m*phi))
    3. U @ D_real(R) @ U† gives valid complex Wigner-D
    """
    import math
    U = real_to_complex_sh(l)
    dim = 2 * l + 1
    eye = torch.eye(dim, dtype=torch.complex128)

    # 1. Unitarity
    err = (U @ U.conj().T - eye).abs().max().item()
    assert err < 1e-5, f"U not unitary for l={l}: {err:.2e}"

    # 2. Diagonalizes Rz
    phi = 1.23
    cp, sp = math.cos(phi), math.sin(phi)
    Rz = torch.tensor([[cp, -sp, 0.], [sp, cp, 0.], [0., 0., 1.]], dtype=torch.float64)
    D_real = o3.Irrep(l, 1).D_from_matrix(Rz).to(torch.complex128)
    D_complex = U @ D_real @ U.conj().T
    offdiag = D_complex.clone()
    offdiag[range(dim), range(dim)] = 0
    assert offdiag.abs().max() < 1e-5, f"D_complex(Rz) not diagonal for l={l}: {offdiag.abs().max():.2e}"

    # Check eigenvalues match exp(-i*m*phi)
    for m in range(-l, l + 1):
        expected = torch.exp(torch.tensor(-1j * m * phi, dtype=torch.complex128))
        actual = D_complex[m + l, m + l]
        assert abs(actual - expected) < 1e-5, f"Wrong eigenvalue for l={l}, m={m}: {abs(actual-expected):.2e}"

    return True


# ============================================================
# Option A: Complex SH Fix
# ============================================================

class SO2Expansion_ComplexSH(nn.Module):
    """SO2Expansion with complex SH intermediate for correct equivariance.

    Precomputes 'effective CG' coefficients that map real SH input features
    to real SH output matrix via complex SH delta_m coupling.

    For each path (l_in, l1, l2) and each band d:
        coupling_d[m1_r, m2_r] = Σ_{m1_c: valid} U†_l1[m1_r, m1_c] * U_l2[m1_c-d, m2_r]

    In forward, the rotation block becomes:
        rot_block_d = D1 @ Re(coupling_d * phase) @ D2^T
    where phase comes from the complex feature extraction.
    """

    def __init__(self, irrep_in, irrep_out_1, irrep_out_2, bandwidth=0):
        super().__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.bandwidth = bandwidth

        # Build CG paths (same as original)
        self.instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        self.instructions.append({
                            'i': i, 'j': j, 'k': k,
                            'l_in': ir_in.l, 'l1': ir_out1.l, 'l2': ir_out2.l,
                            'mul_in': num_in, 'mul1': num_out1, 'mul2': num_out2,
                            'muls': [num_in, num_out1, num_out2],
                        })

        self._num_path_weight = sum(prod(ins['muls']) for ins in self.instructions)
        self._num_bias = sum(
            ins['mul1'] * ins['mul2']
            for ins in self.instructions if ins['l_in'] == 0
        )
        self.num_weights = self._num_path_weight + self._num_bias

        self.lmax = max(
            max((ir.l for _, ir in irrep_in), default=0),
            max((ir.l for _, ir in irrep_out_1), default=0),
            max((ir.l for _, ir in irrep_out_2), default=0),
        )

        # Precompute U matrices for each l
        U_dict = {}
        for l in range(self.lmax + 1):
            U_dict[l] = real_to_complex_sh(l)

        # Precompute effective CG: for each path, for each band d,
        # compute the coupling matrix and U_in row.
        # We store: U_in_rows[path_idx][d_idx] = U_in[d+l_in, :] (complex, 1D)
        #           coupling[path_idx][d_idx] = (2l1+1, 2l2+1) complex matrix
        self._path_data = []
        for ins in self.instructions:
            l_in, l1, l2 = ins['l_in'], ins['l1'], ins['l2']
            U_in = U_dict[l_in]
            U1 = U_dict[l1]
            U2 = U_dict[l2]
            max_dm = min(l_in, self.bandwidth)
            n_bands = 2 * max_dm + 1

            # U_in rows for feature extraction: shape (n_bands, 2*l_in+1)
            u_in_rows = torch.stack([
                U_in[d + l_in, :] for d in range(-max_dm, max_dm + 1)
            ])  # (n_bands, 2*l_in+1) complex

            # Coupling matrices: for each d, coupling_d[m1_r, m2_r]
            couplings = []
            for d in range(-max_dm, max_dm + 1):
                C_d = torch.zeros(2*l1+1, 2*l2+1, dtype=torch.complex128)
                for m1_c in range(-l1, l1 + 1):
                    m2_c = m1_c - d
                    if abs(m2_c) <= l2:
                        # C_d[m1_r, m2_r] += U1†[m1_r, m1_c] * U2[m2_c, m2_r]
                        m1_c_idx = m1_c + l1
                        m2_c_idx = m2_c + l2
                        C_d += (U1.conj().T[:, m1_c_idx:m1_c_idx+1]
                                @ U2[m2_c_idx:m2_c_idx+1, :])
                couplings.append(C_d)
            coupling_stack = torch.stack(couplings)  # (n_bands, 2l1+1, 2l2+1)

            self._path_data.append({
                'u_in_rows_real': u_in_rows.real.float(),   # (n_bands, 2l_in+1)
                'u_in_rows_imag': u_in_rows.imag.float(),
                'coupling_real': coupling_stack.real.float(),  # (n_bands, 2l1+1, 2l2+1)
                'coupling_imag': coupling_stack.imag.float(),
                'max_dm': max_dm,
            })

    @property
    def num_path_weight(self):
        return self._num_path_weight

    @property
    def num_bias(self):
        return self._num_bias

    def _compute_wigner_D(self, edge_vec):
        """Same Rodrigues + e3nn Wigner-D as original."""
        edge_hat = torch.nn.functional.normalize(edge_vec, dim=-1)
        B = edge_hat.shape[0]
        device, dtype = edge_hat.device, edge_hat.dtype

        y_axis = edge_hat.new_tensor([0., 1., 0.]).expand(B, -1)
        v = torch.linalg.cross(y_axis, edge_hat)
        c = (y_axis * edge_hat).sum(-1)
        s_sq = (v * v).sum(-1)

        vx = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        vx[:, 0, 1] = -v[:, 2]; vx[:, 0, 2] = v[:, 1]
        vx[:, 1, 0] = v[:, 2];  vx[:, 1, 2] = -v[:, 0]
        vx[:, 2, 0] = -v[:, 1]; vx[:, 2, 1] = v[:, 0]

        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        factor = ((1 - c) / (s_sq + 1e-8)).reshape(B, 1, 1)
        R = eye + vx + torch.bmm(vx, vx) * factor

        anti = c < -0.99
        if anti.any():
            R[anti] = edge_hat.new_tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])

        angles = o3.matrix_to_angles(R.detach().cpu())
        return {
            l: o3.wigner_D(l, *angles).to(device=device, dtype=dtype)
            for l in range(self.lmax + 1)
        }

    def forward(self, x_in, weights=None, bias_weights=None, edge_vec=None):
        assert edge_vec is not None
        B = x_in.shape[0]
        device, dtype = x_in.device, x_in.dtype

        # 1. Parse input features per l
        x_in_s = [
            x_in[:, sl].reshape(B, mul_ir.mul, mul_ir.ir.dim)
            for sl, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)
        ]

        # 2. Rotate to local frame
        Ds = self._compute_wigner_D(edge_vec)
        x_local_s = []
        for idx, (_, ir_in) in enumerate(self.irrep_in):
            D_inv = Ds[ir_in.l].transpose(-1, -2)
            x_local_s.append(torch.einsum('bij,bcj->bci', D_inv, x_in_s[idx]))

        # 3. For each path, compute using complex SH coupling
        block_outputs = {}
        w_off, b_off = 0, 0

        for path_idx, ins in enumerate(self.instructions):
            i, j, k = ins['i'], ins['j'], ins['k']
            l_in, l1, l2 = ins['l_in'], ins['l1'], ins['l2']
            mul_in, mul1, mul2 = ins['mul_in'], ins['mul1'], ins['mul2']
            n_w = mul_in * mul1 * mul2
            pd = self._path_data[path_idx]

            x_l = x_local_s[i]  # [B, mul_in, 2*l_in+1]

            if weights is not None:
                w = weights[:, w_off:w_off + n_w].reshape(B, mul_in, mul1, mul2)
            else:
                w = torch.ones(1, mul_in, mul1, mul2, device=device, dtype=dtype) / mul_in
            w_off += n_w

            max_dm = pd['max_dm']
            u_real = pd['u_in_rows_real'].to(device=device, dtype=dtype)  # (n_bands, d_in)
            u_imag = pd['u_in_rows_imag'].to(device=device, dtype=dtype)
            coup_real = pd['coupling_real'].to(device=device, dtype=dtype)  # (n_bands, d1, d2)
            coup_imag = pd['coupling_imag'].to(device=device, dtype=dtype)

            D1, D2 = Ds[l1], Ds[l2]

            # Extract complex features: x_complex[d] = U_in[d+l_in, :] @ x_local
            # Real part: u_real @ x_local, Imag part: u_imag @ x_local
            # feat_real[b, w, d] = Σ_m u_real[d, m] * x_l[b, w, m]
            feat_real = torch.einsum('dm, bwm -> bwd', u_real, x_l)  # (B, mul_in, n_bands)
            feat_imag = torch.einsum('dm, bwm -> bwd', u_imag, x_l)

            # h_real + i*h_imag = Σ_w weight * (feat_real + i*feat_imag) / mul_in
            h_real = torch.einsum('bwd, bwuv -> buvd', feat_real, w) / mul_in
            h_imag = torch.einsum('bwd, bwuv -> buvd', feat_imag, w) / mul_in

            # Bias (only for l_in=0, applied to d=0 band)
            if l_in == 0 and bias_weights is not None:
                n_b = mul1 * mul2
                bias = bias_weights[:, b_off:b_off + n_b].reshape(B, mul1, mul2)
                h_real[:, :, :, max_dm] = h_real[:, :, :, max_dm] + bias
                b_off += n_b

            # Rotation: for each band d,
            #   rot_real_d = D1 @ coup_real_d @ D2^T
            #   rot_imag_d = D1 @ coup_imag_d @ D2^T
            # Then H_block += h_real_d * rot_real_d - h_imag_d * rot_imag_d
            # Vectorized over bands:
            rot_real = torch.einsum('bim, dmn, bjn -> bdij', D1, coup_real, D2)  # (B, n_bands, d1, d2)
            rot_imag = torch.einsum('bim, dmn, bjn -> bdij', D1, coup_imag, D2)

            # H_block[b, u, i, v, j] = Σ_d (h_real[b,u,v,d]*rot_real[b,d,i,j]
            #                              - h_imag[b,u,v,d]*rot_imag[b,d,i,j])
            H_spatial = torch.einsum('buvd, bdij -> buivj', h_real, rot_real) \
                      - torch.einsum('buvd, bdij -> buivj', h_imag, rot_imag)

            # Reshape: [B, mul1, d1, mul2, d2] -> [B, mul1*d1, mul2*d2]
            # NO permute needed — direct reshape is correct
            H_block = H_spatial.reshape(B, mul1 * (2*l1+1), mul2 * (2*l2+1))

            key = (j, k)
            if key in block_outputs:
                block_outputs[key] = block_outputs[key] + H_block
            else:
                block_outputs[key] = H_block

        # 4. Assemble
        rows = []
        for ji in range(len(self.irrep_out_1)):
            cols = []
            for ki in range(len(self.irrep_out_2)):
                if (ji, ki) in block_outputs:
                    cols.append(block_outputs[(ji, ki)])
                else:
                    cols.append(torch.zeros(
                        B, self.irrep_out_1[ji].dim, self.irrep_out_2[ki].dim,
                        device=device, dtype=dtype))
            rows.append(torch.cat(cols, dim=-1))
        return torch.cat(rows, dim=-2)

    def __repr__(self):
        return f"SO2Expansion_ComplexSH(bw={self.bandwidth}, weights={self.num_path_weight})"


# ============================================================
# Option B: Z-axis alignment
# ============================================================

class SO2Expansion_ZAxis(nn.Module):
    """SO2Expansion with Z-axis alignment.

    Aligns bond with Z-axis (not Y), then uses complex SH delta_m coupling
    (same effective CG approach as Option A, just different local frame axis).
    """

    def __init__(self, irrep_in, irrep_out_1, irrep_out_2, bandwidth=0):
        super().__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.bandwidth = bandwidth

        self.instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        self.instructions.append({
                            'i': i, 'j': j, 'k': k,
                            'l_in': ir_in.l, 'l1': ir_out1.l, 'l2': ir_out2.l,
                            'mul_in': num_in, 'mul1': num_out1, 'mul2': num_out2,
                            'muls': [num_in, num_out1, num_out2],
                        })

        self._num_path_weight = sum(prod(ins['muls']) for ins in self.instructions)
        self._num_bias = sum(
            ins['mul1'] * ins['mul2']
            for ins in self.instructions if ins['l_in'] == 0
        )
        self.num_weights = self._num_path_weight + self._num_bias

        self.lmax = max(
            max((ir.l for _, ir in irrep_in), default=0),
            max((ir.l for _, ir in irrep_out_1), default=0),
            max((ir.l for _, ir in irrep_out_2), default=0),
        )

        # Same effective CG precomputation as Option A
        U_dict = {}
        for l in range(self.lmax + 1):
            U_dict[l] = real_to_complex_sh(l)

        self._path_data = []
        for ins in self.instructions:
            l_in, l1, l2 = ins['l_in'], ins['l1'], ins['l2']
            U_in, U1, U2 = U_dict[l_in], U_dict[l1], U_dict[l2]
            max_dm = min(l_in, self.bandwidth)

            u_in_rows = torch.stack([
                U_in[d + l_in, :] for d in range(-max_dm, max_dm + 1)
            ])

            couplings = []
            for d in range(-max_dm, max_dm + 1):
                C_d = torch.zeros(2*l1+1, 2*l2+1, dtype=torch.complex128)
                for m1_c in range(-l1, l1 + 1):
                    m2_c = m1_c - d
                    if abs(m2_c) <= l2:
                        m1_c_idx, m2_c_idx = m1_c + l1, m2_c + l2
                        C_d += (U1.conj().T[:, m1_c_idx:m1_c_idx+1]
                                @ U2[m2_c_idx:m2_c_idx+1, :])
                couplings.append(C_d)
            coupling_stack = torch.stack(couplings)

            self._path_data.append({
                'u_in_rows_real': u_in_rows.real.float(),
                'u_in_rows_imag': u_in_rows.imag.float(),
                'coupling_real': coupling_stack.real.float(),
                'coupling_imag': coupling_stack.imag.float(),
                'max_dm': max_dm,
            })

    @property
    def num_path_weight(self):
        return self._num_path_weight

    @property
    def num_bias(self):
        return self._num_bias

    def _compute_wigner_D(self, edge_vec):
        """Rodrigues rotation: Z-axis → edge direction."""
        edge_hat = torch.nn.functional.normalize(edge_vec, dim=-1)
        B = edge_hat.shape[0]
        device, dtype = edge_hat.device, edge_hat.dtype

        # Z-axis alignment: R @ [0,0,1] = edge_hat
        z_axis = edge_hat.new_tensor([0., 0., 1.]).expand(B, -1)
        v = torch.linalg.cross(z_axis, edge_hat)
        c = (z_axis * edge_hat).sum(-1)
        s_sq = (v * v).sum(-1)

        vx = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        vx[:, 0, 1] = -v[:, 2]; vx[:, 0, 2] = v[:, 1]
        vx[:, 1, 0] = v[:, 2];  vx[:, 1, 2] = -v[:, 0]
        vx[:, 2, 0] = -v[:, 1]; vx[:, 2, 1] = v[:, 0]

        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        factor = ((1 - c) / (s_sq + 1e-8)).reshape(B, 1, 1)
        R = eye + vx + torch.bmm(vx, vx) * factor

        anti = c < -0.99
        if anti.any():
            R[anti] = edge_hat.new_tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])

        angles = o3.matrix_to_angles(R.detach().cpu())
        return {
            l: o3.wigner_D(l, *angles).to(device=device, dtype=dtype)
            for l in range(self.lmax + 1)
        }

    def forward(self, x_in, weights=None, bias_weights=None, edge_vec=None):
        """Same as ComplexSH but with Z-axis Wigner-D."""
        assert edge_vec is not None
        B = x_in.shape[0]
        device, dtype = x_in.device, x_in.dtype

        x_in_s = [
            x_in[:, sl].reshape(B, mul_ir.mul, mul_ir.ir.dim)
            for sl, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)
        ]

        Ds = self._compute_wigner_D(edge_vec)
        x_local_s = []
        for idx, (_, ir_in) in enumerate(self.irrep_in):
            D_inv = Ds[ir_in.l].transpose(-1, -2)
            x_local_s.append(torch.einsum('bij,bcj->bci', D_inv, x_in_s[idx]))

        block_outputs = {}
        w_off, b_off = 0, 0

        for path_idx, ins in enumerate(self.instructions):
            i, j, k = ins['i'], ins['j'], ins['k']
            l_in, l1, l2 = ins['l_in'], ins['l1'], ins['l2']
            mul_in, mul1, mul2 = ins['mul_in'], ins['mul1'], ins['mul2']
            n_w = mul_in * mul1 * mul2
            pd = self._path_data[path_idx]

            x_l = x_local_s[i]

            if weights is not None:
                w = weights[:, w_off:w_off + n_w].reshape(B, mul_in, mul1, mul2)
            else:
                w = torch.ones(1, mul_in, mul1, mul2, device=device, dtype=dtype) / mul_in
            w_off += n_w

            max_dm = pd['max_dm']
            u_real = pd['u_in_rows_real'].to(device=device, dtype=dtype)
            u_imag = pd['u_in_rows_imag'].to(device=device, dtype=dtype)
            coup_real = pd['coupling_real'].to(device=device, dtype=dtype)
            coup_imag = pd['coupling_imag'].to(device=device, dtype=dtype)

            D1, D2 = Ds[l1], Ds[l2]

            feat_real = torch.einsum('dm, bwm -> bwd', u_real, x_l)
            feat_imag = torch.einsum('dm, bwm -> bwd', u_imag, x_l)

            h_real = torch.einsum('bwd, bwuv -> buvd', feat_real, w) / mul_in
            h_imag = torch.einsum('bwd, bwuv -> buvd', feat_imag, w) / mul_in

            if l_in == 0 and bias_weights is not None:
                n_b = mul1 * mul2
                bias = bias_weights[:, b_off:b_off + n_b].reshape(B, mul1, mul2)
                h_real[:, :, :, max_dm] = h_real[:, :, :, max_dm] + bias
                b_off += n_b

            rot_real = torch.einsum('bim, dmn, bjn -> bdij', D1, coup_real, D2)
            rot_imag = torch.einsum('bim, dmn, bjn -> bdij', D1, coup_imag, D2)

            H_spatial = torch.einsum('buvd, bdij -> buivj', h_real, rot_real) \
                      - torch.einsum('buvd, bdij -> buivj', h_imag, rot_imag)

            H_block = H_spatial.reshape(B, mul1 * (2*l1+1), mul2 * (2*l2+1))

            key = (j, k)
            if key in block_outputs:
                block_outputs[key] = block_outputs[key] + H_block
            else:
                block_outputs[key] = H_block

        rows = []
        for ji in range(len(self.irrep_out_1)):
            cols = []
            for ki in range(len(self.irrep_out_2)):
                if (ji, ki) in block_outputs:
                    cols.append(block_outputs[(ji, ki)])
                else:
                    cols.append(torch.zeros(
                        B, self.irrep_out_1[ji].dim, self.irrep_out_2[ki].dim,
                        device=device, dtype=dtype))
            rows.append(torch.cat(cols, dim=-1))
        return torch.cat(rows, dim=-2)

    def __repr__(self):
        return f"SO2Expansion_ZAxis(bw={self.bandwidth}, weights={self.num_path_weight})"


# ============================================================
# Tests
# ============================================================

def test_equivariance(name, expansion, irrep_in, irrep_out, B=16, device='cuda'):
    """Test equivariance: D_out @ H(x, e) @ D_out^T == H(D_in x, R e)."""
    x = torch.randn(B, irrep_in.dim, device=device)
    w = torch.randn(B, expansion.num_path_weight, device=device) * 0.01
    bw = torch.randn(B, expansion.num_bias, device=device) * 0.01
    edge_vec = torch.randn(B, 3, device=device)

    R = o3.rand_matrix()
    D_in = irrep_in.D_from_matrix(R).to(device)
    D_out = irrep_out.D_from_matrix(R).to(device)
    x_rot = (D_in @ x.unsqueeze(-1)).squeeze(-1)
    edge_rot = (R.to(device) @ edge_vec.unsqueeze(-1)).squeeze(-1)

    needs_edge = hasattr(expansion, '_compute_wigner_D')
    kw = {'edge_vec': edge_vec} if needs_edge else {}
    kw_rot = {'edge_vec': edge_rot} if needs_edge else {}

    out1 = expansion(x_rot, weights=w, bias_weights=bw, **kw_rot)
    out2 = expansion(x, weights=w, bias_weights=bw, **kw)
    out2_rot = D_out @ out2 @ D_out.T

    diff = (out1 - out2_rot).abs().max().item()
    norm = out2.norm().item()
    rel = diff / (norm + 1e-12)
    status = 'PASS' if diff < 1e-3 else 'FAIL'
    return diff, rel, norm, status


def benchmark_throughput(name, expansion, irrep_in, B=16, N=200, device='cuda'):
    """Benchmark throughput in iterations per second."""
    x = torch.randn(B, irrep_in.dim, device=device)
    w = torch.randn(B, expansion.num_path_weight, device=device) * 0.01
    bw = torch.randn(B, expansion.num_bias, device=device) * 0.01
    edge_vec = torch.randn(B, 3, device=device)

    needs_edge = hasattr(expansion, '_compute_wigner_D')
    kw = {'edge_vec': edge_vec} if needs_edge else {}

    # Warmup
    for _ in range(10):
        expansion(x, weights=w, bias_weights=bw, **kw)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N):
        expansion(x, weights=w, bias_weights=bw, **kw)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return N / elapsed, elapsed * 1000


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda'

    irrep_in = o3.Irreps('32x0e + 32x1e + 32x2e + 32x3e + 32x4e')
    irrep_out = o3.Irreps('3x0e + 2x1e + 1x2e')

    print("=" * 72)
    print("SO2Expansion Fix Comparison")
    print("=" * 72)
    print(f"Input:  {irrep_in}")
    print(f"Output: {irrep_out}")
    print()

    # Verify U matrices
    for l in range(5):
        _verify_U_matrix(l)
    print("U matrices verified for l=0..4 ✓\n")

    # Build all expansions
    expansions = {}

    # Original (broken)
    expansions['Original SO2 (broken)'] = SO2Expansion(
        irrep_in, irrep_out, irrep_out, bandwidth=1).to(device)

    # Option A: Complex SH fix
    for bw in [0, 1, 2, 4]:
        expansions[f'A: ComplexSH bw={bw}'] = SO2Expansion_ComplexSH(
            irrep_in, irrep_out, irrep_out, bandwidth=bw).to(device)

    # Option B: Z-axis fix
    for bw in [0, 1, 2]:
        expansions[f'B: ZAxis bw={bw}'] = SO2Expansion_ZAxis(
            irrep_in, irrep_out, irrep_out, bandwidth=bw).to(device)

    # Option C: CG Expansion (baseline)
    expansions['C: CG Expansion'] = Expansion(
        irrep_in, irrep_out, irrep_out).to(device)

    # ---- Equivariance Tests ----
    print("-" * 72)
    print(f"{'Method':<30} {'Max Diff':>10} {'Rel Err':>10} {'Norm':>8} {'Status':>6}")
    print("-" * 72)

    for name, exp in expansions.items():
        diff, rel, norm, status = test_equivariance(name, exp, irrep_in, irrep_out, device=device)
        mark = '✓' if status == 'PASS' else '✗'
        print(f"{name:<30} {diff:>10.2e} {rel:>10.2e} {norm:>8.4f} {mark:>6}")

    # ---- Throughput Benchmarks ----
    print()
    print("-" * 72)
    print(f"{'Method':<30} {'it/s':>8} {'time(ms)':>10} {'vs CG':>8}")
    print("-" * 72)

    cg_ips = None
    for name, exp in expansions.items():
        ips, ms = benchmark_throughput(name, exp, irrep_in, device=device)
        if 'CG' in name:
            cg_ips = ips
        ratio = f"{ips/cg_ips:.2f}x" if cg_ips else "—"
        print(f"{name:<30} {ips:>8.0f} {ms:>10.1f} {ratio:>8}")

    print()
    print("=" * 72)
    print("Done!")
