# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
"""QH9Stable SO(3) model and its two-element O(3) projection.

These are new architectures, not inference-time replacements for legacy
checkpoints. In particular, spectral gating replaces sampled-grid nonlinearities.
"""

import torch

from .QHFlow_so2_v5_1_ordering_fixed import QHFlow_escn_v5_1_ordering_fixed
from .modules.escn_backbone_v4 import eSCNMDBackbone_ham


class QH9SphericalBackbone(eSCNMDBackbone_ham):
    def _time_features_for_nodes(self, features, data_dict):
        if features.shape[0] == data_dict.pos.shape[0]:
            return features  # QH9 flow training samples one time per atom.
        if features.shape[0] == data_dict.ptr.numel() - 1:
            return features[data_dict.batch]
        raise ValueError("Expected one time per atom or per graph")

    def _matrix_features_for_nodes(self, features, data_dict):
        # QH9 input matrices are per atom, not per graph. Indexing them by
        # graph ID would broadcast the first atoms' blocks across molecules.
        if features.shape[0] != data_dict.pos.shape[0]:
            raise ValueError("Expected one Hamiltonian/overlap conditioning block per atom")
        return features

    def _get_rotmat_and_wigner(self, edge_distance_vecs, use_cuda_graph):
        # QH9's loader stores p AOs as [py, pz, px]. All spherical degrees
        # must use the same coordinate convention, not just the p block.
        return super()._get_rotmat_and_wigner(
            edge_distance_vecs[:, [1, 2, 0]], use_cuda_graph
        )


class QHFlow2_order_fixed(QHFlow_escn_v5_1_ordering_fixed):
    """SO(3)-equivariant QH9 model with corrected packing and spectral gates."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("sh_lmax", 4) != 4:
            raise ValueError("The current QHFlow2 matrix head requires sh_lmax=4")
        if kwargs.get("basis", "def2-svp") != "def2-svp":
            raise ValueError("These QH9Stable models use the def2-svp basis")
        super().__init__(*args, **kwargs)

    def _init_node_vector_backbone(self):
        self.node_attr_backbone = QH9SphericalBackbone(
            lmax=self.order, mmax=self.order,
            sphere_channels=self.hidden_size, hidden_channels=self.hidden_size,
            num_layers=self.num_gnn_layers,
            use_block_S=self.use_block_S, use_block_H=self.use_block_H,
            use_dataset_embedding=False, use_time_embedding=True,
            num_ham_gnn_layers=self.num_ham_gnn_layers,
            cutoff=self.esen_max_radius, ff_type="spectral", act_type="gate",
        )

    def forward(self, data, H, keep_blocks=False):
        if not keep_blocks:
            raise ValueError("QHFlow2 benchmark models require QH9 AO blocks (keep_blocks=True)")
        # The backbone sizes charge/spin from len(data). Populate both keys
        # first so adding one does not change the length used for the other.
        data = data.clone()
        data.charge = torch.zeros(data.ptr.numel() - 1, dtype=torch.long, device=H.device)
        data.spin = torch.zeros_like(data.charge)
        return super().forward(data, H, keep_blocks=True)


class QHFlow2_O3(QHFlow2_order_fixed):
    """Reynolds projection over {identity, inversion}, sharing all weights.

    F_O3(x, H) = (F(x, H) + P F(-x, P H P) P) / 2.
    Every matrix conditioning input is transformed on both AO axes. Together
    with the SO(3) backbone this gives O(3), not just inversion equivariance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "ao_parity", torch.tensor([1.] * 3 + [-1.] * 6 + [1.] * 5),
            persistent=False,
        )

    def _invert_matrix(self, matrix):
        signs = self.ao_parity.to(matrix)
        return matrix * signs[:, None] * signs[None, :]

    def forward(self, data, H, keep_blocks=False):
        if not keep_blocks:
            raise ValueError("QHFlow2-O(3) requires QH9 AO blocks (keep_blocks=True)")
        reflected = data.clone()
        reflected.pos = -data.pos
        for key in ("diagonal_init_ham", "diagonal_overlap"):
            if key in data:
                reflected[key] = self._invert_matrix(data[key])
        direct = super().forward(data, H, keep_blocks=True)
        inverse = super().forward(reflected, self._invert_matrix(H), keep_blocks=True)
        # Hidden features need not possess a single inversion parity. Return
        # only projected observables and the graph metadata used by consumers.
        return {
            key: (direct[key] + self._invert_matrix(inverse[key])) / 2
            for key in ("hamiltonian_diagonal_blocks", "hamiltonian_non_diagonal_blocks")
        } | {
            key: direct[key]
            for key in ("full_edge_index", "full_edge_distance_vec")
        }
