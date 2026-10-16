# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
"""QH9Stable SO(3) model and its two-element O(3) projection.

These are new architectures, not inference-time replacements for legacy
checkpoints. In particular, spectral gating replaces sampled-grid nonlinearities.
"""

import copy
import torch

from common.inference_cache import cached_tensor_metadata

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
        def build():
            return super(QH9SphericalBackbone, self)._get_rotmat_and_wigner(
                edge_distance_vecs[:, [1, 2, 0]], use_cuda_graph)
        if torch.is_grad_enabled() or use_cuda_graph:
            return build()
        result, hit = cached_tensor_metadata(
            (self, "wigner"), (edge_distance_vecs,), build)
        if hit:
            # Preserve the random stream used by subsequent flow noise draws.
            # The original frame builder draws one random roll per edge.
            torch.rand(edge_distance_vecs.shape[0], device=edge_distance_vecs.device,
                       dtype=edge_distance_vecs.dtype)
        return result

    def _generate_graph(self, data_dict):
        build = lambda: super(QH9SphericalBackbone, self)._generate_graph(data_dict)
        if torch.is_grad_enabled():
            return build()
        result, _ = cached_tensor_metadata(
            (self, "backbone_graph", self.cutoff, self.max_neighbors),
            (data_dict.pos, data_dict.batch, *((data_dict.cell,) if "cell" in data_dict else ())), build)
        return result.copy()



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
        # Isolate attribute assignments without cloning every label, Q tensor,
        # overlap and orbital coefficient in the input batch.
        data = copy.copy(data)
        if not self.node_attr_backbone.direct_forces:
            data.pos = data.pos.clone()
        data.charge = torch.zeros(data.ptr.numel() - 1, dtype=torch.long, device=H.device)
        data.spin = torch.zeros_like(data.charge)
        return super().forward(data, H, keep_blocks=True)

    def _prepare_geometry(self, data):
        if torch.is_grad_enabled():
            return super()._prepare_geometry(data)
        keys = ("atomic_numbers", "edge_index", "edge_distance_vec", "edge_distance",
                "full_edge_index", "full_edge_distance_vec", "full_edge_distance",
                "full_edge_sh", "transpose_edge_index")
        def build():
            super(QHFlow2_order_fixed, self)._prepare_geometry(data)
            return {key: data[key] for key in keys}
        result, hit = cached_tensor_metadata(
            (self, "model_geometry", self.max_radius, self.max_num_neighbors),
            (data.pos, data.batch, data.ptr, data.atoms), build)
        for key, value in result.items():
            data[key] = value
        if hit:
            # The radial basis has a learned parameter: recompute even on hits.
            data["full_edge_attr"] = self.distance_expansion(
                data["full_edge_distance"].unsqueeze(-1)).squeeze().type(data.pos.type())

    @staticmethod
    def _calculate_transpose_indices(data, radius_edges):
        """Reverse complete-graph edges without per-molecule CUDA scalars."""
        counts = data.ptr[1:] - data.ptr[:-1]
        edge_counts = counts * (counts - 1)
        offsets = edge_counts.cumsum(0) - edge_counts
        graph = data.batch[radius_edges[0]]
        local = radius_edges - data.ptr[graph].unsqueeze(0)
        return (offsets[graph] + local[0] * (counts[graph] - 1)
                + local[1] - (local[0] < local[1]).long())


class QHFlow2_O3(QHFlow2_order_fixed):
    """Reynolds projection over {identity, inversion}, sharing all weights.

    F_O3(x, H) = (F(x, H) + P F(-x, P H P) P) / 2.
    Every matrix conditioning input is transformed on both AO axes. Together
    with the SO(3) backbone this gives O(3), not just inversion equivariance.
    """

    def __init__(self, *args, batch_branches=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_branches = batch_branches
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
        if self.batch_branches:
            return self._forward_paired(data, H)
        reflected = copy.copy(data)
        if torch.is_grad_enabled():
            reflected.pos = -data.pos
        else:
            reflected.pos, _ = cached_tensor_metadata(
                (self, "inverted_positions"), (data.pos,), lambda: -data.pos)
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

    def _forward_paired(self, data, H):
        """Evaluate the two projected branches as independent graphs in one batch."""
        paired = copy.copy(data)
        n = H.shape[0]
        graphs = data.ptr.numel() - 1
        def positions():
            return torch.cat((data.pos, -data.pos), dim=0)
        def topology():
            return (torch.cat((data.atoms, data.atoms), dim=0),
                    torch.cat((data.batch, data.batch + graphs), dim=0),
                    torch.cat((data.ptr, data.ptr[1:] + n), dim=0))
        if torch.is_grad_enabled():
            paired.pos = positions()
            paired.atoms, paired.batch, paired.ptr = topology()
        else:
            paired.pos, _ = cached_tensor_metadata((self, "paired_pos"), (data.pos,), positions)
            topo, _ = cached_tensor_metadata((self, "paired_topology"),
                (data.atoms, data.batch, data.ptr), topology)
            paired.atoms, paired.batch, paired.ptr = topo
        paired.num_nodes = 2 * n
        # PyG Batch keeps graph count independently of ptr/num_nodes.
        if hasattr(data, "_num_graphs"):
            paired._num_graphs = 2 * graphs
        if "cell" in data:
            paired.cell = torch.cat((data.cell, data.cell), dim=0)
        if "natoms" in data:
            paired.natoms = torch.cat((data.natoms, data.natoms), dim=0)
        paired.t = torch.cat((data.t, data.t), dim=0)
        for key in ("diagonal_init_ham", "diagonal_overlap"):
            if key in data:
                paired[key] = torch.cat((data[key], self._invert_matrix(data[key])), dim=0)
        output = super().forward(paired, torch.cat((H, self._invert_matrix(H)), dim=0), keep_blocks=True)
        projected = {}
        for key in ("hamiltonian_diagonal_blocks", "hamiltonian_non_diagonal_blocks"):
            direct, inverse = output[key].chunk(2, dim=0)
            projected[key] = (direct + self._invert_matrix(inverse)) / 2
        edges = output["full_edge_index"].shape[1] // 2
        projected["full_edge_index"] = output["full_edge_index"][:, :edges]
        projected["full_edge_distance_vec"] = output["full_edge_distance_vec"][:edges]
        return projected
