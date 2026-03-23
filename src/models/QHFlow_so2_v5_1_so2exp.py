"""
QHFlow v5.1 with SO(2) Expansion for off-diagonal blocks.

Uses SO2Expansion (bond-aligned local frame) for off-diagonal (ij) blocks
while keeping standard CG Expansion for diagonal (ii) blocks.

SO(2) approximation exploits the fact that in a bond-aligned frame,
the Hamiltonian block is approximately banded in delta_m = m1 - m2.
This reduces computation while maintaining equivariance.
"""

import torch
from torch import nn
from e3nn import o3

from .QHFlow_so2_v5_1 import QHFlow_escn_v5_1
from .layers import Expansion, SO2Expansion


class QHFlow_escn_v5_1_so2exp(QHFlow_escn_v5_1):
    """QHFlow with SO(2) Expansion for off-diagonal Hamiltonian blocks.

    Inherits from QHFlow_escn_v5_1 and replaces only the off-diagonal
    expansion layer with SO2Expansion.

    Additional Args:
        so2_bandwidth (int): Max |m1 - m2| in local frame.
            0 = diagonal only, 1 = tridiagonal, 2 = pentadiagonal.
    """

    def __init__(self, so2_bandwidth=1, **kwargs):
        self.so2_bandwidth = so2_bandwidth
        super().__init__(**kwargs)

    def _create_matrix_prediction_layers(self, matrix_type):
        """Override: use SO2Expansion for off-diagonal blocks."""
        input_expand_irrep = o3.Irreps(
            f"{self.bottle_hidden_size}x0e + "
            f"{self.bottle_hidden_size}x1e + "
            f"{self.bottle_hidden_size}x2e + "
            f"{self.bottle_hidden_size}x3e + "
            f"{self.bottle_hidden_size}x4e"
        )
        output_irrep = self.output_irrep

        # Diagonal (ii): standard CG expansion (no edge direction)
        self.expand_ii[matrix_type] = Expansion(
            input_expand_irrep, output_irrep, output_irrep
        )
        self.fc_ii[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ii[matrix_type].num_path_weight),
        )
        self.fc_ii_bias[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ii[matrix_type].num_bias),
        )

        # Off-diagonal (ij): SO(2) expansion (uses bond direction)
        self.expand_ij[matrix_type] = SO2Expansion(
            input_expand_irrep, output_irrep, output_irrep,
            bandwidth=self.so2_bandwidth,
        )
        self.fc_ij[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_path_weight),
        )
        self.fc_ij_bias[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_bias),
        )

    def _vector_to_expand_matrix(self, data, fii, fij, keep_blocks):
        """Override: pass edge_vec to SO2Expansion for off-diagonal blocks."""
        node_attr_R_init = data["node_attr_R_init"]
        full_dst, full_src = data["full_edge_index"]
        transpose_edge_index = data["transpose_edge_index"]
        edge_vec = data["full_edge_distance_vec"]

        # Diagonal: standard expansion (unchanged)
        hamiltonian_diagonal_matrix = self.expand_ii["hamiltonian"](
            fii,
            self.fc_ii["hamiltonian"](node_attr_R_init),
            self.fc_ii_bias["hamiltonian"](node_attr_R_init),
        )

        # Off-diagonal: SO(2) expansion with edge direction
        node_pair_embedding = torch.cat(
            [node_attr_R_init[full_dst], node_attr_R_init[full_src]],
            dim=-1,
        )
        hamiltonian_non_diagonal_matrix = self.expand_ij["hamiltonian"](
            fij,
            self.fc_ij["hamiltonian"](node_pair_embedding),
            self.fc_ij_bias["hamiltonian"](node_pair_embedding),
            edge_vec=edge_vec,
        )

        if not keep_blocks:
            hamiltonian_matrix = self._build_final_matrix(
                data, hamiltonian_diagonal_matrix, hamiltonian_non_diagonal_matrix,
            )
            hamiltonian_matrix = (hamiltonian_matrix + hamiltonian_matrix.transpose(-1, -2)) / 2
            return {"hamiltonian": hamiltonian_matrix}
        else:
            ret_diag = (hamiltonian_diagonal_matrix + hamiltonian_diagonal_matrix.transpose(-1, -2)) / 2
            ret_offdiag = (
                hamiltonian_non_diagonal_matrix
                + hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(-1, -2)
            ) / 2
            return {
                "hamiltonian_diagonal_blocks": ret_diag,
                "hamiltonian_non_diagonal_blocks": ret_offdiag,
            }
