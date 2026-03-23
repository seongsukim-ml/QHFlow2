"""
QHFlow v5.1 with CPExpansion (TDN-style vectorized CG).

Drop-in replacement: uses CPExpansion instead of Expansion for both
diagonal (ii) and off-diagonal (ij) blocks. Same weights, same output,
2.5-4x faster expansion layer.
"""

from e3nn import o3

from .QHFlow_so2_v5_1 import QHFlow_escn_v5_1
from .layers import CPExpansion


class QHFlow_escn_v5_1_cpexp(QHFlow_escn_v5_1):
    """QHFlow with CPExpansion for faster Hamiltonian block prediction.

    Additional Args:
        cp_rank (int or None): CP decomposition rank. None = exact (vectorized only).
    """

    def __init__(self, cp_rank=None, **kwargs):
        self.cp_rank = cp_rank
        super().__init__(**kwargs)

    def _create_matrix_prediction_layers(self, matrix_type):
        """Override: use CPExpansion instead of Expansion."""
        input_expand_irrep = o3.Irreps(
            f"{self.bottle_hidden_size}x0e + "
            f"{self.bottle_hidden_size}x1e + "
            f"{self.bottle_hidden_size}x2e + "
            f"{self.bottle_hidden_size}x3e + "
            f"{self.bottle_hidden_size}x4e"
        )
        output_irrep = self.output_irrep

        # Diagonal (ii): CPExpansion
        self.expand_ii[matrix_type] = CPExpansion(
            input_expand_irrep, output_irrep, output_irrep,
            cp_rank=self.cp_rank,
        )
        self.fc_ii[matrix_type] = self._make_fc(
            self.hidden_size, self.expand_ii[matrix_type].num_path_weight
        )
        self.fc_ii_bias[matrix_type] = self._make_fc(
            self.hidden_size, self.expand_ii[matrix_type].num_bias
        )

        # Off-diagonal (ij): CPExpansion
        self.expand_ij[matrix_type] = CPExpansion(
            input_expand_irrep, output_irrep, output_irrep,
            cp_rank=self.cp_rank,
        )
        self.fc_ij[matrix_type] = self._make_fc(
            self.hidden_size * 2, self.expand_ij[matrix_type].num_path_weight
        )
        self.fc_ij_bias[matrix_type] = self._make_fc(
            self.hidden_size * 2, self.expand_ij[matrix_type].num_bias
        )

    @staticmethod
    def _make_fc(in_dim, out_dim):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim),
        )
