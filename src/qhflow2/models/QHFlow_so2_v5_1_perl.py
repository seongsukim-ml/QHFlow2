"""
QHFlow v5.1 with PerLExpansion (per-l_in shared weights).

Middle ground between Expansion (per-path weights) and TDN (single shared W).
Uses one weight matrix per input angular momentum l_in: 5 weight matrices
for lmax=4.

NOT a drop-in replacement — different num_path_weight/num_bias.
"""

from e3nn import o3

from .QHFlow_so2_v5_1 import QHFlow_escn_v5_1
from .layers import PerLExpansion


class QHFlow_escn_v5_1_perl(QHFlow_escn_v5_1):
    """QHFlow with PerLExpansion for Hamiltonian block prediction.

    Per-l_in weight sharing: each input angular momentum l_in gets its own
    weight matrix, giving 5 matrices for lmax=4. This is between the full
    per-path Expansion (19 matrices) and TDN (1 matrix).

    Additional Args:
        cp_rank (int or None): CP decomposition rank for CG tensor.
    """

    def __init__(self, cp_rank=None, **kwargs):
        self.cp_rank = cp_rank
        super().__init__(**kwargs)

    def _create_matrix_prediction_layers(self, matrix_type):
        """Override: use PerLExpansion instead of Expansion."""
        input_expand_irrep = o3.Irreps(
            f"{self.bottle_hidden_size}x0e + "
            f"{self.bottle_hidden_size}x1e + "
            f"{self.bottle_hidden_size}x2e + "
            f"{self.bottle_hidden_size}x3e + "
            f"{self.bottle_hidden_size}x4e"
        )
        output_irrep = self.output_irrep

        # Diagonal (ii): PerLExpansion
        self.expand_ii[matrix_type] = PerLExpansion(
            input_expand_irrep, output_irrep, output_irrep,
            cp_rank=self.cp_rank,
        )
        self.fc_ii[matrix_type] = self._make_fc(
            self.hidden_size, self.expand_ii[matrix_type].num_path_weight
        )
        self.fc_ii_bias[matrix_type] = self._make_fc(
            self.hidden_size, self.expand_ii[matrix_type].num_bias
        )

        # Off-diagonal (ij): PerLExpansion
        self.expand_ij[matrix_type] = PerLExpansion(
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
