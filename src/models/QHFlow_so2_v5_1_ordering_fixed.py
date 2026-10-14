# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""Experimental ordering-corrected QHFlow2; legacy models remain separate.

This variant corrects the verified eSCN-to-e3nn layout bug. It is not a
drop-in replacement for pretrained legacy checkpoints: identical weight
shapes do not imply identical predictions. See docs/EQUIVARIANCE_STATUS.md.
"""

from .QHFlow_so2_v5_1 import QHFlow_escn_v5_1
from .spherical_layout import escn_to_e3nn


class QHFlow_escn_v5_1_ordering_fixed(QHFlow_escn_v5_1):
    """Use channel-major irreps for both node and pair features."""

    def _flatten_escn_embedding(self, embedding):
        return escn_to_e3nn(embedding, self.order)
