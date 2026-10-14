# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""Layout conversion at the eSCN/e3nn feature boundary."""

import torch


def escn_to_e3nn(embedding: torch.Tensor, lmax: int) -> torch.Tensor:
    """Pack ``[batch, (lmax+1)**2, channels]`` as e3nn irreps.

    eSCN stores magnetic components before channels. e3nn stores each
    channel's complete ``2*l+1`` components together within each l block.
    Transposing the whole tensor would also mix the angular-degree blocks,
    so each block must be transposed separately.
    """
    if lmax < 0 or embedding.ndim != 3 or embedding.shape[1] != (lmax + 1) ** 2:
        raise ValueError(
            f"Expected [batch, {(lmax + 1) ** 2}, channels] for lmax={lmax}; "
            f"got {tuple(embedding.shape)}"
        )
    return torch.cat([
        embedding[:, l * l:(l + 1) ** 2, :].transpose(1, 2).flatten(1)
        for l in range(lmax + 1)
    ], dim=-1)
