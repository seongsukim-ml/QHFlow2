"""Conditioning modules for functional/basis transfer.

Provides FiLM and AdaLN conditioning that modulate equivariant features
based on a discrete condition (e.g., DFT functional ID).
"""

import torch
from torch import nn


class ConditionEmbedding(nn.Module):
    """Shared embedding + MLP that maps a discrete condition to a dense vector.

    Parameters
    ----------
    num_classes : int
        Number of discrete conditions (e.g., number of functionals).
    embed_dim : int
        Output embedding dimension.
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [N] long tensor → [N, embed_dim]."""
        return self.mlp(self.embed(ids))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation.

    Given a condition embedding c, modulate features x as:
        x' = (1 + γ(c)) * x + β(c)

    Only modulates the scalar (l=0) channel to preserve equivariance.

    Parameters
    ----------
    hidden_size : int
        Dimension of the scalar channel to modulate.
    cond_dim : int
        Dimension of the condition embedding input.
    """

    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(cond_dim, 2 * hidden_size)
        # Initialize near-identity: γ≈0, β≈0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [N, D]  where D = (lmax+1) * hidden_size (full equivariant feature).
        cond : [N, cond_dim]  condition embedding per node.

        Returns
        -------
        Modulated x with same shape. Only scalar channel is affected.
        """
        gamma_beta = self.proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # Modulate only scalar (l=0) channel
        x_scalar = x[:, :self.hidden_size]
        x_scalar = (1 + gamma) * x_scalar + beta
        return torch.cat([x_scalar, x[:, self.hidden_size:]], dim=-1)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization (DiT-style).

    Replaces standard LayerNorm with condition-dependent scale/shift:
        x' = γ(c) * LayerNorm(x) + β(c)

    Only applied to the scalar (l=0) channel.

    Parameters
    ----------
    hidden_size : int
        Dimension of the scalar channel.
    cond_dim : int
        Dimension of the condition embedding input.
    """

    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_size)
        # Initialize: γ≈1, β≈0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:hidden_size] = 1.0  # γ starts at 1

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [N, D]  full equivariant feature.
        cond : [N, cond_dim]  condition embedding per node.

        Returns
        -------
        Modulated x with same shape.
        """
        gamma_beta = self.proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        x_scalar = self.norm(x[:, :self.hidden_size])
        x_scalar = gamma * x_scalar + beta
        return torch.cat([x_scalar, x[:, self.hidden_size:]], dim=-1)


def build_conditioning(
    method: str,
    hidden_size: int,
    cond_dim: int,
) -> nn.Module:
    """Factory for conditioning modules.

    Parameters
    ----------
    method : str
        One of "film", "adaln", "none".
    """
    if method == "film":
        return FiLM(hidden_size, cond_dim)
    elif method == "adaln":
        return AdaLN(hidden_size, cond_dim)
    elif method == "none":
        return None
    else:
        raise ValueError(f"Unknown conditioning method: {method}")
