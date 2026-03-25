"""Minimal HeadInterface from fairchem.core.models.base."""

from __future__ import annotations


class HeadInterface:
    """Abstract interface for model output heads.

    Subclasses should implement forward() to produce predictions
    (energy, forces, stress, etc.) from backbone embeddings.
    This is a pure interface mixin — does NOT inherit nn.Module.
    """
    pass
