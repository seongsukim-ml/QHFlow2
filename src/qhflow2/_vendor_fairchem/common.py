"""Lightweight shims for fairchem.core.common modules.

Replaces: gp_utils, distutils, registry, utils.
QHFlow2 does not use model parallelism, so gp_utils functions are no-ops.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch


# ── gp_utils shim ──────────────────────────────────────────────
# QHFlow2 uses DDP only, not model parallelism.
# All gp_utils functions used in backbone are no-ops.

def initialized() -> bool:
    return False

def get_gp_rank() -> int:
    return 0

def get_gp_world_size() -> int:
    return 1

def reduce_from_model_parallel_region(x: torch.Tensor) -> torch.Tensor:
    return x

def gather_from_model_parallel_region(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return x

def gather_from_model_parallel_region_sum_grad(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return x


# ── distutils shim ─────────────────────────────────────────────

def get_device_for_local_rank() -> str:
    return "cuda"


# ── registry shim ──────────────────────────────────────────────
# Only used as @registry.register_model("name") decorator — a no-op.

class _Registry:
    def register_model(self, name: str):
        def decorator(cls):
            return cls
        return decorator

registry = _Registry()


# ── conditional_grad shim ──────────────────────────────────────
# Used as @conditional_grad(torch.enable_grad()) in backbone.

def conditional_grad(dec):
    """Decorator that wraps a function with a context manager (e.g. torch.enable_grad()).

    Handles both context manager instances and callables.
    Usage: @conditional_grad(torch.enable_grad())
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # torch.enable_grad() is both callable and has __enter__
            # Always create a fresh context manager for each call
            if hasattr(dec, '__enter__'):
                with dec.__class__():
                    return func(*args, **kwargs)
            else:
                with dec():
                    return func(*args, **kwargs)
        return wrapper
    return decorator
