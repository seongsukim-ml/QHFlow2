"""Bounded, exception-safe cache for one inference call or validation batch.

Only immutable geometry/index metadata belongs here; never learned outputs.
Strong tensor references prevent Python id reuse. Version checks invalidate
in-place edits. Inference-mode tensors without version counters bypass caching.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import torch

_CACHE = ContextVar("qhflow2_inference_cache", default=None)


@contextmanager
def inference_cache_scope():
    if _CACHE.get() is not None:
        yield
        return
    token = _CACHE.set({})
    try:
        yield
    finally:
        _CACHE.reset(token)


def cached_tensor_metadata(namespace, tensors, build):
    cache = _CACHE.get()
    if cache is None:
        return build(), False
    try:
        key = (namespace, tuple((id(t), t._version) for t in tensors))
    except RuntimeError:  # torch.inference_mode tensors have no version counter.
        return build(), False
    if key not in cache:
        cache[key] = (tuple(tensors), build())
        return cache[key][1], False
    return cache[key][1], True


def reuse_inference_metadata(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if torch.is_grad_enabled():
            return function(*args, **kwargs)
        with inference_cache_scope():
            return function(*args, **kwargs)
    return wrapped
