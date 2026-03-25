# nablaDFT Training Bugfixes

**Date:** 2026-03-25
**Environment:** H200 8x, qhflow conda, QHFlow2 + nablaDFT pipeline
**Method:** Bug identification and fix during nablaDFT training integration

## Bugs Fixed

### 1. KeyError: 16/17 (S/Cl elements) in orbital_mask

**Symptom:** `KeyError: 16` / `KeyError: 17` during validation sample metric evaluation.

**Root cause:** `base_module.get_orbital_mask()` hardcodes elements 1-10 (QH9-only). nablaDFT has S(16), Cl(17), Br(35).

```python
# Before (base_module.py:674)
def get_orbital_mask(self):
    for i in range(1, 11):  # ← only elements 1-10
        orbital_mask[i] = ...
```

**Fix:** Use `_get_orbital_mask(basis)` when a `basis` config field is set, with legacy fallback.

```python
# After
def get_orbital_mask(self):
    basis = getattr(self, '_basis', None) or getattr(self.conf, 'basis', None)
    if basis is not None:
        return _get_orbital_mask(basis=basis)
    # Legacy fallback for QH9...
```

Added `basis: "def2-svp-nabla"` to `config_flow_v2_nabla.yaml`.

**Files:** `pl_module/base_module.py`, `configs/qh9/config_flow_v2_nabla.yaml`

### 2. ckpt_path="best" crash after short training

**Symptom:** `ValueError: .test(ckpt_path="best") is set but ModelCheckpoint is not configured to save the best model.`

**Root cause:** `train_qh9.py:155` always calls `trainer.test(ckpt_path="best")`. If `check_val_every_n_epoch=999` or training is too short, no "best" checkpoint is saved.

**Fix:** Check if any ModelCheckpoint callback has a `best_model_path` set. Fall back to `"last"`.

```python
_ckpt_callbacks = [c for c in trainer.callbacks
                   if hasattr(c, 'best_model_path') and c.best_model_path]
_test_ckpt = "best" if _ckpt_callbacks else "last"
trainer.test(lit_model, test_loader, ckpt_path=_test_ckpt)
```

**File:** `experiment/train_qh9.py`

### 3. LMDB deadlock with num_workers > 0

**Symptom:** Training hangs indefinitely with `num_workers > 0` for nablaDFT dataset.

**Root cause:** `NablaDFTDataset.__init__()` opens LMDB environment. When PyTorch DataLoader forks workers, child processes inherit the parent's LMDB handle, which is not fork-safe. QH9 doesn't have this issue because it uses `InMemoryDataset` (loads everything in `__init__`, no LMDB access in `__getitem__`).

**Fix:** Lazy LMDB open — store path in `__init__`, open env on first `__getitem__` call per process.

```python
# Before
def __init__(self, lmdb_path, ...):
    self._env = lmdb.open(lmdb_path, ...)  # fork-unsafe!

# After
def __init__(self, lmdb_path, ...):
    self._lmdb_path = lmdb_path
    self._env = None  # opened lazily

def _get_env(self):
    if self._env is None:
        self._env = lmdb.open(self._lmdb_path, subdir=False, readonly=True, lock=False)
    return self._env
```

**File:** `dataset_module/nabladft_dataset.py`

## Verification

| Test | Before | After |
|------|--------|-------|
| 50-step train + test | KeyError: 16/17 | No errors |
| num_workers=8 | Hangs | 1.88 it/s (3x vs workers=0) |
| ckpt_path="best" | ValueError crash | Falls back to "last" |
| QH9 baseline | Works | Still works (backward compatible) |
