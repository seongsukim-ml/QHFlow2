# expand_lmax 버그 수정 및 ablation 재실행

**Date:** 2026-03-25
**Environment:** 8×H200, PyTorch 2.10+cu128, Lightning 2.6.1
**Method:** 코드 리뷰 + 파라미터 수 검증 + ablation 재실행

## 발견된 버그

### 증상

이전 expandL ablation (v1, v2)에서 expand_lmax=1,2,3,4 모든 실험이 **동일한 결과로 수렴**.

### 원인

`expand_lmax`가 모델에 전달되었지만 **실제로 사용되지 않음**.

```python
# models/__init__.py — expand_lmax를 model_args에 추가
if hasattr(args, "expand_lmax"):
    model_args["expand_lmax"] = args.expand_lmax

# QHFlow_so2_v5_1.__init__ — **kwargs로 흡수, 무시
def __init__(self, ..., **kwargs):
    ...
    # expand_lmax 언급 없음

# _create_matrix_prediction_layers — input_expand_irrep 하드코딩
input_expand_irrep = o3.Irreps(
    f"{self.bottle_hidden_size}x0e + "
    f"{self.bottle_hidden_size}x1e + "
    f"{self.bottle_hidden_size}x2e + "
    f"{self.bottle_hidden_size}x3e + "
    f"{self.bottle_hidden_size}x4e"  # ← 항상 lmax=4
)
```

`expand_lmax`와 무관하게 항상 `l=0~4` irrep으로 Expansion layer가 생성됨 → 모든 모델이 동일.

## 수정 내용

### 1. `__init__`에서 `expand_lmax` 명시적으로 받기

```python
def __init__(self, ..., expand_lmax=None, **kwargs):
    self.expand_lmax = expand_lmax if expand_lmax is not None else sh_lmax
```

### 2. `expand_bottle_irrep` 생성

```python
self.expand_bottle_irrep = o3.Irreps(
    construct_o3irrps(self.bottle_hidden_size, order=self.expand_lmax)
)
```

### 3. `_create_matrix_prediction_layers`에서 동적 irrep

```python
input_expand_irrep = o3.Irreps(
    " + ".join(f"{self.bottle_hidden_size}x{l}e" for l in range(self.expand_lmax + 1))
)
```

### 4. `output_ii`, `output_ij`를 `expand_bottle_irrep`으로

```python
self.output_ii = Linear(self.hidden_irrep, self.expand_bottle_irrep)
self.output_ij = Linear(self.hidden_irrep, self.expand_bottle_irrep)
```

## 검증

### 파라미터 수 비교

| expand_lmax | expand+fc params | total params | expand_bottle_irrep |
|------------|-----------------|-------------|-------------------|
| 1 | 393,656 | 42,599,609 | `32x0e + 32x1o` |
| 2 | 518,456 | 42,732,601 | `+ 32x2e` |
| 3 | 560,056 | 42,782,393 | `+ 32x3o` |
| 4 (baseline) | 568,376 | 42,798,905 | `+ 32x4e` |

수정 전: 모든 lmax에서 568,376으로 동일 → **버그 확인**
수정 후: lmax에 따라 파라미터 수가 달라짐 → **정상**

## 재실행 (v3)

| Model | GPU | expand_lmax | wandb tag |
|-------|-----|------------|-----------|
| QHFlow_so2_v5_1_middle | 2 | 4 (baseline) | v3, fixed |
| QHFlow_so2_v5_1_middle_expandL1 | 3 | 1 | v3, fixed |
| QHFlow_so2_v5_1_middle_expandL2 | 4 | 2 | v3, fixed |
| QHFlow_so2_v5_1_middle_expandL3 | 5 | 3 | v3, fixed |

설정:
- `prefix: -expandL-ablation-v3`
- `wandb.tags: [qh9, flow, expand_lmax_ablation, v3, fixed]`
- `num_workers: 8` (preprocessed LMDB, 4.3x faster dataloader)
- `dataset: QH9Stable-random, bs=32, 260k steps`

## 추가 개선: DataLoader 속도

이번 재실행부터 preprocessed LMDB + `num_workers=8` 적용.

| | 이전 (v1, v2) | 현재 (v3) | Speedup |
|---|---|---|---|
| DataLoader | shard, num_workers=0 | preprocessed LMDB, num_workers=8 | |
| Step time | 27.94s | 6.49s | **4.3x** |
| Time/epoch | ~25h | ~6h | |
| GPU utilization | 낮음 | 38-72% | |
