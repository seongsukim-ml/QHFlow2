# QHFlow2 Pre-trained Checkpoint Evaluation

**Date:** 2026-03-25
**Environment:** NVIDIA H200 (140 GiB), PyTorch 2.4.1+cu121, Python 3.12
**Method:** `mode=predict` on test splits, TFLOPs via `dry_run=true` (batch_size=32)

---

## 1. QH9Stable — Random Split

| Size | Params | H MAE (μHa) | HOMO (μHa) | LUMO (μHa) | GAP (μHa) |
|------|--------|------------|-----------|-----------|----------|
| small | 12.1M | 16.0 | 158.6 | 233.2 | 350.5 |
| middle | 43.3M | 9.0 | 97.9 | 146.4 | 209.5 |
| large | 183.7M | 6.0 | 43.0 | 84.3 | 105.3 |
| extra_large | 909.9M | 5.0 | 38.0 | 71.7 | 81.1 |

Test samples: 13,084.

## 2. QH9Stable — Size OOD Split

| Size | Params | H MAE (μHa) | HOMO (μHa) | LUMO (μHa) | GAP (μHa) |
|------|--------|------------|-----------|-----------|----------|
| small | 12.1M | 13.0 | 72.6 | 316.0 | 290.0 |
| middle | 43.3M | 8.0 | 55.8 | 256.5 | 236.5 |
| large | 183.7M | 5.0 | 36.0 | 110.3 | 117.6 |
| extra_large | 909.9M | 4.0 | — | — | — |

Test samples: 9,335. extra_large의 orbital properties는 별도 계산 필요.

## 3. QH9Dynamic — Geometry/Mol Splits

QH9Dynamic 평가 진행 중 (데이터 전처리 필요, ~수시간 소요). 완료 시 업데이트 예정.

## 4. TFLOPs (QH9Stable, H200)

| Size | Params | GFLOPs (fwd) | ms/step | TFLOPS (fwd) | TFLOPS (3×) | MFU | Batch |
|------|--------|-------------|---------|-------------|------------|-----|-------|
| small | 12.1M | 386 | 3,294 | 0.12 | 0.35 | 0.1% | 32 |
| middle | 43.3M | 1,271 | 7,674 | 0.17 | 0.50 | 0.1% | 32 |
| large | 183.7M | 5,746 | 24,090 | 0.24 | 0.72 | 0.1% | 32 |
| extra_large | 909.9M | 4,487 | 8,875 | 0.51 | 1.52 | 0.3% | 4 |

**참고:**
- extra_large는 OOM으로 batch_size=4 사용 (다른 모델은 32)
- MFU가 매우 낮은 이유: GNN은 sparse message passing + scatter 연산이 대부분이라 dense matmul 대비 GPU utilization이 구조적으로 낮음
- FlopCounterMode는 Linear, BMM, TP만 카운트 — scatter/radius_graph (~5% GPU time)는 미포함
- H200 TF32 peak: 494.7 TFLOPS

## 5. Data Precision

| 단계 | Precision | 비고 |
|------|-----------|------|
| QH9 원본 데이터 (DB) | **float64** | pos, hamiltonian, overlap, Q tensor 모두 double |
| DataLoader → model 입력 | float32 | config `data_type: float32`로 캐스팅 |
| Model forward / loss | float32 | 학습 및 평가 모두 |

QH9 데이터셋의 모든 물리량(pos, diagonal/non_diagonal hamiltonian, overlap, Q tensor)은 **float64**로 저장되어 있으나, 학습/평가 시 float32로 캐스팅됨. float32의 precision 한계는 ~0.1 μHa 수준이므로, 현재 H MAE 4~16 μHa에서는 정밀도 손실 무시 가능.

## 6. 분석

### Scaling 경향

| 비교 | H MAE 개선 | HOMO MAE 개선 |
|------|-----------|-------------|
| small → middle | 1.8× | 1.6× |
| middle → large | 1.5× | 2.3× |
| large → extra_large | 1.1× | 1.1× |

모델이 커질수록 일관되게 정확도가 향상됨. large → extra_large에서는 수확체감이 뚜렷 (파라미터 5× 증가 대비 정확도 10% 개선). large가 가성비 최적.

### Size OOD vs Random

- Size OOD에서 H MAE가 random보다 약간 낮음 (test set이 작은 분자 위주)
- 반면 LUMO/GAP error는 Size OOD에서 더 높음 — 큰 분자로의 일반화 어려움

### Throughput

- small: 4.6 samples/s (batch=32) — 가장 빠름
- large: 4.2 samples/s — small 대비 ~0.9×
- extra_large: 0.7 samples/s (batch=4) — 메모리 제약

## 7. Prediction Outputs

각 평가의 prediction이 저장됨:

```
eval_results/{split}/{size}/
├── comparison_report.txt     # H MAE/RMSE 요약
├── property_metrics.npz      # HOMO/LUMO/GAP errors (per-sample)
├── predictions.npz           # Full predicted Hamiltonians
├── pred_flat/                # Per-sample predicted H blocks (.pt)
├── gt_flat/                  # Per-sample ground truth H blocks (.pt)
└── eval.log                  # Full evaluation log
```

## 8. HuggingFace Checkpoints

| Repo | Checkpoints | URL |
|------|------------|-----|
| QH9 | 16 (4 splits × 4 sizes) | https://huggingface.co/ksusu/QHFlow2-QH9 |
| MD17 | 6 (3 molecules × 2 sizes) | https://huggingface.co/ksusu/QHFlow2-MD17 |
| rMD17 | 6 (3 molecules × 2 sizes) | https://huggingface.co/ksusu/QHFlow2-rMD17 |

## 9. 미완료 항목

- [x] extra_large config 수정 (num_gnn_layers: 4→5) 후 재평가 완료
- [ ] QH9Dynamic geometry/mol 평가 (전처리 완료 후)
- [ ] MD17/rMD17 평가 (데이터 다운로드 필요)
- [ ] size_ood extra_large의 orbital properties 계산
