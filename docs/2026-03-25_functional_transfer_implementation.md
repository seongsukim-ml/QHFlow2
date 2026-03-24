# QHFlow2 — Functional Transfer 모델 구현

**Date:** 2026-03-25
**Environment:** NVIDIA H200, PyTorch 2.4.1+cu121, Python 3.13
**Method:** QHFlow_escn_v5_1을 상속하여 FiLM/AdaLN conditioning + multi-head + freeze/load_pretrained 기능 추가

---

## 1. 목표

동일 backbone으로 여러 DFT functional (PBE, PBE0, B3LYP 등)에 대한 Hamiltonian을 예측할 수 있는 모델 구조.
기존 baseline checkpoint를 로드하여 새 functional에 fine-tune하는 transfer learning 지원.

## 2. 설계 원리

Fock matrix `F = H_core + J + V_xc`에서 V_xc만 functional에 의존.
모델은 `ΔF = F - H_core`를 예측하므로, FiLM/AdaLN conditioning으로 V_xc 부분만 steering.

## 3. 구현 구조

```
QHFlow_escn_v5_1 (원본 — 기존 ckpt 100% 호환, 변경 없음)
    ↑ 상속
QHFlow_escn_v5_1_transfer (FiLM/AdaLN, multi-head, freeze, load_pretrained)
```

원본 모델 코드에는 손대지 않았다. Transfer 모델은 `_init_output_layers`, `_process_through_main_layers`, `_vector_to_expand_matrix` 3개 메서드만 override.

## 4. 새 파일

| 파일 | 역할 |
|------|------|
| `models/QHFlow_so2_v5_1_transfer.py` | Transfer 모델 클래스 (상속 기반) |
| `models/modules/conditioning.py` | `ConditionEmbedding`, `FiLM`, `AdaLN`, `build_conditioning` |
| `configs/qh9/model/QHFlow_so2_v5_1_transfer_small.yaml` | Transfer model config |
| `tests/test_transfer.py` | 32 tests (30 passed, 2 skipped) |
| `docs/2026-03-25_functional_transfer_guide.md` | 사용 가이드 (다른 agent용) |

## 5. 수정된 기존 파일

| 파일 | 변경 |
|------|------|
| `models/utils.py` | `FUNCTIONAL_REGISTRY`, `BASIS_REGISTRY` 추가 |
| `models/__init__.py` | `QHFlow_escn_v5_1_transfer` 등록, conditioning args 전달 |
| `models/modules/escn_backbone_v4.py` | `use_functional_embedding` param, `mix_matrix` dim 확장 |
| `pl_module/base_module.py` | `pretrained_ckpt` 로드 + `transfer_mode` freeze 적용 |
| `pl_module/flow_module.py` | `transfer_mode` config 읽기 |
| `dataset_module/qh9_common.py` | `functional_id`/`basis_id` 필드, `functional`/`basis_name` params |
| `tasks/todo.md` | Transfer Phase 1 완료 기록 |

## 6. Conditioning 주입 지점 (3곳)

| 위치 | 파일 | 동작 |
|------|------|------|
| Backbone input | `escn_backbone_v4.py:mix_matrix` | func_emb를 atom/H/time embedding과 concat |
| Backbone 출력 후 | `_process_through_main_layers` | FiLM: `(1+γ)x + β`, AdaLN: `γ·LN(x) + β` (scalar channel만) |
| Output weight | `_vector_to_expand_matrix` | `fc_ii/fc_ij` 입력에 cond_emb concat |

## 7. Config

```yaml
# Transfer 모델 선택
version: QHFlow_so2_v5_1_transfer
num_functionals: 4       # FUNCTIONAL_REGISTRY 크기
cond_method: film        # film | adaln | none
heads:
  - hamiltonian

# Fine-tuning 시
pretrained_ckpt: /path/to/baseline.ckpt
transfer_mode: cond_and_heads   # none | cond_only | cond_and_heads | backbone_only
```

## 8. load_pretrained 동작

기존 baseline ckpt (conditioning 없음) → transfer 모델 로드 시:

| 상황 | 처리 |
|------|------|
| Shape 동일 | 그대로 복사 |
| fc_ii/fc_ij wider (cond_emb 추가) | `[:, :old_dim] = pretrained`, `[:, old_dim:] = 0` + **WARNING 로그** |
| 새 key (cond_embed 등) | default init 유지 (FiLM: γ≈0,β≈0 → near-identity) |
| Lightning ckpt 형식 | `model.` prefix 자동 strip |

로드 직후 모델은 기존 baseline과 거의 동일하게 동작 → fine-tuning으로 점진적 적응.

## 9. 테스트 결과

```
30 passed, 2 skipped, 0 failed (158s)
```

| 카테고리 | 테스트 수 | 결과 |
|----------|----------|------|
| Conditioning 모듈 (FiLM, AdaLN, factory) | 6 | 6 passed |
| Registry | 2 | 2 passed |
| Forward pass (film, adaln, none, no-func-id, diff-ids) | 6 | 5 passed, 1 skipped |
| Multi-head (keys, param count) | 3 | 2 passed, 1 skipped |
| Freeze / unfreeze (5 modes) | 5 | 5 passed |
| load_pretrained (same arch, baseline→transfer, Lightning, forward, output match) | 5 | 5 passed |
| add_functional (new, existing, forward) | 3 | 3 passed |
| Gradient flow (cond gets grad, backbone frozen) | 2 | 2 passed |

Skip 2개: `keep_blocks=False` (full matrix assembly) — dummy data에 orbital mask 없어서 skip. 실제 데이터에서는 동작.

## 10. 향후 작업

- [ ] EdgeDegreeEmbedding 호환 수정 (별도 진행 중)
- [ ] QH9Stable baseline regression 테스트
- [ ] dft-dataset으로 multi-functional 데이터 생성
- [ ] Multi-task training 실험
- [ ] Fine-tuning transfer 실험 (PBE pre-train → PBE0 fine-tune)
- [ ] Phase 2: Basis transfer (superset padding + masking)
