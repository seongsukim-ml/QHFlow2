# QHFlow2 — Functional/Basis Transfer Guide

**Date:** 2026-03-25
**Status:** Phase 1 구현 완료 (functional conditioning). 검증은 EdgeDegreeEmbedding 호환 수정 후 가능.

---

## 1. 개요

QHFlow2에 DFT functional 조건부 예측 기능을 추가했다.
같은 backbone으로 여러 functional (PBE, PBE0, B3LYP 등)에 대한 Hamiltonian을 예측할 수 있다.

### 핵심 아이디어

Fock matrix `F = H_core + J + V_xc`에서 V_xc만 functional에 의존한다.
모델은 `ΔF = F - H_core` (residual)를 예측하므로, functional conditioning은 V_xc 부분만 steering한다.

### 구현 방식

- **FiLM** (Feature-wise Linear Modulation): `x' = (1 + γ(c)) * x + β(c)`
- **AdaLN** (Adaptive Layer Normalization): `x' = γ(c) * LayerNorm(x) + β(c)`
- 둘 다 scalar (l=0) channel에만 적용하여 equivariance 보존

---

## 2. 수정된 파일 목록

| 파일 | 역할 |
|------|------|
| `src/qhflow2/models/modules/conditioning.py` | **신규**. `ConditionEmbedding`, `FiLM`, `AdaLN`, `build_conditioning` |
| `src/qhflow2/models/utils.py` | `FUNCTIONAL_REGISTRY`, `BASIS_REGISTRY` 추가 |
| `src/qhflow2/models/QHFlow_so2_v5_1.py` | FiLM/adaLN 통합, multi-head, freeze 유틸리티 |
| `src/qhflow2/models/modules/escn_backbone_v4.py` | `use_functional_embedding`, `mix_matrix` 확장 |
| `src/qhflow2/models/__init__.py` | `get_model`에 conditioning args 전달 |
| `src/qhflow2/pl_module/base_module.py` | `transfer_mode` 적용 |
| `src/qhflow2/pl_module/flow_module.py` | `transfer_mode` config 읽기 |
| `src/qhflow2/dataset_module/qh9_common.py` | `functional_id`, `basis_id` 필드, `functional`/`basis_name` params |
| `configs/qh9/model/QHFlow_so2_v5_1_transfer_small.yaml` | **신규**. transfer model config |

---

## 3. Conditioning 주입 지점 (3곳)

### 3.1 Backbone input mix
- **파일**: `escn_backbone_v4.py` forward, `matrix_mix_list`
- **동작**: functional embedding이 atom embedding, H, time 등과 함께 `mix_matrix` Linear에 concat
- **조건**: `use_functional_embedding=True`일 때만 활성

### 3.2 Backbone 출력 후 (FiLM / AdaLN)
- **파일**: `QHFlow_so2_v5_1.py`, `_process_through_main_layers`
- **동작**: backbone이 출력한 `node_attr_R`의 scalar channel에 FiLM/adaLN 적용
- **초기화**: near-identity (γ≈0 for FiLM, γ≈1 for AdaLN, β≈0)

### 3.3 Output weight generation (fc_ii / fc_ij)
- **파일**: `QHFlow_so2_v5_1.py`, `_vector_to_expand_matrix`
- **동작**: `fc_ii` 입력에 `[node_attr_R_init, cond_emb]` concat, `fc_ij` 입력에 `[src, dst, cond_emb]` concat
- **input dim 변경**: `fc_ii`: `hidden_size → 2*hidden_size`, `fc_ij`: `2*hidden_size → 3*hidden_size`

---

## 4. Config 설명

### 4.1 Model config (`configs/qh9/model/QHFlow_so2_v5_1_transfer_small.yaml`)

```yaml
model_name: QHFlow_so2_v5_1
version: QHFlow_so2_v5_1

hidden_size: 64
bottle_hidden_size: 32
use_block_S: True
use_block_H: True
num_gnn_layers: 3
num_ham_gnn_layers: 2

# ── Functional/Basis Transfer ──
num_functionals: 4       # 등록된 functional 수 (FUNCTIONAL_REGISTRY 참조)
cond_method: film        # film | adaln | none
heads:                   # output heads
  - hamiltonian          # 기본 head (필수)
  # - overlap            # 추가 head (선택)
```

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `num_functionals` | 4 | `FUNCTIONAL_REGISTRY` 크기와 일치해야 함 |
| `cond_method` | `film` | `film`: FiLM, `adaln`: AdaLN, `none`: conditioning 비활성 (기존과 동일) |
| `heads` | `[hamiltonian]` | output head 이름 목록. 각 head별로 `expand_ii/ij`, `fc_ii/ij` 별도 생성 |

### 4.2 Transfer mode (training config)

```yaml
transfer_mode: none      # none | cond_only | cond_and_heads | backbone_only
```

| Mode | Frozen | Trainable | 용도 |
|------|--------|-----------|------|
| `none` | 없음 | 전체 | 일반 학습 / multi-task |
| `cond_only` | backbone + ham layers + output heads 전부 | FiLM/adaLN 파라미터만 | 최소 적응 (데이터 < 100 샘플) |
| `cond_and_heads` | backbone + ham layers | conditioning + fc + expand | **표준 fine-tuning (추천)** |
| `backbone_only` | backbone만 | ham layers + conditioning + output 전부 | 넉넉한 데이터일 때 |

### 4.3 Dataset config

`BaseQH9Dataset.__init__`에 추가된 파라미터:

```yaml
functional: b3lyp        # pbe | pbe0 | b3lyp | r2scan
basis_name: def2-svp     # def2-svp | def2-tzvp
```

기존 config에 이 필드가 없으면 기본값 (`b3lyp`, `def2-svp`)이 사용된다.

---

## 5. Registry

`src/qhflow2/models/utils.py`:

```python
FUNCTIONAL_REGISTRY = {"pbe": 0, "pbe0": 1, "b3lyp": 2, "r2scan": 3}
BASIS_REGISTRY = {"def2-svp": 0, "def2-tzvp": 1}
```

- Dataset의 `functional` 파라미터가 여기서 integer ID로 변환됨
- 모델의 `num_functionals`는 registry 크기와 일치해야 함
- 새 functional 추가 시 registry에 항목 추가 + `num_functionals` 증가

---

## 6. 사용 시나리오

### 6.1 Multi-task training (여러 functional 동시 학습)

**목표**: PBE + PBE0 + B3LYP 데이터를 섞어서 하나의 모델로 학습

**Step 1: 데이터 생성**
```bash
# dft-dataset 프로젝트에서 QH9 geometry에 대해 다중 functional로 DFT 계산
# 각 LMDB 샘플에 functional_id 포함
# → QH9Stable_pbe.lmdb, QH9Stable_pbe0.lmdb, QH9Stable_b3lyp.lmdb
# 또는 하나의 mixed LMDB (functional_id가 샘플별로 다름)
```

**Step 2: Dataset config 작성**
```yaml
# configs/qh9/dataset/QH9Stable_multi.yaml
dataset_name: QH9Stable_multi
functional: b3lyp        # 기본값; LMDB에 functional_id가 있으면 이것은 무시됨
basis_name: def2-svp
# ... (나머지는 QH9Stable.yaml과 동일)
```

> **주의**: 현재 `BaseQH9Dataset`은 단일 functional을 가정한다. Multi-functional LMDB를 쓰려면 `_build_data_object`에서 LMDB에 저장된 `functional_id`를 읽도록 수정 필요. 현재 구현은 dataset-level `_functional_id` 하나를 모든 샘플에 적용한다.

**Step 3: 학습 실행**
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m qhflow2.experiment.train_qh9 \
  model=QHFlow_so2_v5_1_transfer_small \
  dataset=QH9Stable_multi \
  wandb.mode=online
```

### 6.2 Fine-tuning transfer (기존 conditioning 모델 → 다른 functional)

**목표**: PBE로 pre-train한 **transfer 모델**을 소량의 PBE0 데이터로 fine-tune

```bash
# Step 1: PBE 데이터로 transfer 모델 pre-train
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m qhflow2.experiment.train_qh9 \
  model=QHFlow_so2_v5_1_transfer_small \
  dataset=QH9Stable_pbe \
  wandb.mode=online
# → outputs/.../checkpoints/best.ckpt

# Step 2: PBE0 데이터로 fine-tune (backbone freeze)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m qhflow2.experiment.train_qh9 \
  model=QHFlow_so2_v5_1_transfer_small \
  dataset=QH9Stable_pbe0 \
  pretrained_ckpt=outputs/.../checkpoints/best.ckpt \
  transfer_mode=cond_and_heads \
  dataset.learning_rate=1e-4 \
  dataset.total_steps=50000 \
  wandb.mode=online
```

이 시나리오는 pre-train과 fine-tune 모두 **같은 모델 구조** (transfer_small)를 사용한다.
`pretrained_ckpt`의 모든 weight가 shape 일치하므로 그대로 로드된다.

### 6.3 기존 baseline 모델에서 transfer 모델로 변환

**목표**: conditioning 없이 학습된 **기존 baseline checkpoint**를 transfer 모델에 로드

이것이 핵심 시나리오다. 기존 모델과 transfer 모델은 구조가 다르다:
- `fc_ii[0]` Linear: 기존 `(hidden_size, hidden_size)` → 새 모델 `(hidden_size + cond_dim, hidden_size)`
- `fc_ij[0]` Linear: 기존 `(2*hidden_size, hidden_size)` → 새 모델 `(3*hidden_size, hidden_size)`
- `cond_embed`, `cond_post_backbone`: 기존 모델에 없음
- `mix_matrix` Linear: 기존보다 wider (func_emb 추가분)

`load_pretrained()`가 이를 자동으로 처리한다:

```bash
# 기존 baseline checkpoint를 transfer 모델에 로드하여 fine-tune
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m qhflow2.experiment.train_qh9 \
  model=QHFlow_so2_v5_1_transfer_small \
  dataset=QH9Stable_pbe0 \
  pretrained_ckpt=/path/to/baseline_best.ckpt \
  transfer_mode=cond_and_heads \
  dataset.learning_rate=1e-4 \
  dataset.total_steps=50000 \
  wandb.mode=online
```

**`load_pretrained`의 동작**:

```
1. Shape 일치하는 key → 그대로 복사
   (backbone, contraction, e3_gnn layers, expand, norm, ...)

2. fc_ii/fc_ij/fc_ii_bias/fc_ij_bias 첫 번째 Linear:
   - pretrained weight: shape (out, hidden_size)
   - new weight:        shape (out, hidden_size + cond_dim)
   - → new[:, :hidden_size] = pretrained, new[:, hidden_size:] = 0
   - 즉, cond_emb 입력 부분은 zero-init → 처음에는 conditioning이 무시됨

3. mix_matrix Linear:
   - 같은 방식으로 partial copy + zero-pad

4. cond_embed, cond_post_backbone:
   - pretrained에 없음 → default init 유지
   - FiLM: γ≈0, β≈0 (near-identity)
   - AdaLN: γ≈1, β≈0 (near-identity)

결과: 로드 직후 모델은 기존 baseline과 거의 동일하게 동작
      → fine-tuning으로 conditioning이 점진적으로 학습됨
```

### 6.4 새 functional 추가 (Python API)

기존 checkpoint에 없던 functional을 추가하고 fine-tune하는 경우:

```python
import torch
from qhflow2.models.QHFlow_so2_v5_1 import QHFlow_escn_v5_1

# 1. Transfer 모델 생성 + pretrained 로드
model = QHFlow_escn_v5_1(**model_config)
model.load_pretrained("path/to/pretrained.ckpt")

# 2. 새 functional 추가
new_id = model.add_functional("wb97x-d")
# → embedding table이 num_functionals+1로 확장
# → 새 embedding은 기존 embedding들의 평균으로 초기화
# → FUNCTIONAL_REGISTRY에 {"wb97x-d": 4} 추가됨

# 3. Fine-tuning 설정
model.freeze_all_except_conditioning_and_heads()
# → backbone frozen, conditioning + heads만 학습
```

### 6.4 Backward compatibility (기존 학습과 동일하게 사용)

기존 config로 conditioning 없이 학습:

```yaml
# 기존 config 그대로 사용 → cond_method, heads 미지정
model_name: QHFlow_so2_v5_1
hidden_size: 64
# ... (num_functionals, cond_method, heads 없음)
```

이 경우:
- `cond_method` 기본값 = `"film"` → conditioning 모듈이 생성되지만 near-identity 초기화
- `data`에 `functional_id`가 없으면 `_get_cond_embedding()`이 `None` 반환 → conditioning 비활성
- **기존 학습에 영향 없음**

명시적으로 비활성화하려면:
```yaml
cond_method: none         # conditioning 모듈 자체를 생성하지 않음
```

---

## 7. Multi-head 출력 구조

### 단일 head (기본)

```yaml
heads:
  - hamiltonian
```

출력 dict:
- `keep_blocks=True`: `{"hamiltonian_diagonal_blocks": ..., "hamiltonian_non_diagonal_blocks": ...}`
- `keep_blocks=False`: `{"hamiltonian": ...}`

기존 flow_module과 100% 호환.

### Multi-head

```yaml
heads:
  - hamiltonian
  - overlap
```

출력 dict:
- `keep_blocks=True`: `{"hamiltonian_diagonal_blocks": ..., "hamiltonian_non_diagonal_blocks": ..., "overlap_diagonal_blocks": ..., "overlap_non_diagonal_blocks": ...}`
- `keep_blocks=False`: `{"hamiltonian": ..., "overlap": ...}`

**주의**: 추가 head의 loss를 flow_module에서 처리하려면 별도의 loss 코드를 추가해야 한다. 현재 flow_module은 `"hamiltonian"` key만 처리한다.

### Multi-head 활용 예시

```python
# 학습 시: hamiltonian head의 loss만 backprop (기존과 동일)
# overlap head는 auxiliary loss로 추가 가능:
#   loss_overlap = MAE(output["overlap_diagonal_blocks"], batch.diagonal_overlap)
#   total_loss = loss_ham + λ * loss_overlap
```

---

## 8. Freeze 유틸리티 API

`QHFlow_escn_v5_1` 모델에 추가된 메서드:

| 메서드 | 동작 |
|--------|------|
| `model.load_pretrained(ckpt_path)` | pretrained ckpt 로드 (shape mismatch 자동 처리, zero-pad) |
| `model.freeze_backbone()` | backbone + contraction + ham layers + output_ii/ij freeze |
| `model.freeze_all_except_conditioning()` | 전체 freeze → conditioning만 unfreeze |
| `model.freeze_all_except_conditioning_and_heads()` | backbone freeze → conditioning + fc + expand unfreeze |
| `model.unfreeze_all()` | 전체 unfreeze |
| `model.add_functional("name")` | embedding table 확장, 새 ID 반환 |

### `pretrained_ckpt` config (Hydra에서 사용)

```yaml
pretrained_ckpt: /path/to/baseline.ckpt   # pretrained weight 로드 (optional)
transfer_mode: cond_and_heads              # freeze 전략
```

**실행 순서**: `get_model()` → `model.set(device)` → `load_pretrained()` → `freeze()`

**`ckpt` vs `pretrained_ckpt` 차이**:
- `ckpt`: Lightning resume — optimizer, scheduler, epoch 등 전체 상태 복원
- `pretrained_ckpt`: model weight만 로드 — shape mismatch 허용, 새 학습 시작

### Freeze 범위 상세

```
freeze_all_except_conditioning_and_heads() 시:

  FROZEN:
  ├── node_attr_backbone        # eSCN backbone 전체 (sphere_embedding, SO2 blocks, ...)
  ├── contraction_layer_H       # input matrix → hidden features
  ├── contraction_layer_S       # (if use_block_S)
  ├── contraction_layer_H_init  # (if use_block_H)
  ├── e3_gnn_node_layer[]       # SelfNetLayer (equivariant ham layers)
  ├── e3_gnn_node_pair_layer[]  # PairNetLayer
  ├── output_ii                 # Linear to bottleneck irrep
  ├── pre_output_ij             # TensorProduct
  └── output_ij                 # Linear to bottleneck irrep

  TRAINABLE:
  ├── cond_embed                # nn.Embedding + MLP (functional → dense vector)
  ├── cond_post_backbone        # FiLM or AdaLN (scalar channel modulation)
  ├── fc_ii["hamiltonian"]      # MLP: [node_attr + cond_emb] → expansion weights
  ├── fc_ij["hamiltonian"]      # MLP: [src + dst + cond_emb] → expansion weights
  ├── fc_ii_bias["hamiltonian"] # MLP: → expansion biases
  ├── fc_ij_bias["hamiltonian"] # MLP: → expansion biases
  ├── expand_ii["hamiltonian"]  # CG expansion (diagonal blocks)
  └── expand_ij["hamiltonian"]  # CG expansion (off-diagonal blocks)
```

---

## 9. 검증 체크리스트

구현 후 반드시 확인할 항목:

- [ ] **Import 테스트**: `from qhflow2.models.QHFlow_so2_v5_1 import QHFlow_escn_v5_1` 성공
  - 현재 `EdgeDegreeEmbedding` 호환 문제로 블록됨 (별도 수정 진행 중)
- [ ] **Forward pass**: `cond_method=film`, `adaln`, `none` 각각 forward 정상 동작
- [ ] **Shape 확인**: output dict의 key와 shape이 기존과 동일 (단일 head 기준)
- [ ] **Backward compat**: `functional_id` 없는 기존 데이터로 기존 config 실행 시 에러 없음
- [ ] **Freeze 확인**: `freeze_all_except_conditioning_and_heads()` 후 `requires_grad` 수 확인
- [ ] **Multi-head**: 2개 head일 때 output key가 올바른지
- [ ] **Conditioning 효과**: 같은 geometry에 다른 `functional_id`를 넣으면 다른 출력
- [ ] **Full training**: QH9Stable에서 `cond_method=film` 으로 baseline 대비 regression 없는지

---

## 10. 향후 작업 (Phase 2: Basis Transfer)

Phase 1 (functional conditioning) 검증 완료 후:

1. **Superset output_irrep**: `"5x0e + 5x1e + 3x2e + 1x3e"` (dim=37, def2-tzvp 기준)
2. **Input padding**: 작은 basis의 H block을 superset dim으로 zero-pad
3. **Orbital mask 확장**: `get_orbital_mask()` → superset dim 기반 mask
4. **Loss masking**: 유효 orbital 영역에만 loss 적용
5. **데이터**: dft-dataset으로 def2-tzvp 데이터 생성

계획 상세: `/home/irteam/.claude/plans/stateless-wiggling-dewdrop.md` 참조
