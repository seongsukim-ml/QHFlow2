# QHFlow2 — TODO

<!-- 진행 중인 작업을 체크리스트로 관리 -->

## Current — 실험 모니터링

- [ ] baseline 실험 완료 대기
  - baseline — 계속 실행 중
- [x] 불필요 실험 중지 (2026-03-25)
  - ~~energy_mse_strong~~ — 중지
  - ~~cpexp~~ — 중지
  - ~~perl~~ — 중지
  - 사유: baseline만 남기고 GPU 자원 확보

## Throughput 개선 (즉시 적용 가능)

- [ ] `num_workers: 8` + `pin_memory: True` 적용 → 예상 2× speedup
  - 실측: nw=4 → 31 samp/s, nw=8 → 61 samp/s
  - config 변경만으로 가능 (QH9Stable.yaml)
- [ ] `num_workers=8` 적용 후 실제 학습 throughput 검증 실험

## Throughput 개선 (중기)

- [ ] Q tensor 사전 계산하여 LMDB에 저장
  - Q tensor가 `__getitem__`의 99% (15,909ms) — 압도적 병목
  - atom 조합에만 의존 → 전처리 가능
  - 저장하면 `__getitem__`이 ~100ms로 감소 예상
- [ ] Graph construction 캐싱
  - radius_graph + spherical_harmonics + Wigner rotation: forward의 ~15%
  - flow matching에서 좌표 불변 → 한 번만 계산하고 재사용
- [ ] torch.compile 적용
  - CC compiler 설치 필요: `conda install gcc_linux-64`
  - kernel fusion으로 GPU compute 1.2-1.5× 예상
- [ ] corrupt() GPU 이동
  - 현재 CPU에서 noise 생성 + interpolation (to_gpu+corrupt = 357ms)
  - batch-level GPU 처리로 ~50ms까지 줄일 수 있음

## Energy / Auxiliary Loss 실험

- [ ] Orbital energy loss 실험 (gradient 간섭 해결)
  - `L_orb = Σ_i |ε_i - ε_i_ref|²` — 모든 eigenvalue에 균등 gradient
  - H_loss와 cosine similarity 0.58 (energy_mse의 0.39 대비)
  - toy 실험에서 H_MAE 5× 개선 확인
- [ ] WALoss (`||C^T H C - E_diag||²`) 와 orbital energy loss 비교
  - WALoss는 QHFlow2에 이미 구현되어 있음 (config_flow_v2_simple_wa.yaml)
  - orbital energy loss와의 gradient 방향 차이 분석
- [ ] 실제 PBE 에너지 vs Tr[H·P] 에너지 비교
  - 현재 E = Tr[F·P] = band energy (one-electron), NOT total DFT energy
  - 전체 DFT 에너지: E_tot = 1/2·Tr[(H_core+F)·P] + V_nn
  - differentiable XC (functionals_torch.py) 활용 가능

## SO2 Expansion

- [ ] SO2Expansion bandwidth sweep (bw=2, 3)
  - bw=1에서 baseline 대비 loss 20× 높음
  - bandwidth 증가로 표현력 보완 가능
- [ ] SO2 vs CG expansion 정확도/속도 trade-off 분석

## Differentiable XC (eg-xc PyTorch)

- [ ] functionals_torch.py를 QHFlow2 energy loss에 통합
  - LDA/PBE/SCAN 전체 PyTorch 구현 완료, JAX와 수치 일치 검증됨
  - Grid + AO 데이터 필요 → dataset에 추가 또는 PySCF에서 on-the-fly 계산
- [ ] Learnable XC functional (Dick2021/Nagai2020) 실험 설계

## Functional/Basis Transfer (Phase 1 구현 완료)

- [x] Conditioning 모듈 구현 (FiLM + AdaLN) → `models/modules/conditioning.py`
- [x] Multi-head output 구조 → `_predict_head()`, configurable heads
- [x] Freeze 유틸리티 → `freeze_backbone()`, `freeze_all_except_conditioning_and_heads()`, etc.
- [x] Dataset에 `functional_id`/`basis_id` 필드 추가
- [x] Config 지원 (`cond_method`, `num_functionals`, `heads`, `transfer_mode`)
- [x] 사용 가이드 작성 → `docs/2026-03-25_functional_transfer_guide.md`
- [ ] EdgeDegreeEmbedding 호환 수정 후 full forward pass 검증
- [ ] QH9Stable baseline regression 테스트 (`cond_method=film` vs 기존)
- [ ] dft-dataset으로 multi-functional 데이터 생성 (PBE, PBE0 for QH9 geometries)
- [ ] Multi-task training 실험
- [ ] Fine-tuning transfer 실험 (PBE pre-train → PBE0 fine-tune)

## Functional/Basis Transfer (Phase 2 — 미착수)

- [ ] Superset output_irrep (def2-tzvp 기준 37dim)
- [ ] Input padding + output masking
- [ ] Orbital mask 확장 (`get_orbital_mask` → superset dim)
- [ ] def2-tzvp 데이터 생성 + 혼합 학습 검증

## Backlog

- [ ] Multi-GPU DDP 실험 (현재 single GPU만 사용)
- [ ] EMA (Exponential Moving Average) 활성화 실험
- [ ] Pre-transformed LMDB 생성 (matrix_transform + cut + Q tensor 전처리)
- [ ] QH9Dynamic dataset 실험
