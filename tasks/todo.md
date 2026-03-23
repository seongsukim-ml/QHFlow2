# QHFlow2 — TODO

<!-- 진행 중인 작업을 체크리스트로 관리 -->

## Current — 실험 모니터링

- [ ] 6개 실험 완료 대기 및 결과 비교
  - baseline (GPU 0, 99h79r7e) — 83K step, loss 0.00028
  - energy_mse λ=0.01 (GPU 1, 73kin396) — 20K step, loss 0.0074
  - energy_mse_strong λ=0.1 (GPU 2, s1g3ozza) — 21K step, loss 0.008
  - so2exp bw=1 (GPU 3, okungs4t) — 83K step, loss 0.006
  - bf16 bs32 (GPU 5, xocgo3z9) — 14K step, loss 0.0013
  - bf16+bs128 (GPU 6, 5t511xch) — 3.4K step, loss 0.0015

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

## Backlog

- [ ] Multi-GPU DDP 실험 (현재 single GPU만 사용)
- [ ] EMA (Exponential Moving Average) 활성화 실험
- [ ] Pre-transformed LMDB 생성 (matrix_transform + cut + Q tensor 전처리)
- [ ] QH9Dynamic dataset 실험
