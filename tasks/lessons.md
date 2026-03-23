# QHFlow2 — Lessons

<!-- 실수·교정에서 배운 패턴을 기록. 같은 실수를 반복하지 않기 위함 -->

## 2026-03-23: Throughput bottleneck은 GPU가 아니라 data pipeline

**상황**: BF16으로 forward를 3.4× 빠르게 했지만 실제 학습은 오히려 느려짐
**원인**: GPU compute가 전체의 42%에 불과. data_load (34%) + to_gpu+corrupt (23%) = 58%가 CPU overhead
**교훈**: 최적화 전에 반드시 component-level profiling 필요. Amdahl's law — bottleneck이 아닌 부분을 빠르게 해도 전체 speedup은 미미

## 2026-03-23: num_workers=8이 sweet spot

**상황**: num_workers=4 (default) → 8로 바꾸면 2× speedup, 12+는 오히려 느려짐
**원인**: nw=12+에서 CPU 메모리 경쟁 + persistent_workers의 overhead
**교훈**: dataloader worker 수는 실측 필요. GPU가 bottleneck이 될 때까지 올리고, 그 이상은 역효과

## 2026-03-23: Q tensor가 __getitem__의 99%

**상황**: `__getitem__` 최적화를 위해 cut_matrix, matrix_transform을 개선 시도
**원인**: Q tensor (torch.block_diag × 60) = 15,909ms/sample. 나머지 합쳐서 ~108ms
**교훈**: profiling 없이 "비싸보이는" 코드를 최적화하면 시간 낭비. Q tensor를 사전 계산하여 LMDB에 저장하는 게 가장 효과적

## 2026-03-23: Energy loss gradient 간섭

**상황**: energy MSE loss를 추가했더니 H loss가 baseline 대비 20× 높아짐
**원인**: E = Tr[H·P]의 gradient가 occupied orbital에만 집중 (virtual = 0). H loss와 cosine sim 0.39로 방향 불일치
**교훈**: auxiliary loss 추가 시 gradient alignment 분석 필수. orbital energy loss (all eigenvalues)가 더 적합 — cosine sim 0.58

## 2026-03-23: Variational energy loss는 발산

**상황**: `loss += λ * E_pred` (에너지를 낮추는 방향) → loss -1.4M으로 발산
**원인**: H를 변경하면서 에너지를 최소화하는 건 variational principle이 아님. GNN이 eigenvalue를 -∞로 보내서 비물리적 H 생성
**교훈**: energy loss는 반드시 `(E_pred - E_ref)²` (MSE) 형태여야 함. `E_pred` 자체를 minimize하면 안 됨

## 2026-03-23: SafeEigh (Lorentzian broadening) 불필요

**상황**: degenerate eigenvalue에서 gradient NaN 방지를 위해 SafeEigh 구현
**원인**: density matrix P = C·C^T는 degenerate subspace 내 rotation에 invariant → G_ij singularity가 chain rule에서 자연 상쇄
**교훈**: DFT의 gauge invariance로 인해 observable 기반 loss에서는 eigensolver 안정화 불필요. SafeEigh의 bias가 오히려 수렴 방해
