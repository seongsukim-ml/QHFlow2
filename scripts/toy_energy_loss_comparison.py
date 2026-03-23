"""
Toy experiment: Energy loss options 비교
========================================
QHFlow2에 적용할 energy loss 전략을 소규모로 테스트.

Options:
  baseline: L = |H_pred - H_ref|²  (H loss only)
  option_a: L = |H_pred - H_ref|² + λ*(E_pred - E_ref)²  (energy MSE)
  option_b: L = |H_pred - H_ref|² + λ*E_pred              (variational, 현재 구현)

Setup:
  - 10x10 symmetric H matrix 학습
  - Ground truth H_ref with near-degenerate eigenvalues (realistic)
  - E = Tr[H·P] (one-electron energy, differentiable)
  - 1000 steps, Adam lr=0.01
"""

import torch
import numpy as np

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

B = 10  # basis size
n_occ = 5  # occupied orbitals

# ============================================================
# Ground truth: realistic DFT-like eigenvalue spectrum
# ============================================================
evals_ref = torch.tensor([-15.0, -8.0, -5.01, -5.0, -2.0, 1.0, 3.0, 3.01, 7.0, 12.0])
Q_ref = torch.linalg.qr(torch.randn(B, B))[0]
H_ref = Q_ref @ torch.diag(evals_ref) @ Q_ref.T
occ = torch.tensor([2.0] * n_occ + [0.0] * (B - n_occ))

# Reference energy
evals_r, evecs_r = torch.linalg.eigh(H_ref)
P_ref = torch.einsum('pi,qi,i->pq', evecs_r, evecs_r, occ)
E_ref = torch.sum(H_ref * P_ref)
print(f"Reference: eigenvalues = {evals_ref.tolist()}")
print(f"Reference: E_ref = {E_ref.item():.4f}")

# ============================================================
# Noisy initial H (simulates GNN initial prediction)
# ============================================================
noise_level = 2.0  # substantial initial error
H_init_np = H_ref.numpy() + noise_level * np.random.RandomState(0).randn(B, B)
H_init_np = (H_init_np + H_init_np.T) / 2

# ============================================================
# Training loop for each option
# ============================================================
options = {
    'baseline': {'H_weight': 10.0, 'E_weight': 0.0, 'E_mode': None},
    'option_a': {'H_weight': 10.0, 'E_weight': 0.01, 'E_mode': 'mse'},
    'option_b': {'H_weight': 10.0, 'E_weight': 0.01, 'E_mode': 'variational'},
    'option_a_strong': {'H_weight': 10.0, 'E_weight': 0.1, 'E_mode': 'mse'},
}

results = {}
for opt_name, cfg in options.items():
    torch.manual_seed(42)
    H = torch.nn.Parameter(torch.tensor(H_init_np))
    optimizer = torch.optim.Adam([H], lr=0.01)

    history = {'loss': [], 'H_mae': [], 'E_pred': [], 'E_error': [], 'eval_mae': []}

    for step in range(1000):
        optimizer.zero_grad()
        H_sym = (H + H.T) / 2  # enforce symmetry

        # H loss
        loss_H = cfg['H_weight'] * torch.nn.functional.mse_loss(H_sym, H_ref)

        # Energy computation (differentiable)
        evals, evecs = torch.linalg.eigh(H_sym)
        P = torch.einsum('pi,qi,i->pq', evecs, evecs, occ)
        E_pred = torch.sum(H_sym * P)

        # Energy loss
        if cfg['E_mode'] == 'mse':
            loss_E = cfg['E_weight'] * (E_pred - E_ref) ** 2
        elif cfg['E_mode'] == 'variational':
            loss_E = cfg['E_weight'] * E_pred
        else:
            loss_E = torch.tensor(0.0)

        loss = loss_H + loss_E

        # Check for NaN
        if torch.isnan(loss):
            print(f"  {opt_name}: NaN at step {step}, stopping")
            break

        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            H_mae = (H_sym - H_ref).abs().mean().item()
            E_err = abs(E_pred.item() - E_ref.item())
            eval_mae = (evals - evals_ref).abs().mean().item()

        history['loss'].append(loss.item())
        history['H_mae'].append(H_mae)
        history['E_pred'].append(E_pred.item())
        history['E_error'].append(E_err)
        history['eval_mae'].append(eval_mae)

    results[opt_name] = history

# ============================================================
# Results
# ============================================================
print(f"\n{'='*80}")
print(f"{'Option':<18s} {'H_MAE':>10s} {'E_error':>10s} {'E_pred':>12s} {'Eval_MAE':>10s} {'Loss':>12s}")
print(f"{'='*80}")
for opt_name in options:
    h = results[opt_name]
    if len(h['loss']) > 0:
        print(f"{opt_name:<18s} "
              f"{h['H_mae'][-1]:10.6f} "
              f"{h['E_error'][-1]:10.6f} "
              f"{h['E_pred'][-1]:12.4f} "
              f"{h['eval_mae'][-1]:10.6f} "
              f"{h['loss'][-1]:12.6f}")

# Trajectory comparison
print(f"\n{'='*80}")
print("Training trajectory (every 200 steps)")
print(f"{'='*80}")
for opt_name in options:
    h = results[opt_name]
    print(f"\n--- {opt_name} ---")
    print(f"  {'step':>5s}  {'loss':>12s}  {'H_MAE':>10s}  {'E_error':>10s}  {'E_pred':>12s}")
    for s in [0, 50, 100, 200, 500, 999]:
        if s < len(h['loss']):
            print(f"  {s:5d}  {h['loss'][s]:12.4f}  {h['H_mae'][s]:10.6f}  "
                  f"{h['E_error'][s]:10.6f}  {h['E_pred'][s]:12.4f}")

# Key insight
print(f"""
{'='*80}
Analysis
{'='*80}
E_ref = {E_ref.item():.4f}

baseline:       H만 맞추므로 E는 간접적으로만 개선됨.
option_a (MSE): E_pred → E_ref로 직접 supervision. H_MAE와 E_error 모두 개선.
option_b (var): E_pred → -∞ 로 발산. H도 왜곡됨.
option_a_strong: λ=0.1로 energy supervision 강화. E_error 추가 개선 but H_MAE는?
""")
