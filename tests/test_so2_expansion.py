"""Test SO2Expansion: equivariance + throughput vs CG Expansion."""

import sys
import os
from math import prod
import builtins
builtins.prod = prod  # layers.py uses prod without import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from e3nn import o3
import time

# Patch relative import in layers.py
import types
modules_mod = types.ModuleType('models.modules')
modules_mod.__dict__['InnerProduct'] = None  # not needed for Expansion
sys.modules['models.modules'] = modules_mod
sys.modules['models'] = types.ModuleType('models')

# Now import layers directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    'models.layers',
    os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'layers.py'),
    submodule_search_locations=[],
)
layers = importlib.util.module_from_spec(spec)
layers.__package__ = 'models'
sys.modules['models.layers'] = layers
spec.loader.exec_module(layers)

Expansion = layers.Expansion
SO2Expansion = layers.SO2Expansion

torch.manual_seed(42)
device = 'cuda'

irrep_in = o3.Irreps('32x0e + 32x1e + 32x2e + 32x3e + 32x4e')
irrep_out = o3.Irreps('3x0e + 2x1e + 1x2e')

cg_exp = Expansion(irrep_in, irrep_out, irrep_out).to(device)
so2_exp = SO2Expansion(irrep_in, irrep_out, irrep_out, bandwidth=0).to(device)

print(f'CG:  weights={cg_exp.num_path_weight}, bias={cg_exp.num_bias}')
print(f'SO2: weights={so2_exp.num_path_weight}, bias={so2_exp.num_bias}')
print(so2_exp)

assert cg_exp.num_path_weight == so2_exp.num_path_weight
assert cg_exp.num_bias == so2_exp.num_bias
print('Weight counts match!\n')

B = 16
x = torch.randn(B, irrep_in.dim, device=device)
w = torch.randn(B, cg_exp.num_path_weight, device=device) * 0.01
bw = torch.randn(B, cg_exp.num_bias, device=device) * 0.01
edge_vec = torch.randn(B, 3, device=device)

out_cg = cg_exp(x, weights=w, bias_weights=bw)
out_so2 = so2_exp(x, weights=w, bias_weights=bw, edge_vec=edge_vec)

print(f'CG  output: {out_cg.shape}, norm={out_cg.norm():.4f}')
print(f'SO2 output: {out_so2.shape}, norm={out_so2.norm():.4f}')
print(f'Norm ratio: {out_so2.norm()/out_cg.norm():.4f}')

# Equivariance
R = o3.rand_matrix()
D_in = irrep_in.D_from_matrix(R).to(device)
D_out = irrep_out.D_from_matrix(R).to(device)
x_rot = (D_in @ x.unsqueeze(-1)).squeeze(-1)
edge_rot = (R.to(device) @ edge_vec.unsqueeze(-1)).squeeze(-1)

out1 = so2_exp(x_rot, weights=w, bias_weights=bw, edge_vec=edge_rot)
out2 = so2_exp(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
out2_rot = D_out.to(device) @ out2 @ D_out.to(device).T

diff = (out1 - out2_rot).abs().max().item()
status = 'PASS' if diff < 1e-3 else 'FAIL'
print(f'\nbw=0 equivariance: {diff:.2e}  ({status})')

# Throughput
N = 200
for _ in range(10):
    cg_exp(x, weights=w, bias_weights=bw)
    so2_exp(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N):
    cg_exp(x, weights=w, bias_weights=bw)
torch.cuda.synchronize()
t_cg = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N):
    so2_exp(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
torch.cuda.synchronize()
t_so2 = time.perf_counter() - t0

print(f'\nThroughput (B={B}, {N} iters):')
print(f'  CG:  {t_cg*1000:.1f}ms ({N/t_cg:.0f} it/s)')
print(f'  SO2: {t_so2*1000:.1f}ms ({N/t_so2:.0f} it/s)')
print(f'  Speedup: {t_cg/t_so2:.2f}x')

# bandwidth=1
print(f'\n{"="*50}')
print('bandwidth=1')
print('='*50)
so2_bw1 = SO2Expansion(irrep_in, irrep_out, irrep_out, bandwidth=1).to(device)
print(so2_bw1)

out_bw1 = so2_bw1(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
print(f'bw=1 output: {out_bw1.shape}, norm={out_bw1.norm():.4f}')

x_rot1 = (D_in @ x.unsqueeze(-1)).squeeze(-1)
edge_rot1 = (R.to(device) @ edge_vec.unsqueeze(-1)).squeeze(-1)
out1 = so2_bw1(x_rot1, weights=w, bias_weights=bw, edge_vec=edge_rot1)
out2 = so2_bw1(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
out2_rot = D_out.to(device) @ out2 @ D_out.to(device).T
diff = (out1 - out2_rot).abs().max().item()
status = 'PASS' if diff < 1e-3 else 'FAIL'
print(f'bw=1 equivariance: {diff:.2e}  ({status})')

for _ in range(10):
    so2_bw1(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    so2_bw1(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
torch.cuda.synchronize()
t_bw1 = time.perf_counter() - t0
print(f'bw=1: {t_bw1*1000:.1f}ms ({N/t_bw1:.0f} it/s), vs CG: {t_cg/t_bw1:.2f}x')

# bandwidth=2
print(f'\n{"="*50}')
print('bandwidth=2')
print('='*50)
so2_bw2 = SO2Expansion(irrep_in, irrep_out, irrep_out, bandwidth=2).to(device)
print(so2_bw2)

out_bw2 = so2_bw2(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
x_rot2 = (D_in @ x.unsqueeze(-1)).squeeze(-1)
edge_rot2 = (R.to(device) @ edge_vec.unsqueeze(-1)).squeeze(-1)
out1 = so2_bw2(x_rot2, weights=w, bias_weights=bw, edge_vec=edge_rot2)
out2 = so2_bw2(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
out2_rot = D_out.to(device) @ out2 @ D_out.to(device).T
diff = (out1 - out2_rot).abs().max().item()
status = 'PASS' if diff < 1e-3 else 'FAIL'
print(f'bw=2 equivariance: {diff:.2e}  ({status})')

for _ in range(10):
    so2_bw2(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    so2_bw2(x, weights=w, bias_weights=bw, edge_vec=edge_vec)
torch.cuda.synchronize()
t_bw2 = time.perf_counter() - t0
print(f'bw=2: {t_bw2*1000:.1f}ms ({N/t_bw2:.0f} it/s), vs CG: {t_cg/t_bw2:.2f}x')

print('\nDone!')
