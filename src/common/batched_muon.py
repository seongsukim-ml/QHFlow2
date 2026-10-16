"""Shape-batched equivalent of native torch.optim.Muon (PyTorch 2.11)."""
import math
from collections import defaultdict
import torch


@torch.no_grad()
def batched_muon_step(group, state):
    buckets = defaultdict(list)
    for parameter in group['params']:
        if parameter.grad is None:
            continue
        if parameter.grad.is_sparse or parameter.is_complex() or parameter.ndim != 2:
            raise ValueError('Muon requires dense real matrix gradients')
        if 'momentum_buffer' not in state[parameter]:
            state[parameter]['momentum_buffer'] = torch.zeros_like(parameter.grad)
        buckets[(parameter.shape, parameter.device, parameter.dtype)].append(parameter)
    for (shape, _, _), parameters in buckets.items():
        gradients = [p.grad for p in parameters]
        buffers = [state[p]['momentum_buffer'] for p in parameters]
        torch._foreach_lerp_(buffers, gradients, 1 - group['momentum'])
        updates = torch._foreach_lerp(gradients, buffers, group['momentum']) if group['nesterov'] else buffers
        x = torch.stack(updates).bfloat16()
        transpose = shape[0] > shape[1]
        if transpose:
            x = x.transpose(-1, -2)
        x.div_(x.norm(dim=(-2, -1), keepdim=True).clamp(min=group['eps']))
        a, b, c = group['ns_coefficients']
        for _ in range(group['ns_steps']):
            gram = x @ x.transpose(-1, -2)
            correction = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            x = torch.baddbmm(x, correction, x, beta=a)
        if transpose:
            x = x.transpose(-1, -2)
        lr = group['lr']
        adjustment = group['adjust_lr_fn']
        ratio = math.sqrt(max(1, shape[0] / shape[1])) if adjustment in (None, 'original') else 0.2 * math.sqrt(max(shape))
        torch._foreach_mul_(parameters, 1 - lr * group['weight_decay'])
        torch._foreach_add_(parameters, list(x.unbind()), alpha=-lr * ratio)
