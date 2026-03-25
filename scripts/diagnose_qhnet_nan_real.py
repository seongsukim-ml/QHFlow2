"""
QHNet NaN 진단 (실제 QH9 데이터 사용)
실제 학습 파이프라인과 동일한 조건에서 activation 추적.
"""

import torch
import sys, os
sys.path.insert(0, "src")

from omegaconf import OmegaConf
from qhflow2.models.Real_QHNet_qh9 import QHNet
from qhflow2.common.qh9_utils import load_qh9_dataset, create_qh9_data_loaders
from qhflow2.pl_module.base_module import LitModel


def stat_str(t, name=""):
    has_nan = t.isnan().any().item()
    finite = t[t.isfinite()]
    if len(finite) == 0:
        return f"{name:<40} ALL NaN/Inf"
    s = f"{name:<40} std={finite.std().item():>12.4f}  max={finite.abs().max().item():>12.4f}"
    if has_nan:
        s += f"  NaN={t.isnan().float().mean().item()*100:.1f}%"
    return s


def main():
    device = torch.device("cuda")

    # Load config
    conf = OmegaConf.load("configs/qh9/qhnet_scaling.yaml")
    dataset_conf = OmegaConf.load("configs/qh9/dataset/QH9Stable.yaml")
    conf.dataset = dataset_conf
    conf.qh9 = True

    # Load real dataset
    from pathlib import Path
    root_path = Path(".")
    dataset = load_qh9_dataset(conf, root_path)
    train_loader, _, _ = create_qh9_data_loaders(dataset, conf)

    for hs, hbs, gnn, label in [(128, 32, 5, "middle"), (256, 64, 7, "large")]:
        print(f"\n{'='*70}")
        print(f"QHNet {label}: hidden={hs}, bottle={hbs}, gnn={gnn}")
        print(f"{'='*70}")

        model = QHNet(hidden_size=hs, bottle_hidden_size=hbs, num_gnn_layers=gnn)
        model = model.to(device)
        model.set(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

        model.train()
        nan_steps = 0
        for step, batch in enumerate(train_loader):
            if step >= 20:
                break

            batch = batch.to(device)
            # Cast all float64 tensors to float32 (PL does this via precision=32)
            for key in list(batch.keys):
                val = batch[key]
                if isinstance(val, torch.Tensor) and val.dtype == torch.float64:
                    batch[key] = val.float()
            optimizer.zero_grad()

            try:
                output = model(batch, keep_blocks=True)
                diag = output["hamiltonian_diagonal_blocks"]
                off = output["hamiltonian_non_diagonal_blocks"]

                # Real loss (MAE of prediction vs actual H blocks)
                # For diagnosis, just use prediction magnitude as proxy
                loss = diag.abs().mean() + off.abs().mean()

                has_nan = loss.isnan().item() or diag.isnan().any().item() or off.isnan().any().item()

                if has_nan:
                    nan_steps += 1
                    print(f"  Step {step}: NaN!")
                    print(f"    {stat_str(diag, 'diag_blocks')}")
                    print(f"    {stat_str(off, 'off_diag_blocks')}")

                    # Check which params have NaN
                    nan_params = []
                    for name, p in model.named_parameters():
                        if p.isnan().any():
                            nan_params.append(name)
                    if nan_params:
                        print(f"    NaN in parameters: {nan_params[:5]}")
                    continue

                loss.backward()

                # Check gradient norms per module group
                grad_stats = {}
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    group = name.split(".")[0]
                    if group not in grad_stats:
                        grad_stats[group] = {"max_norm": 0, "has_nan": False}
                    gn = p.grad.norm().item()
                    grad_stats[group]["max_norm"] = max(grad_stats[group]["max_norm"], gn)
                    if p.grad.isnan().any():
                        grad_stats[group]["has_nan"] = True

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step <= 5 or step % 5 == 0:
                    parts = []
                    for g, s in sorted(grad_stats.items()):
                        if s["max_norm"] > 10 or s["has_nan"]:
                            val = "NaN!" if s["has_nan"] else f"{s['max_norm']:.1f}"
                            parts.append(f"{g}={val}")
                    grad_report = ", ".join(parts)
                    print(f"  Step {step}: loss={loss.item():.6f}  diag_std={diag.std().item():.4f}"
                          f"  off_std={off.std().item():.4f}"
                          + (f"  LARGE_GRAD: {grad_report}" if grad_report else ""))

            except Exception as e:
                print(f"  Step {step}: EXCEPTION: {str(e)[:100]}")
                # Check parameter health
                nan_params = [n for n, p in model.named_parameters() if p.isnan().any()]
                if nan_params:
                    print(f"    NaN params: {nan_params[:5]}")
                break

        print(f"\n  Summary: {nan_steps}/20 steps had NaN")
        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
