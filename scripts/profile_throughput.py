"""QHFlow2 throughput profiling script — run via Hydra."""
import torch
import time
import hydra
from omegaconf import DictConfig

from qhflow2.common.qh9_utils import load_qh9_dataset, create_qh9_data_loaders
from qhflow2.pl_module.flow_module import LitModel_flow
from qhflow2.common.setup import get_root_path, setup_tensor_type_and_seed


def time_cuda(fn, N=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / N


@hydra.main(config_path="../config_qh9", config_name="config_flow_v2_simple", version_base=None)
def main(conf: DictConfig):
    torch.set_float32_matmul_precision('high')
    setup_tensor_type_and_seed(conf)

    root = get_root_path()
    dataset = load_qh9_dataset(conf, root)
    loaders = create_qh9_data_loaders(dataset, conf, batch_size=[32, 32, 32])
    train_loader = loaders[0]

    batches = []
    for i, b in enumerate(train_loader):
        batches.append(b.cuda())
        if i >= 4:
            break
    print(f'\nLoaded {len(batches)} batches (bs=32), atoms/batch: {batches[0].atoms.shape[0]}')

    lit_model = LitModel_flow(conf)
    lit_model = lit_model.cuda()
    lit_model.train()
    model = lit_model.model
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {conf.model.version}, params: {nparams/1e6:.1f}M')

    # Prepare batches with corruption (creates init_ham_t)
    prepared = []
    for b in batches:
        b = lit_model.post_processing(b, lit_model.default_type)
        b = lit_model.corrupt(b, mul=1)
        prepared.append(b)
    batches = prepared

    N = 5
    b = batches[0]

    # Warmup
    for bb in batches[:2]:
        out = model(bb, bb.init_ham_t, keep_blocks=True)
        loss = out['hamiltonian_diagonal_blocks'].pow(2).mean()
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # ============================================================
    # 1. Basic throughput
    # ============================================================
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        for bb in batches:
            with torch.no_grad():
                out = model(bb, bb.init_ham_t, keep_blocks=True)
    torch.cuda.synchronize()
    fwd = (time.perf_counter() - t0) / N / len(batches)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        for bb in batches:
            out = model(bb, bb.init_ham_t, keep_blocks=True)
            loss = (out['hamiltonian_diagonal_blocks'].pow(2).mean()
                    + out['hamiltonian_non_diagonal_blocks'].pow(2).mean())
            loss.backward()
            model.zero_grad()
    torch.cuda.synchronize()
    fwd_bwd = (time.perf_counter() - t0) / N / len(batches)
    bwd = fwd_bwd - fwd

    print(f'\n{"="*60}')
    print(f'1. BASELINE THROUGHPUT (bs=32, FP32)')
    print(f'{"="*60}')
    print(f'Forward:       {fwd*1000:7.1f} ms  ({fwd/fwd_bwd*100:.0f}%)')
    print(f'Backward:      {bwd*1000:7.1f} ms  ({bwd/fwd_bwd*100:.0f}%)')
    print(f'Total:         {fwd_bwd*1000:7.1f} ms/step')
    print(f'Throughput:    {32/fwd_bwd:7.0f} samples/sec')

    # ============================================================
    # 2. Mixed precision (BF16)
    # ============================================================
    print(f'\n{"="*60}')
    print(f'2. MIXED PRECISION (BF16)')
    print(f'{"="*60}')

    # Forward BF16
    try:
        t_bf16_fwd = time_cuda(lambda: (
            model(b, b.init_ham_t, keep_blocks=True)
            if False else
            (lambda: None)()  # placeholder
        ))
    except:
        pass

    def fwd_bf16():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(b, b.init_ham_t, keep_blocks=True)

    def fwd_bwd_bf16():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(b, b.init_ham_t, keep_blocks=True)
            loss = (out['hamiltonian_diagonal_blocks'].pow(2).mean()
                    + out['hamiltonian_non_diagonal_blocks'].pow(2).mean())
        loss.backward()
        model.zero_grad()

    try:
        t_bf16_f = time_cuda(fwd_bf16)
        t_bf16_fb = time_cuda(fwd_bwd_bf16)
        print(f'BF16 Forward:  {t_bf16_f:7.1f} ms  (vs FP32 {fwd*1000:.1f} ms, {fwd*1000/t_bf16_f:.2f}x)')
        print(f'BF16 Fwd+Bwd:  {t_bf16_fb:7.1f} ms  (vs FP32 {fwd_bwd*1000:.1f} ms, {fwd_bwd*1000/t_bf16_fb:.2f}x)')
    except Exception as e:
        print(f'BF16 failed: {e}')

    # ============================================================
    # 3. torch.compile
    # ============================================================
    print(f'\n{"="*60}')
    print(f'3. TORCH.COMPILE')
    print(f'{"="*60}')
    try:
        compiled = torch.compile(model, mode='reduce-overhead')
        # Warmup / compilation
        print('Compiling... (this takes a while)')
        for _ in range(5):
            out = compiled(b, b.init_ham_t, keep_blocks=True)
            loss = out['hamiltonian_diagonal_blocks'].pow(2).mean()
            loss.backward()
            model.zero_grad()

        t_comp_f = time_cuda(lambda: compiled(b, b.init_ham_t, keep_blocks=True))

        def comp_fwd_bwd():
            out = compiled(b, b.init_ham_t, keep_blocks=True)
            loss = (out['hamiltonian_diagonal_blocks'].pow(2).mean()
                    + out['hamiltonian_non_diagonal_blocks'].pow(2).mean())
            loss.backward()
            model.zero_grad()

        t_comp_fb = time_cuda(comp_fwd_bwd)
        print(f'Compiled Fwd:  {t_comp_f:7.1f} ms  (vs FP32 {fwd*1000:.1f} ms, {fwd*1000/t_comp_f:.2f}x)')
        print(f'Compiled F+B:  {t_comp_fb:7.1f} ms  (vs FP32 {fwd_bwd*1000:.1f} ms, {fwd_bwd*1000/t_comp_fb:.2f}x)')
    except Exception as e:
        print(f'torch.compile failed: {type(e).__name__}: {str(e)[:200]}')

    # ============================================================
    # 4. Memory & batch size scaling
    # ============================================================
    print(f'\n{"="*60}')
    print(f'4. MEMORY & BATCH SIZE SCALING')
    print(f'{"="*60}')

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = model(b, b.init_ham_t, keep_blocks=True)
    mem_fwd = torch.cuda.memory_allocated() / 1e9
    loss = (out['hamiltonian_diagonal_blocks'].pow(2).mean()
            + out['hamiltonian_non_diagonal_blocks'].pow(2).mean())
    loss.backward()
    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    model.zero_grad()
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print(f'Model params:  {param_mem:.0f} MB')
    print(f'Peak memory:   {mem_peak:.2f} GB (bs=32)')
    print(f'H200 capacity: 80 GB')
    print(f'Headroom:      {80 - mem_peak:.1f} GB')

    # Try larger batch sizes using train_loader
    for bs in [64, 128, 256]:
        try:
            torch.cuda.empty_cache()
            from torch_geometric.loader import DataLoader
            sub_loader = DataLoader(train_loader.dataset, batch_size=bs, shuffle=False, num_workers=0)
            bb = next(iter(sub_loader)).cuda()
            bb = lit_model.post_processing(bb, lit_model.default_type)
            bb = lit_model.corrupt(bb, mul=1)
            torch.cuda.reset_peak_memory_stats()
            t = time_cuda(lambda: model(bb, bb.init_ham_t, keep_blocks=True), N=3)
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f'bs={bs:3d}: fwd {t:.1f} ms, {bs/(t/1000):.0f} samp/s, {mem:.1f} GB')
        except Exception as e:
            print(f'bs={bs:3d}: {type(e).__name__}: {str(e)[:80]}')
            torch.cuda.empty_cache()

    # ============================================================
    # 5. Profiler (top ops)
    # ============================================================
    print(f'\n{"="*60}')
    print(f'5. TOP 20 CUDA OPS')
    print(f'{"="*60}')
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(3):
            out = model(b, b.init_ham_t, keep_blocks=True)
            loss = out['hamiltonian_diagonal_blocks'].pow(2).mean()
            loss.backward()
            model.zero_grad()
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))


if __name__ == '__main__':
    main()
