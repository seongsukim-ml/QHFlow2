"""Profile real training bottleneck: data loading vs compute."""
import torch, time, numpy as np
import hydra
from omegaconf import DictConfig
from qhflow2.common.qh9_utils import load_qh9_dataset, create_qh9_data_loaders
from qhflow2.common.setup import get_root_path, setup_tensor_type_and_seed
from qhflow2.pl_module.flow_module import LitModel_flow


@hydra.main(config_path="../config_qh9", config_name="config_flow_v2_simple", version_base=None)
def main(conf: DictConfig):
    torch.set_float32_matmul_precision('high')
    setup_tensor_type_and_seed(conf)

    root = get_root_path()
    dataset = load_qh9_dataset(conf, root)
    loaders = create_qh9_data_loaders(dataset, conf, batch_size=[32, 32, 32])
    train_loader = loaders[0]

    lit_model = LitModel_flow(conf).cuda()
    lit_model.train()
    model = lit_model.model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warmup
    it = iter(train_loader)
    for _ in range(3):
        b = next(it).cuda()
        b = lit_model.post_processing(b, lit_model.default_type)
        b = lit_model.corrupt(b, mul=1)
        out = model(b, b.init_ham_t, keep_blocks=True)
        loss = out['hamiltonian_diagonal_blocks'].pow(2).mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad()
    torch.cuda.synchronize()

    # ============================================================
    # 1. Step-by-step breakdown
    # ============================================================
    print('\n' + '='*60)
    print('TRAINING STEP BREAKDOWN (30 steps)')
    print('='*60)

    times = {'data_load': [], 'to_gpu+corrupt': [], 'forward': [], 'loss+backward': [], 'optim_step': []}
    it = iter(train_loader)

    for step in range(30):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b = next(it)
        t1 = time.perf_counter()

        b = b.cuda()
        b = lit_model.post_processing(b, lit_model.default_type)
        b = lit_model.corrupt(b, mul=1)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        out = model(b, b.init_ham_t, keep_blocks=True)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        loss = (out['hamiltonian_diagonal_blocks'].pow(2).mean()
                + out['hamiltonian_non_diagonal_blocks'].pow(2).mean())
        loss.backward()
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t5 = time.perf_counter()

        times['data_load'].append(t1 - t0)
        times['to_gpu+corrupt'].append(t2 - t1)
        times['forward'].append(t3 - t2)
        times['loss+backward'].append(t4 - t3)
        times['optim_step'].append(t5 - t4)

    total = sum(np.mean(v) for v in times.values())
    print(f'\n  {"Component":20s} {"Mean":>8s} {"Std":>8s} {"% Total":>8s}')
    print(f'  {"-"*50}')
    for k, v in times.items():
        m = np.mean(v) * 1000
        s = np.std(v) * 1000
        pct = np.mean(v) / total * 100
        print(f'  {k:20s} {m:7.1f}ms {s:7.1f}ms {pct:7.1f}%')
    print(f'  {"TOTAL":20s} {total*1000:7.1f}ms')
    print(f'  Throughput: {32/total:.0f} samples/s')

    # ============================================================
    # 2. Graph construction inside forward (can we cache?)
    # ============================================================
    print('\n' + '='*60)
    print('GRAPH CONSTRUCTION COST')
    print('='*60)

    b = next(iter(train_loader)).cuda()
    b = lit_model.post_processing(b, lit_model.default_type)
    b = lit_model.corrupt(b, mul=1)

    from torch_cluster import radius_graph
    from e3nn import o3

    def graph_only():
        edge_index = radius_graph(b.pos, r=12, batch=b.batch, max_num_neighbors=32)
        vec = b.pos[edge_index[0]] - b.pos[edge_index[1]]
        dist = vec.norm(dim=-1)
        sh = o3.spherical_harmonics(
            o3.Irreps.spherical_harmonics(4),
            vec[:, [1, 2, 0]], normalize=True, normalization='component'
        ).type(b.pos.type())
        return edge_index, dist, sh

    # Warmup
    for _ in range(3): graph_only()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): graph_only()
    torch.cuda.synchronize()
    graph_t = (time.perf_counter() - t0) / 20 * 1000
    fwd_t = np.mean(times['forward']) * 1000

    print(f'  Graph construction: {graph_t:.1f} ms')
    print(f'  Full forward:      {fwd_t:.1f} ms')
    print(f'  Graph % of forward: {graph_t/fwd_t*100:.0f}%')
    print(f'  (Coords dont change in flow matching → cacheable!)')

    # ============================================================
    # 3. Dataloader num_workers scaling
    # ============================================================
    print('\n' + '='*60)
    print('DATALOADER NUM_WORKERS SCALING')
    print('='*60)

    from torch_geometric.loader import DataLoader
    train_ds = train_loader.dataset

    for nw in [0, 2, 4, 8, 12]:
        loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=nw, pin_memory=True)
        it = iter(loader)
        next(it)  # warmup
        t0 = time.perf_counter()
        for i in range(30):
            next(it)
        dt = (time.perf_counter() - t0) / 30
        print(f'  num_workers={nw:2d}: {dt*1000:.1f} ms/batch, {32/dt:.0f} samples/s')

    # ============================================================
    # 4. Pin memory effect
    # ============================================================
    print('\n' + '='*60)
    print('PIN_MEMORY EFFECT (num_workers=4)')
    print('='*60)

    for pin in [False, True]:
        loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin)
        it = iter(loader)
        next(it)
        t0 = time.perf_counter()
        for i in range(30):
            b = next(it)
            b = b.cuda()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 30
        print(f'  pin_memory={str(pin):5s}: {dt*1000:.1f} ms/batch (load+transfer)')


if __name__ == '__main__':
    main()
