"""Measure checkpoint inference at one and three ODE steps on the same cohort.

CUDA timings include the flow prior, geometry construction and Hamiltonian-block
prediction. Loading, H2D copies, batch cloning, metrics and DFT are excluded.
No training throughput or extrapolation is used. Compilation is disabled for
every model; the same FP32/high-matmul-precision policy is applied to all rows.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def validate(manifest, verify_hashes=True):
    if manifest['ode_steps'] != [1, 3]:
        raise ValueError('Measure both actual 1-step and 3-step inference')
    for key in ('batch_size', 'warmup_sweeps', 'repeats'):
        if not isinstance(manifest[key], int) or manifest[key] < 1:
            raise ValueError(f'Positive integer required: {key}')
    ids = manifest['dataset_indices']
    if not ids or len(set(ids)) != len(ids) or len(ids) % manifest['batch_size']:
        raise ValueError('Cohort must contain unique IDs and complete batches')
    split = json.loads(Path(manifest['split_file']).read_text())
    if not set(ids) <= set(split['test']):
        raise ValueError('Cohort contains molecules outside the test split')
    if sha(manifest['split_file']) != manifest['split_sha256']:
        raise ValueError('Split changed')
    models = manifest['models']
    if not models or len({m['id'] for m in models}) != len(models):
        raise ValueError('Model IDs must be nonempty and unique')
    for m in models:
        if Path(m['id']).name != m['id'] or m['id'] in ('.', '..'):
            raise ValueError('Invalid model ID')
        if verify_hashes and sha(m['checkpoint']) != m['sha256']:
            raise ValueError(f"Checkpoint changed: {m['id']}")


def summarize_sweeps(sweeps, sample_count):
    if not sweeps or any(not math.isfinite(v) or v <= 0 for v in sweeps):
        raise ValueError('Timings must be finite and positive')
    median = statistics.median(sweeps)
    return dict(samples_per_second=sample_count / median,
                milliseconds_per_sample=1000 * median / sample_count,
                median_sweep_seconds=median, sweep_seconds=sweeps,
                min_samples_per_second=sample_count / max(sweeps),
                max_samples_per_second=sample_count / min(sweeps))


def run(args):
    manifest = json.loads(args.manifest.read_text())
    validate(manifest)
    if args.validate:
        print('Manifest, cohort and checkpoint hashes verified')
        return
    import torch
    from omegaconf import OmegaConf, open_dict
    from common.setup import setup_tensor_type_and_seed
    from dataset_module.qh9_stable_readonly import QH9StableReadonly
    from pl_module import get_pl_model
    from torch_geometric.loader import DataLoader

    if not args.cpu_smoke and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for publishable speed measurements')
    device = 'cpu' if args.cpu_smoke else 'cuda:0'
    torch.set_num_threads(manifest.get('cpu_threads', 4))
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = QH9StableReadonly(manifest['processed_directory'], manifest['split_file'],
        compute_q_tensor=False, include_cut_orbital_coefficients=False)
    ids = manifest['dataset_indices'][:1] if args.cpu_smoke else manifest['dataset_indices']
    loader = DataLoader(dataset[torch.tensor(ids)], batch_size=1 if args.cpu_smoke else manifest['batch_size'],
                        shuffle=False, num_workers=0)
    batches = list(loader)
    dataset.close()
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    source = {str(p.relative_to(root)): sha(p) for p in sorted((root / 'src').rglob('*.py'))}
    protocol = dict(manifest=manifest, manifest_sha256=sha(args.manifest), source_commit=commit,
        source_sha256=source, python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, device=device, publishable=not args.cpu_smoke,
        gpu_name=torch.cuda.get_device_name() if device != 'cpu' else None,
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device != 'cpu' else None,
        compile=False, dtype='float32', float32_matmul_precision='high',
        timing='Synchronized perf_counter wall time, median of full-cohort sweeps',
        included=['flow prior', 'geometry construction', 'ODE steps', 'Hamiltonian blocks'],
        excluded=['checkpoint/dataset loading', 'H2D copies', 'batch cloning', 'warmup',
                  'accuracy metrics', 'full-matrix assembly', 'SCF', 'DFT energy/forces'])
    write_json(args.output / 'protocol.json', protocol)
    results = {}
    selected = [m for m in manifest['models'] if not args.model or m['id'] == args.model]
    if not selected:
        raise ValueError('Requested model is absent from manifest')

    def sync():
        if device != 'cpu':
            torch.cuda.synchronize()

    for spec in selected:
        checkpoint = torch.load(spec['checkpoint'], map_location='cpu', mmap=True,
                                weights_only=not spec.get('trust_legacy_checkpoint', False))
        conf = OmegaConf.create(checkpoint['hyper_parameters']['conf'])
        if not conf.qh9 or conf.data_type != 'float32':
            raise ValueError('Protocol requires QH9 FP32 checkpoints')
        with open_dict(conf):
            conf.seed = manifest['seed']
            conf.float32_matmul_precision = 'high'
        setup_tensor_type_and_seed(conf)
        model = get_pl_model(conf)(conf)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        del checkpoint
        model = model.to(device).eval()
        prepared = [model.post_processing(b.clone().to(device), model.default_type) for b in batches]
        per_steps = {}
        with torch.no_grad():
            for steps in manifest['ode_steps']:
                def forward(batch):
                    prediction, _, _ = model.sample(batch, num_timesteps=steps, return_trajectory=False)
                    return prediction

                # Warm all molecular shapes for each step count; clone to prevent
                # geometry from being reused across independent timed samples.
                for _ in range(1 if args.cpu_smoke else manifest['warmup_sweeps']):
                    for batch in prepared:
                        prediction = forward(batch.clone())
                        for key in ('hamiltonian_diagonal_blocks', 'hamiltonian_non_diagonal_blocks'):
                            if not torch.isfinite(prediction[key]).all():
                                raise ValueError(f"Nonfinite prediction: {spec['id']}, {steps} steps")
                        del prediction
                sync()
                if device != 'cpu':
                    torch.cuda.reset_peak_memory_stats()
                sweeps, batch_seconds = [], []
                for repeat in range(1 if args.cpu_smoke else manifest['repeats']):
                    torch.manual_seed(manifest['seed'] + repeat)
                    times = []
                    for batch in prepared:
                        fresh = batch.clone()
                        sync()
                        start = time.perf_counter()
                        prediction = forward(fresh)
                        sync()
                        times.append(time.perf_counter() - start)
                        del prediction, fresh
                    sweeps.append(sum(times))
                    batch_seconds.append(times)
                per_steps[str(steps)] = {**summarize_sweeps(sweeps, len(ids)),
                    'batch_seconds': batch_seconds,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device != 'cpu' else None}
        result = dict(model=spec['model'], group=spec['group'], checkpoint_sha256=spec['sha256'],
                      sample_count=len(ids), ode_steps=per_steps, publishable=not args.cpu_smoke)
        write_json(args.output / (spec['id'] + '.json'), result)
        results[spec['id']] = result
        # CPU smoke timings are diagnostic only; never expose them as a benchmark.
        print(spec['id'], 'CPU smoke passed' if args.cpu_smoke else per_steps, flush=True)
        del forward, prepared, batch, model
        gc.collect()
        if device != 'cpu':
            torch.cuda.empty_cache()
    write_json(args.output / 'summary.json', dict(protocol=protocol, models=results))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--cpu-smoke', action='store_true')
    p.add_argument('--model', help='Optional model ID; use the full manifest for final comparisons')
    run(p.parse_args())


if __name__ == '__main__':
    main()
