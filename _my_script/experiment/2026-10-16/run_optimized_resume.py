#!/usr/bin/env python3
"""Pinned QH9Stable training with local metrics, CUDA preflight, and recovery."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

SOURCE = Path('/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-compute-v020rc2-20261016')
OUTPUTS = Path('/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs')
SPLIT_SHA256 = 'b6bfadb51aedc894f388ffd786c6c3a682d900aa0ff229a94b678b4c72b898c1'
sys.path.insert(0, str(SOURCE / 'src'))


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def resolve(model):
    from hydra import compose, initialize_config_dir
    from experiment.train_qh9_stable_muon import validate_config
    from qhflow2_version import get_code_metadata

    with initialize_config_dir(version_base=None, config_dir=str(SOURCE / 'src/config_qh9')):
        conf = compose(config_name='config_stable_muon_optimized', overrides=[f'model={model}'])
    conf.model.train_batch_size = 16 if model == 'QHFlow2_l' else 32
    conf.model.accumulate_grad_batches = 32 // conf.model.train_batch_size
    conf.dataset.valid_batch_size = 32
    conf.dataset.test_batch_size = 32
    validate_config(conf)
    assert conf.model.version in {"QHFlow2_order_fixed", "QHFlow2_O3"}
    assert conf.model.use_block_H_fix
    assert conf.muon.lr == 1e-3 and conf.muon.adamw_lr == 1e-4
    assert conf.dataset.batch_size * conf.accumulate_grad_batches == 32
    assert conf.num_training_steps == 260000 and conf.warmup_step == 0
    split_path = Path(conf.dataset.split_file)
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == SPLIT_SHA256
    split = json.loads(split_path.read_text())
    counts = {key: len(split[key]) for key in ('train', 'val', 'test')}
    assert counts == {'train': 104664, 'val': 13083, 'test': 13084}
    source = get_code_metadata()
    assert source['code_version'] == '0.2.0-rc.2'
    return conf, {'source': source, 'model_config': model, 'split_sha256': SPLIT_SHA256,
                  'split_counts': counts, 'muon_lr': conf.muon.lr,
                  'adamw_lr': conf.muon.adamw_lr, 'microbatch': conf.dataset.batch_size,
                  'accumulation': conf.accumulate_grad_batches, 'optimizer_steps': conf.num_training_steps}


def run(args):
    conf, receipt = resolve(args.model)
    if args.scope == 'validate':
        print(json.dumps(receipt, indent=2))
        return
    output = Path(args.output)
    if not args.checkpoint:
        raise ValueError('A preserved parent checkpoint is required')
    if not output.is_absolute() or OUTPUTS not in output.parents:
        raise ValueError('Use the absolute canonical MALOQ output directory')
    if receipt['source']['git_dirty']:
        raise RuntimeError('Training requires the committed, frozen source tree')
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch_geometric.loader import DataLoader
    from omegaconf import OmegaConf
    from common.setup import setup_tensor_type_and_seed
    from common.qh9_utils import load_qh9_dataset, create_qh9_data_loaders
    from pl_module.flow_module import LitModel_flow

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('This launcher requires exactly one queue-allocated CUDA GPU')
    setup_tensor_type_and_seed(conf)
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    saved_parent = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    resume_step = int(saved_parent['global_step'])
    expected_rates = [g['lr'] for g in saved_parent['optimizer_states'][0]['param_groups']]
    expected_scheduler = saved_parent['lr_schedulers'][0]
    expected_model_version = saved_parent['qhflow2']['model_version']
    assert expected_model_version == conf.model.version
    receipt['resume'] = {'checkpoint':str(checkpoint_path), 'step':resume_step,
                         'epoch':saved_parent['epoch'], 'learning_rates':expected_rates,
                         'parent_source':saved_parent['qhflow2'],
                         'sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}

    torch.set_num_threads(8)
    receipt.update(compile_scope=args.compile_scope, scope=args.scope, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                   queue_job_id=os.environ.get('SC26_QUEUE_JOB_ID'),
                   queue_attempt=os.environ.get('SC26_QUEUE_ATTEMPT'),
                   physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write_json(output / 'run.json', receipt)
    OmegaConf.save(conf, output / 'resolved-config.yaml', resolve=True)
    dataset = load_qh9_dataset(conf, str(SOURCE))
    try:
        train_loader, val_loader, test_loader = create_qh9_data_loaders(dataset, conf)
        if args.scope == 'smoke':
            # Keep the full training loader for Lightning's restored mid-epoch
            # counters. Only two updates execute; validation uses official rows.
            val_ids = dataset.val_mask[torch.linspace(0, len(dataset.val_mask)-1, 32).long()]
            val_loader = DataLoader(dataset[val_ids], batch_size=32, shuffle=False, num_workers=0)
            write_json(output/'smoke-sample-ids.json', {'val':val_ids.tolist()})

        class Progress(Callback):
            def __init__(self):
                self.last_step = -1
                self.initial_rates = None

            def on_train_start(self, trainer, module):
                self.initial_rates = [group['lr'] for group in trainer.optimizers[0].param_groups]
                assert trainer.global_step == resume_step
                assert self.initial_rates == expected_rates
                assert trainer.lr_scheduler_configs[0].scheduler.state_dict() == expected_scheduler
                optimizer = trainer.optimizers[0]
                groups = saved_parent['optimizer_states'][0]['param_groups']
                for group, original_group in zip(optimizer.param_groups, groups):
                    assert group['param_names'] == original_group['param_names']
                    for parameter, original_id in zip(group['params'], original_group['params']):
                        expected_state = saved_parent['optimizer_states'][0]['state'].get(original_id, {})
                        actual_state = optimizer.state.get(parameter, {})
                        assert actual_state.keys() == expected_state.keys()
                        for key, expected in expected_state.items():
                            actual = actual_state[key]
                            if isinstance(expected, torch.Tensor):
                                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                            else:
                                assert actual == expected
                write_json(output/'resume-verified.json', {'step':trainer.global_step,
                            'learning_rates':self.initial_rates, 'all_optimizer_states_exact':True,
                            'scheduler_state_exact':True, 'strict_model_load':True})
                # Release the CPU copy once exact restoration has been checked.
                saved_parent.clear()

            def on_before_optimizer_step(self, trainer, module, optimizer):
                if trainer.global_step < resume_step + 3:
                    finite = [torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None]
                    if not torch.stack(finite).all():
                        raise FloatingPointError('Nonfinite gradient')

            def record(self, trainer, phase, loss=None):
                metrics = {key: float(value.detach().cpu()) for key, value in trainer.callback_metrics.items()
                           if isinstance(value, torch.Tensor) and value.numel() == 1}
                status = {'phase': phase, 'step': trainer.global_step, 'epoch': trainer.current_epoch,
                          'updated_at': datetime.now(timezone.utc).isoformat(), 'metrics': metrics,
                          'learning_rates': [g['lr'] for g in trainer.optimizers[0].param_groups],
                          'cuda_max_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
                if loss is not None:
                    status['batch_loss'] = float(loss.detach().cpu())
                write_json(output / 'status.json', status)
                with (output / 'progress.jsonl').open('a') as handle:
                    handle.write(json.dumps(status) + '\n')
                print(json.dumps(status), flush=True)

            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                if loss is None or not torch.isfinite(loss).all():
                    raise FloatingPointError('Nonfinite training loss')
                step = trainer.global_step
                if step != self.last_step and (step <= resume_step + 5 or step % 25 == 0):
                    self.last_step = step
                    self.record(trainer, 'training', loss)

            def on_validation_end(self, trainer, module):
                if trainer.sanity_checking:
                    return
                loss = trainer.callback_metrics.get('val/loss')
                if loss is None or not torch.isfinite(loss).all():
                    raise FloatingPointError('Missing or nonfinite validation loss')
                self.record(trainer, 'validation')

        def make_trainer(phase, max_epochs, max_steps):
            progress = Progress()
            checkpoint = ModelCheckpoint(
                dirpath=output / phase / 'checkpoints', filename='epoch{epoch:03d}-step{step:08d}',
                monitor='val/loss', mode='min', save_top_k=1, save_last=True,
                auto_insert_metric_name=False,
            )
            callbacks = [progress, LearningRateMonitor(logging_interval='step'), checkpoint]
            if args.scope == 'full':
                callbacks.append(ModelCheckpoint(
                    dirpath=output / 'recovery', filename='step{step:08d}',
                    every_n_train_steps=250, save_top_k=1, save_last=True,
                    auto_insert_metric_name=False,
                ))
            trainer = pl.Trainer(
                accelerator='gpu', devices=1, precision=32, max_epochs=max_epochs, max_steps=max_steps,
                accumulate_grad_batches=conf.accumulate_grad_batches, gradient_clip_val=conf.gradient_clip_val,
                callbacks=callbacks, logger=CSVLogger(str(output), name='metrics', version=phase),
                default_root_dir=str(output), enable_progress_bar=False, log_every_n_steps=1 if args.scope == 'smoke' else 25,
                num_sanity_val_steps=0, check_val_every_n_epoch=1,
            )
            return trainer, progress, checkpoint

        module = LitModel_flow(conf)
        module.load_state_dict(saved_parent['state_dict'], strict=True)
        # Compilation selection is frozen after the paired GPU checks.
        if args.compile_scope != 'none':
            from common.compile import compile_flow
            compile_flow(module, args.compile_scope)
        torch.cuda.reset_peak_memory_stats()
        max_steps = resume_step + 2 if args.scope == 'smoke' else conf.num_training_steps
        trainer, progress, checkpoint = make_trainer('training', -1, max_steps)
        trainer.fit(module, train_loader, val_loader, ckpt_path=str(checkpoint_path), weights_only=False)
        if args.scope == 'smoke':
            assert trainer.global_step == resume_step + 2
            # Force the complete sampling protocol, including projected O(3)
            # outputs, irrespective of the current loss threshold.
            module.error_threshold = None
            trainer.validate(module, val_loader)
            metrics = {key:float(value.cpu()) for key,value in trainer.callback_metrics.items()
                       if isinstance(value,torch.Tensor) and value.numel()==1}
            assert any('sample_' in key for key in metrics)
            assert all(__import__('math').isfinite(value) for value in metrics.values())
            trainer.save_checkpoint(output/'resumed.ckpt')
            reloaded = torch.load(output/'resumed.ckpt', map_location='cpu', weights_only=False)
            assert reloaded['global_step'] == trainer.global_step
            for key, value in module.state_dict().items():
                torch.testing.assert_close(reloaded['state_dict'][key], value.cpu(), rtol=0, atol=0)
            receipt.update(status='passed', resumed_step=trainer.global_step, metrics=metrics,
                           cuda_max_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            write_json(output/'smoke-result.json', receipt)
        else:
            metrics = trainer.test(module, test_loader, ckpt_path='best', weights_only=False)
            write_json(output/'completed.json', {'step':trainer.global_step, 'test':metrics,
                                                 'best_checkpoint':checkpoint.best_model_path})
    finally:
        dataset.close()


def exclusive_compute():
    """Keep this allocated GPU free of adaptive filler during real training."""
    from contextlib import contextmanager
    import socket
    import threading
    sys.path.insert(0, '/dataset/seongsu/shared-home/workspace/_global_auto_script/experiment_queue')
    import sc26_queue as queue
    paths = queue.QueuePaths(Path('/dataset/seongsu/shared-home/workspace/project/MALOQ'),
                            Path('/dataset/seongsu/shared-home/workspace/_global_auto_script_output/experiment_queue'))
    gpu = int(os.environ['SC26_QUEUE_GPUS'])
    job = os.environ['SC26_QUEUE_JOB_ID']
    host = socket.gethostname()
    @contextmanager
    def guard():
        if not queue.request_autoresearch_drain(paths, host, gpu, job):
            raise RuntimeError('Cannot acquire job-owned compute drain')
        stop = threading.Event()
        def renew():
            while not stop.wait(15):
                queue.request_autoresearch_drain(paths, host, gpu, job)
        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=20)
            queue.release_autoresearch_drain_request(paths, host, gpu, job_id=job)
    return guard()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope', choices=('validate', 'smoke', 'full'))
    parser.add_argument('model', choices=('QHFlow2_s', 'QHFlow2_O3_s', 'QHFlow2_m', 'QHFlow2_l'))
    parser.add_argument('--output')
    parser.add_argument('--checkpoint')
    parser.add_argument('--compile-scope', choices=('none','heads','blocks'), default='none')
    args = parser.parse_args()
    if args.scope == 'validate':
        run(args)
    else:
        with exclusive_compute():
            run(args)
