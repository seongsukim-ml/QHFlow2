"""Summarize the bounded, matched QHFlow2 geometry benchmarks."""
from pathlib import Path
import json
import subprocess
ROOT=Path('/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-geometry-optimized-20261017')
OUT=Path('/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs')
MODELS=OUT/'qhflow2-geometry-models-20261017/attempt-2'
CONFIRM=OUT/'qhflow2-geometry-o3-confirm-20261017/attempt-1'
rows=[]
for model, baseline, candidate in [
 ('QHFlow2-s', MODELS/'QHFlow2_s-baseline-default',MODELS/'QHFlow2_s-candidate-default'),
 ('QHFlow2-O(3)-s', CONFIRM/'baseline',CONFIRM/'paired'),
 ('QHFlow2-m', MODELS/'QHFlow2_m-baseline-default',MODELS/'QHFlow2_m-candidate-default'),
 ('QHFlow2-l', MODELS/'QHFlow2_l-baseline-default',MODELS/'QHFlow2_l-candidate-default')]:
 a=json.loads((baseline/'result.json').read_text());b=json.loads((candidate/'result.json').read_text())
 assert a['status']==b['status']=='passed'
 assert a['train_ids']==b['train_ids'] and a['val_ids']==b['val_ids']
 keys=['train_samples_per_second','validation_samples_per_second','train_mean_seconds', 'validation_mean_seconds', 'train_peak_mib','total_peak_mib','first_gradient_preparation_seconds','first_validation_seconds','checks','microbatch','effective_batch']
 row={'model':model,'before':{k:a[k] for k in keys},'after':{k:b[k] for k in keys},'baseline_evidence':str(baseline/'result.json'),'candidate_evidence':str(candidate/'result.json')}
 for stage in ['train','validation']:
  row[stage+'_throughput_increase_percent']=100*(b[stage+'_samples_per_second']/a[stage+'_samples_per_second']-1)
  row[stage+'_time_reduction_percent']=100*(1-b[stage+'_mean_seconds']/a[stage+'_mean_seconds'])
 rows.append(row)
modes=json.loads((OUT/'qhflow2-geometry-modes-20261017/attempt-2/summary.json').read_text())
smokes={}
for name in ('o3-s','l'):
 p=OUT/f'qhflow2-geometry-{name}-resume-smoke-20261017/attempt-1/smoke-result.json'
 d=json.loads(p.read_text());assert d['status']=='passed'
 smokes[name]={k:d[k] for k in ['status','resumed_step','inference_cache_hits','cuda_max_allocated_mib','minibatch_order_bitwise_continuation']}
 smokes[name]['evidence']=str(p)
report={'schema':'qhflow2-geometry-benchmark/v1','integration_source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'measured_source_commit':'692f0d5e1ffb68edba5700fd93fc8e9dc77c8fd5','baseline_source_commit':'28d72168d3e774210ac187c3f32c147c7c228a8c',
 'protocol':{'gpu':'H100 80GB HBM3','training_updates':24,'warmup_updates':4,'effective_batch':32,'validation_batches':3,'validation_batch_size':32,'sample_definition':'one original molecule, including for paired O(3)','precision':'FP32, TF32 disabled','dataset':'QH9Stable official random split','split_sha256':'b6bfadb51aedc894f388ffd786c6c3a682d900aa0ff229a94b678b4c72b898c1','train_includes':'loader wait, H2D, forward, backward, gradient clipping, native Muon and AdamW','validation_includes':'loader wait, H2D, loss, ODE sampling and all 16 sampling/orbital metrics','scope':'steady-state bounded timings; not a complete epoch, convergence or end-to-end training forecast','gpu_indices':{'s_m_l':4,'o3_confirmation':6,'modes':3},'cpu_shared':True,'compiler_cache_shared':True},
 'rows':rows,'compile_modes':[{k:v for k,v in x.items() if k in ['mode','status','reason','train_samples_per_second','validation_samples_per_second','first_gradient_preparation_seconds']} for x in modes],'resume_smokes':smokes,'full_training_restarted':False,
 'test_log':str(OUT/'qhflow2-geometry-models-20261017/final-tests.log')}
p=OUT/'qhflow2-geometry-models-20261017/final-geometry-report.json';p.write_text(json.dumps(report,indent=2)+'\n')
lines=['# v0.2.0-rc.2: geometry and execution follow-up', '',
 'Development candidate, measured 2026-10-16 UTC. No new release tag or trained checkpoint release is published. Full training remains stopped.', '',
 'This second round compares the already optimized and compiled first-round source (`28d7216`) with geometry caching and paired O(3) execution (`692f0d5`). The [first-round report](v0.2.0-rc.2.md) retains its original measurements.', '',
 '## Throughput', '',
 'One sample means one original molecule; the two internal O(3) branches do not count as two samples. All four models retain the ordering fix.', '',
 '| Model | Training before → after (samples/s) | Training time reduction | Validation before → after (samples/s) | Validation time reduction |',
 '| --- | ---: | ---: | ---: | ---: |']
for row in rows:
 a,b=row['before'],row['after']
 lines.append(f"| {row['model']} | {a['train_samples_per_second']:.1f} → **{b['train_samples_per_second']:.1f}** | {row['train_time_reduction_percent']:.1f}% | {a['validation_samples_per_second']:.1f} → **{b['validation_samples_per_second']:.1f}** | {row['validation_time_reduction_percent']:.1f}% |")
lines += ['', 'Training throughput for s/m/l is effectively unchanged within short-run variation. O(3)-s benefits from evaluating the identity/inversion branches as one batch. Validation benefits from reusing graph geometry, spherical harmonics, Wigner matrices, AO assembly/convention plans and target spectra across ODE/metric calls.', '',
 'H100 80GB HBM3, PyTorch 2.11.0+cu126, FP32 with TF32 disabled, native Muon 1e-3 and AdamW 1e-4. Effective batch 32: microbatch 32 for s/O(3)-s/m, 16 with two accumulation steps for l. Each pair uses the same GPU, initial weights, seed, official split and distinct sampled batches. Four warmups precede 24 measured updates (768 molecules), and three validation batches cover 96 distinct molecules after one warm batch. Sampling/orbital metrics are forced on and all 16 must be finite.', '',
 'Training includes loader wait, host-to-device transfer, forward/backward, clipping and optimizer updates; validation includes loading, loss, ODE sampling and orbital metrics. Disk compiler caches are shared. CPU/I/O resources are shared with other jobs; each GPU is isolated. These bounded measurements do not establish complete-epoch or full-training wall time. The first-round fixed-batch GPU-only measurements use a different protocol and must not be multiplied by these speedups.', '',
 '## Memory and preparation', '',
 '| Model | Peak allocated training memory before → after (GiB) | Candidate first-gradient preparation (s) | Candidate first validation (s) |',
 '| --- | ---: | ---: | ---: |']
for row in rows:
 a,b=row['before'],row['after'];lines.append(f"| {row['model']} | {a['train_peak_mib']/1024:.2f} → {b['train_peak_mib']/1024:.2f} | {b['first_gradient_preparation_seconds']:.1f} | {b['first_validation_seconds']:.1f} |")
lines += ['', 'Peak allocated memory covers the tested batches, not the entire dataset or device-wide reserved memory. First-gradient preparation includes execution and saving the gradient fixture; it is not pure compiler time or a cold-cache bound.', '',
 '## Compiler mode selection', '',
 'Keep selective `torch.compile` on backbone blocks and tensor heads with `dynamic=True`, `fullgraph=True`, and mode `default`. Graph construction stays eager. No FlashSO2 weights, architecture or custom rotation kernels are imported.', '',
 '- `default`: all throughput, gradient, variable-shape, equivariance and checkpoint checks passed.',
 '- `max-autotune-no-cudagraphs`: preparation did not finish within the 600-second per-case budget; no valid throughput is reported.',
 '- `reduce-overhead` and `max-autotune`: the child processes exited with SIGSEGV (code -11) in this environment. They are not enabled. This is an observed failure, not an established root-cause diagnosis.', '',
 'The timeout does not establish that autotuning has worse eventual steady-state performance. It remains unvalidated here. Rejected-case logs and interrupted first attempts are retained.', '',
 '## Correctness and integration', '',
 '- Initial loss and every parameter gradient matched within rtol 3e-4 / atol 2e-6; the paired O(3) matched comparison had maximum gradient difference 2.79e-9. Rotation/inversion and tested dynamic-shape checks passed. State-dict names are unchanged.',
 '- Cache lifetime is one no-grad validation/sampling context and cleanup is exception-safe. Keys include strong tensor references and mutation versions. Gradient-enabled geometry calls and inference tensors without version counters bypass the cache. Learned radial features are recomputed; cached Wigner frames consume the original random draw to preserve downstream RNG state.',
 '- The optimized preset uses Lightning `inference_mode: false`, retaining no-grad validation while preserving tensor version counters. Compile settings now reach the normal QH9 entrypoint, after any pretrained-backbone replacement. Legacy presets retain their prior defaults.',
 '- Fixed missing forwarding of the target-spectrum cache to validation metrics and removed an undefined cache variable from prediction. The first-round implementation contained the cache but its main validation call did not activate it; the earlier description overstated its activation. Its measured timings are preserved.',
 '- Actual Lightning resume checks restored original model/optimizer/scheduler state exactly, then ran two updates: O(3)-s 6000→6002 and l 1000→1002. Both reloaded saved checkpoints and recorded three cache hits per geometry/AO/Wigner namespace during validation. Original checkpoints remain untouched. Shuffled sample order after a changed microbatch is not a bitwise continuation.',
 '- CPU regression results are recorded in the linked final test log. These checks establish implementation compatibility, not convergence, final accuracy, DDP behavior or all possible molecular sizes.', '',
 '## Prepared use', '',
 'Source checkout: `/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-geometry-optimized-20261017`.',
 'The public `experiment.train_qh9_stable_muon` entrypoint accepts `--config-name=config_stable_muon_optimized`; the opt-in O(3)-s paired preset is `--config-name=config_stable_muon_optimized_o3`. Select m with `model=QHFlow2_m model.train_batch_size=32 model.accumulate_grad_batches=1`; select l with `model=QHFlow2_l model.train_batch_size=16 model.accumulate_grad_batches=2`. Keep a new canonical output directory for any future restart.',
 'Paired execution remains opt-in; only O(3)-s was benchmarked on GPU. Larger paired variants have no throughput/memory claim. Full training was not restarted.', '',
 '## Evidence', '',
 f'- Consolidated results: `{p}`',
 f'- Model comparison: `{MODELS}/summary.json`',
 f'- O(3) matched confirmation: `{CONFIRM}/summary.json`',
 f'- Compiler modes: `{OUT}/qhflow2-geometry-modes-20261017/attempt-2/summary.json`',
 f'- Final tests: `{report["test_log"]}`',
 f'- Checkpoint smoke evidence: `{OUT}/qhflow2-geometry-o3-s-resume-smoke-20261017/attempt-1/smoke-result.json` and `{OUT}/qhflow2-geometry-l-resume-smoke-20261017/attempt-1/smoke-result.json`', '']
(ROOT/'docs/releases/v0.2.0-rc.2-geometry.md').write_text('\n'.join(lines))
print(json.dumps([{k:r[k] for k in ['model','train_time_reduction_percent','validation_time_reduction_percent']} for r in rows],indent=2))
