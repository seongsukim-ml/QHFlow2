#!/usr/bin/env python3
"""Matched CUDA benchmarks; fresh interpreter per source/model/batch size."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

SOURCE = Path('/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-compute-optimized-20261016')
BASELINE = Path('/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-stable-muon-ml-training-20261016')
PYTHON = '/home/gpuuser/dataset/_cache/qhflow2-stable-muon-venv/bin/python'


def child(args):
    code = BASELINE if args.variant == 'baseline' else SOURCE
    sys.path.insert(0, str(code / 'src'))
    import torch
    from hydra import compose, initialize_config_dir
    from torch_geometric.data import Batch
    from common.setup import setup_tensor_type_and_seed
    from common.qh9_utils import load_qh9_dataset
    from pl_module.flow_module import LitModel_flow
    with initialize_config_dir(version_base=None, config_dir=str(code/'src/config_qh9')):
        conf = compose(config_name='config_stable_muon', overrides=[f'model={args.model}'])
    setup_tensor_type_and_seed(conf)
    torch.set_num_threads(8)
    dataset = load_qh9_dataset(conf, str(code))
    # Identical official training rows across sources and microbatches.
    ids = dataset.train_mask[torch.linspace(0, len(dataset.train_mask)-1, 32).long()]
    graphs = [dataset[int(i)] for i in ids]
    batches = [Batch.from_data_list(graphs[i:i+args.batch]).to('cuda') for i in range(0, 32, args.batch)]
    val_ids = dataset.val_mask[torch.linspace(0, len(dataset.val_mask)-1, 8).long()]
    val = Batch.from_data_list([dataset[int(i)] for i in val_ids]).to('cuda')
    module = LitModel_flow(conf).cuda()
    logged = {}
    module.log = lambda key, value, **kw: logged.__setitem__(key, value.detach() if isinstance(value, torch.Tensor) else value)
    optimizer = module.configure_optimizers()[0][0]
    module.train()
    def step():
        optimizer.zero_grad(set_to_none=True)
        for index, template in enumerate(batches):
            loss = module.training_step(template.clone(), index) / len(batches)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), conf.gradient_clip_val)
        optimizer.step()
        return loss
    # Strict load compatibility and a matched first gradient fixture, before updates.
    torch.manual_seed(101)
    loss = module.training_step(batches[0].clone(), 0)
    loss.backward()
    fixture = {'loss':loss.detach().cpu(),
               'parameter_names':[n for n,p in module.model.named_parameters()],
               'grads':{n:p.grad.flatten()[::max(1,p.numel()//1024)].cpu() for n,p in module.model.named_parameters() if p.grad is not None}}
    torch.save(fixture, args.output/'first-gradient.pt')
    optimizer.zero_grad(set_to_none=True)
    for _ in range(2):step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings=[]
    for _ in range(6):
        start=time.perf_counter();loss=step();torch.cuda.synchronize();timings.append(time.perf_counter()-start)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite benchmark loss')
    if args.model == 'QHFlow2_l' and args.batch == 4:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            step();torch.cuda.synchronize()
        (args.output/'profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=45))
        (args.output/'profile-cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=45))
    train_peak=torch.cuda.max_memory_allocated()/2**20
    module.eval()
    validation=[]
    # This forces the complete existing validation protocol, including ODE
    # sampling and orbital metrics, even before the training loss threshold.
    module.error_threshold=None
    with torch.no_grad():
        for _ in range(2):
            logged.clear();start=time.perf_counter()
            module.validation_step(val.clone(),0);torch.cuda.synchronize()
            validation.append(time.perf_counter()-start)
    sample_metrics=[key for key in logged if 'sample_' in key]
    if args.variant != 'baseline':
        assert sample_metrics, 'Sampling metrics must actually be present'
        assert all(torch.isfinite(v).all() for v in logged.values() if isinstance(v,torch.Tensor))
    result={'variant':args.variant,'model':args.model,'microbatch':args.batch,'effective_batch':32,
            'train_seconds_per_update':statistics.median(timings),'train_timings':timings,
            'validation_8_graph_seconds':validation,'sample_metric_count':len(sample_metrics),
            'peak_memory_mib':train_peak,'source_commit':subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
            'training_ids':ids.tolist(),'validation_ids':val_ids.tolist()}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True);dataset.close()


def parent(args):
    import threading
    import socket
    sys.path.insert(0,'/dataset/seongsu/shared-home/workspace/_global_auto_script/experiment_queue')
    import sc26_queue as queue
    paths=queue.QueuePaths(Path('/dataset/seongsu/shared-home/workspace/project/MALOQ'),Path('/dataset/seongsu/shared-home/workspace/_global_auto_script_output/experiment_queue'))
    gpu=int(os.environ['SC26_QUEUE_GPUS']);job=os.environ['SC26_QUEUE_JOB_ID'];host=socket.gethostname()
    # Queue-supported, job-scoped drain: prevent adaptive filler GEMMs from
    # contaminating paired measurements. Other GPUs/services are unaffected.
    stop=threading.Event()
    def drain():
        while not stop.is_set():
            if not queue.request_autoresearch_drain(paths,host,gpu,job):
                raise RuntimeError('Cannot acquire benchmark-owned autoresearch drain')
            stop.wait(15)
    thread=threading.Thread(target=drain,daemon=True);thread.start()
    try:
        cases=[('QHFlow2_s',32,32),('QHFlow2_O3_s',16,32),('QHFlow2_m',16,32),('QHFlow2_l',4,16)]
        results=[]
        for model,original,tuned in cases:
            for variant,batch in [('baseline',original),('optimized',original)]+([('optimized',tuned)] if tuned!=original else []):
                out=args.output/f'{model}-{variant}-b{batch}';out.mkdir()
                cmd=[PYTHON,str(SOURCE/'_my_script/experiment/2026-10-16/benchmark_compute.py'),'child','--variant',variant,'--model',model,'--batch',str(batch),'--output',str(out)]
                with (out/'console.log').open('w') as log:
                    completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
                if completed.returncode:raise RuntimeError(f'Benchmark failed: {out}')
                result=json.loads((out/'result.json').read_text());results.append(result)
                print(json.dumps(result),flush=True)
        # Compare first gradients at the same microbatch and seed.
        import torch
        comparison={}
        for model,batch,_ in cases:
            ref=torch.load(args.output/f'{model}-baseline-b{batch}/first-gradient.pt',weights_only=True)
            opt=torch.load(args.output/f'{model}-optimized-b{batch}/first-gradient.pt',weights_only=True)
            assert ref['parameter_names']==opt['parameter_names']
            torch.testing.assert_close(ref['loss'],opt['loss'],rtol=1e-5,atol=1e-7)
            assert ref['grads'].keys()==opt['grads'].keys()
            maxima=[]
            for name in ref['grads']:
                torch.testing.assert_close(ref['grads'][name],opt['grads'][name],rtol=3e-4,atol=2e-6,msg=name)
                maxima.append(float((ref['grads'][name]-opt['grads'][name]).abs().max()))
            comparison[model]={'maximum_sampled_gradient_difference':max(maxima),'loss_difference':float((ref['loss']-opt['loss']).abs())}
        (args.output/'benchmark-summary.json').write_text(json.dumps({'results':results,'equivalence':comparison},indent=2)+'\n')
    finally:
        stop.set();thread.join(timeout=20)
        queue.release_autoresearch_drain_request(paths,host,gpu,job_id=job)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('scope',choices=['parent','child'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--variant');parser.add_argument('--model');parser.add_argument('--batch',type=int)
    args=parser.parse_args();(parent if args.scope=='parent' else child)(args)
