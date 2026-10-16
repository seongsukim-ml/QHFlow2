#!/usr/bin/env python3
"""Paired, varying-batch throughput and full-validation comparisons."""
import argparse,json,os,statistics,subprocess,sys,time
from pathlib import Path
SOURCE=Path('/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-geometry-optimized-20261017')
BASELINE=Path('/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-compute-v020rc2-20261016')
PYTHON='/home/gpuuser/dataset/_cache/qhflow2-stable-muon-venv/bin/python'
sys.path.insert(0,'/dataset/seongsu/shared-home/workspace/project/MALOQ/_auto_script/qhflow2_compute_20261016')
from gpu_guard import exclusive_compute,check_gpu_processes


def child(args):
    code=BASELINE if args.variant=='baseline' else SOURCE
    sys.path.insert(0,str(code/'src'))
    sys.path.insert(0,str(BASELINE/'_my_script/experiment/2026-10-16'))
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from hydra import compose,initialize_config_dir
    from common.setup import setup_tensor_type_and_seed
    from common.qh9_utils import load_qh9_dataset
    from common.compile import compile_flow
    from pl_module.flow_module import LitModel_flow
    from benchmark_compute import check_dynamic_and_equivariance
    with initialize_config_dir(version_base=None,config_dir=str(code/'src/config_qh9')):
        conf=compose(config_name='config_stable_muon_optimized',overrides=[f'model={args.model}'])
    setup_tensor_type_and_seed(conf);torch.set_num_threads(8)
    dataset=load_qh9_dataset(conf,str(code))
    micro=16 if args.model=='QHFlow2_l' else 32
    accumulation=32//micro
    generator=torch.Generator().manual_seed(3407)
    train_ids=dataset.train_mask[torch.randperm(len(dataset.train_mask),generator=generator)[:32*28]].tolist()
    val_ids=dataset.val_mask[torch.randperm(len(dataset.val_mask),generator=generator)[:32*4]].tolist()
    train=DataLoader(Subset(dataset,train_ids),batch_size=micro,shuffle=False,num_workers=4,
                     pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(11))
    validation=DataLoader(Subset(dataset,val_ids),batch_size=32,shuffle=False,num_workers=4,
                     pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(12))
    module=LitModel_flow(conf).cuda()
    if args.variant=='paired': module.model.batch_branches=True
    if args.variant=='baseline':compile_flow(module,'blocks')
    else:compile_flow(module,'blocks',mode=None if args.mode=='default' else args.mode)
    optimizer=module.configure_optimizers()[0][0]
    logged={};module.log=lambda key,value,**kw:logged.__setitem__(key,value.detach() if isinstance(value,torch.Tensor) else value)
    # Read one fixed microbatch for full gradient parity; loader has its own RNG.
    fixture=next(iter(train)).to('cuda',non_blocking=True)
    module.train();torch.manual_seed(101)
    start=time.perf_counter();loss=module.training_step(fixture.clone(),0);loss.backward();torch.cuda.synchronize()
    torch.save({'loss':loss.detach().cpu(),'parameter_names':[n for n,p in module.model.named_parameters()],
                'grads':{n:p.grad.detach().cpu() for n,p in module.model.named_parameters() if p.grad is not None}},args.output/'first-gradient.pt')
    preparation=time.perf_counter()-start
    optimizer.zero_grad(set_to_none=True)
    def update(iterator,index):
        optimizer.zero_grad(set_to_none=True)
        data_wait=0.
        for part in range(accumulation):
            tick=time.perf_counter();batch=next(iterator);data_wait+=time.perf_counter()-tick
            batch=batch.to('cuda',non_blocking=True)
            loss=module.training_step(batch,index*accumulation+part)/accumulation
            loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(),conf.gradient_clip_val);optimizer.step()
        torch.cuda.synchronize()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        return data_wait,float(loss.detach())
    iterator=iter(train)
    for i in range(4):update(iterator,i)
    check_gpu_processes(int(os.environ['SC26_QUEUE_GPUS']))
    torch.cuda.reset_peak_memory_stats();timings=[];waits=[];losses=[];graph_counts=[]
    start=time.perf_counter()
    for i in range(24):
        tick=time.perf_counter();wait,loss=update(iterator,i+4);timings.append(time.perf_counter()-tick)
        waits.append(wait);losses.append(loss);graph_counts.append(int(torch._dynamo.utils.counters['stats']['unique_graphs']))
    train_wall=time.perf_counter()-start
    train_peak=torch.cuda.max_memory_allocated()/2**20
    check_gpu_processes(int(os.environ['SC26_QUEUE_GPUS']))
    module.eval();module.error_threshold=None
    val_iterator=iter(validation)
    with torch.no_grad():
        tick=time.perf_counter();module.validation_step(next(val_iterator).to('cuda',non_blocking=True),0);torch.cuda.synchronize()
        first_validation=time.perf_counter()-tick
        val_times=[];metric_counts=[]
        for index in range(3):
            logged.clear();tick=time.perf_counter()
            module.validation_step(next(val_iterator).to('cuda',non_blocking=True),index+1);torch.cuda.synchronize()
            val_times.append(time.perf_counter()-tick)
            metric_counts.append(sum('sample_' in name for name in logged))
            assert metric_counts[-1]>=16, f'Sampling metrics missing: {list(logged)}'
            assert all(torch.isfinite(v).all() for v in logged.values() if isinstance(v,torch.Tensor))
    peak=torch.cuda.max_memory_allocated()/2**20
    check_gpu_processes(int(os.environ['SC26_QUEUE_GPUS']))
    # Numerical transform and shape checks, outside the timing window.
    graphs=[dataset[int(i)] for i in train_ids[:32]]
    checks=check_dynamic_and_equivariance(module,graphs,optimizer,'blocks')
    assert len(set(graph_counts[4:]))==1, f'Continued shape recompilation: {graph_counts}'
    saved={'model':module.state_dict(),'optimizer':optimizer.state_dict()}
    path=args.output/'checkpoint-smoke.pt';torch.save(saved,path)
    restored=torch.load(path,map_location='cpu',weights_only=False)
    module.load_state_dict(restored['model'],strict=True);optimizer.load_state_dict(restored['optimizer'])
    for key,value in module.state_dict().items():torch.testing.assert_close(value.cpu(),restored['model'][key],atol=0,rtol=0)
    result={'model':args.model,'variant':args.variant,'mode':args.mode,'source_commit':subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
            'microbatch':micro,'effective_batch':32,'updates':24,'train_wall_seconds':train_wall,
            'train_mean_seconds':train_wall/24,'train_median_seconds':statistics.median(timings),'train_timings':timings,
            'loader_wait_seconds':sum(waits),'train_peak_mib':train_peak,'total_peak_mib':peak,
            'validation_32_graph_seconds':val_times,'validation_mean_seconds':statistics.mean(val_times),
            'first_validation_seconds':first_validation,'first_gradient_preparation_seconds':preparation,
            'sample_metric_counts':metric_counts,'train_ids':train_ids,'val_ids':val_ids,
            'losses':losses,'unique_graphs':graph_counts,'checks':checks,'checkpoint_reload':True,'status':'passed'}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('train_ids','val_ids','losses','train_timings')}),flush=True)
    dataset.close()


def parent(args):
    if args.suite=='models':
        cases=[(m,v,'default') for m in ['QHFlow2_s','QHFlow2_O3_s','QHFlow2_m','QHFlow2_l'] for v in ['baseline','candidate']]
    elif args.suite=='modes':
        cases=[('QHFlow2_m','candidate',m) for m in ['default','max-autotune-no-cudagraphs','reduce-overhead','max-autotune']]
    else:cases=[('QHFlow2_O3_s',v,'default') for v in ['candidate','paired']]
    import torch
    results=[]
    with exclusive_compute():
        for model,variant,mode in cases:
            out=args.output/f'{model}-{variant}-{mode}';out.mkdir()
            command=[PYTHON,str(SOURCE/'_auto_script/benchmark_geometry.py'),'child','--model',model,'--variant',variant,'--mode',mode,'--output',str(out)]
            result={'model':model,'variant':variant,'mode':mode}
            try:
                with (out/'console.log').open('w') as log:
                    proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
                if proc.returncode:raise RuntimeError(f'exit code {proc.returncode}')
                result.update(json.loads((out/'result.json').read_text()))
                if len(results) and ((args.suite=='models' and variant=='candidate') or args.suite!='models'):
                    ref=results[-1] if args.suite=='models' else results[0]
                    refpath=args.output/f"{ref['model']}-{ref['variant']}-{ref['mode']}"/'first-gradient.pt'
                    a=torch.load(refpath,weights_only=True);b=torch.load(out/'first-gradient.pt',weights_only=True)
                    assert a['parameter_names']==b['parameter_names'] and a['grads'].keys()==b['grads'].keys()
                    torch.testing.assert_close(a['loss'],b['loss'],rtol=1e-5,atol=1e-7)
                    maxima=[]
                    for key in a['grads']:
                        torch.testing.assert_close(a['grads'][key],b['grads'][key],rtol=3e-4,atol=2e-6,msg=key)
                        maxima.append(float((a['grads'][key]-b['grads'][key]).abs().max()))
                    result['gradient_max_abs_error']=max(maxima)
            except Exception as exc:
                result.update(status='rejected',reason=str(exc))
            results.append(result)
            (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('train_ids','val_ids','losses','train_timings')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope',choices=['parent','child']);p.add_argument('--suite',choices=['models','modes','o3'],default='models')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model');p.add_argument('--variant');p.add_argument('--mode',default='default')
    args=p.parse_args();(parent if args.scope=='parent' else child)(args)
