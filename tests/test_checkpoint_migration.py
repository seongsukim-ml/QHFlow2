import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from types import SimpleNamespace
import pytest
from common.checkpoint_migration import adapt_microbatch_progress


@pytest.mark.parametrize('old,new,processed,expected', [(16,32,3458,1729),(4,16,8000,2000),(32,32,2729,2729)])
def test_resume_batch_units_keep_optimizer_step(old,new,processed,expected):
    counter=dict(ready=processed,started=processed,processed=processed,completed=processed-1)
    checkpoint={'global_step':1000,'hyper_parameters':{'conf':SimpleNamespace(
        dataset=SimpleNamespace(batch_size=old),accumulate_grad_batches=32//old)},
        'state_dict':{'weight':object()},'optimizer_states':[object()], 'lr_schedulers':[object()],
        'loops':{'fit_loop':{'epoch_loop.batch_progress':{
            'current':counter.copy(),'total':counter.copy(),'is_last_batch':False},
            'epoch_loop.automatic_optimization.optim_progress':{'step':1000}}}}
    before=copy.deepcopy(checkpoint['loops'])
    migrated,receipt=adapt_microbatch_progress(checkpoint,new)
    assert checkpoint['loops']==before
    assert migrated['global_step']==1000
    for key in ('state_dict','optimizer_states','lr_schedulers'):
        assert migrated[key] is checkpoint[key]
    result=migrated['loops']['fit_loop']['epoch_loop.batch_progress']['current']
    assert result['processed']==expected
    assert result['completed']==expected-1
    assert receipt['effective_batch']==32
