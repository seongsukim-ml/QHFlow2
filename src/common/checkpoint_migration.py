"""Translate Lightning minibatch counters when effective batch is unchanged."""
import copy
import math


def adapt_microbatch_progress(checkpoint, microbatch):
    """Preserve weights, optimizer, scheduler and global step; scale batch units.

    Lightning recovery saves during on_train_batch_end, before the completed
    counter catches up to processed. Keep that difference for restart logic.
    A shuffled non-stateful loader cannot reproduce the original sample order.
    """
    conf = checkpoint['hyper_parameters']['conf']
    old = int(conf.dataset.batch_size)
    old_accum = int(conf.accumulate_grad_batches)
    effective = old * old_accum
    if microbatch <= 0 or effective % microbatch:
        raise ValueError('New microbatch must divide the original effective batch')
    receipt = {'changed':old != microbatch, 'old_microbatch':old,
               'new_microbatch':microbatch, 'effective_batch':effective,
               'optimizer_step':int(checkpoint['global_step'])}
    if old == microbatch:
        return checkpoint, receipt
    migrated = dict(checkpoint)
    migrated['loops'] = copy.deepcopy(checkpoint['loops'])
    progress = migrated['loops']['fit_loop']['epoch_loop.batch_progress']
    before = copy.deepcopy(progress)
    for scope in ('total', 'current'):
        counter = progress[scope]
        previous_processed = counter['processed']
        processed = math.ceil(previous_processed * old / microbatch)
        for name in counter:
            if name == 'completed':
                counter[name] = max(0, processed - (previous_processed - counter[name]))
            else:
                counter[name] = math.ceil(counter[name] * old / microbatch)
    receipt.update(before=before, after=copy.deepcopy(progress))
    return migrated, receipt
