import pytest
from reproduction.training_costs import summarize_run


def row(history, step, timestamp, runtime, epoch=0):
    return dict(history_step=history, optimizer_step=step, timestamp=timestamp, runtime=runtime, epoch=epoch)


def test_resume_replaces_discarded_steps_and_excludes_downtime():
    rows = [row(0,49,100,0), row(1,99,150,50), row(2,149,200,100),
            row(3,99,1000,110), row(4,149,1025,135), row(5,199,1050,160)]
    rows.insert(2, rows[1].copy())
    result = summarize_run(rows, gpu_count=2, effective_batch=32)
    assert result["discarded_events"] == 2
    assert result["duplicate_api_events"] == 1
    assert result["train_gpu_hours"] == pytest.approx(50 * 2 / 3600)
    assert result["samples_per_second"] == 64


def test_validation_boundary_is_excluded_from_throughput_but_included_in_cost():
    rows = [row(0,49,100,0), row(1,99,125,25),
            row(2,149,1125,1025,epoch=1), row(3,199,1150,1050,epoch=1)]
    result = summarize_run(rows, gpu_count=1, effective_batch=32)
    assert result["train_gpu_hours"] == pytest.approx(1050 / 3600)
    assert result["samples_per_second"] == 64
    assert result["throughput_windows"] == 2


def test_conflicting_api_duplicates_are_rejected():
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        summarize_run([row(0,49,100,0),row(0,49,101,0)], 1, 32)


def test_missing_training_interval_is_rejected():
    with pytest.raises(ValueError, match="Missing or duplicated optimizer steps"):
        summarize_run([row(0,49,100,0),row(1,149,150,50),row(2,199,175,75)], 1, 32)
