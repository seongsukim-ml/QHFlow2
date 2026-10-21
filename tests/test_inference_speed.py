import pytest
from reproduction.inference_speed import summarize_sweeps


def test_rate_uses_real_samples_and_median_sweep_not_mean_batch_rates():
    r = summarize_sweeps([8.0, 4.0, 2.0], 32)
    assert r['samples_per_second'] == 8
    assert r['milliseconds_per_sample'] == 125
    assert r['min_samples_per_second'] == 4
    assert r['max_samples_per_second'] == 16


@pytest.mark.parametrize('times', [[], [0], [-1], [float('nan')], [float('inf')]])
def test_rejects_invalid_timings(times):
    with pytest.raises(ValueError):
        summarize_sweeps(times, 32)
