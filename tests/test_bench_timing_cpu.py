"""The benchmark timer's attribution and summary rules, on synthetic records.

These are the two places a timing bug would go unnoticed: a device record that
the profiler dropped, and a partial iteration entering the median. Both are pure
functions, so they are tested here with no GPU.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

import timer  # noqa: E402


def capture(markers=2, iters=2, per_iteration=3, drop=()):
    """A synthetic capture: two guard trains around `iters` iterations."""
    host, device = [], []
    launches = []
    corr = 1
    for _ in range(markers):
        launches.append((corr, timer.GUARD_KERNEL_NAME, 1.0))
        corr += 1
    for iteration in range(iters):
        for position in range(per_iteration):
            if position == 0:
                key = timer.FLUSH_KERNEL_NAME
            else:
                key = "decode_kernel"
            launches.append((corr, key, float(10 * iteration + position)))
            corr += 1
    for _ in range(markers):
        launches.append((corr, timer.GUARD_KERNEL_NAME, 1.0))
        corr += 1

    for index, (cid, key, us) in enumerate(launches):
        host.append((cid, float(index), "cudaLaunchKernel"))
        if cid in drop:
            continue
        device.append((cid, key, us))
    return host, device


def test_min_observed_is_one_drop_or_a_tenth():
    assert timer.min_observed(5) == 4
    assert timer.min_observed(20) == 18
    assert timer.min_observed(40) == 36


def test_each_iteration_sums_its_own_kernels():
    host, device = capture()
    buckets, lead, tail = timer.attribute_launches(host, device, markers=2, iters=2)
    assert (lead, tail) == (2, 2)
    assert [b["n_kernels"] for b in buckets] == [2, 2]
    assert [b["kernel_us"] for b in buckets] == [3.0, 23.0]
    assert [b["flush_us"] for b in buckets] == [0.0, 10.0]


def test_a_lost_record_empties_only_its_own_iteration():
    # Correlation id 4 is the second decode kernel of the first iteration.
    host, device = capture(drop=(4,))
    buckets, _lead, _tail = timer.attribute_launches(host, device, markers=2, iters=2)
    assert buckets[0]["n_kernels"] == 0
    assert buckets[1]["kernel_us"] == 23.0


def test_a_partial_iteration_stays_out_of_the_median():
    whole = [{"kernel_us": us, "n_kernels": 2, "flush_us": 0.0}
             for us in (30.0, 31.0, 32.0, 33.0)]
    partial = {"kernel_us": 15.0, "n_kernels": 1, "flush_us": 0.0}
    median, n_kernels, _flush, observed = timer.summarize(whole + [partial], iters=5)
    assert n_kernels == 2
    assert observed == 4
    assert median == pytest.approx(31.5)


def test_an_empty_capture_is_refused():
    with pytest.raises(RuntimeError, match="observed whole"):
        timer.summarize([{"kernel_us": 0.0, "n_kernels": 0, "flush_us": 0.0}] * 5,
                        iters=5)


def test_too_few_whole_iterations_are_refused():
    buckets = [{"kernel_us": 10.0, "n_kernels": 2, "flush_us": 0.0}] * 2
    buckets += [{"kernel_us": 0.0, "n_kernels": 0, "flush_us": 0.0}] * 3
    with pytest.raises(RuntimeError, match="observed whole"):
        timer.summarize(buckets, iters=5)
