"""GPU kernel time for one decode call, from CUPTI activity records.

A call's latency is the device time of every kernel it launches, summed over one
call, taken as the median over the calls whose records the profiler delivered
whole. Host dispatch is outside it, and so is stream idle, because CUPTI reports
each kernel's own duration rather than the span of a stream between two events.

Every call is timed against a cold L2. The loop launches a 1 GiB write before
each call: that reallocates every L2 set, so the cache the call sees is the one
it would see on a cache miss. The flush is excluded from the total by exact
name, and the name is ours because the kernel is compiled here through NVRTC.

Nothing here times a second stream correctly. A call that enqueues work on a
stream of its own is still charged for it, but that work may overlap the next
call's flush and so may not have run against a cold L2. A capture cannot show
whether it did.

    from timer import measure
    m = measure(lambda: run_once(...))
    m.median_us
"""

from __future__ import annotations

import collections
import dataclasses
import statistics
import typing

#: Calls timed per measurement, and calls run first to warm the code paths.
#: These are the protocol's values; a number measured at others is not
#: comparable with a number measured at these.
ITERS = 5
WARMUP = 5

#: The cold-L2 flush. 1 GiB reallocates every set of an 80 GB or 180 GB card's
#: L2 with a wide margin, so one size serves both architectures.
FLUSH_KERNEL_NAME = "foscar_l2_flush"
FLUSH_KERNEL_SOURCE = r"""
extern "C" __global__ void foscar_l2_flush(unsigned int* buf,
                                           unsigned int val,
                                           long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) buf[i] = val;
}
"""
FLUSH_BYTES = 1 << 30

#: Guards bracket every capture. The profiler can drop the device records of the
#: kernels at either end of a capture while keeping their launch records, and a
#: guard that loses its record costs nothing, where a decode that lost one would
#: have to be discarded.
GUARD_KERNEL_NAME = "foscar_capture_guard"
GUARD_KERNEL_SOURCE = r"""
extern "C" __global__ void foscar_capture_guard(int cycles) {
    long long start = clock64();
    while (clock64() - start < (long long)cycles) { }
}
"""
GUARD_MARKER_CYCLES = 250_000
GUARD_MIN_MARKERS = 8
GUARD_MAX_MARKERS = 4096


@dataclasses.dataclass(frozen=True)
class Measurement:
    """One timed call."""

    #: Median device time of one call, over the calls that were observed whole.
    median_us: float
    #: Kernels the call launched per iteration, as the modal count.
    n_kernels: int
    #: Median device time of the flush, published so a reader can see it ran.
    flush_us: float
    #: Iterations requested, and how many of them entered the median.
    iters: int
    iters_observed: int
    #: The sample `median_us` is the median of.
    samples_us: tuple


def min_observed(iters: int) -> int:
    """How many whole iterations a measurement needs before it may publish.

    One dropped iteration, or a tenth of them, whichever is the more forgiving:
    at five iterations that is four, at forty it is thirty-six.
    """
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    return max(1, iters - max(1, -(-iters // 10)))


def is_flush_key(key: str) -> bool:
    return key == FLUSH_KERNEL_NAME


def issues_device_work(key: str) -> bool:
    """True for a host record of a call that puts work on the device.

    Kernel launches, memsets and memcpys each leave one device record, so a call
    whose device record is missing is a loss the capture can see. A call of any
    other name is counted only when its device record arrives; teach this
    predicate a call before timing something that issues it.
    """
    return ("Launch" in key and "Kernel" in key) or "Memset" in key or "Memcpy" in key


def _self_device_us(event) -> float:
    total = getattr(event, "self_device_time_total", None)
    if total is None:  # torch < 2.5
        total = getattr(event, "self_cuda_time_total", 0.0)
    return float(total or 0.0)


def attribute_launches(host, device, *, markers: int, iters: int):
    """Group one capture's device records into per-call buckets. Pure.

    `host` holds `(correlation_id, host_start_us, key)` for every host record of
    the capture and `device` holds `(correlation_id, key, device_us)` for every
    device record. A launch and the work it launched share a correlation id, so
    a record belongs to the iteration whose launch made it however late it
    starts. A launch that lost its device record therefore empties its own
    iteration and can never move a duration into another one.

    The capture is refused when a device record has no launch record, when two
    share one launch, when the launch count is not two guard trains around a
    whole number of equal iterations, or when a record's name does not fit its
    position. In each case the iteration a duration belongs to would be a guess.

    Returns `(buckets, lead_seen, tail_seen)`; an iteration that lost a record
    is passed with `n_kernels` 0 so it cannot enter a median.
    """
    records = {}
    for corr, key, device_us in device:
        if corr in records:
            raise RuntimeError(
                f"two device records ({records[corr][0][:60]!r} and {key[:60]!r}) share "
                f"one launch record, so neither belongs to a known iteration")
        records[corr] = (key, float(device_us))
    issued = {corr for corr, _start, _key in host}
    orphans = [key for corr, (key, _us) in records.items() if corr not in issued]
    if orphans:
        raise RuntimeError(
            f"{len(orphans)} device record(s) of this capture have no launch record "
            f"(first {orphans[0][:60]!r}), so their iteration cannot be determined")

    # A launch is a correlation id rather than a host record: the profiler can
    # file a record of its own under a launch's id, so each id counts once, at
    # the earliest host record carrying it.
    issued_at: dict = {}
    for corr, start, key in host:
        if corr in records or issues_device_work(key):
            issued_at[corr] = min(start, issued_at.get(corr, start))
    launches = sorted(issued_at, key=issued_at.get)

    measured = len(launches) - 2 * markers
    if measured <= 0 or measured % iters:
        raise RuntimeError(
            f"the capture holds {len(launches)} launches, which is not two trains of "
            f"{markers} guards around {iters} iterations of one launch count; a record "
            f"was lost where its launch cannot be identified, or the call launched a "
            f"different number of kernels per call")
    per_iteration = measured // iters
    if per_iteration < 2:
        raise RuntimeError(
            "the call launched nothing measurable: every iteration holds only the "
            "L2 flush")

    def seen(position: int, role: str):
        record = records.get(launches[position])
        if record is None:
            return None
        key = record[0]
        if role == "guard":
            fits = key == GUARD_KERNEL_NAME
        elif role == "flush":
            fits = is_flush_key(key)
        else:
            fits = key != GUARD_KERNEL_NAME and not is_flush_key(key)
        if not fits:
            raise RuntimeError(
                f"launch {position} of the capture is a {role} position and its device "
                f"record is {key[:60]!r}; the capture is not the sequence this module "
                f"issued, so no duration can be attributed")
        return record

    tail = markers + measured
    lead_seen = sum(seen(i, "guard") is not None for i in range(markers))
    tail_seen = sum(seen(i, "guard") is not None for i in range(tail, len(launches)))
    buckets = []
    for iteration in range(iters):
        first = markers + iteration * per_iteration
        flush = seen(first, "flush")
        decode = [seen(first + j, "decode") for j in range(1, per_iteration)]
        if flush is None or None in decode:
            buckets.append({"kernel_us": 0.0, "n_kernels": 0, "flush_us": 0.0})
        else:
            buckets.append({"kernel_us": sum(us for _key, us in decode),
                            "n_kernels": len(decode), "flush_us": flush[1]})
    return buckets, lead_seen, tail_seen


def modal_kernel_count(buckets) -> int:
    """The call's kernel count, as the most frequent value.

    Ties break toward the larger count: a lost kernel can only make a bucket
    short, so of two equally frequent counts the larger is the one loss cannot
    explain.
    """
    counts: dict = {}
    for bucket in buckets:
        counts[int(bucket["n_kernels"])] = counts.get(int(bucket["n_kernels"]), 0) + 1
    return max(counts, key=lambda n: (counts[n], n))


def summarize(buckets, *, iters: int) -> tuple:
    """`(median_us, n_kernels, flush_us, iters_observed)` from per-call buckets.

    Only whole iterations enter the median. A bucket that lost a record reads 0,
    and a bucket short one kernel of several is biased downward, so both are
    dropped rather than averaged in.
    """
    observed = [b for b in buckets if b["n_kernels"] > 0]
    if not observed:
        raise RuntimeError(
            f"no iteration of this capture was observed whole: each of {len(buckets)} "
            f"lost a device record or launched nothing measurable")
    modal = modal_kernel_count(observed)
    complete = [b for b in observed if int(b["n_kernels"]) == modal]
    if len(complete) < min_observed(iters):
        histogram = sorted(collections.Counter(int(b["n_kernels"])
                                               for b in observed).items())
        raise RuntimeError(
            f"only {len(complete)} of {iters} iterations were observed whole at "
            f"{modal} kernel(s), against {len(observed)} that held a kernel "
            f"(kernels per iteration: {histogram}); too few to take a median over")
    return (statistics.median([b["kernel_us"] for b in complete]),
            modal,
            statistics.median([b["flush_us"] for b in complete]),
            len(complete))


def _compile(source: str, name: str):
    # torch is imported at the point of use so the attribution and summary rules
    # above stay testable on a machine with no CUDA build of torch installed.
    from torch.cuda import _compile_kernel

    return _compile_kernel(source, name)


class _Flusher:
    """The 1 GiB eviction buffer, driven by the kernel compiled above."""

    def __init__(self) -> None:
        import torch

        self.words = FLUSH_BYTES // 4
        self._buf = torch.empty(self.words, dtype=torch.int32, device="cuda")
        self._kernel = _compile(FLUSH_KERNEL_SOURCE, FLUSH_KERNEL_NAME)
        self._threads = 256
        self._blocks = max(1, min(65535,
                                  (self.words + self._threads - 1) // self._threads))
        # A fresh allocation has untouched pages; pay the mapping cost here
        # rather than on whichever iterations happen to be timed first.
        for _ in range(8):
            self.flush()
        torch.cuda.synchronize()

    def flush(self) -> None:
        self._kernel(grid=(self._blocks, 1, 1), block=(self._threads, 1, 1),
                     args=[self._buf, 1, self.words])


_flusher: typing.Optional[_Flusher] = None
_guard = None
_guard_markers = GUARD_MIN_MARKERS


def _get_flusher() -> _Flusher:
    global _flusher
    if _flusher is None:
        _flusher = _Flusher()
    return _flusher


def _guard_train(kernel, markers: int) -> None:
    for _ in range(markers):
        kernel(grid=(1, 1, 1), block=(1, 1, 1), args=[GUARD_MARKER_CYCLES])


def _next_markers(markers: int, lead_seen: int, tail_seen: int) -> int:
    worst = min(lead_seen, tail_seen)
    if worst == 0:
        markers *= 4
    elif 2 * worst < markers:
        markers *= 2
    return min(markers, GUARD_MAX_MARKERS)


def measure(fn, *, iters: int = ITERS, warmup: int = WARMUP) -> Measurement:
    """Time one call of `fn` as GPU kernel time, always against a cold L2."""
    global _guard, _guard_markers
    if iters <= 0 or warmup < 0:
        raise ValueError(f"bad iters/warmup: {iters}/{warmup}")

    import torch
    from torch.profiler import ProfilerActivity, profile

    flusher = _get_flusher()

    def body() -> None:
        flusher.flush()
        fn()

    # Warm up on the exact timed sequence, flush included: warming with `fn`
    # alone leaves the flush buffer cold, so the first timed calls would pay
    # first-touch cost.
    for _ in range(warmup):
        body()
    torch.cuda.synchronize()

    if _guard is None:
        _guard = _compile(GUARD_KERNEL_SOURCE, GUARD_KERNEL_NAME)
    markers = _guard_markers
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _guard_train(_guard, markers)
        for _ in range(iters):
            body()
        _guard_train(_guard, markers)
        torch.cuda.synchronize()

    host, device = [], []
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CPU:
            host.append((event.id, float(event.time_range.start), event.key))
        else:
            device.append((event.id, event.key, _self_device_us(event)))

    buckets, lead_seen, tail_seen = attribute_launches(
        host, device, markers=markers, iters=iters)
    _guard_markers = _next_markers(markers, lead_seen, tail_seen)
    median_us, n_kernels, flush_us, observed = summarize(buckets, iters=iters)
    samples = tuple(float(b["kernel_us"]) for b in buckets
                    if int(b["n_kernels"]) == n_kernels)
    return Measurement(median_us=median_us, n_kernels=n_kernels, flush_us=flush_us,
                       iters=iters, iters_observed=observed, samples_us=samples)
