# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""Split-KV planning for the SM90 quantized-KV decode kernel.

At plan time this module decides how many thread blocks one request's KV is
spread over, and how many blocks per SM the kernel is built for.

THE COST MODEL

Each candidate split count ``o`` is scored by an occupancy model fitted per head
dimension:

    cost(o) = T[bits] * ceil(nt / o) * max(1, base * o / sm_count)
              + P + K * o + (combine if o > 1 else 0)

``nt`` is the request's tile count and ``base = batch * num_kv_heads`` is the
unsplit grid. ``T`` is the per-tile stream time, ``K`` the cost of one more split
(its partial output buffer and its share of the merge), ``P`` the fixed launch
cost, and ``combine`` the cost of merging the partials.

The saturation term ``max(1, base * o / sm_count)`` is what separates this model
from a plain wave estimate. Past one CTA per SM, extra CTAs are co-resident and
share the SM, so halving the tiles per CTA stops halving the makespan. The term
is deliberately not capped at one wave, because the cap removes exactly the
slowdown it exists to express.

Every candidate is floored at ``min_chunk_tokens``, because a split has to reach
pipeline steady state before it pays for itself. The kernel runs TMA rings into
wgmma pipelines and strides two tiles per consumer warpgroup, so a split needs
about eight 64-token tiles.

THE ONE-WAVE RULE

When a head dimension has no fitted entry, or the bit width has no constant, the
planner falls back to FlashInfer's work estimate: the split budget is one
occupancy wave, ``max_grid = blocks_per_sm * sm_count``. If the unsplit grid
already fills it, the request is not split; otherwise the planner takes the
largest split count that still fits, floored at the minimum chunk. The kernel
ceil-partitions each request's own tile count and tolerates empty tail splits, so
any split count is legal.

``min_blocks_per_sm`` is 2 and is architectural: the kernel's 32/104
``setmaxnregs`` split is built in, so the planner passes it through rather than
searching it.
"""
import math

# Occupancy model constants, fitted per head dimension.
#
# ``T_us`` is physical: the median per-tile stream time from unsplit cells at
# B >= 32, where the model reduces to ``T * nt * base / sm``. ``K_us`` is fitted.
# Only the ratios of T, K and combine rank candidates, so the zero P and combine
# values in the 256 entry mean the fit did not need them, not that those costs
# are absent.
#
# A head dimension with no entry plans by the one-wave rule. No entry is copied
# from another head dimension or added as a placeholder, because a placeholder
# would replace the baseline a first fit of that head dimension is compared
# against.
PINGPONG_MODEL = {
    128: {
        "T_us": {2: 0.358, 4: 0.420},
        "P_us": 5.867, "combine_us": 10.315, "K_us": 0.08,
    },
    256: {
        "T_us": {2: 0.85, 4: 0.15},
        "P_us": 0.0, "combine_us": 0.0, "K_us": 0.05,
    },
}

_OUTER_CANDIDATES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def _plan_pingpong_model(base, kv_len, nt, bits, sm_count, min_chunk_tokens,
                         legacy, *, head_dim):
    """(outer, min_blocks) from the occupancy model.

    Returns ``legacy`` unchanged when the model has no entry for this head
    dimension or for this bit width, so an unfitted configuration plans exactly
    as it did before the model existed.
    """
    entry = PINGPONG_MODEL.get(head_dim)
    if entry is None:                  # no fit at this head dimension: do not guess
        return legacy
    t = entry["T_us"].get(bits)
    if t is None:                      # unknown bit width: do not guess
        return legacy
    best = None
    for o in sorted(set(_OUTER_CANDIDATES) | {legacy[0]}):
        if o > nt or (o > 1 and kv_len // o < min_chunk_tokens):
            continue
        c = (t * math.ceil(nt / o) * max(1.0, base * o / sm_count)
             + entry["P_us"]
             + entry["K_us"] * o
             + (entry["combine_us"] if o > 1 else 0.0))
        if best is None or c < best[0] - 1e-9:
            best = (c, o)
    if best is None:
        return legacy
    # min_blocks is architectural on this kernel (the 32/104 setmaxnregs split is
    # built in), so it is passed through rather than searched.
    return best[1], legacy[1]


def plan_fused_combine(batch, num_kv_heads, outer_splits):
    """Whether to run the in-kernel fused split-K combine instead of the external
    SplitKCombine launch.

    The fused tail folds the merge into the last CTA. It pays when the unsplit
    grid is small, ``batch * num_kv_heads <= 16``, where the external launch
    cannot hide its latency. Above that the grid fills a wave, every CTA pays the
    release fence and atomic scaffolding while the per-group merge shrinks, and
    the external launch overlaps the merge across B x HQ CTAs anyway.
    """
    return outer_splits > 1 and batch * num_kv_heads <= 16


def plan_splits(batch, num_kv_heads, kv_len, *, bits, head_dim, page_size=64,
                sm_count=None, min_chunk_tokens=512):
    """Plan (outer_splits, min_blocks_per_sm) for one decode launch.

    ``kv_len`` is the sequence length in tokens, either an int for a uniform
    batch or a sequence of per-request lengths, in which case the plan follows
    the longest request. ``head_dim`` is required because the model is fitted per
    head dimension. The returned ``min_blocks_per_sm`` is the value the kernel
    constructor accepts, which is always 2.
    """
    if sm_count is None:
        import torch
        sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if not isinstance(kv_len, int):
        kv_len = max(int(l) for l in kv_len)
    nt = max(1, math.ceil(kv_len / page_size))
    base = batch * num_kv_heads

    minb = 2                           # architectural: the 32/104 setmaxnregs split is built in
    capacity = minb * sm_count         # one occupancy wave (FlashInfer's max_grid_size)

    # The one-wave rule. Its answer is kept even where the occupancy model
    # overrides it, because the model takes it as one candidate and falls back to
    # it for an unfitted head dimension or bit width.
    outer = 1
    if base < capacity:
        outer = max(1, capacity // base)
        outer = min(outer, nt, max(1, kv_len // min_chunk_tokens))
    return _plan_pingpong_model(base, kv_len, nt, bits, sm_count,
                                min_chunk_tokens, legacy=(outer, minb),
                                head_dim=head_dim)


if __name__ == "__main__":
    # self-test against the measured H100 optima (sm_count pinned to 132)
    cases = [
        # (batch, hkv, kv, bits) -> (outer, min_blocks)
        ((8, 8, 4096, 2), (4, 2)),
        ((8, 8, 16384, 2), (4, 2)),
        ((8, 8, 8192, 4), (4, 2)),
        ((32, 8, 8192, 2), (1, 2)),
        ((64, 8, 4096, 2), (1, 2)),
        ((128, 8, 16384, 2), (1, 2)),
        # bs=1 latency regime: the chunk floor binds
        ((1, 8, 4096, 2), (8, 2)),                 # 4096/512 -> 8
        # B=1 long KV: the cells the occupancy model exists to fix. The one-wave
        # rule returns 32 and 33 (= 264//8) here; the model returns 16, because
        # 128 CTAs is about the 132 SMs.
        ((1, 8, 16384, 2), (16, 2)),
        ((1, 8, 32768, 2), (16, 2)),
        # floors: a tiny KV caps the split count
        ((1, 8, 256, 2), (1, 2)),                  # 256 tokens < 512 floor: no split
        ((1, 8, 64, 2), (1, 2)),                   # single tile: no split
    ]
    for (b, h, kv, bits), want in cases:
        # Every case is a D=128 optimum, so the cases plan at head_dim=128.
        got = plan_splits(b, h, kv, bits=bits, head_dim=128, sm_count=132)
        assert got == want, f"{(b, h, kv, bits)}: got {got}, want {want}"
    print(f"scheduler self-test: {len(cases)} cases OK")
