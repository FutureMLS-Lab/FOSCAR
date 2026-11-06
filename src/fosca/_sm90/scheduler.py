# Copyright (c) 2026 Donglin Zhuang
# SPDX-License-Identifier: MIT
"""Split-KV planning for the SM90 quantized-KV decode kernel (``IntPingPongDecodeSm90``):
pick the CTA partition along the sequence dimension at plan time, FlashInfer-style.

The pick comes from ``PINGPONG_MODEL``, an occupancy model. The legacy one-wave rule
below is still computed, but only as one candidate of that model and as its answer for
a bit width the model has no constant for. The calibrations and the validation scores
are recorded beside the constants below.

THE LEGACY ONE-WAVE RULE

A port of FlashInfer's decode work estimation
(``BatchDecodeWithPagedKVCacheWorkEstimationDispatched``): the split budget is ONE
occupancy wave, max_grid = CTAs/SM x SM count. If the unsplit grid
(batch x num_kv_heads) already fills it, do not split; otherwise use the largest
split count that still fits inside the wave, floored at a minimum chunk per split.
FlashInfer reaches the same point by binary-searching the minimal per-request chunk
size; with a single uniform ``n_outer`` kernel argument the closed form below is
equivalent.

The kernel ceil-partitions each request's own tile count and tolerates empty tail
splits, so any split count is legal.

ONE DELIBERATE DEVIATION FROM FLASHINFER: the minimum chunk is 512 tokens, not 128.
FlashInfer's decode CTA is a shallow streaming loop; this one runs TMA rings feeding
wgmma pipelines (at D=128 4 stages deep at INT2 and 2 at INT4, at D=256 2 at both) and
strides 2 tiles per consumer WG,
so a split needs about 8 x 64-token tiles to reach steady state. The occupancy model applies this floor to
every candidate, and it is the reason the derivation above is kept.

WHY PINGPONG NEEDS A DIFFERENT SHAPE

The wave form's per-tile term is ``ceil(base*o/(minb*sm)) * T*ceil(nt/o)``, which at
B=1 is ONE WAVE for every o the planner can reach and is therefore strictly
decreasing in o. It ranks the largest legal split best where the measured optimum is
o=16, and NO REFIT CAN CHANGE THAT: searched exhaustively, its floor is 4.75% max
regret, which is a property of the FORM rather than of its constants. On the H100
sweep that form costs 11-14% at B=1 and kv >= 16k.

The measured fact the wave form cannot express: at B=1 the int2 optimum is o=16 at
KV 8k, 16k AND 32k, CONSTANT in KV. (int4 at KV=32768 is a tie inside the
measurement's own reproducibility band, s32 and s16 measuring 36.86 and 37.38us, so
it neither supports nor contradicts this.) A second H100 reading has int2 s16 and s32
within 1.2% at KV=32768, and an int4 KV=32768 optimum of s24, 3.1% under s16.
Any model balancing a decreasing ``T*tiles/o`` against an increasing ``K*o`` has its
optimum at ``o ~ sqrt(T*nt/K)``, which cannot be constant. What makes it constant is SATURATION at about 1 CTA/SM: past that, extra
CTAs are co-resident and timeshare, so halving the tiles per CTA stops halving the
makespan. ``max(1, base*o/sm_count)`` is that term, and it is deliberately NOT capped
at ``minb*sm``, because the cap discards exactly the slowdown it exists to model.

Every B>=8 pick is the searched optimum under both rules, so the model changes only
the bs=1 latency regime.
"""
import math

# --- PINGPONG occupancy model, fit on measured regret ------------------------
#
#   cost(o) = T[bits] * ceil(nt/o) * max(1, base*o/sm_count)
#             + P + K*o + (COMBINE if o > 1 else 0)
#
# THE SHAPE IS WHAT MATTERS HERE, NOT THE CONSTANTS; the module docstring gives
# the argument for why the wave form cannot be refit into this one.
#
# `max(1, base*o/sm_count)` is the saturation term, and it is deliberately NOT
# capped at minb*sm: capping it scores 8.57%, because the cap discards exactly
# the slowdown the term exists to model.
#
# T IS PHYSICAL, K IS FIT, both on the same H100 sweep. T is the median implied per-tile
# stream time from its nine unsplit B>=32 cells per bit width, where the model reduces to
# `T*nt*base/sm`. K is the per-split cost, and o_partial is [o,B,HQ,D] f32, so
# partial traffic and the combine's reduction depth both scale with o; on that
# sweep max regret is <= 1.5% for ANY K in 0.03-0.17, so 0.08 is mid-range rather
# than a knife edge. P and COMBINE barely affect ranking, so they are inherited
# unchanged from the D=128 calibration ladder.
#
# THE MODEL IS KEYED BY HEAD DIMENSION, the whole entry and not only T, because K
# prices the [o,B,HQ,D] partial output and the combine, both of which grow with D. Each
# entry is fitted on its own head dimension's sweep. The 128 entry is the D=128 fit above.
# The 256 entry comes from a fit on a D=256 ladder alone. Unlike the 128 entry, its
# constants are not physical: only the ratios of T, K and COMBINE rank the candidates, and
# the fit's optimum sets P and COMBINE to zero. No second ladder checks it out of sample.
# A head dimension with no entry plans by the legacy one-wave rule; no entry is copied from
# another head dimension or added as a placeholder, because that would replace the legacy
# planning a first fit of that head dimension is compared against.
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
    """(outer, minb) for pingpong from the occupancy model. See PINGPONG_MODEL.

    NO legacy-anchored acceptance gate and NO latency-regime bypass: either would keep
    legacy's pick at B=1, where it is up to about 14% off at KV=16384.

    A head dimension without an entry in PINGPONG_MODEL returns `legacy`, as an
    unknown bit width does.
    """
    entry = PINGPONG_MODEL.get(head_dim)
    if entry is None:                  # with no fit at this head dimension, do not guess
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
    # minb is ARCHITECTURAL on pingpong (the 32/104 setmaxnregs split is built
    # into the kernel), so it is passed through rather than searched.
    return best[1], legacy[1]


def plan_fused_combine(batch, num_kv_heads, outer_splits, *, head_tiles=1):
    """Whether IntPingPongDecodeSm90 should run its in-kernel fused split-K combine
    (v2 tree form) instead of the external SplitKCombine launch.

    Measured on H100 across int2/int4 and kv 4k-32k: the fused tail wins the latency
    regime, where base = batch x num_kv_heads <= 16 saves 4-8% end to end by folding the
    combine launch into the last CTA. It is a wash at base=32, and loses 2-3% at
    base >= 64 where the grid fills a wave, because every CTA then pays the release-fence
    and atomic scaffolding while the per-group merge shrinks, and the external launch
    hides its latency across B x HQ parallel CTAs anyway."""
    if isinstance(head_tiles, bool) or not isinstance(head_tiles, int) or head_tiles < 1:
        raise ValueError("head_tiles must be a positive integer")
    return outer_splits > 1 and batch * num_kv_heads * head_tiles <= 16


def plan_splits(batch, num_kv_heads, kv_len, *, bits, head_dim, page_size=64,
                sm_count=None, min_chunk_tokens=512, head_tiles=1):
    """Plan (outer_splits, min_blocks_per_sm) for one decode launch.

    kv_len is the sequence length in tokens, either an int for a uniform batch or a
    sequence of per-request lengths, in which case the plan follows the longest request,
    as FlashInfer's binary search does. head_dim is the head dimension planned for, and it
    is required because the model is fitted per head dimension (PINGPONG_MODEL). The
    returned min_blocks_per_sm is resolved to the value the kernel constructor accepts,
    which is always 2.
    """
    if isinstance(head_tiles, bool) or not isinstance(head_tiles, int) or head_tiles < 1:
        raise ValueError("head_tiles must be a positive integer")
    if sm_count is None:
        import torch
        sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if not isinstance(kv_len, int):
        kv_len = max(int(l) for l in kv_len)
    nt = max(1, math.ceil(kv_len / page_size))
    base = batch * num_kv_heads * head_tiles

    minb = 2                           # architectural: the 32/104 setmaxnregs split is built in
    capacity = minb * sm_count         # one occupancy wave (FlashInfer's max_grid_size)

    outer = 1
    if base < capacity:
        outer = max(1, capacity // base)
        outer = min(outer, nt, max(1, kv_len // min_chunk_tokens))
    # This is the occupancy model, fit against a MEASURED 12-point ladder at every
    # shape (see PINGPONG_MODEL), and it supersedes the legacy one-wave rule outright.
    # `outer` above is still computed, because the model takes it as a candidate and
    # falls back to it for an unknown bit width or for a head dimension with no model
    # entry. Where the model has an entry, `outer` is only a candidate.
    return _plan_pingpong_model(base, kv_len, nt, bits, sm_count,
                                min_chunk_tokens, legacy=(outer, minb),
                                head_dim=head_dim)


if __name__ == "__main__":
    # self-test against the H100 ladder optima (sm_count pinned to 132)
    cases = [
        # (batch, hkv, kv, bits) -> (outer, minb)
        ((8, 8, 4096, 2), (4, 2)),
        ((8, 8, 16384, 2), (4, 2)),
        ((8, 8, 8192, 4), (4, 2)),
        ((32, 8, 8192, 2), (1, 2)),
        ((64, 8, 4096, 2), (1, 2)),
        ((128, 8, 16384, 2), (1, 2)),
        # bs=1 latency regime (ladder optima within ~5%: chunk floor binds)
        ((1, 8, 4096, 2), (8, 2)),                 # 4096/512 -> 8
        # B=1 LONG-KV: the two cells the occupancy model is fitted for.
        # The legacy wave rule returns 32 and 33 (= 264//8) here, and the model
        # returns 16 because 128 CTAs is about the 132 SMs.
        ((1, 8, 16384, 2), (16, 2)),
        ((1, 8, 32768, 2), (16, 2)),
        # floors: tiny kv caps the split count
        ((1, 8, 256, 2), (1, 2)),                  # 256 tokens < 512 floor -> no split
        ((1, 8, 64, 2), (1, 2)),                   # single tile -> no split
    ]
    for (b, h, kv, bits), want in cases:
        # Every case is a D=128 ladder optimum, so the cases plan at head_dim=128.
        got = plan_splits(b, h, kv, bits=bits, head_dim=128, sm_count=132)
        assert got == want, f"{(b, h, kv, bits)}: got {got}, want {want}"
    print(f"scheduler self-test: {len(cases)} cases OK")
