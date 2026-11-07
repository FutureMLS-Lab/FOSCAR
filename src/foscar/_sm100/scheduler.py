# Copyright (c) 2026 Donglin Zhuang
# SPDX-License-Identifier: MIT
"""Split-KV planner for the SM100 quantized-KV fold decode kernel.

Picks ``(kv_splits, reduction_mode)`` for ``IntDecodeSm100`` at plan time from a
closed-form cost model. Wave-fill alone is NOT sufficient on B200: splits also win
at multi-wave grids, because the makespan is CEIL-QUANTIZED in waves.

    cost(s) = ceil(base*s / SM) * (T*ceil(tiles/s) + P) + (s>1 ? COMBINE : 0) + C0

      base   = batch * num_kv_heads * head_tiles  (independent head-tile items)
      tiles  = ceil(kv_len / sequence_tile)  (sequence_tile = 256)

    BOTH ceilings are required and neither is cosmetic. The inner
    ``ceil(tiles/s)`` is there because the makespan of a split is set by the split
    that got the MOST tiles, and the kernel ceil-partitions; writing it ``tiles/s``
    makes the model wrong for exactly the candidates that do not divide the tile
    count, which is what separates s=24 and s=48 from s=16, 32 and 64.
      T      = per-tile stream time, per bits (measured; see below)
      P      = fixed per-CTA prologue (measured)
      COMBINE= split-K reduction cost (kernel-mode launch + run)
      C0     = launch floor, constant across s, kept for readability

    pick   = argmin over s in {1,2,4,8,16,24,32,48,64} s.t. kv_len/s >= 512

The s == 1 arm gets P amortized by persistence (the C8 static round-robin, which is
the ctor default); persistence silently downgrades for any split-K reduction, so
split arms pay full P per CTA. The 512-token chunk floor is the same
pipeline-steady-state bound as SM90's: rings need about 2 tiles per split to fill,
and 2 x 256 = 512.

Every constant below is measured on B200.

THREE RULES A CHANGE HERE MUST RESPECT
--------------------------------------

1. **The T pair is a RATIO, and the ratio is what must be right.** ``T_US`` holds
   (unsplit, split) per bit width. Every pair that ranks the arms correctly has
   ``T_split / T_unsplit`` inside a narrow band around 1.0; the absolute level is
   nearly free, because a level error largely cancels in the argmin and a ratio
   error does not. DO NOT re-derive the pair from a single cell. Refit from
   medians over well-conditioned cells, taking the centroid of the
   correctly-ranking region. Re-run the fit whenever the kernel's per-tile cost
   changes.

2. **The candidate ladder deliberately runs past 1 CTA/SM.** Capping it hides the
   turnaround rather than preventing it: past one wave the ceil quantisation makes
   the work term invariant for split counts that DIVIDE the tile count, so those
   arms differ only by one prologue per extra wave, and non-divisors are charged
   strictly more. The measured ladder turns around at 1 CTA/SM on its own, so the
   cap is verified as not needed rather than merely removed.

3. **DO NOT PORT ``_sm90/scheduler.PINGPONG_MODEL``'s OCCUPANCY TERM HERE.** It is
   a 16.10% regression on B200 at the best K anywhere in [0, 4.0].
   ``max(1, base*s/sm_count)`` is CONTINUOUS, so past saturation the work term is
   scale-invariant in s, the arms tie at zero cost, and K*s is left as the only
   separator; no K satisfies both the cells that want it large and those that want
   it small. The two architectures differ in ONE constant and it is the whole
   story: SM90's wave boundary sits at 2x its saturation point, because it counts
   register-limited RESIDENCY, and it needed a continuous term to see what its own
   quantisation stepped over. Here the boundary lands ON the saturation point. Same
   form, opposite verdicts; copy neither the constants nor the shape.

The self-test below is the cheap always-on version of the held-out validation, pinned
to the measured optima so it fails loudly if these constants drift again.
Run: uv run python src/foscar/_sm100/scheduler.py
"""
import math

SM_COUNT = 148
SEQ_TILE = 256
P_US = 4.5
COMBINE_US = 4.0
C0_US = 11.0
#: (unsplit, split) per-tile stream time in us, per bit width. The RATIO rather
#: than the level is what has to be right, for the reason the module docstring
#: gives; refit the pair from medians over well-conditioned cells whenever the
#: kernel's per-tile cost moves.
#:
#: The int4 pair reflects the one-warpgroup dequant tuning that `BITS_TUNING` in
#: `fold_runtime` applies to every compile, because per-tile cost is exactly what
#: that tuning changes.
#:
#: On the calibration grid the pair ranks 28 of 30 cells. Two int2 cells miss,
#: B=1/KV=8192 at 4.04% and B=64/KV=16384 at 0.37%; both survive every candidate
#: pair, so they are a limit of the cost model's form rather than of its
#: calibration, and the first is the tightest margin on the grid at 1.6 us.
#:
#: Fit the pair from a timing column that holds kernel time only; a file whose
#: columns mix an event-bracket protocol with kernel time cannot calibrate it.
T_US = {2: (0.791, 0.795), 4: (1.25, 1.291)}
#: Deliberately widened past 1 CTA/SM. A ladder that stops at 16 never fills the
#: machine at B=1 (base=8 gives 128 CTAs on 148 SMs), so the wave term
#: `ceil(base*s/sm_count)` is pinned at 1 for every arm the planner can reach and
#: the cap, not the model, decides the answer. See the module docstring for why
#: the wider ladder is safe.
SPLIT_CANDIDATES = (1, 2, 4, 8, 16, 24, 32, 48, 64)
MIN_CHUNK_TOKENS = 512


def _cost(base, tiles, s, t_us, sm_count):
    # P is charged per wave for EVERY arm, including s == 1: the C8
    # persistent walk still pays an (amortized-but-nonzero) per-item
    # boundary cost, and the calibration of T at an s=1 cell folds the
    # residual in. This exact form reproduced the ladder optimum within 3%
    # on every swept int2 cell.
    #
    # `ceil(tiles / s)`, NOT `tiles / s`. The makespan of a split is set by the
    # split that got the MOST tiles, and the kernel ceil-partitions, so a split
    # count that does not divide the tile count lengthens nothing while still
    # adding a combine input. Exact division cannot see that. It agrees with ceil
    # only when every candidate divides the tile count, which holds for
    # tiles=32/64/128 against powers of two up to 16 and fails as soon as either
    # side is not a power of two: tiles=128 with s=24 is 5.33 tiles per split
    # under exact division and 6 in reality, and SPLIT_CANDIDATES carries 24 and
    # 48.
    #
    # SM90 measures what the exact form costs when the case is live: at 512
    # tiles, s=33 and s=32 both do ceil(512/33) == ceil(512/32) == 16 tiles per
    # split, so the 33rd CTA lengthens nothing and only adds a combine input --
    # an IDENTICAL critical path priced 7.6% apart. See
    # `_sm90/scheduler.PINGPONG_MODEL`, which ceils for this reason.
    waves = math.ceil(base * s / sm_count)
    t = t_us[0] if s == 1 else t_us[1]
    return (waves * (t * math.ceil(tiles / s) + P_US)
            + (COMBINE_US if s > 1 else 0.0) + C0_US)


def plan_splits_sm100(batch, num_kv_heads, kv_len, *, bits,
                      sm_count=SM_COUNT, sequence_tile=SEQ_TILE,
                      prefer_atomic=False, head_tiles=1):
    """Return ``(kv_splits, reduction_mode)`` for one decode launch.

    ``kv_len``: int or per-request iterable (the longest request governs,
    mirroring the SM90 planner)."""
    if isinstance(head_tiles, bool) or not isinstance(head_tiles, int) or head_tiles < 1:
        raise ValueError("head_tiles must be a positive integer")
    if not isinstance(kv_len, int):
        kv_len = max(int(l) for l in kv_len)
    base = batch * num_kv_heads * head_tiles
    tiles = max(1, math.ceil(kv_len / sequence_tile))
    t_us = T_US[bits]

    best_s, best_c = 1, None
    for s in SPLIT_CANDIDATES:
        if s > 1 and kv_len // s < MIN_CHUNK_TOKENS:
            continue
        if s > tiles:
            continue
        c = _cost(base, tiles, s, t_us, sm_count)
        if best_c is None or c < best_c - 1e-9:
            best_s, best_c = s, c
    if best_s == 1:
        return 1, "none"
    return best_s, ("atomic" if prefer_atomic else "kernel")


if __name__ == "__main__":
    # MEASURED_BEST holds the L2-flushed medians of an exhaustive
    # {1,2,4,8,16,24,32,48,64} search at every shape. The ground truth goes stale
    # when the kernel's per-tile cost moves, and a self-test pinned to stale
    # ground truth does not merely fail to catch that drift, it actively DEFENDS
    # it. Re-pin whenever per-tile cost moves.
    MEASURED_BEST = {  # (bits, batch, kv) -> measured-best split
        (2, 1, 8192): 8, (2, 1, 16384): 16, (2, 1, 32768): 16,
        (2, 8, 8192): 2, (2, 8, 16384): 2, (2, 8, 32768): 2,
        (2, 32, 8192): 1, (2, 32, 16384): 1, (2, 32, 32768): 1,
        (2, 64, 8192): 1, (2, 64, 16384): 1, (2, 64, 32768): 2,
        (2, 128, 8192): 1, (2, 128, 16384): 1, (2, 128, 32768): 1,
        (4, 1, 8192): 16, (4, 1, 16384): 16, (4, 1, 32768): 16,
        (4, 8, 8192): 2, (4, 8, 16384): 2, (4, 8, 32768): 2,
        (4, 32, 8192): 1, (4, 32, 16384): 1, (4, 32, 32768): 4,
        (4, 64, 8192): 1, (4, 64, 16384): 2, (4, 64, 32768): 2,
        (4, 128, 8192): 1, (4, 128, 16384): 1, (4, 128, 32768): 1,
    }
    # KNOWN MISSES, WITH WHAT EACH COSTS. A miss is named with its measured cost,
    # never called a tie: a self-test that DEFENDS drift rather than catching it is
    # the defect the pin comment above warns about, one layer up. The assertion
    # below requires the miss set to match EXACTLY. A new miss fails, and so does a
    # miss that disappears, because either means the ground truth needs re-pinning.
    #
    # Both entries are int2, and no int2 pair ranks every cell correctly, which
    # makes these a limit of the cost model's FORM rather than of its calibration;
    # `(2, 1, 8192)` is also the tightest margin on the grid at about 1.6 us.
    KNOWN_MISSES = {
        (2, 1, 8192): (16, 8, 4.04),      # plan s16 13.38 us vs best s8 12.86 us
        (2, 64, 16384): (2, 1, 0.37),     # plan s2 231.26 us vs best s1 230.41 us
    }
    misses = {}
    for (bits, b, kv), want in sorted(MEASURED_BEST.items()):
        got, mode = plan_splits_sm100(b, 8, kv, bits=bits)
        known = KNOWN_MISSES.get((bits, b, kv))
        ok = got == want
        note = ""
        if not ok:
            misses[(bits, b, kv)] = got
            note = (f" KNOWN, costs {known[2]:.2f}%" if known and known[0] == got
                    else " UNEXPECTED")
        print(f"int{bits} b{b:>3} kv{kv:>6}: plan s{got:<2} ({mode:<6}) "
              f"measured-best s{want:<2} {'OK' if ok else 'MISS' + note}")
    expected = {k: v[0] for k, v in KNOWN_MISSES.items()}
    if misses == expected:
        print(f"SELF-TEST PASS ({len(misses)} known misses, worst "
              f"{max(v[2] for v in KNOWN_MISSES.values()):.2f}%)")
    else:
        new_misses = {k: v for k, v in misses.items() if expected.get(k) != v}
        gone = {k: v for k, v in expected.items() if misses.get(k) != v}
        print(f"SELF-TEST FAIL: unexpected {new_misses}, no longer missing {gone}")
    assert misses == expected, (
        f"the planner's miss set moved: got {misses}, expected {expected}. Re-pin "
        f"MEASURED_BEST and KNOWN_MISSES from a fresh sweep rather than widening "
        f"the tolerance.")
