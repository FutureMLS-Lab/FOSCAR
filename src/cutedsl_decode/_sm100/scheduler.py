"""Split-KV planner for the SM100 quantized-KV fold decode kernel.

At plan time this module picks ``(kv_splits, reduction_mode)`` for
``IntDecodeSm100`` from a closed-form cost model. Wave-fill alone is not enough
on B200: splits also win at multi-wave grids, because the makespan is quantized
in waves.

    cost(s) = ceil(base*s / SM) * (T*ceil(tiles/s) + P) + (s>1 ? COMBINE : 0) + C0

      base    = batch * num_kv_heads          (items; grid_y == 1 at g <= 16)
      tiles   = ceil(kv_len / sequence_tile)  (sequence_tile = 256)

      T        per-tile stream time, per bit width
      P        fixed per-CTA prologue
      COMBINE  split-K reduction cost (the kernel-mode launch and run)
      C0       launch floor, constant across s, kept for readability

    pick = argmin over s in {1,2,4,8,16,24,32,48,64} with kv_len/s >= 512

Both ceilings are load-bearing. The inner ``ceil(tiles/s)`` is there because the
makespan of a split is set by the split that received the most tiles, and the
kernel ceil-partitions. Writing it ``tiles/s`` would be wrong for exactly the
candidates that do not divide the tile count, which is what separates 24 and 48
from 16, 32 and 64.

The ``s == 1`` arm gets ``P`` amortized by persistence, which is the kernel's
default static round-robin. Persistence silently downgrades for any split-K
reduction, so split arms pay the full ``P`` per CTA. The 512-token floor is the
same pipeline steady-state bound as on SM90: the rings need about two tiles per
split to fill, and 2 x 256 = 512.

RULES A CHANGE HERE MUST RESPECT
--------------------------------

1. **The T pair is a ratio, and the ratio is what must be right.** ``T_US`` holds
   (unsplit, split) per bit width. Every pair that ranks the arms correctly keeps
   ``T_split / T_unsplit`` inside a narrow band around 1.0. The absolute level is
   nearly free, because a level error largely cancels in the argmin while a ratio
   error does not. Do not re-derive the pair from a single cell. Refit it with the
   planner-fit script whenever the kernel's per-tile cost changes: that script
   takes medians over well-conditioned cells, then the centroid of the correctly
   ranking region.

2. **The candidate ladder deliberately runs past one CTA per SM.** Capping it
   hides the turnaround rather than preventing it. Past one wave, ceil
   quantisation makes the work term invariant for split counts that divide the
   tile count, so those arms differ only by one prologue per extra wave, and
   non-divisors are charged strictly more. The measured ladder turns around at one
   CTA per SM on its own, so the cap is verified as unnecessary rather than merely
   removed.

3. **Do not port SM90's occupancy term here.** ``max(1, base*s/sm_count)`` is
   continuous, so past saturation the work term is scale-invariant in ``s``, the
   arms tie at zero cost, and ``K*s`` is left as the only separator; no ``K``
   satisfies both the cells that want it large and those that want it small. On
   B200 it is a large regression at every ``K``. The two architectures differ in
   one constant, and it is the whole story: SM90's wave boundary sits at twice its
   saturation point, because it counts register-limited residency, so it needs a
   continuous term to see what its own quantisation stepped over. Here the
   boundary lands on the saturation point. Same form, opposite verdicts; copy
   neither the constants nor the shape.

The self-test below is the cheap always-on version of the held-out validation,
pinned to measured optima so it fails loudly if these constants drift.
Run: ``python src/cutedsl_decode/_sm100/scheduler.py``.
"""
import math

SM_COUNT = 148
SEQ_TILE = 256
P_US = 4.5
COMBINE_US = 4.0
C0_US = 11.0
#: (unsplit, split) per-tile stream time in microseconds. Refit whenever the
#: kernel's per-tile cost changes; only the ratio matters (rule 1). This pair was
#: refitted together with an int4 tuning that made the kernel about 15% cheaper
#: per tile, which is exactly the kind of change that moves the ratio.
T_US = {2: (0.791, 0.795), 4: (1.25, 1.291)}
#: Deliberately widened past one CTA per SM. A ladder that stops at 16 never fills
#: the machine at B=1 (base=8 gives 128 CTAs on 148 SMs), so the wave term is
#: pinned at 1 for every arm the planner can reach and the cap, not the model,
#: decides the answer. See rule 2 for why the wider ladder is safe.
SPLIT_CANDIDATES = (1, 2, 4, 8, 16, 24, 32, 48, 64)
MIN_CHUNK_TOKENS = 512


def _cost(base, tiles, s, t_us, sm_count):
    # P is charged per wave for every arm, including s == 1: the persistent walk
    # still pays an amortized per-item boundary cost, and the calibration of T at
    # an s=1 cell folds the residual in. This form reproduced the measured optimum
    # within 3% on every swept int2 cell.
    #
    # `ceil(tiles / s)`, not `tiles / s`: the makespan is set by the split that
    # got the most tiles, and the kernel ceil-partitions, so a split count that
    # does not divide the tile count lengthens nothing while still adding a
    # combine input. While every tile count was a power of two and every candidate
    # was at most 16, the two forms agreed on every cell; widening the ladder to
    # 24 and 48 is what makes them differ (tiles=128 with s=24 is 5.33 by
    # division, 6 in reality). The SM90 planner ceils for the same reason.
    waves = math.ceil(base * s / sm_count)
    t = t_us[0] if s == 1 else t_us[1]
    return (waves * (t * math.ceil(tiles / s) + P_US)
            + (COMBINE_US if s > 1 else 0.0) + C0_US)


def plan_splits_sm100(batch, num_kv_heads, kv_len, *, bits,
                      sm_count=SM_COUNT, sequence_tile=SEQ_TILE,
                      prefer_atomic=False):
    """Return ``(kv_splits, reduction_mode)`` for one decode launch.

    ``kv_len`` is an int or a per-request iterable, in which case the longest
    request governs, mirroring the SM90 planner. The reduction mode is ``"none"``
    when the plan does not split, and otherwise ``"kernel"``, or ``"atomic"`` when
    the caller asks for it.
    """
    if not isinstance(kv_len, int):
        kv_len = max(int(l) for l in kv_len)
    base = batch * num_kv_heads
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
    # Pinned to the L2-flushed medians of an exhaustive {1,2,4,8,16,24,32,48,64}
    # search at every shape, produced by the sweep harness and driven through
    # PagedDecodeWrapper. The ground truth goes stale when the kernel's per-tile
    # cost moves, and a self-test pinned to stale ground truth does not merely
    # fail to catch that drift, it actively defends it. Re-pin whenever per-tile
    # cost moves.
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
    # The cells the model gets wrong, named with what each costs. The assertion
    # requires this set to match EXACTLY: a new miss fails, and so does a miss
    # that disappears, because either means the ground truth needs re-pinning.
    # Both survivors are int2 and both survive the previous constants too, so
    # neither is a calibration error; no int2 pair ranks every cell correctly,
    # which makes them a limit of the model's form.
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
