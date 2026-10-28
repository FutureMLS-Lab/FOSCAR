# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""Library-side compile/launch glue for ``IntDecodeSm100`` (quantized-KV decode).

This module is the SINGLE place that knows how to build and launch the SM100
fold kernel. It was lifted out of ``scripts/_fold_harness.py`` (a benchmark
helper) so that :class:`cutedsl_decode.PagedDecodeWrapper` — the public API —
can drive the quantized kernels through exactly the same code the gates and
benchmarks drive. ``scripts/_fold_harness.py`` now re-exports from here; there
is one compile path and one launch path, not two that can drift.

What lives here
---------------
``BITS_TUNING``      per-bits ctor knobs applied to every compile (see its docstring)
``compile_fold``     ctor + symbolic-shape ``cute.compile``
``launch_fold``      the raw 20-argument launch (ONE call site)
``run_fold``         ``launch_fold`` + workspace handling (harness-compatible)
``alloc_fold_workspace``  out/m/o_partial/l_partial/m_partial allocation
``global_pscale_lp`` / ``global_pscale_lp_flex``  launch-global P scale exponent

What does NOT live here: pool CONSTRUCTION (``build_pools_lut`` /
``build_pools_flex`` / ``v_store_perm_sm100*``). Those are in
``scripts/probes/repack_sm100.py`` and are a test/benchmark fixture — a serving
integration packs its own KV cache. ``_fold_harness.v_pool_perm`` therefore
stays in the harness.
"""
import os

import torch


#: Per-bits constructor knobs applied to every compile unless the caller
#: overrides them.
#:
#: They exist because a benchmark round measured these values and then left the
#: ctor defaults conservative, on the understanding that callers would tune. No
#: caller did, so int4 shipped wrong on three knobs in a published table. Applying
#: them here is what makes the measured configuration the one a caller gets.
#:
#: NOT BIT-IDENTICAL. Applying the tuning is a tiny reassociation, not a no-op: it
#: holds at 9 of 12 shipped head configs under torch.equal and differs at
#: QH32/KVH8 with reduction="none" by rel_l2 2.03e-04, about 25x inside the 5.0e-03
#: gate. Do not describe it as bit-identical.
#:
#: The dequant warpgroup count was a tuning knob until 2026-09-14: int2 always ran
#: one warpgroup, and int4 moved from two to one on 2026-09-04, when dwg=2 measured
#: 1.3% to 2.5% slower at every batch of 32 and above with no numerics difference
#: (`data/b200_i{2,4}_dwg_ab_{fwd,rev}_20260831.csv`, both directions, ssr-b200 GPU 7).
#: One warpgroup has been hardcoded since 2026-09-14, as the paragraph below records.
#:
#: scale_stages STAYS UNSET ON int4, AND THE ISOLATION DOES NOT REPRODUCE. Four arms
#: on the g=8 gate, worst int4 rel_l2 against a 5.0e-03 bar:
#:
#:     arm                     2026-07      2026-09-04
#:     dwg2 + cs16             PASS         PASS, PASS  (1.796e-03)
#:     dwg1 + cs16             --           PASS, PASS  (1.796e-03)
#:     dwg1 + cs16 + ss16      PASS         FAIL x2     (1.741e-02, 2.294e-02)
#:     dwg2 + cs16 + ss16      FAIL         PASS, PASS  (1.796e-03)
#:
#: THE TWO COLUMNS ARE INVERTED ON THE ss16 ROWS. July concluded the fault was the
#: TRIPLE with dwg2 and that ss16 was safe without it; September measures the exact
#: opposite. An intermediate draft of this comment asserted September's half as the
#: rule -- "ss=16 is the leg, at any dwg" -- and arm dwg2+cs16+ss16 then passed twice,
#: which is how that overclaim was caught.
#:
#: So the honest reading is neither table: `scale_stages=16` at int4 produces a
#: NONDETERMINISTIC failure at g=8, and the September samples did not reproduce
#: July's arm pattern. Two or three trials per arm cannot establish more than that. Every failure lands on the same cell,
#: int4/uniform/kv8192/g=8/QH64-KVH8/splits=4/scale=f32. Until someone explains the
#: mechanism, ss=16 does not ship at int4 under EITHER count, and a passing arm is not
#: evidence: three of these arms passed on a day the configuration was broken.
#:
#: That costs something real. ss=16 is worth 0.5% to 3.3% on int4, measured both
#: directions with the arms swapped and `arms_bitwise_equal` true:
#:   data/b200_i4_ss16_ab_fwd_20260904.csv  and  ..._rev_20260904.csv
#:
#: THE EVIDENCE FOR THE TABLE ABOVE, so no number here is prose only:
#:   data/b200_int4_ss16_g4_isolation_20260904.log   the four arms, two runs each,
#:                                                   from baselines/isolate_int4_ss16_g4.sh
#:   data/b200_int4_ss16_g4_firstruns_20260904.log   the three earlier ad-hoc runs of
#:                                                   dwg1+cs16+ss16, and the int2 worst Re-run
#: scripts/test_int_decode_sm100.py at g=8 SEVERAL times before believing any pass: the
#: failure is nondeterministic and neighbouring passes prove nothing.
#:
#: int2 keeps ss=16 and is unaffected; its worst is 1.768e-03 across every arm above.
#:
#: SINCE 2026-09-14 THERE IS ONE DEQUANT WARPGROUP, HARDCODED. `dequant_warpgroups`
#: is no longer a constructor knob (the constructor refuses it with TypeError, and
#: every kernel_kwargs door refuses it through `_RETIRED_KERNEL_KWARGS` below), so
#: the dwg2 arms above are history that cannot be re-run as written; the ss=16
#: finding stands on the dwg1 rows, and `baselines/isolate_int4_ss16_g4.sh` now
#: runs the two arms that still build. `k_op_stages` is the K operand ring's depth
#: of the split K/V operand rings, unconditional at both widths since the same
#: date; 3 is the measured INT4 best (the int2 default, min(2, kv_stages // 2),
#: resolves to 2 and is left to the trace-time default).
BITS_TUNING = {
    2: {"code_stages": 16, "scale_stages": 16},
    4: {"code_stages": 16, "k_op_stages": 3},
}

#: The quant-pool page size. `IntDecodeSm100` asserts page_size == 64 (one
#: scale block / one code block per (page, kv-head)).
POOL_PAGE_SIZE = 64
POOL_HEAD_DIM = 128


def blk_tile_n_of(h_q, h_kv):
    """GQA packing factor (`q_head_tile`) for an (h_q, h_kv) pair, incl.
    the fp8 1x1-tile bump (the wrapper convention: g == 1 is not a legal fp8
    MMA N-tile, so it is bumped to 2)."""
    npo2 = lambda x: 1 if x <= 1 else 1 << (x - 1).bit_length()
    return max(min(32, npo2(h_q // h_kv)), 2)


#: Names retired from `IntDecodeSm100` and from `compile_fold`, mapped to the
#: message that says so. Retired OUTRIGHT means nothing replaced the knob, so every
#: value is refused, including False and None. The two RENAMED depth knobs are here
#: as well, because a dict reader would otherwise believe the old spelling; their
#: messages name the new spelling.
_RETIRED_KERNEL_KWARGS = {
    "identity_dequant": (
        "'identity_dequant' was retired together with the raw-byte relay mode "
        "it selected; there is no value of it to pass and nothing replaced it. "
        "The packed pools mKq/mVq/mScale are the kernel's only input contract."),
    "operands_in_tmem": (
        "'operands_in_tmem' was retired together with the K-in-tmem staging arm it "
        "selected (the dequant warpgroup's STTM deposit, the tmem operand ring and "
        "the TMEM-source BMM1 atom); it measured neutral against the shared-memory "
        "operand ring, and its only host, the whole-warpgroup interleave, is deleted "
        "with the 2026-09-14 consolidation. There is no value of it to pass and "
        "nothing replaced it."),
    "dump_stage": (
        "'dump_stage' was retired together with the operand-stage dump it selected "
        "(the mDump tensor, the dequant warpgroup's named-barrier-14 copy and "
        "scripts/_g3_dump_check.py); its only host, the whole-warpgroup interleave, "
        "is deleted with the 2026-09-14 consolidation, and the bitwise output anchors "
        "and the end-to-end operand-matched bar carry what it checked. There is no "
        "value of it to pass and nothing replaced it."),
    "dump": (
        "'dump' was retired together with 'dump_stage' on 2026-09-14: it was "
        "compile_fold's spelling of that knob (it set the constructor's dump_stage and "
        "allocated the mDump tensor). There is no value of it to pass and nothing "
        "replaced it."),
    "softmax_warpgroups": (
        "'softmax_warpgroups' was retired on 2026-09-14 together with the dual-softmax "
        "topology it selected (the second softmax warpgroup, the tile-parity "
        "skip-walks of the S, P and scale rings, and the sM mutex barrier); it "
        "measured flat to slower than one softmax warpgroup, and the in-loop scale "
        "acquire that was its last host is deleted with the scale-quad prefetch made "
        "unconditional. One softmax warpgroup is the only topology, so there is no "
        "value of it to pass and nothing replaced it."),
    "f32x2_fold": (
        "'f32x2_fold' was retired on 2026-09-14 together with the two packed-f32x2 "
        "sites it selected (the softmax P hi/lo quant multiplies and the correction "
        "warpgroup's V-fold epilogue FMAs) and the four packed-f32x2 asm helpers; it "
        "was bit-exact with the scalar fma.rn path by construction and benched flat, "
        "so the scalar path is the only path. There is no value of it to pass and "
        "nothing replaced it."),
    "elide_unit_correction": (
        "'elide_unit_correction' was retired on 2026-09-14 together with the "
        "correction warpgroup's unit-correction elision arm (skip the tmem O "
        "read-modify-write when the correction vector is exactly all-1.0); it was "
        "bit-exact by construction and measured no wall effect, so the unconditional "
        "read-modify-write is the only path. There is no value of it to pass and "
        "nothing replaced it."),
    "int4_k_op_stages": (
        "'int4_k_op_stages' was renamed to 'k_op_stages' on 2026-09-14, when the "
        "split K/V operand rings became unconditional at both widths; pass "
        "`k_op_stages`, the K operand ring's depth (None resolves to "
        "min(2, kv_stages // 2) at trace time, and BITS_TUNING sets 3 at bits == 4)."),
    "int4_code_stages_k": (
        "'int4_code_stages_k' was renamed to 'code_stages_k' on 2026-09-14, when the "
        "split K/V code rings became unconditional at both widths; pass "
        "`code_stages_k`, the K code ring's depth (None splits code_stages in half)."),
    "dequant_warpgroups": (
        "'dequant_warpgroups' was retired from IntDecodeSm100 on 2026-09-14: the "
        "kernel has had one dequant warpgroup, hardcoded, since that date and takes "
        "no such argument (the knob selected two or four cooperating warpgroups, "
        "which measured slower at both widths). There is no value of it to pass "
        "and nothing replaced it."),
    "kernel_cls": (
        "'kernel_cls' was retired on 2026-10-07 together with the SM100 ping-pong "
        "fork it selected (FoldDecodeSm100PP, kernel_cls=\"pingpong\"), an "
        "unshipped experiment that failed the end-to-end check on long "
        "single-split cells; IntDecodeSm100 is the only SM100 kernel class. Every "
        "value is refused, None included, because there is nothing left to "
        "select, so drop the argument. The Hopper ping-pong kernel "
        "(IntPingPongDecodeSm90) is a different kernel and is unaffected."),
}


def refuse_retired_kernel_kwargs(kernel_kwargs, where="compile_fold"):
    """Raise if a `kernel_kwargs` dict carries a name retired outright.

    THIS IS THE ONE PLACE, and it is a function rather than a line inside
    `compile_fold` because a dict read BY NAME is a door that discards instead
    of raising. A keyword argument that no longer exists raises on its own; a
    dict key that no longer means anything is simply ignored, and the caller
    gets a default it did not ask for and no warning. So every helper that
    consumes a `kernel_kwargs` dict has to consult this, not only the compile
    entry point.

    The doors, counted here because two rounds of codex review each found one
    more (2026-09-08): `compile_fold`, whose `**kernel_kwargs` and
    `bits_tuning` merge into one dict (it refuses after that merge, so a
    retired name raises this message rather than the constructor's generic
    TypeError, and so a compile_fold-level parameter such as `dump` or
    `kernel_cls` that the splat would otherwise hand to the constructor is
    named as retired); `quant_cfg_of`, and `scale_pool_dtype_of` through it,
    which resolve the quant-flex configuration out of the same dict;
    `_fold_harness.k_pool_perm` / `v_pool_perm`, which resolve the POOL ORDER
    out of it, where a silently defaulted answer is wrong at every K channel
    and raises nothing; and `PagedDecodeWrapper`, whose plan refuses the dict
    it is handed and whose run refuses the planned dict it reads the K pool
    order out of.
    """
    if not kernel_kwargs:
        return
    for name, why in _RETIRED_KERNEL_KWARGS.items():
        if name in kernel_kwargs:
            raise TypeError(f"{where}: {why}")


def quant_cfg_of(page_size, head_dim, kernel_kwargs=None):
    """Normalized (k_quant, v_quant, sdtype_str, qoff) from a kernel-kwargs
    dict (the compile-keying entry for the quant-flex configs; absent kwargs
    mean the legacy token-128 pair, f32 pool — the shipped default).

    K token-64 keeps its shipped bf16 default, but an explicit ``sdtype="f32"``
    selects the full-precision metadata path. Bf16 metadata remains exclusive
    to K token-64.
    """
    from ..quant_layout import quant_block_offsets

    kkw = kernel_kwargs or {}
    refuse_retired_kernel_kwargs(kkw, where="quant_cfg_of")
    kq = tuple(kkw.get("k_quant") or ("token", head_dim))
    vq = tuple(kkw.get("v_quant") or ("token", head_dim))
    kq, vq = (kq[0], int(kq[1])), (vq[0], int(vq[1]))
    sdt = kkw.get("sdtype")
    if sdt is None:
        sdt = "bf16" if kq == ("token", head_dim // 2) else "f32"
    if sdt not in ("f32", "bf16"):
        raise ValueError(
            f"sdtype must be 'f32' or 'bf16'; got {sdt!r}")
    if sdt == "bf16" and kq != ("token", head_dim // 2):
        raise ValueError(
            "sdtype='bf16' is only legal with K token-64; every other "
            "IntDecodeSm100 configuration uses an f32 scale pool")
    qoff = quant_block_offsets(kq, vq, N=page_size, D=head_dim)
    return kq, vq, sdt, qoff


def scale_pool_dtype_of(page_size, head_dim, kernel_kwargs=None):
    """Return the torch dtype required by the resolved SM100 scale-pool config."""
    _, _, sdt, _ = quant_cfg_of(page_size, head_dim, kernel_kwargs)
    return torch.bfloat16 if sdt == "bf16" else torch.float32


def require_scale_pool_dtype(mScale, expected_dtype, *, expected_sdtype=None):
    """Fail before launch when a scale pool does not match the compiled dtype."""
    if mScale.dtype != expected_dtype:
        planned = (
            f"sdtype={expected_sdtype!r} ({expected_dtype})"
            if expected_sdtype is not None else str(expected_dtype)
        )
        raise ValueError(
            f"scale_pool.dtype={mScale.dtype} mismatches planned {planned}")


def resolve_fold_tiling(k_quant, v_quant, *, sequence_tile=None, o_stages=None):
    """`IntDecodeSm100.resolve_tiling`, importable without naming the kernel
    module. Drivers and the split planner MUST go through this rather than
    duplicating the table -- a driver that hardcodes 256 while the kernel
    resolves 128 is exactly the planner/kernel disagreement this indirection
    exists to prevent."""
    from .int_fold_decode import IntDecodeSm100

    return IntDecodeSm100.resolve_tiling(
        k_quant, v_quant, sequence_tile=sequence_tile, o_stages=o_stages)


def global_pscale_lp(mScale):
    """Launch-global P pscale exponent: lp = floor(log2(448 / max
    s_v over the pool)) (s_v = mScale[..., 2]). Certified conservative-safe
    to 64x (scripts/probes/certify_global_pscale_sm100.py).

    Takes the LEGACY 4-D pool [NP, HKV, N, 4]."""
    import math
    sv_max = float(mScale[..., 2].max().item())
    return float(math.floor(math.log2(448.0 / max(sv_max, 1e-30))))


def global_pscale_lp_flex(mScale, k_quant, v_quant, *,
                          page_size=POOL_PAGE_SIZE, head_dim=POOL_HEAD_DIM):
    """`global_pscale_lp` for a flex (flat sectioned) scale pool: token-wise V
    reads max s_v from the V ring section; channel-wise V has NO s_v inside
    the P operand (the per-channel scale drains post-MMA), so max |w| = max P
    = 1 and lp = floor(log2(448)) = 8."""
    import math
    from ..quant_layout import quant_block_offsets

    off = quant_block_offsets(k_quant, v_quant, N=page_size, D=head_dim)
    if off["legacy"]:
        return global_pscale_lp(mScale.reshape(*mScale.shape[:2], page_size, 4))
    if v_quant[0] == "channel":
        return float(math.floor(math.log2(448.0)))
    ring = mScale.float()[..., off["v_ring"]:off["v_ring"] + 2 * page_size]
    sv_max = float(ring.reshape(*ring.shape[:-1], page_size, 2)[..., 0].max().item())
    return float(math.floor(math.log2(448.0 / max(sv_max, 1e-30))))


def pscale_lp_for_pool(mScale, k_quant=None, v_quant=None, *,
                       page_size=POOL_PAGE_SIZE, head_dim=POOL_HEAD_DIM):
    """`pscale_lp` for either pool form, dispatching on the tensor rank.

    A 4-D pool is the legacy [NP, HKV, N, 4] interleaved block; a 3-D pool is
    the flex flat sectioned block and NEEDS its (k_quant, v_quant) to locate
    the V ring section. NOTE: this does a ``.max().item()`` — a device sync.
    Hoist it out of a hot loop and pass the value explicitly.
    """
    if mScale is None:
        return 0.0
    if mScale.dim() == 4:
        return global_pscale_lp(mScale)
    if k_quant is None or v_quant is None:
        raise ValueError(
            "a flat sectioned (3-D) scale pool needs k_quant/v_quant to locate "
            "the V ring section; pass them or pass pscale_lp explicitly")
    return global_pscale_lp_flex(mScale, k_quant, v_quant,
                                 page_size=page_size, head_dim=head_dim)


# ---------------------------------------------------------------------------
# IntDecodeSm100 compile
# ---------------------------------------------------------------------------


def _refuse_cuda12_dsl_backend():
    """Blackwell builds need the DSL's CUDA 13 backend (`cutedsl-decode[cu13]`).

    nvidia-cutlass-dsl 4.8.0's CUDA 12 backend vendors ptxas 12.9, and on B200 it left
    output elements of this kernel unwritten (2026-10-07, the shipped B = 1 cell and a
    channel-64 K cell) while its CUDA 13 backend (ptxas 13.4) is correct. Blackwell runs
    CUDA 13; the cu12 extra is for Hopper. Called before tracing, so it costs nothing at
    launch and a cached object, keyed on the DSL's CUDA version, never reaches it.
    """
    from cutlass.base_dsl import version_info

    major = getattr(version_info.CUDA_VERSION, "major", None)
    if major is not None and major < 13:
        raise RuntimeError(
            f"compile_fold: the SM100 kernel needs nvidia-cutlass-dsl's CUDA 13 backend "
            f"(install cutedsl-decode[cu13]); this DSL loaded CUDA "
            f"{version_info.CUDA_VERSION}, whose ptxas miscompiles this kernel on B200")


def compile_fold(
    *,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    reduction,
    bits,
    in_dtype=None,
    out_dtype=None,
    debug_s=False,
    sequence_tile=None,
    tma_mask=False,
    q_head_tile=None,
    bits_tuning=None,
    knobs_out=None,
    kernel_out=None,
    **kernel_kwargs,
):
    """Compile `IntDecodeSm100` with symbolic batch/page-count shapes
    (wrapper-style fake tensors). The kernel's only input contract is the
    packed LUT pools `mKq`/`mVq`/`mScale`. `debug_s` adds the matching
    gmem debug tensor (`mDbgS`).

    ONE compile serves every batch size and every KV length — the batch, page
    count and kv_splits dims are `cute.sym_int()`. Memoise on the remaining
    (static) arguments; a trace costs 10 to 13 s on B200 with DSL 4.5.2 (measured
    2026-09-11), and `cutedsl_decode.kernel_cache` serves a repeat of the same
    trace from disk in milliseconds.

    `bits_tuning`: the per-bits ctor knobs to apply (default `BITS_TUNING`).
    Overridable so an A/B driver can compile a control arm WITHOUT the
    promoted tuning (`baselines/check_bit_identity.py` does exactly that);
    pass `{}` for "no tuning". Whatever is passed is also what the
    post-construction guard below checks, so a caller cannot silently get a
    kernel that did not receive the knobs it asked for.

    `kernel_out`: optional dict, filled with `{"kernel": <the constructed
    IntDecodeSm100>}` before the compile. This function returns the compiled
    CALLABLE, which carries none of the resolution the constructor did, so a caller
    had no way to ask whether the kernel it just built actually reached the
    configuration it asked for. That is not hypothetical: the `i2_persist_multi`
    anchor cell sat at `reduction="kernel"` for weeks, where `self.persistent`
    resolves to False, so the one cell named for the persistent item loop never
    entered it and the anchor could not tell. (item E3)

    `knobs_out`: optional dict, filled with the tuned knobs READ BACK OFF THE
    CONSTRUCTED KERNEL. The guard below already asserts on those values, but a
    memoising caller (the wrapper) only traces once and would never see the
    assert again on a cache hit — so it stores this and re-checks per plan().
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import Float32, Int32
    from .int_fold_decode import IntDecodeSm100

    in_dtype = in_dtype or cutlass.Float8E4M3FN
    out_dtype = out_dtype or cutlass.BFloat16
    acc_dtype = cutlass.Float32
    tuning = BITS_TUNING if bits_tuning is None else bits_tuning

    if q_head_tile is None:
        q_head_tile = blk_tile_n_of(num_qo_heads, num_kv_heads)

    # Apply the per-bits tuning unless the caller overrode it. This MUST precede
    # construction. Placed after it, it is dead code, and the only symptom is
    # timings that come back bit-for-bit unchanged, so measure the wiring rather
    # than assuming it.
    kernel_kwargs = dict(kernel_kwargs)
    for _k, _v in tuning.get(bits, {}).items():
        if _k not in kernel_kwargs:
            kernel_kwargs[_k] = _v

    # A RETIRED SPELLING MUST RAISE BY NAME, and `**kernel_kwargs` is where one
    # hides. Splatting into `IntDecodeSm100` already gives TypeError, but a generic
    # one that does not say what replaced the name, so the renamed spellings are
    # refused here with the new spelling in the message.
    #
    # AFTER the bits_tuning merge, not before it. `bits_tuning` is a caller-supplied
    # dict of ctor knobs, so it is a second way in, and a guard placed ahead of the
    # merge misses it.
    for _retired, _now in (("grouped_head_tile", "q_head_tile"),
                           ("headdim", "head_dim"),
                           ("n_tile", "page_size")):
        if _retired in kernel_kwargs:
            raise TypeError(
                f"compile_fold: {_retired!r} was renamed to {_now!r}; pass that "
                f"instead of routing it through kernel_kwargs or bits_tuning")
    # RETIRED OUTRIGHT, not renamed. The constructor splat would raise on its own,
    # but these messages say why the name is gone.
    if "fused_correction" in kernel_kwargs:
        raise TypeError(
            "compile_fold: 'fused_correction' was retired together "
            "with the alternate correction epilogue it selected; there is no "
            "value of it to pass.")
    if "prediction_tile" in kernel_kwargs:
        raise TypeError(
            "compile_fold: 'prediction_tile' was retired. The "
            "query-token tile is pinned at 1, so passing 2 would have been "
            "accepted and ignored rather than honoured. The runtime prediction "
            "axis is unaffected; what the knob bought was packing several query "
            "tokens into one MMA N tile.")
    # RETIRED OUTRIGHT, and NOTHING REPLACED IT. `identity_dequant` was a
    # bring-up scaffold: it swapped the input contract for plain e4m3
    # `k_bshd`/`v_bshd` tensors, turned the affine fold off everywhere, and left
    # the dequant warpgroup relaying raw bytes through shared memory. It existed
    # to be compared bitwise against an unquantized baseline kernel, and that
    # baseline is gone, so there is no mode left for it to select. Refused HERE,
    # after the bits_tuning merge and before construction, for the same reason
    # as the guards above. The messages, and the other doors that consult the
    # same table, live in `refuse_retired_kernel_kwargs`. `kernel_cls` is in that
    # table too: it was a parameter of this function until 2026-10-07, so a
    # caller still passing it lands in `**kernel_kwargs` and is told it is
    # retired rather than handed to the constructor.
    refuse_retired_kernel_kwargs(kernel_kwargs, where="compile_fold")

    # Quant-config tiling. Resolved HERE as well as inside the ctor because the
    # compile CACHE KEY upstream is built from what we pass. Same classmethod
    # both places, so the two cannot disagree; the assert after the ctor proves
    # the value actually reached the object. BITS_TUNING is NOT given a
    # sequence_tile/o_stages key -- it is per-bits, and callers replace the
    # whole dict (`{2: {}, 4: {}}`) to compile an untuned kernel.
    _kq_t, _vq_t, _sdt_t, _ = quant_cfg_of(page_size, head_dim, kernel_kwargs)
    sequence_tile, _o_res = IntDecodeSm100.resolve_tiling(
        _kq_t, _vq_t, sequence_tile=sequence_tile,
        o_stages=kernel_kwargs.get("o_stages"))
    kernel_kwargs["o_stages"] = _o_res

    # g <= 8 is the P hi/lo row budget (`2 * blk_tile_n <= mma_tile_n`).
    # `__call__` asserts the same bound against the tensors it is handed;
    # refusing here names the head tile rather than an MMA shape.
    assert q_head_tile <= 8, "fold: g <= 8 (the P hi/lo row budget)"

    fmha = IntDecodeSm100(
        page_size,
        head_dim,
        q_head_tile,
        sequence_tile=sequence_tile,
        reduction_mode=reduction,
        tma_mask=tma_mask,
        bits=bits,
        debug_s=debug_s,
        **kernel_kwargs,
    )

    # Prove the resolved tiling reached the object. Same guard class, and the
    # same recorded reason, as the bits_tuning assert below: three separate
    # false nulls came from an override wired to a ctor the caller never
    # reached (int_fold_decode.py's FOLD_TRACE_TILING comment). Reads
    # `_o_stages` -- `o_stages` does not exist until __call__.
    assert fmha.sequence_tile == sequence_tile, (
        f"resolve_tiling disagreement: compile_fold resolved "
        f"sequence_tile={sequence_tile}, kernel has {fmha.sequence_tile}")
    assert fmha._o_stages == kernel_kwargs["o_stages"], (
        f"resolve_tiling disagreement: compile_fold resolved "
        f"o_stages={kernel_kwargs['o_stages']}, kernel has "
        f"{fmha._o_stages}")

    sym_batch = cute.sym_int()
    sym_virtual_pages = cute.sym_int()
    sym_total_logical_pages = cute.sym_int()
    sym_kv_splits = cute.sym_int()
    sym_prediction = cute.sym_int()

    fct = cute.runtime.make_fake_compact_tensor
    seqlens_fake = fct(Int32, (sym_batch,), assumed_align=16)
    table_offsets_fake = fct(Int32, (sym_batch,), assumed_align=16)
    page_table_fake = fct(Int32, (sym_total_logical_pages,), assumed_align=16)
    q_fake = fct(in_dtype, (sym_batch, sym_prediction, num_qo_heads, head_dim),
                 stride_order=(3, 2, 1, 0), assumed_align=16)
    o_fake = fct(out_dtype, (sym_batch, sym_prediction, num_qo_heads, head_dim),
                 stride_order=(3, 2, 1, 0), assumed_align=16)

    k_row_bytes = head_dim * bits // 8
    v_row_bytes = page_size * bits // 8
    mKq_fake = fct(cutlass.Uint8, (sym_virtual_pages, num_kv_heads, page_size,
                                   k_row_bytes),
                   stride_order=(3, 2, 1, 0), assumed_align=16)
    mVq_fake = fct(cutlass.Uint8, (sym_virtual_pages, num_kv_heads, head_dim,
                                   v_row_bytes),
                   stride_order=(3, 2, 1, 0), assumed_align=16)
    kq_cfg, vq_cfg, sdt, qoff = quant_cfg_of(page_size, head_dim, kernel_kwargs)
    if qoff["legacy"]:
        mScale_fake = fct(cutlass.Float32,
                          (sym_virtual_pages, num_kv_heads, page_size, 4),
                          stride_order=(3, 2, 1, 0), assumed_align=16)
    else:
        # flex configs: flat sectioned block, sdtype elements
        sdt_cutlass = cutlass.BFloat16 if sdt == "bf16" else cutlass.Float32
        mScale_fake = fct(sdt_cutlass,
                          (sym_virtual_pages, num_kv_heads, qoff["block_tot"]),
                          stride_order=(2, 1, 0), assumed_align=16)

    mDbgS_fake = (fct(cutlass.Float32, (sequence_tile, q_head_tile),
                      stride_order=(1, 0), assumed_align=16) if debug_s else None)

    if reduction == "kernel":
        m_fake = fct(acc_dtype, (sym_batch, sym_prediction, num_qo_heads),
                     stride_order=(2, 1, 0), assumed_align=16)
        o_partial_fake = fct(
            acc_dtype, (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads, head_dim),
            stride_order=(4, 3, 2, 1, 0), assumed_align=16)
        m_partial_fake = fct(acc_dtype, (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads),
                             stride_order=(3, 2, 1, 0), assumed_align=16)
        l_partial_fake = fct(acc_dtype, (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads),
                             stride_order=(3, 2, 1, 0), assumed_align=16)
    else:
        m_fake = o_partial_fake = m_partial_fake = l_partial_fake = None

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    # Opt-level default is bits-scoped, and measured: int2 is faster at 2 and int4
    # regresses hard at 2, so int4 keeps 3. An env override wins, which is what the
    # correctness gates use (OPT_LEVEL=0).
    opt_level = os.environ.get(
        "CUTE_DSL_OPT_LEVEL", "2" if bits == 2 else "3")
    # GUARD: prove the per-bits tuning actually reached the constructed kernel.
    # Applying it as a dead statement AFTER construction is a silent no-op, whose
    # only symptom is timings that come back bit-for-bit identical. A measured
    # optimisation that the object never receives is a recurring failure in this
    # repo, the split-K planner having been unwired the same way, so assert rather
    # than trust.
    for _k, _v in tuning.get(bits, {}).items():
        _got = getattr(fmha, _k, None)
        if knobs_out is not None:
            knobs_out[_k] = _got
        if _k in kernel_kwargs and kernel_kwargs[_k] != _v:
            continue                      # caller overrode deliberately
        assert _got == _v, (
            f"per-bits tuning did not reach the kernel: bits={bits} {_k} is "
            f"{_got!r}, expected {_v!r}; see BITS_TUNING.")

    if kernel_out is not None:
        kernel_out["kernel"] = fmha

    options = f"--enable-tvm-ffi --opt-level {opt_level}"

    def _trace():
        from ..kernel_cache import init_cuda_driver
        init_cuda_driver()
        _refuse_cuda12_dsl_backend()
        return cute.compile(
            fmha,
            Int32(1),
            seqlens_fake,
            table_offsets_fake,
            page_table_fake,
            q_fake,
            o_fake,
            None,           # l
            m_fake,
            o_partial_fake,
            l_partial_fake,
            m_partial_fake,
            Float32(1.0),
            Float32(1.0),
            None,           # threshold_scale_factor
            stream_fake,
            True,           # enable_pdl
            mKq_fake,
            mVq_fake,
            mScale_fake,
            mDbgS_fake,
            Float32(0.0),  # pscale_lp (runtime scalar)
            options=options,
        )

    # THE TRACE GOES THROUGH THE PERSISTENT KERNEL CACHE. Everything above this line
    # still runs on a cache hit: the constructor resolves the tiling, the guards
    # assert the tuning reached the object, and `kernel_out` / `knobs_out` read the
    # constructed kernel, so a hit hands back the same evidence a trace would. What a
    # hit skips is `cute.compile` itself, 10 to 13 s on B200. The key carries the
    # constructed kernel's RESOLVED attributes (not the raw kernel kwargs: an omitted
    # knob and its explicit default resolve to the same attribute and must share a
    # key), the trace arguments that are not attributes, the options string and the DSL
    # identity.
    from .. import kernel_cache
    from . import int_fold_decode as _ifd

    sources = kernel_cache.source_closure(_ifd.__file__, __file__)
    config = {
        "kernel": kernel_cache.simple_attributes(fmha),
        "trace": {
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "reduction": reduction,
            "bits": bits,
            "in_dtype": in_dtype,
            "out_dtype": out_dtype,
            "acc_dtype": acc_dtype,
            "debug_s": bool(debug_s),
            "scale_fake": {"legacy": bool(qoff["legacy"]),
                           "block_tot": qoff.get("block_tot"), "sdtype": sdt},
        },
    }
    return kernel_cache.compile_or_load(
        kind=f"sm100_{type(fmha).__name__}",
        compile_fn=_trace,
        sources=sources,
        config=config,
        options=options,
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch_fold(compiled, *, kv_splits, seqlens, table_offsets, page_table,
                q_bshd, o_bshd, m=None, o_partial=None, l_partial=None,
                m_partial=None, scale_s=1.0, scale_o=1.0,
                mKq=None, mVq=None, mScale=None, pscale_lp=0.0,
                dbg=None, enable_pdl=True):
    """THE launch site for a `compile_fold`-compiled kernel.

    Everything (the harness `run_fold`, `PagedDecodeWrapper.run`) funnels
    through here, so the 20-argument positional contract exists once. All
    tensors are torch tensors; scalars are Python floats/ints.

    Note the `tvm_ffi.use_torch_stream()` scope: the kernel is compiled with
    `use_tvm_ffi_env_stream=True`, so this is what binds the launch to torch's
    current stream. Dropping it silently moves the launch to another stream.
    """
    import tvm_ffi
    from cutlass.cute.typing import Float32, Int32

    with tvm_ffi.use_torch_stream():
        compiled(
            Int32(kv_splits),
            seqlens,
            table_offsets,
            page_table,
            q_bshd,
            o_bshd,
            None,          # l (lse) — the fold kernel has no LSE output wired
            m,
            o_partial,
            l_partial,
            m_partial,
            Float32(scale_s),
            Float32(scale_o),
            None,          # threshold_scale_factor (BLASST out of scope)
            enable_pdl,
            mKq,
            mVq,
            mScale,
            dbg,
            Float32(pscale_lp),
        )


def alloc_fold_workspace(p, kv_splits, reduction):
    """Pre-allocate everything `run_fold` would otherwise allocate per call.

    Pass the result as `run_fold(..., ws=...)` to keep ALLOCATION out of a
    timed region: five `torch.empty`/`torch.full` calls per launch at
    reduction="kernel", one at "none". Every external baseline in `baselines/`
    allocates in `build()`, so ours has to as well or the protocol is not
    uniform -- which is the one thing that harness exists to guarantee.

    WHAT DOES NOT HOIST, AND WHY. The `m` buffer is not a scratch output; the
    kernel reduces into it with a device-side atomic max
    (`gqa_decode.reduction_epilogue` -> `gmem_fmax`), so it must hold the
    identity `-inf` at the start of EVERY launch. `run_fold` therefore refills
    it per call even with a supplied workspace. That fill is a real launch, but
    it is the kernel's own precondition -- a serving integration pays it too --
    not benchmark overhead, so removing it would flatter our column rather than
    make the comparison fair.
    """
    dev = torch.device("cuda")
    b, h_q = p["b"], p["h_q"]
    d = p.get("d", POOL_HEAD_DIM)
    out = torch.empty(b, 1, h_q, d, device=dev, dtype=torch.bfloat16)
    if reduction == "kernel":
        return dict(
            out=out,
            m=torch.full((b, 1, h_q), -float("inf"), device=dev, dtype=torch.float32),
            o_partial=torch.empty(kv_splits, b, 1, h_q, d, device=dev, dtype=torch.float32),
            l_partial=torch.empty(kv_splits, b, 1, h_q, device=dev, dtype=torch.float32),
            m_partial=torch.empty(kv_splits, b, 1, h_q, device=dev, dtype=torch.float32),
        )
    return dict(out=out, m=None, o_partial=None, l_partial=None, m_partial=None)


def run_fold(compiled, p, kv_splits, reduction, scale_s, scale_o=1.0, *,
             pools=None, dbg=None, pscale_lp=None, ws=None):
    """Launch a `compile_fold`-compiled kernel on a harness problem dict.

    `pools=(mKq,mVq,mScale)` are the packed quantized pools the kernel reads.
    Returns the squeezed [b, h_q, d] bf16 output.

    `ws` is an `alloc_fold_workspace()` dict. Supply it to hoist allocation out
    of the call; omit it and the workspace is allocated per call as before.
    A reused `m` is re-filled with `-inf` here because the kernel atomic-maxes
    into it -- see `alloc_fold_workspace` for why that one launch must stay.
    """
    if ws is None:
        ws = alloc_fold_workspace(p, kv_splits, reduction)
    elif ws["m"] is not None:
        ws["m"].fill_(-float("inf"))
    out = ws["out"]
    mKq, mVq, mScale = pools if pools is not None else (None, None, None)
    if pscale_lp is None:
        assert mScale is None or mScale.dim() == 4, (
            "flex (flat sectioned) mScale pools need an explicit pscale_lp — "
            "use global_pscale_lp_flex(mScale, k_quant, v_quant)")
        pscale_lp = global_pscale_lp(mScale) if mScale is not None else 0.0
    launch_fold(
        compiled,
        kv_splits=kv_splits,
        seqlens=p["seq_lens"],
        table_offsets=p["table_offsets"],
        page_table=p["indices"],
        q_bshd=p["q_fp8"],
        o_bshd=out,
        m=ws["m"],
        o_partial=ws["o_partial"],
        l_partial=ws["l_partial"],
        m_partial=ws["m_partial"],
        scale_s=scale_s,
        scale_o=scale_o,
        mKq=mKq, mVq=mVq, mScale=mScale,
        pscale_lp=pscale_lp,
        dbg=dbg,
    )
    return out.squeeze(1)


def plan_fold(p, bits, *, sequence_tile=256):
    """(kv_splits, reduction) the SM100 planner would choose for this problem.

    Thin adapter from a problem dict (the `_fold_harness.make_pool_problem`
    shape: keys "b", "h_kv", "seq_lens") to
    `cutedsl_decode._sm100.scheduler.plan_splits_sm100`, so the planner is
    reachable from the one place every fold launch goes through instead of
    being library code nobody calls.

    `PagedDecodeWrapper` does NOT go through here — it calls
    `plan_splits_sm100` directly from real plan() arguments. This entry point
    exists for the benchmark/gate drivers that already speak problem dicts.
    """
    from .scheduler import plan_splits_sm100

    seq = p["seq_lens"]
    kv_len = int(seq.max().item()) if hasattr(seq, "max") else max(seq)
    return plan_splits_sm100(p["b"], p["h_kv"], kv_len, bits=bits,
                             sequence_tile=sequence_tile)
