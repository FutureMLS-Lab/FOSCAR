# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""Library-side compile/launch glue for ``IntDecodeSm100`` (quantized-KV decode).

This module is the SINGLE place that knows how to build and launch the SM100
fold kernel, so that :class:`fosca.PagedDecodeWrapper` — the public API —
drives the quantized kernels through exactly the same code a benchmark drives.
There is one compile path and one launch path, not two that can drift.

What lives here
---------------
``BITS_TUNING``      per-bits ctor knobs applied to every compile (see its docstring)
``compile_fold``     ctor + symbolic-shape ``cute.compile``
``launch_fold``      the raw 21-argument launch (ONE call site)
``launch_fold_graph``  the graph split build's launch, apart from eager's
``graph_work_views``  that build's work list and split counts, as views of one buffer
``run_fold``         ``launch_fold`` + workspace handling
``alloc_fold_workspace``  out/o_partial/l_partial/m_partial allocation
``v_page_max_for_pool``   the per-page V-scale maxima of a host-built scale pool
``v_max_hist_of``         the count table of page maxima the decode reads
``v_max_hist_for_pool``   that table for a host-built pool, where the build reads one
``vmax_bucket``, ``pscale_lp_of_bucket``  the exponent bin and its P-scale exponent

What does NOT live here: pool CONSTRUCTION. A serving integration packs its own
KV cache through :class:`fosca.QuantKVCache` and :class:`fosca.QuantKVLayout`;
pool builders are test and benchmark fixtures, not library code.
"""
import os

import torch

from .item_loop import reads_exponent_table


#: Per-bits constructor knobs applied to every compile unless the caller
#: overrides them.
#:
#: The constructor defaults are conservative, so a caller that does not tune gets
#: the conservative configuration. Applying these here is what makes the measured
#: configuration the one a caller gets.
#:
#: NOT BIT-IDENTICAL. Applying the tuning is a tiny reassociation, not a no-op: it
#: holds at 9 of 12 shipped head configs under torch.equal and differs at
#: QH32/KVH8 with reduction="none" by rel_l2 2.03e-04, about 25x inside the 5.0e-03
#: gate. Do not describe it as bit-identical.
#:
#: The dequant warpgroup count is not a knob: one warpgroup is hardcoded at both
#: widths. Two warpgroups measured 1.3% to 2.5% slower at every batch of 32 and
#: above, with no numerics difference.
#:
#: scale_stages STAYS UNSET ON int4. `scale_stages=16` at int4 produces a
#: nondeterministic failure at g=8, and every observed failure lands on the same
#: cell: int4/uniform/kv8192/g=8/QH64-KVH8/splits=4/scale=f32. The mechanism is not
#: understood, so no count of passing trials establishes it as safe and it does not
#: ship at int4; the measured gain it would buy is 0.5% to 3.3%. int2 keeps ss=16
#: and is unaffected; its worst observed rel_l2 is 1.768e-03.
#:
#: `k_op_stages` is the K operand ring's depth of the split K/V operand rings,
#: unconditional at both widths. 3 is the measured INT4 best; the int2 default,
#: min(2, kv_stages // 2), resolves to 2 and is left to the trace-time default.
BITS_TUNING = {
    2: {"code_stages": 16, "scale_stages": 16},
    4: {"code_stages": 16, "k_op_stages": 3},
}

#: The quant-pool page size. `IntDecodeSm100` asserts page_size == 64 (one
#: scale block / one code block per (page, kv-head)).
POOL_PAGE_SIZE = 64
POOL_HEAD_DIM = 128


def blk_tile_n_of(h_q, h_kv):
    """Physical query-head tile: four-row padding or eight-row head tiling."""
    npo2 = lambda x: 1 if x <= 1 else 1 << (x - 1).bit_length()
    return max(min(8, npo2(h_q // h_kv)), 4)






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
    kq = tuple(kkw.get("k_quant") or ("token", head_dim))
    vq = tuple(kkw.get("v_quant") or ("token", head_dim))
    kq, vq = (kq[0], int(kq[1])), (vq[0], int(vq[1]))
    sdt = kkw.get("sdtype")
    if sdt is None:
        sdt = "bf16" if kq == ("token", head_dim // 2) else "f32"
    if sdt not in ("f32", "bf16"):
        raise ValueError(
            f"sdtype must be 'f32' or 'bf16'; got {sdt!r}")
    if head_dim != 128 and sdt != "f32":
        raise ValueError(
            f"at head_dim={head_dim} IntDecodeSm100 uses an f32 scale pool; "
            f"got sdtype={sdt!r}")
    qoff = quant_block_offsets(kq, vq, N=page_size, D=head_dim)
    return kq, vq, sdt, qoff


def scale_pool_dtype_of(page_size, head_dim, kernel_kwargs=None):
    """Return the torch dtype required by the resolved SM100 scale-pool config."""
    _, _, sdt, _ = quant_cfg_of(page_size, head_dim, kernel_kwargs)
    return torch.bfloat16 if sdt == "bf16" else torch.float32


def resolve_fold_tiling(k_quant, v_quant, *, sequence_tile=None, o_stages=None):
    """`IntDecodeSm100.resolve_tiling`, importable without naming the kernel
    module. Drivers and the split planner MUST go through this rather than
    duplicating the table -- a driver that hardcodes 256 while the kernel
    resolves 128 is exactly the planner/kernel disagreement this indirection
    exists to prevent."""
    from .int_fold_decode import IntDecodeSm100

    return IntDecodeSm100.resolve_tiling(
        k_quant, v_quant, sequence_tile=sequence_tile, o_stages=o_stages)


#: The f32 bit pattern of 1e-30. The P-scale exponent takes the largest s_v at
#: no less than this value, as the host formula takes max(max s_v, 1e-30): no
#: stored s_v is that small (a quantizer's scale is at least 1e-8 / qmax), and a
#: pool that holds no page maximum gets lp = 108.
PSCALE_VMAX_FLOOR_BITS = 0x0DA24260

#: Bins of the count table `v_max_hist`, one per value `vmax_bucket` can take.
VMAX_BINS = 256


def vmax_bucket(bits, minimum=min):
    """The exponent bin of a page maximum, computed from its f32 bit pattern in
    integers, so that lp = floor(log2(448 / x)) = `pscale_lp_of_bucket(b)`.

    For a positive normal x = 1.m * 2^E, 448 = 1.75 * 2^8 gives lp = 8 - E, less
    one when 1.m is above 1.75, which is a mantissa field above 0x600000. Adding
    0x1FFFFF to the bit pattern carries into the exponent field exactly when the
    mantissa field is above 0x600000, so b = (bits + 0x1FFFFF) >> 23 gives both
    terms and lp = 135 - b, where 135 = 8 + 127 removes the exponent bias. Exact
    for every normal x (`tests/test_sm100_scale_dtype_host.py` checks it against
    the float formula at every binade and at the boundary mantissas). b grows with
    x, so the highest bin of a set of maxima is the bin of the largest of them.

    The exponent this gives equals the float formula for finite positive scales,
    which is every scale a quantizer stores. Invalid input is not rejected: an
    +inf or NaN maximum lands in bin 255, beside the largest finite values, and
    gives the finite lp = -120, where the float formula raises ValueError.

    The sign bit is masked off first, so -0.0 lands in bin 0 beside +0.0, and the
    pattern is capped at +inf's before the add, so every 32-bit pattern lands in
    [0, 255] with no signed overflow: the cache writer indexes the table with this
    value. `minimum` is the two-argument minimum of the caller's integer type:
    Python's `min` (the default), `cutlass.min` in a kernel, `torch.clamp_max` on
    a tensor. The writer, the decode, `v_max_hist_of` and the CPU tests share this
    one function.
    """
    return (minimum(bits & 0x7FFFFFFF, 0x7F800000) + 0x1FFFFF) >> 23


#: The bin of 1e-30 (27). The decode takes lp at max(highest nonzero bin, this).
PSCALE_FLOOR_BUCKET = vmax_bucket(PSCALE_VMAX_FLOOR_BITS)


def pscale_lp_of_bucket(bucket):
    """lp = floor(log2(448 / x)) for every positive normal x in bin `bucket` of
    `vmax_bucket`. Python ints and DSL Int32 alike."""
    return 135 - bucket


#: The P-scale exponent of a build that reads no table (channel-wise V, where P
#: carries no s_v and max P = 1, and the fold-off ablation): floor(log2(448)).
#: `launch_fold` passes it as the kernel's `pscale_lp` launch scalar; the kernel's
#: "THE P-SCALE EXPONENT" comment says why it is not a constant in the kernel.
TABLE_FREE_PSCALE_LP = 8.0


def v_page_max_for_pool(mScale, k_quant=None, v_quant=None, *,
                        page_size=POOL_PAGE_SIZE, head_dim=POOL_HEAD_DIM):
    """The per-(page, KV head) V-scale maxima of a scale pool.

    Returns uint32 [NP, HKV] on the pool's device: the bit pattern of the largest
    s_v of each page, over every position of the page, widened to f32 first when
    the pool is bf16 (the value the decode multiplies). A maximum that is not
    positive (a page of zeros, a -0.0, or a value no quantizer produces) becomes
    +0.0, which means "no maximum"; so does a NaN maximum, so a table rebuilt from
    a pool with a NaN scale counts nothing for that page. The host formula
    max(max s_v, 1e-30) never lets such a value raise the exponent, and the cache
    writer never stores a sign-set cell, so every cell stays in one sign-free
    domain. None for channel-wise V, whose P-scale exponent is the constant 8. This
    is the host-side counterpart of what the cache writer keeps, for pools built on
    the host (test fixtures, `QuantKVCache._adopt`); `v_max_hist_of` turns it into
    the count table the decode reads. No host sync.

    A 4-D pool is the legacy [NP, HKV, N, 4] interleaved block; a 3-D pool is the
    flex flat sectioned block and NEEDS its (k_quant, v_quant) to locate the V
    ring section.
    """
    from ..quant_layout import quant_block_offsets

    if mScale.dim() == 4:
        sv = mScale[..., 2]
    else:
        if k_quant is None or v_quant is None:
            raise ValueError(
                "a flat sectioned (3-D) scale pool needs k_quant/v_quant to locate "
                "the V ring section")
        off = quant_block_offsets(k_quant, v_quant, N=page_size, D=head_dim)
        if off["legacy"]:
            sv = mScale.reshape(*mScale.shape[:2], page_size, 4)[..., 2]
        elif v_quant[0] == "channel":
            return None
        else:
            ring = mScale[..., off["v_ring"]:off["v_ring"] + 2 * page_size]
            sv = ring.reshape(*ring.shape[:-1], page_size, 2)[..., 0]
    m = sv.float().amax(dim=-1)
    m = torch.where(m > 0, m, torch.zeros_like(m))
    return m.contiguous().view(torch.uint32)


def v_max_hist_of(v_page_max, out=None):
    """The count table of a set of page maxima, as the cache writer keeps it.

    `v_page_max` is uint32 [NP, HKV] (`v_page_max_for_pool`, or a cache's own).
    Returns uint32 [VMAX_BINS] on its device: bin b counts the cells whose
    `vmax_bucket` is b, over every cell that holds a maximum, which is every cell
    whose pattern is not +0.0 or -0.0. The decode turns the highest nonzero bin
    into the P-scale exponent of the largest maximum. `out`, when given, is
    overwritten in place and returned (`QuantKVCache.recount_v_max`).

    Device only: no boolean indexing, no `bincount`, no `.item()`, so it neither
    waits for the device nor changes shape with the data, and a CUDA graph can
    capture it.
    """
    bits = v_page_max.view(torch.int32).long() & 0x7FFFFFFF
    bucket = vmax_bucket(bits, torch.clamp_max)
    if out is None:
        out = torch.empty(VMAX_BINS, dtype=torch.uint32, device=v_page_max.device)
    counts = out.view(torch.int32)
    counts.zero_()
    counts.index_add_(0, bucket.reshape(-1), (bits != 0).reshape(-1).to(torch.int32))
    return out


def reads_v_max_hist(page_size, head_dim, kernel_kwargs=None):
    """Whether a `compile_fold(**kernel_kwargs)` build reads `v_max_hist`.

    It does exactly when its P operand carries s_v, which is the fold
    (`fold_scale_zero`, on unless a caller turns it off) with token-wise V.
    Channel-wise V and the fold-off ablation take the constant exponent
    `TABLE_FREE_PSCALE_LP` as a launch scalar instead.
    """
    kkw = kernel_kwargs or {}
    _, vq, _, _ = quant_cfg_of(page_size, head_dim, kkw)
    return reads_exponent_table(vq[0], kkw.get("fold_scale_zero", 1))


def v_max_hist_for_pool(mScale, kernel_kwargs=None, *, page_size=POOL_PAGE_SIZE,
                        head_dim=POOL_HEAD_DIM):
    """The count table a `compile_fold(**kernel_kwargs)` build reads for a host-built
    scale pool, or None when the build reads none (`reads_v_max_hist`). What
    `run_fold`'s callers pass as `v_max_hist`."""
    if not reads_v_max_hist(page_size, head_dim, kernel_kwargs):
        return None
    kq, vq, _, _ = quant_cfg_of(page_size, head_dim, kernel_kwargs)
    return v_max_hist_of(v_page_max_for_pool(mScale, kq, vq, page_size=page_size,
                                             head_dim=head_dim))


# ---------------------------------------------------------------------------
# IntDecodeSm100 compile
# ---------------------------------------------------------------------------


def _refuse_cuda12_dsl_backend(where="compile_fold"):
    """Blackwell builds need the DSL's CUDA 13 backend (`fosca[cu13]`).

    nvidia-cutlass-dsl 4.8.0's CUDA 12 backend vendors ptxas 12.9, which on B200
    leaves output elements of this kernel unwritten at the B = 1 cell and at a
    channel-64 K cell, while its CUDA 13 backend (ptxas 13.4) is correct. Blackwell
    runs CUDA 13; the cu12 extra is for Hopper. Called before tracing, so it costs
    nothing at launch and a cached object, keyed on the DSL's CUDA version, never
    reaches it. `where` names the compile in the message: `compile_fold`, or the
    Blackwell cache writer's.
    """
    from cutlass.base_dsl import version_info

    major = getattr(version_info.CUDA_VERSION, "major", None)
    if major is not None and major < 13:
        raise RuntimeError(
            f"{where}: the SM100 kernel needs nvidia-cutlass-dsl's CUDA 13 backend "
            f"(install fosca[cu13]); this DSL loaded CUDA "
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
    has_q_descale=False,
    has_lse=False,
    graph_split=False,
    **kernel_kwargs,
):
    """Compile `IntDecodeSm100` with symbolic batch/page-count shapes
    (wrapper-style fake tensors). The kernel's only input contract is the
    packed LUT pools `mKq`/`mVq`/`mScale`. `debug_s` adds the matching
    gmem debug tensor (`mDbgS`).

    ONE compile serves every batch size and every KV length — the batch, page
    count and kv_splits dims are `cute.sym_int()`. Memoise on the remaining
    (static) arguments; a trace costs 10 to 13 s on B200, and
    `fosca.kernel_cache` serves a repeat of the same trace from disk in
    milliseconds.

    `bits_tuning`: the per-bits ctor knobs to apply (default `BITS_TUNING`).
    Overridable so an A/B driver can compile a control arm WITHOUT the
    promoted tuning; pass `{}` for "no tuning". Whatever is passed is also what
    the post-construction guard below checks, so a caller cannot silently get a
    kernel that did not receive the knobs it asked for.

    `kernel_out`: optional dict, filled with `{"kernel": <the constructed
    IntDecodeSm100>}` before the compile. This function returns the compiled
    CALLABLE, which carries none of the resolution the constructor did, so a caller
    has no other way to ask whether the kernel it just built actually reached the
    configuration it asked for.

    `knobs_out`: optional dict, filled with the tuned knobs READ BACK OFF THE
    CONSTRUCTED KERNEL. The guard below already asserts on those values, but a
    memoising caller (the wrapper) only traces once and would never see the
    assert again on a cache hit — so it stores this and re-checks per plan().

    `has_q_descale`: compile the variant that reads an fp32 `q_descale` [B, HQ]
    (any strides) and multiplies each query row's scores by it. False compiles the
    kernel without the per-row factor, where the host folds a per-tensor descale
    into `scale_s`. The variant follows from whether a launch passes `q_descale`;
    it is not a tuning knob.

    `has_lse`: compile the variant that returns the natural-log log-sum-exp of every query
    row into an fp32 `[B, 1, HQ]` tensor (`run(lse=)`), the same way. An unsplit build
    stores it from its reduction warp; a split build hands it to the reduction kernel only, so
    its decode kernel is the one compiled without it. The atomic reduction is refused: its
    store is base 2 and predicated on the tile alone (`GqaDecode.reduction_cluster`).

    The compiled kernel takes `v_max_hist` (uint32 [VMAX_BINS]) exactly when
    `reads_v_max_hist(page_size, head_dim, kernel_kwargs)`.

    `graph_split`: compile the graph split build, the split build (`reduction="kernel"`)
    that CUDA-graph mode captures. It launches a fixed grid of W CTAs, one per slot of a
    work list the host rewrites before each replay, and takes each CTA's request, KV head,
    split and length from its slot (`IntDecodeSm100.call_graph`, `mWork`); the reduction
    kernel merges each request's own split count. `launch_fold_graph` launches it. The
    default (False) compiles the eager build.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import Float32, Int32
    from .int_fold_decode import IntDecodeSm100

    in_dtype = in_dtype or cutlass.Float8E4M3FN
    out_dtype = out_dtype or cutlass.BFloat16
    acc_dtype = cutlass.Float32
    tuning = BITS_TUNING if bits_tuning is None else bits_tuning
    if has_lse and reduction == "atomic":
        raise ValueError(
            "compile_fold: has_lse with reduction='atomic' is refused; that reduction's LSE "
            "store is base 2 and has no head predicate (GqaDecode.reduction_cluster)")
    if graph_split and reduction != "kernel":
        raise ValueError(
            f"compile_fold: graph_split is the split build, reduction='kernel'; got "
            f"reduction={reduction!r}")

    if q_head_tile is None:
        q_head_tile = blk_tile_n_of(num_qo_heads, num_kv_heads)

    # The keys that decide which inputs the build takes and which pool it reads
    # (`reads_v_max_hist`, `quant_cfg_of`) are the caller's kernel arguments, which
    # the pool helpers are handed too; through the tuning they would reach the
    # constructor but not those helpers, which would then hand the launch a table
    # the build does not take, or none where it takes one.
    selectors = sorted(set(tuning.get(bits, {}))
                       & {"fold_scale_zero", "k_quant", "v_quant", "sdtype"})
    if selectors:
        raise TypeError(
            f"compile_fold: {selectors} decide the kernel's inputs and its pool, "
            "not its tuning; pass them as kernel keyword arguments, not through "
            "bits_tuning")

    # Apply the per-bits tuning unless the caller overrode it. This MUST precede
    # construction. Placed after it, it is dead code, and the only symptom is
    # timings that come back bit-for-bit unchanged, so measure the wiring rather
    # than assuming it.
    kernel_kwargs = dict(kernel_kwargs)
    for _k, _v in tuning.get(bits, {}).items():
        if _k not in kernel_kwargs:
            kernel_kwargs[_k] = _v

    # A RENAMED SPELLING MUST RAISE BY NAME, and `**kernel_kwargs` is where one
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
    # REFUSED OUTRIGHT, not renamed. The constructor splat would raise on its own,
    # but these messages say why the name is refused.
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
    # REFUSED OUTRIGHT, and NOTHING REPLACED IT. `identity_dequant` swapped the
    # input contract for plain e4m3 `k_bshd`/`v_bshd` tensors, turned the affine
    # fold off everywhere, and left the dequant warpgroup relaying raw bytes through
    # shared memory. The packed pools mKq/mVq/mScale are the kernel's only input
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
    # The graph split build decodes one query token per request: its work entries name a
    # (request, KV head) pair, and its partials and merge hold one prediction row, so its
    # prediction length is the static 1 and a call with more rows is refused by the compiled
    # function's argument binding. Every other build keeps it symbolic.
    sym_prediction = 1 if graph_split else cute.sym_int()

    fct = cute.runtime.make_fake_compact_tensor
    seqlens_fake = fct(Int32, (sym_batch,), assumed_align=16)
    table_offsets_fake = fct(Int32, (sym_batch,), assumed_align=16)
    page_table_fake = fct(Int32, (sym_total_logical_pages,), assumed_align=16)
    q_fake = fct(in_dtype, (sym_batch, sym_prediction, num_qo_heads, head_dim),
                 stride_order=(3, 2, 1, 0), assumed_align=16)
    o_fake = fct(out_dtype, (sym_batch, sym_prediction, num_qo_heads, head_dim),
                 stride_order=(3, 2, 1, 0), assumed_align=16)
    l_fake = (fct(acc_dtype, (sym_batch, sym_prediction, num_qo_heads),
                  stride_order=(2, 1, 0), assumed_align=16) if has_lse else None)

    k_row_bytes = head_dim * bits // 8
    # The kernel stores V exactly like K where `v_token_rows` holds; every
    # other configuration keeps channel rows.
    if fmha.v_token_rows:
        v_rows, v_row_bytes = page_size, k_row_bytes
    else:
        v_rows, v_row_bytes = head_dim, page_size * bits // 8
    mKq_fake = fct(cutlass.Uint8, (sym_virtual_pages, num_kv_heads, page_size,
                                   k_row_bytes),
                   stride_order=(3, 2, 1, 0), assumed_align=16)
    mVq_fake = fct(cutlass.Uint8, (sym_virtual_pages, num_kv_heads, v_rows,
                                   v_row_bytes),
                   stride_order=(3, 2, 1, 0), assumed_align=16)
    kq_cfg, vq_cfg, sdt, qoff = quant_cfg_of(page_size, head_dim, kernel_kwargs)
    sdt_cutlass = cutlass.BFloat16 if sdt == "bf16" else cutlass.Float32
    if qoff["legacy"]:
        mScale_fake = fct(sdt_cutlass,
                          (sym_virtual_pages, num_kv_heads, page_size, 4),
                          stride_order=(3, 2, 1, 0), assumed_align=16)
    else:
        # flex configs: flat sectioned block, sdtype elements
        mScale_fake = fct(sdt_cutlass,
                          (sym_virtual_pages, num_kv_heads, qoff["block_tot"]),
                          stride_order=(2, 1, 0), assumed_align=16)

    mDbgS_fake = (fct(cutlass.Float32, (sequence_tile, q_head_tile),
                      stride_order=(1, 0), assumed_align=16) if debug_s else None)
    # The count table of page maxima, one fixed size; the cache allocates it
    # 16-byte aligned.
    v_max_hist_fake = (
        fct(cutlass.Uint32, (VMAX_BINS,), assumed_align=16)
        if reads_v_max_hist(page_size, head_dim, kernel_kwargs) else None)
    q_descale_fake = (
        cute.runtime.make_fake_tensor(
            cutlass.Float32, (sym_batch, num_qo_heads),
            stride=(cute.sym_int(), cute.sym_int()), assumed_align=4)
        if has_q_descale else None)

    if reduction == "kernel":
        o_partial_fake = fct(
            acc_dtype, (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads, head_dim),
            stride_order=(4, 3, 2, 1, 0), assumed_align=16)
        m_partial_fake = fct(acc_dtype, (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads),
                             stride_order=(3, 2, 1, 0), assumed_align=16)
        l_partial_fake = fct(acc_dtype, (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads),
                             stride_order=(3, 2, 1, 0), assumed_align=16)
    else:
        o_partial_fake = m_partial_fake = l_partial_fake = None

    # The graph split build's work list (int32 [W, 4], one 16-byte entry per CTA), the
    # engine page table's row stride, and each request's split count (int32 [B]).
    graph_args = ()
    if graph_split:
        from ..graph_plan import ENTRY_WORDS

        graph_args = (
            fct(Int32, (cute.sym_int(), ENTRY_WORDS), stride_order=(1, 0),
                assumed_align=16),
            Int32(1),
            fct(Int32, (sym_batch,), assumed_align=4),
        )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    # Opt-level default is bits-scoped, and measured: int2 is faster at 2 and int4
    # regresses hard at 2, so int4 keeps 3. An env override wins, which is what the
    # correctness tests use (OPT_LEVEL=0).
    opt_level = os.environ.get(
        "CUTE_DSL_OPT_LEVEL", "2" if bits == 2 else "3")
    # GUARD: prove the per-bits tuning actually reached the constructed kernel.
    # Applying it as a dead statement AFTER construction is a silent no-op, whose
    # only symptom is timings that come back bit-for-bit identical. A measured
    # optimisation that the object never receives is a silent failure, so assert
    # rather than trust.
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
        # The graph split build compiles its own entry point, which adds the work list's three
        # arguments, so that every other build's compiled function keeps `launch_fold`'s list.
        return cute.compile(
            fmha.call_graph if graph_split else fmha,
            Int32(1),
            seqlens_fake,
            None if graph_split else table_offsets_fake,
            page_table_fake,
            q_fake,
            o_fake,
            l_fake,
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
            v_max_hist_fake,
            q_descale_fake,
            None if v_max_hist_fake is not None else Float32(TABLE_FREE_PSCALE_LP),
            *graph_args,
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
            "v_max_hist": v_max_hist_fake is not None,
            "has_q_descale": bool(has_q_descale),
            "has_lse": bool(has_lse),
            "graph_split": bool(graph_split),
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
                q_bshd, o_bshd, o_partial=None, l_partial=None, m_partial=None, scale_s=1.0, scale_o=1.0,
                mKq=None, mVq=None, mScale=None, v_max_hist=None,
                q_descale=None, lse=None, dbg=None, enable_pdl=True):
    """THE launch site for a `compile_fold`-compiled kernel.

    Everything (`run_fold`, `PagedDecodeWrapper.run`) funnels
    through here, so the 21-argument positional contract exists once. All
    tensors are torch tensors; scalars are Python floats/ints.

    `v_max_hist`: uint32 [VMAX_BINS], the count table of the cache's page maxima
    (`QuantKVCache.v_max_hist`, or `v_max_hist_for_pool` for a host-built pool),
    exactly when the build reads one (`reads_v_max_hist`). `q_descale`: fp32 [B, HQ],
    any strides, exactly when the kernel was compiled with `has_q_descale=True`;
    `scale_s` is then the model's softmax scale, with no descale folded in. `lse`: fp32
    `[B, 1, HQ]`, exactly when the kernel was compiled with `has_lse=True`, receives the
    natural-log log-sum-exp of each query row. The
    compiled function's argument binding refuses a tensor where it was compiled
    for None and the reverse. A build that takes no `v_max_hist` gets the exponent
    `TABLE_FREE_PSCALE_LP` as its last argument; no caller passes it.

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
            lse,
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
            v_max_hist,
            q_descale,
            None if v_max_hist is not None else Float32(TABLE_FREE_PSCALE_LP),
        )


def launch_fold_graph(compiled, *, seqlens, page_table, q_bshd, o_bshd, o_partial, l_partial,
                      m_partial, work, split_counts, max_pages, scale_s=1.0, scale_o=1.0,
                      mKq=None, mVq=None, mScale=None, v_max_hist=None, q_descale=None,
                      lse=None, enable_pdl=True):
    """THE launch site for the graph split build (`compile_fold(graph_split=True)`), kept apart
    from `launch_fold` so that eager's launch is untouched by graph mode.

    `work` and `split_counts` are the two views `graph_work_views` cuts from the work list the
    host writes before each replay, and `max_pages` the row stride of the engine's int32
    [B, max_pages] page table, which `page_table` holds flat. `seqlens` is any int32 [B] tensor:
    the build reads its lengths from `work` and takes this one only for the shared-memory map of
    the split build it shares code with. The partials are [S, B, 1, HQ, D] (fp32; `l_partial`,
    `m_partial` without D), with S at least the largest split count in `work`. The other
    arguments are `launch_fold`'s. It is the launch a CUDA graph captures: nothing here depends
    on the plan's values.
    """
    import tvm_ffi
    from cutlass.cute.typing import Float32, Int32

    with tvm_ffi.use_torch_stream():
        compiled(
            Int32(1),      # kv_splits: each CTA's split count comes from its work entry
            seqlens,
            None,          # table_offsets: the build computes b * max_pages
            page_table,
            q_bshd,
            o_bshd,
            lse,
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
            None,          # dbg
            v_max_hist,
            q_descale,
            None if v_max_hist is not None else Float32(TABLE_FREE_PSCALE_LP),
            work,
            Int32(max_pages),
            split_counts,
        )


def graph_work_views(buf, slots, batch):
    """`(work, split_counts)`: the graph split build's two inputs, as views of the work list
    `fosca.graph_plan` writes (`Bucket.write`), one int32 device buffer of
    `graph_plan.words(slots, batch)` words that the host plan reaches with one copy.

    `work` is [slots, 4], one 16-byte entry per CTA of the launch's grid; `split_counts` is
    [batch], each request's k, from word `graph_plan.k_offset(slots)` on. The format is the
    planner's, so the two cannot drift."""
    from ..graph_plan import ENTRY_WORDS, k_offset, words

    n = words(slots, batch)
    if buf.dtype != torch.int32 or buf.dim() != 1 or buf.numel() < n:
        raise ValueError(f"graph_work_views: need a flat int32 buffer of at least {n} words; "
                         f"got {buf.dtype} {tuple(buf.shape)}")
    if buf.data_ptr() % 16:
        raise ValueError("graph_work_views: the buffer must start 16-byte aligned")
    k0 = k_offset(slots)
    return buf[:k0].view(slots, ENTRY_WORDS), buf[k0:k0 + batch]


def alloc_fold_workspace(p, kv_splits, reduction):
    """Pre-allocate everything `run_fold` would otherwise allocate per call.

    Pass the result as `run_fold(..., ws=...)` to keep ALLOCATION out of a
    timed region: four `torch.empty` calls per launch at reduction="kernel", one
    at "none". Every external baseline allocates in its own `build()`, so this
    path has to as well or the comparison protocol is not uniform. Nothing in the
    workspace needs initializing: each split stores its partials, and the
    reduction kernel takes the final maximum from the per-split ones.
    """
    dev = torch.device("cuda")
    b, h_q = p["b"], p["h_q"]
    d = p.get("d", POOL_HEAD_DIM)
    out = torch.empty(b, 1, h_q, d, device=dev, dtype=torch.bfloat16)
    if reduction == "kernel":
        return dict(
            out=out,
            o_partial=torch.empty(kv_splits, b, 1, h_q, d, device=dev, dtype=torch.float32),
            l_partial=torch.empty(kv_splits, b, 1, h_q, device=dev, dtype=torch.float32),
            m_partial=torch.empty(kv_splits, b, 1, h_q, device=dev, dtype=torch.float32),
        )
    return dict(out=out, o_partial=None, l_partial=None, m_partial=None)


def run_fold(compiled, p, kv_splits, reduction, scale_s, scale_o=1.0, *,
             pools=None, dbg=None, v_max_hist=None, q_descale=None, ws=None):
    """Launch a `compile_fold`-compiled kernel on a problem dict.

    `pools=(mKq,mVq,mScale)` are the packed quantized pools the kernel reads.
    Returns the squeezed [b, h_q, d] bf16 output.

    `v_max_hist` is the count table the build reads, which a caller derives from
    its pool and its kernel arguments with `v_max_hist_for_pool` (None where
    `reads_v_max_hist` is false). `q_descale` goes to a kernel compiled with
    `has_q_descale=True`.

    `ws` is an `alloc_fold_workspace()` dict. Supply it to hoist allocation out
    of the call; omit it and the workspace is allocated per call.
    """
    if ws is None:
        ws = alloc_fold_workspace(p, kv_splits, reduction)
    out = ws["out"]
    mKq, mVq, mScale = pools if pools is not None else (None, None, None)
    launch_fold(
        compiled,
        kv_splits=kv_splits,
        seqlens=p["seq_lens"],
        table_offsets=p["table_offsets"],
        page_table=p["indices"],
        q_bshd=p["q_fp8"],
        o_bshd=out,
        o_partial=ws["o_partial"],
        l_partial=ws["l_partial"],
        m_partial=ws["m_partial"],
        scale_s=scale_s,
        scale_o=scale_o,
        mKq=mKq, mVq=mVq, mScale=mScale,
        v_max_hist=v_max_hist,
        q_descale=q_descale,
        dbg=dbg,
    )
    return out.squeeze(1)


def plan_fold(p, bits, *, sequence_tile=256):
    """(kv_splits, reduction) the SM100 planner would choose for this problem.

    Thin adapter from a problem dict (keys "b", "h_kv", "seq_lens") to
    `fosca._sm100.scheduler.plan_splits_sm100`, so the planner is
    reachable from the one place every fold launch goes through instead of
    being library code nobody calls.

    `PagedDecodeWrapper` does NOT go through here — it calls
    `plan_splits_sm100` directly from real plan() arguments. This entry point
    exists for benchmark drivers that speak problem dicts.
    """
    from .scheduler import plan_splits_sm100

    seq = p["seq_lens"]
    kv_len = int(seq.max().item()) if hasattr(seq, "max") else max(seq)
    return plan_splits_sm100(p["b"], p["h_kv"], kv_len, bits=bits,
                             sequence_tile=sequence_tile,
                             head_tiles=(p["h_q"] // p["h_kv"] + 7) // 8)
