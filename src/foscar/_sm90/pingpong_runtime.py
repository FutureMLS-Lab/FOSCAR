# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""Library-side compile/launch glue for ``IntPingPongDecodeSm90`` (SM90 quantized KV).

The Hopper sibling of ``_sm100/fold_runtime.py``: it is what
:class:`foscar.PagedDecodeWrapper` calls when ``plan(..., layout)`` runs on
an SM90 device, so the quantized path has ONE public entry point on both
architectures.

Validated on H100 SXM5.

TWO THINGS THAT ARE NOT BUGS AND LOOK LIKE THEM

  * The accuracy floor is ~2.4e-3, not SM100's ~1.8e-3, because ``mQ`` here is
    **bf16** where ``IntDecodeSm100`` takes fp8_e4m3. 2**-9 is the floor.
  * ``plan_splits`` can emit split counts outside ``_sm100``'s ``{1,2,4,8,16}``.
    It returns the occupancy-model pick (16 at B=1, KV 8k-32k); a coarser
    ``capacity // base`` candidate (33 at B=1/KV=32768) survives only as the
    fallback. Any count is legal, because the kernel ceil-partitions and tolerates
    empty tail splits, which the split-parity comparison exercises deliberately at
    up to 2x the tile count.

DESIGN NOTES THAT MATTER IF YOU CHANGE THIS
-------------------------------------------
* ``mCnt`` (int32 ``[B*HKV]``) is the fused-combine arrival counter. The kernel
  contract says: allocate zeroed ONCE, the reducing CTA rearms its own cell to
  0 each launch. So it must NOT be re-zeroed per call, and it must NOT be
  carved out of a buffer that something else writes. It is allocated in
  ``plan()`` and owned by the wrapper.
* ``fused_combine`` is a TRACE-TIME ctor flag, not a runtime one, and
  ``plan_fused_combine`` is its enable policy. It must therefore be decided
  before the compile — which is why the split plan happens first in ``plan()``,
  same ordering rule as the SM100 side (reduction mode is also compile-time
  there).
* When ``outer_splits > 1`` and the fused combine is OFF, a SECOND kernel
  (``SplitKCombine``) has to run after the decode. When it is ON, the decode
  kernel's last CTA does the merge and the second launch must be skipped —
  running both would double-reduce.
"""
import os
import re
import sys

import torch

#: The page/tile size the SM90 int kernels are built around (`page_size == 64` is
#: a ctor assert: the single-WG QK atom and the 2-D-half PV loop assume it).
POOL_PAGE_SIZE = 64
#: Grouped-head tile. `q_head_tile == 8` is a ctor assert (inline RS wgmma m64n8k16).
PG = 8
#: Code-generation options of NVVM, the DSL's LLVM backend. NVVM 23, which nvidia-cutlass-dsl
#: 4.8.0 ships, turns on a register rematerialization pass that recomputes loop-invariant values
#: inside the consumer tile loop, chiefly the shared-memory stage addresses and the wgmma
#: descriptors built from them, instead of keeping them in registers across iterations, and makes
#: machine CSE weigh register pressure.
#:
#: libNVVM keeps an option's value for the rest of the process once one compile sets it, so a
#: compile that leaves an option out inherits whatever an earlier compile in the same process
#: set. Every compile therefore states every option below, with an explicit value either way.
#: The rematerialization pass on or off:
NVVM_REMAT = "-Xllc -enable-new-nvvm-remat={}"
#: Restricts the pass, where it runs, to cheaper recomputations; with the pass off it has no
#: effect, so every decode compile states the same value (the other SM90 kernels state their own,
#: `NVVM_DEFAULTS`).
NVVM_REMAT_COST = "-Xllc -cost-threshold-remat=6"
#: Machine CSE that weighs register pressure on or off (`mcse_off`):
NVVM_MCSE_OFF = "-Xllc -rp-aware-mcse={}"
#: Loop strength reduction on or off (`lsr_off`):
NVVM_LSR_OFF = "-Xllc -disable-lsr={}"
#: The head dimensions at which the kernel stays free of spills with the remat pass off. Keeping
#: loop invariants in registers raises register pressure, and at head_dim 256 each consumer holds
#: twice the output accumulators, so there the pass stays on, restricted to cheap
#: recomputations. The setmaxnregs split cannot pay for the extra registers instead: at two CTAs
#: per SM the consumers' 104 is already the most the pool allows.
REMAT_OFF_HEAD_DIMS = (64, 128)
#: The head dimensions whose direct A/Bs named `lsr_off`'s classes; at any other head dimension
#: strength reduction stays on.
LSR_OFF_HEAD_DIMS = (128,)
#: What every SM90 kernel other than the decode compiles with under DSL 4.8, so that such a kernel
#: does not inherit a decode compile's values: the remat pass on, pressure-aware CSE and loop
#: strength reduction on, which are libNVVM's defaults, and a remat cost threshold of 10, a reset
#: value. libNVVM does not expose that threshold's default; 10 is the lowest value with which each
#: non-decode SM90 kernel tried compiles to the same binary as with no options at all, which does
#: not establish that it is the default.
NVVM_DEFAULTS = " ".join([NVVM_REMAT.format("true"), "-Xllc -cost-threshold-remat=10",
                          NVVM_MCSE_OFF.format("true"), NVVM_LSR_OFF.format("false")])
#: The nvidia-cutlass-dsl releases the SM90 decode compiles under, by major.minor: whether that
#: release takes NVVM options. 4.5 ships NVVM 20, which needs none and has no option class to
#: carry them; 4.8 ships NVVM 23 and `ExtraCompilerOpts`. Any other release is refused, because
#: the options are chosen for one NVVM and an unlisted release may read them differently.
DSL_TAKES_NVVM_OPTIONS = {(4, 5): False, (4, 8): True}
#: A whole normalized PEP 440 version without an epoch: release, then optional pre-release,
#: post-release, development release and local segments, each in its normalized spelling, so
#: numbers are ASCII digits without leading zeros and local segments are lowercase.
_N = r"(?:0|[1-9][0-9]*)"
_LOCAL_SEGMENT = rf"(?:[a-z0-9]*[a-z][a-z0-9]*|{_N})"
_NORMALIZED_VERSION = re.compile(
    rf"({_N})\.({_N})(?:\.{_N})*(?:(?:a|b|rc){_N})?(?:\.post{_N})?(?:\.dev{_N})?"
    rf"(?:\+{_LOCAL_SEGMENT}(?:\.{_LOCAL_SEGMENT})*)?")


def mcse_off(kern):
    """Whether `kern` compiles with machine CSE that ignores register pressure. It does except in
    the configuration classes in which a direct A/B of the two settings measured the
    pressure-aware CSE, NVVM 23's default, faster, which is no known mechanism, so the classes
    are named rather than derived: INT4 with K token-wise per half of the head dimension, INT4
    with channel-wise K whose group spans more than one tile and token-wise V, and a
    fused-combine kernel at a head dimension where the remat pass stays on."""
    if kern.fused_combine and kern.head_dim not in REMAT_OFF_HEAD_DIMS:
        return False
    if kern.bits == 4:
        if kern.k_axis == "token" and kern.k_group == kern.head_dim // 2:
            return False
        if (kern.k_axis == "channel" and kern.k_group > kern.page_size
                and kern.v_axis == "token"):
            return False
    return True


def lsr_off(kern):
    """Whether `kern` compiles with loop strength reduction off: the configuration classes at
    head dimension 128 in which two independent direct A/Bs of the two settings both measured it
    faster, which is no known mechanism, so the classes are named rather than derived. Each is
    channel-wise K whose group spans more than one tile, outside a fused-combine kernel, and
    either INT2 with token-wise V or INT4 with channel-wise V whose group is one tile. Classes
    that only one of the two A/Bs favoured off keep it on. Everywhere else strength reduction
    stays on, including head dimension 256, where the A/B favoured it on in four cells and off
    in one."""
    if (kern.head_dim not in LSR_OFF_HEAD_DIMS or kern.fused_combine
            or kern.k_axis != "channel" or kern.k_group <= kern.page_size):
        return False
    if kern.bits == 2:
        return kern.v_axis == "token"
    return kern.v_axis == "channel" and kern.v_group <= kern.page_size


def nvvm_options(kern):
    """The NVVM options an `IntPingPongDecodeSm90` compiles with under DSL 4.8, derived from the
    kernel, every one stated explicitly (see `NVVM_REMAT`).

    The head dimension decides the remat setting as a compile-time stand-in for spill risk,
    because the options must be fixed when the kernel is traced. `mcse_off` and `lsr_off`
    decide the other two."""
    remat_off = kern.head_dim in REMAT_OFF_HEAD_DIMS
    return " ".join([
        NVVM_REMAT.format("false" if remat_off else "true"),
        NVVM_REMAT_COST,
        NVVM_MCSE_OFF.format("false" if mcse_off(kern) else "true"),
        NVVM_LSR_OFF.format("true" if lsr_off(kern) else "false"),
    ])


def installed_dsl_version():
    """The installed nvidia-cutlass-dsl version string."""
    import importlib.metadata

    return importlib.metadata.version("nvidia-cutlass-dsl")


def dsl_release(version):
    """`(major, minor)` of a normalized version string, or None for anything else.

    The supported distributions publish their version in the normalized PEP 440 form, which
    `importlib.metadata` returns as written, so only that form is read:
    `4.8.0`, `4.8`, `4.8+cu129`, `4.8.0.post1` and `4.6.0.dev0` read as (4, 8), (4, 8), (4, 8),
    (4, 8) and (4, 6). A string that is not one whole normalized version, including one with an
    epoch, returns None, so the caller refuses it rather than guessing at its release."""
    match = _NORMALIZED_VERSION.fullmatch(version.strip())
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def dsl_takes_nvvm_options(version=None):
    """Whether the installed DSL (or `version`) compiles the decode kernel with NVVM options.

    Raises RuntimeError for a release `DSL_TAKES_NVVM_OPTIONS` does not list."""
    version = installed_dsl_version() if version is None else version
    key = dsl_release(version)
    if key not in DSL_TAKES_NVVM_OPTIONS:
        supported = ", ".join(f"{a}.{b}.x" for a, b in sorted(DSL_TAKES_NVVM_OPTIONS))
        raise RuntimeError(
            f"the SM90 decode compiles under nvidia-cutlass-dsl {supported}; the installed "
            f"release is {version}, whose NVVM the compile options were not chosen for")
    return DSL_TAKES_NVVM_OPTIONS[key]


def plan_pingpong(batch, num_kv_heads, kv_len, *, bits, head_dim, sm_count=None, head_tiles=1):
    """``(outer_splits, min_blocks_per_sm, fused_combine)`` for one launch.

    Thin adapter over ``_sm90/scheduler.plan_splits`` + ``plan_fused_combine``,
    so the wrapper reaches the calibrated H100 cost model for the quantized path.
    ``head_dim`` is required and forwarded, because the planner's model is fitted per
    head dimension.
    """
    from .scheduler import plan_fused_combine, plan_splits

    if not isinstance(kv_len, int):
        kv_len = max(int(v) for v in kv_len)
    outer, min_blocks = plan_splits(batch, num_kv_heads, kv_len, bits=bits,
                                    head_dim=head_dim, page_size=POOL_PAGE_SIZE,
                                    sm_count=sm_count, head_tiles=head_tiles)
    fused = plan_fused_combine(batch, num_kv_heads, outer, head_tiles=head_tiles,
                               head_dim=head_dim)
    return outer, min_blocks, fused


def block_table_2d(indptr, indices, batch_size, *, device=None):
    """Flat (indptr, indices) paging -> the ``[B, maxPages]`` table SM90 wants.

    The SM100 fold kernel indexes ``page_table[table_offsets[b] + gt]``; the
    SM90 kernel indexes ``mPT[(b, gt)]``. Same information, different shape —
    convert once at plan() time rather than per launch.
    """
    device = indices.device if device is None else device
    counts = (indptr[1:batch_size + 1] - indptr[:batch_size]).tolist()
    max_pages = max(1, max(counts) if counts else 1)
    bt = torch.zeros(batch_size, max_pages, dtype=torch.int32, device=device)
    for b, n in enumerate(counts):
        if n:
            start = int(indptr[b])
            bt[b, :n] = indices[start:start + n].to(torch.int32)
    return bt


def compile_pingpong(*, head_dim, num_qo_heads, num_kv_heads, bits,
                     min_blocks_per_sm=2,
                     fused_combine=False, num_stages=None,
                     sdtype=torch.float32, k_quant=None,
                     v_quant=None, page_size=POOL_PAGE_SIZE,
                     opt_level=None, out_dtype=torch.bfloat16, has_lse=False, graph=False):
    """Compile ``IntPingPongDecodeSm90`` with symbolic batch / page-count shapes.

    ``k_quant`` / ``v_quant`` pass unchanged to the constructor, which resolves a
    ``None`` to ``("token", head_dim)``; the default is spelled there and only there.

    ``out_dtype``, ``has_lse`` and ``graph`` select the variants `compile_constructed`
    describes; their defaults compile the default build.

    Only the dims a serving integration varies (batch, physical page count,
    pages-per-request, split count) are symbolic; the per-(page, head) code and
    scale block shapes are static because the kernel's TMA views assert on
    their exact strides.

    Returns the compiled callable. Cross-compiles on a non-Hopper box, where it
    must NOT then be invoked (sticky CUDA 209); launched and validated on H100 —
    see the module docstring.

    THE FOLD IS NOT OPTIONAL HERE. The kernel accepts `fold_scale_zero=False`, the
    classic-dequant ablation arm that measures what folding is worth, but this function
    does not take it and will not: a public compile API that offers a caller the choice
    between a fast path and a slow one lets a caller get the slow path by forgetting.
    Callers of this function get the fold, always. A test that must compile the ablation
    constructs the kernel itself and calls `compile_constructed`.
    """

    from .int_decode_pingpong import IntPingPongDecodeSm90

    if opt_level is None:
        opt_level = os.environ.get("CUTE_DSL_OPT_LEVEL", "3")

    # The GQA ratio is trace-time, so it is settled here rather than at launch.
    kern = IntPingPongDecodeSm90(
        head_dim, page_size, PG, num_stages=num_stages, bits=bits,
        sdtype=sdtype, min_blocks_per_sm=min_blocks_per_sm,
        fold_scale_zero=True, fused_combine=fused_combine,
        k_quant=k_quant, v_quant=v_quant,
        heads_per_kv=num_qo_heads // num_kv_heads,
    )
    return compile_constructed(kern, head_dim=head_dim, page_size=page_size,
                               num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                               opt_level=opt_level, out_dtype=out_dtype, has_lse=has_lse,
                               graph=graph)


def _compiler(opt_level, nvvm):
    """`(compile, options)`: a function that traces a kernel with its call arguments,
    positionally, the way `cute.compile` does, under the NVVM options `nvvm` when the installed
    DSL takes them and none when it does not, and the text that names its options in a kernel
    cache key.

    The function refuses every keyword argument: `cute.compile` takes compile-time overrides as
    keywords (`options=` and `pipeline=` among them) and would let one replace these options. The
    NVVM options are passed as option objects because `cute.compile(options=...)` splits its
    string on spaces and so cannot carry `-Xllc` together with its argument. Under a release
    `dsl_takes_nvvm_options` does not list, nothing compiles.
    """
    import cutlass.cute as cute

    options = f"--enable-tvm-ffi --opt-level {opt_level}"
    if dsl_takes_nvvm_options():
        from cutlass.base_dsl.compiler import ExtraCompilerOpts

        options += f" --nvvm-options '{nvvm}'"
        bracketed = cute.compile[(
            cute.EnableTVMFFI(True),
            cute.OptLevel(int(opt_level)),
            ExtraCompilerOpts(f'nvvm-options="{nvvm}"'),
        )]

        def compile_with(*args):
            return bracketed(*args)
    else:
        def compile_with(*args):
            return cute.compile(*args, options=options)

    def compile_fn(*args, **kwargs):
        if kwargs:
            raise TypeError(
                f"the SM90 compile derives its compile options and takes the kernel's call "
                f"arguments positionally; refused keyword(s): {', '.join(sorted(kwargs))}")
        return compile_with(*args)

    return compile_fn, options


def pingpong_compiler(kern, opt_level):
    """`(compile, options)` for the constructed `IntPingPongDecodeSm90` `kern`, with the NVVM
    options `nvvm_options` derives from it (`_compiler`).

    Every compile of the decode kernel goes through this, so that a test compiles the object the
    library ships.
    """
    return _compiler(opt_level, nvvm_options(kern))


def sm90_compiler(opt_level):
    """`(compile, options)` for an SM90 kernel other than the decode, such as `SplitKCombine` or
    the cache writer `QuantAppendSm90`, with every NVVM option a decode compile sets stated at
    a reset value (`NVVM_DEFAULTS`, `_compiler`), so that its binary does not depend on what
    the process compiled before it."""
    return _compiler(opt_level, NVVM_DEFAULTS)


#: The output dtypes the SM90 decode and its external combine are compiled for. bf16 is every
#: plan's default; f32 is the engine's opt-in (`plan(o_data_type=torch.float32)`), a separate
#: compile in which every output store keeps its f32 value instead of rounding it.
OUT_DTYPES = (torch.bfloat16, torch.float32)


def _out_element_type(out_dtype):
    """The CuTe element type of an `OUT_DTYPES` output dtype; any other dtype raises."""
    import cutlass

    if out_dtype not in OUT_DTYPES:
        raise ValueError(f"the SM90 decode writes {list(OUT_DTYPES)}; got {out_dtype}")
    return cutlass.BFloat16 if out_dtype == torch.bfloat16 else cutlass.Float32


def compile_constructed(kern, *, head_dim, num_qo_heads, num_kv_heads,
                        page_size=POOL_PAGE_SIZE, opt_level=None,
                        out_dtype=torch.bfloat16, has_lse=False, graph=False):
    """Compile an ALREADY-CONSTRUCTED ``IntPingPongDecodeSm90`` against symbolic shapes.

    Three variants follow from what a caller asks for, two of `plan()` and `run()` (the output
    dtype and the LSE) and one of the wrapper's mode, never from a knob, and each is its own
    compile and its own kernel-cache entry. The
    defaults compile the default build, which the bitwise comparison holds to a reference build
    at the builds it compiles: every layout, scale dtype and combine form at 32/8 and 64/8
    heads, plus its GQA and head dimension 256 rows. Head counts are compile keys, so that is a
    bounded claim, not one about every head pair. The variants:

      * ``out_dtype`` (`OUT_DTYPES`): the output tensor's dtype, which every output store takes.
      * ``has_lse``: the fused-combine build that also stores the request's merged LSE into an
        extra f32 ``[B, HQ]`` argument (``mLseOut``). Only a fused-combine kernel takes it,
        because every other build already returns its LSE through ``mLSE``.
      * ``graph``: the graph split build (`IntPingPongDecodeSm90.call_graph`), which a CUDA-graph
        bucket that can split launches: a 1-D grid of one wave of CTAs reading the step's work
        list, an int32 ``[W, 4]`` argument (`graph_plan`), in place of the grid's split count
        and of ``mSL``. With ``has_lse`` it is the fused build that also returns the LSE of a
        request of one split. Eager plans never compile it.

    Separate from `compile_pingpong` so that the fold remains non-optional on the
    public entry point while a test that must compile the classic-dequant ablation
    still can. The rule it serves: a caller must not be able to select the slow
    path by passing a flag, so the flag is not on the API -- a caller that genuinely
    wants a non-default kernel builds that kernel itself and brings it here.

    Besides `compile_pingpong`, its callers are the tests that compile a kernel the
    public entry point does not build, such as the fold-off ablation arm or a kernel
    with the int2 phase lock switched off, and those that need the constructed kernel
    object itself, which `compile_pingpong` does not return.
    """
    # THE KERNEL AND THE TENSORS MUST AGREE ON D. `kern` is already built and carries
    # its own head_dim; the `head_dim` argument here only sets the fake Q and output
    # extents. Nothing else ties the two together, so a D=128 kernel could be compiled
    # against D=64 tensors: the traced code would still write both 64-channel halves,
    # past the end of a 64-wide row. The constructor guard cannot see this, because
    # both values are individually legal.
    if head_dim != kern.head_dim:
        raise ValueError(
            f"compile_constructed was given head_dim={head_dim} for a kernel built at "
            f"head_dim={kern.head_dim}; the kernel traces against its OWN head_dim, so "
            f"the fake tensors must match it or the compiled kernel indexes past them")

    import cutlass  # noqa: F401
    import cutlass.cute as cute
    from cutlass.cute.typing import Float32, Int32

    if opt_level is None:
        opt_level = os.environ.get("CUTE_DSL_OPT_LEVEL", "3")


    from .serving_cache import code_pool_shape

    per_k = kern.k_code_bytes_per_thread
    per_v = kern.v_code_bytes_per_thread
    block_tot = kern.qoff["block_tot"]

    fct = cute.runtime.make_fake_compact_tensor
    sym_batch = cute.sym_int()
    sym_pages = cute.sym_int()          # physical pool pages (NP)
    sym_maxpg = cute.sym_int()          # block-table width (maxPagesPerSeq)
    sym_outer = cute.sym_int()          # split-K partitions
    sym_cnt = cute.sym_int()            # B * HKV fused-combine counters
    # The code pools in the order the kernel reads, asked of the kernel.
    pool_k = code_pool_shape(kern.code_pool_order, sym_pages, num_kv_heads, per_k)
    pool_v = code_pool_shape(kern.code_pool_order, sym_pages, num_kv_heads, per_v)

    mQ = fct(cutlass.BFloat16, (sym_batch, num_qo_heads, head_dim),
             stride_order=(2, 1, 0), assumed_align=16)
    mKq = fct(cutlass.Uint8, pool_k,
              stride_order=tuple(range(len(pool_k) - 1, -1, -1)), assumed_align=16)
    mVq = fct(cutlass.Uint8, pool_v,
              stride_order=tuple(range(len(pool_v) - 1, -1, -1)), assumed_align=16)
    if kern.qoff["legacy"]:
        # legacy interleaved pool [NP, HKV, N, 4] = (s_k, z_k, s_v, z_v)
        mScale = fct(kern.scale_element_type,
                     (sym_pages, num_kv_heads, page_size, 4),
                     stride_order=(3, 2, 1, 0), assumed_align=16)
    else:
        mScale = fct(kern.scale_element_type,
                     (sym_pages, num_kv_heads, block_tot),
                     stride_order=(2, 1, 0), assumed_align=16)
    mPT = fct(Int32, (sym_batch, sym_maxpg), stride_order=(1, 0), assumed_align=4)
    mSL = fct(Int32, (sym_batch,), assumed_align=4)
    mOp = fct(cutlass.Float32, (sym_outer, sym_batch, num_qo_heads, head_dim),
              stride_order=(3, 2, 1, 0), assumed_align=16)
    mOut = fct(_out_element_type(out_dtype), (sym_batch, num_qo_heads, head_dim),
               stride_order=(2, 1, 0), assumed_align=16)
    mLSE = fct(cutlass.Float32, (sym_outer, sym_batch, num_qo_heads),
               stride_order=(2, 1, 0), assumed_align=16)
    mCnt = fct(Int32, (sym_cnt,), assumed_align=16)
    if has_lse and not kern.fused_combine:
        raise ValueError("has_lse compiles the fused combine's merged-LSE store; a kernel "
                         "without the fused combine returns its LSE through mLSE")
    lse_out = ((fct(cutlass.Float32, (sym_batch, num_qo_heads), stride_order=(1, 0),
                    assumed_align=16),) if has_lse else ())
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_fn, options = pingpong_compiler(kern, opt_level)
    if graph:
        # one 16-byte slot per CTA of the wave (`graph_plan.ENTRY_WORDS`)
        mWork = fct(Int32, (cute.sym_int(), 4), stride_order=(1, 0), assumed_align=16)
        args = (kern.call_graph, mQ, mKq, mVq, mScale, mPT, mOp, mOut, mLSE, mCnt,
                Float32(1.0), Float32(1.0), Int32(num_kv_heads), stream_fake, mWork,
                lse_out[0] if has_lse else None)
    else:
        # The merged-LSE build compiles its own entry point, `call_with_lse`, which takes the
        # LSE after the stream, so the default build keeps its own argument list.
        args = (kern.call_with_lse if has_lse else kern,
                mQ, mKq, mVq, mScale, mPT, mSL, mOp, mOut, mLSE, mCnt,
                Float32(1.0),        # bmm1 (softmax scale)
                Float32(1.0),        # bmm2 (output scale)
                Int32(1),            # outer_splits
                Int32(num_kv_heads),
                Int32(1),            # batch
                stream_fake,
                *lse_out)

    def _trace():
        from ..kernel_cache import init_cuda_driver
        init_cuda_driver()
        return compile_fn(*args)

    # Through the persistent kernel cache; `_sm100/fold_runtime.compile_fold` carries
    # the reasoning. The kernel is already constructed here, so its resolved
    # attributes (`fold_scale_zero`, `fused_combine`, the quantization pair, the scale
    # dtype and the rest) are the key, together with the fake-tensor extents that are
    # not attributes.
    from .. import kernel_cache

    kernel_module = sys.modules[type(kern).__module__]
    sources = kernel_cache.source_closure(kernel_module.__file__, __file__)
    attributes = kernel_cache.simple_attributes(kern)
    # The constructor keeps the caller's `min_blocks_per_sm` (None or 2) beside the
    # value it resolved to, `_min_blocks`; only the resolved one shapes the trace.
    attributes.pop("min_blocks_per_sm", None)
    config = {
        "kernel": attributes,
        "trace": {
            "head_dim": head_dim,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "page_size": page_size,
            "per_k": per_k,
            "per_v": per_v,
            "code_pool_order": kern.code_pool_order,
            "block_tot": block_tot,
            "legacy": bool(kern.qoff["legacy"]),
            "scale_element_type": kern.scale_element_type,
            "out_dtype": str(out_dtype),
            "has_lse": bool(has_lse),
        },
    }
    if graph:
        config["trace"]["graph"] = True
    return kernel_cache.compile_or_load(
        kind=f"sm90_{type(kern).__name__}",
        compile_fn=_trace,
        sources=sources,
        config=config,
        options=options,
    )


def compile_splitk_combine(*, head_dim, opt_level=None, out_dtype=torch.bfloat16,
                           has_lse=False, graph=False):
    """Compile the external ``SplitKCombine`` used when ``outer_splits > 1``
    and the in-kernel fused combine is off. Validated on H100; on a non-Hopper
    box this compiles and must not be invoked.

    ``out_dtype`` (`OUT_DTYPES`) is the output's dtype, and ``has_lse`` compiles
    ``SplitKCombineLse``, which also stores the merged LSE into an f32 ``[B, HQ]`` argument
    after the output; each is its own compile and kernel-cache entry. ``graph`` compiles the
    graph split build's merge instead, ``SplitKCombineGraph`` or ``SplitKCombineGraphLse``,
    which reads each request's split count from an int32 ``[B]`` argument."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import Int32

    from .splitk_combine import (
        SplitKCombine, SplitKCombineGraph, SplitKCombineGraphLse, SplitKCombineLse)

    if opt_level is None:
        opt_level = os.environ.get("CUTE_DSL_OPT_LEVEL", "3")

    fct = cute.runtime.make_fake_compact_tensor
    sym_outer = cute.sym_int()
    sym_batch = cute.sym_int()
    sym_hq = cute.sym_int()
    mOp = fct(cutlass.Float32, (sym_outer, sym_batch, sym_hq, head_dim),
              stride_order=(3, 2, 1, 0), assumed_align=16)
    mLSE = fct(cutlass.Float32, (sym_outer, sym_batch, sym_hq),
               stride_order=(2, 1, 0), assumed_align=16)
    mOut = fct(_out_element_type(out_dtype), (sym_batch, sym_hq, head_dim),
               stride_order=(2, 1, 0), assumed_align=16)
    lse_out = ((fct(cutlass.Float32, (sym_batch, sym_hq), stride_order=(1, 0),
                    assumed_align=16),) if has_lse else ())
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_fn, options = sm90_compiler(opt_level)
    if graph:
        combine = (SplitKCombineGraphLse if has_lse else SplitKCombineGraph)(head_dim)
        args = (combine, mOp, mLSE, mOut, fct(Int32, (sym_batch,), assumed_align=16), *lse_out,
                Int32(1), Int32(1), stream_fake)
    else:
        combine = (SplitKCombineLse if has_lse else SplitKCombine)(head_dim)
        args = (combine, mOp, mLSE, mOut, *lse_out, Int32(1), Int32(1), Int32(1), stream_fake)

    def _trace():
        from ..kernel_cache import init_cuda_driver
        init_cuda_driver()
        return compile_fn(*args)

    from .. import kernel_cache
    from . import splitk_combine as _combine_module

    return kernel_cache.compile_or_load(
        kind=f"sm90_{type(combine).__name__}",
        compile_fn=_trace,
        sources=kernel_cache.source_closure(_combine_module.__file__, __file__),
        config={"kernel": kernel_cache.simple_attributes(combine),
                "trace": {"head_dim": head_dim, "out_dtype": str(out_dtype)}},
        options=options,
    )


def alloc_pingpong_workspace(batch, num_qo_heads, num_kv_heads, head_dim,
                             outer_splits, *, device):
    """``(o_partial, lse, counters)`` for one plan.

    ``counters`` is allocated ZEROED and must stay that way across launches —
    the kernel rearms its own cell (module docstring).
    """
    return dict(
        o_partial=torch.zeros(outer_splits, batch, num_qo_heads, head_dim,
                              device=device, dtype=torch.float32),
        lse=torch.full((outer_splits, batch, num_qo_heads), -float("inf"),
                       device=device, dtype=torch.float32),
        counters=torch.zeros(batch * num_kv_heads * ((num_qo_heads // num_kv_heads + 7) // 8),
                             device=device, dtype=torch.int32),
    )


def launch_pingpong(compiled, combine, *, q_bhd, mKq, mVq, mScale, block_table,
                    seq_lens, o_partial, out_bhd, lse, counters,
                    scale_s, scale_o, outer_splits, num_kv_heads, batch, lse_out=None):
    """Launch the decode (+ the external combine when supplied).

    ``lse`` is the split LSE buffer the decode writes (`[OUTER, B, HQ]`; a one-split caller may
    pass its own `[B, HQ]` LSE viewed `[1, B, HQ]`, which the decode then fills directly).
    ``lse_out``, f32 ``[B, HQ]``, is where a split launch returns the merged LSE: to the decode
    when it fuses the combine (compiled with ``has_lse``), else to the external combine
    (``SplitKCombineLse``).

    Validated on H100 SXM5 at every split count and both combine mechanisms.

    ``combine`` must be None when the kernel was compiled with
    ``fused_combine=True`` or when ``outer_splits == 1``; passing it anyway
    would double-reduce. The caller (`PagedDecodeWrapper`) owns that choice —
    see ``plan_pingpong``.

    ``mKq``/``mVq`` must be C-contiguous and in the kernel's ``code_pool_order``;
    a row-major ``[NP, HKV, 128, per]`` pool is refused here by name, before the
    compiled kernel's own argument check would refuse it by rank.
    """
    import tvm_ffi
    from cutlass.cute.typing import Float32, Int32

    from .int_decode_pingpong import IntPingPongDecodeSm90
    from .serving_cache import require_code_pool

    for name, pool in (("mKq", mKq), ("mVq", mVq)):
        require_code_pool(name, pool.shape, IntPingPongDecodeSm90.code_pool_order,
                          stride=pool.stride())

    decode_lse_out = (lse_out,) if lse_out is not None and combine is None else ()
    combine_lse_out = (lse_out,) if lse_out is not None and combine is not None else ()
    with tvm_ffi.use_torch_stream():
        compiled(q_bhd, mKq, mVq, mScale, block_table, seq_lens,
                 o_partial, out_bhd, lse, counters,
                 Float32(scale_s), Float32(scale_o), Int32(outer_splits),
                 Int32(num_kv_heads), Int32(batch), *decode_lse_out)
        if combine is not None:
            combine(o_partial, lse, out_bhd, *combine_lse_out, Int32(outer_splits),
                    Int32(batch), Int32(q_bhd.shape[1]))


def launch_pingpong_graph(compiled, combine, *, q_bhd, mKq, mVq, mScale, block_table, o_partial,
                          out_bhd, lse, counters, work, splits, scale_s, scale_o, num_kv_heads,
                          lse_out=None):
    """Launch the graph split build (`compile_constructed(graph=True)`) and, when supplied, its
    external combine (`compile_splitk_combine(graph=True)`).

    ``work`` is the bucket's device work list viewed int32 ``[W, 4]`` and ``splits`` its k_b
    words viewed int32 ``[B]`` (`graph_plan`); ``o_partial`` and ``lse`` hold the bucket's
    partial capacity of splits. ``lse_out``, f32 ``[B, HQ]``, goes to the fused decode, which
    stores it for every request, or else to the combine, which merges it for a request of
    several splits and copies a one-split request's own. Every argument is a tensor whose
    address the graph keeps, or a scalar fixed at capture, so a replay launches the same calls.
    """
    import tvm_ffi
    from cutlass.cute.typing import Float32, Int32

    from .int_decode_pingpong import IntPingPongDecodeSm90
    from .serving_cache import require_code_pool

    for name, pool in (("mKq", mKq), ("mVq", mVq)):
        require_code_pool(name, pool.shape, IntPingPongDecodeSm90.code_pool_order,
                          stride=pool.stride())
    with tvm_ffi.use_torch_stream():
        compiled(q_bhd, mKq, mVq, mScale, block_table, o_partial, out_bhd, lse, counters,
                 Float32(scale_s), Float32(scale_o), Int32(num_kv_heads), work,
                 lse_out if combine is None else None)
        if combine is not None:
            combine(o_partial, lse, out_bhd, splits, *(() if lse_out is None else (lse_out,)),
                    Int32(out_bhd.shape[0]), Int32(out_bhd.shape[1]))
