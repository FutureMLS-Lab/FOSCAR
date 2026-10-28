# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: Apache-2.0
"""Compile and launch glue for ``IntPingPongDecodeSm90``, the SM90 quantized KV kernel.

This is the Hopper sibling of ``_sm100/fold_runtime.py``: it is what
:class:`cutedsl_decode.PagedDecodeWrapper` calls when ``plan(bits=...)`` runs on
an SM90 device, so the quantized path has one public entry point on both
architectures. The entry points are ``compile_pingpong`` and
``compile_constructed`` for compilation, ``alloc_pingpong_workspace`` for the
per-plan workspace, and ``launch_pingpong`` for the launch.

INVARIANTS

  * The accuracy floor is ~2.4e-3, not SM100's ~1.8e-3, because ``mQ`` here is
    bf16 where ``IntDecodeSm100`` takes fp8_e4m3. 2**-9 is the floor.
  * ``plan_splits`` may emit split counts outside ``_sm100``'s ``{1,2,4,8,16}``.
    It returns the occupancy model's pick (16 at B=1, KV 8k-32k), and it falls
    back to ``capacity // base`` for an unknown width. Any count is legal,
    because the kernel ceil-partitions and tolerates empty tail splits.

DESIGN NOTES THAT MATTER IF YOU CHANGE THIS
-------------------------------------------
* ``mCnt`` (int32 ``[B*HKV]``) is the fused-combine arrival counter. The kernel
  contract is: allocate it zeroed once, and the reducing CTA rearms its own cell
  to 0 each launch. So it must not be re-zeroed per call, and it must not be
  carved out of a buffer that something else writes. It is allocated in
  ``plan()`` and owned by the wrapper.
* ``fused_combine`` is a trace-time constructor flag, not a runtime one, and
  ``plan_fused_combine`` is its enable policy. It must therefore be decided
  before the compile, which is why the split plan happens first in ``plan()``.
  The SM100 side follows the same ordering, because its reduction mode is also
  compile-time.
* When ``outer_splits > 1`` and the fused combine is off, a second kernel
  (``SplitKCombine``) runs after the decode. When it is on, the decode kernel's
  last CTA does the merge and the second launch must be skipped, because
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
REMAT_OFF_HEAD_DIMS = (128,)
#: The head dimensions at which strength reduction is off; at any other head dimension it stays
#: on.
LSR_OFF_HEAD_DIMS = (128,)
#: What every SM90 kernel other than the decode compiles with under DSL 4.8, so that such a kernel
#: does not inherit a decode compile's values: the remat pass on, pressure-aware CSE and loop
#: strength reduction on, which are libNVVM's defaults, and a remat cost threshold of 10, a reset
#: value. libNVVM does not expose that threshold's default; 10 is the lowest value that compiles
#: each SM90 non-decode kernel to the same binary as with no options at all.
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
    """Whether `kern` compiles with machine CSE that ignores register pressure.

    It does except in the configuration classes below, where pressure-aware CSE, NVVM 23's
    default, is faster: INT4 with K token-wise per half of the head dimension, INT4 with
    channel-wise K whose group spans more than one tile and token-wise V, and a fused-combine
    kernel at a head dimension where the remat pass stays on. The classes are named rather than
    derived."""
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
    """Whether `kern` compiles with loop strength reduction off.

    It is off in the configuration classes below, all at head dimension 128, where strength
    reduction is faster: each is channel-wise K whose group spans more than one tile, outside a
    fused-combine kernel, and either INT2 with token-wise V or INT4 with channel-wise V whose
    group is one tile. The classes are named rather than derived. Everywhere else strength
    reduction stays on, including head dimension 256."""
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


def plan_pingpong(batch, num_kv_heads, kv_len, *, bits, head_dim, sm_count=None):
    """``(outer_splits, min_blocks_per_sm, fused_combine)`` for one launch.

    Thin adapter over ``_sm90/scheduler.plan_splits`` + ``plan_fused_combine``,
    so the wrapper reaches the calibrated H100 cost model. The quantized path
    plans only through ``plan_splits``. ``head_dim`` is required and forwarded,
    because the planner's model is fitted per head dimension.
    """
    from .scheduler import plan_fused_combine, plan_splits

    if not isinstance(kv_len, int):
        kv_len = max(int(v) for v in kv_len)
    outer, min_blocks = plan_splits(batch, num_kv_heads, kv_len, bits=bits,
                                    head_dim=head_dim, page_size=POOL_PAGE_SIZE,
                                    sm_count=sm_count)
    fused = plan_fused_combine(batch, num_kv_heads, outer)
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
                     opt_level=None):
    """Compile ``IntPingPongDecodeSm90`` with symbolic batch / page-count shapes.

    ``k_quant`` / ``v_quant`` pass unchanged to the constructor, which resolves a
    ``None`` to ``("token", head_dim)``; the default is spelled there and only there.

    Only the dims a serving integration varies (batch, physical page count,
    pages-per-request, split count) are symbolic; the per-(page, head) code and
    scale block shapes are static because the kernel's TMA views assert on
    their exact strides.

    Returns the compiled callable. It cross-compiles on a non-Hopper machine, where it must
    not then be launched, because the device returns a sticky CUDA error 209; it is launched
    and validated on H100.

    The fused fold is not optional here. The kernel accepts `fold_scale_zero=False`, but
    this function does not take that flag: a public compile API that offers a caller the
    choice between a fast path and a slow one lets a caller reach the slow path by
    forgetting the flag. Callers of this function always get the folded path. A caller
    that must compile the unfolded kernel constructs the kernel itself and calls
    `compile_constructed`.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import Float32, Int32

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
                               opt_level=opt_level)


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

    Every compile of the decode kernel goes through this, including the static-shape
    compiles in the tests, so that a test exercises the object the library ships.
    """
    return _compiler(opt_level, nvvm_options(kern))


def sm90_compiler(opt_level):
    """`(compile, options)` for an SM90 kernel other than the decode, such as `SplitKCombine` or
    `IntAppend`, with every NVVM option a decode compile sets stated at a reset value
    (`NVVM_DEFAULTS`, `_compiler`), so that its binary does not depend on what the process
    compiled before it."""
    return _compiler(opt_level, NVVM_DEFAULTS)


def compile_constructed(kern, *, head_dim, num_qo_heads, num_kv_heads,
                        page_size=POOL_PAGE_SIZE, opt_level=None):
    """Compile an already constructed ``IntPingPongDecodeSm90`` against symbolic shapes.

    Separate from `compile_pingpong` so that the fold remains non-optional on the
    public entry point while a harness that must compile the unfolded kernel still
    can. A caller must not be able to select the slow path by passing a flag, so the
    flag is not on the public API; a caller that genuinely wants a non-default kernel
    builds that kernel itself and brings it here.

    Besides `compile_pingpong`, it is called by tests that compile a kernel the public
    entry point does not build, such as the unfolded kernel or a kernel with the int2
    phase lock switched off, and by callers that need the constructed kernel object
    itself, which `compile_pingpong` does not return.
    """
    # THE KERNEL AND THE TENSORS MUST AGREE ON D. `kern` is already built and carries
    # its own head_dim; the `head_dim` argument here only sets the fake Q and output
    # extents. A D=128 kernel compiled against D=64 tensors would trace code that still
    # writes both 64-channel halves, past the end of a 64-wide row. The constructor
    # guard cannot see this, because both values are individually legal.
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
    mOut = fct(cutlass.BFloat16, (sym_batch, num_qo_heads, head_dim),
               stride_order=(2, 1, 0), assumed_align=16)
    mLSE = fct(cutlass.Float32, (sym_outer, sym_batch, num_qo_heads),
               stride_order=(2, 1, 0), assumed_align=16)
    mCnt = fct(Int32, (sym_cnt,), assumed_align=16)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_fn, options = pingpong_compiler(kern, opt_level)

    def _trace():
        from ..kernel_cache import init_cuda_driver
        init_cuda_driver()
        return compile_fn(
            kern, mQ, mKq, mVq, mScale, mPT, mSL, mOp, mOut, mLSE, mCnt,
            Float32(1.0),        # bmm1 (softmax scale)
            Float32(1.0),        # bmm2 (output scale)
            Int32(1),            # outer_splits
            Int32(num_kv_heads),
            Int32(1),            # batch
            stream_fake,
        )

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
        },
    }
    return kernel_cache.compile_or_load(
        kind=f"sm90_{type(kern).__name__}",
        compile_fn=_trace,
        sources=sources,
        config=config,
        options=options,
    )


def compile_splitk_combine(*, head_dim, opt_level=None):
    """Compile the external ``SplitKCombine`` used when ``outer_splits > 1``
    and the in-kernel fused combine is off. Validated on H100; on a non-Hopper
    box this compiles and must not be invoked."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import Int32

    from .splitk_combine import SplitKCombine

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
    mOut = fct(cutlass.BFloat16, (sym_batch, sym_hq, head_dim),
               stride_order=(2, 1, 0), assumed_align=16)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_fn, options = sm90_compiler(opt_level)
    combine = SplitKCombine(head_dim)

    def _trace():
        from ..kernel_cache import init_cuda_driver
        init_cuda_driver()
        return compile_fn(
            combine, mOp, mLSE, mOut,
            Int32(1), Int32(1), Int32(1), stream_fake,
        )

    from .. import kernel_cache
    from . import splitk_combine as _combine_module

    return kernel_cache.compile_or_load(
        kind="sm90_SplitKCombine",
        compile_fn=_trace,
        sources=kernel_cache.source_closure(_combine_module.__file__, __file__),
        config={"kernel": kernel_cache.simple_attributes(combine),
                "trace": {"head_dim": head_dim}},
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
        counters=torch.zeros(batch * num_kv_heads, device=device, dtype=torch.int32),
    )


def launch_pingpong(compiled, combine, *, q_bhd, mKq, mVq, mScale, block_table,
                    seq_lens, o_partial, out_bhd, lse, counters,
                    scale_s, scale_o, outer_splits, num_kv_heads, batch):
    """Launch the decode, and the combine kernel when one is supplied.

    Validated on H100 SXM5 at every split count and with both combine mechanisms.

    ``combine`` must be None when the kernel was compiled with
    ``fused_combine=True`` or when ``outer_splits == 1``, because passing it anyway
    would double-reduce. The caller (`PagedDecodeWrapper`) owns that choice; see
    ``plan_pingpong``.

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

    with tvm_ffi.use_torch_stream():
        compiled(q_bhd, mKq, mVq, mScale, block_table, seq_lens,
                 o_partial, out_bhd, lse, counters,
                 Float32(scale_s), Float32(scale_o), Int32(outer_splits),
                 Int32(num_kv_heads), Int32(batch))
        if combine is not None:
            combine(o_partial, lse, out_bhd, Int32(outer_splits), Int32(batch),
                    Int32(q_bhd.shape[1]))
