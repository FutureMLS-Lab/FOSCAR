# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Donglin Zhuang (standalone foscar adaptation).
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from FlashInfer's CuTe DSL paged-decode wrapper (see NOTICE).

"""PagedDecodeWrapper — PyTorch-facing API for paged GQA decode over a quantized KV cache.

Provides :class:`PagedDecodeWrapper` for SM90 (Hopper) and SM100 (Blackwell) GPUs. The
architecture is the one of the device the wrapper's workspace lives on, and it selects the
reader: ``IntPingPongDecodeSm90`` on Hopper, ``IntDecodeSm100`` on Blackwell.

ONE PLAN PER STEP, ONE RUN PER LAYER
------------------------------------
``plan(indptr, indices, seq_lens, num_qo_heads, layout)`` fixes one step's batch: the page
table, the lengths, the split count and the compiled kernel. ``layout`` is the
:class:`~foscar.kv_cache.QuantKVLayout` the caches were built with, which carries the
stored width, the head dimension, the KV-head count, the quantization granularity, the scale
dtype and the pool orders. ``run(q, kv_cache)`` then decodes one layer's cache. It refuses a
cache of any other layout, a query or output of another shape, and any tensor on another
device, so a pool cannot meet a reader that reads it in another order.

Kernels compile with symbolic dimensions (``cute.sym_int``) so the same compiled module is
reused across batch sizes and lengths, and compilation is memoized on the static
configuration, in process and on disk (:mod:`foscar.kernel_cache`).

The split-K planners are deliberately NOT shared between the two architectures; see
:meth:`PagedDecodeWrapper.plan` for why.

CUDA-GRAPH MODE
---------------
``PagedDecodeWrapper(workspace, use_cuda_graph=True)`` is the serving path under CUDA graphs,
beside eager mode, whose planners, kernels and host paths it leaves as they are: the constructor
returns the graph-mode implementation (:class:`_GraphDecodeWrapper`), so an eager wrapper runs
no graph-mode code at all. One graph-mode wrapper serves one captured batch size (a bucket) for
its whole life: ``plan(page_table=, seq_lens=, ...)`` once, records the engine's fixed buffers
and compiles the bucket's builds; ``update(lengths)`` once per step, outside the graph, plans the
step on the host (:mod:`foscar.graph_plan`) and copies its work list to the device;
``run()`` is captured once per layer and replayed. ``graph_workspace_bytes`` sizes the workspace
that buckets share.

The pre-launch guards in ``run()`` refuse a query, an output, a cache or an LSE tensor
that does not match the plan, before anything is launched.
"""

import functools
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:  # `cutlass` is imported where it is used, so a host import stays cheap
    import cutlass

from . import _utils
from ._utils import (
    api as flashinfer_api,
    get_num_sm,
)
from . import graph_plan
from .kv_cache import QuantKVCache, QuantKVLayout

_QUANT_BITS = (2, 4)
#: The quantized kernels are built around a 64-token page: both
#: ``IntDecodeSm100`` and ``IntPingPongDecodeSm90`` assert it (one code block and
#: one scale block per (page, kv-head)). `kv_cache.PAGE_SIZE` is the layout's statement of
#: it; this literal is the one read from this file's syntax tree without importing the
#: package, and `tests/test_public_api.py` holds the two equal.
_QUANT_PAGE_SIZE = 64

# The decode kernels are specialized to the head dimensions they tile (64, 128, 256).
# The two architectures reach that from different mechanisms, so do not state one of
# them as if it covered both:
#   SM90 builds 128 in every layout, and 64 / 256 with whole-head token-wise K and V,
#   the fold, an f32 scale pool and 2 or 4 stages (`quant_layout.sm90_wide_refusal`).
#   At 256 it cuts both contractions into 128-channel groups and tiles its output
#   accumulators and its fused split-K epilogue by D.
#   SM100 builds 128 in every layout, and 64 / 256 with whole-head token-wise K/V,
#   f32 scales and default settings (64 in both eager and graph modes; 256 in eager mode).
# Both constructors RAISE rather than assert, so `python -O` cannot drop the guard.
# The accepted set is therefore per architecture, keyed by the names `_utils.get_capabilities`
# reports:
#   "sm90" lists 64, 128 and 256; its planner has head_dim-specific entries in
#   `_sm90/scheduler.py` (`PINGPONG_MODEL`).
#   "sm100" lists 64, 128 and 256; CUDA graphs support 64 and 128.
# `kv_cache.QuantKVLayout.create` refuses anything else, so a plan never sees it.
# It stays one literal, because tooling reads it from this file's syntax tree with
# `ast.literal_eval` rather than importing the package.
_ACCEPTED_HEAD_DIMS = {"sm90": (64, 128, 256), "sm100": (64, 128, 256)}

#: The output dtypes each architecture's kernel is compiled for. float32 is the engine's opt-in:
#: a plan for it compiles its own build, and a bf16 plan compiles the default build. A dtype
#: outside the list is refused at plan time rather than at the launch
#: (`_sm90/pingpong_runtime.OUT_DTYPES` is that kernel's own list).
_OUT_DTYPES = {"sm90": (torch.bfloat16, torch.float32),
               "sm100": (torch.bfloat16, torch.float16, torch.float32)}
#: The query dtype each architecture's kernel reads.
_Q_DTYPES = {"sm90": torch.bfloat16, "sm100": torch.float8_e4m3fn}

#: Arguments of the per-argument `plan()` and the tensor-form `run()` that the current call
#: forms refuse. Each raises and names what replaced it.
_RETIRED_PLAN_ARGS = ("num_kv_heads", "head_dim", "page_size", "bits", "k_quant", "v_quant")
_RETIRED_RUN_ARGS = ("k_cache", "v_cache", "scale_pool", "pscale_lp", "k_pool_order")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _torch_to_cutlass(dtype: torch.dtype):
    """Map a torch dtype to the CuTe DSL dtype of the same width.

    Imported here rather than at module scope so that importing the package does
    not require a CUDA toolkit.
    """
    import cutlass

    table = {
        torch.float32: cutlass.Float32,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float8_e5m2: cutlass.Float8E5M2,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    }
    if dtype not in table:
        raise TypeError(
            f"cute-dsl decode does not support {dtype}; "
            f"supported: {list(table)}"
        )
    return table[dtype]



def _slice_workspace(
    workspace: torch.Tensor,
    kv_splits: int,
    batch_size: int,
    prediction: int,
    num_qo_heads: int,
    head_dim: int,
):
    """Slice pre-allocated workspace into partial result workspace"""
    o_partial_shape = torch.Size(
        (kv_splits, batch_size, prediction, num_qo_heads, head_dim)
    )
    m_partial_shape = l_partial_shape = o_partial_shape[:-1]

    workspace = workspace.view(dtype=torch.float32)

    required_elts = (
        o_partial_shape.numel()
        + m_partial_shape.numel()
        + l_partial_shape.numel()
    )
    if workspace.numel() < required_elts:
        raise RuntimeError(
            f"kernel reduction with kv_splits={kv_splits} "
            f"batch_size={batch_size} q_len_per_req={prediction} "
            f"num_qo_heads={num_qo_heads} head_dim={head_dim} "
            f"requires {required_elts * 4} byte workspace "
            f"which exceeds provided workspace of {workspace.nbytes} bytes"
        )

    start, end = 0, o_partial_shape.numel()
    o_partial = workspace[start:end].view(o_partial_shape)

    start, end = end, end + l_partial_shape.numel()
    l_partial = workspace[start:end].view(l_partial_shape)

    start, end = end, end + m_partial_shape.numel()
    m_partial = workspace[start:end].view(m_partial_shape)

    return o_partial, l_partial, m_partial


# ---------------------------------------------------------------------------
# Compile helpers (memoized)
# ---------------------------------------------------------------------------


@functools.cache
def _get_compiled_fold_kernel(
    page_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    reduction: str,
    bits: int,
    out_dtype: "cutlass.dtype",
    kernel_kwargs: Tuple[Tuple[str, Any], ...],
    has_q_descale: bool = False,
    has_lse: bool = False,
):
    """Compile + cache ``IntDecodeSm100`` (SM100 quantized-KV decode).

    Keyed on the STATIC configuration only: ``compile_fold`` traces batch,
    physical page count and kv_splits as ``cute.sym_int()``, so one compiled
    module serves every batch size and KV length. A trace costs 10 to 13 s on
    B200, so this in-process memo is the difference between a usable plan() and
    an unusable one, and :mod:`foscar.kernel_cache` underneath it serves a repeat
    of the same trace from disk across processes.

    ``kernel_kwargs`` arrives as a sorted tuple of pairs because
    :func:`functools.cache` needs hashable arguments. ``has_q_descale`` selects the
    variant that reads a per-row query descale (``run(q_descale=)``), and ``has_lse`` the one
    that returns the log-sum-exp (``run(lse=)``); each is a different compiled kernel, so both
    are part of the key.

    Returns ``(compiled, applied_knobs)``. ``applied_knobs`` is read back off
    the CONSTRUCTED kernel object and is cached with it, so
    :meth:`PagedDecodeWrapper._assert_bits_tuning_applied` can re-check the
    promoted tuning on every plan() — including cache hits, where the trace
    (and therefore ``compile_fold``'s own assert) does not run again.
    """
    from ._sm100.fold_runtime import compile_fold

    knobs: Dict[str, Any] = {}
    compiled = compile_fold(
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        reduction=reduction,
        bits=bits,
        out_dtype=out_dtype,
        tma_mask=False,
        knobs_out=knobs,
        has_q_descale=has_q_descale,
        has_lse=has_lse,
        **dict(kernel_kwargs),
    )
    return compiled, knobs


@functools.cache
def _persistent_sm_count() -> int:
    """The most CTAs `IntDecodeSm100`'s persistent grid launches when the layout sets no
    `persistent_sm_count` of its own: the constructor's default, read off its signature so
    that the plan and the kernel cannot disagree."""
    import inspect

    from ._sm100.int_fold_decode import IntDecodeSm100

    return int(inspect.signature(IntDecodeSm100).parameters["persistent_sm_count"].default)


@functools.cache
def _get_compiled_pingpong_kernel(
    page_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    bits: int,
    min_blocks_per_sm: int,
    fused_combine: bool,
    k_quant: Tuple[str, int],
    v_quant: Tuple[str, int],
    sdtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.bfloat16,
    has_lse: bool = False,
):
    """Compile + cache ``IntPingPongDecodeSm90`` (SM90 quantized-KV decode).

    Validated on H100 SXM5; see ``_sm90/pingpong_runtime.py``'s module docstring
    for what was checked.

    ``sdtype`` is part of the cache key, not a default to be dropped: it sets
    the scale pool's CuTe element type and its TMA transaction byte count, so
    two granularity configs that differ only in pool precision are two
    different compiled kernels. So are ``out_dtype``, the output's dtype, and ``has_lse``, the
    fused-combine build that also stores the merged log-sum-exp (``run(lse=)``).
    """
    from ._sm90.pingpong_runtime import compile_pingpong

    return compile_pingpong(
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        bits=bits,
        min_blocks_per_sm=min_blocks_per_sm,
        fused_combine=fused_combine,
        k_quant=k_quant,
        v_quant=v_quant,
        page_size=page_size,
        sdtype=sdtype,
        out_dtype=out_dtype,
        has_lse=has_lse,
    )


@functools.cache
def _get_compiled_splitk_combine(head_dim: int, out_dtype: torch.dtype = torch.bfloat16,
                                 has_lse: bool = False):
    """Compile + cache the SM90 external split-K combine, for the output dtype and, with
    ``has_lse``, the form that also stores the merged log-sum-exp. Validated on H100."""
    from ._sm90.pingpong_runtime import compile_splitk_combine

    return compile_splitk_combine(head_dim=head_dim, out_dtype=out_dtype, has_lse=has_lse)


# ---------------------------------------------------------------------------
# CUDA-graph mode's builds, compiled apart from eager's (`_GraphDecodeWrapper`)
# ---------------------------------------------------------------------------


@functools.cache
def _get_compiled_fold_graph_kernel(page_size, num_qo_heads, num_kv_heads, head_dim, bits,
                                    out_dtype, kernel_kwargs, has_q_descale, has_lse):
    """Compile + cache Blackwell's graph split build (`compile_fold(graph_split=True)`), the
    split build (`reduction="kernel"`) that reads its pieces from a work list, keyed and read
    back as `_get_compiled_fold_kernel` keys and reads back eager's builds."""
    from ._sm100.fold_runtime import compile_fold

    knobs: Dict[str, Any] = {}
    compiled = compile_fold(
        page_size=page_size, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, reduction="kernel", bits=bits, out_dtype=out_dtype, tma_mask=False,
        knobs_out=knobs, has_q_descale=has_q_descale, has_lse=has_lse, graph_split=True,
        **dict(kernel_kwargs))
    return compiled, knobs


@functools.cache
def _get_compiled_pingpong_graph_kernel(page_size, num_qo_heads, num_kv_heads, head_dim, bits,
                                        min_blocks_per_sm, fused_combine, k_quant, v_quant,
                                        sdtype, out_dtype, has_lse):
    """Compile + cache Hopper's graph split build (`compile_pingpong(graph=True)`), keyed as
    `_get_compiled_pingpong_kernel` keys eager's builds."""
    from ._sm90.pingpong_runtime import compile_pingpong

    return compile_pingpong(
        head_dim=head_dim, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, bits=bits,
        min_blocks_per_sm=min_blocks_per_sm, fused_combine=fused_combine, k_quant=k_quant,
        v_quant=v_quant, page_size=page_size, sdtype=sdtype, out_dtype=out_dtype,
        has_lse=has_lse, graph=True)


@functools.cache
def _get_compiled_splitk_combine_graph(head_dim, out_dtype, has_lse):
    """Compile + cache Hopper's graph combine, which reads each request's split count from
    memory, with the LSE or without."""
    from ._sm90.pingpong_runtime import compile_splitk_combine

    return compile_splitk_combine(head_dim=head_dim, out_dtype=out_dtype, has_lse=has_lse,
                                  graph=True)


# ---------------------------------------------------------------------------
# CUDA-graph mode: buckets, their scratch, and the host lengths of a step
# ---------------------------------------------------------------------------

#: The pinned host buffers `update()` writes its work lists into, in turn. Before it rewrites one
#: it waits for that buffer's previous copy to the device to complete, so a host running more
#: than this many steps ahead of the device waits there; nothing waits on the device as a whole.
_GRAPH_HOST_BUFFERS = 4
#: The longest context a graph bucket takes: the kernels hold lengths and tile counts in int32,
#: and a length is rounded up to a whole 64-token tile.
_GRAPH_MAX_KV_LEN = 2**31 - 1 - 63


def _graph_tile(arch, layout):
    """The sequence tile of the graph builds of `layout`, in tokens: Hopper's 64-token page, and
    Blackwell's `resolve_fold_tiling` (256, 512 at D=64 int2, or 128 for K token-128 with V
    channel-64), which is also what eager's planner is told (`_plan_quantized_sm100`)."""
    if arch == "sm90":
        return _QUANT_PAGE_SIZE
    from ._sm100.fold_runtime import resolve_fold_tiling

    return resolve_fold_tiling(
        layout.k_quant, layout.v_quant, head_dim=layout.head_dim, bits=layout.bits
    )[0]


def _graph_bucket(arch, layout, batch, max_kv_len, sm_count, num_qo_heads=None):
    """The `graph_plan.Bucket` of `batch` requests of `layout` with contexts up to `max_kv_len`
    on a device of `sm_count` SMs."""
    head_tiles = 1 if num_qo_heads is None else (num_qo_heads // layout.num_kv_heads + 7) // 8
    return graph_plan.Bucket(batch, layout.num_kv_heads,
                             wave=graph_plan.SPLIT_CTAS_PER_SM[arch] * sm_count,
                             tile=_graph_tile(arch, layout), max_kv_len=max_kv_len,
                             head_tiles=head_tiles)


def _graph_scratch_words(arch, bucket, num_qo_heads, head_dim):
    """The f32 words of the shared workspace one launch of `bucket` uses: on Hopper the decode's
    split outputs and split LSEs at the bucket's split capacity (one split when it cannot split,
    because the unsplit build stores its LSE there too); on Blackwell the split build's partial
    output, maxima and sums, and nothing for an unsplit bucket."""
    rows = bucket.s_tight * bucket.batch * num_qo_heads
    if arch == "sm90":
        return rows * (head_dim + 1)
    return rows * (head_dim + 2) if bucket.split_capable else 0


def graph_workspace_bytes(max_batch: int, num_qo_heads: int, layout: QuantKVLayout, *,
                          max_kv_len: int, device=None) -> int:
    """The float workspace that graph-mode wrappers of every batch size up to `max_batch` can
    share, for `num_qo_heads` query heads over caches of `layout` with contexts up to
    `max_kv_len`, on `device` (default: the current CUDA device): the largest scratch any one
    bucket's launch uses. Buckets may share it because a launch writes every scratch word it
    reads, and because no two of their graphs run at the same time; an engine that overlaps two
    batches gives each stream its own workspace."""
    if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
        raise ValueError(f"max_batch must be a positive int; got {max_batch!r}")
    device = torch.device("cuda", _utils.cuda_index(device))
    arch = _utils.require_arch(device)
    if layout.arch != arch:
        raise ValueError(f"the layout was resolved for {layout.arch} and {device} is {arch}")
    _check_num_qo_heads(arch, num_qo_heads, layout.num_kv_heads)
    sm_count = get_num_sm(device)
    wave = graph_plan.SPLIT_CTAS_PER_SM[arch] * sm_count
    head_tiles = (num_qo_heads // layout.num_kv_heads + 7) // 8
    words = 0
    for batch in range(1, max_batch + 1):
        if batch * layout.num_kv_heads * head_tiles >= wave and batch < max_batch:
            continue            # unsplit, and its scratch grows with the batch: max_batch's is larger
        bucket = _graph_bucket(arch, layout, batch, max_kv_len, sm_count, num_qo_heads)
        words = max(words, _graph_scratch_words(arch, bucket, num_qo_heads, layout.head_dim))
    return 4 * words


def _pinned_words(n):
    """A page-locked int32 host buffer of `n` words: a copy from pageable memory is not
    asynchronous, so `update()` would wait for it."""
    return torch.empty(n, dtype=torch.int32, pin_memory=True)


def _graph_host_lengths(lengths, batch, max_kv_len):
    """`update()`'s lengths as int64 [B], refused before any conversion unless they are integers
    on the host: a CUDA tensor would need a device sync, and a float would be rounded."""
    if isinstance(lengths, torch.Tensor):
        if lengths.device.type != "cpu":
            raise ValueError(
                f"update() takes the step's lengths on the host (a list, a numpy array or a CPU "
                f"tensor); reading a tensor on {lengths.device} would wait for the device")
        if lengths.dtype.is_floating_point or lengths.dtype.is_complex \
                or lengths.dtype == torch.bool:
            raise ValueError(f"update() takes integer lengths; got a {lengths.dtype} tensor")
        lengths = lengths.numpy()
    else:
        lengths = np.asarray(lengths)
        if lengths.dtype.kind not in "iu":
            raise ValueError(f"update() takes integer lengths; got {lengths.dtype} values")
    if lengths.shape != (batch,):
        raise ValueError(f"update() takes one length per request of the bucket, [{batch}]; got "
                         f"shape {tuple(lengths.shape)}")
    lengths = lengths.astype(np.int64)
    if int(lengths.min()) < 1 or int(lengths.max()) > max_kv_len:
        raise ValueError(
            f"every length must be in [1, max_kv_len = {max_kv_len}]; got "
            f"[{int(lengths.min())}, {int(lengths.max())}]. A padded row decodes one token of a "
            f"reserved page.")
    return lengths


class _GraphPlan:
    """What a graph-mode `plan()` fixed and keeps for the wrapper's life, because a captured graph
    keeps the address of every tensor here."""

    def __init__(self, bucket, page_table, seq_lens, lse, q_descale):
        self.bucket = bucket
        self.page_table, self.seq_lens = page_table, seq_lens
        #: the declared variants: whether every `run()` passes `lse=` and `q_descale=`
        self.lse, self.q_descale = lse, q_descale
        self.decode = self.combine = None
        # views of the shared workspace: the split outputs, and on Hopper the split LSEs, on
        # Blackwell the split maxima and sums
        self.o_partial = self.split_lse = self.l_partial = self.m_partial = None
        # Hopper: the fused-combine arrival counters, private and zeroed once
        self.counters = None
        # Blackwell: the unsplit build's row offsets into the engine's page table
        self.table_offsets = None
        # split buckets: the device work list (`work_views` cuts its [W, 4] slots and [B] split
        # counts out of it at each launch, so that it is one tensor), and the pinned host
        # buffers it is copied from with each copy's completion event
        self.work = None
        self.host, self.host_words, self.events = [], [], []
        self.recorded = [False] * _GRAPH_HOST_BUFFERS
        #: the split counts each pinned buffer was last written with: a step with the same
        #: counts rewrites only its lengths (`Bucket.write_lengths`)
        self.host_splits = [None] * _GRAPH_HOST_BUFFERS
        self.next_host = 0
        #: the stream of the first `update()`, which every later one must use
        self.stream = None

    def work_views(self):
        """`(slots, splits)`: the work list's int32 [W, 4] slots and [B] split counts."""
        wave, batch = self.bucket.wave, self.bucket.batch
        k0 = graph_plan.k_offset(wave)
        return (self.work[:k0].view(wave, graph_plan.ENTRY_WORDS),
                self.work[k0:k0 + batch])



def _check_num_qo_heads(arch, num_qo_heads, num_kv_heads):
    """Refuse a query-head count the architecture's decode does not build for `num_kv_heads`:
    Shared by eager and graph planning before any compilation."""
    if (isinstance(num_qo_heads, bool) or not isinstance(num_qo_heads, int)
            or num_qo_heads <= 0 or num_qo_heads % num_kv_heads != 0):
        raise ValueError(
            f"num_qo_heads ({num_qo_heads!r}) must be a positive multiple of the layout's "
            f"num_kv_heads ({num_kv_heads})"
        )
    ratio = num_qo_heads // num_kv_heads
    if arch == "sm90" and ratio > 8 and ratio % 8:
        raise NotImplementedError(
            "SM90 GQA ratios above eight must be multiples of eight; "
            f"got {ratio} ({num_qo_heads} / {num_kv_heads})")


def _resolve_io_dtypes(arch, layout, q_data_type, o_data_type):
    """`(q_data_type, o_data_type)` with the defaults resolved, refusing what the architecture's
    decode does not read or write: graph mode's copy of eager `plan()`'s resolution."""
    if q_data_type is None:
        q_data_type = _Q_DTYPES[arch]
    if q_data_type != _Q_DTYPES[arch]:
        raise ValueError(
            f"the {layout.reader} decode reads a {_Q_DTYPES[arch]} query; got "
            f"q_data_type={q_data_type}. The SM100 kernel takes fp8 e4m3 with its "
            f"per-tensor descale folded into sm_scale; the SM90 kernel takes bf16.")
    if o_data_type is None:
        o_data_type = torch.bfloat16
    if o_data_type not in _OUT_DTYPES[arch]:
        raise ValueError(
            f"the {layout.reader} decode writes a {list(_OUT_DTYPES[arch])} output; got "
            f"o_data_type={o_data_type}")
    return q_data_type, o_data_type



# ---------------------------------------------------------------------------
# Paged-KV decode wrapper
# ---------------------------------------------------------------------------


def _fail_closed_plan(method):
    """Make a planning method transactional from the caller's perspective."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        self._invalidate_plan()
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self._invalidate_plan()
            raise

    return wrapped


def _refuse_retired(where, positional, keywords, retired, replacement):
    """Raise for arguments the current call forms refuse, naming what replaced them.

    Such an argument must RAISE rather than be accepted and ignored, on every entry point,
    or a caller who still passes it believes something false."""
    if "kernel_kwargs" in keywords:
        raise TypeError(
            f"{where}: kernel_kwargs is retired from the public API, because what it still "
            f"carried selects kernel paths and pool layouts; the layout carries every "
            f"configuration a plan needs (QuantKVLayout.create)")
    named = sorted(set(keywords) & set(retired))
    if named or positional:
        what = (f"{named}" if named else f"{len(positional)} extra positional argument(s)")
        raise TypeError(f"{where}: {what} belong to the retired call form; {replacement}")
    if keywords:
        raise TypeError(f"{where}: unexpected keyword argument(s) {sorted(keywords)}")


class PagedDecodeWrapper:
    """PyTorch-facing wrapper for paged GQA decode over a quantized KV cache (SM90/SM100)."""

    def __new__(cls, float_workspace_buffer: torch.Tensor = None, use_cuda_graph: bool = False):
        # CUDA-graph mode is a separate implementation, chosen here, once: a wrapper built with
        # `use_cuda_graph=True` is a `_GraphDecodeWrapper`, and an eager wrapper's plan() and
        # run() run no graph-mode code (module docstring, CUDA-GRAPH MODE).
        if use_cuda_graph and cls is PagedDecodeWrapper:
            cls = _GraphDecodeWrapper
        return super().__new__(cls)

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        # The workspace's device is the wrapper's device, and it is concrete: a tensor's
        # `.device` always carries its ordinal. Every plan resolves the architecture from it,
        # never from whichever device happens to be current.
        self.device = float_workspace_buffer.device
        # `use_cuda_graph` selects the implementation in `__new__`; nothing here reads it.
        self._invalidate_plan()

    def _invalidate_plan(self) -> None:
        """Discard every state object that could make a failed replan runnable."""
        self._planned = False
        self._layout = None
        self._arch = None
        self._compiled_fold = None
        # the plan's compile key, and the variants of the same kernel, keyed by
        # (has_q_descale, has_lse), which `run(q_descale=)` and `run(lse=)` fetch the first
        # time they are passed
        self._fold_key = None
        self._fold_variants = None
        # whether the plan's Blackwell build reads the cache's count table of page maxima
        self._reads_v_max_hist = None
        self._compiled_combine = None
        self._quant_kernel_kwargs = None
        self._quant_offsets = None
        self._planned_splits = None
        self._planned_reduction = None
        self._sm90_min_blocks = None
        self._sm90_fused_combine = None
        self._sm90_scale_dtype = None
        self._sm90_block_table = None
        self._sm90_ws = None
        # the compile arguments of the plan's decode, and its LSE-returning merge (the fused
        # decode or the external combine built with `has_lse`), fetched on first `run(lse=)`
        self._sm90_compile_args = None
        self._sm90_lse_merge = None
        self._bits = None
        self._k_quant = None
        self._v_quant = None
        self._q_data_type = None
        self._o_data_type = None
        self._is_sm90 = None
        self._num_qo_heads = None
        self._num_kv_heads = None
        self._head_dim = None
        self._page_size = None
        self._sm_scale = None
        self._batch_size = None
        self._max_kv_len = None
        self._kv_splits = None
        self._reduction = None
        self._has_workspace = False
        self._o_partial = None
        self._l_partial = None
        self._m_partial = None
        self._seqlens = None
        self._table_offsets = None
        self._page_table = None

    # -- read-only view of what plan() decided -------------------------------
    # Integrators need these to size their own buffers and to log what the
    # planner picked.

    @property
    def kv_splits(self) -> int:
        """Split-K partitions the current plan will launch with."""
        return self._kv_splits

    @property
    def reduction(self) -> str:
        """Resolved reduction mode ("none" or "kernel")."""
        return self._reduction

    @property
    def bits(self) -> Optional[int]:
        """The planned layout's stored bit width, 2 or 4; ``None`` before a plan."""
        return self._bits

    @flashinfer_api
    @_fail_closed_plan
    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        num_qo_heads: int,
        layout: QuantKVLayout = None,
        *retired_positional,
        q_data_type: Optional[torch.dtype] = None,
        o_data_type: Optional[torch.dtype] = None,
        sm_scale: Optional[float] = None,
        kv_splits: Optional[int] = None,
        reduction: str = "auto",
        q_len_per_req: int = 1,
        max_kv_len: Optional[int] = None,
        non_blocking: bool = True,
        **retired,
    ) -> None:
        """Plan paged GQA decode for one step's batch over caches of ``layout``.

        Parameters
        ----------
        indptr : torch.Tensor (int32, [batch_size + 1])
            Prefix-sum offsets into ``indices``.
        indices : torch.Tensor (int32, [num_pages_total])
            Flat per-sequence physical page indices, in logical order.
        seq_lens : torch.Tensor (int32 or int64, [batch_size])
            Per-sequence KV length in tokens. Plan copies it to the device as int32. Callers
            that have ``last_page_len`` instead can convert via
            ``(indptr.diff() - 1) * 64 + last_page_len``.
        num_qo_heads : int
            Query heads; a positive multiple of ``layout.num_kv_heads``. Eager
            kernels pad small groups and tile larger groups into eight-head CTAs.
            On SM90 ratios above eight must be multiples of eight. Graph mode
            retains ratios 1--8 on SM90 and 3--8 on SM100.
        layout : QuantKVLayout
            The layout the caches ``run()`` will read were built with, from
            ``QuantKVLayout.create(...)`` for this wrapper's device. It carries the stored
            width, head dimension, KV-head count, granularity, scale dtype and pool orders,
            and it replaces the ``num_kv_heads``, ``head_dim``, ``page_size``,
            ``bits``, ``k_quant``, ``v_quant`` and ``kernel_kwargs`` arguments, each of which
            raises. A layout resolved for another architecture than this wrapper's device is
            refused.
        q_data_type, o_data_type : Optional[torch.dtype]
            The query dtype must be the reader's: ``torch.float8_e4m3fn`` on SM100 and
            ``torch.bfloat16`` on SM90 (the default resolves it). The output defaults to
            bfloat16; both GPUs also write float32, and SM100 float16. Each output dtype is
            its own compiled build, so a bfloat16 plan's kernel is the same whether or not a
            float32 plan exists.
        sm_scale : Optional[float]
            Softmax scale; defaults to ``1 / sqrt(head_dim)``. On SM100 the fp8 query's
            per-tensor descale is folded in here by the caller (``sm_scale / q_scale``).
        kv_splits : Optional[int]
            Threadblocks per sequence (flash decoding).  ``None`` takes the
            architecture's planner pick from the planned shapes and SM count.
        reduction : str
            ``"kernel"`` (deterministic with workspace), ``"none"`` (no
            flash-decoding split-K; requires kv_splits == 1), or ``"auto"``
            (the arch cost model picks). On SM90 the reduction is derived
            from the split count, so only ``"auto"`` or the derived value is accepted.
        q_len_per_req : int
            Must be 1: this is decode, one query token per request.
        max_kv_len : Optional[int]
            Maximum KV sequence length across the batch, which the planners
            key on; passing it skips the ``seq_lens.max().item()`` that waits for the
            device. Hopper's plan still waits while it builds its block table.
        non_blocking : bool
            Async device copies for the plan-time integer buffers.

        THE TWO PLANNERS ARE NOT SHARED, and must not be unified. Both quantized cost models
        (``_sm100/scheduler.plan_splits_sm100``, ``_sm90/scheduler.plan_splits``) are KEYED
        BY ``bits`` and calibrated against their own kernels, with per-tile costs from
        measured int2/int4 sweeps (int2 0.791 us/tile unsplit against int4 1.25 on B200).
        """
        if retired_positional or retired or not isinstance(layout, QuantKVLayout):
            _refuse_retired(
                "PagedDecodeWrapper.plan", retired_positional, retired, _RETIRED_PLAN_ARGS,
                "build the cache layout with QuantKVLayout.create(bits=, head_dim=, "
                "num_kv_heads=, k_quant=, v_quant=) and pass it as plan(indptr, indices, "
                "seq_lens, num_qo_heads, layout)")
            raise TypeError(f"PagedDecodeWrapper.plan: layout must be a QuantKVLayout; got "
                            f"{type(layout).__name__}")
        # The architecture of the WRAPPER'S device, whatever device is current: the workspace,
        # the planner's SM count and every launch belong to it.
        arch = _utils.require_arch(self.device)
        if layout.arch != arch:
            raise ValueError(
                f"the layout was resolved for {layout.arch} and this wrapper's device "
                f"{self.device} is {arch}; create the layout for the device that decodes it")
        # Before any device work: the DSL compiles for one architecture per process.
        _utils.require_dsl_target(arch)
        if q_len_per_req != 1:
            raise NotImplementedError(
                "quantized decode is plain decode only (the fold kernels pack "
                f"one query token per threadblock); got "
                f"q_len_per_req={q_len_per_req}"
            )
        num_kv_heads, head_dim = layout.num_kv_heads, layout.head_dim
        _check_num_qo_heads(arch, num_qo_heads, num_kv_heads)
        if seq_lens.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"seq_lens must be int32 or int64; got {seq_lens.dtype}")
        batch_size = seq_lens.numel()
        if batch_size == 0:
            raise ValueError("plan() needs at least one request; seq_lens is empty")
        if indptr.numel() < batch_size:
            raise ValueError(
                f"indptr length ({indptr.numel()}) must be at least batch_size "
                f"({batch_size})"
            )

        if q_data_type is None:
            q_data_type = _Q_DTYPES[arch]
        if q_data_type != _Q_DTYPES[arch]:
            raise ValueError(
                f"the {layout.reader} decode reads a {_Q_DTYPES[arch]} query; got "
                f"q_data_type={q_data_type}. The SM100 kernel takes fp8 e4m3 with its "
                f"per-tensor descale folded into sm_scale; the SM90 kernel takes bf16.")
        if o_data_type is None:
            o_data_type = torch.bfloat16
        if o_data_type not in _OUT_DTYPES[arch]:
            raise ValueError(
                f"the {layout.reader} decode writes a {list(_OUT_DTYPES[arch])} output; got "
                f"o_data_type={o_data_type}")
        if sm_scale is None:
            sm_scale = head_dim**-0.5

        if max_kv_len is None:
            # Unlike the dense path, we cannot fall back to "assume the largest
            # possible sequence": the cost model turns kv_len into a tile count
            # and 2**31-1 tiles would pin every plan to the maximum split.
            # Accept the sync and tell the caller how to skip it.
            max_kv_len = int(seq_lens.max().item())

        def to_int32_device(tensor: torch.Tensor):
            return tensor.to(dtype=torch.int32, device=self.device,
                             non_blocking=non_blocking)

        seq_lens = to_int32_device(seq_lens)
        indices = to_int32_device(indices)
        indptr = to_int32_device(indptr)

        self._layout = layout
        self._arch = arch
        self._is_sm90 = arch == "sm90"
        self._bits = layout.bits
        self._k_quant = layout.k_quant
        self._v_quant = layout.v_quant
        self._q_data_type = q_data_type
        self._o_data_type = o_data_type
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._page_size = _QUANT_PAGE_SIZE
        self._sm_scale = sm_scale
        self._batch_size = batch_size
        self._seqlens = seq_lens
        self._table_offsets = indptr[:batch_size]
        self._page_table = indices
        self._max_kv_len = max_kv_len
        self._quant_kernel_kwargs = dict(layout._kernel_kwargs)

        if self._is_sm90:
            self._plan_quantized_sm90(indptr, indices, kv_splits, reduction)
        else:
            self._plan_quantized_sm100(kv_splits, reduction)
        self._planned = True

    def _plan_quantized_sm100(self, kv_splits, reduction) -> None:
        from ._sm100.fold_runtime import quant_cfg_of, reads_v_max_hist
        from ._sm100.item_loop import drops_item_loop
        from .quant_layout import quant_block_offsets
        from ._sm100.scheduler import plan_splits_sm100

        bits = self._bits
        k_quant, v_quant = self._k_quant, self._v_quant
        self._quant_offsets = quant_block_offsets(
            k_quant, v_quant, N=self._page_size, D=self._head_dim)

        # PLAN BEFORE COMPILE: the reduction mode is a CONSTRUCTOR argument of
        # IntDecodeSm100 (it selects the epilogue), so the planner's choice of
        # reduction has to be known before the trace. Doing this the other way
        # round is how the planner ends up unwired.
        # The planner costs per-TILE work, so it has to be told the tile the
        # kernel will actually be built with -- compile_fold resolves 128 for
        # V-ch64 while the planner's default is 256, and a planner costing
        # 256-token tiles against a 128-token kernel picks the wrong split
        # count.
        from ._sm100.fold_runtime import resolve_fold_tiling

        _seq_res, _ = resolve_fold_tiling(
            k_quant, v_quant, head_dim=self._head_dim, bits=bits
        )
        self._planned_splits, self._planned_reduction = plan_splits_sm100(
            self._batch_size, self._num_kv_heads, self._max_kv_len, bits=bits,
            sm_count=get_num_sm(self.device), sequence_tile=_seq_res,
            head_tiles=(self._num_qo_heads // self._num_kv_heads + 7) // 8)
        if kv_splits is None or kv_splits <= 0:
            kv_splits = self._planned_splits
            if reduction == "auto":
                reduction = self._planned_reduction
        if reduction == "auto":
            reduction = "none" if kv_splits == 1 else "kernel"
        if reduction == "atomic":
            raise NotImplementedError(
                "quantized decode has no validated atomic-reduction arm. The "
                "cluster/CGA reduce was A/B'd and rejected as a default "
                "(parity 2.3-4.3e-3 vs kernel-mode's 0.7-1.1e-3, and "
                "~1 us SLOWER at every small-batch cell); "
                "plan_splits_sm100(prefer_atomic=True) can still return it for "
                "a retest, but nothing here has been gated on it."
            )
        if reduction not in ("none", "kernel"):
            raise ValueError(
                f'quantized reduction must be "none" or "kernel"; got {reduction!r}')
        if reduction == "none" and kv_splits != 1:
            raise ValueError(
                f'reduction="none" requires kv_splits == 1, got {kv_splits}')

        self._kv_splits = kv_splits
        self._reduction = reduction
        self._has_workspace = reduction == "kernel"

        if self._has_workspace:
            o_partial, l_partial, m_partial = _slice_workspace(
                self._float_workspace_buffer, kv_splits, self._batch_size,
                1, self._num_qo_heads, self._head_dim)
            self._o_partial = o_partial
            self._l_partial = l_partial
            self._m_partial = m_partial

        # The layout's kernel keyword arguments (empty in production), plus the
        # granularity. At D128 whole-head token-wise K/V is left OUT on purpose:
        # the compile key, and therefore the traced kernel, is then byte-for-byte the
        # default build's. When the layout's scale_dtype is bfloat16 on a layout that
        # defaults to f32 scales, sdtype="bf16" is forwarded explicitly.
        # compile_fold's constructor defaults are explicitly token-128, so D64 and D256
        # must forward their whole-head pair even though quant_cfg_of derives it when omitted.
        kkw = dict(self._quant_kernel_kwargs)
        if self._head_dim != 128 or k_quant != ("token", self._head_dim) or v_quant != ("token", self._head_dim):
            kkw["k_quant"] = k_quant
            kkw["v_quant"] = v_quant
        if self._layout.scale_dtype == torch.bfloat16 and not (
            k_quant[0] == "token" and k_quant[1] < self._head_dim
        ):
            kkw["sdtype"] = "bf16"
        cfg_k, cfg_v, scale_sdtype, _ = quant_cfg_of(self._page_size, self._head_dim, kkw)
        assert (cfg_k, cfg_v) == (k_quant, v_quant), (
            f"quant_cfg_of drifted from the layout: "
            f"{cfg_k}/{cfg_v} != {k_quant}/{v_quant}")
        compiled_dtype = torch.bfloat16 if scale_sdtype == "bf16" else torch.float32
        assert compiled_dtype == self._layout.scale_dtype, (
            f"the kernel would read a {compiled_dtype} scale pool and the layout holds "
            f"{self._layout.scale_dtype}")
        # ONE WORK ITEM PER CTA. Without a split, the kernel is persistent by default: it
        # launches at most `persistent_sm_count` CTAs (the constructor's default, or the
        # layout's own value), and each loops over the (KV head, request) work items it is
        # given. A persistent build that reads the exponent table (the fold with token-wise
        # V) reads it before its first item's set-up barrier, which every role then waits
        # for. When the batch has no more items than the CTA cap, no CTA gets a second item,
        # so the plan compiles such a build without the item loop, where the warpgroups that
        # use the exponent read the table themselves while the first tile is produced. On
        # B200 that build runs 0.3 to 7.6% faster than the persistent one on such grids, with
        # bitwise the same output (INT2 and INT4; 32/8, 64/8, 24/8, 16/4, 8/2 and 4/1 heads;
        # token-wise and channel-wise K, K token-64; KV 63 to 8192). Channel-wise
        # V reads no table, and there the build without the loop measures up to 3.8% slower
        # at long requests, so it keeps the persistent build. A layout that names
        # `persistent` among its kernel arguments keeps its own choice. The predicate is
        # `_sm100/item_loop.py`.
        self._reads_v_max_hist = reads_v_max_hist(self._page_size, self._head_dim, kkw)
        ckw = dict(kkw)
        cap = int(ckw.get("persistent_sm_count", _persistent_sm_count()))
        if drops_item_loop(reduction, self._reads_v_max_hist, "persistent" in ckw,
                           self._batch_size * self._num_kv_heads, cap):
            ckw["persistent"] = False
        self._fold_key = (
            self._page_size,
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim,
            reduction,
            bits,
            _torch_to_cutlass(self._o_data_type),
            tuple(sorted(ckw.items())),
        )
        self._compiled_fold, applied = _get_compiled_fold_kernel(*self._fold_key)
        self._fold_variants = {(False, False): self._compiled_fold}
        self._assert_bits_tuning_applied(applied)

    def _assert_bits_tuning_applied(self, applied: Dict[str, Any]) -> None:
        """Prove the promoted per-bits tuning reached the object we compiled.

        ``_sm100/fold_runtime.compile_fold`` guards this, so that a promoted
        optimisation cannot be applied as a dead statement after the constructor
        and leave the kernel object without it. The wrapper shares the same
        ``BITS_TUNING`` dict (imported, never copied, so the two cannot drift)
        and therefore shares the guard.

        ``applied`` holds the knob values read back off the constructed kernel;
        it is cached alongside the compiled module so this still fires on a
        compile-cache HIT, where the trace-time assert does not re-run.
        """
        from ._sm100.fold_runtime import BITS_TUNING

        overrides = self._quant_kernel_kwargs
        want_all = BITS_TUNING.get(self._bits, {})
        assert set(applied) == set(want_all), (
            f"BITS_TUNING drift: compile_fold reported knobs {sorted(applied)} "
            f"for bits={self._bits}, wrapper expected {sorted(want_all)}")
        for key, want in want_all.items():
            if key in overrides and overrides[key] != want:
                continue                       # a layout overrode it deliberately
            assert applied[key] == want, (
                f"per-bits tuning did not reach the kernel: bits={self._bits} "
                f"{key} is {applied[key]!r}, expected {want!r}; see "
                f"_sm100/fold_runtime.BITS_TUNING")

    def _plan_quantized_sm90(self, indptr, indices, kv_splits, reduction) -> None:
        """Plan ``IntPingPongDecodeSm90``. Validated on H100.

        See ``_sm90/pingpong_runtime.py``'s module docstring for what was checked.

        WHY ``kv_splits`` IS HONOURED HERE
        ----------------------------------
        A caller's ``kv_splits`` and ``reduction`` reach the plan, so that
        ``plan(kv_splits=4)`` is what launches. Two things depend on it, both of
        which the SM100 path also supports:

          * a correctness gate must reach the split counts the planner does not
            happen to choose, because "only wrong at outer > 1" is exactly this
            path's exposure;
          * the planner must be SCORABLE, because scoring it means timing the
            same problem at every candidate split and comparing.

        REDUCTION IS DERIVED, NOT CHOSEN, and that is an arch difference worth
        stating: on SM100 the reduction mode selects a compile-time epilogue,
        while ``IntPingPongDecodeSm90`` always writes ``o_partial``/``lse`` and the
        merge is either its own fused tail or the external ``SplitKCombine``.
        So ``outer == 1`` IS ``"none"`` and ``outer > 1`` IS ``"kernel"``; a
        caller asking for the other pairing is asking for something that does
        not exist, and gets an error rather than a silent substitution.

        The third axis -- fused tail vs external combine -- is a real choice
        (``plan_fused_combine`` is a POLICY, not a constraint). A layout reaches it
        through ``QuantKVLayout._create(..., kernel_kwargs={"fused_combine": bool})``;
        production layouts carry nothing.

        The scale-pool precision is the layout's ``scale_dtype``, which ``create()`` checked
        against the granularity (token-wise K over D/2 needs bfloat16, D in (64, 256) needs
        float32).
        """
        from .quant_layout import quant_block_offsets
        from ._sm90.pingpong_runtime import (
            alloc_pingpong_workspace, block_table_2d, plan_pingpong,
        )
        from ._sm90.scheduler import plan_fused_combine

        fused_override = self._quant_kernel_kwargs.get("fused_combine")
        self._quant_offsets = quant_block_offsets(
            self._k_quant, self._v_quant, N=self._page_size, D=self._head_dim)
        self._sm90_scale_dtype = self._layout.scale_dtype
        planned_outer, planned_minb, planned_fused = plan_pingpong(
            self._batch_size, self._num_kv_heads, self._max_kv_len,
            bits=self._bits, head_dim=self._head_dim, sm_count=get_num_sm(self.device),
            head_tiles=(self._num_qo_heads // self._num_kv_heads + 7) // 8)
        self._planned_splits = planned_outer
        self._planned_reduction = "none" if planned_outer == 1 else "kernel"

        if kv_splits is None or kv_splits <= 0:
            outer, min_blocks, fused = planned_outer, planned_minb, planned_fused
        else:
            outer = int(kv_splits)
            # min_blocks_per_sm is ARCHITECTURAL on pingpong (the 32/104
            # setmaxnregs split is built into the kernel), so it does not vary
            # with the split count -- plan_splits returns 2 unconditionally.
            # Take the planner's value rather than hardcode.
            min_blocks = planned_minb
            fused = plan_fused_combine(
                self._batch_size, self._num_kv_heads, outer,
                head_tiles=(self._num_qo_heads // self._num_kv_heads + 7) // 8,
                head_dim=self._head_dim)
        if fused_override is not None:
            fused = bool(fused_override) and outer > 1
        if reduction == "atomic":
            raise NotImplementedError(
                "IntPingPongDecodeSm90 has no atomic-reduction arm; the split-K "
                "merge is either its fused tail or the external SplitKCombine")
        derived = "none" if outer == 1 else "kernel"
        if reduction not in ("auto", derived):
            raise ValueError(
                f'on SM90 the reduction mode is derived from the split count, '
                f'not chosen: kv_splits={outer} implies reduction="{derived}", '
                f'got reduction="{reduction}"')
        self._kv_splits = outer
        self._reduction = derived
        self._has_workspace = False        # SM90 owns its own partial buffers
        self._sm90_min_blocks = min_blocks
        self._sm90_fused_combine = fused
        self._sm90_block_table = block_table_2d(indptr, indices, self._batch_size,
                                                device=self.device)
        self._sm90_ws = alloc_pingpong_workspace(
            self._batch_size, self._num_qo_heads, self._num_kv_heads,
            self._head_dim, outer, device=self.device)
        self._sm90_compile_args = (
            self._page_size, self._num_qo_heads, self._num_kv_heads,
            self._head_dim, self._bits, min_blocks, fused,
            tuple(self._k_quant), tuple(self._v_quant),
            self._sm90_scale_dtype, self._o_data_type)
        self._compiled_fold = _get_compiled_pingpong_kernel(*self._sm90_compile_args)
        # The external combine runs ONLY when the in-kernel fused tail is off.
        self._compiled_combine = (
            _get_compiled_splitk_combine(self._head_dim, self._o_data_type)
            if outer > 1 and not fused else None)
        self._sm90_lse_merge = None

    def pscale_lp(self, *args, **kwargs):
        """Always raises: ``run(q, kv_cache)`` derives the Blackwell P-scale exponent
        itself, so there is no value to precompute or pass."""
        raise TypeError(
            "PagedDecodeWrapper.pscale_lp() is retired: run(q, kv_cache) derives the "
            "Blackwell P-scale exponent itself, so there is no value to precompute or pass")

    def _check_tensor(self, name, x, dtype):
        """The query or output: ``[B, HQ, D]`` or ``[B, 1, HQ, D]`` with the planned B, HQ and
        D exactly, C-contiguous, 16-byte aligned, on this wrapper's device. Returns True for
        the 3-D form. Constant time; it never reshapes by element count, so a ``[1, 16, 256]``
        query cannot pass a plan for HQ=32, D=128."""
        if x.dtype != dtype:
            raise ValueError(f"{name}.dtype={x.dtype} mismatches the planned {dtype}")
        if x.device != self.device:
            raise ValueError(f"{name} is on {x.device}; this wrapper plans for {self.device}")
        b, hq, d = self._batch_size, self._num_qo_heads, self._head_dim
        shape = x.shape
        if shape == (b, hq, d):
            three_d = True
        elif shape == (b, 1, hq, d):
            three_d = False
        else:
            raise ValueError(
                f"{name} has shape {tuple(shape)}; the plan is for [{b}, {hq}, {d}] or "
                f"[{b}, 1, {hq}, {d}] (batch, query heads, head dimension)")
        if not x.is_contiguous():
            raise ValueError(f"{name} must be C-contiguous; got strides {tuple(x.stride())}")
        if x.data_ptr() % 16:
            raise ValueError(f"{name} must start 16-byte aligned")
        return three_d

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        kv_cache: QuantKVCache = None,
        *retired_positional,
        out: Optional[torch.Tensor] = None,
        sm_scale: Optional[float] = None,
        o_scale: Optional[float] = None,
        enable_pdl: bool = True,
        lse: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None,
        **retired,
    ) -> torch.Tensor:
        """Decode one layer: attend each request's query over its pages of ``kv_cache``.

        Parameters
        ----------
        q : torch.Tensor
            ``[batch_size, num_qo_heads, head_dim]`` or ``[batch_size, 1, num_qo_heads,
            head_dim]`` with exactly the planned sizes, C-contiguous and 16-byte aligned, in the
            planned query dtype (fp8 e4m3 on SM100, bf16 on SM90). A query sliced out of a fused
            QKV projection is not contiguous; pass ``q.contiguous()``.
        kv_cache : QuantKVCache
            The layer's cache, built by ``layout.allocate`` or ``layout.bind`` for the layout
            this plan was made with; a cache of any other layout is refused, even one whose
            pools have the same shapes. It replaces the ``k_cache``, ``v_cache``,
            ``scale_pool=``, ``pscale_lp=`` and ``k_pool_order=`` arguments, each of which
            raises: the pools, their orders and the scale dtype all come from the cache's
            layout, which ``create()`` resolved from the reader's own predicates, so there is
            nothing left for a caller to declare.
        out : Optional[torch.Tensor]
            Keyword only. The output buffer, with the query's batch, head and dimension
            sizes in either form, C-contiguous, 16-byte aligned, in the planned output dtype.
            ``None`` allocates one per call; a serving loop should pass its own.
        sm_scale : Optional[float]
            Per-call override of the softmax scale set at plan() time.
        o_scale : Optional[float]
            Output scale applied to the final O before it is written, in the kernel's
            epilogue. Defaults to 1.0.
        enable_pdl : bool
            Programmatic dependent launch on SM100.
        lse : Optional[torch.Tensor]
            Keyword only. An fp32 ``[batch_size, num_qo_heads]`` tensor, C-contiguous, 16-byte
            aligned, on this wrapper's device, that receives each query row's log-sum-exp in
            the NATURAL log: the log of the normalizer the kernel divides O by,
            ``LSE[b, h] = ln sum_j exp(z_j)`` over the kernel's own scores ``z_j``, so merging
            ``(out, lse)`` with another attention state by the natural-log rule (SGLang's
            ``merge_state``) reproduces this decode's output exactly. With token-wise K the
            scores are ``z_j = sm_scale * d_{b,h} * (q_{b,h} . k_j)``: ``sm_scale`` the call's
            softmax scale, ``d`` the SM100 query descale (1 without one), ``q`` the query the
            kernel reads (fp8 e4m3 on SM100, bf16 on SM90) and ``k_j`` the dequantized key.
            With channel-wise K the kernel first multiplies the query by each channel's K scale
            and rounds the product to its MMA operand (e4m3 on SM100; bf16 on SM90, with the
            softmax scale in it), so its scores, and the LSE, differ from that formula by that
            rounding. ``o_scale`` is in neither. Every
            request must have ``seq_len >= 1``:
            a row of length 0 gets an unspecified LSE (and, in SM100's unsplit builds, an
            unspecified output row). The first call with ``lse`` compiles the variant that
            stores it, where one is needed (docs/integration.md); SM90's one-split plans
            return it from the plan's own kernel.
        q_descale : Optional[torch.Tensor]
            SM100 only: an fp32 device tensor ``[batch_size, num_qo_heads]``, any strides (a
            per-tensor value may be a stride-0 expand), with the effective query
            ``q_fp8 * q_descale``. SM90 reads a bf16
            query and refuses it. Every factor must be finite and strictly positive: the
            kernel takes each row's score maximum before it applies the factor, so a zero,
            negative or non-finite factor gives wrong output without an error (a value check
            would wait for the device, so none is made). With ``q_descale``, ``sm_scale`` stays
            the model's attention scale; fold a per-tensor descale into ``sm_scale`` only when
            ``q_descale`` is None, or it is applied twice.

        Returns the output: ``out`` when passed, else a new tensor shaped like ``q``.
        """
        if retired_positional or retired or not isinstance(kv_cache, QuantKVCache):
            _refuse_retired(
                "PagedDecodeWrapper.run", retired_positional, retired, _RETIRED_RUN_ARGS,
                "build the cache with layout.allocate() or layout.bind() and call "
                "run(q, kv_cache, out=...)")
            raise TypeError(f"PagedDecodeWrapper.run: kv_cache must be a QuantKVCache; got "
                            f"{type(kv_cache).__name__}. The tensor form run(q, k_cache, "
                            f"v_cache, scale_pool=, pscale_lp=, k_pool_order=) is retired")
        if not self._planned:
            raise RuntimeError("Call plan() before run().")
        # IDENTITY FIRST, so the usual case costs one pointer comparison; the field-wise
        # comparison of two frozen dataclasses is the fallback, and it tells apart layouts
        # whose pools have identical shapes (K channel-64 against channel-128).
        layout = kv_cache.layout
        if layout is not self._layout and layout != self._layout:
            raise ValueError(
                f"kv_cache was built for another layout than this plan's:\n  cache: {layout}\n"
                f"  plan:  {self._layout}")
        if kv_cache.device != self.device:
            raise ValueError(f"kv_cache is on {kv_cache.device}; this wrapper plans for "
                             f"{self.device}")
        out_was_3d = self._check_tensor("q", q, self._q_data_type)
        if lse is not None:
            # The kernels store [B, HQ] through a compact view and assume 16-byte alignment.
            if (lse.dtype != torch.float32 or lse.device != self.device
                    or lse.shape != (self._batch_size, self._num_qo_heads)
                    or not lse.is_contiguous() or lse.data_ptr() % 16):
                raise ValueError(
                    f"lse must be a C-contiguous, 16-byte aligned fp32 "
                    f"[{self._batch_size}, {self._num_qo_heads}] tensor on {self.device}; got "
                    f"{lse.dtype} {tuple(lse.shape)} strides {tuple(lse.stride())} on "
                    f"{lse.device}")
        if out is not None:
            self._check_tensor("out", out, self._o_data_type)
        if q_descale is not None:
            if self._is_sm90:
                raise ValueError(
                    "q_descale is for SM100's fp8 query; the SM90 decode reads a bf16 query")
            if q_descale.dtype != torch.float32 or q_descale.device != self.device \
                    or q_descale.shape != (self._batch_size, self._num_qo_heads):
                raise ValueError(
                    f"q_descale must be an fp32 [{self._batch_size}, {self._num_qo_heads}] "
                    f"tensor on {self.device}; got {q_descale.dtype} "
                    f"{tuple(q_descale.shape)} on {q_descale.device}")
        scale_s = self._sm_scale if sm_scale is None else sm_scale
        scale_o = 1.0 if o_scale is None else o_scale
        with _utils.on_device(self.device):
            return self._run_quantized(q, kv_cache, out, scale_s, scale_o, enable_pdl,
                                       out_was_3d, q_descale, lse)

    def update(self, lengths) -> None:
        """Graph mode's per-step call (`_GraphDecodeWrapper.update`); an eager wrapper plans
        each step with `plan()` instead."""
        raise RuntimeError("update() is graph mode's per-step call: build the wrapper with "
                           "use_cuda_graph=True and plan() it first")

    def _run_quantized(self, q, kv_cache, out, scale_s, scale_o, enable_pdl,
                       out_was_3d, q_descale=None, lse=None) -> torch.Tensor:
        """The launch, after `run()`'s checks: SM90 through `_run_quantized_sm90`, SM100 here."""
        if self._is_sm90:
            return self._run_quantized_sm90(q, kv_cache, out, scale_s, scale_o, out_was_3d,
                                            lse)
        from ._sm100.fold_runtime import launch_fold

        b, hq, d = self._batch_size, self._num_qo_heads, self._head_dim
        q_bshd = q.view(b, 1, hq, d)
        out_4d = (torch.empty((b, 1, hq, d), dtype=self._o_data_type, device=self.device)
                  if out is None else out.view(b, 1, hq, d))

        o_partial = l_partial = m_partial = None
        if self._has_workspace:
            o_partial = self._o_partial
            l_partial, m_partial = self._l_partial, self._m_partial

        # THE VARIANT FOLLOWS FROM THE CALL (`_fold_variant`): the plan's own kernel, or the one
        # that reads a query descale, returns the log-sum-exp, or both.
        variant = (q_descale is not None, lse is not None)
        compiled = (self._compiled_fold if variant == (False, False)
                    else self._fold_variant(*variant))
        launch_fold(
            compiled,
            kv_splits=self._kv_splits,
            seqlens=self._seqlens,
            table_offsets=self._table_offsets,
            page_table=self._page_table,
            q_bshd=q_bshd,
            o_bshd=out_4d,
            o_partial=o_partial, l_partial=l_partial, m_partial=m_partial,
            scale_s=scale_s, scale_o=scale_o,
            mKq=kv_cache.k_codes, mVq=kv_cache.v_codes, mScale=kv_cache.scales,
            v_max_hist=kv_cache.v_max_hist if self._reads_v_max_hist else None,
            q_descale=q_descale,
            lse=None if lse is None else lse.view(b, 1, hq),
            enable_pdl=enable_pdl,
        )
        if out is not None:
            return out
        return out_4d.view(b, hq, d) if out_was_3d else out_4d

    def _fold_variant(self, has_q_descale: bool, has_lse: bool):
        """The plan's SM100 kernel in the variant a call asks for, compiled on first use and
        kept in `_fold_variants`: at most four objects per plan, the plan's own and the ones
        that read a query descale, return the log-sum-exp, or both. A split build's decode
        kernel is the same either way; only its reduction kernel stores the LSE
        (`IntDecodeSm100.__call__`)."""
        key = (has_q_descale, has_lse)
        compiled = self._fold_variants.get(key)
        if compiled is None:
            compiled, _ = _get_compiled_fold_kernel(
                *self._fold_key, has_q_descale=has_q_descale, has_lse=has_lse)
            self._fold_variants[key] = compiled
        return compiled

    def _sm90_lse_kernel(self):
        """The SM90 split plan's merge that returns the log-sum-exp, compiled on first use and
        kept in `_sm90_lse_merge`: the fused-combine decode built with `has_lse` when the plan
        fuses its combine, else `SplitKCombineLse`, which runs after the plan's own decode."""
        if self._sm90_lse_merge is None:
            self._sm90_lse_merge = (
                _get_compiled_pingpong_kernel(*self._sm90_compile_args, True)
                if self._sm90_fused_combine else
                _get_compiled_splitk_combine(self._head_dim, self._o_data_type, True))
        return self._sm90_lse_merge

    def _run_quantized_sm90(self, q, kv_cache, out, scale_s, scale_o,
                            out_was_3d, lse=None) -> torch.Tensor:
        """Validated on H100 SXM5.

        The SM90 kernel's tensors are 3-D ``[B, HQ, D]`` (no prediction axis)
        and its query is bf16, so this is not a shape-for-shape copy of the
        SM100 path. It launches through ``pingpong_runtime.launch_pingpong``.

        ``lse`` takes one of three routes, each the smallest change to its plan. A one-split
        plan's decode already writes every split's natural-log LSE, so it is handed the
        caller's tensor viewed ``[1, B, HQ]`` in place of the workspace's, and the kernel is
        the plan's own. A split plan returns it from its merge: the fused-combine decode built
        with ``has_lse``, or, after the plan's own decode, ``SplitKCombineLse``.
        """
        from ._sm90.pingpong_runtime import launch_pingpong

        b, hq, d = self._batch_size, self._num_qo_heads, self._head_dim
        q_bhd = q.view(b, hq, d)
        out_bhd = (torch.zeros((b, hq, d), dtype=self._o_data_type, device=self.device)
                   if out is None else out.view(b, hq, d))
        ws = self._sm90_ws
        compiled, combine, split_lse, lse_out = (
            self._compiled_fold, self._compiled_combine, ws["lse"], None)
        if lse is not None:
            if self._kv_splits == 1:
                split_lse = lse.view(1, b, hq)
            elif self._sm90_fused_combine:
                compiled, lse_out = self._sm90_lse_kernel(), lse
            else:
                combine, lse_out = self._sm90_lse_kernel(), lse
        launch_pingpong(
            compiled, combine,
            q_bhd=q_bhd, mKq=kv_cache.k_codes, mVq=kv_cache.v_codes, mScale=kv_cache.scales,
            block_table=self._sm90_block_table, seq_lens=self._seqlens,
            o_partial=ws["o_partial"], out_bhd=out_bhd, lse=split_lse,
            counters=ws["counters"], scale_s=scale_s, scale_o=scale_o,
            outer_splits=self._kv_splits, num_kv_heads=self._num_kv_heads,
            batch=b, lse_out=lse_out,
        )
        if out is not None:
            return out
        return out_bhd if out_was_3d else out_bhd.view(b, 1, hq, d)


class _GraphDecodeWrapper(PagedDecodeWrapper):
    """CUDA-graph mode (module docstring): what ``PagedDecodeWrapper(workspace,
    use_cuda_graph=True)`` builds. It shares eager mode's `run()` checks, kernels' compile
    arguments and unsplit builds, and replaces only `plan()`, `update()` and the launch, so
    that eager mode's own `plan()` and `run()` run no graph-mode code."""

    def __init__(self, float_workspace_buffer: torch.Tensor, use_cuda_graph: bool = True) -> None:
        super().__init__(float_workspace_buffer, use_cuda_graph)
        #: the bucket's `_GraphPlan`, set by its one `plan()` and kept for the wrapper's life
        self._graph = None

    @flashinfer_api
    def plan(self, indptr=None, indices=None, seq_lens=None, num_qo_heads=None, layout=None,
             *retired_positional, q_data_type=None, o_data_type=None, sm_scale=None,
             kv_splits=None, reduction="auto", q_len_per_req=1, max_kv_len=None,
             non_blocking=True, page_table=None, lse=False, q_descale=False,
             **retired) -> None:
        """Graph mode's plan: ONE per wrapper, for one captured batch size (a bucket).

        ``page_table`` (int32 ``[B, max_pages]``, row-major, row stride ``max_pages``) and
        ``seq_lens`` (int32 ``[B]``) are the engine's own device buffers, which it keeps in step
        with the lengths it passes to `update`; their addresses are recorded and no value is
        read. ``max_kv_len`` is the bucket's largest context, which every `update` length must
        stay within. ``lse`` and ``q_descale`` declare whether every `run` passes them, so that
        exactly the builds the bucket launches are compiled here and none inside a capture.

        It decides the bucket's build class (`graph_plan.Bucket`): a bucket that can split runs
        the graph split build over a work list, and any other bucket runs eager's unsplit build
        for its batch, reading the engine's buffers in place. It allocates the bucket's private
        buffers (the device work list, the pinned host buffers, Hopper's fused-combine
        counters), carves its scratch from the shared workspace (`graph_workspace_bytes`),
        writes a one-chunk-per-request plan at length 1 so that a warm-up or capture before the
        first `update` runs correctly, and waits for the device once, at set-up.

        A second call raises, so that nothing a captured graph points at is ever freed or
        replaced while the wrapper lives, and so does a call under stream capture, before any
        set-up.
        """
        if self._graph is not None:
            raise RuntimeError(
                "a graph-mode wrapper plans once: a captured graph keeps the addresses of its "
                "work list, counters and scratch for the wrapper's life; build one wrapper per "
                "captured batch size")
        # Before any set-up: plan() compiles, allocates and waits for the device once, none of
        # which a stream capture may contain.
        with _utils.on_device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "plan() runs once, before capture: it compiles, allocates and waits for the "
                    "device, which a stream capture may not contain")
        if retired_positional or retired or not isinstance(layout, QuantKVLayout):
            _refuse_retired(
                "PagedDecodeWrapper.plan", retired_positional, retired, _RETIRED_PLAN_ARGS,
                "graph mode plans with plan(page_table=, seq_lens=, num_qo_heads=, layout=, "
                "max_kv_len=)")
            raise TypeError(f"PagedDecodeWrapper.plan: layout must be a QuantKVLayout; got "
                            f"{type(layout).__name__}")
        if indptr is not None or indices is not None:
            raise TypeError("graph mode's plan() takes the engine's page_table=, not indptr and "
                            "indices")
        if kv_splits is not None or reduction != "auto":
            raise TypeError("graph mode plans each step's splits in update(); kv_splits= and "
                            "reduction= are eager mode's")
        if non_blocking is not True:
            raise TypeError("non_blocking= is eager mode's; graph mode copies nothing at plan()")
        if q_len_per_req != 1:
            raise NotImplementedError(
                f"quantized decode is plain decode only; got q_len_per_req={q_len_per_req}")
        for name, flag in (("lse", lse), ("q_descale", q_descale)):
            if not isinstance(flag, bool):
                raise TypeError(f"graph mode's {name}= declares whether every run() passes "
                                f"{name}; it is a bool, got {flag!r}")
        arch = _utils.require_arch(self.device)
        if layout.arch != arch:
            raise ValueError(
                f"the layout was resolved for {layout.arch} and this wrapper's device "
                f"{self.device} is {arch}; create the layout for the device that decodes it")
        _utils.require_dsl_target(arch)
        _check_num_qo_heads(arch, num_qo_heads, layout.num_kv_heads)
        if q_descale and arch == "sm90":
            raise ValueError("q_descale is for SM100's fp8 query; the SM90 decode reads bf16")
        for name, t, rank in (("page_table", page_table, 2), ("seq_lens", seq_lens, 1)):
            if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or t.dim() != rank \
                    or t.device != self.device or not t.is_contiguous():
                raise ValueError(
                    f"graph mode's {name} is the engine's contiguous int32 {rank}-D buffer on "
                    f"{self.device}; got "
                    + (f"{t.dtype} {tuple(t.shape)} strides {tuple(t.stride())} on {t.device}"
                       if isinstance(t, torch.Tensor) else type(t).__name__))
        batch, max_pages = page_table.shape
        if batch == 0 or seq_lens.shape != (batch,):
            raise ValueError(f"page_table holds {batch} rows and seq_lens {tuple(seq_lens.shape)}; "
                             f"a bucket has one length per row, and at least one row")
        if batch * max_pages > 2**31 - 1:
            raise ValueError(f"the page table's {batch} x {max_pages} entries exceed int32 offsets")
        if (isinstance(max_kv_len, bool) or not isinstance(max_kv_len, int)
                or not 1 <= max_kv_len <= min(_GRAPH_MAX_KV_LEN, _QUANT_PAGE_SIZE * max_pages)):
            raise ValueError(
                f"max_kv_len is the bucket's largest context, an int in [1, "
                f"{min(_GRAPH_MAX_KV_LEN, _QUANT_PAGE_SIZE * max_pages)}] (at most 2^31 - 64, "
                f"and the {max_pages} pages a page-table row holds); got {max_kv_len!r}")
        q_data_type, o_data_type = _resolve_io_dtypes(arch, layout, q_data_type, o_data_type)
        bucket = _graph_bucket(arch, layout, batch, max_kv_len, get_num_sm(self.device),
                               num_qo_heads)
        graph = _GraphPlan(bucket, page_table, seq_lens, lse, q_descale)
        try:
            self._layout, self._arch, self._is_sm90 = layout, arch, arch == "sm90"
            self._bits, self._k_quant, self._v_quant = layout.bits, layout.k_quant, layout.v_quant
            self._q_data_type, self._o_data_type = q_data_type, o_data_type
            self._num_qo_heads, self._num_kv_heads = num_qo_heads, layout.num_kv_heads
            self._head_dim, self._page_size = layout.head_dim, _QUANT_PAGE_SIZE
            self._sm_scale = layout.head_dim**-0.5 if sm_scale is None else sm_scale
            self._batch_size, self._max_kv_len = batch, max_kv_len
            self._quant_kernel_kwargs = dict(layout._kernel_kwargs)
            with _utils.on_device(self.device):
                if self._is_sm90:
                    self._plan_graph_sm90(graph)
                else:
                    self._plan_graph_sm100(graph)
                if bucket.split_capable:
                    self._graph_buffers(graph)
                torch.cuda.synchronize(self.device)
        except Exception:
            self._invalidate_plan()
            raise
        self._graph = graph
        self._planned = True

    def _graph_scratch(self, words):
        """The first `words` f32 words of the shared workspace, refusing a workspace too small."""
        workspace = self._float_workspace_buffer.view(torch.float32)
        if workspace.numel() < words:
            raise RuntimeError(
                f"this graph bucket's launch uses {4 * words} bytes of workspace, and the "
                f"workspace holds {workspace.nbytes}; size it with graph_workspace_bytes()")
        return workspace[:words]

    def _plan_graph_sm90(self, graph) -> None:
        """Hopper's builds and scratch for a graph bucket: eager's unsplit build, or the graph split
        build with the merge eager would use at the same batch, the fused last-CTA combine at
        B * HKV <= 16 and the external combine above (`scheduler.plan_fused_combine`)."""
        from ._sm90.scheduler import plan_fused_combine

        bucket, b, hq, d = graph.bucket, self._batch_size, self._num_qo_heads, self._head_dim
        fused = False
        if bucket.split_capable:
            fused = plan_fused_combine(b, self._num_kv_heads, bucket.s_tight, head_dim=d,
                                       head_tiles=bucket.head_tiles)
            override = self._quant_kernel_kwargs.get("fused_combine")
            if override is not None:
                fused = bool(override)
        args = (self._page_size, hq, self._num_kv_heads, d, self._bits,
                graph_plan.SPLIT_CTAS_PER_SM["sm90"], fused, tuple(self._k_quant),
                tuple(self._v_quant), self._layout.scale_dtype, self._o_data_type)
        if bucket.split_capable:
            graph.decode = _get_compiled_pingpong_graph_kernel(*args, graph.lse and fused)
            if not fused:
                graph.combine = _get_compiled_splitk_combine_graph(d, self._o_data_type,
                                                                   graph.lse)
        else:
            graph.decode = _get_compiled_pingpong_kernel(*args)
        self._sm90_fused_combine = fused
        scratch = self._graph_scratch(bucket.s_tight * b * hq * (d + 1))
        graph.o_partial = scratch[:bucket.s_tight * b * hq * d].view(bucket.s_tight, b, hq, d)
        graph.split_lse = scratch[bucket.s_tight * b * hq * d:].view(bucket.s_tight, b, hq)
        graph.counters = torch.zeros(b * self._num_kv_heads * bucket.head_tiles,
                                     dtype=torch.int32, device=self.device)

    def _plan_graph_sm100(self, graph) -> None:
        """Blackwell's builds for a graph bucket: eager's unsplit build for the batch, planned by
        eager's own step (`_plan_quantized_sm100` at one split), persistent or one item per CTA
        by eager's rule, which reads the engine's page table through `table_offsets[b] = b *
        max_pages`; or the graph split build with the reduction kernel
        (`_get_compiled_fold_graph_kernel`), which computes that offset itself, its partials at
        the bucket's split capacity carved from the shared workspace."""
        from ._sm100.fold_runtime import reads_v_max_hist

        bucket = graph.bucket
        if not bucket.split_capable:
            self._plan_quantized_sm100(1, "none")
            graph.decode = (self._fold_variant(graph.q_descale, graph.lse)
                            if graph.q_descale or graph.lse else self._compiled_fold)
            graph.table_offsets = torch.arange(self._batch_size, dtype=torch.int32,
                                               device=self.device) * graph.page_table.shape[1]
            return
        # the layout's kernel arguments, as eager's plan passes them (`_plan_quantized_sm100`)
        kkw = dict(self._quant_kernel_kwargs)
        if self._head_dim != 128 or (self._k_quant, self._v_quant) != (("token", self._head_dim),) * 2:
            kkw["k_quant"], kkw["v_quant"] = self._k_quant, self._v_quant
        if self._layout.scale_dtype == torch.bfloat16 and not (
            self._k_quant[0] == "token" and self._k_quant[1] < self._head_dim
        ):
            kkw["sdtype"] = "bf16"
        self._reads_v_max_hist = reads_v_max_hist(self._page_size, self._head_dim, kkw)
        graph.decode, applied = _get_compiled_fold_graph_kernel(
            self._page_size, self._num_qo_heads, self._num_kv_heads, self._head_dim, self._bits,
            _torch_to_cutlass(self._o_data_type), tuple(sorted(kkw.items())), graph.q_descale,
            graph.lse)
        self._assert_bits_tuning_applied(applied)
        graph.o_partial, graph.l_partial, graph.m_partial = _slice_workspace(
            self._float_workspace_buffer, bucket.s_tight, self._batch_size, 1,
            self._num_qo_heads, self._head_dim)

    def _graph_buffers(self, graph) -> None:
        """A split bucket's device work list, holding the one-chunk plan at length 1 until the
        first `update`, and the pinned host buffers each `update` writes into in turn."""
        bucket = graph.bucket
        words = graph_plan.words(bucket.wave, bucket.batch)
        ones = np.ones(bucket.batch, dtype=np.int64)
        initial = np.empty(words, dtype=np.int32)
        bucket.write(initial, ones, bucket.chunks(ones))
        graph.work = torch.from_numpy(initial).to(self.device)
        for _ in range(_GRAPH_HOST_BUFFERS):
            host = _pinned_words(words)
            graph.host.append(host)
            graph.host_words.append(host.numpy())
            graph.events.append(torch.cuda.Event())

    def update(self, lengths) -> None:
        """Graph mode, once per step, outside the graph and before its replay: plan the step from
        its host `lengths` (one integer per request of the bucket, each in [1, max_kv_len]; a list,
        a numpy array or a CPU tensor) and copy the work list to the device.

        It runs on the stream the graph replays on, the same stream at every call, and refuses
        to run under stream capture, because a captured update would replay one frozen plan.
        It never waits for the device as a whole: it writes the plan into the next of
        `_GRAPH_HOST_BUFFERS` pinned buffers, waiting first for that buffer's previous copy, and
        enqueues one asynchronous copy of the whole work list. A bucket that cannot split has no
        work list and reads the lengths from the engine's `seq_lens`, so its `update` checks the
        lengths and the stream, and copies nothing. One update serves every layer of the step.
        """
        graph = self._graph
        if graph is None:
            raise RuntimeError("update() follows the graph-mode wrapper's plan(); plan() it "
                               "first")
        with _utils.on_device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "update() runs outside the graph, before each replay: under capture it "
                    "would freeze one step's plan into every replay")
            lengths = _graph_host_lengths(lengths, self._batch_size, self._max_kv_len)
            stream = torch.cuda.current_stream()
            if graph.stream is None:
                graph.stream = stream
            elif stream != graph.stream:
                raise RuntimeError(
                    "update() runs on the stream the graph replays on, the stream of its first "
                    f"call ({graph.stream}); this call is on {stream}")
            if graph.work is None:
                return
            slot = graph.next_host
            graph.next_host = (slot + 1) % _GRAPH_HOST_BUFFERS
            if graph.recorded[slot]:
                graph.events[slot].synchronize()
            bucket = graph.bucket
            splits = bucket.chunks(lengths)
            if graph.host_splits[slot] is not None and np.array_equal(graph.host_splits[slot],
                                                                       splits):
                bucket.write_lengths(graph.host_words[slot], lengths, splits)
            else:
                bucket.write(graph.host_words[slot], lengths, splits)
                graph.host_splits[slot] = splits
            graph.work.copy_(graph.host[slot], non_blocking=True)
            graph.events[slot].record(stream)
            graph.recorded[slot] = True

    def _run_quantized(self, q, kv_cache, out, scale_s, scale_o, enable_pdl,
                       out_was_3d, q_descale=None, lse=None) -> torch.Tensor:
        """Graph mode's launch, after `run()`'s checks, which this mode shares with eager's:
        only host comparisons and the launches of the builds `plan()` compiled, so that a
        captured call allocates, compiles and reads nothing. ``out`` is required, and the call
        must pass exactly the variants the plan declared."""
        graph = self._graph
        if out is None:
            raise ValueError("graph mode's run() needs out=: a captured call writes a buffer the "
                             "engine owns, and allocates nothing")
        if (lse is not None) != graph.lse or (q_descale is not None) != graph.q_descale:
            raise ValueError(
                f"this bucket was planned with lse={graph.lse} and q_descale={graph.q_descale}, "
                f"and plan() compiled only those builds; this call passes "
                f"lse={'a tensor' if lse is not None else None} and "
                f"q_descale={'a tensor' if q_descale is not None else None}")
        b, hq, d = self._batch_size, self._num_qo_heads, self._head_dim
        if self._is_sm90:
            from ._sm90.pingpong_runtime import launch_pingpong, launch_pingpong_graph

            q_bhd, out_bhd = q.view(b, hq, d), out.view(b, hq, d)
            pools = dict(mKq=kv_cache.k_codes, mVq=kv_cache.v_codes, mScale=kv_cache.scales)
            if graph.work is not None:
                slots, splits = graph.work_views()
                launch_pingpong_graph(
                    graph.decode, graph.combine, q_bhd=q_bhd, **pools,
                    block_table=graph.page_table, o_partial=graph.o_partial, out_bhd=out_bhd,
                    lse=graph.split_lse, counters=graph.counters, work=slots, splits=splits,
                    scale_s=scale_s, scale_o=scale_o, num_kv_heads=self._num_kv_heads,
                    lse_out=lse)
            else:
                # eager's one-split launch, whose decode stores the LSE where it is pointed
                launch_pingpong(
                    graph.decode, None, q_bhd=q_bhd, **pools, block_table=graph.page_table,
                    seq_lens=graph.seq_lens, o_partial=graph.o_partial, out_bhd=out_bhd,
                    lse=graph.split_lse if lse is None else lse.view(1, b, hq),
                    counters=graph.counters, scale_s=scale_s, scale_o=scale_o, outer_splits=1,
                    num_kv_heads=self._num_kv_heads, batch=b)
            return out
        from ._sm100.fold_runtime import launch_fold, launch_fold_graph

        common = dict(
            page_table=graph.page_table.view(-1), q_bshd=q.view(b, 1, hq, d),
            o_bshd=out.view(b, 1, hq, d), scale_s=scale_s, scale_o=scale_o,
            mKq=kv_cache.k_codes, mVq=kv_cache.v_codes, mScale=kv_cache.scales,
            v_max_hist=kv_cache.v_max_hist if self._reads_v_max_hist else None,
            q_descale=q_descale, lse=None if lse is None else lse.view(b, 1, hq),
            enable_pdl=enable_pdl)
        if graph.work is not None:
            slots, splits = graph.work_views()
            launch_fold_graph(graph.decode, seqlens=graph.seq_lens, o_partial=graph.o_partial,
                              l_partial=graph.l_partial, m_partial=graph.m_partial, work=slots,
                              split_counts=splits, max_pages=graph.page_table.shape[1],
                              **common)
        else:
            launch_fold(graph.decode, kv_splits=1, seqlens=graph.seq_lens,
                        table_offsets=graph.table_offsets, **common)
        return out
