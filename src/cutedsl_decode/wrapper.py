# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Donglin Zhuang (standalone cutedsl-decode adaptation).
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer's CuTe DSL paged-decode wrapper (see NOTICE).

"""PyTorch-facing API for paged GQA decode attention.

:class:`PagedDecodeWrapper` drives the SM90 (Hopper) and SM100 (Blackwell)
quantized-KV decode kernels behind one interface, dispatching on the compute
capability of the current device.

Every decode path is quantized. K and V live in the cache as packed 2-bit or
4-bit integer values plus a separate scale-and-offset pool, and ``plan()``
requires ``bits=2`` or ``bits=4``. In this mode ``run()``'s ``k_cache`` and
``v_cache`` are the packed pools, and the affine block arrives as
``scale_pool``.

Kernels compile with symbolic dimensions, so one compiled module serves varying
batch sizes, and compilation is memoized in process and cached on disk keyed by
the static configuration.
"""

import functools
from typing import Any, Dict, Optional, Tuple

import torch

from ._utils import (
    api as flashinfer_api,
    get_num_sm,
    is_sm90 as _is_sm90,
    is_supported as _is_supported,
)

_QUANT_BITS = (2, 4)
#: The quantized kernels are built around a 64-token page: both
#: ``IntDecodeSm100`` and ``IntPingPongDecodeSm90`` assert it (one code block and
#: one scale block per (page, kv-head)).
_QUANT_PAGE_SIZE = 64

# The decode kernels are specialized to the head dimensions they tile, and this is NOT a
# "multiple of 64" envelope. The two architectures reach that from different mechanisms,
# so do not state one of them as if it covered both:
#   SM90 builds 128 in every layout, and 256 in one: token-wise K and V over all 256
#   channels, the fold, an f32 scale pool and 2 or 4 stages (`quant_layout.sm90_wide_refusal`).
#   At 256 it cuts both contractions into 128-channel groups and tiles its output
#   accumulators and its fused split-K epilogue by D.
#   SM100 is any positive multiple of 128, because it tiles D at 128 and masks no partial
#   tile. D=256 is therefore not rejected by that kernel, but nothing here measures it.
# Both constructors RAISE rather than assert, so `python -O` cannot drop the guard.
# The accepted set is therefore per architecture, keyed by the names `_utils.get_capabilities`
# reports. A value is listed once a kernel implements AND a run covers it, or plan() admits a
# configuration that writes out of range (below 128) or that nothing has measured:
#   "sm90" lists 256 because the SM90 kernel implements it (#119), its ring depth and phase
#   lock were timed there (profile/h100_planb_prd_20261002/), and the SM90 planner has a
#   D=256 entry fitted on its own split ladder (`_sm90/scheduler.py`, PINGPONG_MODEL).
#   "sm100" lists 128 only: its kernel builds 256, but no run measures it there.
# It stays one literal, because `baselines/bench_sweep.py` reads it from this file's syntax
# tree with `ast.literal_eval` rather than importing the package.
_ACCEPTED_HEAD_DIMS = {"sm90": (128, 256), "sm100": (128,)}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _torch_to_cutlass(dtype: torch.dtype):
    """Map a torch dtype to the CuTe DSL dtype of the same width.

    The DSL is imported here rather than at module scope so that importing the
    package does not require a CUDA toolkit to be installed.
    """
    import cutlass

    table = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float8_e5m2: cutlass.Float8E5M2,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    }
    if dtype not in table:
        raise TypeError(
            f"cutedsl-decode does not support {dtype}; supported: {list(table)}"
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
    m_shape = o_partial_shape[1:-1]

    workspace = workspace.view(dtype=torch.float32)

    required_elts = (
        o_partial_shape.numel()
        + m_partial_shape.numel()
        + l_partial_shape.numel()
        + m_shape.numel()
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

    start, end = end, end + m_shape.numel()
    m = workspace[start:end].view(m_shape)

    return m, o_partial, l_partial, m_partial


# ---------------------------------------------------------------------------
# Compile helpers (memoized)
# ---------------------------------------------------------------------------


@functools.cache
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
):
    """Compile + cache ``IntDecodeSm100`` (SM100 quantized-KV decode).

    Keyed on the STATIC configuration only: ``compile_fold`` traces batch,
    physical page count and kv_splits as ``cute.sym_int()``, so one compiled
    module serves every batch size and KV length. A trace costs 10 to 13 s on
    B200 (DSL 4.5.2, measured 2026-09-11), so this in-process memo is the
    difference between a usable plan() and an unusable one, and
    :mod:`cutedsl_decode.kernel_cache` underneath it serves a repeat of the same
    trace from disk across processes.

    ``kernel_kwargs`` arrives as a sorted tuple of pairs because
    :func:`functools.cache` needs hashable arguments.

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
        **dict(kernel_kwargs),
    )
    return compiled, knobs


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
):
    """Compile + cache ``IntPingPongDecodeSm90`` (SM90 quantized-KV decode).

    Validated on H100 SXM5; see ``_sm90/pingpong_runtime.py``'s module docstring
    for what was checked.

    ``sdtype`` is part of the cache key, not a default to be dropped: it sets
    the scale pool's CuTe element type and its TMA transaction byte count, so
    two granularity configs that differ only in pool precision are two
    different compiled kernels.
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
    )


@functools.cache
def _get_compiled_splitk_combine(head_dim: int):
    """Compile + cache the SM90 external split-K combine. Validated on H100."""
    from ._sm90.pingpong_runtime import compile_splitk_combine

    return compile_splitk_combine(head_dim=head_dim)


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


class PagedDecodeWrapper:
    """PyTorch-facing wrapper for the paged CuTe DSL GQA decode kernel (SM90/SM100)."""

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        # `use_cuda_graph` is accepted and ignored: it was already dead before
        # the dense path was removed (nothing read self._use_cuda_graph), and it
        # is a public constructor argument, so dropping it would break callers
        # for no gain.
        self._planned = False
        # None = dense mode (everything below is unused); 2 or 4 = quantized.
        self._bits: Optional[int] = None
        # Resolved only for the SM100 quantized path. SM90 retains its existing
        # contract until its separately active resource work lands.
        self._scale_pool_dtype: Optional[torch.dtype] = None
        self._scale_pool_sdtype: Optional[str] = None

    def _invalidate_plan(self) -> None:
        """Discard every state object that could make a failed replan runnable."""
        self._planned = False
        self._compiled_fmha_std = None
        self._compile_args = None
        self._compiled_fold = None
        self._compiled_combine = None
        self._quant_kernel_kwargs = None
        self._quant_offsets = None
        self._planned_splits = None
        self._planned_reduction = None
        self._sm90_min_blocks = None
        self._sm90_fused_combine = None
        self._sm90_block_table = None
        self._sm90_ws = None
        self._bits = None
        self._k_quant = None
        self._v_quant = None
        self._q_data_type = None
        self._o_data_type = None
        self._scale_pool_dtype = None
        self._scale_pool_sdtype = None
        self._is_sm90 = None
        self._num_qo_heads = None
        self._num_kv_heads = None
        self._head_dim = None
        self._page_size = None
        self._q_len_per_req = None
        self._sm_scale = None
        self._batch_size = None
        self._max_kv_len = None
        self._tma_mask = None
        self._kv_splits = None
        self._reduction = None
        self._has_workspace = False
        self._m = None
        self._o_partial = None
        self._l_partial = None
        self._m_partial = None
        self._seqlens = None
        self._table_offsets = None
        self._page_table = None

    # -- read-only view of what plan() decided -------------------------------
    # Integrators need these to size their own buffers and to log what the
    # planner picked; they were previously reachable only as private
    # attributes, which is how the split plan stayed invisible.

    @property
    def kv_splits(self) -> int:
        """Split-K partitions the current plan will launch with."""
        return self._kv_splits

    @property
    def reduction(self) -> str:
        """Resolved reduction mode ("none" / "kernel" / "atomic")."""
        return self._reduction

    @property
    def bits(self) -> Optional[int]:
        """``None`` in dense mode; 2 or 4 in quantized mode."""
        return self._bits

    @property
    def scale_pool_dtype(self) -> Optional[torch.dtype]:
        """Resolved IntDecodeSm100 metadata dtype, else ``None``.

        K token-64 defaults to ``torch.bfloat16`` and records
        ``torch.float32`` when ``kernel_kwargs={"sdtype": "f32"}`` is passed.
        Dense plans and SM90 plans return ``None``.
        """
        return self._scale_pool_dtype

    @flashinfer_api
    @_fail_closed_plan
    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        q_data_type: torch.dtype = torch.bfloat16,
        o_data_type: Optional[torch.dtype] = None,
        sm_scale: Optional[float] = None,
        kv_splits: Optional[int] = None,
        reduction: str = "auto",
        q_len_per_req: int = 1,
        max_kv_len: Optional[int] = None,
        non_blocking: bool = True,
        bits: Optional[int] = None,
        k_quant: Optional[Tuple[str, int]] = None,
        v_quant: Optional[Tuple[str, int]] = None,
        kernel_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Plan paged GQA decode for the given problem.

        Parameters
        ----------
        indptr : torch.Tensor (int32, [batch_size + 1])
            Prefix-sum offsets into ``indices``.
        indices : torch.Tensor (int32, [num_pages_total])
            Flat per-sequence virtual page indices.
        seq_lens : torch.Tensor (int32, [batch_size])
            Per-sequence KV length in tokens. The kernel reads this
            directly; callers that have ``last_page_len`` instead can
            convert via ``(indptr.diff() - 1) * page_size + last_page_len``.
        num_qo_heads, num_kv_heads, head_dim, page_size : int
            GQA + paging configuration.  ``page_size`` must be 64 (the
            quant-pool contract: one scale block per (page, kv-head)), and
            ``head_dim`` one of the running architecture's ``_ACCEPTED_HEAD_DIMS``:
            128 or 256 on SM90, where 256 is built in one layout (token-wise K and V
            over all 256 channels with a float32 scale pool) and every other layout is
            refused with the kernel constructor's own ``ValueError`` before anything is
            compiled, and 128 on SM100 (the kernels are specialized to these, see that
            constant).
        q_data_type, o_data_type : torch.dtype
            Q/O dtypes.  K/V are the packed code pools and are always
            ``torch.uint8``, so there is no ``kv_data_type``.
        sm_scale : Optional[float]
            Softmax scale; defaults to ``1 / sqrt(head_dim)``.
        kv_splits : Optional[int]
            Threadblocks per sequence (flash decoding).  ``None`` auto-tunes
            from the planned shapes and SM count.
        reduction : str
            ``"kernel"`` (deterministic with workspace), ``"none"`` (no
            flash-decoding split-K; requires kv_splits == 1), or ``"auto"``
            (the arch cost model picks).  ``"atomic"`` was dense-only and both
            quantized planners reject it.
        q_len_per_req : int
            Must be 1.  Speculative decode was dense-only.
        max_kv_len : Optional[int]
            Maximum KV sequence length across the batch.  Used to auto-tune
            ``kv_splits``; pass it explicitly to avoid a GPU→CPU sync.
        non_blocking : bool
            Async device copies for the plan-time integer buffers.
        bits : int
            REQUIRED -- ``2`` or ``4``.  Selects the QUANTIZED-KV kernel (``IntDecodeSm100`` on SM100,
            ``IntPingPongDecodeSm90`` on SM90).  In quantized mode ``run``'s
            ``k_cache``/``v_cache`` are the PACKED CODE POOLS and the affine
            block is passed as ``run(scale_pool=...)``.  See
            :meth:`_plan_quantized` for the extra constraints (page_size 64,
            fp8 query on SM100, ``kv_splits`` chosen by the arch cost model).
        k_quant, v_quant : Optional[Tuple[str, int]]
            Quantized mode only: per-tensor quantization granularity as
            ``(axis, group)`` — ``("token", head_dim)`` (the default and the
            shipped fast path), ``("token", head_dim // 2)``, or
            ``("channel", G)`` with ``G`` in ``{64, 128, 256}``.  The pool
            layout must match; see ``quant_layout.quant_block_offsets``.
        kernel_kwargs : Optional[dict]
            Quantized mode only, ADVANCED: extra constructor knobs forwarded
            verbatim to the quantized kernel (e.g. ``sdtype``,
            ``dequant_int2_to_f4``, ``o_stages``).  Anything set here
            OVERRIDES the promoted per-bits tuning, so use it only when you
            know which knob you are moving and why. On SM100, K token-64 keeps
            its bf16 scale-pool default and also accepts explicit
            ``sdtype="f32"``; :attr:`scale_pool_dtype` records the resolved
            torch dtype.
        """
        if bits is not None:
            return self._plan_quantized(
                indptr, indices, seq_lens, num_qo_heads, num_kv_heads,
                head_dim, page_size, bits,
                q_data_type=q_data_type, o_data_type=o_data_type,
                sm_scale=sm_scale, kv_splits=kv_splits, reduction=reduction,
                q_len_per_req=q_len_per_req, max_kv_len=max_kv_len,
                non_blocking=non_blocking, k_quant=k_quant, v_quant=v_quant,
                kernel_kwargs=kernel_kwargs,
            )
        if k_quant is not None or v_quant is not None or kernel_kwargs:
            raise ValueError(
                "k_quant / v_quant / kernel_kwargs are quantized-mode "
                "arguments; pass bits=2 or bits=4 to select that mode"
            )
        raise ValueError(
            "bits=2 or bits=4 is required: the dense (unquantized) decode path "
            "was removed. It used to be plan()'s default via bits=None."
        )

    def _plan_quantized(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        bits: int,
        *,
        q_data_type: Optional[torch.dtype],
        o_data_type: Optional[torch.dtype],
        sm_scale: Optional[float],
        kv_splits: Optional[int],
        reduction: str,
        q_len_per_req: int,
        max_kv_len: Optional[int],
        non_blocking: bool,
        k_quant: Optional[Tuple[str, int]],
        v_quant: Optional[Tuple[str, int]],
        kernel_kwargs: Optional[Dict[str, Any]],
    ) -> None:
        """Plan the quantized-KV kernel. Called by :meth:`plan` when ``bits``
        is set.

        Both cost models (``_sm100/scheduler.plan_splits_sm100`` and
        ``_sm90/scheduler.plan_splits``) are keyed by ``bits`` and calibrated
        against the quantized kernels, whose per-tile cost differs by about 1.6
        times between the two bit widths. They are not shared with each other
        and must not be unified: the two architectures plan by different cost
        forms.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "paged decode needs a CUDA GPU, and torch reports no CUDA "
                "device. Install a CUDA build of torch and run on an SM90 "
                "(Hopper) or SM100 (Blackwell) device."
            )
        if not _is_supported():
            major, minor = torch.cuda.get_device_capability()
            raise ValueError(
                f"this package runs on SM90 (Hopper) and SM100 (Blackwell) "
                f"only; the current device reports compute capability "
                f"{major}.{minor}. Use get_capabilities() to check a device "
                f"before planning."
            )
        if bits not in _QUANT_BITS:
            raise ValueError(f"bits must be one of {_QUANT_BITS} or None; got {bits}")
        if page_size != _QUANT_PAGE_SIZE:
            raise ValueError(
                f"quantized decode requires page_size == {_QUANT_PAGE_SIZE} "
                f"(one code block and one scale block per (page, kv-head)); "
                f"got {page_size}"
            )
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_qo_heads ({num_qo_heads}) must be a multiple of "
                f"num_kv_heads ({num_kv_heads})"
            )
        # First the union over architectures, before any device query, so a head dimension
        # no architecture accepts is refused on a box without a GPU. The running
        # architecture's own set is checked once the device is known, below.
        accepted_anywhere = sorted({d for dims in _ACCEPTED_HEAD_DIMS.values() for d in dims})
        if head_dim not in accepted_anywhere:
            raise ValueError(
                f"head_dim ({head_dim}) is not accepted; the decode kernels accept "
                f"{accepted_anywhere} on some architecture "
                f"({', '.join(f'{a}: {list(d)}' for a, d in _ACCEPTED_HEAD_DIMS.items())}). "
                f"A multiple of 64 is NOT sufficient: below 128 the kernels write past the "
                f"output row, and above it a value is accepted only where a kernel builds it "
                f"and a run measures it."
            )
        if q_len_per_req != 1:
            raise NotImplementedError(
                "quantized decode is plain decode only (the fold kernels pack "
                f"one query token per threadblock); got "
                f"q_len_per_req={q_len_per_req}"
            )
        is_sm90 = _is_sm90()
        self._is_sm90 = is_sm90
        arch_key, arch_name = ("sm90", "SM90") if is_sm90 else ("sm100", "SM100")
        if head_dim not in _ACCEPTED_HEAD_DIMS[arch_key]:
            raise ValueError(
                f"head_dim {head_dim} is built by the {arch_name} kernel but not measured, "
                f"so plan() accepts "
                f"{' and '.join(str(d) for d in _ACCEPTED_HEAD_DIMS[arch_key])} only on "
                f"{arch_name}"
            )

        batch_size = seq_lens.numel()
        if indptr.numel() < batch_size:
            raise ValueError(
                f"indptr length ({indptr.numel()}) must be at least batch_size "
                f"({batch_size})"
            )

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

        if q_data_type is None:
            q_data_type = torch.float8_e4m3fn if not is_sm90 else torch.bfloat16
        if o_data_type is None:
            o_data_type = torch.bfloat16
        if sm_scale is None:
            sm_scale = head_dim**-0.5

        k_quant = ("token", head_dim) if k_quant is None else (k_quant[0], int(k_quant[1]))
        v_quant = ("token", head_dim) if v_quant is None else (v_quant[0], int(v_quant[1]))

        self._bits = bits
        self._k_quant = k_quant
        self._v_quant = v_quant
        self._q_data_type = q_data_type
        self._o_data_type = o_data_type
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._page_size = page_size
        self._q_len_per_req = 1
        self._sm_scale = sm_scale
        self._batch_size = batch_size
        self._seqlens = seq_lens
        self._table_offsets = indptr[:batch_size]
        self._page_table = indices
        self._max_kv_len = max_kv_len
        self._scale_pool_dtype = None
        self._scale_pool_sdtype = None

        if is_sm90:
            self._plan_quantized_sm90(indptr, indices, kv_splits, reduction,
                                      kernel_kwargs)
        else:
            self._plan_quantized_sm100(kv_splits, reduction, kernel_kwargs)
        self._planned = True

    def _plan_quantized_sm100(self, kv_splits, reduction, kernel_kwargs) -> None:
        from ._sm100.fold_runtime import (
            blk_tile_n_of, quant_cfg_of, refuse_retired_kernel_kwargs)
        from .quant_layout import quant_block_offsets, validate_quant_flex
        from ._sm100.scheduler import plan_splits_sm100

        # A retired knob must raise rather than be accepted and ignored, and this
        # is the public door: refuse it before the split is planned or anything is
        # compiled, naming plan() rather than the helper that would otherwise catch
        # it further in.
        refuse_retired_kernel_kwargs(kernel_kwargs,
                                     where="PagedDecodeWrapper.plan")
        bits = self._bits
        tile_n = blk_tile_n_of(self._num_qo_heads, self._num_kv_heads)
        if tile_n > 8:
            raise NotImplementedError(
                f"IntDecodeSm100 packs at most 8 Q head rows per threadblock (the "
                f"P hi/lo row budget, 2 * blk_tile_n <= mma_tile_n), and this "
                f"wrapper picks the head tile as the GQA ratio rounded up to a "
                f"power of two rather than spreading a larger ratio over several "
                f"threadblocks; num_qo_heads / num_kv_heads = "
                f"{self._num_qo_heads // self._num_kv_heads} needs a "
                f"{tile_n}-wide N tile")
        if self._q_data_type != torch.float8_e4m3fn:
            raise ValueError(
                "IntDecodeSm100 is fp8-mode-only: q_data_type must be "
                f"torch.float8_e4m3fn, got {self._q_data_type}. The per-tensor "
                "Q descale is folded into sm_scale by the caller "
                "(scale_s = sm_scale / q_scale) — the kernel takes no q_scale."
            )
        k_quant, v_quant = validate_quant_flex(
            self._k_quant, self._v_quant, self._page_size, self._head_dim)
        self._k_quant, self._v_quant = k_quant, v_quant
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
        # count. Safe here: k_quant/v_quant were validated just above.
        from ._sm100.fold_runtime import resolve_fold_tiling

        _seq_res, _ = resolve_fold_tiling(k_quant, v_quant)
        self._planned_splits, self._planned_reduction = plan_splits_sm100(
            self._batch_size, self._num_kv_heads, self._max_kv_len, bits=bits,
            sm_count=get_num_sm(self.device), sequence_tile=_seq_res)
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
            m, o_partial, l_partial, m_partial = _slice_workspace(
                self._float_workspace_buffer, kv_splits, self._batch_size,
                1, self._num_qo_heads, self._head_dim)
            self._m = m
            self._o_partial = o_partial
            self._l_partial = l_partial
            self._m_partial = m_partial

        kkw = dict(kernel_kwargs or {})
        clash = {"k_quant", "v_quant"} & set(kkw)
        if clash:
            raise ValueError(
                f"pass granularity through the named plan() arguments, not "
                f"kernel_kwargs: {sorted(clash)}. Only the named ones are "
                f"validated (validate_quant_flex) and recorded for the "
                f"pscale_lp derivation in run().")
        # Legacy ("token", head_dim) for BOTH tensors is the shipped default and
        # is left OUT of the ctor kwargs on purpose: it keeps the compile key —
        # and therefore the traced kernel — byte-for-byte what the benchmark and
        # gate harness produce, which is what the equivalence gate compares.
        if k_quant != ("token", self._head_dim) or v_quant != ("token", self._head_dim):
            kkw["k_quant"] = k_quant
            kkw["v_quant"] = v_quant
        cfg_k, cfg_v, scale_sdtype, _ = quant_cfg_of(
            self._page_size, self._head_dim, kkw)
        assert (cfg_k, cfg_v) == (k_quant, v_quant), (
            f"quant_cfg_of drifted from the validated plan: "
            f"{cfg_k}/{cfg_v} != {k_quant}/{v_quant}")
        self._scale_pool_sdtype = scale_sdtype
        self._scale_pool_dtype = (
            torch.bfloat16 if scale_sdtype == "bf16" else torch.float32)
        self._quant_kernel_kwargs = kkw
        self._compiled_fold, applied = _get_compiled_fold_kernel(
            self._page_size,
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim,
            reduction,
            bits,
            _torch_to_cutlass(self._o_data_type),
            tuple(sorted(kkw.items())),
        )
        self._assert_bits_tuning_applied(applied)

    def _assert_bits_tuning_applied(self, applied: Dict[str, Any]) -> None:
        """Prove the promoted per-bits tuning reached the object we compiled.

        ``_sm100/fold_runtime.compile_fold`` guards this because the tuning was
        once applied as a dead statement AFTER the constructor and the int4
        timings came back bit-for-bit identical (1674.2 us both ways) — a
        promoted optimisation the kernel object never received. The wrapper
        shares the same ``BITS_TUNING`` dict (imported, never copied, so the
        two cannot drift) and therefore shares the guard.

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
                continue                       # caller overrode deliberately
            assert applied[key] == want, (
                f"per-bits tuning did not reach the kernel: bits={self._bits} "
                f"{key} is {applied[key]!r}, expected {want!r}; see "
                f"_sm100/fold_runtime.BITS_TUNING")

    def _plan_quantized_sm90(self, indptr, indices, kv_splits, reduction,
                             kernel_kwargs) -> None:
        """Plan ``IntPingPongDecodeSm90``. Validated on H100.

        See ``_sm90/pingpong_runtime.py``'s module docstring for what was checked.
        This wires the SAME API surface (`plan` -> `run`) as the SM100 path.

        WHY ``kv_splits`` IS HONOURED HERE AND WAS NOT BEFORE
        ----------------------------------------------------
        This function used to call ``plan_pingpong`` unconditionally and drop
        the caller's ``kv_splits`` and ``reduction`` on the floor, so
        ``plan(kv_splits=4)`` silently returned the planner's pick. Two things
        that breaks, both of which the SM100 path supports:

          * a correctness gate cannot reach the split counts the planner did
            not happen to choose -- and "only wrong at outer > 1" is exactly
            this path's exposure;
          * the planner cannot be SCORED, because scoring it means timing the
            same problem at every candidate split and comparing.

        REDUCTION IS DERIVED, NOT CHOSEN, and that is an arch difference worth
        stating: on SM100 the reduction mode selects a compile-time epilogue,
        while ``IntPingPongDecodeSm90`` always writes ``o_partial``/``lse`` and the
        merge is either its own fused tail or the external ``SplitKCombine``.
        So ``outer == 1`` IS ``"none"`` and ``outer > 1`` IS ``"kernel"``; a
        caller asking for the other pairing is asking for something that does
        not exist, and gets an error rather than a silent substitution.

        The third axis -- fused tail vs external combine -- is a real choice
        (``plan_fused_combine`` is a POLICY, not a constraint), so it is
        exposed as ``kernel_kwargs={"fused_combine": bool}`` for A/B work.

        The fourth is ``kernel_kwargs={"sdtype": torch.float32 | torch.bfloat16}``,
        the scale-pool precision, which the CALLER owns because the caller is
        the one who built the pool -- the same surface SM100 exposes. It is
        validated, not derived: ``("token", head_dim // 2)`` requires the
        bfloat16 pool (token-64 doubles the scale ring's smem section and
        halving the element is what funds it), and asking for anything else
        with it is an error here rather than an assert about a smem budget from
        inside the kernel constructor. Default ``torch.float32``.

        Nothing else is accepted: the remaining SM100 knobs do not exist on
        this kernel.
        """
        from .kernel_contract import normalize_sdtype
        from .quant_layout import SM90_WIDE_STAGES, sm90_wide_refusal, validate_quant_flex
        from ._sm90.pingpong_runtime import (
            alloc_pingpong_workspace, block_table_2d, plan_pingpong,
        )
        from ._sm90.scheduler import plan_fused_combine
        from ._sm90.serving_cache import quant_block_offsets

        kkw = dict(kernel_kwargs or {})
        fused_override = kkw.pop("fused_combine", None)
        sdtype = kkw.pop("sdtype", torch.float32)
        if kkw:
            raise NotImplementedError(
                f"the SM90 quantized path takes its knobs from plan_splits / "
                f"plan_fused_combine; the accepted kernel_kwargs keys are "
                f"'fused_combine' and 'sdtype' (got {sorted(kkw)})")
        if sdtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                f"kernel_kwargs['sdtype'] is the scale-pool precision and must "
                f"be torch.float32 (default) or torch.bfloat16; got {sdtype}")
        if self._q_data_type != torch.bfloat16:
            raise ValueError(
                "IntPingPongDecodeSm90 takes a bf16 query (mQ); got "
                f"{self._q_data_type}. Note the arch difference: the SM100 "
                "fold kernel takes fp8 e4m3."
            )
        # AT A HEAD DIMENSION OTHER THAN 128 THE KERNEL BUILDS ONE LAYOUT, and every other pair or
        # pool precision is refused here with the constructor's own reason
        # (`quant_layout.sm90_wide_refusal`), before anything is planned or compiled. It has to
        # come before `validate_quant_flex` and the token-D/2 pool check below, because both
        # describe the envelope at 128: the first raises an AssertionError for V token-128 at
        # head_dim 256, and the second advises a bfloat16 pool for K token-128 at head_dim 256,
        # which no pool precision builds. `compile_pingpong` always folds and leaves the ring
        # depth to the constructor, which resolves it inside `SM90_WIDE_STAGES` at these head
        # dimensions, so neither can be refused here, and an admitted depth stands in for it.
        if self._head_dim != 128:
            refusal = sm90_wide_refusal(
                self._head_dim, self._k_quant, self._v_quant, sdtype=normalize_sdtype(sdtype),
                fold_scale_zero=True, num_stages=SM90_WIDE_STAGES[0])
            if refusal is not None:
                raise ValueError(refusal)
        # VALIDATE THE GRANULARITY, which this path used to accept blind. It
        # forwarded `k_quant`/`v_quant` straight into the compile without ever
        # calling `validate_quant_flex`, and `run()` skipped the `scale_pool`
        # shape check on SM90 as well -- so a pool built at a different
        # granularity surfaced either as a DSL trace assert naming an argument
        # index, or as silently wrong output. `validate_quant_flex` is
        # arch-neutral despite living under `_sm100/`: its own docstring records
        # that it encodes the SM90 IntPingPongDecodeSm90 envelope verbatim.
        k_quant, v_quant = validate_quant_flex(
            self._k_quant, self._v_quant, self._page_size, self._head_dim)
        self._k_quant, self._v_quant = k_quant, v_quant
        # SM90's OWN offsets table (`_sm90/serving_cache`), not the SM100 mirror
        # of it -- same keys, and this is the one the SM90 pool builders use.
        self._quant_offsets = quant_block_offsets(
            k_quant, v_quant, N=self._page_size, D=self._head_dim)
        # CHECK the caller's scale-pool precision against the granularity; do
        # not pick it for them. The pool is the caller's tensor -- deriving a
        # dtype here would mean plan() silently compiling for a precision the
        # pool they already built does not have, and the mismatch would surface
        # as wrong numbers rather than as an error.
        #
        # The constraint itself is the kernel's: token-64 doubles the scale
        # ring's smem section and halving the element is what funds it, so
        # `IntPingPongDecodeSm90.__init__` asserts `sdtype == bfloat16`. Raising
        # here turns that into a message naming the two arguments that
        # disagree. The sdtype MUST be forwarded: without it `compile_pingpong`'s
        # f32 default applies unconditionally and every `k_quant=("token", 64)`
        # plan dies on the constructor assert, even though
        # `validate_quant_flex` accepts ("token", D//2) and the capability
        # surface advertises it.
        if ("token", self._head_dim // 2) in (k_quant, v_quant):
            if sdtype != torch.bfloat16:
                raise ValueError(
                    f"k_quant={k_quant} v_quant={v_quant}: token-"
                    f"{self._head_dim // 2} requires a bfloat16 scale pool (it "
                    f"doubles the scale-ring smem section and halving the "
                    f"element funds it), but kernel_kwargs['sdtype'] is "
                    f"{sdtype}. Build the pool with "
                    f"make_serving_cache_flex(..., sdtype=torch.bfloat16) and "
                    f"pass kernel_kwargs={{'sdtype': torch.bfloat16}}.")
        self._sm90_scale_dtype = sdtype
        planned_outer, planned_minb, planned_fused = plan_pingpong(
            self._batch_size, self._num_kv_heads, self._max_kv_len,
            bits=self._bits, head_dim=self._head_dim, sm_count=get_num_sm(self.device))
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
            fused = plan_fused_combine(self._batch_size, self._num_kv_heads, outer)
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
        self._compiled_fold = _get_compiled_pingpong_kernel(
            self._page_size, self._num_qo_heads, self._num_kv_heads,
            self._head_dim, self._bits, min_blocks, fused,
            tuple(self._k_quant), tuple(self._v_quant),
            self._sm90_scale_dtype)
        # The external combine runs ONLY when the in-kernel fused tail is off.
        self._compiled_combine = (
            _get_compiled_splitk_combine(self._head_dim)
            if outer > 1 and not fused else None)
        # What the caller actually asked for, kept so a benchmark label cannot
        # claim a config the plan did not use (the SM100 side needs this for
        # its BITS_TUNING override check; here it is pure auditability).
        self._quant_kernel_kwargs = dict(kernel_kwargs or {})

    def pscale_lp(self, scale_pool: torch.Tensor) -> float:
        """Precompute `run(pscale_lp=...)` for this pool. **Does a device sync.**

        `run()` derives this itself when you omit it, which costs a
        `.max().item()` on every call -- a full device sync sitting in the
        middle of the decode path. The value is a property of the scale pool
        alone, so a serving loop should call this when the KV cache is written
        and pass the cached float to every `run()` after that.

        Exposed as a method because the alternative was pointing users at
        `_sm100.fold_runtime.pscale_lp_for_pool` and asking them to thread
        `(k_quant, v_quant, page_size, head_dim)` through by hand -- four
        values this object already holds, and a mismatch on any of them is
        silent.
        """
        if not self._planned:
            raise RuntimeError("Call plan() before pscale_lp().")
        if self._bits is None:
            raise RuntimeError("pscale_lp() is quantized-mode only; this "
                               "wrapper was planned without bits=")
        from ._sm100.fold_runtime import pscale_lp_for_pool
        self._require_scale_pool_dtype(scale_pool)
        return pscale_lp_for_pool(
            scale_pool, self._k_quant, self._v_quant,
            page_size=self._page_size, head_dim=self._head_dim)

    def _require_scale_pool_dtype(self, scale_pool: torch.Tensor) -> None:
        """Enforce the resolved SM100 metadata dtype before any sync or launch."""
        if self._scale_pool_dtype is None:
            return
        from ._sm100.fold_runtime import require_scale_pool_dtype
        require_scale_pool_dtype(
            scale_pool,
            self._scale_pool_dtype,
            expected_sdtype=self._scale_pool_sdtype,
        )

    def _require_declared_k_pool_order(self, k_pool_order) -> None:
        """The caller must SAY which K order its pool carries, when it matters.

        NOTHING CAN DETECT THIS FROM THE BYTES. The kernel reads a K pool
        whose codes were permuted at store time by
        `cutedsl_decode.pool.k_store_perm_sm100_f4`: at int4 at EVERY configuration
        (the raw-feed 8-slot shuffle) and at int2 with the E2M1 feed at either
        K axis. A pool written in
        `pack_rows_lut` order has the same dtype, the same shape and the same byte
        count, and decodes to finite, plausible values with every K channel taken
        from the wrong slot inside its packed-word group. No validation can see
        that, so the only alternative to a silent wrong answer is to make the
        caller state what it wrote and to check the statement against the
        configuration.

        This is the public surface, and it is where the risk actually lives: the
        installed package ships no pool builder, no serving append path emits this
        order, and `docs/K_STORE_PERM.md` is the specification an external writer
        has to meet. Routing every in-repo caller through
        `_fold_harness.k_pool_perm` makes the wrong pairing unreachable for them
        and does nothing for a library caller, which is why this guard exists too.

        `k_pool_order` is "permuted" or "logical", and it is required only for the
        configurations that read a permuted pool: INT4 everywhere and INT2 with
        the E2M1 feed at either K axis. Dense callers may omit it. An undeclared
        pool is refused rather than read wrong.
        """
        if self._is_sm90 or self._bits is None:
            return
        # `getattr`, because a partially built wrapper must produce a USEFUL
        # error rather than an AttributeError from inside a validator. `__init__`
        # sets this attribute, so a real wrapper always carries it; the host
        # tests reach the pre-launch guards through `__new__`.
        kkw = dict(getattr(self, "_quant_kernel_kwargs", None) or {})
        # This reads the planned dict BY NAME, so it consults the retired-kwargs
        # table as every such door does: a retired key here would otherwise be
        # discarded and the order answered as though it were absent.
        from ._sm100.fold_runtime import refuse_retired_kernel_kwargs
        refuse_retired_kernel_kwargs(kkw, where="PagedDecodeWrapper.run")
        # The SAME predicate the pool writer asks (`_fold_harness.k_pool_perm`),
        # so the pool writer and the launch agree.
        from ._sm100.int_fold_decode import IntDecodeSm100
        want_permuted = IntDecodeSm100.k_pool_is_permuted(
            self._bits,
            dequant_int2_to_f4=kkw.get("dequant_int2_to_f4"),
            k_quant=tuple(self._k_quant or ("token", self._head_dim)),
            fold_scale_zero=bool(int(kkw.get("fold_scale_zero", 1) or 0)))
        if k_pool_order not in (None, "permuted", "logical"):
            raise ValueError(
                f"k_pool_order must be 'permuted', 'logical' or None; got "
                f"{k_pool_order!r}")
        if not want_permuted:
            if k_pool_order == "permuted":
                raise ValueError(
                    "this plan reads K in LOGICAL (pack_rows_lut) order, so a "
                    "permuted pool would be read wrong. Ask "
                    "IntDecodeSm100.k_pool_is_permuted; "
                    "see docs/K_STORE_PERM.md")
            return
        if k_pool_order is None:
            raise ValueError(
                "this plan reads a PERMUTED K pool, and the order cannot be "
                "detected from the pool itself, so run() requires you to declare "
                "it: pass k_pool_order='permuted' if the K codes were permuted "
                "by cutedsl_decode.pool.k_store_perm_sm100_f4 before packing, which is "
                "what this configuration reads (bits=4 at every configuration, "
                "or bits=2 with the E2M1 feed at either K axis). A pool in "
                "pack_rows_lut order is silently wrong "
                "here at every K channel. See docs/K_STORE_PERM.md")
        if k_pool_order != "permuted":
            raise ValueError(
                "k_pool_order='logical' contradicts this plan, which reads a "
                "PERMUTED K pool (bits=4 everywhere; bits=2 with the E2M1 feed "
                "at either K axis). Permute "
                "the K codes with cutedsl_decode.pool.k_store_perm_sm100_f4 before "
                "packing, or plan a configuration that reads logical order; see "
                "docs/K_STORE_PERM.md")

    def _run_quantized(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        scale_pool: Optional[torch.Tensor],
        out: Optional[torch.Tensor],
        sm_scale: Optional[float],
        o_scale: Optional[float],
        pscale_lp: Optional[float],
        enable_pdl: bool,
        k_pool_order: Optional[str] = None,
    ) -> torch.Tensor:
        if scale_pool is None:
            raise ValueError(
                "quantized decode needs the affine (scale, zero) block: pass "
                "run(..., scale_pool=mScale)")
        self._require_scale_pool_dtype(scale_pool)
        # AFTER the scale-pool guards, deliberately. A caller with both a wrong
        # metadata dtype and no declared K order should hear about the dtype
        # first: that is the older, narrower error, and
        # `tests/test_sm100_scale_dtype_host.py` pins it as the first thing a
        # mismatched pool hits. Both run before any launch, which is what matters.
        self._require_declared_k_pool_order(k_pool_order)
        if q.dtype != self._q_data_type:
            raise ValueError(
                f"q.dtype={q.dtype} mismatches planned {self._q_data_type}")
        if k_cache.dtype != torch.uint8 or v_cache.dtype != torch.uint8:
            raise ValueError(
                "in quantized mode k_cache/v_cache are the PACKED CODE POOLS "
                f"(uint8 mKq/mVq); got {k_cache.dtype}/{v_cache.dtype}")
        # Catch a pool built for a different granularity HERE. Downstream it
        # surfaces as a DSL trace assert on argument #19, which does not say
        # "your k_quant and your pool disagree".
        #
        # RUNS ON BOTH ARCHES, and must. Guarding it on `not self._is_sm90` left
        # Hopper with either that opaque trace assert or silently wrong output.
        # Both arches build the same sectioned block, so one check covers both.
        legacy = self._quant_offsets["legacy"]
        if legacy:
            want_rank, want_tail = 4, 4
            want = "[num_pages, num_kv_heads, page_size, 4]"
        else:
            want_rank = 3
            want_tail = self._quant_offsets["block_tot"]
            want = f"[num_pages, num_kv_heads, {want_tail}]"
        if scale_pool.dim() != want_rank or scale_pool.shape[-1] != want_tail:
            raise ValueError(
                f"scale_pool shape {tuple(scale_pool.shape)} does not match "
                f"k_quant={self._k_quant} v_quant={self._v_quant}: "
                f"expected {want}")
        # And the POOL PRECISION, which the shape check above cannot see. The
        # kernel's scale TMA is traced against a fixed element type and
        # transaction size, so handing it a pool of the other dtype reads the
        # right bytes as the wrong numbers -- finite, plausibly scaled, wrong.
        # SM90 only, because that is where `plan()` records what it compiled
        # for. On SM100 the same value goes through `kernel_kwargs["sdtype"]`
        # straight into the trace and is not tracked separately here.
        if self._is_sm90 and scale_pool.dtype != self._sm90_scale_dtype:
            raise ValueError(
                f"scale_pool dtype {scale_pool.dtype} does not match the "
                f"precision this plan compiled for "
                f"({self._sm90_scale_dtype}); pass the same dtype to "
                f"make_serving_cache_flex(..., sdtype=...) and to "
                f"plan(kernel_kwargs={{'sdtype': ...}}).")

        # q accepted as [B, HQ, D] or [B, 1, HQ, D]; the fold kernel wants BSHD.
        if q.dim() == 3:
            q_view = q.view(self._batch_size, 1, self._num_qo_heads, self._head_dim)
            out_was_3d = True
        elif q.dim() == 4:
            if q.shape[1] != 1:
                raise ValueError(
                    f"quantized decode is plain decode (q_len_per_req == 1); "
                    f"got q.shape={tuple(q.shape)}")
            q_view = q
            out_was_3d = False
        else:
            raise ValueError(f"q must be 3D or 4D; got ndim={q.dim()}")
        if q_view.shape[0] != self._batch_size:
            raise ValueError(
                f"q batch {q_view.shape[0]} mismatches planned {self._batch_size}")

        scale_s = self._sm_scale if sm_scale is None else sm_scale
        scale_o = 1.0 if o_scale is None else o_scale

        if self._is_sm90:
            return self._run_quantized_sm90(q_view, k_cache, v_cache, scale_pool,
                                            out, scale_s, scale_o, out_was_3d)

        from ._sm100.fold_runtime import launch_fold, pscale_lp_for_pool

        device = q.device
        if out is None:
            out_4d = torch.empty(
                (self._batch_size, 1, self._num_qo_heads, self._head_dim),
                dtype=self._o_data_type, device=device)
            user_out = None
        else:
            user_out = out
            out_4d = out.view(self._batch_size, 1, self._num_qo_heads,
                              self._head_dim) if out.dim() == 3 else out

        m = o_partial = l_partial = m_partial = None
        if self._has_workspace:
            m, o_partial = self._m, self._o_partial
            l_partial, m_partial = self._l_partial, self._m_partial
            # The kernel atomic-maxes into `m`, so it must hold the identity
            # -inf at the start of EVERY launch. This fill is the kernel's own
            # precondition, not benchmark overhead.
            m.fill_(float("-inf"))

        if pscale_lp is None:
            # A device sync (`.max().item()` over the pool). Hoist it: the
            # value only changes when the scale pool's max s_v changes, so a
            # serving loop should compute it on cache writes and pass it here.
            pscale_lp = pscale_lp_for_pool(
                scale_pool, self._k_quant, self._v_quant,
                page_size=self._page_size, head_dim=self._head_dim)

        launch_fold(
            self._compiled_fold,
            kv_splits=self._kv_splits,
            seqlens=self._seqlens,
            table_offsets=self._table_offsets,
            page_table=self._page_table,
            q_bshd=q_view,
            o_bshd=out_4d,
            m=m, o_partial=o_partial, l_partial=l_partial, m_partial=m_partial,
            scale_s=scale_s, scale_o=scale_o,
            mKq=k_cache, mVq=v_cache, mScale=scale_pool,
            pscale_lp=pscale_lp,
            enable_pdl=enable_pdl,
        )

        if user_out is not None:
            return user_out
        if out_was_3d:
            return out_4d.view(-1, self._num_qo_heads, self._head_dim)
        return out_4d

    def _run_quantized_sm90(self, q_view, k_cache, v_cache, scale_pool, out,
                            scale_s, scale_o, out_was_3d) -> torch.Tensor:
        """Validated on H100 SXM5.

        The SM90 kernel's tensors are 3-D ``[B, HQ, D]`` (no prediction axis)
        and its query is bf16, so this is not a shape-for-shape copy of the
        SM100 path. It launches through ``pingpong_runtime.launch_pingpong``.
        """
        from ._sm90.pingpong_runtime import launch_pingpong

        q_bhd = q_view[:, 0]
        if out is None:
            out_bhd = torch.zeros((self._batch_size, self._num_qo_heads,
                                   self._head_dim), dtype=self._o_data_type,
                                  device=q_bhd.device)
        else:
            out_bhd = out.view(self._batch_size, self._num_qo_heads, self._head_dim)
        ws = self._sm90_ws
        launch_pingpong(
            self._compiled_fold, self._compiled_combine,
            q_bhd=q_bhd, mKq=k_cache, mVq=v_cache, mScale=scale_pool,
            block_table=self._sm90_block_table, seq_lens=self._seqlens,
            o_partial=ws["o_partial"], out_bhd=out_bhd, lse=ws["lse"],
            counters=ws["counters"], scale_s=scale_s, scale_o=scale_o,
            outer_splits=self._kv_splits, num_kv_heads=self._num_kv_heads,
            batch=self._batch_size,
        )
        if out is not None:
            return out
        if out_was_3d:
            return out_bhd
        return out_bhd.view(self._batch_size, 1, self._num_qo_heads, self._head_dim)

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        sm_scale: Optional[float] = None,
        o_scale: Optional[float] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        lse: Optional[torch.Tensor] = None,
        enable_pdl: bool = True,
        scale_pool: Optional[torch.Tensor] = None,
        pscale_lp: Optional[float] = None,
        k_pool_order: Optional[str] = None,
    ) -> torch.Tensor:
        """Run paged GQA decode.

        Parameters
        ----------
        q : torch.Tensor
            ``[batch_size * q_len_per_req, num_qo_heads, head_dim]`` or
            ``[batch_size, q_len_per_req, num_qo_heads, head_dim]``.
            ``q_len_per_req`` is read from ``q.shape`` at run time; it does
            not have to match the value passed to :meth:`plan` (which is
            only a compile-time tile-size hint).
        k_cache, v_cache : torch.Tensor
            Logical shape ``[num_pages, page_size, num_kv_heads, head_dim]``.
            Both NHD-contiguous layouts and HND layouts (presented as a
            transposed view) are accepted; the kernel handles arbitrary
            strides as long as ``head_dim`` is innermost.
        out : Optional[torch.Tensor]
            Pre-allocated output buffer.  For atomic reduction it must be
            zero-initialized before being passed in.
        sm_scale : Optional[float]
            Per-call override of the softmax scale set at plan() time.
        o_scale : Optional[float]
            Output scale applied to the final O before it is written. The
            cute-dsl kernel folds this in for free in the reduction
            epilogue (no separate post-kernel multiply). Defaults to 1.0.
        skip_softmax_threshold_scale_factor : Optional[float]
            BLASST skip-softmax scale factor. The kernel divides this by
            each batch's KV seqlen to obtain the per-request effective
            threshold. Must be > 0 when set. ``None`` (default) dispatches
            to the standard kernel; a value triggers lazy compile of the
            BLASST variant on first use, or hits the precompiled cache if
            ``plan(precompile_skip_softmax_kernel=True)`` was used.
        lse : Optional[torch.Tensor]
            Pre-allocated float32 buffer of shape
            ``(batch_size, q_len_per_req, num_qo_heads)`` (or the flat
            equivalent ``(batch_size * q_len_per_req, num_qo_heads)``) to
            receive the log-sum-exp (log2 base, matching flashinfer
            convention). When ``None`` (default) the kernel skips the LSE
            write; otherwise an LSE variant is lazily compiled on first
            use.
        scale_pool : Optional[torch.Tensor]
            QUANTIZED MODE ONLY (``plan(bits=...)``), and REQUIRED there: the
            per-(page, kv-head) affine block.  With the default
            ``("token", head_dim)`` granularity for both tensors this is the
            legacy interleaved f32 pool ``[num_pages, num_kv_heads,
            page_size, 4] = (s_k, z_k, s_v, z_v)``; with any flex granularity
            it is the flat sectioned block ``[num_pages, num_kv_heads,
            block_tot]`` (see ``quant_layout.quant_block_offsets``) in
            the dtype recorded by :attr:`scale_pool_dtype`.
            In this mode ``k_cache``/``v_cache`` carry the packed uint8 code
            pools (mKq/mVq) rather than dense K/V.
            V has no pool-order declaration or runtime layout check. For the
            main SM100 kernel, rebuild INT4 V caches created before the
            K-major V alignment: their shape is unchanged but their byte
            order is incompatible. Resolve the writer's order with
            ``_fold_harness.v_pool_perm(bits, kernel_kwargs)``; see
            ``docs/K_STORE_PERM.md``. INT2 layouts are unchanged by that
            alignment. On Hopper both code pools are in the chunk16 order that
            ``IntPingPongDecodeSm90.code_pool_order`` names,
            ``[num_pages, num_kv_heads, per/16, 128, 16]``, and a row-major
            ``[num_pages, num_kv_heads, 128, per]`` pool is refused by shape
            (``_sm90/serving_cache.py``, CODE POOL ORDERS). A row-major pool
            RESHAPED to that shape is not detectable and reads wrong at INT4;
            convert with ``serving_cache.code_rows_to_order``.
        pscale_lp : Optional[float]
            QUANTIZED MODE ONLY: the launch-global P-scale exponent
            ``floor(log2(448 / max s_v))``.  ``None`` (default) derives it
            from ``scale_pool``, which costs a device sync per call — pass it
            explicitly in a serving loop and recompute it only when the pool's
            max scale changes.
        k_pool_order : Optional[str]
            QUANTIZED MODE ONLY, and MANDATORY for every configuration that reads
            a permuted K pool: INT4 at every configuration (the raw-feed store
            shuffle) and INT2 with the E2M1 feed at either K axis. A caller
            that omits it where it is required is refused rather than run wrong.
            ``"permuted"`` says the K codes were reordered by
            ``cutedsl_decode.pool.k_store_perm_sm100_f4`` before packing; ``"logical"``
            says they are in plain ``pack_rows_lut`` order. ``None`` is accepted
            only for configurations that read logical order: dense callers and
            the INT2 e4m3 feed.

            IT IS REQUIRED BECAUSE NOTHING IN THE POOL REVEALS IT. Both orders
            have the same dtype, the same shape and the same byte count, and the
            wrong pairing decodes to finite, plausible values with every K channel
            taken from the wrong slot inside its packed-word group (16 channels
            at INT2, 8 at INT4). This argument
            is checked against the planned configuration, so a declaration that
            contradicts the plan raises; a declaration that contradicts the POOL
            cannot be caught here, which is why a caller should resolve it from
            ``IntDecodeSm100.k_pool_is_permuted``. That is what a pool writer and
            this guard both ask. The kernel itself no longer consults it to
            select a deposit: its f4 K read is unconditional two masks.
            See ``docs/K_STORE_PERM.md``.
        """
        if not self._planned:
            raise RuntimeError("Call plan() before run().")
        if self._bits is not None:
            if skip_softmax_threshold_scale_factor is not None or lse is not None:
                raise NotImplementedError(
                    "quantized decode supports neither BLASST skip-softmax nor "
                    "an LSE output (the fold kernels have no l_bsh wiring)")
            return self._run_quantized(q, k_cache, v_cache, scale_pool, out,
                                       sm_scale, o_scale, pscale_lp, enable_pdl,
                                       k_pool_order=k_pool_order)
        raise RuntimeError(
            "unreachable: run() requires a quantized plan. plan() rejects a "
            "missing `bits`, so self._bits cannot be None here."
        )
