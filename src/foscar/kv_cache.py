# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""The quantized KV cache's layout and the cache object a decode reads.

`QuantKVLayout` is one layer's cache layout, resolved once per model against the reader that will
decode it, and `QuantKVCache` is one layer's pools in global memory, laid out by that layout. The
decode (`PagedDecodeWrapper.plan(..., layout)` and `run(q, kv_cache)`), the cache writes
(`QuantKVCache.append`, and `append_quantized` for tokens the engine quantized) and the read-back
into bf16 rows (`QuantKVCache.dequantize`) all take their configuration from the same layout
object, which is what keeps a pool from meeting the wrong reader.

WHY THE ORDERS ARE RESOLVED HERE AND NEVER ACCEPTED AS INPUT. Nothing in a pool's bytes says which
order it was written in. On Blackwell an INT4 K pool in `pack_rows_lut` order and one in the store
order the kernel reads have the same dtype, shape and byte count, and the wrong pairing decodes to
finite, plausible values with every K channel taken from the wrong slot of its packed word
(docs/K_STORE_PERM.md). So the orders are the reader's own answers, asked in `create()` for the
architecture of the device the cache will live on: `IntPingPongDecodeSm90.code_pool_order` on
Hopper, and `IntDecodeSm100.k_pool_is_permuted` / `v_pool_token_rows` on Blackwell. A caller
cannot pass one, and a layout built any other way is refused by `__post_init__`, which derives
the same answers again and compares.

WHAT A LAYOUT FIXES. A layout fixes the stored bit width, the head dimension, the KV-head count,
the quantization granularity of K and of V, the scale pool's dtype, and, through those, the shape
and dtype of every pool. The page is always 64 tokens, one decode tile. A layout holds no page count;
`allocate(num_pages)` and `bind(buffer, num_pages)` supply one.

THE PUBLIC `create()` BUILDS PRODUCTION LAYOUTS ONLY. It takes no kernel keyword arguments,
because a key such as `dequant_int2_to_f4=0` selects another INT2 operand path and changes both
Blackwell pool orders, and the library forbids a public knob that selects a path or implies a
layout. The harnesses build ablation and tuning layouts through the private
`_create(..., kernel_kwargs=)`, take their pools from the host packers, and wrap those with the
private `QuantKVCache._adopt`.

THE BYTES OF A BOUND CACHE are region-major, `[all K blocks | all V blocks | all scale blocks |
all page maxima]`, because each reader requires each pool to be C-contiguous on its own. Every
region starts at a multiple of 16 bytes, because every per-page block of the first three regions
is a multiple of 16 bytes. `bind` zeroes the page maxima and nothing else: the writer sets bit
fields rather than OR-ing into them, and when a run starts a page at position 0 it clears that
page's remaining per-token scale records, so no stale scale is read (`QuantKVCache.append`).

THE P-SCALE EXPONENT (Blackwell, token-wise V). The
Blackwell decode quantizes P with one power-of-two scale 2^lp, lp = floor(log2(448 / m)), where m
is the largest V scale in the pool, so lp is the pool-wide exponent. The cache keeps what
gives m without a pass over the pool: `v_page_max`, the largest V scale of each (page, KV head)
in the page's current use (a duplicated slot's own scale excepted, which an append with
duplicate slots leaves unspecified), inside the bound buffer, and `v_max_hist`, a 256-bin count
table outside it, in which bin b counts the page maxima whose exponent bin
(`fold_runtime.vmax_bucket`) is b; a maximum of zero means none and counts nowhere. The writer
keeps the two consistent with atomics, so the table is the histogram of `v_page_max` whenever no
write is in flight, duplicate slots included, and the decode turns the table's highest nonzero
bin into lp before its first tile. Pages of finished requests still count until a run reuses them from
position 0.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from . import _utils
from .quant_layout import quant_block_offsets

#: The most tokens one `append` takes. The writer counts tokens in 32-bit integers, and its page
#: scan steps its token index past the last token by THREADS * SCAN_ROUNDS = 512
#: (`_append_core`) before it stops, so the count must stay that far below 2^31.
MAX_APPEND_TOKENS = 2**31 - 512

#: The most requests one `dequantize` takes. Its kernels count requests in 32-bit integers: a
#: checking CTA's request index ends at most its thread count less one (255 at most) past the
#: last request, and the request search's up to 31 past it, so the count stays this far below
#: 2^31 (`tests/test_dequant_host_limits_cpu.py` holds the kernels to it).
MAX_DEQUANT_REQUESTS = 2**31 - 512

#: The requests per checking CTA that the host counts a `dequantize` launch's grid with. A launch
#: adds one checking CTA per chunk of requests to its page CTAs, a chunk being its kernel's thread
#: count and the last chunk possibly holding a single request (`_dequant_core`, GRID). This value
#: is no larger than either kernel's thread count, so counting with it gives at least as many
#: checking CTAs as either kernel launches, and a grid bounded with it is bounded on both GPUs
#: (`tests/test_dequant_host_limits_cpu.py` holds the kernels to it).
DEQUANT_MIN_CHECK_REQUESTS = 128

#: Tokens per page: one decode tile, one code block and one scale block per (page, KV head).
#: Both readers are built around it.
PAGE_SIZE = 64

#: The reader each architecture's layouts are resolved against.
READERS = {"sm90": "IntPingPongDecodeSm90", "sm100": "IntDecodeSm100"}

#: The scale-pool dtypes a layout may name.
SCALE_DTYPES = (torch.float32, torch.bfloat16)

#: The dtype of `QuantKVCache.v_page_max`: the f32 bit pattern of a page's largest stored V scale.
#: Positive floats order like their bit patterns, so the writer can raise it with `atom.max.u32`.
V_PAGE_MAX_DTYPE = torch.uint32

#: Blackwell kernel keyword arguments a layout carries in fields of its own, refused in
#: `kernel_kwargs` so that one value cannot arrive by two routes.
_SM100_FIELD_KWARGS = ("k_quant", "v_quant", "sdtype")
#: The only kernel keyword argument a Hopper plan takes: the fused-combine policy override,
#: which implies no pool layout.
_SM90_KWARGS = ("fused_combine",)


def clip_index(ratio, head_dim: int) -> int:
    """The order-statistic index of `append`'s per-row clip, or -1 for no clip.

    A clip ratio r in (0, 1] clamps every (token, KV head) row of `head_dim` channels to
    [-t, t] before it is quantized, where t is the element at index int(r * head_dim) of the
    row's |x| sorted ascending, capped at head_dim - 1, so r = 1 leaves every row unchanged.
    That is OSCAR's extend quantizer, `quantized_set_kv_int2_pretransformed_clip_triton` and its
    `_clip_index`, not its decode-flush quantizer, whose bisection lands on the next order
    statistic down. None means no clip; any
    other value outside (0, 1] is refused rather than read as "off"."""
    if ratio is None:
        return -1
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0.0 < ratio <= 1.0:
        raise ValueError(f"a clip ratio is a number in (0, 1], or None for no clip; got {ratio!r}")
    return min(int(ratio * head_dim), head_dim - 1)


def _norm_pair(pair, head_dim, name):
    if pair is None:
        return ("token", head_dim)
    try:
        axis, group = pair
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an (axis, group) pair; got {pair!r}") from None
    return (str(axis), int(group))


def _require_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int; got {value!r}")


def _derive(arch, bits, head_dim, num_kv_heads, k_quant, v_quant, scale_dtype, kernel_kwargs):
    """Every field a layout resolves, as a dict, after the readers' own checks.

    The order is the architecture, then the
    envelope (`validate_quant_flex`, `sm90_wide_refusal`, the accepted head dimensions), then the
    scale dtype, then the pool-order predicates. `kernel_kwargs` is a dict, and empty for a
    production layout.
    """
    from .quant_layout import SM90_WIDE_STAGES, sm90_wide_refusal, validate_quant_flex
    from .wrapper import _ACCEPTED_HEAD_DIMS, _QUANT_BITS

    if arch not in READERS:
        raise ValueError(f"arch must be one of {sorted(READERS)}; got {arch!r}")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in _QUANT_BITS:
        raise ValueError(f"bits must be one of {_QUANT_BITS}; got {bits!r}")
    _require_int("head_dim", head_dim)
    _require_int("num_kv_heads", num_kv_heads)
    if head_dim not in _ACCEPTED_HEAD_DIMS[arch]:
        raise ValueError(
            f"head_dim {head_dim} is not accepted on {arch}; the decode accepts "
            f"{list(_ACCEPTED_HEAD_DIMS[arch])} there (wrapper._ACCEPTED_HEAD_DIMS)")
    k_quant = _norm_pair(k_quant, head_dim, "k_quant")
    v_quant = _norm_pair(v_quant, head_dim, "v_quant")
    token_half = ("token", head_dim // 2)
    if scale_dtype is None:
        scale_dtype = torch.bfloat16 if k_quant == token_half else torch.float32
    if scale_dtype not in SCALE_DTYPES:
        raise ValueError(f"scale_dtype must be one of {SCALE_DTYPES}; got {scale_dtype!r}")
    kkw = dict(kernel_kwargs)

    if arch == "sm90":
        unknown = sorted(set(kkw) - set(_SM90_KWARGS))
        if unknown:
            raise ValueError(
                f"the Hopper decode takes {list(_SM90_KWARGS)} as a kernel keyword argument "
                f"and nothing else; got {unknown}. Its layout fields carry the rest")
        if type(kkw.get("fused_combine", False)) is not bool:
            raise ValueError(f"fused_combine must be a bool; got {kkw['fused_combine']!r}")
        # At a head dimension other than 128 the kernel builds one layout, and every other pair
        # or pool precision is refused with the constructor's own reason, before
        # `validate_quant_flex`, which describes the envelope at 128.
        if head_dim != 128:
            refusal = sm90_wide_refusal(
                head_dim, k_quant, v_quant,
                sdtype="bf16" if scale_dtype == torch.bfloat16 else "f32",
                fold_scale_zero=True, num_stages=SM90_WIDE_STAGES[0])
            if refusal is not None:
                raise ValueError(refusal)
    else:
        if head_dim != 128 and (k_quant != ("token", head_dim)
                                or v_quant != ("token", head_dim)
                                or scale_dtype is not torch.float32 or kkw):
            raise ValueError(f"at head_dim={head_dim} SM100 eager decode builds only whole-head "
                             "token-wise K and V with float32 scales and default kernel settings")
        clash = sorted(set(kkw) & set(_SM100_FIELD_KWARGS))
        if clash:
            raise ValueError(f"{clash} are layout fields; pass them as k_quant=, v_quant= and "
                             f"scale_dtype=, not as kernel keyword arguments")
        if bits == 4 and kkw.get("scale_stages") == 16:
            raise ValueError(
                "scale_stages=16 at INT4 fails nondeterministically on Blackwell and does not "
                "ship (_sm100/fold_runtime.py, the note above BITS_TUNING)")
    for key, value in kkw.items():
        try:
            hash(value)
        except TypeError:
            raise ValueError(f"kernel keyword argument {key}={value!r} must be hashable, "
                             f"because a layout is a frozen, hashable value") from None

    try:
        k_quant, v_quant = validate_quant_flex(k_quant, v_quant, PAGE_SIZE, head_dim)
    except AssertionError as error:
        raise ValueError(f"k_quant={k_quant} v_quant={v_quant} is outside the decode's "
                         f"envelope: {error}") from None

    if arch == "sm90":
        # Token-wise K over D/2 keeps two (s, z) pairs per token, and a bfloat16 element is what
        # funds the second pair's shared memory (`IntPingPongDecodeSm90.__init__`).
        if k_quant == token_half and scale_dtype != torch.bfloat16:
            raise ValueError(f"k_quant={k_quant} needs a bfloat16 scale pool on Hopper; got "
                             f"{scale_dtype}")
        from ._sm90.int_decode_pingpong import IntPingPongDecodeSm90
        orders = dict(code_pool_order=IntPingPongDecodeSm90.code_pool_order,
                      k_pool_permuted=None, v_pool_permuted=None)
    else:
        from ._sm100.int_fold_decode import IntDecodeSm100
        # Token-wise K over D/2 keeps two (s, z) pairs per token, and a bfloat16 element is what
        # funds the second pair's shared memory (`IntDecodeSm100.__init__`), matching Hopper.
        if k_quant == token_half and scale_dtype != torch.bfloat16:
            raise ValueError(f"k_quant={k_quant} needs a bfloat16 scale pool on Blackwell; got "
                             f"{scale_dtype}")
        if kkw:
            _require_buildable(bits, head_dim, k_quant, v_quant, scale_dtype, kkw)
        feed = _sm100_feed(kkw)
        orders = dict(code_pool_order=None,
                      k_pool_permuted=bool(IntDecodeSm100.k_pool_is_permuted(
                          bits, k_quant=k_quant, **feed)),
                      v_pool_permuted=bool(IntDecodeSm100.v_pool_token_rows(bits, **feed)))
    return dict(arch=arch, bits=bits, head_dim=head_dim, num_kv_heads=num_kv_heads,
                k_quant=k_quant, v_quant=v_quant, scale_dtype=scale_dtype,
                reader=READERS[arch], **orders,
                _kernel_kwargs=tuple(sorted(kkw.items(), key=lambda item: item[0])))


def _sm100_feed(kernel_kwargs):
    """The Blackwell keyword arguments both pool-layout predicates read, from a harness's kernel
    keyword arguments; both absent in a production layout, which is the shipped feed."""
    return dict(dequant_int2_to_f4=kernel_kwargs.get("dequant_int2_to_f4"),
                fold_scale_zero=bool(int(kernel_kwargs.get("fold_scale_zero", 1) or 0)))


def _require_buildable(bits, head_dim, k_quant, v_quant, scale_dtype, kernel_kwargs):
    """Refuse harness kernel keyword arguments the Blackwell reader refuses.

    The reader's constructor is the authority, so it is asked: constructed as `compile_fold`
    constructs it, with the per-bits tuning under the harness's keys. Construction runs on the
    host and traces nothing. An unknown name, a feed the width does not take, or a
    tuning value the constructor bounds is refused here, before any pool is built for the
    layout. The query-head tile is a plan's (it follows the GQA ratio), so this asks at 4, the
    tile of the shipped 32/8 shape, and `plan()`'s compile remains the check at the real one."""
    from ._sm100.fold_runtime import BITS_TUNING
    from ._sm100.int_fold_decode import IntDecodeSm100

    knobs = {**BITS_TUNING.get(bits, {}), **kernel_kwargs}
    sdtype = "bf16" if scale_dtype == torch.bfloat16 else "f32"
    try:
        IntDecodeSm100(PAGE_SIZE, head_dim, 4, bits=bits, reduction_mode="kernel",
                       k_quant=k_quant, v_quant=v_quant, sdtype=sdtype, **knobs)
    except (AssertionError, TypeError, ValueError, NotImplementedError) as error:
        raise ValueError(f"IntDecodeSm100 refuses kernel_kwargs={kernel_kwargs} at bits={bits}, "
                         f"k_quant={k_quant}, v_quant={v_quant}: "
                         f"{type(error).__name__}: {error}") from None


@dataclasses.dataclass(frozen=True)
class QuantKVLayout:
    """One layer's quantized KV-cache layout, resolved against the reader that will decode it.

    Build it with `QuantKVLayout.create(...)`. The fields from `reader` on are the reader's own
    answers and are never accepted as input:

    - `code_pool_order`: Hopper, `IntPingPongDecodeSm90.code_pool_order` ("chunk16"); None on
      Blackwell.
    - `k_pool_permuted`: Blackwell, whether the K codes are stored in the kernel's permuted
      within-word order (`IntDecodeSm100.k_pool_is_permuted`); None on Hopper.
    - `v_pool_permuted`: Blackwell, whether V is stored exactly like K
      (`IntDecodeSm100.v_pool_token_rows`): token rows `[64, head_dim * bits / 8]` per (page, KV
      head) in K's within-word order. True in every production layout. False only in the harness's
      INT2 ablation layouts (the E4M3 feed and the fold-off arm), whose V pool is channel rows
      `[head_dim, 64 * bits / 8]` in a token-major order. None on Hopper.

    Two layouts are equal when every field is, which is the comparison `run()` makes between a
    cache and its plan, so two layouts with identical pool shapes but different granularities
    (K channel-64 against channel-128, say) are told apart.
    """
    arch: str
    bits: int
    head_dim: int
    num_kv_heads: int
    k_quant: tuple
    v_quant: tuple
    scale_dtype: torch.dtype
    reader: str
    code_pool_order: Optional[str]
    k_pool_permuted: Optional[bool]
    v_pool_permuted: Optional[bool]
    #: Harness-only kernel keyword arguments, as sorted (name, value) pairs; () in production.
    _kernel_kwargs: tuple = ()

    def __post_init__(self):
        derived = _derive(self.arch, self.bits, self.head_dim, self.num_kv_heads, self.k_quant,
                          self.v_quant, self.scale_dtype, dict(self._kernel_kwargs))
        mine = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        if mine != derived:
            raise ValueError(
                f"a QuantKVLayout is built by QuantKVLayout.create(); these fields disagree "
                f"with what the reader resolves: "
                f"{ {k: (mine[k], derived[k]) for k in derived if mine[k] != derived[k]} }")

    @classmethod
    def create(cls, *, bits, head_dim, num_kv_heads, k_quant=None, v_quant=None,
               scale_dtype=None, device=None) -> "QuantKVLayout":
        """The production layout for `device` (default: the current CUDA device).

        `bits` is 2 or 4. `k_quant` and `v_quant` are `(axis, group)` pairs, `("token",
        head_dim)` by default (the shipped layout); see `quant_layout.validate_quant_flex` for
        the envelope. `scale_dtype` defaults to the reader's: bfloat16 for token-wise K over
        `head_dim // 2`, float32 otherwise; at head dimension 128 either reader also takes
        bfloat16 for the other pairs. The architecture of `device` selects the reader; a device
        with no decode kernel is refused.
        """
        return cls._create(_utils.require_arch(device), bits=bits, head_dim=head_dim,
                           num_kv_heads=num_kv_heads, k_quant=k_quant, v_quant=v_quant,
                           scale_dtype=scale_dtype)

    @classmethod
    def _create(cls, arch, *, bits, head_dim, num_kv_heads, k_quant=None, v_quant=None,
                scale_dtype=None, kernel_kwargs=None) -> "QuantKVLayout":
        """`create()` for a named architecture, with harness kernel keyword arguments. Private.

        It needs no GPU, so CPU tests can build either architecture's layouts. `kernel_kwargs`
        is how a harness reaches an ablation or a tuning knob: on Blackwell any key
        `compile_fold` takes except the layout's own fields, and on Hopper `fused_combine`. A key
        that changes a pool order (`dequant_int2_to_f4`, `fold_scale_zero`) changes the resolved
        order with it, so the harness's packers ask the layout and get the pool that plan reads.
        """
        return cls(**_derive(arch, bits, head_dim, num_kv_heads, k_quant, v_quant, scale_dtype,
                             dict(kernel_kwargs or {})))

    # -- geometry ---------------------------------------------------------------------------

    def _blocks(self) -> dict:
        """Per (page, KV head) block shape of each pool, without the [num_pages, HKV] prefix."""
        off = quant_block_offsets(self.k_quant, self.v_quant, N=PAGE_SIZE, D=self.head_dim)
        scales = (PAGE_SIZE, 4) if off["legacy"] else (off["block_tot"],)
        if self.arch == "sm90":
            from ._sm90.serving_cache import code_block_shape, code_row_bytes
            per_k, per_v = code_row_bytes(self.head_dim, PAGE_SIZE, self.bits)
            k = code_block_shape(self.code_pool_order, per_k)
            v = code_block_shape(self.code_pool_order, per_v)
        else:
            from ._sm100.int_fold_decode import IntDecodeSm100
            k = (PAGE_SIZE, self.head_dim * self.bits // 8)
            # The reader's own slab shape for this layout's feed: token rows like K's on the
            # shipped feeds, channel rows in the harness's INT2 ablation layouts.
            v = tuple(IntDecodeSm100.v_pool_rows(PAGE_SIZE, self.head_dim, self.bits,
                                                 **_sm100_feed(dict(self._kernel_kwargs))))
        return {"k_codes": k, "v_codes": v, "scales": scales,
                "v_page_max": () if self.has_v_page_max else None}

    @property
    def has_v_page_max(self) -> bool:
        """Whether the cache keeps `v_page_max` and `v_max_hist`: Blackwell with token-wise V.

        The Blackwell decode takes its P-scale exponent from the largest V scale in the pool,
        which it reads from the count table the writer keeps (module docstring). Channel-wise V
        keeps a constant exponent and needs none, and Hopper has no such exponent."""
        return self.arch == "sm100" and self.v_quant[0] == "token"

    def pool_dtypes(self) -> dict:
        """The dtype of each pool, keyed as `pool_shapes` keys them."""
        return {"k_codes": torch.uint8, "v_codes": torch.uint8, "scales": self.scale_dtype,
                "v_page_max": V_PAGE_MAX_DTYPE if self.has_v_page_max else None}

    def pool_shapes(self, num_pages: int) -> dict:
        """The shape of each pool of a `num_pages`-page cache; `v_page_max` is None where the
        layout keeps none."""
        _require_int("num_pages", num_pages)
        return {name: None if block is None else (num_pages, self.num_kv_heads, *block)
                for name, block in self._blocks().items()}

    def _page_bytes(self) -> dict:
        """Bytes per page of each pool, all KV heads."""
        sizes = {}
        for name, block in self._blocks().items():
            if block is None:
                continue
            elems = self.num_kv_heads
            for extent in block:
                elems *= extent
            sizes[name] = elems * self.pool_dtypes()[name].itemsize
        return sizes

    def bytes_per_page(self) -> int:
        """Bytes one page takes across every pool and KV head: K, V, scales and, where the
        layout keeps them, page maxima. An engine sizes its per-layer KV buffer with this."""
        return sum(self._page_bytes().values())

    # -- caches -----------------------------------------------------------------------------

    def allocate(self, num_pages: int, *, device=None) -> "QuantKVCache":
        """A fresh `num_pages`-page cache on `device` (default: the current CUDA device), laid out
        and initialized as `bind` does it, under the same rules: not inside a CUDA graph capture.
        Its other bytes are not initialized, and need not be, provided every use of a page begins
        with a write at position 0 (`QuantKVCache`)."""
        _require_int("num_pages", num_pages)
        index = _utils.cuda_index(device)
        buffer = torch.empty(num_pages * self.bytes_per_page(), dtype=torch.uint8,
                             device=torch.device("cuda", index))
        return self.bind(buffer, num_pages)

    def bind(self, buffer: torch.Tensor, num_pages: int) -> "QuantKVCache":
        """A `num_pages`-page cache over an engine-owned uint8 `buffer`, region by region.

        The buffer must be uint8, C-contiguous, 16-byte aligned at its start, at least
        `num_pages * bytes_per_page()` bytes, and on a device of this layout's architecture. Its
        rank does not matter (an engine's `[num_blocks, bytes_per_page]` tensor works). Row `i`
        of such a tensor is NOT page `i`: the pools are region-major, so an engine path that
        copies blocks by indexing its first dimension must go through the cache instead.

        Where the layout keeps page maxima (`has_v_page_max`), `bind` zeroes that region with an
        asynchronous fill on the current stream and creates the cache's own zeroed count table
        `v_max_hist`, outside the buffer, so the table starts as the histogram of the maxima.
        That makes it a write, with these rules: bind exactly once per buffer, before any append,
        on the stream of the first append or ordered before it; never over a buffer whose pages
        live requests still reference, because their maxima are zeroed and stop counting; and
        never inside a CUDA graph capture, because a replay would empty a live cache. Nothing
        else is written.
        """
        _require_int("num_pages", num_pages)
        if not isinstance(buffer, torch.Tensor):
            raise TypeError(f"bind() takes a torch.Tensor; got {type(buffer).__name__}")
        if buffer.dtype != torch.uint8:
            raise ValueError(f"bind() takes a uint8 buffer; got {buffer.dtype}")
        if not buffer.is_contiguous():
            raise ValueError("bind() takes a C-contiguous buffer")
        need = num_pages * self.bytes_per_page()
        if buffer.numel() < need:
            raise ValueError(f"a {num_pages}-page cache of this layout needs {need} bytes; the "
                             f"buffer holds {buffer.numel()}")
        if buffer.data_ptr() % 16:
            raise ValueError("bind() takes a buffer whose start is 16-byte aligned")
        _require_device(self, buffer.device)
        flat = buffer.view(-1)
        shapes, dtypes = self.pool_shapes(num_pages), self.pool_dtypes()
        pools, start = {}, 0
        for name, per_page in self._page_bytes().items():
            region = flat[start:start + num_pages * per_page]
            if name == "v_page_max":
                region.zero_()           # on the uint8 bytes: zero means "no maximum"
            if dtypes[name] != torch.uint8:
                region = region.view(dtypes[name])
            pools[name] = region.view(shapes[name])
            start += num_pages * per_page
        v_page_max, v_max_hist = pools.get("v_page_max"), None
        if v_page_max is not None:
            from ._sm100.fold_runtime import VMAX_BINS
            # the histogram of all-zero maxima: empty
            v_max_hist = torch.zeros(VMAX_BINS, dtype=torch.int32,
                                     device=buffer.device).view(V_PAGE_MAX_DTYPE)
        return QuantKVCache._build(self, pools["k_codes"], pools["v_codes"], pools["scales"],
                                   v_page_max, v_max_hist)


#: The Blackwell V layout a refused `v_codes` must be rebuilt in, by `v_pool_permuted`: the two
#: slab shapes hold the same bytes, so a pool of the other layout reaches `_adopt` only as a
#: shape mismatch.
_V_LAYOUT_HINT = {
    True: ("; this layout stores V exactly like K, in token rows (docs/SM100_ALIGNED_KV.md), so "
           "a V pool written in the earlier channel-row layout must be rebuilt"),
    False: ("; this ablation layout reads channel-row V, so a V pool stored like K must be "
            "rebuilt"),
}


def _require_device(layout, device):
    """Refuse a device whose architecture is not the layout's."""
    arch = _utils.require_arch(device)
    if arch != layout.arch:
        raise ValueError(f"this layout was resolved for {layout.arch}, and {device} is {arch}; "
                         f"create the layout for the device the cache lives on")


class QuantKVCache:
    """One layer's quantized KV cache in global memory: the K and V code pools, the scale pool
    and, where the layout keeps them, the per-page V-scale maxima.

    Built only by `QuantKVLayout.allocate`, `QuantKVLayout.bind`, or, for host-built fixtures,
    the private `QuantKVCache._adopt`, each of which checks the device, shapes, dtypes,
    contiguity and alignment against the layout once, so that `run()` and `append()` need not.

    On Blackwell with token-wise V (`QuantKVLayout.has_v_page_max`) the cache also keeps, in
    global memory, `v_page_max` (uint32 [NP, HKV], inside the bound buffer), the bit pattern of
    the largest V scale written during each (page, KV head)'s current use, 0 for none; and
    `v_max_hist` (uint32 [256], the cache's own tensor, at a fixed address for its life), the
    count of page maxima in each exponent bin, which the decode reads (module docstring).
    Elsewhere both are None.

    CONTRACTS the engine keeps:
    - Every use of a page begins with a write at position 0, on both GPUs, which paged
      allocation guarantees: a request reaches a new page at offset 0, prefill writes from
      position 0, and a recomputed request rewrites from position 0. That write clears the
      page's stale scale records (`append`), which the decodes read with a weight of zero past
      the length, and it makes the Blackwell maximum exact. A never-written page whose first
      use started mid-page would keep whatever the buffer held, which can turn those zero
      weights into NaN. On a page written before, a use that starts mid-page keeps the earlier
      use's maximum, which can lower the decode's precision but never lets it overflow.
    - No decode of a layer runs concurrently with an append to the same layer, on any stream.
      Stream order on one stream gives this; across streams the engine orders them. On
      Blackwell the layer's count table is shared by all its pages, which is why the rule
      covers disjoint requests too.
    - Bytes copied into the buffer outside `append` (swap-in, offload, disaggregated transfer,
      defragmentation) change `v_page_max` behind the table's back, so after such a copy the
      engine calls `recount_v_max()`, ordered after the copy and before the next decode of that
      layer, with no append or decode of the layer concurrent with it.
    - `bind` once per buffer, before any append (`QuantKVLayout.bind`).
    """

    __slots__ = ("layout", "num_pages", "device", "k_codes", "v_codes", "scales", "v_page_max",
                 "v_max_hist", "_writer", "_writer_prequantized", "_clipped_writers",
                 "_dequantizers")

    def __init__(self, *args, **kwargs):
        raise TypeError("a QuantKVCache is built by QuantKVLayout.allocate() or "
                        "QuantKVLayout.bind()")

    @classmethod
    def _build(cls, layout, k_codes, v_codes, scales, v_page_max, v_max_hist) -> "QuantKVCache":
        self = object.__new__(cls)
        self.layout = layout
        self.num_pages = int(k_codes.shape[0])
        self.device = k_codes.device
        self.k_codes, self.v_codes, self.scales = k_codes, v_codes, scales
        self.v_page_max, self.v_max_hist = v_page_max, v_max_hist
        self._writer = None
        self._writer_prequantized = None
        # The clipped writers this cache has used, keyed by their (K, V) clip indices.
        self._clipped_writers = {}
        self._dequantizers = {}
        return self

    @classmethod
    def _adopt(cls, layout, k_codes, v_codes, scales, v_page_max=None,
               v_max_hist=None) -> "QuantKVCache":
        """A cache over pools a host packer built. Private: the fixture path of the gates,
        benchmarks and adapters, whose packers take their orders from `layout`.

        The pools must have exactly the layout's shapes and dtypes for one page count, be
        C-contiguous and 16-byte aligned, and share one device of the layout's architecture.
        They are wrapped in place, never copied. Every value in them must be one the cache
        writer could have stored for an input inside its envelope (every K and V element within
        +-2^64), at every position of every page, tails included, because the decodes read a
        partial page's tail positions with a weight of zero and a scale record outside that set
        (a NaN, an infinity, or a tiny scale with a huge zero point) can still reach the output.

        On Blackwell with token-wise V, `v_page_max` is derived from `scales` over every
        position of each page (`fold_runtime.v_page_max_for_pool`) and `v_max_hist` from it
        (`fold_runtime.v_max_hist_of`), unless the caller passes both: a benchmark that places
        every tensor the decode reads passes placed copies of a pair derived earlier. One
        without the other is refused, because the table must be the histogram of the maxima, and
        so is passing them where the layout keeps none.
        """
        if not isinstance(layout, QuantKVLayout):
            raise TypeError(f"_adopt() takes a QuantKVLayout; got {type(layout).__name__}")
        pools = {"k_codes": k_codes, "v_codes": v_codes, "scales": scales}
        if (v_page_max is None) != (v_max_hist is None):
            raise ValueError("pass v_page_max and v_max_hist together, or neither")
        if v_page_max is not None:
            if not layout.has_v_page_max:
                raise ValueError("this layout keeps no v_page_max (Hopper, or channel-wise V)")
            pools["v_page_max"] = v_page_max
            pools["v_max_hist"] = v_max_hist
        for name, pool in pools.items():
            if not isinstance(pool, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor; got {type(pool).__name__}")
        num_pages = int(k_codes.shape[0]) if k_codes.dim() else 0
        shapes, dtypes = layout.pool_shapes(num_pages), layout.pool_dtypes()
        if v_max_hist is not None:
            from ._sm100.fold_runtime import VMAX_BINS
            shapes = {**shapes, "v_max_hist": (VMAX_BINS,)}
            dtypes = {**dtypes, "v_max_hist": V_PAGE_MAX_DTYPE}
        for name, pool in pools.items():
            if tuple(pool.shape) != shapes[name] or pool.dtype != dtypes[name]:
                raise ValueError(f"{name} is {tuple(pool.shape)} {pool.dtype}; this layout's "
                                 f"{num_pages}-page {name} is {shapes[name]} {dtypes[name]}"
                                 + (_V_LAYOUT_HINT[layout.v_pool_permuted]
                                    if name == "v_codes" and layout.arch == "sm100" else ""))
            if not pool.is_contiguous():
                raise ValueError(f"{name} must be C-contiguous")
            if pool.data_ptr() % 16:
                raise ValueError(f"{name} must start 16-byte aligned")
            if pool.device != k_codes.device:
                raise ValueError(f"{name} is on {pool.device} and k_codes on {k_codes.device}")
        _require_device(layout, k_codes.device)
        if v_page_max is None and layout.has_v_page_max:
            from ._sm100.fold_runtime import v_max_hist_of, v_page_max_for_pool
            v_page_max = v_page_max_for_pool(scales, layout.k_quant, layout.v_quant,
                                             page_size=PAGE_SIZE, head_dim=layout.head_dim)
            v_max_hist = v_max_hist_of(v_page_max)
        return cls._build(layout, k_codes, v_codes, scales, v_page_max, v_max_hist)

    def recount_v_max(self) -> None:
        """Rebuild `v_max_hist` from `v_page_max`, in place, on the current stream.

        Call it after any copy into the cache's bytes that bypasses `append` (swap-in, offload,
        disaggregated transfer, defragmentation), before the next decode of this layer: such a
        copy moves page maxima without moving their counts, and a copied-in maximum above the
        counted ones would let the decode's P products overflow. Device only, with no wait for
        the device, no allocation that depends on the data, and a CUDA graph can capture it. No
        append or decode of this layer may run concurrently with it. A no-op where the layout
        keeps no page maxima.

        It repairs the cells as well as the counts. A cell must hold the bit pattern of a
        sign-free f32 page maximum, +0.0 meaning no maximum, because the writer compares cells
        as unsigned words: a copied-in cell with its sign bit set (a -0.0, or a negated scale)
        would compare above every scale a later append brings and keep that append from
        raising it. Recount first clears every cell's sign bit, in place, then rebuilds the
        table from the cleared cells.
        """
        if self.v_page_max is not None:
            from ._sm100.fold_runtime import v_max_hist_of
            self.v_page_max.view(torch.int32).bitwise_and_(0x7FFFFFFF)
            v_max_hist_of(self.v_page_max, out=self.v_max_hist)

    def append(self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor, *,
               k_clip=None, v_clip=None) -> None:
        """Quantize and write tokens into the cache, one launch for K and V of every token.

        `k` and `v` are bfloat16 `[T, num_kv_heads, head_dim]` with the head dimension
        contiguous and a free token stride (a view of a fused QKV output works), every row
        16-byte aligned. `slot_mapping` is int64 `[T]`: token `t` goes to page
        `slot // 64`, position `slot % 64`; a negative slot is skipped, and a slot at or past
        `num_pages * 64` traps on the device. The write runs on the current stream, so a
        decode queued after it on that stream reads it, and no decode of this layer may run
        concurrently with it on another stream (the class docstring), and appends that touch the
        same pages must be ordered on one stream too, because a run that starts a page at
        position 0 clears and resets what another append of that page may be writing. Duplicate
        valid slots in one call leave unspecified the whole page's contents on Hopper, where
        tokens share code words, and the duplicated slot's contents on Blackwell; route padding
        writes to a page no request uses (docs/integration.md).

        Every use of a page must begin with a write at position 0 (the class docstring). A run
        of tokens that starts a page at position 0 also clears that page's other per-token scale
        records to zero, except at the positions other tokens of this launch write, so that a
        page reused from position 0 holds no stale scale past the request's length. A rewrite
        of position 0 inside a live use therefore zeroes the use's later positions too, which
        paged allocation never does. On Blackwell with token-wise V the
        launch also keeps `v_page_max` and `v_max_hist`.

        Token-wise layouts only: a layout with a channel-wise K or V is written one complete
        quantization group at a time, which `append` cannot do.

        `k_clip` and `v_clip` (default None, no clip) clamp each (token, KV head) row of K or V to
        plus or minus one of its own order statistics before the minimum and maximum are taken,
        OSCAR's per-row quantile clip (`clip_index`); with K token-wise over D/2 the threshold
        comes from the whole row of D channels and each half then takes its own range. The clip
        changes the values written, not the pools' format, so every decode reads a clipped cache
        as it reads any other, and a call without them launches exactly the writer it always did.
        Each distinct pair compiles its own writer once per cache.
        """
        layout = self.layout
        self._refuse_channel_wise("append")
        shape = (layout.num_kv_heads, layout.head_dim)
        for name, x in (("k", k), ("v", v)):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.dim() != 3 \
                    or tuple(x.shape[1:]) != shape:
                raise ValueError(f"{name} must be a bfloat16 [T, {shape[0]}, {shape[1]}] tensor; "
                                 f"got {getattr(x, 'dtype', None)} "
                                 f"{tuple(getattr(x, 'shape', ()))}")
            if x.device != self.device:
                raise ValueError(f"{name} is on {x.device}; the cache is on {self.device}")
            if x.stride(2) != 1 or x.stride(1) != layout.head_dim:
                raise ValueError(f"{name} must have a contiguous [num_kv_heads, head_dim] row "
                                 f"per token (strides (*, {layout.head_dim}, 1)); got "
                                 f"{tuple(x.stride())}")
            if x.data_ptr() % 16 or (x.stride(0) * x.element_size()) % 16:
                raise ValueError(f"every token row of {name} must start 16-byte aligned")
        if k.shape[0] != v.shape[0]:
            raise ValueError(f"k has {k.shape[0]} tokens and v has {v.shape[0]}")
        if k.shape[0] > MAX_APPEND_TOKENS:
            raise ValueError(f"one append takes at most {MAX_APPEND_TOKENS} tokens (the writer "
                             f"counts tokens in 32-bit integers); got {k.shape[0]}: split it "
                             f"into several appends")
        self._check_slots(slot_mapping, k.shape[0])
        clip = (clip_index(k_clip, layout.head_dim), clip_index(v_clip, layout.head_dim))
        if clip == (-1, -1):
            writer = self._writer
            if writer is None:
                writer = self._writer = _writer_for(layout)
        else:
            writer = self._clipped_writers.get(clip)
            if writer is None:
                writer = self._clipped_writers[clip] = _writer_for(layout, clip=clip)
        self._launch(writer, (k, v), slot_mapping)

    def append_quantized(self, k_codes: torch.Tensor, k_scale: torch.Tensor,
                         k_zero: torch.Tensor, v_codes: torch.Tensor, v_scale: torch.Tensor,
                         v_zero: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Write tokens the engine has already quantized into the cache, one launch for K and V
        of every token: the layout-only form of `append`.

        `k_codes` and `v_codes` are uint8 `[T, num_kv_heads, head_dim]`, C-contiguous and
        16-byte aligned, one quantized value per byte, 0 to qmax = 2^bits - 1; the bits of a
        byte above the stored width are ignored. `k_scale` and `k_zero` are float32
        `[T, num_kv_heads]` when K is token-wise over the whole head, and
        `[T, num_kv_heads, 2]` when it is token-wise over `head_dim // 2`, where group 0 is
        channels `[0, head_dim / 2)` and group 1 the rest; `v_scale` and `v_zero` are float32
        `[T, num_kv_heads]`; all four C-contiguous. `slot_mapping` is as in `append`. A stored
        token is x = s * q + z per channel, the library's convention (`quant_ref`).

        Everything but the quantization is `append`'s: the same layouts, page runs, tail clear
        of a page a run starts at position 0, Blackwell page maximum and count table, and the
        duplicate-slot, ordering and position-0 contracts (`append`, the class docstring). The
        scale record is rounded to the pool's dtype as `append` rounds its own.

        SATURATION. Each scale is saturated to [div.rn(1e-8, qmax), div.rn(2^65, qmax)] and each
        zero point to [-2^64, 2^64], and a NaN to the lower bound: the ranges of the scales and
        zero points `append` stores under its input envelope (every K and V element within
        +-2^64), which keep the decodes' products finite and every page maximum positive. The
        two bounds are independent, so they do
        not make every record one `append` could store; pass the records your own quantizer
        made from finite inputs.

        FROM OSCAR. OSCAR stores scale = max(mx - mn, 1e-8) / 3, zero = -mn / scale and
        dequantizes (q - zero) * scale. The engine passes s = scale and z = fl(-zero * scale)
        (one f32 multiply, round to nearest): the integers and the scale are identical, and
        |z - (-zero * scale)| <= 2^-24 * |z|. Computed in f32, s * q + z then differs from
        OSCAR's (q - zero) * scale only by rounding: each form rounds a product once and a sum
        or difference once, so to first order in 2^-24 the two differ by at most
        6 * 2^-24 * max(|x|, |z|), a few f32 ulps of max(|x|, |z|), where x is (q - zero) * scale
        evaluated exactly from OSCAR's stored zero and scale. That is small against the INT2
        step s only while max(|x|, |z|) / s is well below 2^24 / 6, about 2.8e6. A row whose
        range is tiny beside its offset goes past it: mn = 1000 and mx = 1000.00006103515625
        give s = 2.03e-5 and |z| / s = 4.9e7, and at q = 3 the two forms differ by 3 s.
        Neither form resolves the step there: the spacing at which each rounds its sum,
        ulp(|z|) = 6.1e-5 for s * q + z and ulp(|zero|) * scale = 8.1e-5 for OSCAR's, is
        larger than s.

        The first call for a layout compiles this writer, apart from `append`'s, so an engine
        warms it before it captures a CUDA graph. Token-wise layouts only, as `append`.
        """
        layout = self.layout
        self._refuse_channel_wise("append_quantized")
        H, D = layout.num_kv_heads, layout.head_dim
        T = k_codes.shape[0] if isinstance(k_codes, torch.Tensor) and k_codes.dim() else None
        k_group = (T, H, 2) if tuple(layout.k_quant) == ("token", D // 2) else (T, H)
        for name, x, dtype, shape in (("k_codes", k_codes, torch.uint8, (T, H, D)),
                                      ("k_scale", k_scale, torch.float32, k_group),
                                      ("k_zero", k_zero, torch.float32, k_group),
                                      ("v_codes", v_codes, torch.uint8, (T, H, D)),
                                      ("v_scale", v_scale, torch.float32, (T, H)),
                                      ("v_zero", v_zero, torch.float32, (T, H))):
            if (not isinstance(x, torch.Tensor) or x.dtype != dtype
                    or tuple(x.shape) != shape):
                want = ["T" if extent is None else extent for extent in shape]
                raise ValueError(f"{name} must be a {dtype} {want} tensor; got "
                                 f"{getattr(x, 'dtype', None)} {tuple(getattr(x, 'shape', ()))}")
            if not x.is_contiguous():
                raise ValueError(f"{name} must be C-contiguous")
            if x.device != self.device:
                raise ValueError(f"{name} is on {x.device}; the cache is on {self.device}")
            if dtype == torch.uint8 and x.data_ptr() % 16:
                raise ValueError(f"{name} must start 16-byte aligned")
        if T > MAX_APPEND_TOKENS:
            raise ValueError(f"one append takes at most {MAX_APPEND_TOKENS} tokens (the writer "
                             f"counts tokens in 32-bit integers); got {T}: split it into "
                             f"several appends")
        self._check_slots(slot_mapping, T)
        writer = self._writer_prequantized
        if writer is None:
            writer = self._writer_prequantized = _writer_for(layout, prequantized=True)
        self._launch(writer, (k_codes, k_scale, k_zero, v_codes, v_scale, v_zero),
                     slot_mapping)

    def _refuse_channel_wise(self, method):
        layout = self.layout
        if layout.k_quant[0] == "channel" or layout.v_quant[0] == "channel":
            # append_groups must clear the token-wise section of a page it starts at
            # position 0, as `append` does, because no decode masks a stale tail scale.
            raise NotImplementedError(
                f"{method}() writes token-wise layouts only; k_quant={layout.k_quant} "
                f"v_quant={layout.v_quant} has a channel-wise tensor, whose groups are written "
                f"whole by append_groups (docs/SERVING_READINESS.md, section 5), which is not "
                f"available yet")

    def _check_slots(self, slot_mapping, num_tokens):
        if (not isinstance(slot_mapping, torch.Tensor) or slot_mapping.dtype != torch.int64
                or tuple(slot_mapping.shape) != (num_tokens,)
                or not slot_mapping.is_contiguous()):
            raise ValueError(f"slot_mapping must be a contiguous int64 [{num_tokens}] tensor")
        if slot_mapping.device != self.device:
            raise ValueError(f"slot_mapping is on {slot_mapping.device}; the cache is on "
                             f"{self.device}")

    def _launch(self, writer, inputs, slot_mapping):
        compiled, launch = writer
        with _utils.on_device(self.device):
            launch(compiled, inputs=inputs, slot_mapping=slot_mapping, k_codes=self.k_codes,
                   v_codes=self.v_codes, scales=self.scales, v_page_max=self.v_page_max,
                   v_max_hist=self.v_max_hist)

    def dequantize(self, k_out: torch.Tensor, v_out: torch.Tensor, page_indptr: torch.Tensor,
                   page_indices: torch.Tensor, seq_lens: torch.Tensor,
                   out_indptr: torch.Tensor) -> None:
        """Read the K and V of a batch of requests back into bf16 rows, one launch for both.

        `page_indptr` is int32 `[B + 1]`, nondecreasing: request `b` owns entries
        `[page_indptr[b], page_indptr[b + 1])` of `page_indices`, int32 `[P]`, its physical
        pages in logical order (FlashInfer's `kv_indptr` and `kv_indices`, in pages). Size
        `page_indices` to the pages passed, because the launch is sized from its length.
        `seq_lens` is int32 `[B]`: request `b`'s positions `[0, seq_lens[b])`, which need
        `ceil(seq_lens[b] / 64)` of its pages and must be at least 0; its entries past those are
        not read and may hold anything. `out_indptr` is int32 or int64 `[B]`, the row of `k_out`
        and `v_out` that request `b`'s position 0 goes to. `k_out` and `v_out` are bfloat16
        `[N, num_kv_heads, head_dim]` with the head dimension contiguous, a free token stride
        and 16-byte aligned rows (32-byte aligned rows avoid partial-sector stores); the rows of
        different requests may leave gaps but must not overlap.

        It writes `k_out[out_indptr[b] + t] = bf16(fma(s_k, q, z_k))` channel by channel, and
        `v_out` likewise, for every `t < seq_lens[b]`, from the stored code q and the token's
        stored scale and zero point, rounded once to f32 and then to bf16; nothing else of
        `k_out` or `v_out` is written. Every tensor stays on the device: no host
        synchronization, allocation or device value read, so a CUDA graph can capture it. A
        page id outside the cache on a page the call reads, an output row outside `k_out` or
        `v_out`, and a negative length or one that needs more pages than its request's entries
        (none included) trap on the device, as `append` does for a slot outside the cache. The
        launch runs on the current stream, ordered after the appends queued before it; no
        append to the pages it reads may run concurrently with it on another stream. Its first
        call for a dtype of `out_indptr` compiles the kernel, which then serves every batch and
        page count.

        One call takes at most `MAX_DEQUANT_REQUESTS` requests, and its page entries times
        `num_kv_heads` plus `ceil(B / DEQUANT_MIN_CHECK_REQUESTS)` must not pass 2^31 - 1 (the
        launch's CTAs: one per entry and KV head, and the checking CTAs, which the host counts
        at `DEQUANT_MIN_CHECK_REQUESTS` requests each on both GPUs, at least as many as either
        kernel launches); a larger one is refused on the host, from the tensor shapes, and must
        be split. Token-wise layouts only, the layouts `append` writes.
        """
        layout = self.layout
        if layout.k_quant[0] == "channel" or layout.v_quant[0] == "channel":
            raise NotImplementedError(
                f"dequantize() reads token-wise layouts only, the layouts append() writes; "
                f"k_quant={layout.k_quant} v_quant={layout.v_quant} has a channel-wise tensor")
        H, D = layout.num_kv_heads, layout.head_dim
        for name, x in (("k_out", k_out), ("v_out", v_out)):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.dim() != 3 \
                    or tuple(x.shape[1:]) != (H, D):
                raise ValueError(f"{name} must be a bfloat16 [N, {H}, {D}] tensor; got "
                                 f"{getattr(x, 'dtype', None)} "
                                 f"{tuple(getattr(x, 'shape', ()))}")
            if x.device != self.device:
                raise ValueError(f"{name} is on {x.device}; the cache is on {self.device}")
            if x.stride(2) != 1 or x.stride(1) != D or (x.shape[0] > 1 and x.stride(0) < H * D):
                raise ValueError(f"{name} must hold each token's [num_kv_heads, head_dim] row "
                                 f"contiguously, rows not overlapping (strides (>= {H * D}, "
                                 f"{D}, 1)); got {tuple(x.stride())}")
            if x.data_ptr() % 16 or (x.stride(0) * x.element_size()) % 16:
                raise ValueError(f"every token row of {name} must start 16-byte aligned")
        index = {"page_indptr": (page_indptr, (torch.int32,)),
                 "page_indices": (page_indices, (torch.int32,)),
                 "seq_lens": (seq_lens, (torch.int32,)),
                 "out_indptr": (out_indptr, (torch.int32, torch.int64))}
        for name, (x, dtypes) in index.items():
            if not isinstance(x, torch.Tensor) or x.dtype not in dtypes or x.dim() != 1 \
                    or not x.is_contiguous():
                raise ValueError(f"{name} must be a contiguous one-dimensional "
                                 f"{' or '.join(str(d) for d in dtypes)} tensor; got "
                                 f"{getattr(x, 'dtype', None)} {tuple(getattr(x, 'shape', ()))}")
            if x.device != self.device:
                raise ValueError(f"{name} is on {x.device}; the cache is on {self.device}")
        batch = seq_lens.shape[0]
        if page_indptr.shape[0] != batch + 1 or out_indptr.shape[0] != batch:
            raise ValueError(f"seq_lens has {batch} requests, so page_indptr must have "
                             f"{batch + 1} entries and out_indptr {batch}; got "
                             f"{page_indptr.shape[0]} and {out_indptr.shape[0]}")
        refusal = _dequant_size_refusal(batch, page_indices.shape[0], H)
        if refusal is not None:
            raise ValueError(refusal)
        wide = out_indptr.dtype == torch.int64
        dequantizer = self._dequantizers.get(wide)
        if dequantizer is None:
            dequantizer = self._dequantizers[wide] = _dequantizer_for(layout, wide)
        compiled, launch = dequantizer
        with _utils.on_device(self.device):
            launch(compiled, k_codes=self.k_codes, v_codes=self.v_codes, scales=self.scales,
                   k_out=k_out, v_out=v_out, page_indptr=page_indptr,
                   page_indices=page_indices, seq_lens=seq_lens, out_indptr=out_indptr)


def _dequant_size_refusal(num_requests, num_entries, num_kv_heads):
    """Why `dequantize` refuses a call of this many requests and page entries, or None: its
    kernels count requests in 32-bit integers (`MAX_DEQUANT_REQUESTS`), and it launches one CTA
    per entry and KV head and then one checking CTA per thread count of requests, a grid size of
    at most 2^31 - 1. The checking CTAs are counted here at `DEQUANT_MIN_CHECK_REQUESTS`
    requests each, at least as many as either kernel launches, so that one bound holds on both
    GPUs; where a kernel checks more requests per CTA, it refuses some calls whose real grid
    would fit."""
    if num_requests > MAX_DEQUANT_REQUESTS:
        return (f"one dequantize takes at most {MAX_DEQUANT_REQUESTS} requests (its kernels count "
                f"them in 32-bit integers); got {num_requests}: split it into several calls")
    checking = -(-num_requests // DEQUANT_MIN_CHECK_REQUESTS)
    if num_entries * num_kv_heads + checking > 2 ** 31 - 1:
        return (f"one dequantize launches a CTA per page entry and KV head plus its checking "
                f"CTAs, which the host counts at {DEQUANT_MIN_CHECK_REQUESTS} requests per "
                f"checking CTA on both GPUs (at least as many CTAs as either kernel launches), "
                f"at most 2^31 - 1 in all; got {num_entries} entries, {num_kv_heads} KV heads "
                f"and {num_requests} requests: split it into several calls")
    return None


def _dequantizer_for(layout, out_indptr_64):
    """(compiled, launch) of this layout's dequantize for the out_indptr dtype (int64 when
    True), compiled for its architecture."""
    _utils.require_dsl_target(layout.arch)
    if layout.arch == "sm90":
        from ._sm90.quant_dequant import compile_quant_dequant, launch_quant_dequant
    else:
        from ._sm100.quant_dequant import compile_quant_dequant, launch_quant_dequant
    return compile_quant_dequant(layout, out_indptr_64), launch_quant_dequant


def _writer_for(layout, prequantized=False, clip=(-1, -1)):
    """(compiled, launch) of this layout's cache writer, compiled for its architecture: the
    quantizing writer of `append` with the (K, V) clip indices `clip`, or with `prequantized`
    the writer of `append_quantized`."""
    _utils.require_dsl_target(layout.arch)
    if layout.arch == "sm90":
        from ._sm90.quant_append import compile_quant_append, launch_quant_append
    else:
        from ._sm100.quant_append import compile_quant_append, launch_quant_append
    return compile_quant_append(layout, prequantized, clip), launch_quant_append
