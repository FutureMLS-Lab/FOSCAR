# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""The paged dequantize both GPUs share: finding a page's request, the scale records, and the f32
arithmetic that turns a stored value into bf16.

`_sm90/quant_dequant.QuantDequantSm90` and `_sm100/quant_dequant.QuantDequantSm100` read a
layer's quantized cache back into bf16 K and V rows, for an engine whose prefill attends over the
cache with a bf16 attention kernel (`QuantKVCache.dequantize`). Everything that does not depend on
the code-pool layout lives here; each subclass supplies the kernel body that reads its own code
pools.

INPUTS. `page_indptr` is int32 [B + 1], nondecreasing: request b owns the entries
[page_indptr[b], page_indptr[b + 1]) of `page_indices`, int32 [P], its physical pages in logical
order (FlashInfer's kv_indptr and kv_indices, in pages). `seq_lens` is int32 [B]: request b's
logical positions [0, seq_lens[b]), which need ceil(seq_lens[b] / 64) of its pages; the entries
past those are slack and are not read. `out_indptr` is int32 or int64 [B]: the row of k_out and
v_out that request b's position 0 goes to. `k_out` and `v_out` are bf16 [N, HKV, D], D
contiguous, head stride D, a free token stride, 16-byte aligned rows.

WHAT IS WRITTEN. k_out[out_indptr[b] + t] = bf16(fma(s_k, q, z_k)) channel by channel, and v_out
likewise, for every b and t < seq_lens[b], where q is the stored code and (s, z) the token's
stored scale record widened to f32 (a bf16 pool's values exactly). The fma is one f32
`fma.rn`, and the code enters it exactly: a code q < 256 placed in the low byte of 0x4B000000
is the f32 2^23 + q, and subtracting 2^23 leaves q (`code_value`), so no integer-to-float
conversion is issued. bf16 is round-to-nearest-even of that f32 (`cvt.rn.bf16x2.f32`). Nothing
else of k_out or v_out is touched.

GRID. One CTA per (entry p of page_indices, KV head), P x HKV CTAs with the head fastest, so
the CTAs of one page, which write the interleaved heads of the same output rows, run together,
and after them ceil(B / THREADS) checking CTAs (THREADS, the kernel's thread count, is 128 on
Blackwell and 256 on Hopper), which check the requests' lengths against their entries
(`_validate`), THREADS requests each, one per thread, and store nothing. So a request
with no entries, or a call whose page list is empty, is checked too, and no page's CTA waits for
the check. Each checking CTA makes one round of loads and exits.
A CTA finds its request by a warp-parallel search of page_indptr (`_locate`): each round, 32
lanes sample the remaining range and a ballot narrows it 32 times, so B requests take
ceil(log32 B) rounds of one load each. The loads that do not depend on the request, the page id
and then the page's code words and scale records, are issued before the search, so the search
overlaps them. A CTA whose entry lies outside [page_indptr[0], page_indptr[B]), or whose
logical page starts at or past its request's length (a slack entry), exits without storing.
Until the search resolves the page id is not known to be one this call reads, so the loads use
page 0 in place of an id outside [0, NP), and a CTA that needs such a page traps.

One CTA per (entry, KV head) is the fastest grid measured. One CTA per page for all its heads,
with one search per page, is slower at every size, by far the most at small sizes, where it
leaves most SMs idle. Half-page CTAs are slightly faster on B200 at the smallest sizes and
slower on ragged pages there and on H100, so no size regime gains enough for a second form.

TRAPS. Like the cache writer's slot trap, a call that would write or read
outside its tensors executes a device trap rather than corrupting memory or returning rows it
never wrote: a page id outside [0, NP) on a page the call reads, an output row outside [0, N)
of k_out or v_out, and a request whose length is negative or needs more pages than its entries
hold. A trap
leaves the CUDA context in an error state, so it is a bug report, not a refusal an engine can
recover from.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from ._append_core import (
    PAGE, _asm, compile_writer, envelope_refusal, f32_from_bits, ld_nc_b32x2, ld_nc_b32x4,
    scale_fake, select, trap)
from .quant_layout import quant_block_offsets

#: Channels of one output unit: eight bf16 values, one 16-byte store.
UNIT_CHANNELS = 8
#: The f32 2^23, whose bit pattern 0x4B000000 takes a code below 256 in its low byte as 2^23 + q.
MAGIC_BITS = 0x4B000000
MAGIC = 2.0 ** 23


@dsl_user_op
def ld_nc_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
                      "ld.global.nc.b32 $0, [$1];", "=r,l", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def prmt(a, b, selector, *, loc=None, ip=None):
    """Byte permute: byte i of the result is byte (selector >> 4i) & 7 of the pair {b:a}."""
    return Int32(_asm(T.i32(), [Int32(a).ir_value(loc=loc, ip=ip),
                                Int32(b).ir_value(loc=loc, ip=ip),
                                Int32(selector).ir_value(loc=loc, ip=ip)],
                      "prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", side_effects=False,
                      loc=loc, ip=ip))


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None):
    """a * b + c rounded once to f32, round to nearest even: the dequantize's one rounding
    before bf16, written as the instruction so no compiler choice moves it."""
    return Float32(_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip),
                                  Float32(b).ir_value(loc=loc, ip=ip),
                                  Float32(c).ir_value(loc=loc, ip=ip)],
                        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", side_effects=False,
                        loc=loc, ip=ip))


def field_lanes(bits, field):
    """(k, byte) such that field `field` of a code word (its bits [bits*field, bits*field + bits))
    is byte `byte` of `byte_lanes(word, bits, k)`: a byte holds 8 / bits fields, so lane word k
    keeps the k-th field of every byte."""
    per_byte = 8 // bits
    return field % per_byte, field // per_byte


def byte_lanes(word, bits, k):
    """Field k of every byte of `word`, one per byte: (word >> bits*k) & 0x0F0F0F0F at INT4,
    & 0x03030303 at INT2."""
    mask = 0x0F0F0F0F if bits == 4 else 0x03030303
    return (word >> (bits * k)) & Int32(mask)


def code_value(lanes, byte):
    """The f32 value of the code in byte `byte` of a `byte_lanes` word, exactly: the byte placed
    under 0x4B000000 is 2^23 + q, and subtracting 2^23 is exact."""
    return f32_from_bits(prmt(lanes, Int32(MAGIC_BITS), 0x7440 | byte)) - Float32(MAGIC)


class QuantDequantBase:
    """The shared parts of the kernel (module docstring). A subclass sets `ARCH` and implements
    `kernel` and `__call__`."""

    ARCH = None

    def __init__(self, layout, out_indptr_64):
        why = envelope_refusal(layout, self.ARCH)
        if why is not None:
            raise ValueError(f"{type(self).__name__}: {why}")
        D = layout.head_dim
        self.bits = layout.bits
        self.head_dim = D
        self.num_kv_heads = layout.num_kv_heads
        self.k_groups = D // int(layout.k_quant[1])
        self.k_half = self.k_groups > 1
        self.scale_bf16 = str(layout.scale_dtype) == "torch.bfloat16"
        off = quant_block_offsets(tuple(layout.k_quant), tuple(layout.v_quant), N=PAGE, D=D)
        self.legacy = bool(off["legacy"])
        self.block_tot = off["block_tot"]
        self.v_ring = off["v_ring"]
        #: The dtype of `out_indptr`, int64 when True: each is its own compile.
        self.out_indptr_64 = bool(out_indptr_64)

    # -----------------------------------------------------------------------------------------
    # The page's request
    # -----------------------------------------------------------------------------------------

    @cute.jit
    def _locate(self, mPageIndptr, mSeqLens, mOutIndptr, mKo, mVo, p, lane):
        """(active, n, row0) of entry p of page_indices: active is 1 when the entry is a page this
        call reads, n its tokens (1 to 64) and row0 the output row of its first token. Every warp
        runs the search itself, with the same loads and ballots, so the warps agree without
        exchanging anything. Warp-uniform.

        The search keeps [lo, hi] holding b, the last request with page_indptr[b] <= p, which is
        p's request because p < page_indptr[B]; a round samples it at 32 evenly spaced points,
        and since page_indptr is nondecreasing the samples at or below p are a prefix of the
        lanes, so the ballot's population count names the last of them. An input that is not
        nondecreasing still leaves lo inside [0, B), so it cannot send a load out of bounds."""
        num_req = Int32(mSeqLens.shape[0])
        active = Int32(0)
        n = Int32(0)
        row0 = Int64(0)
        if num_req > 0:
            if p >= mPageIndptr[0] and p < mPageIndptr[num_req]:
                lo = Int32(0)
                hi = num_req - Int32(1)
                while lo < hi:
                    step = (hi - lo + Int32(32)) // Int32(32)
                    i = lo + lane * step
                    take = Int32(0)
                    if i <= hi:
                        if mPageIndptr[i] <= p:
                            take = Int32(1)
                    ballot = cute.arch.vote_ballot_sync(take != 0)
                    lo = lo + (cute.arch.popc(ballot) - Int32(1)) * step
                    hi = cutlass.min(lo + step - Int32(1), hi)
                # In 64 bits: a slack entry's page ordinal times 64 passes 2^31 from ordinal
                # 2^25 on, which a request's page list may legally reach.
                first = (Int64(p) - Int64(mPageIndptr[lo])) * Int64(PAGE)
                length = Int64(mSeqLens[lo])
                if first < length:
                    active = Int32(1)
                    n = Int32(cutlass.min(Int64(PAGE), length - first))
                    row0 = Int64(mOutIndptr[lo]) + first
                    # TRAPS (module docstring): its rows lie outside k_out or v_out.
                    if row0 < Int64(0) or row0 + Int64(n) > Int64(mKo.shape[0]) \
                            or row0 + Int64(n) > Int64(mVo.shape[0]):
                        trap()
        return active, n, row0

    @cute.jit
    def _validate(self, mPageIndptr, mPageIndices, mSeqLens, cta, tidx,
                  threads: cutlass.Constexpr):
        """A checking CTA's job (GRID): trap when a request's length is negative or needs more
        than 64 tokens per entry it owns. Checking CTA c, the grid's CTA P * HKV + c, checks
        requests [c * threads, (c + 1) * threads), one per thread and once, those without
        entries included, so the launch's ceil(B / threads) checking CTAs check every request.
        The request index is 32-bit: c * threads is at most B - 1, and the index ends at most
        `threads` - 1 past the last request, which the host's request limit keeps in range
        (`kv_cache.MAX_DEQUANT_REQUESTS`). P * HKV is in range because the host bounds the
        whole grid (`kv_cache._dequant_size_refusal`)."""
        chunk = cta - Int32(mPageIndices.shape[0]) * Int32(self.num_kv_heads)
        b = chunk * Int32(threads) + tidx
        if b < Int32(mSeqLens.shape[0]):
            length = Int64(mSeqLens[b])
            span = Int64(mPageIndptr[b + 1]) - Int64(mPageIndptr[b])
            if length < Int64(0) or length > span * Int64(PAGE):
                trap()

    @cute.jit
    def _page_of(self, mPageIndices, num_pages, p):
        """(page, safe): entry p's page id, and that id where it is a page of the pool, else 0,
        which the request-independent loads use before the search says whether p is read."""
        page = mPageIndices[p]
        safe = select((page >= 0) & (page < num_pages), page, 0)
        return page, safe

    @cute.jit
    def _require_page(self, page, num_pages):
        """Trap on a page id outside the pool, for an entry this call reads (TRAPS)."""
        if page < 0 or page >= num_pages:
            trap()

    # -----------------------------------------------------------------------------------------
    # Scale records
    # -----------------------------------------------------------------------------------------

    def _block_bytes(self):
        return (4 * PAGE if self.legacy else self.block_tot) * (2 if self.scale_bf16 else 4)

    @cute.jit
    def _scale_words(self, mScale, page, head, token, tensor: cutlass.Constexpr, group):
        """Issue the load of one token's (s, z) of `tensor` ("K" or "V") from the scale pool
        and return its raw words; `_scale_pair` decodes them. `group` is K's
        channel half (0 or 1) for K token-wise over D/2, and is ignored otherwise."""
        block = mScale.iterator.toint() + (Int64(page) * Int64(self.num_kv_heads)
                                            + Int64(head)) * Int64(self._block_bytes())
        words = []
        if cutlass.const_expr(self.legacy):
            if cutlass.const_expr(self.scale_bf16):
                # (s_k, z_k, s_v, z_v) bf16: 8 bytes per token, K's pair first.
                words = [ld_nc_b32(block + Int64(8) * Int64(token)
                                   + Int64(0 if tensor == "K" else 4))]
            else:
                words = ld_nc_b32x2(block + Int64(16) * Int64(token)
                                    + Int64(0 if tensor == "K" else 8))
        else:
            # K token-wise over sub-head groups, whose pool is bf16 (`envelope_refusal`): K ring
            # has `k_groups` (s, z) pairs per token, then the V ring (s, z) per token.
            if cutlass.const_expr(tensor == "K"):
                words = [ld_nc_b32(block + Int64(4 * self.k_groups) * Int64(token)
                                   + Int64(4) * Int64(group))]
            else:
                words = [ld_nc_b32(block + Int64(2 * self.v_ring) + Int64(4) * Int64(token))]
        return words

    def _scale_pair(self, words):
        """(s, z) as f32 from `_scale_words`: f32 bit patterns, or a bf16 pair widened exactly."""
        if len(words) == 2:
            return f32_from_bits(words[0]), f32_from_bits(words[1])
        return f32_from_bits(words[0] << 16), f32_from_bits(words[0] & Int32(-65536))

    @cute.jit
    def _token_record(self, mScale, page, head, token, k_group):
        """Issue the loads of one token's K and V scale records for a unit of output channels
        and return the raw words; `_record_pairs` decodes them. In the legacy layout that is the
        token's whole record, 16 bytes in f32 or 8 in bf16, one load; for K token-wise over D/2
        it is the K half `k_group` of the unit's channels and the V pair, two loads."""
        words = []
        if cutlass.const_expr(self.legacy):
            block = mScale.iterator.toint() + (Int64(page) * Int64(self.num_kv_heads)
                                                + Int64(head)) * Int64(self._block_bytes())
            if cutlass.const_expr(self.scale_bf16):
                words = ld_nc_b32x2(block + Int64(8) * Int64(token))
            else:
                words = ld_nc_b32x4(block + Int64(16) * Int64(token))
        else:
            words = (self._scale_words(mScale, page, head, token, "K", k_group)
                     + self._scale_words(mScale, page, head, token, "V", 0))
        return words

    def _record_pairs(self, words):
        """((s_k, z_k), (s_v, z_v)) from `_token_record`'s words."""
        if self.legacy and not self.scale_bf16:
            return self._scale_pair(words[:2]), self._scale_pair(words[2:])
        return self._scale_pair(words[:1]), self._scale_pair(words[1:])


# ---------------------------------------------------------------------------------------------
# Compile and launch
# ---------------------------------------------------------------------------------------------

def index_fakes(out_indptr_64):
    """Fake page_indptr, page_indices, seq_lens and out_indptr, every length symbolic, so one
    compile serves every batch and page count; seq_lens and out_indptr share B."""
    batch = cute.sym_int()
    indptr = cute.runtime.make_fake_compact_tensor(Int32, (cute.sym_int(),), assumed_align=4)
    indices = cute.runtime.make_fake_compact_tensor(Int32, (cute.sym_int(),), assumed_align=4)
    lens = cute.runtime.make_fake_compact_tensor(Int32, (batch,), assumed_align=4)
    out = cute.runtime.make_fake_compact_tensor(Int64 if out_indptr_64 else Int32, (batch,),
                                                assumed_align=8 if out_indptr_64 else 4)
    return indptr, indices, lens, out


def out_fakes(kern):
    """Fake k_out and v_out: bf16 [N, HKV, D] with a symbolic row count and a free token stride
    of a multiple of 8 elements (16 bytes), each tensor its own."""
    return [cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (cute.sym_int(), kern.num_kv_heads, kern.head_dim),
        stride=(cute.sym_int(divisibility=8), kern.head_dim, 1), assumed_align=16)
        for _ in range(2)]


def compile_dequant(kern, module_file, layout, pools):
    """Compile `kern` through the persistent kernel cache, under the cache writer's compile
    rules (`_append_core.compile_writer`), keyed on the layout and the constructed kernel's
    attributes, the out_indptr dtype among them. `pools` is (k_codes, v_codes) fakes with a
    symbolic page count shared with the scale pool's; the trailing Int32 is the CTA count, the
    checking CTAs included."""
    k_pool, v_pool, num_pages = pools
    return compile_writer(kern, module_file, layout,
                          (k_pool, v_pool, scale_fake(kern, num_pages), *out_fakes(kern),
                           *index_fakes(kern.out_indptr_64)))


def launch_dequant(compiled, *, threads, k_codes, v_codes, scales, k_out, v_out, page_indptr,
                   page_indices, seq_lens, out_indptr):
    """Launch a compiled dequantize on torch's current stream: one CTA per (entry of
    page_indices, KV head), then one checking CTA per `threads` requests, the kernel's thread
    count (module docstring, GRID). A call with no requests launches nothing; one with requests
    and no entries launches the checks alone."""
    import tvm_ffi

    batch = seq_lens.shape[0]
    if batch == 0:
        return
    ctas = page_indices.shape[0] * k_codes.shape[1] + (batch + threads - 1) // threads
    with tvm_ffi.use_torch_stream():
        compiled(k_codes, v_codes, scales, k_out, v_out, page_indptr, page_indices, seq_lens,
                 out_indptr, Int32(ctas))
