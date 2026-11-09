# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""The cache write both serving writers share: page runs, quantization and scale records.

`_sm90/quant_append.QuantAppendSm90` and `_sm100/quant_append.QuantAppendSm100` quantize new K
and V tokens and write them into the pools their GPU's decode reads. Everything that does not
depend on the code-pool layout lives here: finding the page runs, quantizing each token into
shared memory, writing its scale record, clearing the stale scale records of a page a run starts
at position 0, and the Blackwell page maximum `v_page_max` with its count table `v_max_hist`.
Each subclass supplies `_store_run`, which builds the words of its own code pools from shared
memory and stores them.

INPUTS. `k` and `v` are bf16 [T, HKV, D] in global memory, D contiguous, head stride D and a
free token stride, so a view of a fused QKV projection works without a copy; rows are 16-byte
aligned. `slot_mapping` is int64 [T]: token i goes to page slot // 64, in-page position
slot % 64. A negative slot is skipped. A slot at or past NP * 64 traps, because a write past
the K pool would land in the next region of an engine-bound buffer.

PAGE RUNS. A run is a maximal sequence of consecutive tokens whose slots are non-negative,
rise by exactly 1 from one token to the next and stay in one page. Token i starts a run when its
slot is non-negative and it is the first token, follows a negative slot, sits at in-page
position 0, or does not continue its predecessor's slot. Runs are a property of the slot map,
so every CTA finds the same runs, and no token belongs to two.

GRID. The writer launches G x HKV CTAs of 128 threads, with G = min(T, the resident-CTA budget /
HKV) and the KV head as the fastest grid dimension, so that the CTAs of one token range land on
different SMs.
CTA (g, h) owns the tokens [g T / G, (g + 1) T / G) and writes, for KV head h, every run whose
first token it owns, reading past the end of its range when a run continues there. Each warp
finds the runs itself, with the same loads and ballots, so the warps agree on every run
without exchanging it.

ONE RUN, in three steps.
  1. Quantize (global memory to registers to shared memory). An aligned group of lanes takes
     one token: D / 16 lanes for a run of at least one pass of the CTA (so a warp quantizes
     several tokens at once), a whole warp for a shorter run such as a decode token. Each lane
     loads its contiguous slice of the token's K and V rows, and the group reduces the exact
     minimum and maximum, over each half of the group separately for K token-wise over D/2.
     The writer's input envelope (`INPUT_ENVELOPE`) saturates every value to [-2^64, 2^64];
     saturation is monotone, so it is applied to the minimum and maximum, and to the values
     themselves only when one lies outside. In f32, z = mn, s = div.rn(max(mx - z, 1e-8), qmax) and
     q = clamp(cvt.rni(div.rn(x - z, s)), 0, qmax): no step is a product, so nothing can be
     contracted into an fma (`_store_quantized` says why a dividend floor changes no q). Each
     quantized value goes to shared memory as one byte, at [in-page position][channel] of
     `sQk` / `sQv` (rows padded by 16 bytes, so the reads of step 3 spread across banks). The
     group's first lane stores the token's scale records, rounded to the pool's dtype; the
     codes always use the f32 scale.
  2. A CTA barrier, then on Blackwell the tail clear (below) when the run starts its page at
     position 0 and ends before the page does, and thread 0 issues the run's atomic on
     `v_page_max` unless the update would change nothing (below). Hopper clears those tails in
     the page-start pass, after all of the CTA's runs.
  3. `_store_run` (the subclass): every thread builds words of its destination rows from
     shared memory and stores them. A word all of whose fields belong to this run gets a plain
     store, 16 bytes at a time where a whole chunk is covered; any other word the run touches
     gets red.and of the cleared fields and then red.or of their new values, so the fields of
     every other token keep whatever they hold. Blackwell stores K and V in token rows, so
     every word it writes is a plain store; Hopper's words mix tokens and take both forms. On
     Blackwell thread 0 moves the count of the
     run's atomic in `v_max_hist` before or after this step (below). A second barrier protects
     shared memory before the next run.

WHY THIS CANNOT RACE for a slot map without duplicate slots: a plain store happens only when
the run holds every slot of the word, and slots are distinct, so no other run touches that
word; every other write changes only this run's fields, field-wise operations on disjoint
fields commute, and the and and the or come from one thread to one address in program order.
Duplicate slots in one launch void that argument for their page: on Hopper, where several
tokens share a code word, a duplicate's field-wise and/or can race another run's plain store of
the whole word, so the whole page's contents are unspecified; on Blackwell, where no two runs
share a word, only the duplicated slot's contents are, and the page maximum can miss the scale
that slot ends up holding (THE SKIP TEST AND THE SETTLE). Every value written is still a valid
quantized value or scale. Engines route padding writes to a page no request uses
(docs/integration.md).

THE TAIL CLEAR. A run that starts its page at in-page position 0 and holds fewer than 64 tokens
begins a new use of that page (every use of a page begins with a write at position 0, which
SGLang's paged allocation guarantees), so the page's other positions hold an
earlier use's scale records or never-written bytes. Before the decodes multiply a position past
the sequence length by a weight of zero, its scale record must be one that cannot turn that zero
into a NaN, so the run stores (s, z) = (0, 0) at every position of the page outside the run that
no other token of this launch writes. The run finds those tokens by scanning the slot map, each
warp collecting the in-page positions it meets as two 32-bit masks in `sHits`, so the cleared
positions are disjoint from every token's store in the launch, duplicates included. Only another
position-0 run of the same page (a duplicate of position 0) clears the same positions, and the
two find each other in the scan, clear the same set and store the same zero bytes. Code
bytes are not cleared: with s = z = 0 every product with a stale code is 0, and every code byte
decodes to a finite operand on both GPUs. A run that starts later in the page continues a use
and clears nothing; a run that fills the page leaves nothing to clear. Because the cleared
positions are disjoint from every store of the launch, the clear commutes with them, so Hopper
does it in a second pass over the CTA's runs, after the run loop and only when some run started a
use, which keeps the scan and the clear out of the run loop's code; Blackwell scans in the run,
because the scan also feeds the page maximum's exchange (below), and clears there too.

V_PAGE_MAX AND V_MAX_HIST (Blackwell, token-wise V). `v_page_max` is uint32 [NP,
HKV] in global memory, the bit pattern of the largest stored s_v of each (page, KV head) in the
page's current use (after bf16 rounding when the pool is bf16, the value the decode multiplies), 0
when the cell holds none. `v_max_hist` is uint32 [256] in global memory: bin b counts the cells
whose `fold_runtime.vmax_bucket` is b among the cells that hold a maximum, and the decode takes its
P-scale exponent from the highest nonzero bin. Thread 0 takes `best`, the largest stored s_v of the
run's tokens and, for a position-0 run shorter than the page, of every other token of that page in
this launch (the same scan). A position-0 run's update is `old = atom.exch(cell, best)` and any
other run's `old = atom.max.u32(cell, best)`, so the cell's new value is best or max(old, best),
and the update moves one count from old's bin to the new value's bin when the two differ
(`_settle_v_max`). Every write of a cell is one of these atomics, which `bind` starts from zeroed
cells and an empty table, and the atomics on one cell are totally ordered, each returning the value
its predecessor left; so the moves of one cell telescope from its first value to its last whatever
the order of the issuers and of their adds, and the table equals the histogram of the cells
whenever no writer is in flight. Counts inside a launch may be transiently off; the decode reads
the table only after `griddepcontrol.wait` or stream order. In a launch without duplicate slots, a
position-0 run either fills its page or scans the launch, so the exchange's best includes every
token of the page in this launch, and exchange and max give the same final cell in either order.

THE SKIP TEST AND THE SETTLE (Blackwell). A run that continues its page reads the cell while the
run is quantized, and thread 0 skips its update when best <= that value as unsigned words. The read
is a relaxed gpu-scope load (`ld_relaxed_b32`), which is single-copy atomic, so it returns a value
the cell held and never a mix of two of them; a cp.async copy is a weak read with no such
guarantee. For INT2 with K token-wise over the whole head, thread 0 loads the cell into a
register and holds it while the run is quantized; for every
other layout, lane 0 of the last warp loads it and stores it to shared memory (`sSeen`) before it
quantizes anything, so no register holds it across the quantization, and the barrier before 2a
publishes it to thread 0. A short run of fewer tokens than warps (a one-token decode step) leaves
the last warp without a token, so its wait for the load delays no quantization. The choice is `seen_in_register`, made per layout from measurement
(below). Either way the value read is one the cell held earlier in this launch (the launch starts after every earlier write, and the
cell changes only by this launch's atomics): with no exchange since, the cell only grew, so it is
at least best and the max would change nothing; with an exchange since, that exchange's best
includes this run's tokens, and the cell is at least that. A skipped update writes nothing, so the
table stays the histogram of the cells. Every other update issues its atomic after the run's
quantization and settles it within the run, before the run's code words are stored when the cell
is read into a register and after them otherwise, so the atomic returns while the run works and
the run loop carries no returned value into the next run. A loop-carried value would spill to
local memory as soon as it arrived at the writer's 64-register budget, which waits for the atomic
inside every run and makes the decode-step write 6% to 36% slower on B200. The two read
forms keep the same guarantee. On B200 the register form makes the mid-page decode-step
write 5% to 11% slower than the shared-memory form in the other three
layouts, and the shared-memory form spills in INT2 with K over the whole head and is 3% to 11%
slower there. The relaxed load in the shared-memory form makes the write alone
0.973 of the cp.async form's time (geometric mean over the 68 cases of the Blackwell append
sweep, 0.852 to 1.008). With duplicate slots in
one launch (a run that fills the page from position 0 does not scan, and a duplicate's update may
land before its exchange or be skipped against a value read before it), the cell can miss the s_v a
duplicated slot ends up holding, while the table stays the histogram of the cells; that slot's
contents are unspecified anyway, and the decode converts its fp8 products with
satfinite, so the slot's term saturates and stays finite while every other position stays covered.

SCALES are value-exact with the oracle; their bytes are too, except that a zero point of zero
may carry the other sign than torch's amin gives when a row mixes +0.0 and -0.0.

THE PREQUANTIZED WRITE (`QuantKVCache.append_quantized`). An engine that quantizes its tokens
itself hands the writer uint8 codes [T, HKV, D], one value per byte, and f32 scales and zero
points [T, HKV] ([T, HKV, 2] for K token-wise over D/2, group 0 the channels [0, D/2)), and the
writer only lays them out. It is a writer built with `prequantized=True`, a kernel of its own
(`kernel_prequantized`) that is compiled and cached apart from the quantizing one, and both
kernels run the one run loop `_write`, whose every difference is a trace-time branch: step 1
and the page scan's s_v. Its step 1 (`_stage_given`) gives each token D / 16 lanes on every run,
long or short, as the quantizing writer does on a long run: each lane reads its 16 bytes of the
token's K codes and of its V codes with one 16-byte load each, masks every byte to the stored
width, and stores the four words where the quantizing writer stores its bytes, so steps 2 and 3
are unchanged. The lanes read the token's scales and zero points and saturate them, and the
group's first lane stores the scale record, rounded to the pool's dtype; the stored s_v enters
the page maximum as before. The page scan takes a token's stored s_v from its given V scale,
saturated and rounded, where the quantizing writer re-quantizes the row (`_stored_sv_bits`).
The prequantized write divides nothing.

ITS SATURATION (`_saturated`). s goes to [div.rn(1e-8, qmax), div.rn(2^65, qmax)] and z to
[-2^64, 2^64], with fmax then fmin, which take a NaN to the lower bound. Those are the ranges of
the quantizing writer's s and z under its input envelope, and they are the magnitudes the
finiteness argument rests on: the decodes' z / s' and the Blackwell INT4
x512 stay finite, and every stored scale is positive, so a page maximum stays sign-free. The two
bounds are independent, so a stored record may pair a z and an s that no single row of the
quantizing writer's input gives together (z = 2^64 with s = 2^65 / qmax decodes q = qmax near
3 * 2^64); they give arithmetic headroom and do not make every record one `append` could store.
The engine is expected to pass the records its own quantizer made from finite inputs.
"""
import functools
import struct

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from ._sm100.fold_runtime import vmax_bucket
from .quant_layout import quant_block_offsets

#: Tokens per page, the page size both decodes read: one Hopper decode tile; a Blackwell
#: sequence tile spans several pages.
PAGE = 64
#: Threads per CTA: one warpgroup.
THREADS = 128
WARPS = THREADS // 32
#: Channels per lane when the lanes of a long run's token quantize it together: D / 16 lanes
#: per token (8 at D = 128), so a warp quantizes several tokens at once. A run shorter than one
#: such pass of the CTA takes a whole warp per token instead.
GROUP_CHANNELS = 16
#: Passes whose K and V rows a warp loads before quantizing any of them, by lanes per token:
#: two for a warp per token (D / 32 values a lane), one for a group (16 values a lane), which
#: keeps the registers within eight CTAs per SM.
PASSES_IN_FLIGHT = {4: 1, 8: 1, 16: 1, 32: 2}
#: The clamp of a token's range before it is divided by qmax.
S_CLAMP = 1.0e-8
#: The writer's input envelope: every K and V element is saturated to
#: [-2^64, 2^64] before it is quantized, so mx - mn, z / s and the INT4 x512 of the Blackwell
#: decode stay finite in f32 for any finite bf16 input.
INPUT_ENVELOPE = 2.0 ** 64
#: The smallest dividend the quantization divides (`_store_quantized`): 2^-60 over the
#: smallest scale, 1e-8 / 15, is far below 0.5, and over the largest, 2^65 / 3, still a normal
#: f32, so div.rn keeps its fast path.
DIVIDEND_FLOOR = 2.0 ** -60
#: The bits of |x| the clip's order-statistic search decides, highest first. The writer's inputs
#: are bf16, so as f32 every |x| holds its value in bits 30..16 and zeros below, and an f32 bit
#: pattern of a non-negative value orders like the value; deciding bits 30..16 therefore finds the
#: order statistic exactly.
CLIP_BITS = tuple(range(30, 15, -1))
#: Bytes added to each row of `sQk` / `sQv`, so the token rows of one warp's step-3 reads start
#: in different banks while every row stays 16-byte aligned.
ROW_PAD = 16
#: Slot entries each warp tests per round of the page scan (`_page_scan`), four rounds of 32 per
#: lane in flight at once.
SCAN_ROUNDS = 4
#: Runs that start a new use of their page a Hopper CTA records in shared memory (`sStarts`) for
#: its page-start pass; a CTA with more finds its runs again from the slot map instead.
PAGE_START_LIST = 16
#: The prequantized write (`_stage_given`): the bits of each 32-bit word of codes a lane keeps,
#: the low `bits` of each of its four bytes, by stored width.
CODE_MASK = {2: 0x03030303, 4: 0x0F0F0F0F}
#: Passes whose codes and scales a warp of the prequantized write loads before it stages any of
#: them, so that the loads of a whole-page run overlap.
GIVEN_PASSES_IN_FLIGHT = 2


def _f32(x):
    """x rounded to the nearest f32, as a Python float."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def scale_range(qmax):
    """(lowest, highest) f32 scale the quantizing writer stores at `qmax`: div.rn(1e-8, qmax) and
    div.rn(2^65, qmax), the range clamped to S_CLAMP and the widest range the input envelope
    allows, each divided with round to nearest. The prequantized write saturates a given scale to
    it (`_saturated`). A quotient of two f32 values formed in double and rounded once more to f32
    is the correctly rounded f32 quotient, because 53 >= 2 * 24 + 2 bits."""
    return _f32(_f32(S_CLAMP) / qmax), _f32(2.0 * INPUT_ENVELOPE / qmax)


def _asm(ret, args, text, constraints, *, side_effects, loc, ip):
    return llvm.inline_asm(ret, args, text, constraints, has_side_effects=side_effects,
                           is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
                           loc=loc, ip=ip)


@dsl_user_op
def div_rn(a, b, *, loc=None, ip=None):
    """IEEE round-to-nearest f32 division, which the CPU oracle's torch division is."""
    return Float32(_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip),
                                  Float32(b).ir_value(loc=loc, ip=ip)],
                        "div.rn.f32 $0, $1, $2;", "=f,f,f", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def cvt_rni_s32(x, *, loc=None, ip=None):
    """f32 to s32, rounding half to even as torch.round does."""
    return Int32(_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)],
                      "cvt.rni.s32.f32 $0, $1;", "=r,f", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def f32_from_bits(x, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Int32(x).ir_value(loc=loc, ip=ip)],
                        "mov.b32 $0, $1;", "=f,r", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def bits_from_f32(x, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)],
                      "mov.b32 $0, $1;", "=r,f", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def bf16x2(lo, hi, *, loc=None, ip=None):
    """Two f32 rounded to bf16 (round to nearest even, as torch's .to(bfloat16)) in one word,
    `lo` in the low half, which is the lower address of a little-endian bf16 pair."""
    return Int32(_asm(T.i32(), [Float32(hi).ir_value(loc=loc, ip=ip),
                                Float32(lo).ir_value(loc=loc, ip=ip)],
                      "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", side_effects=False,
                      loc=loc, ip=ip))


@dsl_user_op
def ld_nc_b32(addr, *, loc=None, ip=None):
    """One word of read-only global memory."""
    return Int32(_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
                      "ld.global.nc.b32 $0, [$1];", "=r,l", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def ld_nc_b32x2(addr, *, loc=None, ip=None):
    st = _asm(llvm.StructType.get_literal([T.i32()] * 2), [Int64(addr).ir_value(loc=loc, ip=ip)],
              "ld.global.nc.v2.b32 {$0, $1}, [$2];", "=r,=r,l", side_effects=False,
              loc=loc, ip=ip)
    return [Int32(llvm.extractvalue(T.i32(), st, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def ld_nc_b32x4(addr, *, loc=None, ip=None):
    st = _asm(llvm.StructType.get_literal([T.i32()] * 4), [Int64(addr).ir_value(loc=loc, ip=ip)],
              "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l",
              side_effects=False, loc=loc, ip=ip)
    return [Int32(llvm.extractvalue(T.i32(), st, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_relaxed_b32(addr, *, loc=None, ip=None):
    """A relaxed gpu-scope load of one global word, which reads it where the atomics on it
    resolve (L2)."""
    return Int32(_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
                      "ld.relaxed.gpu.global.b32 $0, [$1];", "=r,l", side_effects=True,
                      loc=loc, ip=ip))


@dsl_user_op
def st_b32(addr, words, *, loc=None, ip=None):
    """A plain global store of 1, 2 or 4 consecutive words at a correspondingly aligned address."""
    n = len(words)
    regs = ", ".join(f"${i + 1}" for i in range(n))
    text = ("st.global.b32 [$0], $1;" if n == 1
            else f"st.global.v{n}.b32 [$0], {{{regs}}};")
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)]
         + [Int32(w).ir_value(loc=loc, ip=ip) for w in words],
         text, ",".join(["l"] + ["r"] * n), side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def red_b32(op, addr, value, *, loc=None, ip=None):
    """A global reduction with no return value: op is "and.b32", "or.b32" or "add.u32"."""
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
         f"red.global.{op} [$0], $1;", "l,r", side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def atom_b32(op, addr, value, *, loc=None, ip=None):
    """A global atomic that returns the word's previous value: op is "exch.b32" or "max.u32".
    Relaxed at gpu scope, PTX's default for atom."""
    return Int32(_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip),
                                Int32(value).ir_value(loc=loc, ip=ip)],
                      f"atom.global.{op} $0, [$1], $2;", "=r,l,r", side_effects=True,
                      loc=loc, ip=ip))


@dsl_user_op
def trap(*, loc=None, ip=None):
    _asm(None, [], "trap;", "", side_effects=True, loc=loc, ip=ip)


def s32(x):
    """A 32-bit pattern given as a Python int, as the signed value an Int32 holds."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >> 31 else x


def ctz(x):
    """Index of the lowest set bit of a nonzero Int32."""
    return cute.arch.popc((x & (Int32(0) - x)) - Int32(1))


def select(cond, a, b):
    return Int32(cutlass.select_(cond, Int32(a), Int32(b)))


def in_run(position, t0, n):
    """Whether in-page position `position` lies in [t0, t0 + n): one unsigned compare."""
    return cutlass.Uint32(position - t0) < cutlass.Uint32(n)


def check_clip_indices(clip, head_dim, who="the writer"):
    """`clip` as a tuple of two (K, V) clip indices (`kv_cache.clip_index`), each an int in
    [-1, head_dim); anything else, a float or a bool included, raises. The compile entry points
    call it before their memoized compile, whose key would otherwise equate 122.0 with 122."""
    clip = tuple(clip)
    if len(clip) != 2 or any(isinstance(c, bool) or not isinstance(c, int)
                             or not -1 <= c < head_dim for c in clip):
        raise ValueError(f"{who}: clip indices are two ints in [-1, {head_dim}); got {clip!r}")
    return clip


def envelope_refusal(layout, arch):
    """Why the serving writer of `arch` refuses `layout`, or None.

    Plain Python, so the constructors raise ValueError with it under `python -O` too. The orders
    are the reader's own answers, carried by the layout. On Blackwell the writer implements the
    production orders only, so any other answer is a configuration with no serving writer, whose
    pools keep coming from the host packers. On Hopper it places every word through
    `serving_cache.code_word_offset`, which takes the order, so it writes whichever order the
    pool geometry defines, and the geometry refuses any other (`QuantAppendSm90`)."""
    if layout.arch != arch:
        return f"this writer serves {arch}; the layout is for {layout.arch}"
    D = layout.head_dim
    if arch == "sm100" and not (layout.k_pool_permuted is True and layout.v_pool_permuted is True):
        return (f"no serving writer for this configuration: the reader's pool layout is "
                f"k_pool_permuted={layout.k_pool_permuted}, "
                f"v_pool_permuted={layout.v_pool_permuted}, and the writer implements the "
                f"production layout only (K permuted, V stored exactly like K)")
    if layout.bits not in (2, 4):
        return f"the writer stores 2 or 4 bits; got {layout.bits}"
    if layout.num_kv_heads < 1:
        return f"num_kv_heads must be at least 1; got {layout.num_kv_heads}"
    k_quant, v_quant = tuple(layout.k_quant), tuple(layout.v_quant)
    if k_quant[0] == "channel" or v_quant[0] == "channel":
        return "a channel-wise K or V is not supported by the per-token append writer"
    heads = (64, 128, 256) if arch == "sm90" else (64, 128)
    if D not in heads:
        return f"the {arch} writer serves head_dim in {heads}; got {D}"
    full, half = ("token", D), ("token", D // 2)
    if v_quant != full or k_quant not in (full, half):
        return (f"the writer serves V token-wise over {D} channels and K token-wise over {D} or "
                f"{D // 2}; got k_quant={k_quant}, v_quant={v_quant}")
    if D != 128 and k_quant != full:
        return f"at head_dim={D} the reader decodes token-wise K over all {D} channels only"
    name = {"torch.float32": "f32", "torch.bfloat16": "bf16"}.get(str(layout.scale_dtype))
    if k_quant == half:
        allowed = ("bf16",)
    elif D == 128:
        allowed = ("f32", "bf16")
    else:
        allowed = ("f32",)
    if name not in allowed:
        return (f"the scale pool of this pair is {' or '.join(allowed)}; "
                f"got {layout.scale_dtype}")
    return None


class QuantAppendBase:
    """The shared kernel body (module docstring). A subclass sets `ARCH` and implements
    `_store_run`; `WRITES_V_PAGE_MAX` turns on the Blackwell page maximum. `prequantized`
    builds the writer of `QuantKVCache.append_quantized` (THE PREQUANTIZED WRITE), which launches
    `kernel_prequantized`; otherwise it is the quantizing writer of `append`, which launches
    `kernel`."""

    ARCH = None
    WRITES_V_PAGE_MAX = False
    #: (K, V) clip indices (`kv_cache.clip_index`), -1 for no clip. A class attribute that the
    #: constructor overrides only for a clipped writer, so an unclipped writer's attributes, and
    #: with them its compile key, carry no clip.
    clip_indices = (-1, -1)

    def __init__(self, layout, prequantized=False, clip=(-1, -1)):
        why = envelope_refusal(layout, self.ARCH)
        if why is not None:
            raise ValueError(f"{type(self).__name__}: {why}")
        self.prequantized = bool(prequantized)
        D = layout.head_dim
        clip = check_clip_indices(clip, D, type(self).__name__)
        if clip != (-1, -1):
            if self.prequantized:
                raise ValueError(f"{type(self).__name__}: the prequantized write lays out "
                                 f"codes the engine quantized, so it takes no clip")
            self.clip_indices = clip
        self.bits = layout.bits
        self.qmax = (1 << layout.bits) - 1
        self.head_dim = D
        self.num_kv_heads = layout.num_kv_heads
        self.k_half = tuple(layout.k_quant) == ("token", D // 2)
        # Blackwell, where a run that continues its page reads its page maximum's cell for 2a's
        # test (THE SKIP TEST AND THE SETTLE), with a relaxed gpu-scope load either way: by
        # thread 0 into a register when True, else by lane 0 of the last warp into shared memory,
        # the form measured faster for the layout on B200.
        self.seen_in_register = layout.bits == 2 and not self.k_half
        self.scale_bf16 = str(layout.scale_dtype) == "torch.bfloat16"
        off = quant_block_offsets(tuple(layout.k_quant), tuple(layout.v_quant), N=PAGE, D=D)
        self.legacy = bool(off["legacy"])
        self.block_tot = off["block_tot"]
        self.v_ring = off["v_ring"]
        self.smem_row = D + ROW_PAD

    # -----------------------------------------------------------------------------------------
    # Inputs, scales and the run scan
    # -----------------------------------------------------------------------------------------

    @cute.jit
    def _load_row(self, mX, token, head, part, parts: cutlass.Constexpr):
        """Channels [part D / parts, (part + 1) D / parts) of row (token, head) of K or V, as
        f32 in channel order. bf16 to f32 is exact; the input envelope is applied by
        `_scale_zero` and `_quantize_to_smem`."""
        D = self.head_dim
        count = D // parts
        elem = (Int64(token) * Int64(mX.layout.stride[0]) + Int64(head) * Int64(D)
                + Int64(part * count))
        addr = mX.iterator.toint() + elem * Int64(2)
        words = []
        if cutlass.const_expr(count == 2):
            words = [ld_nc_b32(addr)]
        elif cutlass.const_expr(count == 4):
            words = ld_nc_b32x2(addr)
        else:
            for piece in cutlass.range_constexpr(count // 8):
                words += ld_nc_b32x4(addr + Int64(16 * piece))
        out = []
        for i in cutlass.range_constexpr(2 * len(words)):
            # A word holds channels 2m (low half) and 2m + 1 (high half).
            w = words[i // 2]
            out.append(f32_from_bits(w << 16 if i % 2 == 0 else w & Int32(-65536)))
        return out

    @cute.jit
    def _clip(self, xs, lanes: cutlass.Constexpr, index: cutlass.Constexpr):
        """The values the `lanes` lanes of an aligned lane group hold together (one token's row),
        each clamped to [-t, t], where t is the element at `index` of the row's |x| sorted
        ascending: OSCAR's per-row clip (`kv_cache.clip_index`).

        t is found by a radix select on the bit patterns of |x| (CLIP_BITS): for each bit, highest
        first, the group counts the values that match the bits decided so far and have this bit
        clear, with a lane-local count and a butterfly sum; if fewer than the remaining rank have
        it clear, the bit is set and the rank drops by that count. The bits decided are the bits
        of t, ties included. The clamp is min(max(x, -t), t), as OSCAR's and the CPU oracle's."""
        mags = []
        for i in cutlass.range_constexpr(len(xs)):
            mags.append(bits_from_f32(xs[i]) & Int32(0x7FFFFFFF))
        prefix = Int32(0)
        rank = Int32(index)
        for b in cutlass.range_constexpr(len(CLIP_BITS)):
            bit = CLIP_BITS[b]
            decided = Int32(s32((1 << 31) - (1 << bit)))
            count = Int32(0)
            for i in cutlass.range_constexpr(len(mags)):
                count = count + select((mags[i] & decided) == prefix, 1, 0)
            for step in cutlass.range_constexpr(lanes.bit_length() - 1):
                count = count + cute.arch.shuffle_sync_bfly(count, offset=1 << step)
            above = rank >= count
            rank = select(above, rank - count, rank)
            prefix = select(above, prefix | Int32(1 << bit), prefix)
        t = f32_from_bits(prefix)
        clipped = []
        for i in cutlass.range_constexpr(len(xs)):
            clipped.append(-cute.arch.fmax(-cute.arch.fmax(xs[i], -t), -t))
        return clipped

    @cute.jit
    def _scale_zero(self, xs, lanes: cutlass.Constexpr):
        """(s, z, inside) of the values the `lanes` lanes of an aligned lane group hold together.

        The exact minimum and maximum come from a lane-local pass and a butterfly over the
        group. Saturating to the input envelope is monotone, so the envelope is applied to them
        rather than to every value: z = clamp(mn) and s = div.rn(max(clamp(mx) - z, 1e-8),
        qmax), which is what the saturated values give. `inside` says no value of the group
        needs saturating, so `_quantize_to_smem` can skip it."""
        mx = xs[0]
        mn = xs[0]
        for i in cutlass.range_constexpr(1, len(xs)):
            mx = cute.arch.fmax(mx, xs[i])
            mn = -cute.arch.fmax(-mn, -xs[i])
        for step in cutlass.range_constexpr(lanes.bit_length() - 1):
            off = 1 << step
            mx = cute.arch.fmax(mx, cute.arch.shuffle_sync_bfly(mx, offset=off))
            mn = -cute.arch.fmax(-mn, -cute.arch.shuffle_sync_bfly(mn, offset=off))
        bound = Float32(INPUT_ENVELOPE)
        inside = (mx <= bound) & (mn >= -bound)
        z = -cute.arch.fmax(-cute.arch.fmax(mn, -bound), -bound)
        top = -cute.arch.fmax(-cute.arch.fmax(mx, -bound), -bound)
        s = div_rn(cute.arch.fmax(top - z, Float32(S_CLAMP)), Float32(self.qmax))
        return s, z, inside

    @cute.jit
    def _quantize_to_smem(self, xs, s, z, inside, sQ32, t, first_word):
        """Quantize this lane's values, saturating them first unless `inside`, and store them as
        bytes of row `t` of a staging buffer, from word `first_word` of the row."""
        if inside:
            self._store_quantized(xs, s, z, sQ32, t, first_word)
        else:
            bound = Float32(INPUT_ENVELOPE)
            clamped = []
            for i in cutlass.range_constexpr(len(xs)):
                clamped.append(-cute.arch.fmax(-cute.arch.fmax(xs[i], -bound), -bound))
            self._store_quantized(clamped, s, z, sQ32, t, first_word)

    @cute.jit
    def _store_quantized(self, xs, s, z, sQ32, t, first_word):
        """q = clamp(cvt.rni(div.rn(x - z, s)), 0, qmax) of each value, four to a word.

        The dividend is raised to at least DIVIDEND_FLOOR first. That changes no q: a dividend
        below the floor gives a quotient below 0.5 (s is at least 1e-8 / qmax), so q is 0 either
        way. What it changes is the division's path: div.rn takes a slow subroutine for a zero
        or denormal dividend, and every token's minimum has a zero one. The floor also makes the
        quotient positive, a NaN input included (fmax returns the floor), so the clamp at 0 is
        implied and only the clamp at qmax is executed."""
        floor = Float32(DIVIDEND_FLOOR)
        for k in cutlass.range_constexpr(len(xs) // 4):
            word = Int32(0)
            for i in cutlass.range_constexpr(4):
                q = cvt_rni_s32(div_rn(cute.arch.fmax(xs[4 * k + i] - z, floor), s))
                word = word | (cutlass.min(q, Int32(self.qmax)) << (8 * i))
            sQ32[(t, first_word + k)] = word

    @cute.jit
    def _stored_bits(self, s, z):
        """The bit pattern of a scale as the pool stores it: f32, or rounded to bf16."""
        bits = bits_from_f32(s)
        if cutlass.const_expr(self.scale_bf16):
            bits = bf16x2(s, z) << 16
        return bits

    @cute.jit
    def _store_scales(self, mScale, page, head, t, sk, zk, sk_hi, zk_hi, sv, zv):
        """Store one token's scale records, rounded to the pool dtype. Called
        by one lane."""
        bf16 = self.scale_bf16
        esize = 2 if bf16 else 4
        base = mScale.iterator.toint()
        block = Int64(page) * Int64(self.num_kv_heads) + Int64(head)
        if cutlass.const_expr(self.legacy):
            addr = base + (block * Int64(PAGE) + Int64(t)) * Int64(4 * esize)
            if cutlass.const_expr(bf16):
                st_b32(addr, [bf16x2(sk, zk), bf16x2(sv, zv)])
            else:
                st_b32(addr, [bits_from_f32(sk), bits_from_f32(zk),
                              bits_from_f32(sv), bits_from_f32(zv)])
        else:
            # K token-wise over D/2, whose pool is bf16 (`envelope_refusal`): K ring (s_lo, z_lo,
            # s_hi, z_hi) at 4t, V ring (s, z) at v_ring + 2t, element offsets inside the
            # (page, head) block.
            block_addr = base + block * Int64(self.block_tot * esize)
            st_b32(block_addr + Int64(4 * esize) * Int64(t),
                   [bf16x2(sk, zk), bf16x2(sk_hi, zk_hi)])
            st_b32(block_addr + Int64(self.v_ring * esize) + Int64(2 * esize) * Int64(t),
                   [bf16x2(sv, zv)])

    @cute.jit
    def _quantize_tokens(self, mK, mV, mScale, sQk32, sQv32, p, n, t0, page, head, warp, lane,
                         lanes: cutlass.Constexpr):
        """Step 1, by every warp: quantize the run's tokens into the staging buffers and store
        their scale records. Each token takes an aligned group of `lanes` lanes, D / lanes
        channels each, so a warp quantizes 32 / lanes tokens at once and the CTA covers
        WARPS * 32 / lanes consecutive tokens per pass; the rows of PASSES_IN_FLIGHT[lanes]
        passes are loaded before any is quantized, so their loads overlap. A group past the
        run's end loads the run's last row again, joins the butterflies, and stores nothing.
        Returns the largest stored s_v of the warp's tokens, as a bit pattern (every lane)."""
        D = self.head_dim
        per_pass = WARPS * (32 // lanes)
        group = lane // lanes
        sub = lane % lanes
        first_word = sub * (D // lanes // 4)
        # The lane of this group holding K's high half, for K token-wise over D/2.
        hi_lane = lane - sub + lanes // 2
        vmax = Int32(0)
        k_base = warp * (32 // lanes)
        passes = PASSES_IN_FLIGHT[lanes]
        while k_base < n:
            rows = []
            for u in cutlass.range_constexpr(passes):
                token = p + cutlass.min(k_base + per_pass * u + group, n - 1)
                rows.append((self._load_row(mK, token, head, sub, lanes),
                             self._load_row(mV, token, head, sub, lanes)))
            for u in cutlass.range_constexpr(passes):
                if k_base + per_pass * u < n:
                    k = k_base + per_pass * u + group
                    xk, xv = rows[u]
                    # The clip's threshold comes from the whole row, also for K over D/2.
                    if cutlass.const_expr(self.clip_indices[0] >= 0):
                        xk = self._clip(xk, lanes, self.clip_indices[0])
                    if cutlass.const_expr(self.clip_indices[1] >= 0):
                        xv = self._clip(xv, lanes, self.clip_indices[1])
                    sk, zk, k_in = self._scale_zero(xk, lanes // 2 if self.k_half else lanes)
                    sv, zv, v_in = self._scale_zero(xv, lanes)
                    sk_hi = cute.arch.shuffle_sync(sk, hi_lane)
                    zk_hi = cute.arch.shuffle_sync(zk, hi_lane)
                    if k < n:
                        t = t0 + k
                        self._quantize_to_smem(xk, sk, zk, k_in, sQk32, t, first_word)
                        self._quantize_to_smem(xv, sv, zv, v_in, sQv32, t, first_word)
                        vmax = cutlass.max(vmax, self._stored_bits(sv, zv))
                        if sub == 0:
                            self._store_scales(mScale, page, head, t, sk, zk, sk_hi, zk_hi,
                                               sv, zv)
            k_base = k_base + per_pass * passes
        for step in cutlass.range_constexpr(5):
            vmax = cutlass.max(vmax, cute.arch.shuffle_sync_bfly(vmax, offset=1 << step))
        return vmax

    @cute.jit
    def _load_codes(self, mC, token, head, part, parts: cutlass.Constexpr):
        """Code bytes [part D / parts, (part + 1) D / parts) of row (token, head) of a uint8
        [T, HKV, D] input of the prequantized write, as words of four channels each in channel
        order (the staging rows' order), every byte masked to the stored width."""
        D = self.head_dim
        count = D // parts
        addr = (mC.iterator.toint()
                + (Int64(token) * Int64(self.num_kv_heads) + Int64(head)) * Int64(D)
                + Int64(part * count))
        words = []
        if cutlass.const_expr(count == 16):
            words = ld_nc_b32x4(addr)
        elif cutlass.const_expr(count == 8):
            words = ld_nc_b32x2(addr)
        else:
            words = [ld_nc_b32(addr)]
        mask = Int32(CODE_MASK[self.bits])
        return [w & mask for w in words]

    @cute.jit
    def _saturated(self, s, z):
        """A given scale and zero point saturated to the quantizing writer's ranges (ITS
        SATURATION): s to `scale_range(qmax)`, z to the input envelope; fmax then fmin, so a NaN
        becomes the lower bound."""
        lo, hi = scale_range(self.qmax)
        s = cute.arch.fmin(cute.arch.fmax(s, Float32(lo)), Float32(hi))
        z = cute.arch.fmin(cute.arch.fmax(z, Float32(-INPUT_ENVELOPE)), Float32(INPUT_ENVELOPE))
        return s, z

    @cute.jit
    def _given_f32(self, mX, index):
        """Element `index` (Int64) of a C-contiguous f32 input of the prequantized write."""
        return f32_from_bits(ld_nc_b32(mX.iterator.toint() + index * Int64(4)))

    @cute.jit
    def _given_scales(self, mKs, mKz, mVs, mVz, token, head):
        """(sk, zk, sk_hi, zk_hi, sv, zv) of (token, head) as handed to the prequantized write,
        saturated. For K token-wise over D/2, (sk, zk) is channel group 0 and (sk_hi, zk_hi)
        group 1; otherwise the hi pair repeats (sk, zk), which `_store_scales` does not read.
        The element offsets are 64-bit, as every input offset of the writer is."""
        row = Int64(token) * Int64(self.num_kv_heads) + Int64(head)
        if cutlass.const_expr(self.k_half):
            sk, zk = self._saturated(self._given_f32(mKs, row * Int64(2)),
                                     self._given_f32(mKz, row * Int64(2)))
            sk_hi, zk_hi = self._saturated(self._given_f32(mKs, row * Int64(2) + Int64(1)),
                                           self._given_f32(mKz, row * Int64(2) + Int64(1)))
        else:
            sk, zk = self._saturated(self._given_f32(mKs, row), self._given_f32(mKz, row))
            sk_hi, zk_hi = sk, zk
        sv, zv = self._saturated(self._given_f32(mVs, row), self._given_f32(mVz, row))
        return sk, zk, sk_hi, zk_hi, sv, zv

    @cute.jit
    def _stage_given(self, mKc, mKs, mKz, mVc, mVs, mVz, mScale, sQk32, sQv32, p, n, t0, page,
                     head, warp, lane, lanes: cutlass.Constexpr):
        """Step 1 of the prequantized write, by every warp: stage the run's given codes and store
        their scale records (THE PREQUANTIZED WRITE). Each token takes an aligned group of
        `lanes` lanes, D / lanes channels each, so the CTA covers WARPS * 32 / lanes consecutive
        tokens per pass, and the codes and scales of GIVEN_PASSES_IN_FLIGHT passes are loaded
        before any is staged. A group past the run's end loads the run's last token again and
        stages nothing. Returns the largest stored s_v of the warp's tokens, as a bit pattern
        (every lane)."""
        D = self.head_dim
        per_pass = WARPS * (32 // lanes)
        group = lane // lanes
        sub = lane % lanes
        words = D // lanes // 4
        first_word = sub * words
        vmax = Int32(0)
        k_base = warp * (32 // lanes)
        while k_base < n:
            rows = []
            for u in cutlass.range_constexpr(GIVEN_PASSES_IN_FLIGHT):
                token = p + cutlass.min(k_base + per_pass * u + group, n - 1)
                rows.append((self._load_codes(mKc, token, head, sub, lanes),
                             self._load_codes(mVc, token, head, sub, lanes),
                             self._given_scales(mKs, mKz, mVs, mVz, token, head)))
            for u in cutlass.range_constexpr(GIVEN_PASSES_IN_FLIGHT):
                k = k_base + per_pass * u + group
                if k < n:
                    kw, vw, scales = rows[u]
                    sk, zk, sk_hi, zk_hi, sv, zv = scales
                    t = t0 + k
                    for i in cutlass.range_constexpr(words):
                        sQk32[(t, first_word + i)] = kw[i]
                        sQv32[(t, first_word + i)] = vw[i]
                    vmax = cutlass.max(vmax, self._stored_bits(sv, zv))
                    if sub == 0:
                        self._store_scales(mScale, page, head, t, sk, zk, sk_hi, zk_hi, sv, zv)
            k_base = k_base + per_pass * GIVEN_PASSES_IN_FLIGHT
        for step in cutlass.range_constexpr(5):
            vmax = cutlass.max(vmax, cute.arch.shuffle_sync_bfly(vmax, offset=1 << step))
        return vmax

    @cute.jit
    def _stored_sv_bits(self, mV, token, head, lane):
        """The bit pattern of the s_v the writer stores for (token, head), without storing
        anything (the `v_page_max` scan). The quantizing writer re-quantizes the row, which the
        whole warp reads; the prequantized write saturates the given V scale of (token, head),
        which `mV` holds, every lane reading the same word (the zero point does not enter the
        pattern)."""
        if cutlass.const_expr(self.prequantized):
            row = Int64(token) * Int64(self.num_kv_heads) + Int64(head)
            sv, _ = self._saturated(self._given_f32(mV, row), Float32(0.0))
            return self._stored_bits(sv, sv)
        else:
            xv = self._load_row(mV, token, head, lane, 32)
            if cutlass.const_expr(self.clip_indices[1] >= 0):
                xv = self._clip(xv, 32, self.clip_indices[1])
            sv, zv, _ = self._scale_zero(xv, 32)
            return self._stored_bits(sv, zv)

    @cute.jit
    def _run_start(self, mSlot, cursor, hi, lane):
        """The first token in [cursor, hi) that starts a run, or hi. Warp-uniform."""
        p = hi
        base = cursor
        while base < hi:
            j = base + lane
            start = Int32(0)
            if j < hi:
                s = mSlot[j]
                prev = Int64(-1)
                if j > 0:
                    prev = mSlot[j - 1]
                if s >= Int64(0):
                    if prev < Int64(0) or (s & Int64(PAGE - 1)) == Int64(0) or s != prev + Int64(1):
                        start = Int32(1)
            ballot = cute.arch.vote_ballot_sync(start != 0)
            if ballot != 0:
                p = base + ctz(ballot)
                base = hi
            else:
                base = base + 32
        return p

    @cute.jit
    def _run_length(self, mSlot, p, slot0, num_tokens, lane):
        """Tokens in the run that starts at token p with slot `slot0`: the first k >= 1 whose
        slot is not slot0 + k, capped at the page end. Warp-uniform."""
        room = Int32(PAGE) - Int32(slot0 & Int64(PAGE - 1))
        n = room
        for window in cutlass.range_constexpr(2):
            k = lane + 1 + 32 * window
            stop = Int32(0)
            if k < room:
                stop = Int32(1)
                if p + k < num_tokens:
                    if mSlot[p + k] == slot0 + Int64(k):
                        stop = Int32(0)
            ballot = cute.arch.vote_ballot_sync(stop != 0)
            if n == room and ballot != 0:
                n = ctz(ballot) + 1 + 32 * window
        return n

    @cute.jit
    def _page_scan(self, mSlot, mV, head, page, p, n, num_tokens, warp, lane, vmax):
        """The other tokens of `page` in this launch, outside the run [p, p + n): returns
        (vmax, hits_lo, hits_hi), where hits_lo and hits_hi are this warp's masks of the in-page
        positions [0, 32) and [32, 64) those tokens write (every lane holds them), and, on
        Blackwell, `vmax` is folded with the largest stored s_v among them. Warp w tests slot
        entries w*32 + lane + 128 i."""
        base = warp * 32
        hits_lo = Int32(0)
        hits_hi = Int32(0)
        while base < num_tokens:
            slots = []
            for r in cutlass.range_constexpr(SCAN_ROUNDS):
                j = cutlass.min(base + lane + THREADS * r, num_tokens - 1)
                slots.append(mSlot[j])
            for r in cutlass.range_constexpr(SCAN_ROUNDS):
                j = base + lane + THREADS * r
                hit = Int32(0)
                if j < num_tokens and (j < p or j >= p + n):
                    if slots[r] >= Int64(0) and (slots[r] >> 6) == page:
                        hit = Int32(1)
                        position = Int32(slots[r] & Int64(PAGE - 1))
                        if position < 32:
                            hits_lo = hits_lo | (Int32(1) << position)
                        else:
                            hits_hi = hits_hi | (Int32(1) << (position - 32))
                if cutlass.const_expr(self.WRITES_V_PAGE_MAX):
                    ballot = cute.arch.vote_ballot_sync(hit != 0)
                    while ballot != 0:
                        other = base + THREADS * r + ctz(ballot)
                        vmax = cutlass.max(vmax, self._stored_sv_bits(mV, other, head, lane))
                        ballot = ballot & (ballot - 1)
            base = base + THREADS * SCAN_ROUNDS
        hits_lo = cute.arch.warp_redux_sync(hits_lo, kind="or")
        hits_hi = cute.arch.warp_redux_sync(hits_hi, kind="or")
        return vmax, hits_lo, hits_hi

    @cute.jit
    def _tail_clear(self, mScale, sHits, page, head, n, tidx):
        """Step 2b, the tail clear of a run that starts a new use of `page` at position 0 with
        `n` < 64 tokens: thread q < 64 stores (s, z) = (0, 0) at position q when q lies past the
        run and no token of the launch writes q, by the warps' masks in `sHits`."""
        if tidx < PAGE and tidx >= n:
            taken = sHits[0, tidx // 32]
            for w in cutlass.range_constexpr(1, WARPS):
                taken = taken | sHits[w, tidx // 32]
            if ((taken >> (tidx % 32)) & Int32(1)) == 0:
                zero = Float32(0.0)
                self._store_scales(mScale, page, head, tidx, zero, zero, zero, zero, zero, zero)

    @cute.jit
    def _page_start_run(self, mSlot, mV, mScale, sHits, head, p, n, num_tokens, warp, lane,
                        tidx):
        """Hopper's page-start pass for one run [p, p + n) that starts a new use of its page:
        the scan, then the tail clear, each behind a CTA barrier."""
        page = mSlot[p] >> 6
        _, hits_lo, hits_hi = self._page_scan(mSlot, mV, head, page, p, n, num_tokens, warp,
                                              lane, Int32(0))
        if lane == 0:
            sHits[warp, 0] = hits_lo
            sHits[warp, 1] = hits_hi
        cute.arch.barrier()
        self._tail_clear(mScale, sHits, page, head, n, tidx)
        cute.arch.barrier()

    @cute.jit
    def _settle_v_max(self, mVMaxHist, old, best, reset):
        """Move the count of one `v_page_max` update in `v_max_hist`: the cell held `old` and
        now holds `best` when `reset` is 1 (an exchange) or max(old, best) as unsigned words
        (a max). A cell counts when its pattern is not +0.0 or -0.0. When old and new both
        count in one bin nothing moves; otherwise the new bin gains one before the old bin
        loses one, so this thread's own move never passes through a table that counts the cell
        in neither bin."""
        new = best
        if reset == 0:
            if cutlass.Uint32(old) > cutlass.Uint32(best):
                new = old
        b_old = vmax_bucket(old, cutlass.min)
        b_new = vmax_bucket(new, cutlass.min)
        counts_old = (old & Int32(0x7FFFFFFF)) != 0
        counts_new = (new & Int32(0x7FFFFFFF)) != 0
        same = Int32(0)
        if counts_old and counts_new and b_old == b_new:
            same = Int32(1)
        if same == 0:
            base = mVMaxHist.iterator.toint()
            if counts_new:
                red_b32("add.u32", base + Int64(b_new) * Int64(4), Int32(1))
            if counts_old:
                red_b32("add.u32", base + Int64(b_old) * Int64(4), Int32(-1))

    # -----------------------------------------------------------------------------------------
    # The kernel
    # -----------------------------------------------------------------------------------------

    @cute.kernel
    def kernel(self, mK, mV, mSlot, mKq, mVq, mScale, mVPageMax, mVMaxHist, num_slices: Int32):
        """The quantizing writer of `append`: bf16 K and V [T, HKV, D]."""
        self._write(mK, mV, None, None, None, None, mSlot, mKq, mVq, mScale, mVPageMax,
                    mVMaxHist, num_slices)

    @cute.kernel
    def kernel_prequantized(self, mKc, mKs, mKz, mVc, mVs, mVz, mSlot, mKq, mVq, mScale,
                            mVPageMax, mVMaxHist, num_slices: Int32):
        """The prequantized write of `append_quantized` (THE PREQUANTIZED WRITE): uint8 codes
        `mKc`, `mVc` [T, HKV, D] and f32 scales and zero points `mKs`, `mKz`, `mVs`, `mVz`."""
        self._write(mKc, mVc, mKs, mKz, mVs, mVz, mSlot, mKq, mVq, mScale, mVPageMax,
                    mVMaxHist, num_slices)

    @cute.jit
    def _write(self, mK, mV, mKs, mKz, mVs, mVz, mSlot, mKq, mVq, mScale, mVPageMax, mVMaxHist,
               num_slices):
        """The run loop both kernels run. `mK` and `mV` are the bf16 rows of the quantizing
        writer, whose `mKs`, `mKz`, `mVs` and `mVz` are None, or the codes of the prequantized
        write, which reads its scales and zero points from those four; every difference
        between the two is a trace-time branch on `prequantized`."""
        D = self.head_dim
        tidx, _, _ = cute.arch.thread_idx()
        # The KV heads are the grid's fastest dimension, so the CTAs of one token slice, which
        # have the same amount of work, are spread over SMs rather than stacked on one.
        head, slice_idx, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        num_tokens = Int32(mSlot.shape[0])
        num_pages = Int64(mKq.shape[0])
        lo = Int32(Int64(slice_idx) * Int64(num_tokens) // Int64(num_slices))
        hi = Int32(Int64(slice_idx + 1) * Int64(num_tokens) // Int64(num_slices))

        smem = cutlass.utils.SmemAllocator()
        row = self.smem_row
        staging = cute.make_layout((PAGE, D), stride=(row, 1))
        sQk = smem.allocate_tensor(cutlass.Uint8, staging, byte_alignment=16)
        sQv = smem.allocate_tensor(cutlass.Uint8, staging, byte_alignment=16)
        words = cute.make_layout((PAGE, row // 4), stride=(row // 4, 1))
        sQk32 = cute.make_tensor(cute.recast_ptr(sQk.iterator, dtype=Int32), words)
        sQv32 = cute.make_tensor(cute.recast_ptr(sQv.iterator, dtype=Int32), words)
        sVmax = None
        sSeen = None
        if cutlass.const_expr(self.WRITES_V_PAGE_MAX):
            sVmax = smem.allocate_tensor(Int32, cute.make_layout(WARPS), byte_alignment=16)
        if cutlass.const_expr(self.WRITES_V_PAGE_MAX and not self.seen_in_register):
            # Blackwell: the page maximum's cell value, read for 2a's test.
            sSeen = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
        # Each warp's masks of the in-page positions the launch's other tokens write (the tail
        # clear), positions [0, 32) in column 0 and [32, 64) in column 1.
        sHits = smem.allocate_tensor(Int32, cute.make_layout((WARPS, 2), stride=(2, 1)),
                                     byte_alignment=16)
        # Hopper: the first and the length of each run that starts a new use of its page.
        sStarts = None
        if cutlass.const_expr(not self.WRITES_V_PAGE_MAX):
            sStarts = smem.allocate_tensor(
                Int32, cute.make_layout((PAGE_START_LIST, 2), stride=(2, 1)), byte_alignment=16)

        # Where the page scan takes the stored s_v of the page's other tokens from: their bf16 V
        # rows, quantized again, or their given V scales (`_stored_sv_bits`).
        vscan = mV
        if cutlass.const_expr(self.prequantized):
            vscan = mVs
        # Hopper: how many runs of this CTA started a new use of their page (the page-start pass).
        starts = Int32(0)
        cursor = lo
        while cursor < hi:
            p = self._run_start(mSlot, cursor, hi, lane)
            if p < hi:
                slot0 = mSlot[p]
                page = slot0 >> 6
                t0 = Int32(slot0 & Int64(PAGE - 1))
                n = self._run_length(mSlot, p, slot0, num_tokens, lane)
                if page >= num_pages:
                    trap()
                # Blackwell: for a run that continues its page, the page maximum's cell is read
                # for 2a's test with a relaxed gpu-scope load, a single-copy atomic read: by
                # thread 0 into a register held while the run is quantized, or by lane 0 of the
                # last warp, which stores it to `sSeen` before it quantizes anything
                # (`seen_in_register`).
                seen = Int32(0)
                if cutlass.const_expr(self.WRITES_V_PAGE_MAX and self.seen_in_register):
                    if tidx == 0 and t0 != 0:
                        seen = ld_relaxed_b32(mVPageMax.iterator.toint()
                                              + (page * Int64(self.num_kv_heads)
                                                 + Int64(head)) * Int64(4))
                if cutlass.const_expr(self.WRITES_V_PAGE_MAX and not self.seen_in_register):
                    if tidx == (WARPS - 1) * 32 and t0 != 0:
                        sSeen[0] = ld_relaxed_b32(mVPageMax.iterator.toint()
                                                  + (page * Int64(self.num_kv_heads)
                                                     + Int64(head)) * Int64(4))
                if cutlass.const_expr(self.prequantized):
                    # 1. stage the given codes and store the given scales, D / GROUP_CHANNELS
                    # lanes per token on every run (THE PREQUANTIZED WRITE).
                    vmax = self._stage_given(mK, mKs, mKz, mV, mVs, mVz, mScale, sQk32, sQv32,
                                             p, n, t0, page, head, warp, lane,
                                             D // GROUP_CHANNELS)
                else:
                    # 1. quantize: a long run with D / GROUP_CHANNELS lanes per token, a short
                    # one (decode tokens) with a whole warp per token, which issues fewer
                    # instructions per token when most lanes would hold no token.
                    lanes = D // GROUP_CHANNELS
                    short_lanes = min(32, D // 4)
                    vmax = Int32(0)
                    if n >= WARPS * (32 // lanes):
                        vmax = self._quantize_tokens(mK, mV, mScale, sQk32, sQv32, p, n, t0,
                                                     page, head, warp, lane, lanes)
                    else:
                        vmax = self._quantize_tokens(mK, mV, mScale, sQk32, sQv32, p, n, t0,
                                                     page, head, warp, lane, short_lanes)
                # A run that starts a new use of its page and ends before the page does. On
                # Blackwell it scans the launch for the page's other tokens here, because their
                # s_v enter this run's exchange of the page maximum (2a), and clears the tail in
                # 2b. On Hopper the scan and the clear wait for the page-start pass after the run
                # loop, so that this loop compiles without them (see the pass).
                starts_use = Int32(0)
                if t0 == 0 and n < PAGE:
                    starts_use = Int32(1)
                    if cutlass.const_expr(self.WRITES_V_PAGE_MAX):
                        vmax, hits_lo, hits_hi = self._page_scan(mSlot, vscan, head, page, p, n,
                                                                 num_tokens, warp, lane, vmax)
                        if lane == 0:
                            sHits[warp, 0] = hits_lo
                            sHits[warp, 1] = hits_hi
                if cutlass.const_expr(self.WRITES_V_PAGE_MAX):
                    if lane == 0:
                        sVmax[warp] = vmax
                cute.arch.barrier()
                # 2a. the page maximum: a run that continues its page and whose best is no
                # larger than the cell value read skips its update; every other run issues its
                # atomic here and settles it before step 3 (`seen_in_register`) or after it, so
                # that the atomic returns while the tail is cleared or the code words stored.
                upd = Int32(0)
                old = Int32(0)
                best = Int32(0)
                reset = Int32(0)
                if cutlass.const_expr(self.WRITES_V_PAGE_MAX):
                    if tidx == 0:
                        best = sVmax[0]
                        for w in cutlass.range_constexpr(1, WARPS):
                            best = cutlass.max(best, sVmax[w])
                        if t0 == 0:
                            reset = Int32(1)
                        if cutlass.const_expr(not self.seen_in_register):
                            if t0 != 0:
                                seen = sSeen[0]
                        if reset != 0 or cutlass.Uint32(best) > cutlass.Uint32(seen):
                            upd = Int32(1)
                            addr = (mVPageMax.iterator.toint()
                                    + (page * Int64(self.num_kv_heads) + Int64(head)) * Int64(4))
                            if reset != 0:
                                old = atom_b32("exch.b32", addr, best)
                            else:
                                old = atom_b32("max.u32", addr, best)
                # 2b. the tail clear (Blackwell; Hopper's runs in the page-start pass)
                if cutlass.const_expr(self.WRITES_V_PAGE_MAX):
                    if starts_use != 0:
                        self._tail_clear(mScale, sHits, page, head, n, tidx)
                else:
                    if starts_use != 0:
                        if tidx == 0 and starts < PAGE_START_LIST:
                            sStarts[starts, 0] = p
                            sStarts[starts, 1] = n
                        starts = starts + 1
                if cutlass.const_expr(self.WRITES_V_PAGE_MAX and self.seen_in_register):
                    if upd != 0:
                        self._settle_v_max(mVMaxHist, old, best, reset)
                # 3. the code words
                self._store_run(sQk, sQv, mKq, mVq, page, head, t0, n, tidx)
                if cutlass.const_expr(self.WRITES_V_PAGE_MAX and not self.seen_in_register):
                    if upd != 0:
                        self._settle_v_max(mVMaxHist, old, best, reset)
                cute.arch.barrier()
                cursor = p + n
            else:
                cursor = hi
        # THE PAGE-START PASS (Hopper). The scan and the clear of every run that started a new use
        # of its page, after all of this CTA's runs, so that the run loop above compiles without
        # them: inside it they raised the 64-register writer's spills on every run, a 4.5% to 12%
        # slower decode-step write on H100. The clear stores only positions no token
        # of the launch writes, so it commutes with every store of the launch and may come later.
        # The runs come from `sStarts`, which thread 0 filled before the run loop's last barrier;
        # a CTA that met more of them than the list holds finds them again from the slot map.
        # `starts` is the same in every thread (every warp finds the same runs), so the pass, and
        # its barriers, run in every thread or in none; a step whose runs all continue a page
        # skips it.
        if cutlass.const_expr(not self.WRITES_V_PAGE_MAX):
            if starts != 0:
                if starts <= PAGE_START_LIST:
                    for i in cutlass.range(starts, unroll=1):
                        self._page_start_run(mSlot, vscan, mScale, sHits, head, sStarts[i, 0],
                                             sStarts[i, 1], num_tokens, warp, lane, tidx)
                else:
                    cursor = lo
                    while cursor < hi:
                        p = self._run_start(mSlot, cursor, hi, lane)
                        if p < hi:
                            slot0 = mSlot[p]
                            n = self._run_length(mSlot, p, slot0, num_tokens, lane)
                            if Int32(slot0 & Int64(PAGE - 1)) == 0 and n < PAGE:
                                self._page_start_run(mSlot, vscan, mScale, sHits, head, p, n,
                                                     num_tokens, warp, lane, tidx)
                            cursor = p + n
                        else:
                            cursor = hi
        if cutlass.const_expr(self.WRITES_V_PAGE_MAX):
            # The Blackwell decode follows griddepcontrol.wait before any global read, so it
            # may launch as soon as every CTA has issued its stores; thread
            # 0 signals for its CTA, after its last table move.
            if tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _store_word_group(self, addr, masks, values):
        """Store 2 or 4 consecutive words of one destination row at a correspondingly aligned
        address: one vector store when the run covers all of them, otherwise a plain store per
        covered word and red.and then red.or per partly covered one. A mask holds the bits of
        the fields this run owns; a word with no such field is left alone."""
        n = len(values)
        full = Int32(-1)
        if cutlass.const_expr(n == 2):
            all_full = (masks[0] & masks[1]) == full
        else:
            all_full = (masks[0] & masks[1] & masks[2] & masks[3]) == full
        if all_full:
            st_b32(addr, values)
        else:
            for u in cutlass.range_constexpr(n):
                if masks[u] == full:
                    st_b32(addr + Int64(4 * u), [values[u]])
                else:
                    if masks[u] != 0:
                        red_b32("and.b32", addr + Int64(4 * u), ~masks[u])
                        red_b32("or.b32", addr + Int64(4 * u), values[u])


# ---------------------------------------------------------------------------------------------
# Compile and launch
# ---------------------------------------------------------------------------------------------

#: Resident CTAs per SM the writer is built for (its register bound, `min_blocks_per_mp`) and
#: the grid is sized for, where shared memory allows: G = min(T, SMs * resident / HKV).
CTAS_PER_SM = 8
#: Shared memory per SM on both GPUs, less the 1 KiB the hardware reserves for each CTA.
_SMEM_PER_SM = 228 * 1024


def resident_ctas_per_sm(head_dim):
    """CTAS_PER_SM, or fewer where the two staging buffers do not leave room for that many:
    six at head_dim 256. Beside the staging buffers: `sVmax` (16 bytes), `sHits` (32),
    `sStarts` (8 per entry of `PAGE_START_LIST`) and `sSeen` (16)."""
    smem = (2 * PAGE * (head_dim + ROW_PAD) + 16 + 32 + 8 * PAGE_START_LIST
            + 16 + 1024)
    return min(CTAS_PER_SM, _SMEM_PER_SM // smem)


def layout_key(layout):
    """The layout fields a writer reads, which is what its compile is keyed on."""
    return (("arch", layout.arch), ("bits", int(layout.bits)), ("head_dim", int(layout.head_dim)),
            ("num_kv_heads", int(layout.num_kv_heads)),
            ("k_quant", (layout.k_quant[0], int(layout.k_quant[1]))),
            ("v_quant", (layout.v_quant[0], int(layout.v_quant[1]))),
            ("scale_dtype", str(layout.scale_dtype)),
            ("code_pool_order", layout.code_pool_order),
            ("k_pool_permuted", layout.k_pool_permuted),
            ("v_pool_permuted", layout.v_pool_permuted))


def input_fakes(kern):
    """Fake k, v and slot_mapping for a symbolic token count T and free token strides, plus the
    symbolic page count the pool fakes share. One compile serves every T, stride and NP."""
    num_tokens, num_pages = cute.sym_int(), cute.sym_int()
    shape = (num_tokens, kern.num_kv_heads, kern.head_dim)
    rows = [cute.runtime.make_fake_tensor(
        cutlass.BFloat16, shape, stride=(cute.sym_int(divisibility=8), kern.head_dim, 1),
        assumed_align=16) for _ in range(2)]
    slots = cute.runtime.make_fake_compact_tensor(Int64, (num_tokens,), assumed_align=8)
    return rows[0], rows[1], slots, num_pages


def given_fakes(kern):
    """Fake inputs of the prequantized write (THE PREQUANTIZED WRITE), in its kernel's order:
    uint8 K codes [T, HKV, D], f32 K scales and zero points [T, HKV] ([T, HKV, 2] for K
    token-wise over D/2), the same for V ([T, HKV]), all C-contiguous, and the slot map; plus the
    symbolic page count the pool fakes share. T is symbolic, so one compile serves every T."""
    num_tokens, num_pages = cute.sym_int(), cute.sym_int()
    H, D = kern.num_kv_heads, kern.head_dim

    def compact(dtype, shape, align):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(range(len(shape) - 1, -1, -1)), assumed_align=align)

    k_shape = (num_tokens, H, 2) if kern.k_half else (num_tokens, H)
    k_codes, v_codes = (compact(cutlass.Uint8, (num_tokens, H, D), 16) for _ in range(2))
    k_scale, k_zero = (compact(Float32, k_shape, 4) for _ in range(2))
    v_scale, v_zero = (compact(Float32, (num_tokens, H), 4) for _ in range(2))
    slots = cute.runtime.make_fake_compact_tensor(Int64, (num_tokens,), assumed_align=8)
    return (k_codes, k_scale, k_zero, v_codes, v_scale, v_zero, slots), num_pages


def scale_fake(kern, num_pages):
    """The scale pool: the legacy [NP, HKV, 64, 4] or the sectioned [NP, HKV, block_tot]."""
    dtype = cutlass.BFloat16 if kern.scale_bf16 else Float32
    if kern.legacy:
        shape = (num_pages, kern.num_kv_heads, PAGE, 4)
    else:
        shape = (num_pages, kern.num_kv_heads, kern.block_tot)
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(range(len(shape) - 1, -1, -1)), assumed_align=16)


def pool_fake(shape):
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, shape, stride_order=tuple(range(len(shape) - 1, -1, -1)), assumed_align=16)


def compile_writer(kern, module_file, layout, fakes):
    """Compile `kern` against `fakes` through the persistent kernel cache, keyed on the layout
    and the constructed writer's resolved attributes, `prequantized` among them.

    The writer compiles under the rules the decode kernels follow under nvidia-cutlass-dsl 4.8:
    the CUDA driver is initialized before the trace (`kernel_cache.init_cuda_driver`), a Hopper
    writer compiles through `pingpong_runtime.sm90_compiler`, which states every NVVM option a
    decode compile sets at its reset value so that the writer's binary does not depend on what
    the process compiled before it, and a Blackwell writer is refused under the DSL's CUDA 12
    backend (`fold_runtime._refuse_cuda12_dsl_backend`), as the Blackwell decode is."""
    import os
    from . import kernel_cache

    opt_level = os.environ.get("CUTE_DSL_OPT_LEVEL", "3")
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if kern.ARCH == "sm90":
        from ._sm90 import pingpong_runtime

        compile_fn, options = pingpong_runtime.sm90_compiler(opt_level)
        sources = kernel_cache.source_closure(module_file, pingpong_runtime.__file__)
    else:
        from ._sm100.fold_runtime import _refuse_cuda12_dsl_backend

        options = f"--enable-tvm-ffi --opt-level {opt_level}"

        def compile_fn(*args):
            _refuse_cuda12_dsl_backend(where=f"{type(kern).__name__} compile")
            return cute.compile(*args, options=options)

        sources = kernel_cache.source_closure(module_file)

    # The quantizing writer is traced through `__call__`, the prequantized write through
    # `call_prequantized`; each launches its own kernel. The dequantize
    # (`_dequant_core.compile_dequant`) compiles here too, through its `__call__`, and has no
    # prequantized form.
    entry = kern.call_prequantized if getattr(kern, "prequantized", False) else kern

    def _trace():
        kernel_cache.init_cuda_driver()
        return compile_fn(entry, *fakes, Int32(1), stream_fake)

    return kernel_cache.compile_or_load(
        kind=f"{kern.ARCH}_{type(kern).__name__}",
        compile_fn=_trace,
        sources=sources,
        config={"kernel": kernel_cache.simple_attributes(kern), "trace": dict(layout_key(layout))},
        options=options)


@functools.cache
def _sm_count(device_index):
    import torch
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def launch_writer(compiled, tensors):
    """Launch a compiled writer on torch's current stream. `tensors` starts with k (the K codes
    for the prequantized write), [T, HKV, D]; the grid width G = min(T, SMs *
    resident_ctas_per_sm / HKV) is the last argument. A launch with no tokens does nothing."""
    import tvm_ffi

    k = tensors[0]
    num_tokens, num_kv_heads, head_dim = k.shape
    if num_tokens == 0:
        return
    budget = _sm_count(k.device.index) * resident_ctas_per_sm(head_dim) // num_kv_heads
    with tvm_ffi.use_torch_stream():
        compiled(*tensors, Int32(max(1, min(num_tokens, budget))))
