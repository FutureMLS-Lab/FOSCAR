# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-instruction inline-asm primitives for the SM100 fold decode kernel.

Each function here wraps one PTX instruction, or one short straight-line sequence
with no control flow, that the DSL has no operator for. They carry no kernel
state, read nothing from a ring, and mutate no pipeline, so they can move out of
`int_fold_decode.py` without any check beyond "the emitted PTX did not change".

The byte maps (`map_int4`, `map_int2_k`, ...) belong in the kernel module
instead. They encode a pool layout, so they stay beside the deposits that read
it, and their closure defaults let a concurrent stream hand the same map its own
staging tensors.
"""
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def _lift_exp(amax, *, loc=None, ip=None):
    """The channel-K Q'-plane po2 lift exponent, read off the f32 exponent field.

    Returns ``m = clamp(floor(log2(amax)) - 7, -14, 10)`` for every NONNEGATIVE
    FINITE ``amax``, which is the largest ``e`` in [-13, 10] with
    ``amax >= 2^(e+7)``, else -14:

      * normal amax: biased exponent ``e_b = (bits >> 23) & 0xFF`` gives
        ``floor(log2(amax)) = e_b - 127``, so ``m = e_b - 134`` before the clamp.
      * ``amax == 0`` and subnormal amax: ``e_b == 0`` -> ``-134`` -> clamps to
        ``-14``, below the 2^-6 floor. The zero case is not hypothetical — an
        out-of-range group loads ``s_raw == 0``, so it happens on the tail tile
        of most sequences.

    The mantissa does not affect the answer, which is why four integer
    instructions suffice.

    The domain is the nonnegative finites, and only there does this match an
    ordered compare ladder. Outside it the two disagree: the ladder's ``>=``
    compares are false for NaN so it yields -14, while the exponent read sees
    ``e_b == 255`` and clamps to +10, and a negative input yields its
    magnitude's exponent rather than -14. Neither is reachable here — ``amax``
    starts at ``Float32(0)`` and is an ``fmax`` chain over
    ``s_eff[j] * rMaxq[rr, j]``, where ``rMaxq`` is ``fmax(qv, -qv) >= 0`` and
    ``s_eff`` is a stored quantization scale
    (``(max-min).clamp_min(1e-8)/qmax``, strictly positive) times 1.0 or 2.0. So
    the input is nonnegative by construction, and NaN requires already-NaN pool
    data. A caller that widens that domain gets a different result, and the
    host reference must follow.
    """
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(amax).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 %eb;"
        " mov.b32 %eb, $1;"
        " shr.u32 %eb, %eb, 23;"
        " and.b32 %eb, %eb, 255;"
        " sub.s32 %eb, %eb, 134;"
        " max.s32 %eb, %eb, -14;"
        " min.s32 $0, %eb, 10; }",
        "=r,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _prmt_r(a, b, c, *, loc=None, ip=None):
    # PRMT with a REGISTER selector, the byte-LUT dequant primitive: each of c's
    # 4 low nibbles picks a byte of {a,b}; a selector nibble with bit3 set
    # sign-replicates the selected byte's MSB (exploited: MSB-0 table bytes make
    # out-of-half encodings read 0x00, so the two int4 table halves OR together).
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip),
                  Int32(c).ir_value(loc=loc, ip=ip)],
        "prmt.b32 $0, $1, $2, $3;", "=r,r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


# ---- fold-off (fold_scale_zero=False) element-wise dequant primitives ----
#
# The fold-off path dequantizes with ONE PACKED FMA PER TWO ELEMENTS, in bf16,
# and only then converts to the e4m3 operand. That is deliberate: it is the
# baseline of a fold-versus-dequant comparison, so it must pay what element-wise
# affine dequantization costs, not what a clever encoding of it costs. bf16 also
# matches the sm90 path's `_hfma2_bf16` dequant, so the two architectures differ
# in operand dtype and nothing else.
#
# There is no bf16 -> fp8 pack in PTX (`cvt.rn.satfinite.e4m3x2.bf16x2` is
# rejected by ptxas on sm_100a, as is `cvt.rn.bf16x2.e4m3x2` the other way), so
# the pack widens each bf16 lane to f32 first — free in register terms, two
# integer ops — and uses the f32 form. `fma.rn.f16x2` + the direct
# `cvt.rn.satfinite.e4m3x2.f16x2` would save those two ops and carry 10 mantissa
# bits instead of 7; bf16 is chosen for parity with sm90. Either way the value is
# rounded TWICE (to the FMA dtype, then to e4m3), which the reference and the
# byte-exact gate both model.
@dsl_user_op
def _fma2_bf16(a, b, c, *, loc=None, ip=None):
    # d = fma.rn per bf16 lane on .b32-packed bf16x2 words: ONE rounding of the
    # exact a*b + c into bf16, which is what lets the host expectation replay it
    # by rounding an f64 product-sum once (`fosca.pool.dequant_bf16_fma_e4m3`).
    return Int32(llvm.inline_asm(
        T.i32(),
        [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip),
         Int32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _sub2_bf16(a, b, *, loc=None, ip=None):
    # d = a - b per bf16 lane. Takes the magic base back off the feed
    # (y = 4 + q -> q), which is EXACT here by Sterbenz.
    return Int32(llvm.inline_asm(
        T.i32(),
        [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "sub.rn.bf16x2 $0, $1, $2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _cvt2_bf16x2(hi, lo, *, loc=None, ip=None):
    # (f32 lo, f32 hi) -> one bf16x2 word. Used once per (thread, page) to
    # broadcast the token's s and z' into both lanes.
    return Int32(llvm.inline_asm(
        T.i32(),
        [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _e4m3x4_of_bf16x2(w01, w23, *, loc=None, ip=None):
    """Two bf16x2 words -> u32 of FOUR e4m3 bytes (byte j = lane j in order).

    bf16 lane -> f32 is a pure bit placement (bf16 IS the top half of an f32),
    so each widening is one integer op — `shl 16` for the low lane, a mask for
    the high one — and the `mov.b32` reinterpretations cost nothing because PTX
    registers are untyped. Then one `cvt.rn.satfinite.e4m3x2.f32` per pair and a
    vector `mov.b32` to place the two halves, so the whole pack is 7 ops per 4
    elements. (`cvt.u32.u16` is NOT the way to widen the 16-bit result: ptxas
    rejects it — "Unknown field '.u32'" — hence the vector `mov.b32`.)

    RNE + satfinite is `torch.Tensor.to(float8_e4m3fn)`'s rounding, and the
    saturation matters: a dequantized value past 448 must clamp rather than
    become NaN, exactly as an fp8 KV cache would store it.
    """
    return Int32(llvm.inline_asm(
        T.i32(),
        [Int32(w01).ir_value(loc=loc, ip=ip), Int32(w23).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 %tl, %th;\n"
        " .reg .f32 %fl, %fh;\n"
        " .reg .b16 %e0, %e1;\n"
        " shl.b32 %tl, $1, 16;\n"
        " and.b32 %th, $1, 0xffff0000;\n"
        " mov.b32 %fl, %tl;\n"
        " mov.b32 %fh, %th;\n"
        " cvt.rn.satfinite.e4m3x2.f32 %e0, %fh, %fl;\n"   # bytes {1, 0}
        " shl.b32 %tl, $2, 16;\n"
        " and.b32 %th, $2, 0xffff0000;\n"
        " mov.b32 %fl, %tl;\n"
        " mov.b32 %fh, %th;\n"
        " cvt.rn.satfinite.e4m3x2.f32 %e1, %fh, %fl;\n"   # bytes {3, 2}
        " mov.b32 $0, {%e0, %e1}; }\n",
        "=r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _tid_x_fresh(*, loc=None, ip=None):
    """`%tid.x`, read at the point of the call.

    `cute.arch.thread_idx()` lowers to an LLVM intrinsic that LLVM may hoist and merge with
    every other read, so the kernel-entry value is what every later use reads. The persistent
    item loop of the Blackwell kernel dispatches the warp roles on the warp index at the top
    of every item, and a warp index computed once at kernel entry stays live through every
    role's body. ptxas then sometimes keeps it in a vector register rather than a uniform
    one, compiles the role dispatch as a divergent branch, and every uniform operand of the
    MMA and TMA warps' instructions is rebuilt from vector registers on every issue
    (R2UR.BROADCAST under ELECT). A side-effecting `mov` is neither hoisted nor merged, so the
    index derived from it is short-lived.
    """
    return Int32(llvm.inline_asm(
        T.i32(), [], "mov.u32 $0, %tid.x;", "=r",
        has_side_effects=True, is_align_stack=False))


@dsl_user_op
def _opaque_i32(x, *, loc=None, ip=None):
    """`x`, through a side-effecting `mov`, so LLVM can neither merge two copies nor see that a
    copy equals `x`.

    The persistent item loop of the Blackwell kernel dispatches four warp roles by comparing the
    item's warp index with each role's warp number. Compared on one value, the four checks lose
    their source order: the PTX tests the value against the MMA-KQ and TMA-VO warp numbers
    first, and ptxas then places those two bodies after the dequant loops, which moves every
    other role's loop and costs instruction-cache hits. With each check comparing its own copy,
    the checks stay a chain in source order and the bodies keep the layout produced by reading
    the warp index once at kernel entry.
    """
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,r",
        has_side_effects=True, is_align_stack=False))
