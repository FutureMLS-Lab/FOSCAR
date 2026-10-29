# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""Pool construction for the SM100 quantized-KV decode kernel.

The kernel reads packed pools, not dequantized tensors. This module turns a bf16
KV cache into those pools, and provides a reference implementation of the same
attention so a caller can check a kernel run against a storage-matched oracle.

POOLS (page_size=64, D=head_dim)

  K codes   uint8 [NP, HKV, 64, D*bits//8]   one token's channels per row
  V codes   uint8 [NP, HKV, D, 64*bits//8]   one channel's tokens per row
  scales    f32   [NP, HKV, 64, 4]           (s_k, z_k, s_v, z_v) per token

Every stored integer value is permuted in a way the kernel requires and the bytes
do not reveal. Ask `k_store_perm_sm100_f4` for K and `v_store_perm_sm100_f4` for
V, keyed by the bit width and the resolved feed, and apply the permutation before
packing. `PagedDecodeWrapper.run(k_pool_order=...)` refuses a K pool whose order
the caller has not declared.

QUANTIZED Q

The kernel takes fp8 Q with a per-tensor scale. `quantize_q_fp8` produces it in
the operation order the kernel expects: ``qs = 448 / amax(|q|)``,
``q_fp8 = e4m3(clamp(q * qs))``, and the effective Q the kernel sees is
``q_fp8.float() / qs``. A divide-form scale is one fp8 ULP different on some
elements and is not this contract.

REFERENCE

`dequant_pools_flex` inverts the pools, and `ref_paged_attention_fp8q_flex`
scores attention over the dequantized logical stream through a block table. The
self-gates in ``__main__`` check the pack/unpack round trip and the paged
reference against an independent path.
"""
import os
import sys

import torch

D, PS = 128, 64      # head_dim, page_size
E4M3_MAX = 448.0      # fp8 e4m3 saturation magnitude (trtllm-style per-tensor Q contract)


def quant_tokenwise(x, bits):
    """x [..., T, D] -> codes uint8 in [0, 2^bits-1], s, z per token (f32).

    Computes on ``x``'s device, and the device is part of the answer: f32 division
    rounds differently on the CPU and on CUDA, and ``torch.round`` sits on top of
    it, so the same bf16 values quantise to codes that differ by +-1 in a small
    fraction of positions between the two. A fixture whose bytes are anchored
    must therefore quantise on the CPU.

    One float32 materialisation and one ``aminmax`` pass rather than three
    ``.float()`` conversions. The values are identical: the minimum and maximum
    are exact, and the division and the rounding see the same float32 inputs.
    """
    qmax = 2**bits - 1
    xf = x.float()
    mn, mx = torch.aminmax(xf, dim=-1, keepdim=True)
    s = (mx - mn).clamp_min(1e-8) / qmax
    q = torch.round((xf - mn) / s).clamp_(0, qmax).to(torch.uint8)
    return q, s.squeeze(-1), mn.squeeze(-1)


def _plane_pack_u32(codes, bits):
    """codes [..., G] with G = 32//bits codes per u32 -> u32 plane-interleaved words.
    evens -> low half fields, odds -> high half fields (field width = bits)."""
    c = codes.to(torch.int64)
    G = 32 // bits
    w = torch.zeros(c.shape[:-1], dtype=torch.int64, device=c.device)
    for i in range(G):
        half, slot = i % 2, i // 2
        w |= c[..., i] << (16 * half + bits * slot)
    return w.to(torch.uint32)


def _plane_unpack_u32(w, bits):
    G = 32 // bits
    w = w.to(torch.int64)
    out = []
    for i in range(G):
        half, slot = i % 2, i // 2
        out.append((w >> (16 * half + bits * slot)) & (2**bits - 1))
    return torch.stack(out, -1).to(torch.uint8)


def _pack_rows_plane(codes, bits):
    """codes [..., R, E] (E elems per row) -> uint8 [..., R, E*bits//8] plane-word rows."""
    G = 32 // bits
    *lead, R, E = codes.shape
    assert E % G == 0
    w = _plane_pack_u32(codes.reshape(*lead, R, E // G, G), bits)  # [..., R, E//G] u32
    return w.view(torch.uint8).reshape(*lead, R, E * bits // 8)


def _unpack_rows_plane(packed, bits, E):
    G = 32 // bits
    *lead, R, B = packed.shape
    w = packed.reshape(*lead, R, B // 4, 4).view(torch.uint32).squeeze(-1)
    return _plane_unpack_u32(w, bits).reshape(*lead, R, E)


def _build_pools_plane(k, v, bits):
    """k, v: [NP, PS, HKV, D] bf16 (paged, physical pages). Returns (mKq, mVq, mScale).
    Pools are physical-page-indexed; the block table only matters at read time."""
    NP, ps, HKV, d = k.shape
    assert ps == PS and d == D
    qk, sk, zk = quant_tokenwise(k.permute(0, 2, 1, 3), bits)   # [NP,HKV,PS,D]
    qv, sv, zv = quant_tokenwise(v.permute(0, 2, 1, 3), bits)
    mKq = _pack_rows_plane(qk, bits).contiguous()                       # [NP,HKV,PS,D*bits//8]
    mVq = _pack_rows_plane(qv.transpose(-1, -2).contiguous(), bits).contiguous()  # [NP,HKV,D,PS*bits//8]
    mScale = torch.stack([sk, zk, sv, zv], -1).float().contiguous()        # [NP,HKV,PS,4]
    # Byte contract the GPU kernel depends on: contiguous pools, 16B-aligned code rows.
    assert mKq.is_contiguous() and mVq.is_contiguous() and mScale.is_contiguous()
    assert mKq.shape[-1] % 16 == 0, mKq.shape   # K row bytes = D*bits//8
    assert mVq.shape[-1] % 16 == 0, mVq.shape   # V row bytes = PS*bits//8
    return mKq, mVq, mScale


def _dequant_pools_plane(mKq, mVq, mScale, bits):
    """Inverse of _build_pools_plane -> (k_hat, v_hat) [NP,HKV,PS,D] f32 (STORAGE-MATCHED values)."""
    qk = _unpack_rows_plane(mKq, bits, D).float()
    qv = _unpack_rows_plane(mVq, bits, PS).float().transpose(-1, -2)   # back to [NP,HKV,PS,D]
    sk, zk, sv, zv = mScale.unbind(-1)
    k_hat = qk * sk[..., None] + zk[..., None]
    v_hat = qv * sv[..., None] + zv[..., None]
    return k_hat, v_hat


def pack_rows(codes, bits):
    """codes [..., R, E] (E % 16 == 0) -> uint8 [..., R, E*bits//8], LUT byte order.
    The bit layout is fixed by the kernel's byte-lane extraction and is checked by
    `check_k_store_perm_f4` and `check_k_store_perm_i4`; do not change it without
    re-validating those:
      int4: sequential nibbles — byte i holds codes 2i (low nibble), 2i+1 (high nibble); the
            raw-code compaction's masks split a u32's even/odd nibbles (this layout was
            originally the PRMT selector format of the retired two-table LUT dequant).
      int2: code a of each 16-code group sits at bit 16*(a//8) + 4*(a%4) + 2*((a%8)//4) of a
            u32, so (w >> (16*(a//2)+2*(a%2))) & 0x3333 gathers codes 4a..4a+3 one per nibble
            (the OR-magic int2 dequant's selector nibbles).
    The packer takes a flat [R, E] tensor; this wrapper flattens any leading batch
    dims together with R (the packing is per-row, batch-independent) so it drops into
    `build_pools` the same way `_pack_rows_plane` drops into `_build_pools_plane`.
    """
    *lead, R, E = codes.shape
    assert E % 16 == 0, E
    flat = codes.reshape(-1, E)
    if bits == 4:
        packed = (flat[:, 0::2] | (flat[:, 1::2] << 4)).contiguous()
    elif bits == 2:
        # On the codes' device: this is pure integer shifting and or-ing, so the bytes it
        # produces are the same on every device, and building a GPU fixture's pools on
        # the GPU is what keeps a large cell from spending its time here. IN ROW CHUNKS,
        # because the int64 widening is eight bytes per code: expanded whole, the
        # B=128/KV=32768 pool would need 32 GiB of scratch beside the codes themselves.
        packed = torch.empty(flat.shape[0], E // 4, dtype=torch.uint8, device=flat.device)
        rows_per_chunk = 1 << 16
        for start in range(0, flat.shape[0], rows_per_chunk):
            rows = flat[start:start + rows_per_chunk]
            c = rows.reshape(rows.shape[0], -1, 16).to(torch.int64)
            w = torch.zeros(c.shape[:2], dtype=torch.int64, device=c.device)
            for i in range(16):
                w |= c[..., i] << (16 * (i // 8) + 4 * (i % 4) + 2 * ((i % 8) // 4))
            packed[start:start + rows.shape[0]] = (
                w.to(torch.uint32).view(torch.uint8).reshape(rows.shape[0], -1))
            del c, w
    else:
        raise ValueError(bits)
    return packed.reshape(*lead, R, E * bits // 8)


def unpack_rows(packed, bits, E):
    """Inverse of `pack_rows`. The round trip is exact at both widths."""
    *lead, R, B = packed.shape
    flat = packed.reshape(-1, B)
    if bits == 4:
        b = flat.to(torch.int16)
        codes = torch.stack([b & 0xF, b >> 4], -1).reshape(flat.shape[0], -1).to(torch.uint8)
    elif bits == 2:
        w = flat.reshape(flat.shape[0], -1, 4).view(torch.uint32).squeeze(-1).to(torch.int64)
        out = [(w >> (16 * (i // 8) + 4 * (i % 4) + 2 * ((i % 8) // 4))) & 0x3 for i in range(16)]
        codes = torch.stack(out, -1).reshape(flat.shape[0], -1).to(torch.uint8)
    else:
        raise ValueError(bits)
    return codes.reshape(*lead, R, E)


def v_store_perm_sm100(bits):
    """Legacy MN-major V operand-ring store order, matching the kernel's
    dequant-warpgroup V path in `_sm100/int_fold_decode.py`.

    Kernel-side contract being satisfied: dequant thread (col = t%64, half = t//64) of a
    V code stage (one page) loads the packed flat byte run
    [col*(128*bits//8) + half*(64*bits//8), col*(128*bits//8) + (half+1)*(64*bits//8))
    (32B at int4 / 16B at int2), byte-maps it (int4 raw-code compaction / int2 OR-magic —
    both byte-ORDER-PRESERVING), and stores dims [64*half, 64*half+64) of token column `col`
    as 4x STS.128 (MN-major stage: dims contiguous). So stored flat code position
    j = col*128 + d must hold q[dim=d, tok=col] -> token-major: p[j] = (j%D)*PS + j//D.

    int2 extra: `pack_rows` puts input slot i of each 16-code group at bit
    b(i) = 16*(i//8) + 4*(i%4) + 2*((i%8)//4); the kernel's OR-magic
    (w >> 2m) & 0x03030303 reads byte lane l from bits 8l+2m and assigns it local dim
    delta = 4m + l. Solving b(i) = 8l + 2m gives delta(i) = 4*((b%8)//2) + b//8,
    i.e. input slot i must hold the code of local dim delta(i):
    delta = [0,8,1,9,4,12,5,13,2,10,3,11,6,14,7,15]. Verified byte-exact on host
    (all 128 threads x both paths) and on-GPU by the G3 stage dumps."""
    j = torch.arange(D * PS)
    if bits == 4:
        jj = j
    elif bits == 2:
        delta = torch.tensor([0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15])
        jj = 16 * (j // 16) + delta[j % 16]
    else:
        raise ValueError(bits)
    return (jj % D) * PS + (jj // D)


#: The int4 8-slot store shuffle, as position -> logical within each packed
#: word's 8 codes: stored nibble sequence [q0, q4, q1, q5, q2, q6, q3, q7], so
#: `w & 0x0F0F0F0F` yields bytes [q0..q3] and `(w >> 4) & 0x0F0F0F0F` yields
#: [q4..q7] with NO PRMT. Unlike int2's sigma it is NOT an involution
#: (t8[t8] = [0,2,4,6,1,3,5,7] != identity), so store and reference use this
#: array and its argsort respectively rather than one function.
_T8_I4 = (0, 4, 1, 5, 2, 6, 3, 7)


def _reduction_store_perm(bits, length):
    """Within-word order for the two-mask expansion along an operand's K axis."""
    assert bits in (2, 4), f"store permutation requires bits in (2, 4); got {bits}"
    group = 32 // bits
    assert length % group == 0
    slots = (_T8_I4 if bits == 4 else
             tuple((i & 3) | ((i & 4) << 1) | ((i & 8) >> 1)
                   for i in range(16)))
    x = torch.arange(length)
    return group * (x // group) + torch.tensor(slots)[x % group]


def v_store_perm_sm100_f4(bits):
    """MODE-KEYED V store order for the main kernel's reinterpreting feeds.
    (The ``_f4`` name is historical: it comes from the int2 E2M1 feed that
    introduced this order. At 4 bits the same order serves the E4M3 raw-code
    feed.)

    int2 (dequant_int2_to_f4): dim-major runs (dim d's 64 tokens = one
    16B LDS.128), pre-interleaved so the kernel's 2->4 expand is
    lo = w & 0x33333333 (tokens 16w..16w+7), hi = (w >> 2) & 0x33333333 (16w+8..15).
    Composed against pack_rows's int2 slot->bit map b(i) via the self-inverse
    16-slot sub-perm sigma(i) = (i&3) | ((i&4)<<1) | ((i&8)>>1). A pool built with
    this permutation is valid only for kernels compiled with
    ``dequant_int2_to_f4=1``. ``check_k_store_perm_f4`` is the enforcing oracle;
    do not edit one without the other.

    int4: the same dim-major base order, with `_T8_I4` along TOKENS inside
    each packed word. The two masks produce tokens 8w..8w+3 and 8w+4..8w+7
    as E4M3 bytes for a K-major V operand. This replaces the old token-major
    INT4 pool without changing its shape: old pools MUST be rebuilt.
    Checked by `_gate_i4_v_perm` below.

    Source frame (checked by `_gate_f4_v_perm` against `build_pools`'s V leg): the array
    this perm indexes into is the DIM-MAJOR flat V view
    `qv_t.reshape(D * PS)` (qv_t = q^T, shape [D, PS], i.e. flat position k = d*PS + tok)
    -- the SAME frame `v_store_perm_sm100` composes against. So the closed form's flat
    output position (d, w, i) sources from dim-major position d*PS + tok, NOT tok*D + d."""
    return (torch.arange(D)[:, None] * PS
            + _reduction_store_perm(bits, PS)[None, :]).reshape(-1)


def k_store_perm_sm100_f4(bits):
    """MODE-KEYED K CHANNEL order, the counterpart of `v_store_perm_sm100_f4`
    for the K pool. (The _f4 name is historical, from the int2 E2M1 feed that
    introduced it; the bits==4 branch serves the E4M3 raw-code feed.)

    int2 (dequant_int2_to_f4): the SAME self-inverse 16-slot sub-perm sigma
    that the V order already composes, applied along D within each 16-channel
    group. K rows stay token-major and every row keeps its byte count, so
    pages, TMA boxes and the block table are untouched; only which channel
    each packed slot holds changes.

    WHY IT WORKS. `pack_rows` puts code i of a u32 at
    b(i) = 16*(i//8) + 4*(i%4) + 2*((i%8)//4), a bijection onto the even bits
    0..30. The b4x16_p64 K image wants out0 nibble j = code j and out1 nibble
    j = code 8+j. Reading `w & 0x33333333` leaves each kept pair in place, so
    out0 nibble j sources bit 4j and out1 nibble j sources bit 4j+2; sending
    logical channel j to the slot with b = 4j and channel 8+j to the slot with
    b = 4j+2 is exactly sigma. The deposit then collapses from the 7-op
    two-part gather to V's 3-op two masks.

    int4 (raw-code feed): the 8-slot shuffle `_T8_I4` within each
    8-channel group (one packed u32). The stored nibble sequence becomes
    [q0, q4, q1, q5, q2, q6, q3, q7], so `w & 0x0F0F0F0F` already IS the
    operand word for channels 8w..8w+3 and `(w >> 4) & 0x0F0F0F0F` for
    8w+4..7, retiring both PRMTs (5 ops per 8 codes -> 3). NOT an involution;
    references invert with `_k_store_perm_inverse`.

    Gates: `check_k_store_perm_f4` (int2) and `check_k_store_perm_i4` (int4)
    are the enforcing oracles. A pool built without the perm and read by the
    mask deposit is wrong at every K channel and raises nothing, so the
    pairing is settled in one place: `IntDecodeSm100.k_pool_is_permuted`
    states which configurations carry this order."""
    return _reduction_store_perm(bits, D)


def check_k_store_perm_f4():
    """Prove the permuted pool read with two masks equals the unpermuted pool
    read with the production gather, as operand WORDS rather than as inputs.

    This is the K counterpart of `_gate_f4_v_perm`, and it is why the permutation may be trusted.

    IT IS EXHAUSTIVE, not a sample, and the second loop is what makes it so. Every
    output bit of both deposits depends on exactly one input code position, since
    both are AND/OR/shift over disjoint bit fields, so agreement on a BASIS implies
    agreement everywhere: the basis sweep below sets one position to one code value
    with the rest zero, over all 16 positions and all 4 values. The random vectors
    are kept as a cross-check that the basis argument was applied to the right
    functions, since a sampling test alone would only license "for these codes",
    which is weaker than the claim this oracle is cited for."""
    perm = k_store_perm_sm100_f4(2)

    def deposits(codes):
        """(gather from the logical pool, masks from the permuted pool), per group."""
        wp = (pack_rows(codes.reshape(1, 1, D), 2).reshape(-1)
              .view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
        wq = (pack_rows(codes[:, perm].reshape(1, 1, D), 2).reshape(-1)
              .view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
        out = []
        for grp in range(D // 16):
            a, b = int(wp[grp]), int(wq[grp])
            out.append((
                [((a & 0x3333) | ((a << 14) & 0x33330000)) & 0xFFFFFFFF,
                 ((a >> 16) & 0x3333) | ((a >> 2) & 0x33330000)],
                [b & 0x33333333, (b >> 2) & 0x33333333]))
        return out

    # BASIS: one position, one value, everything else zero.
    for pos in range(D):
        for val in range(1, 4):
            codes = torch.zeros((1, D), dtype=torch.uint8)
            codes[0, pos] = val
            for grp, (gather, masks) in enumerate(deposits(codes)):
                assert gather == masks, ("basis", pos, val, grp,
                                         [hex(x) for x in gather],
                                         [hex(x) for x in masks])

    g = torch.Generator().manual_seed(0)
    for _ in range(64):
        codes = torch.randint(0, 4, (1, D), generator=g, dtype=torch.uint8)
        wp = (pack_rows(codes.reshape(1, 1, D), 2).reshape(-1)
              .view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
        wq = (pack_rows(codes[:, perm].reshape(1, 1, D), 2).reshape(-1)
              .view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
        for grp in range(D // 16):
            a, b = int(wp[grp]), int(wq[grp])
            gather = [((a & 0x3333) | ((a << 14) & 0x33330000)) & 0xFFFFFFFF,
                      ((a >> 16) & 0x3333) | ((a >> 2) & 0x33330000)]
            masks = [b & 0x33333333, (b >> 2) & 0x33333333]
            assert gather == masks, (grp, [hex(x) for x in gather],
                                     [hex(x) for x in masks])
    return True


def check_k_store_perm_i4():
    """Prove the int4-permuted pool read with two masks equals the logical pool
    read with the shipped 5-op compaction (masks + immediate-selector PRMTs),
    as operand WORDS. The int4 counterpart of `check_k_store_perm_f4`, and it
    is exhaustive by the same basis argument: every output bit of both
    deposits depends on exactly one input nibble (AND/shift/byte-move over
    disjoint fields), so agreement on one-hot inputs over all positions and
    all 15 nonzero code values implies agreement everywhere. The random
    vectors cross-check that the basis argument was applied to the right
    functions."""
    perm = k_store_perm_sm100_f4(4)

    def prmt_pair(t0, t1):
        # PRMT(t0, t1, 0x5140) and PRMT(t0, t1, 0x7362) as byte gathers.
        b = lambda x, i: (x >> (8 * i)) & 0xFF
        return [b(t0, 0) | (b(t1, 0) << 8) | (b(t0, 1) << 16) | (b(t1, 1) << 24),
                b(t0, 2) | (b(t1, 2) << 8) | (b(t0, 3) << 16) | (b(t1, 3) << 24)]

    def deposits(codes):
        """(5-op compaction from the logical pool, masks from the permuted
        pool), per 8-code word."""
        wp = (pack_rows(codes.reshape(1, 1, D), 4).reshape(-1)
              .view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
        wq = (pack_rows(codes[:, perm].reshape(1, 1, D), 4).reshape(-1)
              .view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
        out = []
        for grp in range(D // 8):
            a, b = int(wp[grp]), int(wq[grp])
            out.append((
                prmt_pair(a & 0x0F0F0F0F, (a >> 4) & 0x0F0F0F0F),
                [b & 0x0F0F0F0F, (b >> 4) & 0x0F0F0F0F]))
        return out

    # BASIS: one position, one value, everything else zero.
    for pos in range(D):
        for val in range(1, 16):
            codes = torch.zeros((1, D), dtype=torch.uint8)
            codes[0, pos] = val
            for grp, (gather, masks) in enumerate(deposits(codes)):
                assert gather == masks, ("basis", pos, val, grp,
                                         [hex(x) for x in gather],
                                         [hex(x) for x in masks])

    g = torch.Generator().manual_seed(0)
    for _ in range(64):
        codes = torch.randint(0, 16, (1, D), generator=g, dtype=torch.uint8)
        for grp, (gather, masks) in enumerate(deposits(codes)):
            assert gather == masks, (grp, [hex(x) for x in gather],
                                     [hex(x) for x in masks])
    return True


def _check_k_store_perm(p, bits=2):
    """k_store_perm contract: an int tensor of shape (D,) that IS a permutation
    of range(D), acting within each packed-word channel group (16 channels at
    int2, 8 at int4 -- one u32 of codes either way).

    The group clause is not decoration. A code moved across a group crosses a
    packed-word boundary, so the kernel's mask read would gather channels
    belonging to another word and the pool would be silently wrong. The
    default bits=2 serves `_k_store_perm_inverse`'s bits-agnostic callers; an
    int4 perm passes it too (within-8 implies within-16), so the tight check
    runs at the pool builders, which know bits."""
    assert bits in (2, 4), f"k_store_perm is defined for bits in (2, 4), got {bits}"
    group = 32 // bits
    assert torch.is_tensor(p) and p.shape == (D,), getattr(p, "shape", p)
    assert torch.equal(torch.sort(p.long()).values, torch.arange(D)), (
        "not a permutation of range(D)")
    groups = p.long() // group == torch.arange(D) // group
    assert bool(groups.all()), (
        f"perm moves a code across a {group}-channel group")


def _k_store_perm_inverse(k_store_perm, bits=2):
    # The default bits=2 keeps the bits-agnostic reference callers unchanged;
    # an int4 perm passes the looser 16-group check too (within-8 implies
    # within-16), so the default under-constrains but never mis-inverts.
    _check_k_store_perm(k_store_perm, bits)
    inv = torch.empty_like(k_store_perm)
    inv[k_store_perm] = torch.arange(k_store_perm.numel(),
                                     device=k_store_perm.device)
    return inv


def build_pools(k, v, bits, v_store_perm=None, k_store_perm=None):
    """Pack a bf16 KV cache into the pools the SM100 kernel reads.

    ``k`` and ``v`` are ``[pages, page_size, kv_heads, head_dim]`` and are
    quantized token-wise, one scale and offset per token over the head dimension.
    The returned ``(codes_k, codes_v, scales)`` are what ``run()`` takes as
    ``k_cache``, ``v_cache`` and ``scale_pool``.

    The integer values are stored in an order the kernel's byte-lane extraction
    requires and the pool bytes do not reveal, so pass the permutations rather
    than reproducing them: ``k_store_perm=k_store_perm_sm100_f4(bits)`` and
    ``v_store_perm=v_store_perm_sm100_f4(bits)``. ``dequant_pools`` inverts them.

    ``k_store_perm`` (default None = identity = K rows token-major and unpermuted)
    reorders the K CHANNEL positions before packing, which is what the mask
    deposits read (int2 f4 at either K axis; int4 everywhere since
    see `k_store_perm_sm100_f4` and ask
    `IntDecodeSm100.k_pool_is_permuted` whether a given configuration needs it.
    `v_store_perm` (default None = identity = plain
    dim-major V^T rows) reorders the V code positions BEFORE `pack_rows`:
      - (PS,) tensor: token-axis permutation within each V row (kept for the
        plumbing regression);
      - (D*PS,) tensor: full per-page code-position permutation — stored flat position j
        holds the code at dim-major flat position perm[j]. This is what the SM100 kernel
        consumes. Resolve it through `v_store_perm_sm100_f4(bits)`;
        the main INT2/FP4 and INT4 feeds use `v_store_perm_sm100_f4(bits)`."""
    NP, ps, HKV, d = k.shape
    assert ps == PS and d == D
    qk, sk, zk = quant_tokenwise(k.permute(0, 2, 1, 3), bits)   # [NP,HKV,PS,D]
    qv, sv, zv = quant_tokenwise(v.permute(0, 2, 1, 3), bits)
    return pack_pools(qk, (sk, zk), qv, (sv, zv), bits,
                          v_store_perm=v_store_perm, k_store_perm=k_store_perm)


def pack_pools(qk, k_sz, qv, v_sz, bits, v_store_perm=None, k_store_perm=None):
    """The packing half of `build_pools`, from codes already quantised.

    `qk`/`qv` are uint8 codes [NP,HKV,PS,D] and `k_sz`/`v_sz` the (s, z) pairs
    `quant_tokenwise` returned for them. Everything here is a permutation, an integer
    shift or an or, so the bytes are the same on every device, and the work runs where
    the codes are; the pools come back there too. A fixture that has to keep its
    anchored bytes quantises on the CPU and hands the codes to the GPU before
    calling this.
    """
    dev = qk.device
    sk, zk = k_sz
    sv, zv = v_sz
    qv_t = qv.transpose(-1, -2).contiguous()                     # [NP,HKV,D,PS] dim-major
    if v_store_perm is not None:
        _check_v_store_perm(v_store_perm)
        if v_store_perm.numel() == PS:
            qv_t = qv_t[..., v_store_perm.to(dev)]
        else:
            lead = qv_t.shape[:-2]
            qv_t = (qv_t.reshape(*lead, D * PS)[..., v_store_perm.long().to(dev)]
                    .reshape(*lead, D, PS))
    if k_store_perm is not None:
        _check_k_store_perm(k_store_perm, bits)
        qk = qk[..., k_store_perm.long().to(dev)]
    mKq = pack_rows(qk, bits).contiguous()                   # [NP,HKV,PS,D*bits//8]
    mVq = pack_rows(qv_t, bits).contiguous()                 # [NP,HKV,D,PS*bits//8]
    mScale = (torch.stack([sk, zk, sv, zv], -1).float().contiguous()
              .to(dev))                                          # [NP,HKV,PS,4]
    assert mKq.is_contiguous() and mVq.is_contiguous() and mScale.is_contiguous()
    assert mKq.shape[-1] % 16 == 0, mKq.shape   # K row bytes = D*bits//8
    assert mVq.shape[-1] % 16 == 0, mVq.shape   # V row bytes = PS*bits//8
    return mKq, mVq, mScale


def _check_v_store_perm(p):
    """v_store_perm contract: int tensor of shape (PS,) (legacy token perm) or (D*PS,)
    (full per-page code-position perm) that IS a permutation of its index range."""
    assert torch.is_tensor(p) and p.shape in ((PS,), (D * PS,)), getattr(p, "shape", p)
    n = p.numel()
    assert torch.equal(torch.sort(p.long()).values, torch.arange(n)), "not a permutation"


def dequant_pools(mKq, mVq, mScale, bits, v_store_perm=None,
                      k_store_perm=None):
    """Invert `build_pools`: (k_hat, v_hat) ``[pages, kv_heads, page_size, head_dim]``
    f32, the values the kernel reads. ``v_store_perm`` must match what `build_pools`
    was called with
    (default None = identity)."""
    qk = unpack_rows(mKq, bits, D).float()
    if k_store_perm is not None:
        qk = qk[..., _k_store_perm_inverse(k_store_perm).long()]
    qv_t = unpack_rows(mVq, bits, PS).float()                # [NP,HKV,D,PS], perm order
    if v_store_perm is not None:
        _check_v_store_perm(v_store_perm)
        inv = torch.empty_like(v_store_perm)
        inv[v_store_perm] = torch.arange(v_store_perm.numel(), device=v_store_perm.device)
        if v_store_perm.numel() == PS:
            qv_t = qv_t[..., inv]
        else:
            lead = qv_t.shape[:-2]
            qv_t = (qv_t.reshape(*lead, D * PS)[..., inv.long()]
                    .reshape(*lead, D, PS))
    qv = qv_t.transpose(-1, -2)                                   # [NP,HKV,PS,D]
    sk, zk, sv, zv = mScale.unbind(-1)
    k_hat = qk * sk[..., None] + zk[..., None]
    v_hat = qv * sv[..., None] + zv[..., None]
    return k_hat, v_hat


# ---------------------------------------------------------------------------
# Quant-granularity flex pools: per-tensor (axis, group)
# configs — token-wise {D, D/2} / channel-wise token-groups {64, 128, 256},
# K and V independent. Codes keep the LUT pack (pack_rows; v_store_perm
# applies unchanged); the scale pool becomes the SECTIONED flat block of
# fosca.quant_layout.quant_block_offsets:
#   [K ring | V ring | K channel | V channel] per (page, kv-head), sdtype.
# Ring sections: ("token", D) -> [N,2] (s,z)/token; ("token", D/2) -> [N,4]
# (s_lo,z_lo,s_hi,z_hi)/token (lo = d < D/2). Channel sections: [2,D] row-0
# s_d / row-1 z_d — the page's group slice, DUPLICATED per page when Gt > N
# (page-local dequant then equals logical-stream dequant, which is what makes
# the storage-matched flex reference page-local — see dequant_pools_flex).
# Channel-wise quantization does NOT commute with the page scatter (a group
# spans Gt logical tokens), so the flex builder takes LOGICAL page streams
# and scatters itself, unlike build_pools's physical-page contract.
# ---------------------------------------------------------------------------


def _quant_ref_mod():
    from . import quant_ref

    return quant_ref


def _grouped_sz(x, bits, quant):
    """x [B,HKV,S,D] -> (q uint8 [B,HKV,S,D], (s, z)) per the (axis, group)
    config, via quant_ref.quantize_grouped. Token-wise G is the CHANNEL group
    (D or D/2); channel-wise G is the TOKEN group Gt."""
    axis, g = quant
    q, s, z = _quant_ref_mod().quantize_grouped(x, bits, axis, g)
    return q.to(torch.uint8), (s, z)


def build_scale_block_flex(k_sz, v_sz, k_quant, v_quant, sdtype=torch.float32):
    """Per-page sectioned scale blocks [B, HKV, NT, block_tot] (sdtype) from
    quantize_grouped (s, z) pairs in LOGICAL token order (S_pad = NT * PS):
      ("token", D):    s/z [B,HKV,S_pad]
      ("token", D/2):  s/z [B,HKV,S_pad,2]  ([..., 0] = lo half d < D/2)
      ("channel", Gt): s/z [B,HKV,S_pad//Gt,D]
    Layout mirrors _sm90.serving_cache.make_serving_cache_flex exactly."""
    from fosca.quant_layout import quant_block_offsets

    off = quant_block_offsets(k_quant, v_quant, N=PS, D=D)
    s_ref = (k_sz if k_quant[0] == "token" else v_sz)[0]
    B, HKV = s_ref.shape[0], s_ref.shape[1]

    def ring_block(qc, sz):
        axis, g = qc
        if axis == "channel":
            return None
        s, z = sz
        if g == D:      # [B,HKV,S_pad] -> [B,HKV,NT,N*2] (s,z interleaved)
            blk = torch.stack([s, z], dim=-1)
            return blk.reshape(B, HKV, -1, PS * 2).float()
        # g == D//2: (s_lo, z_lo, s_hi, z_hi) per token
        blk = torch.stack([s[..., 0], z[..., 0], s[..., 1], z[..., 1]], dim=-1)
        return blk.reshape(B, HKV, -1, PS * 4).float()

    def ch_block(qc, sz):
        axis, g = qc
        if axis != "channel":
            return None
        s, z = sz                                   # [B,HKV,S_pad//Gt,D]
        n_grp = s.shape[2]
        blk = torch.stack([s, z], dim=-2).float().reshape(B, HKV, n_grp, 2 * D)
        NT = n_grp * g // PS
        pages = torch.arange(NT) * PS // g          # page p -> its group
        return blk[:, :, pages.clamp(max=n_grp - 1)]  # [B,HKV,NT,2D]

    if off["legacy"]:
        sk, zk = k_sz
        sv, zv = v_sz
        blk = torch.stack([sk, zk, sv, zv], dim=-1).float()  # [B,HKV,S_pad,4]
        return blk.reshape(B, HKV, -1, PS * 4).to(sdtype)

    parts = []
    for qc, sz in ((k_quant, k_sz), (v_quant, v_sz)):
        rb = ring_block(qc, sz)
        if rb is not None:
            parts.append(rb)
    for qc, sz in ((k_quant, k_sz), (v_quant, v_sz)):
        cb = ch_block(qc, sz)
        if cb is not None:
            parts.append(cb)
    NT = min(p.shape[2] for p in parts)
    block = torch.cat([p[:, :, :NT] for p in parts], dim=-1)
    assert block.shape[-1] == off["block_tot"], (block.shape, off)
    return block.to(sdtype)


def build_pools_flex(k_log, v_log, bits, k_quant=("token", 128),
                     v_quant=("token", 128), sdtype=torch.float32,
                     v_store_perm=None, perm=None, k_store_perm=None,
                     pack_device=None):
    """Flex-granularity pool builder. `k_log`/`v_log`: [B, NT, PS, HKV, D]
    bf16 LOGICAL page streams (padded to whole pages; NT * PS must be a
    multiple of every channel token-group in use). Quantizes the logical
    stream per (axis, group), packs codes per page with `pack_rows`
    (V through `v_store_perm`, same contract as `build_pools`), builds
    the sectioned scale blocks, and scatters pages into a physical pool via
    `perm` (default: fresh randperm over B*NT).

    Quantisation runs on the inputs' device, because the codes depend on it
    (see `quant_tokenwise`); the permutations, the packing and the scatter are
    integer work that runs on `pack_device` (default: the inputs' device) and
    the pools come back there. `aux` stays on the inputs' device.

    Returns (mKq, mVq, mScale, block_table, aux):
      mKq  uint8 [NP,HKV,PS,D*bits//8], mVq uint8 [NP,HKV,D,PS*bits//8]
      mScale sdtype [NP,HKV,block_tot]  (flat sectioned block)
      block_table int32 [B, NT]         (logical page -> physical page)
      aux = dict(k_sz=..., v_sz=..., qk=..., qv=...) for references."""
    B, NT, ps, HKV, d = k_log.shape
    assert ps == PS and d == D
    for qc in (k_quant, v_quant):
        if qc[0] == "channel":
            assert (NT * PS) % qc[1] == 0, (
                f"channel-wise {qc}: NT*PS = {NT * PS} must be a multiple of "
                f"the token group (pad the logical page stream)")
    x_k = k_log.permute(0, 3, 1, 2, 4).reshape(B, HKV, NT * PS, D)
    x_v = v_log.permute(0, 3, 1, 2, 4).reshape(B, HKV, NT * PS, D)
    qk, k_sz = _grouped_sz(x_k, bits, k_quant)
    qv, v_sz = _grouped_sz(x_v, bits, v_quant)

    block = build_scale_block_flex(k_sz, v_sz, k_quant, v_quant, sdtype=sdtype)
    assert block.shape[2] == NT, (block.shape, NT)

    dev = qk.device if pack_device is None else torch.device(pack_device)
    qk_p = qk.reshape(B, HKV, NT, PS, D).to(dev)
    if k_store_perm is not None:
        # A (D,) CHANNEL permutation applied before packing, so the kernel's K
        # deposit reads two masks instead of a two-part gather. Rows keep their
        # length, so nothing about pages, TMA boxes or the block table changes.
        _check_k_store_perm(k_store_perm, bits)
        qk_p = qk_p[..., k_store_perm.long().to(dev)]
    qv_p = qv.reshape(B, HKV, NT, PS, D).to(dev).transpose(-1, -2).contiguous()
    if v_store_perm is not None:
        _check_v_store_perm(v_store_perm)
        if v_store_perm.numel() == PS:
            qv_p = qv_p[..., v_store_perm.to(dev)]
        else:
            lead = qv_p.shape[:-2]
            qv_p = (qv_p.reshape(*lead, D * PS)[..., v_store_perm.long().to(dev)]
                    .reshape(*lead, D, PS))
    kq_pages = pack_rows(qk_p, bits)            # [B,HKV,NT,PS,kb]
    vq_pages = pack_rows(qv_p, bits)            # [B,HKV,NT,D,vb]

    NP = B * NT
    if perm is None:
        perm = torch.randperm(NP)
    perm = perm.to(torch.long)
    assert perm.numel() == NP and torch.equal(
        torch.sort(perm).values, torch.arange(NP)
    ), "perm must be a permutation of range(B*NT)"
    perm_dev = perm.to(dev)
    mKq = torch.empty(NP, HKV, PS, kq_pages.shape[-1], dtype=torch.uint8, device=dev)
    mVq = torch.empty(NP, HKV, D, vq_pages.shape[-1], dtype=torch.uint8, device=dev)
    mScale = torch.empty(NP, HKV, block.shape[-1], dtype=sdtype, device=dev)
    mKq[perm_dev] = kq_pages.permute(0, 2, 1, 3, 4).reshape(NP, HKV, PS, -1)
    mVq[perm_dev] = vq_pages.permute(0, 2, 1, 3, 4).reshape(NP, HKV, D, -1)
    mScale[perm_dev] = block.permute(0, 2, 1, 3).reshape(NP, HKV, -1).to(dev)
    block_table = perm.reshape(B, NT).to(torch.int32)
    aux = dict(k_sz=k_sz, v_sz=v_sz, qk=qk, qv=qv)
    return (mKq.contiguous(), mVq.contiguous(), mScale.contiguous(),
            block_table, aux)


def dequant_pools_flex(mKq, mVq, mScale, bits, k_quant, v_quant,
                       v_store_perm=None, k_store_perm=None):
    """Storage-matched PAGE-LOCAL dequant of a flex pool -> (k_hat, v_hat)
    [NP,HKV,PS,D] f32. Every page's block carries its OWN copy of the channel
    section (duplicated per page at build), so page-local dequant equals
    logical-stream dequant exactly; scales are read back at pool precision
    (a bf16 pool dequantizes with the bf16-ROUNDED values — the reference
    convention the flex gates need)."""
    from fosca.quant_layout import quant_block_offsets

    off = quant_block_offsets(k_quant, v_quant, N=PS, D=D)
    if off["legacy"]:
        return dequant_pools(mKq, mVq, mScale.reshape(*mScale.shape[:2], PS, 4).float(),
                                 bits, v_store_perm=v_store_perm,
                                 k_store_perm=k_store_perm)
    qk = unpack_rows(mKq, bits, D).float()
    if k_store_perm is not None:
        qk = qk[..., _k_store_perm_inverse(k_store_perm).long()]
    qv_t = unpack_rows(mVq, bits, PS).float()
    if v_store_perm is not None:
        _check_v_store_perm(v_store_perm)
        inv = torch.empty_like(v_store_perm)
        inv[v_store_perm] = torch.arange(v_store_perm.numel(),
                                         device=v_store_perm.device)
        lead = qv_t.shape[:-2]
        if v_store_perm.numel() == PS:
            qv_t = qv_t[..., inv]
        else:
            qv_t = (qv_t.reshape(*lead, D * PS)[..., inv.long()]
                    .reshape(*lead, D, PS))
    qv = qv_t.transpose(-1, -2)
    blk = mScale.float()

    def hat(q, quant, ring_off, ch_off):
        axis, g = quant
        if axis == "channel":
            sec = blk[..., ch_off:ch_off + 2 * D].reshape(*blk.shape[:2], 2, D)
            s_d, z_d = sec[..., 0, :], sec[..., 1, :]        # [NP,HKV,D]
            return q * s_d[..., None, :] + z_d[..., None, :]
        if g == D:
            ring = blk[..., ring_off:ring_off + 2 * PS].reshape(
                *blk.shape[:2], PS, 2)
            s, z = ring[..., 0], ring[..., 1]                # [NP,HKV,PS]
            return q * s[..., None] + z[..., None]
        # token D/2: (s_lo, z_lo, s_hi, z_hi) per token
        ring = blk[..., ring_off:ring_off + 4 * PS].reshape(
            *blk.shape[:2], PS, 4)
        lo = q[..., :D // 2] * ring[..., 0, None] + ring[..., 1, None]
        hi = q[..., D // 2:] * ring[..., 2, None] + ring[..., 3, None]
        return torch.cat([lo, hi], dim=-1)

    k_hat = hat(qk, k_quant, off["k_ring"], off["k_ch"])
    v_hat = hat(qv, v_quant, off["v_ring"], off["v_ch"])
    return k_hat, v_hat


def ref_paged_attention_fp8q_flex(q_fp8, q_scale, mKq, mVq, mScale, bits,
                                  k_quant, v_quant, block_table, seq_lens,
                                  sm_scale, v_store_perm=None,
                                  k_store_perm=None):
    """fp8-Q storage-matched paged reference over a FLEX pool (the fp8q_fast
    einsum structure with the grouped page-local dequant swapped in)."""
    B, HQ, d = q_fp8.shape
    assert d == D
    q = q_fp8.float() / q_scale
    k_hat, v_hat = dequant_pools_flex(mKq, mVq, mScale, bits, k_quant, v_quant,
                                      v_store_perm=v_store_perm,
                                      k_store_perm=k_store_perm)
    HKV = k_hat.shape[1]
    g = HQ // HKV
    o = torch.zeros(B, HQ, D)
    for b in range(B):
        T = int(seq_lens[b])
        pages = block_table[b, : (T + PS - 1) // PS].long()
        kb = k_hat[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]
        vb = v_hat[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]
        qb = q[b].reshape(HKV, g, D)
        P = torch.softmax(torch.einsum("hgd,htd->hgt", qb, kb) * sm_scale, -1)
        o[b] = torch.einsum("hgt,htd->hgd", P, vb).reshape(HQ, D)
    return o


def quantize_q_fp8(q):
    """Per-tensor absmax fp8 Q, in the operation order the kernel's scale fold assumes:
        qs    = 448 / amax(|q.float()|).clamp_min(1e-30)     (f32 MULTIPLY-scale)
        q_fp8 = e4m3(clamp(q.float() * qs, ±448))            (multiply, NOT divide-by-s)
        Q_eff = q_fp8.float() / qs                           (dequant DIVIDES by qs)
    Returns (q_fp8, q_scale) with q_scale = qs, the MULTIPLY-scale, as a python float.
    The algebraically-equal divide form (s = amax/448; q/s; codes*s) rounds differently in
    f32 (~12% of elements land 1 fp8 ULP off) and MUST NOT be used: the Task-4 G3.5
    folded-S gate (rel <= 1e-5 vs this oracle) requires op-order identity with the sim.
    A caller folding the Q dequant into the kernel's scale_s knob uses
    scale_s = sm_scale / (q_scale * k_scale), which is the natural form under
    multiply-scales."""
    qs = E4M3_MAX / q.float().abs().amax().clamp_min(1e-30)
    q_fp8 = (q.float() * qs).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q_fp8, qs.item()


def dequant_bf16_fma_e4m3(codes, s, z):
    """THE host model of `IntDecodeSm100(fold_scale_zero=False)`'s operand.

    One implementation, used by the operand-matched reference
    (`ref_paged_attention_fp8q(kv_round="bf16_fma")`); the byte-exact stage
    Kept as the single copy because two hand-transcribed copies of a numerics
    model drift.

    The kernel dequantizes element-wise with ONE `fma.rn.bf16x2` per element
    pair and then converts to the e4m3 operand, so there are exactly three
    roundings and their ORDER matters:

      1. the FMA's operands: s, z -> bf16 (the kernel's `dq_coeffs`, once per
         token). This is the dominant term and the reason a reference that
         rounds only the FINAL value is NOT operand-matched: with mean |err|
         ~0.005 on N(0,1) rows it is a different perturbation than rounding the
         product-sum, and pretending otherwise made the "operand-matched"
         reference disagree with the kernel MORE than the exact-dequant one did.
      2. q*s + z -> bf16, ONE rounding: an FMA is the exactly-rounded
         product-sum, so the model computes it in f64 and rounds once. Doing the
         arithmetic in bf16 would round twice and disagree near bf16 ties.
      3. bf16 -> e4m3, RNE + satfinite (`cvt.rn.satfinite.e4m3x2.f32` after the
         free bf16 -> f32 widening).

    `codes` holds the raw q (the kernel's feed carries q itself: the magic's
    base-4 offset is subtracted in bf16 and that subtraction is exact), and
    s / z broadcast against it. Returns e4m3-typed values.

    TWO PLACES WHERE TORCH IS NOT THE HARDWARE, both asserted rather than
    trusted, because either one silently breaks a byte-exact comparison:

    * `torch.to(bfloat16)` from f64 DOUBLE-ROUNDS (it goes through f32), so it
      is NOT step 2 in general. Witness: 1 + 2^-8 + 2^-25 rounds to 1.0 in
      torch and to 1.0078125 under a true single rounding — the f32 step lands
      exactly on the bf16 tie and RNE then pulls it down. Here it is harmless
      only because the exact product-sum is REPRESENTABLE in f32 (s and z carry
      <= 8 significant bits, q is 0..3, and |z/s| is a few), which makes the f32
      hop lossless and the bf16 step the only rounding. That is asserted below,
      so the day a caller arrives with a scale/zero ratio wide enough to need
      more than 24 bits, this fails loudly instead of drifting one ULP.
    * `torch.to(float8_e4m3fn)` does NOT saturate like `cvt.rn.satfinite`: it
      rounds to nearest and then NaNs. 449 and 464 both give 448 (0x7e), but
      500 gives 0x7f = NaN where the hardware clamps to 448. The two therefore
      agree only for |v| <= 464, asserted below. Unreachable for the pools these
      gates use (max |dequantized value| ~4.75 on N(0,1) rows), and a wrong
      number rather than a crash if it ever is reached.
    """
    s_b = s.to(torch.bfloat16).double()
    z_b = z.to(torch.bfloat16).double()
    tot = codes.double() * s_b + z_b
    assert torch.equal(tot.float().double(), tot), (
        "the exact product-sum does not survive f32, so torch's f64->bf16 "
        "double rounding is no longer a single rounding and this model has "
        "stopped matching fma.rn.bf16x2 (widen the model, do not loosen it)"
    )
    assert tot.abs().max().item() <= 464.0, (
        "a dequantized value exceeds 464, where torch's float8_e4m3fn cast "
        "returns NaN but the kernel's cvt.rn.satfinite clamps to 448 — this "
        "model and the hardware diverge above that point"
    )
    fma = tot.to(torch.bfloat16)
    return fma.float().to(torch.float8_e4m3fn)


def ref_paged_attention_fp8q(q_fp8, q_scale, mKq, mVq, mScale, bits, block_table, seq_lens,
                              sm_scale, v_store_perm=None, kv_round=None,
                              k_store_perm=None):
    """fp8-Q storage-matched paged reference: Q_eff = q_fp8.float()/q_scale
    (q_scale is `quantize_q_fp8`'s MULTIPLY-scale qs = 448/amax; the DIVIDE here matches
    the kernel's dequant direction exactly), KV dequantized
    from the LUT pools (`build_pools`/`unpack_rows`), fp32 attention, paged via
    block_table/seq_lens with ragged tails. Mirrors `ref_paged_attention`'s structure
    exactly, swapping in the LUT dequant and the fp8 Q.

    `kv_round="bf16_fma"` rebuilds K/V through `dequant_bf16_fma_e4m3` — the
    OPERAND-MATCHED reference for the `fold_scale_zero=False` ablation arm of
    `IntDecodeSm100`, which materializes exactly those values in its operand
    ring (the fold never materializes them, which is the point of the fold).
    It models all three of the arm's roundings, including the bf16 rounding of
    the (s, z) FMA OPERANDS, which is the dominant one; see that helper.
    Isolating them keeps the arm's e2e gate from absorbing the requantization
    the arm is defined to do rather than measuring it.

    `kv_round="e4m3"` is the e4m3-only variant: the right operand model for a
    dequant arm that FMAs in f32, and NOT the right one for the bf16 arm.
    """
    B, HQ, d = q_fp8.shape
    assert d == D
    assert kv_round in (None, "e4m3", "bf16_fma"), kv_round
    q = q_fp8.float() / q_scale
    k_hat, v_hat = dequant_pools(mKq, mVq, mScale, bits,
                                     v_store_perm=v_store_perm,
                                     k_store_perm=k_store_perm)
    if kv_round == "e4m3":
        k_hat = k_hat.to(torch.float8_e4m3fn).float()
        v_hat = v_hat.to(torch.float8_e4m3fn).float()
    elif kv_round == "bf16_fma":
        # rebuild from the CODES: the operand model needs (q, s, z) separately,
        # not the already-dequantized value (see dequant_bf16_fma_e4m3)
        sk, zk, sv, zv = mScale.unbind(-1)
        qk = unpack_rows(mKq, bits, D).float()
        # This branch rebuilds from the CODES, so it must undo the K order
        # itself: the `dequant_pools` call above is not on this path.
        if k_store_perm is not None:
            qk = qk[..., _k_store_perm_inverse(k_store_perm).long()]
        qv_t = unpack_rows(mVq, bits, PS).float()
        if v_store_perm is not None:
            _check_v_store_perm(v_store_perm)
            inv = torch.empty_like(v_store_perm)
            inv[v_store_perm] = torch.arange(v_store_perm.numel(),
                                             device=v_store_perm.device)
            if v_store_perm.numel() == PS:
                qv_t = qv_t[..., inv]
            else:
                lead = qv_t.shape[:-2]
                qv_t = (qv_t.reshape(*lead, D * PS)[..., inv.long()]
                        .reshape(*lead, D, PS))
        qv = qv_t.transpose(-1, -2)
        k_hat = dequant_bf16_fma_e4m3(qk, sk[..., None], zk[..., None]).float()
        v_hat = dequant_bf16_fma_e4m3(qv, sv[..., None], zv[..., None]).float()
    HKV = k_hat.shape[1]
    g = HQ // HKV
    o = torch.zeros(B, HQ, D)
    for b in range(B):
        T = int(seq_lens[b])
        pages = block_table[b, : (T + PS - 1) // PS].long()
        kb = k_hat[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]  # [HKV,T,D]
        vb = v_hat[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]
        for hq in range(HQ):
            hk = hq // g
            S = (q[b, hq] @ kb[hk].T) * sm_scale
            P = torch.softmax(S, -1)
            o[b, hq] = P @ vb[hk]
    return o


def ref_paged_attention_fp8q_fast(q_fp8, q_scale, mKq, mVq, mScale, bits, block_table,
                                   seq_lens, sm_scale, v_store_perm=None,
                                   k_store_perm=None):
    """Head-batched einsum version of `ref_paged_attention_fp8q` for LARGE problems (Task-5
    matrix at bs64/HQ32/kv8k — the per-(b,hq) loop is too slow there). Numerics: NOT
    bit-equal to the loop version — einsum's f32 reduction order differs (measured ~4e-7
    max abs on O(1) outputs); cross-checked against the loop at atol=rtol=1e-6. The loop
    version remains the oracle; use this one where a
    5e-3-class tolerance makes the ~1e-6 difference invisible."""
    B, HQ, d = q_fp8.shape
    assert d == D
    q = q_fp8.float() / q_scale
    k_hat, v_hat = dequant_pools(mKq, mVq, mScale, bits,
                                     v_store_perm=v_store_perm,
                                     k_store_perm=k_store_perm)
    HKV = k_hat.shape[1]
    g = HQ // HKV
    o = torch.zeros(B, HQ, D)
    for b in range(B):
        T = int(seq_lens[b])
        pages = block_table[b, : (T + PS - 1) // PS].long()
        kb = k_hat[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]  # [HKV,T,D]
        vb = v_hat[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]
        qb = q[b].reshape(HKV, g, D)                                       # heads grouped by hk
        P = torch.softmax(torch.einsum("hgd,htd->hgt", qb, kb) * sm_scale, -1)
        o[b] = torch.einsum("hgt,htd->hgd", P, vb).reshape(HQ, D)
    return o


def ref_paged_attention(q, mKq, mVq, mScale, bits, block_table, seq_lens, sm_scale):
    """q [B,HQ,D] -> o [B,HQ,D] f32, over the same pools the kernel reads."""
    B, HQ, d = q.shape
    assert d == D
    k_hat, v_hat = _dequant_pools_plane(mKq, mVq, mScale, bits)
    HKV = k_hat.shape[1]
    g = HQ // HKV
    o = torch.zeros(B, HQ, D)
    for b in range(B):
        T = int(seq_lens[b])
        pages = block_table[b, : (T + PS - 1) // PS].long()
        kb = k_hat[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]  # [HKV,T,D]
        vb = v_hat[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]
        for hq in range(HQ):
            hk = hq // g
            S = (q[b, hq].float() @ kb[hk].T) * sm_scale
            P = torch.softmax(S, -1)
            o[b, hq] = P @ vb[hk]
    return o


def _words_of(flat_u8):
    """flat uint8 [..., 4k] LE bytes -> int64 u32 words (LSB-first), for reassembling a
    packed byte stream into the u32s the kernel's OR-magic expand operates on."""
    w = flat_u8.reshape(-1, 4).to(torch.int64)
    return w[:, 0] | (w[:, 1] << 8) | (w[:, 2] << 16) | (w[:, 3] << 24)


def _gate_f4_v_perm(bits=2, NP=3, seed=0):
    """`v_store_perm_sm100_f4` property gate: the kernel's two-mask expand
    reconstructs dim-major token codes from perm+pack pool words, bit-exact."""
    torch.manual_seed(seed)
    ok = True
    q = torch.randint(0, 4, (NP, PS, D), dtype=torch.uint8)     # [page, token, dim]
    perm = v_store_perm_sm100_f4(bits)
    for p in range(NP):
        qv_flat = q[p].t().contiguous().reshape(-1)              # SAME source frame as
        stored = qv_flat[perm]                                   # build_pools's V leg
        words = _words_of(pack_rows(stored.reshape(D, PS), bits).reshape(-1))
        words = words.reshape(D, PS // 16)                       # 4 u32 per dim run
        lo = words & 0x33333333
        hi = (words >> 2) & 0x33333333
        for j in range(8):
            ok &= bool(torch.equal((lo >> (4 * j)) & 0x3,
                                   q[p, j::16, :].t().to(lo.dtype)))       # token 16w+j
            ok &= bool(torch.equal((hi >> (4 * j)) & 0x3,
                                   q[p, j + 8::16, :].t().to(hi.dtype)))   # token 16w+8+j
    print(f"[v perm int2, bits={bits}] dim-major two-mask expand property: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _gate_i4_v_perm(NP=3, seed=0):
    """`v_store_perm_sm100_f4(4)` property gate: the kernel's two-mask
    int4 deposit reconstructs each channel row's token runs from perm+pack pool
    words, byte-exact, with no PRMT. The int4 twin of the gate above."""
    torch.manual_seed(seed)
    ok = True
    q = torch.randint(0, 16, (NP, PS, D), dtype=torch.uint8)   # [page, token, dim]
    perm = v_store_perm_sm100_f4(4)
    for p in range(NP):
        qv_flat = q[p].t().contiguous().reshape(-1)             # dim-major source frame
        stored = qv_flat[perm]
        words = _words_of(pack_rows(stored.reshape(D, PS), 4).reshape(-1))
        words = words.reshape(D, PS // 8)                       # 8 u32 per channel row
        lo = words & 0x0F0F0F0F
        hi = (words >> 4) & 0x0F0F0F0F
        for b in range(4):
            ok &= bool(torch.equal((lo >> (8 * b)) & 0xF,
                                   q[p, b::8, :].t().to(lo.dtype)))       # token 8w+b
            ok &= bool(torch.equal((hi >> (8 * b)) & 0xF,
                                   q[p, b + 4::8, :].t().to(hi.dtype)))   # token 8w+4+b
    print(f"[v perm int4] dim-major two-mask int4 deposit property: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _gate_i4_v_perm_roundtrip(NP=2, seed=1):
    """`v_store_perm_sm100_f4(4)` is a true permutation of the dim-major
    flat V view -- inverting it (argsort) recovers the original source exactly."""
    torch.manual_seed(seed)
    ok = True
    q = torch.randint(0, 16, (NP, PS, D), dtype=torch.uint8)
    perm = v_store_perm_sm100_f4(4)
    inv = torch.argsort(perm)
    for p in range(NP):
        qv_flat = q[p].t().contiguous().reshape(-1)
        stored = qv_flat[perm]
        ok &= bool(torch.equal(stored[inv], qv_flat))
    print(f"[v perm int4 round-trip] argsort recovers the source: {'PASS' if ok else 'FAIL'}")
    return ok


def _gate_f4_v_perm_roundtrip(bits=2, NP=2, seed=1):
    """`v_store_perm_sm100_f4` is a true permutation of the dim-major flat V
    view -- inverting it (argsort) recovers the original source exactly."""
    torch.manual_seed(seed)
    ok = True
    q = torch.randint(0, 4, (NP, PS, D), dtype=torch.uint8)
    perm = v_store_perm_sm100_f4(bits)
    inv = torch.argsort(perm)
    for p in range(NP):
        qv_flat = q[p].t().contiguous().reshape(-1)
        stored = qv_flat[perm]
        ok &= bool(torch.equal(stored[inv], qv_flat))
    print(f"[v perm int2 round-trip, bits={bits}] argsort recovers the source: {'PASS' if ok else 'FAIL'}")
    return ok


# First 16 values + full-perm sha256 of v_store_perm_sm100(2), recorded BEFORE
# v_store_perm_sm100_f4 was added (T1 report `.superpowers/sdd/f4vp/task-1-report.md`),
# so a future edit that accidentally perturbs the flag-off perm while touching this file
# trips a gate instead of silently regressing the kernel's contract.
_V_STORE_PERM_SM100_BITS2_FIRST16 = [0, 512, 64, 576, 256, 768, 320, 832,
                                      128, 640, 192, 704, 384, 896, 448, 960]
# sha256 of the full 8192-length perm as little-endian int64 bytes.
_V_STORE_PERM_SM100_BITS2_SHA256 = (
    "11fc9b7414453a2e3dbf6c258325724d176772406dd8347f3e7690c4075aab1e")


def _gate_f4_flagoff_unchanged():
    """Adding `v_store_perm_sm100_f4` must not perturb the existing flag-off
    v_store_perm_sm100(2) -- checksummed against the pre-change values (first-16 spot
    values + full-perm sha256)."""
    import hashlib
    p = v_store_perm_sm100(2)
    ok = p[:16].tolist() == _V_STORE_PERM_SM100_BITS2_FIRST16
    dig = hashlib.sha256(p.to(torch.int64).numpy().tobytes()).hexdigest()
    ok &= dig == _V_STORE_PERM_SM100_BITS2_SHA256
    print(f"[v perm flag-off] v_store_perm_sm100(2) unchanged (first-16 + full sha256): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


FLEX_GATE_CONFIGS = [
    # (k_quant, v_quant, sdtype)
    (("token", 128), ("token", 128), torch.float32),   # legacy via flex path
    (("channel", 64), ("token", 128), torch.float32),
    (("channel", 128), ("token", 128), torch.float32),
    (("channel", 256), ("channel", 128), torch.float32),
    (("token", 128), ("channel", 64), torch.float32),
    (("token", 128), ("channel", 256), torch.float32),
    (("token", 64), ("token", 128), torch.bfloat16),
    (("token", 64), ("channel", 128), torch.bfloat16),
]


def _gate_flex_pools(seed=3):
    """Self-gates for the flex pool builders.
    G1g-legacy: build_pools_flex at the legacy config == build_pools on
    the SAME data through the SAME page perm, byte-identical (mKq/mVq/mScale).
    G1g-rt: per flex config, page-local dequant of the built pool gathered
    through the block table == quantize+dequantize of the LOGICAL stream
    directly (dequant_grouped at pool precision) — proves sectioning,
    channel-per-page duplication, packing, and scatter, exactly."""
    torch.manual_seed(seed)
    qr = _quant_ref_mod()
    B, NT, HKV = 2, 8, 2                       # S_pad = 512 (covers Gt=256)
    ok = True
    for bits in (4, 2):
        k_log = torch.randn(B, NT, PS, HKV, D, dtype=torch.bfloat16)
        v_log = torch.randn(B, NT, PS, HKV, D, dtype=torch.bfloat16)
        NP = B * NT
        perm = torch.randperm(NP)
        vperm = v_store_perm_sm100(bits)

        # G1g-legacy
        k_phys = k_log.permute(0, 1, 2, 3, 4).reshape(NP, PS, HKV, D)
        v_phys = v_log.reshape(NP, PS, HKV, D)
        k_scat = torch.empty_like(k_phys)
        v_scat = torch.empty_like(v_phys)
        k_scat[perm] = k_phys
        v_scat[perm] = v_phys
        mKq_l, mVq_l, mScale_l = build_pools(k_scat, v_scat, bits,
                                                 v_store_perm=vperm)
        mKq_f, mVq_f, mScale_f, bt, _ = build_pools_flex(
            k_log, v_log, bits, ("token", 128), ("token", 128),
            v_store_perm=vperm, perm=perm)
        g_leg = (torch.equal(mKq_f, mKq_l) and torch.equal(mVq_f, mVq_l)
                 and torch.equal(mScale_f, mScale_l.reshape(NP, HKV, -1))
                 and torch.equal(bt.reshape(-1).long(), perm))
        ok &= g_leg
        print(f"[G1g-legacy bits={bits}] flex builder == build_pools "
              f"byte-identical: {'OK' if g_leg else 'FAIL'}")

        # G1g-rt per config
        g_rt = True
        for kq, vq, sdt in FLEX_GATE_CONFIGS:
            mKq, mVq, mScale, bt, aux = build_pools_flex(
                k_log, v_log, bits, kq, vq, sdtype=sdt,
                v_store_perm=vperm, perm=perm)
            # k-order: logical -- the pool above is built without a K perm,
            # so this self-gate exercises the logical path on both sides. The
            # permuted path has its own oracle, `check_k_store_perm_f4`.
            k_hat, v_hat = dequant_pools_flex(mKq, mVq, mScale, bits, kq, vq,
                                              v_store_perm=vperm)
            # independent logical-stream path at pool precision
            for x_log, quant, hat in ((k_log, kq, k_hat), (v_log, vq, v_hat)):
                x = x_log.permute(0, 3, 1, 2, 4).reshape(B, HKV, NT * PS, D)
                q, s, z = qr.quantize_grouped(x, bits, *quant)
                s, z = s.to(sdt).float(), z.to(sdt).float()
                ref = qr.dequant_grouped(q, s, z, *quant)
                got = (hat[bt.reshape(-1).long()]
                       .reshape(B, NT, HKV, PS, D).permute(0, 2, 1, 3, 4)
                       .reshape(B, HKV, NT * PS, D))
                g_rt &= bool(torch.equal(got, ref))
        ok &= g_rt
        print(f"[G1g-rt bits={bits}] flex pool page-local dequant == logical "
              f"grouped dequant ({len(FLEX_GATE_CONFIGS)} configs): "
              f"{'OK' if g_rt else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    ok = True
    for bits in (4, 2):
        # G1a: pack<->unpack bit-exact
        codes = torch.randint(0, 2**bits, (3, 2, PS, D), dtype=torch.uint8)
        rt_ok = bool((_unpack_rows_plane(_pack_rows_plane(codes, bits), bits, D) == codes).all())
        ok &= rt_ok
        print(f"[G1a bits={bits}] pack round-trip: {'OK' if rt_ok else 'FAIL'}")

        # G1a-lut: pack_rows<->unpack_rows bit-exact (LUT order, both K-shaped E=D
        # rows and V-shaped E=PS rows, since the two row shapes hit different reshape paths).
        # k-order: logical -- a pack/unpack round trip over codes this block just
        # generated, so there is no stored order to invert on either side.
        rt_ok_k = bool((unpack_rows(pack_rows(codes, bits), bits, D) == codes).all())
        codes_v = torch.randint(0, 2**bits, (3, 2, D, PS), dtype=torch.uint8)
        rt_ok_v = bool((unpack_rows(pack_rows(codes_v, bits), bits, PS) == codes_v).all())
        rt_ok_lut = rt_ok_k and rt_ok_v
        ok &= rt_ok_lut
        print(f"[G1a-lut bits={bits}] pack_rows round-trip (K-shape/V-shape): "
              f"{'OK' if rt_ok_lut else 'FAIL'}")


        # G1b: paged ref through a SCRAMBLED block table == independent logical-stream ref.
        # Logical KV is quantized BEFORE any paging; token-wise quant commutes with the
        # page scatter, so any mismatch pins a pack/pool/gather bug.
        B, HKV, HQ, NPB = 2, 2, 8, 4
        NP = B * NPB
        sl = torch.tensor([3 * PS + 17, 2 * PS], dtype=torch.int32)  # ragged tail + exact boundary
        k_log = torch.randn(B, NPB, PS, HKV, D, dtype=torch.bfloat16)
        v_log = torch.randn(B, NPB, PS, HKV, D, dtype=torch.bfloat16)
        q = torch.randn(B, HQ, D, dtype=torch.bfloat16)
        bt = torch.randperm(NP).to(torch.int32).reshape(B, NPB)  # scrambled physical pages
        # scatter logical pages into the physical pool: page bt[b,i] holds logical page i of b
        k_phys = torch.empty(NP, PS, HKV, D, dtype=torch.bfloat16)
        v_phys = torch.empty_like(k_phys)
        k_phys[bt.reshape(-1).long()] = k_log.reshape(NP, PS, HKV, D)
        v_phys[bt.reshape(-1).long()] = v_log.reshape(NP, PS, HKV, D)
        mKq, mVq, mScale = _build_pools_plane(k_phys, v_phys, bits)
        o1 = ref_paged_attention(q, mKq, mVq, mScale, bits, bt, sl, D**-0.5)
        # independent path: quantize+dequantize the LOGICAL stream directly
        # (no pack/unpack, no pools, no block table, no gather code shared with o1)
        qk, sk, zk = quant_tokenwise(k_log.permute(0, 1, 3, 2, 4), bits)  # [B,NPB,HKV,PS(,D)]
        qv, sv, zv = quant_tokenwise(v_log.permute(0, 1, 3, 2, 4), bits)
        k_hat_log = qk.float() * sk[..., None] + zk[..., None]           # [B,NPB,HKV,PS,D]
        v_hat_log = qv.float() * sv[..., None] + zv[..., None]
        o2 = torch.zeros_like(o1)
        for b in range(B):
            T = int(sl[b])
            kb = k_hat_log[b].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]  # [HKV,T,D]
            vb = v_hat_log[b].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]
            for hq in range(HQ):
                hk = hq // (HQ // HKV)
                P = torch.softmax((q[b, hq].float() @ kb[hk].T) * D**-0.5, -1)
                o2[b, hq] = P @ vb[hk]
        same = bool(torch.equal(o1, o2))
        ok &= same
        print(f"[G1b bits={bits}] scrambled-paged ref == logical-stream ref: "
              f"{'OK' if same else 'FAIL'}")

        # G1b-lut: same independence structure as G1b (fresh scrambled table, reuse the
        # G1b logical KV/Q so this exercises the identical inputs the legacy gate does), but
        # LUT pools (build_pools/pack_rows) + the fp8-Q reference
        # (ref_paged_attention_fp8q), with the SAME q_fp8/q_scale fed to BOTH sides.
        bt_lut = torch.randperm(NP).to(torch.int32).reshape(B, NPB)  # independent scramble
        k_phys_lut = torch.empty(NP, PS, HKV, D, dtype=torch.bfloat16)
        v_phys_lut = torch.empty_like(k_phys_lut)
        k_phys_lut[bt_lut.reshape(-1).long()] = k_log.reshape(NP, PS, HKV, D)
        v_phys_lut[bt_lut.reshape(-1).long()] = v_log.reshape(NP, PS, HKV, D)
        q_fp8, q_scale = quantize_q_fp8(q)
        mKq_l, mVq_l, mScale_l = build_pools(k_phys_lut, v_phys_lut, bits)
        o1_l = ref_paged_attention_fp8q(q_fp8, q_scale, mKq_l, mVq_l, mScale_l, bits,
                                        bt_lut, sl, D**-0.5)
        # independent path: quantize+dequantize the LOGICAL stream directly, fp8-Q applied to
        # the SAME q_fp8/q_scale — no pack/unpack, no pools, no block table, no gather code
        # shared with o1_l (mirrors the G1b independence structure exactly).
        qk_l, sk_l, zk_l = quant_tokenwise(k_log.permute(0, 1, 3, 2, 4), bits)
        qv_l, sv_l, zv_l = quant_tokenwise(v_log.permute(0, 1, 3, 2, 4), bits)
        k_hat_log_l = qk_l.float() * sk_l[..., None] + zk_l[..., None]
        v_hat_log_l = qv_l.float() * sv_l[..., None] + zv_l[..., None]
        qf = q_fp8.float() / q_scale   # q_scale = multiply-scale qs; dequant divides
        o2_l = torch.zeros_like(o1_l)
        for b in range(B):
            T = int(sl[b])
            kb = k_hat_log_l[b].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]
            vb = v_hat_log_l[b].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :T]
            for hq in range(HQ):
                hk = hq // (HQ // HKV)
                P = torch.softmax((qf[b, hq] @ kb[hk].T) * D**-0.5, -1)
                o2_l[b, hq] = P @ vb[hk]
        same_lut = bool(torch.equal(o1_l, o2_l))
        ok &= same_lut
        print(f"[G1b-lut bits={bits}] scrambled-paged fp8-Q ref == logical-stream fp8-Q ref: "
              f"{'OK' if same_lut else 'FAIL'}")

        # G1e: v_store_perm plumbing — build with a RANDOM synthetic permutation, dequant
        # with the SAME permutation, must equal the identity-perm dequant exactly; the
        # permuted mVq bytes must differ (perm applied, not a no-op) and mKq/mScale must be
        # untouched. Proves the invert-reapply logic before Task 3 wires the real perm.
        kh_i, vh_i = dequant_pools(mKq_l, mVq_l, mScale_l, bits)
        g1e = True
        for tag, perm in (("rand-token", torch.randperm(PS)),
                          ("rand-full", torch.randperm(D * PS)),
                          ("sm100", v_store_perm_sm100(bits))):
            mKq_p, mVq_p, mScale_p = build_pools(k_phys_lut, v_phys_lut, bits,
                                                     v_store_perm=perm)
            kh_p, vh_p = dequant_pools(mKq_p, mVq_p, mScale_p, bits, v_store_perm=perm)
            g1e &= (torch.equal(kh_p, kh_i) and torch.equal(vh_p, vh_i)
                    and not torch.equal(mVq_p, mVq_l)      # perm really changed V bytes
                    and torch.equal(mKq_p, mKq_l) and torch.equal(mScale_p, mScale_l))
            if tag == "sm100":
                # kernel-perm paged ref must also match the independent stream ref (o2_l)
                o1_p = ref_paged_attention_fp8q(q_fp8, q_scale, mKq_p, mVq_p, mScale_p,
                                                bits, bt_lut, sl, D**-0.5,
                                                v_store_perm=perm)
                g1e &= bool(torch.equal(o1_p, o2_l))
        ok &= g1e
        print(f"[G1e bits={bits}] v_store_perm round-trips (rand-token/rand-full/sm100 "
              f"+ sm100 paged ref): {'OK' if g1e else 'FAIL'}")

        # G1f: einsum-vectorized fast reference vs the loop gate reference. Not bit-equal
        # (einsum reduction order, ~4e-7 abs class) — documented tolerance cross-gate.
        o_fast = ref_paged_attention_fp8q_fast(q_fp8, q_scale, mKq_l, mVq_l, mScale_l, bits,
                                               bt_lut, sl, D**-0.5)
        fast_diff = (o_fast - o1_l).abs().max().item()
        g1f = bool(torch.allclose(o_fast, o1_l, rtol=1e-6, atol=1e-6))
        ok &= g1f
        print(f"[G1f bits={bits}] fp8q_fast vs loop ref (max abs {fast_diff:.1e}, "
              f"tol 1e-6): {'OK' if g1f else 'FAIL'}")

        # G1c: all-equal-token rows (mx==mn hits clamp_min) must round-trip exactly:
        # q = round(0/s) = 0 everywhere, dequant = z = the constant. 0 error required.
        kc = torch.randn(NP, PS, HKV, 1, dtype=torch.bfloat16).expand(NP, PS, HKV, D).contiguous()
        vc = torch.randn(NP, PS, HKV, 1, dtype=torch.bfloat16).expand(NP, PS, HKV, D).contiguous()
        kc_hat, vc_hat = _dequant_pools_plane(*_build_pools_plane(kc, vc, bits), bits)
        err = max((kc_hat - kc.permute(0, 2, 1, 3).float()).abs().max().item(),
                  (vc_hat - vc.permute(0, 2, 1, 3).float()).abs().max().item())
        g1c = err == 0.0
        ok &= g1c
        print(f"[G1c bits={bits}] constant-token round-trip (max abs err {err:g}): "
              f"{'OK' if g1c else 'FAIL'}")

    # Quant-granularity flex pools.
    ok &= _gate_flex_pools()

    # The flag-on (dequant_int2_to_f4) V pool permutation
    # (int2-only) -- see v_store_perm_sm100_f4 docstring for the frame derivation.
    g1f4 = _gate_f4_v_perm()
    ok &= g1f4
    g1f4_rt = _gate_f4_v_perm_roundtrip()
    ok &= g1f4_rt
    g1f4_flagoff = _gate_f4_flagoff_unchanged()
    ok &= g1f4_flagoff

    # The int4 raw-feed store shuffle -- see
    # v_store_perm_sm100_f4's int4 branch and check_k_store_perm_i4 for K.
    ok &= _gate_i4_v_perm()
    ok &= _gate_i4_v_perm_roundtrip()


    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
