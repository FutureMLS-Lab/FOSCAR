# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""Reference quantization of a KV cache to low-bit integer values.

This is the host-side contract the kernel's on-chip dequantization has to match.

  - Granularity: one ``(scale, zero)`` pair per group, where a group is a token
    (all ``head_dim`` channels) or a channel (a group of tokens).
  - Affine asymmetric: ``q = round((x - zero) / scale)`` clamped to
    ``[0, 2**bits - 1]``, and ``deq = q * scale + zero``.
  - Token-wise: ``zero`` is the group minimum and ``scale`` is
    ``(max - min) / (2**bits - 1)``.
  - Packing: along the head dimension, little-endian within a byte. There are
    ``8 // bits`` values per byte, and element ``j`` lands in byte ``j // vpb``
    at bit ``(j % vpb) * bits``.
"""
import math
import torch


def quantize_tokenwise(x: torch.Tensor, bits: int):
    """x: [..., d] -> (q_int[...,d] int32, scale[...] f32, zero[...] f32). deq = q*scale+zero."""
    qmax = (1 << bits) - 1
    xf = x.float()
    mn = xf.amin(dim=-1, keepdim=True)
    mx = xf.amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / qmax
    zero = mn
    q = torch.round((xf - zero) / scale).clamp_(0, qmax).to(torch.int32)
    return q, scale.squeeze(-1), zero.squeeze(-1)


def dequant_tokenwise(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.unsqueeze(-1) + zero.unsqueeze(-1)


# --------------------------------------------------------------------------------------------
# Grouped affine quant, unified over the group AXIS (the only difference between the two schemes):
#   axis="token"   : group along D  -> scale [..., S, D//G]  (each token; G channels share a scale)
#   axis="channel" : group along S  -> scale [..., S//G, D]  (each channel; G tokens share a scale)
# The integer values q are identical in shape ([..., S, D]); only WHICH elements share a (scale,
# zero) differs. Token-wise with G=D is the per-token scheme `quantize_tokenwise` above.
# Per-channel (axis=channel) isolates K's outlier *channels* -> much lower K quant error (KIVI).
# --------------------------------------------------------------------------------------------
def quantize_grouped(x: torch.Tensor, bits: int, axis: str, G: int):
    """x: [..., S, D] -> (q[...,S,D] int32, scale, zero). scale/zero shape depends on axis:
    token -> [..., S, D//G]; channel -> [..., S//G, D]. deq = q*scale + zero (per group)."""
    qmax = (1 << bits) - 1
    xf = x.float()
    *lead, S, D = xf.shape
    if axis == "token":
        assert D % G == 0, (D, G)
        xg = xf.reshape(*lead, S, D // G, G)          # group the G channels
        mn = xg.amin(-1, keepdim=True); mx = xg.amax(-1, keepdim=True)
        scale = (mx - mn).clamp_min(1e-8) / qmax; zero = mn
        q = torch.round((xg - zero) / scale).clamp_(0, qmax).to(torch.int32).reshape(*lead, S, D)
        return q, scale.squeeze(-1), zero.squeeze(-1)          # [..., S, D//G]
    elif axis == "channel":
        assert S % G == 0, (S, G)
        xg = xf.reshape(*lead, S // G, G, D)          # group the G tokens
        mn = xg.amin(-2, keepdim=True); mx = xg.amax(-2, keepdim=True)
        scale = (mx - mn).clamp_min(1e-8) / qmax; zero = mn
        q = torch.round((xg - zero) / scale).clamp_(0, qmax).to(torch.int32).reshape(*lead, S, D)
        return q, scale.squeeze(-2), zero.squeeze(-2)          # [..., S//G, D]
    raise ValueError(axis)


def dequant_grouped(q: torch.Tensor, scale, zero, axis: str, G: int) -> torch.Tensor:
    *lead, S, D = q.shape
    qf = q.float()
    if axis == "token":
        qg = qf.reshape(*lead, S, D // G, G)
        out = qg * scale.unsqueeze(-1) + zero.unsqueeze(-1)
        return out.reshape(*lead, S, D)
    elif axis == "channel":
        qg = qf.reshape(*lead, S // G, G, D)
        out = qg * scale.unsqueeze(-2) + zero.unsqueeze(-2)
        return out.reshape(*lead, S, D)
    raise ValueError(axis)


def _interleave_perm(bits: int):
    """Within each 32-bit group (G = 32//bits values), reorder to [evens, odds] so the kernel
    can extract a bf16-lane pair with one LOP3 `((w>>e*bits) & MASK) | MAGIC` and NO per-nibble
    positioning shift (Marlin/Machete-style). evens land in the low 16 bits, odds in the high
    16 bits; extraction e then yields (value 2e, value 2e+1) in natural d-order."""
    G = 32 // bits
    return list(range(0, G, 2)) + list(range(1, G, 2)), G


def pack_along_last(q: torch.Tensor, bits: int, interleave: bool = True) -> torch.Tensor:
    """q: [..., d] int in [0,2^bits-1] -> uint8 [..., d*bits//8]. Little-endian within byte.
    With interleave=True the values are permuted within each 32-bit group for the branchless
    LOP3 unpack (see _interleave_perm)."""
    vpb = 8 // bits
    d = q.shape[-1]
    assert d % vpb == 0, (d, vpb)
    if interleave:
        perm, G = _interleave_perm(bits)
        assert d % G == 0, (d, G)
        q = q.reshape(*q.shape[:-1], d // G, G)[..., perm].reshape(*q.shape[:-1], d).contiguous()
    qn = q.to(torch.int32).reshape(*q.shape[:-1], d // vpb, vpb)
    out = torch.zeros(*qn.shape[:-1], device=q.device, dtype=torch.int32)
    for k in range(vpb):
        out |= (qn[..., k] & ((1 << bits) - 1)) << (k * bits)
    return out.to(torch.uint8)


def unpack_along_last(packed: torch.Tensor, bits: int, d: int, interleave: bool = True) -> torch.Tensor:
    """uint8 [..., d_bytes] -> int32 [..., d]. Inverse of pack_along_last."""
    vpb = 8 // bits
    mask = (1 << bits) - 1
    p = packed.to(torch.int32)
    outs = [((p >> (k * bits)) & mask) for k in range(vpb)]
    out = torch.stack(outs, dim=-1).reshape(*packed.shape[:-1], d)  # sequential (interleaved) order
    if interleave:
        perm, G = _interleave_perm(bits)
        inv = sorted(range(G), key=lambda i: perm[i])  # argsort(perm)
        out = out.reshape(*out.shape[:-1], d // G, G)[..., inv].reshape(*out.shape[:-1], d)
    return out


def build_quantized_paged_kv(b, kv_len, page_size, h_kv, h_q, d, bits, device, seed=0):
    """Build a paged KV cache, quantize token-wise, pack. Returns a dict with both the
    packed tensors (kernel inputs) and the dequantized bf16 caches (reference + bf16 baseline)."""
    torch.manual_seed(seed)
    pps = (kv_len + page_size - 1) // page_size
    total = pps * b
    # true bf16 caches
    k_cache = torch.randn(total, page_size, h_kv, d, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total, page_size, h_kv, d, device=device, dtype=torch.bfloat16)
    q = torch.randn(b, h_q, d, device=device, dtype=torch.bfloat16)

    qk, sk, zk = quantize_tokenwise(k_cache, bits)
    qv, sv, zv = quantize_tokenwise(v_cache, bits)
    packed_k = pack_along_last(qk, bits).contiguous()       # [total, ps, h_kv, d_bytes]
    packed_v = pack_along_last(qv, bits).contiguous()
    # dequantized bf16 caches (what BOTH the kernel and the reference effectively attend over)
    k_deq = dequant_tokenwise(qk, sk, zk).to(torch.bfloat16).contiguous()
    v_deq = dequant_tokenwise(qv, sv, zv).to(torch.bfloat16).contiguous()

    indptr = torch.arange(b + 1, device=device, dtype=torch.int32) * pps
    indices = torch.arange(total, device=device, dtype=torch.int32)
    seq_lens = torch.full((b,), kv_len, device=device, dtype=torch.int32)
    return dict(
        q=q, k_cache=k_cache, v_cache=v_cache, k_deq=k_deq, v_deq=v_deq,
        packed_k=packed_k, packed_v=packed_v,
        scale_k=sk.contiguous(), zero_k=zk.contiguous(),
        scale_v=sv.contiguous(), zero_v=zv.contiguous(),
        indptr=indptr, indices=indices, seq_lens=seq_lens,
        b=b, kv_len=kv_len, page_size=page_size, h_kv=h_kv, h_q=h_q, d=d, bits=bits,
    )


def _to_paged(x_log, b, pps, ps, kv_len):
    """[b, kv_len, h_kv, last] logical -> [b*pps, ps, h_kv, last] paged (pad tail with 0)."""
    _, _, h_kv, last = x_log.shape
    full = torch.zeros(b, pps * ps, h_kv, last, dtype=x_log.dtype, device=x_log.device)
    full[:, :kv_len] = x_log
    return full.reshape(b * pps, ps, h_kv, last).contiguous()


def build_quantized_paged_kv_channel(b, kv_len, page_size, h_kv, h_q, d, bits, G, device, seed=0):
    """CHANNEL-WISE (grouped) paged KV: quantize per (channel d, group of G logical tokens), so
    scale/zero are [B, H_kv, S//G, D] in LOGICAL seq order (NOT paged). The packed integers stay
    paged [pages, ps, h_kv, d_bytes] (packed along d, same interleave as token-wise). Both K and V
    are channel-wise here (the symmetric 'both-channel' variant)."""
    assert kv_len % G == 0, f"kv_len={kv_len} must be a multiple of group_size={G}"
    torch.manual_seed(seed)
    pps = (kv_len + page_size - 1) // page_size
    total = pps * b
    Sg = kv_len // G
    k_cache = torch.randn(total, page_size, h_kv, d, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total, page_size, h_kv, d, device=device, dtype=torch.bfloat16)
    q = torch.randn(b, h_q, d, device=device, dtype=torch.bfloat16)

    # logical seq per batch [b, kv_len, h_kv, d] -> per-(b,h_kv) [.., S, D] for channel grouping
    k_log = k_cache.reshape(b, pps * page_size, h_kv, d)[:, :kv_len].permute(0, 2, 1, 3)  # [b,h,S,d]
    v_log = v_cache.reshape(b, pps * page_size, h_kv, d)[:, :kv_len].permute(0, 2, 1, 3)
    qk, sk, zk = quantize_grouped(k_log, bits, "channel", G)   # qk [b,h,S,d]; sk/zk [b,h,Sg,d]
    qv, sv, zv = quantize_grouped(v_log, bits, "channel", G)
    # The kernel dequantizes with bf16 (scale, zero) -- store/round them to bf16 so the golden
    # attends over EXACTLY what the kernel produces (q*bf16(scale)+bf16(zero)). Vectorizing the
    # per-channel affine as HFMA2 needs bf16 group-scale in smem anyway.
    sk, zk, sv, zv = sk.bfloat16(), zk.bfloat16(), sv.bfloat16(), zv.bfloat16()
    k_deq = dequant_grouped(qk, sk, zk, "channel", G).to(torch.bfloat16)   # [b,h,S,d]
    v_deq = dequant_grouped(qv, sv, zv, "channel", G).to(torch.bfloat16)

    # pack along d (logical), then map logical->paged for packed + dequantized caches
    pk = pack_along_last(qk, bits).permute(0, 2, 1, 3)        # [b, S, h, d_bytes]
    pv = pack_along_last(qv, bits).permute(0, 2, 1, 3)
    packed_k = _to_paged(pk, b, pps, page_size, kv_len)        # [total, ps, h, d_bytes]
    packed_v = _to_paged(pv, b, pps, page_size, kv_len)
    k_deq_p = _to_paged(k_deq.permute(0, 2, 1, 3), b, pps, page_size, kv_len)  # [total,ps,h,d]
    v_deq_p = _to_paged(v_deq.permute(0, 2, 1, 3), b, pps, page_size, kv_len)

    indptr = torch.arange(b + 1, device=device, dtype=torch.int32) * pps
    indices = torch.arange(total, device=device, dtype=torch.int32)
    seq_lens = torch.full((b,), kv_len, device=device, dtype=torch.int32)
    return dict(
        q=q, k_cache=k_cache, v_cache=v_cache, k_deq=k_deq_p, v_deq=v_deq_p,
        packed_k=packed_k, packed_v=packed_v,
        scale_k=sk.contiguous(), zero_k=zk.contiguous(),
        scale_v=sv.contiguous(), zero_v=zv.contiguous(),
        indptr=indptr, indices=indices, seq_lens=seq_lens,
        b=b, kv_len=kv_len, page_size=page_size, h_kv=h_kv, h_q=h_q, d=d, bits=bits, G=G,
    )


def paged_decode_reference(q, k_cache_deq, v_cache_deq, indptr, indices, seq_lens, sm_scale):
    """q:[b,h_q,d]; k/v:[pages,ps,h_kv,d] (already dequantized). Returns out[b,h_q,d] f32."""
    b = indptr.numel() - 1
    page_size, h_kv, d = k_cache_deq.shape[1], k_cache_deq.shape[2], k_cache_deq.shape[3]
    h_q = q.shape[1]
    g = h_q // h_kv
    out = torch.empty(b, h_q, d, device=q.device, dtype=torch.float32)
    for bi in range(b):
        pgs = indices[indptr[bi]:indptr[bi + 1]]
        vlen = int(seq_lens[bi].item())
        keys = k_cache_deq[pgs].float().reshape(-1, h_kv, d)[:vlen].repeat_interleave(g, 1)
        vals = v_cache_deq[pgs].float().reshape(-1, h_kv, d)[:vlen].repeat_interleave(g, 1)
        logits = torch.einsum("hd,nhd->hn", q[bi].float(), keys) * sm_scale
        out[bi] = torch.einsum("hn,nhd->hd", torch.softmax(logits, -1), vals)
    return out


if __name__ == "__main__":
    # self-test: pack/unpack round-trip + dequant attention sanity vs true bf16
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for bits in (4, 2):
        data = build_quantized_paged_kv(b=2, kv_len=257, page_size=16, h_kv=2, h_q=8, d=128,
                                        bits=bits, device=dev)
        qk_rt = unpack_along_last(data["packed_k"], bits, data["d"])
        qk_ref, _, _ = quantize_tokenwise(data["k_cache"], bits)
        assert torch.equal(qk_rt, qk_ref), f"pack/unpack mismatch bits={bits}"
        sm = 1.0 / math.sqrt(data["d"])
        out_deq = paged_decode_reference(data["q"], data["k_deq"], data["v_deq"],
                                         data["indptr"], data["indices"], data["seq_lens"], sm)
        out_true = paged_decode_reference(data["q"], data["k_cache"], data["v_cache"],
                                          data["indptr"], data["indices"], data["seq_lens"], sm)
        rel = (out_deq - out_true).norm() / out_true.norm()
        print(f"bits={bits}: pack/unpack OK; dequant-vs-true-bf16 rel-err = {rel.item():.4f}")
