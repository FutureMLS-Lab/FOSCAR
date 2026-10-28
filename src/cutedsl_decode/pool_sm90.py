# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""Pool construction for the SM90 quantized-KV decode kernel.

The Hopper kernel reads chunk-major code pools plus an interleaved scale pool.
This module turns a bf16 KV cache into those pools, and exposes the host
repacker on its own for a caller that already holds integer values.

TWO ENTRY POINTS

`build_pools_sm90` is the whole path: quantize, repack, and lay the pages out as
one global pool addressed through a block table. Its input is a logical cache of
shape ``[batch, kv_heads, kv_len, head_dim]``, which is the order
`quant_ref.quantize_grouped` works in.

`repack_K_rs` and `repack_V_rs` take integer values of shape
``[batch, kv_len, kv_heads, head_dim]`` and produce the thread rows alone, for a
caller that quantized somewhere else or keeps its own page table. Ask
`frag_maps` for the fragment coordinate maps they need.

WHAT THE REPACK DOES

The kernel reads each 64-token page through the wgmma A-fragment layout: one
warpgroup of 128 threads, each thread holding a fixed set of (row, contraction)
coordinates. The repacker walks that same map on the host, so a pool written here
is the pool the kernel's load pattern expects, with no runtime shuffle. The map
is closed form in the head dimension, so no fixture file is needed.

The code pool order is ``"chunk16"`` unless the caller says otherwise. It is the
order `IntPingPongDecodeSm90.code_pool_order` names, and it is not inferable from
the pool bytes: a pool in the other order, ``"row"``, reads wrong without
raising.
"""
import torch

from . import quant_ref
from ._sm90.serving_cache import (
    CODE_V_HALF_CHANNELS,
    code_row_bytes,
    code_rows_to_order,
    make_serving_cache_flex,
)

#: Tokens per KV tile. It is the page size and the M of the m64 wgmma.
PAGE = 64


def wgmma_a_frag_map(k_steps):
    """The RS wgmma ``m64nNk16`` A-fragment coordinate map for ``k_steps`` k-steps.

    Returns int32 ``[128 thread, k_steps, 8 element, 2]`` holding ``(m, k)``. Warp
    ``t // 32`` of the warpgroup owns rows ``16 * (t // 32)`` to
    ``16 * (t // 32) + 15``, and inside one 16-column k-step, lane ``t % 32``
    holds the m16n8k16 A fragment. Element ``s`` of k-step ``kt`` is

        m = 16 * (t // 32) + (t % 32) // 4 + 8 * ((s // 2) % 2)
        k = 16 * kt + 2 * (t % 4) + 8 * (s // 4) + s % 2

    For K the pair is (token, channel) and there are ``head_dim // 16`` k-steps;
    for V transposed it is (channel within an output half, token) and there are
    ``PAGE // 16``.
    """
    t = torch.arange(128).view(128, 1, 1)
    kt = torch.arange(k_steps).view(1, k_steps, 1)
    s = torch.arange(8).view(1, 1, 8)
    m = 16 * (t // 32) + (t % 32) // 4 + 8 * ((s // 2) % 2)
    k = 16 * kt + 2 * (t % 4) + 8 * (s // 4) + s % 2
    return torch.stack(torch.broadcast_tensors(m, k), dim=-1).to(torch.int32)


def frag_maps(head_dim):
    """The A-fragment maps a packer needs at ``head_dim``: ``{"A_qk", "A_pv"}``.

    ``A_qk`` covers ``head_dim // 16`` k-steps and depends on the head dimension.
    ``A_pv`` covers one 64-channel output half, so it is the same at every head
    dimension.
    """
    return {
        "A_qk": wgmma_a_frag_map(head_dim // 16),
        "A_pv": wgmma_a_frag_map(PAGE // 16),
    }


def _pack_plane_words(g, bits):
    """``g [..., KT, 8]`` integer values in fragment-element order to plane-packed
    ``uint8 [..., nwords * 4]``.

    At 4 bits a word carries one k-step, and at 2 bits a word carries a k-step
    pair.
    """
    assert bits in (4, 2) and g.shape[-1] == 8
    mask = (1 << bits) - 1
    # The intermediate is exact: values are at most 4 bits and shifts at most 14.
    g = g.to(torch.int32) & mask
    lo, hi = g[..., 0::2], g[..., 1::2]
    if bits == 4:
        sh = 4 * torch.arange(4, device=g.device, dtype=torch.int32)
    else:
        *lead, KT, _ = lo.shape
        assert KT % 2 == 0, "int2 packs k-step PAIRS into u32 words"
        lo = lo.reshape(*lead, KT // 2, 8)
        hi = hi.reshape(*lead, KT // 2, 8)
        sh = 2 * torch.arange(8, device=g.device, dtype=torch.int32)
    w = (lo << sh).sum(-1) | (hi << (16 + sh)).sum(-1)
    b = torch.stack([(w >> (8 * k)) & 0xFF for k in range(4)], dim=-1)
    return b.reshape(*b.shape[:-2], -1).to(torch.uint8)


def _straight_decode(rep, bits, KT):
    """The inverse of `_pack_plane_words`, read sequentially.

    ``rep [..., nwords * 4]`` to ``[..., KT, 8]`` in fragment-element order. This
    is the kernel's own read-and-build-pairs, expressed on the host.
    """
    by = rep.reshape(*rep.shape[:-1], -1, 4).long()
    w = by[..., 0] | by[..., 1] << 8 | by[..., 2] << 16 | by[..., 3] << 24
    npw = 4 if bits == 4 else 8
    sh = bits * torch.arange(npw, device=rep.device)
    mask = (1 << bits) - 1
    lo = (w.unsqueeze(-1) >> sh) & mask
    hi = (w.unsqueeze(-1) >> (16 + sh)) & mask
    return torch.stack([lo, hi], dim=-1).reshape(*w.shape[:-1], KT, 8)


def _tiles(x_int):
    """``[B, KV, HKV, D]`` to ``[B, NT, HKV, PAGE, D]``, one entry per 64-token tile."""
    B, KV, HKV, D = x_int.shape
    assert KV % PAGE == 0, "kv_len must be a multiple of the 64-token tile"
    return x_int.reshape(B, KV // PAGE, PAGE, HKV, D).permute(0, 1, 3, 2, 4)


def _require_map(name, m, k_steps):
    """A map that does not cover every k-step would pack only the channels it names."""
    if tuple(m.shape) != (128, k_steps, 8, 2):
        raise ValueError(
            f"the {name} map has shape {tuple(m.shape)}; it must cover all {k_steps} "
            f"k-steps, [128, {k_steps}, 8, 2], which wgmma_a_frag_map({k_steps}) is.")


def _require_row_bytes(name, rows, per):
    """The packed rows against the pool geometry, so packer and reader agree on `per`."""
    if rows.shape[-1] != per:
        raise ValueError(f"packed {name} rows are {rows.shape[-1]} bytes; "
                         f"serving_cache.code_row_bytes says {per}")


def repack_K_rs(k_int, mapA, bits, *, order):
    """``k_int [B, KV, HKV, D]`` integer values to ``uint8 [B, NT, HKV, <block>]``.

    The result holds the plane-word thread rows stored in ``order``, the code-pool
    order of the kernel that reads the pool. There is no default, because a pool
    in the wrong order reads wrong rather than raising: ``"chunk16"`` is the
    shipped SM90 kernel's order and ``"row"`` is the thread-row layout itself.
    ``mapA`` is ``frag_maps(head_dim)["A_qk"]``.
    """
    B, KV, HKV, D = k_int.shape
    per_k, _ = code_row_bytes(D, PAGE, bits)
    m = mapA.to(k_int.device).long()
    _require_map("A_qk", m, D // 16)
    g = _tiles(k_int)[:, :, :, m[..., 0], m[..., 1]]
    rows = _pack_plane_words(g, bits)
    _require_row_bytes("K", rows, per_k)
    return code_rows_to_order(rows, order)


def repack_V_rs(v_int, mapA, bits, *, order):
    """``v_int [B, KV, HKV, D]`` integer values to ``uint8 [B, NT, HKV, <block>]``.

    The thread rows ``[128, per_v]``, stored in ``order`` exactly as `repack_K_rs`
    does. ``mapA`` is ``frag_maps(head_dim)["A_pv"]``, which carries the
    transposed V coordinates (channel within an output half, token) and is the
    same at every head dimension. The ``D // 64`` output halves are stacked along
    the row, half ``h`` at row bytes ``[h * per_h, (h + 1) * per_h)`` with
    ``per_h = serving_cache.code_v_half_bytes``, so in the chunk16 order at 4 bits
    chunk ``h`` is half ``h``, and at 2 bits chunk ``c`` is halves ``2c`` and
    ``2c + 1``.
    """
    B, KV, HKV, D = v_int.shape
    _, per_v = code_row_bytes(D, PAGE, bits)
    m = mapA.to(v_int.device).long()
    _require_map("A_pv", m, PAGE // 16)
    xv = _tiles(v_int)
    halves = [
        _pack_plane_words(
            xv[:, :, :, m[..., 1], CODE_V_HALF_CHANNELS * h + m[..., 0]], bits)
        for h in range(D // CODE_V_HALF_CHANNELS)
    ]
    rows = torch.cat(halves, dim=-1)
    _require_row_bytes("V", rows, per_v)
    return code_rows_to_order(rows, order)


def assert_rs_layout(rep, n_ksteps_words):
    """Check that packed rows are contiguous, aligned, and a whole number of words.

    Rows are contiguous ``uint8`` runs of stride ``per`` (a multiple of 16, so the
    kernel can load them 16 bytes at a time), the base pointer is 16-byte aligned,
    and every (thread, k-step) lives inside one aligned word of its row.
    """
    per = rep.shape[-1]
    assert rep.dtype == torch.uint8 and rep.is_contiguous()
    assert rep.stride(-2) == per and per % 16 == 0, f"row stride {per} is not a 16-byte multiple"
    assert rep.data_ptr() % 16 == 0
    assert per == 4 * n_ksteps_words


def _quantize(x, bits, quant, sdtype):
    """Quantize one logical tensor and return (values, scale, zero) for the pool."""
    axis, group = quant
    q, scale, zero = quant_ref.quantize_grouped(x, bits, axis, group)
    scale, zero = scale.to(sdtype), zero.to(sdtype)
    if axis == "token" and group == x.shape[-1]:
        scale, zero = scale.squeeze(-1), zero.squeeze(-1)
    return q, scale, zero


def build_pools_sm90(k_log, v_log, bits, *, seq_lens, k_quant=None, v_quant=None,
                     sdtype=torch.float32, order="chunk16"):
    """Build the Hopper code pools and scale pool from a bf16 logical KV cache.

    ``k_log`` and ``v_log`` are ``[batch, kv_heads, kv_len, head_dim]``, and
    ``kv_len`` must be a multiple of 64. ``k_quant`` and ``v_quant`` are the
    ``(axis, group)`` pairs the plan will use, and default to
    ``("token", head_dim)``, which is the shipped fast path. ``seq_lens`` is the
    per-request KV length in tokens.

    Returns the `ServingCache` the kernel reads: ``codes_k``, ``codes_v``,
    ``scales`` and the block table. ``sdtype`` is the scale-pool element type and
    must match what ``plan()`` compiles for.
    """
    head_dim = k_log.shape[-1]
    k_quant = k_quant or ("token", head_dim)
    v_quant = v_quant or ("token", head_dim)
    maps = frag_maps(head_dim)

    qk, sk, zk = _quantize(k_log, bits, k_quant, sdtype)
    qv, sv, zv = _quantize(v_log, bits, v_quant, sdtype)
    # The repacker works in [batch, kv_len, kv_heads, head_dim].
    kq = repack_K_rs(qk.permute(0, 2, 1, 3).to(torch.uint8), maps["A_qk"], bits, order=order)
    vq = repack_V_rs(qv.permute(0, 2, 1, 3).to(torch.uint8), maps["A_pv"], bits, order=order)

    return make_serving_cache_flex(
        kq, vq, (sk, zk), (sv, zv), seq_lens,
        k_quant=k_quant, v_quant=v_quant, N=PAGE, D=head_dim,
        sdtype=sdtype, shuffle=False,
    )
