# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: MIT
"""The cache API end to end, on the device.

Build a layout, allocate a cache, write a KV cache into it with `append`, decode
with `run`, and score the output against the values the cache reads back. The
reference is the cache's own `dequantize`, so what is measured is the kernel's
own error rather than the quantization's.
"""
import pytest
import torch

from foscar import PagedDecodeWrapper, QuantKVLayout, get_capabilities, is_supported

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not is_supported(), reason="needs an SM90 or SM100 device"),
]

PS = 64
BATCH, KV = 2, 4096
#: Over the whole output, `||out - reference|| / ||reference||`. The maximum over the 64
#: (batch, head) pairs is a tail statistic that ranges from 4e-3 to 1e-2 depending on the
#: query draw, while this one stays between 3.2e-3 and 3.8e-3 across seeds and both widths,
#: so it is what the bar is set on and the worst head is reported beside it. The reference is
#: the cache's own read-back, so this bounds the decode's arithmetic, not the quantization.
TOLERANCE = 6e-3


def _decoded(bits, head_dim=128, hq=32, hkv=8, kv_splits=None, prequantized=False, short_tail=False):
    """Write a cache, decode one step, and return (out, reference, lse_got, lse_want)."""
    device = "cuda"
    arch = get_capabilities()["arch"]
    q_dtype = torch.float8_e4m3fn if arch == "sm100" else torch.bfloat16
    sm_scale = head_dim ** -0.5
    nt = KV // PS
    npages = BATCH * nt
    tokens = BATCH * KV

    g = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(tokens, hkv, head_dim, device=device, dtype=torch.bfloat16, generator=g)
    v = torch.randn(tokens, hkv, head_dim, device=device, dtype=torch.bfloat16, generator=g)

    layout = QuantKVLayout.create(bits=bits, head_dim=head_dim, num_kv_heads=hkv, device=device)
    cache = layout.allocate(npages)
    slots = torch.arange(tokens, device=device, dtype=torch.int64)
    if prequantized:
        def _qtok(x):
            f = x.float()
            mn, mx = f.amin(dim=-1), f.amax(dim=-1)
            s = (mx - mn).clamp_min(1e-8) / ((1 << bits) - 1)
            q = ((f - mn.unsqueeze(-1)) / s.unsqueeze(-1)).round().clamp(0, (1 << bits) - 1).to(torch.uint8)
            return q.contiguous(), s.contiguous(), mn.contiguous()
        kq, ks, kz = _qtok(k)
        vq, vs, vz = _qtok(v)
        cache.append_quantized(kq, ks, kz, vq, vs, vz, slots)
    elif short_tail:
        cache.append(k[:-3], v[:-3], slots[:-3])
        cache.append(k[-3:], v[-3:], slots[-3:])
    else:
        cache.append(k, v, slots)

    indptr = torch.arange(BATCH + 1, device=device, dtype=torch.int32) * nt
    indices = torch.arange(npages, device=device, dtype=torch.int32)
    seq = torch.full((BATCH,), KV, device=device, dtype=torch.int32)

    gq = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(BATCH, hq, head_dim, device=device, dtype=torch.bfloat16,
                    generator=gq).to(q_dtype)
    w = PagedDecodeWrapper(torch.empty(256 * 1024 * 1024, dtype=torch.uint8,
                                       device=device))
    w.plan(indptr, indices, seq, num_qo_heads=hq, layout=layout, kv_splits=kv_splits)
    out = torch.empty(BATCH, hq, head_dim, device=device, dtype=torch.bfloat16)
    lse_got = torch.empty(BATCH, hq, device=device, dtype=torch.float32)
    w.run(q, cache, out=out, lse=lse_got)
    torch.cuda.synchronize()

    k_hat = torch.empty(tokens, hkv, head_dim, device=device, dtype=torch.bfloat16)
    v_hat = torch.empty_like(k_hat)
    out_indptr = (torch.arange(BATCH, device=device, dtype=torch.int32) * KV)
    cache.dequantize(k_hat, v_hat, indptr, indices, seq, out_indptr)
    torch.cuda.synchronize()

    group = hq // hkv
    want = torch.zeros(BATCH, hq, head_dim, device=device, dtype=torch.float32)
    lse_want = torch.zeros(BATCH, hq, device=device, dtype=torch.float32)
    for b in range(BATCH):
        kb = k_hat[b * KV:(b + 1) * KV].permute(1, 0, 2).float()
        vb = v_hat[b * KV:(b + 1) * KV].permute(1, 0, 2).float()
        qb = q[b].float().reshape(hkv, group, head_dim)
        scores = torch.einsum("hgd,htd->hgt", qb, kb) * sm_scale
        lse_want[b] = torch.logsumexp(scores, dim=-1).reshape(hq)
        p = torch.softmax(scores, -1)
        want[b] = torch.einsum("hgt,htd->hgd", p, vb).reshape(hq, head_dim)
    return out, want, lse_got, lse_want


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("bits", [2, 4])
def test_the_cache_api_decodes_to_its_own_read_back(bits, head_dim):
    out, want, _, _ = _decoded(bits, head_dim=head_dim)
    got = out.reshape(BATCH, 32, head_dim).float()
    err = got - want
    rel = (err.norm() / want.norm()).item()
    worst = (err.norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-9)).max().item()
    assert rel < TOLERANCE, f"global rel_l2 {rel:.3e}, worst head {worst:.3e}"


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("kv_splits", [1, 4, None])
@pytest.mark.parametrize("mode", ["normal", "prequantized", "short_tail"])
def test_muse_spark_tp8_head_dim_64(bits, kv_splits, mode):
    """Muse-Spark-Open-816B-A42B TP8 shape: HQ=16, HKV=2, head_dim=64."""
    out, want, lse_got, lse_want = _decoded(
        bits, head_dim=64, hq=16, hkv=2, kv_splits=kv_splits,
        prequantized=(mode == "prequantized"), short_tail=(mode == "short_tail"))
    got = out.reshape(BATCH, 16, 64).float()
    err = got - want
    rel = (err.norm() / want.norm()).item()
    worst = (err.norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-9)).max().item()
    assert rel < TOLERANCE, f"global rel_l2 {rel:.3e}, worst head {worst:.3e}"
    assert (lse_got - lse_want).abs().max().item() < 5e-2
