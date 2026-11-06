# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""The cache API end to end, on the device.

Build a layout, allocate a cache, write a KV cache into it with `append`, decode
with `run`, and score the output against the values the cache reads back. The
reference is the cache's own `dequantize`, so what is measured is the kernel's
own error rather than the quantization's.
"""
import pytest
import torch

from fosca import PagedDecodeWrapper, QuantKVLayout, get_capabilities, is_supported

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not is_supported(), reason="needs an SM90 or SM100 device"),
]

D, PS, HQ, HKV = 128, 64, 32, 8
BATCH, KV = 2, 4096
SM_SCALE = D ** -0.5
#: Over the whole output, `||out - reference|| / ||reference||`. The maximum over the 64
#: (batch, head) pairs is a tail statistic that ranges from 4e-3 to 1e-2 depending on the
#: query draw, while this one stays between 3.2e-3 and 3.8e-3 across seeds and both widths,
#: so it is what the bar is set on and the worst head is reported beside it. The reference is
#: the cache's own read-back, so this bounds the decode's arithmetic, not the quantization.
TOLERANCE = 6e-3


def _decoded(bits):
    """Write a cache, decode one step, and return (out, reference)."""
    device = "cuda"
    arch = get_capabilities()["arch"]
    q_dtype = torch.float8_e4m3fn if arch == "sm100" else torch.bfloat16
    nt = KV // PS
    npages = BATCH * nt
    tokens = BATCH * KV

    g = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(tokens, HKV, D, device=device, dtype=torch.bfloat16, generator=g)
    v = torch.randn(tokens, HKV, D, device=device, dtype=torch.bfloat16, generator=g)

    layout = QuantKVLayout.create(bits=bits, head_dim=D, num_kv_heads=HKV, device=device)
    cache = layout.allocate(npages)
    cache.append(k, v, torch.arange(tokens, device=device, dtype=torch.int64))

    indptr = torch.arange(BATCH + 1, device=device, dtype=torch.int32) * nt
    indices = torch.arange(npages, device=device, dtype=torch.int32)
    seq = torch.full((BATCH,), KV, device=device, dtype=torch.int32)

    gq = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(BATCH, HQ, D, device=device, dtype=torch.bfloat16,
                    generator=gq).to(q_dtype)
    w = PagedDecodeWrapper(torch.empty(256 * 1024 * 1024, dtype=torch.uint8,
                                       device=device))
    w.plan(indptr, indices, seq, num_qo_heads=HQ, layout=layout)
    out = torch.empty(BATCH, HQ, D, device=device, dtype=torch.bfloat16)
    w.run(q, cache, out=out)
    torch.cuda.synchronize()

    k_hat = torch.empty(tokens, HKV, D, device=device, dtype=torch.bfloat16)
    v_hat = torch.empty_like(k_hat)
    out_indptr = (torch.arange(BATCH, device=device, dtype=torch.int32) * KV)
    cache.dequantize(k_hat, v_hat, indptr, indices, seq, out_indptr)
    torch.cuda.synchronize()

    group = HQ // HKV
    want = torch.zeros(BATCH, HQ, D, device=device, dtype=torch.float32)
    for b in range(BATCH):
        kb = k_hat[b * KV:(b + 1) * KV].permute(1, 0, 2).float()
        vb = v_hat[b * KV:(b + 1) * KV].permute(1, 0, 2).float()
        qb = q[b].float().reshape(HKV, group, D)
        p = torch.softmax(torch.einsum("hgd,htd->hgt", qb, kb) * SM_SCALE, -1)
        want[b] = torch.einsum("hgt,htd->hgd", p, vb).reshape(HQ, D)
    return out, want


@pytest.mark.parametrize("bits", [2, 4])
def test_the_cache_api_decodes_to_its_own_read_back(bits):
    out, want = _decoded(bits)
    got = out.reshape(BATCH, HQ, D).float()
    err = got - want
    rel = (err.norm() / want.norm()).item()
    worst = (err.norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-9)).max().item()
    assert rel < TOLERANCE, f"global rel_l2 {rel:.3e}, worst head {worst:.3e}"
