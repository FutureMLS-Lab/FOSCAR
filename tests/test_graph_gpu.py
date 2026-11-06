# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: MIT
"""`run()` inside a CUDA graph, on the device.

A serving engine captures its decode step, so the call has to be capturable and
has to replay the same answer. The capture uses the fast path it needs: a
caller-owned `out` buffer.
"""
import pytest
import torch

from foscar import PagedDecodeWrapper, QuantKVLayout, get_capabilities, is_supported

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not is_supported(), reason="needs an SM90 or SM100 device"),
]

D, PS, HQ, HKV = 128, 64, 32, 8
BATCH, KV = 2, 8192


def _planned(bits):
    device = "cuda"
    arch = get_capabilities()["arch"]
    q_dtype = torch.float8_e4m3fn if arch == "sm100" else torch.bfloat16
    nt = KV // PS
    npages = BATCH * nt
    tokens = BATCH * KV

    k = torch.randn(tokens, HKV, D, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    layout = QuantKVLayout.create(bits=bits, head_dim=D, num_kv_heads=HKV, device=device)
    cache = layout.allocate(npages)
    cache.append(k, v, torch.arange(tokens, device=device, dtype=torch.int64))

    indptr = torch.arange(BATCH + 1, device=device, dtype=torch.int32) * nt
    indices = torch.arange(npages, device=device, dtype=torch.int32)
    seq = torch.full((BATCH,), KV, device=device, dtype=torch.int32)

    q = torch.randn(BATCH, HQ, D, device=device, dtype=torch.bfloat16).to(q_dtype)
    w = PagedDecodeWrapper(torch.empty(256 * 1024 * 1024, dtype=torch.uint8,
                                       device=device))
    w.plan(indptr, indices, seq, num_qo_heads=HQ, layout=layout)
    out = torch.empty(BATCH, HQ, D, device=device, dtype=torch.bfloat16)
    return (lambda: w.run(q, cache, out=out)), out


@pytest.mark.parametrize("bits", [2, 4])
def test_run_replays_identically_from_a_cuda_graph(bits):
    call, out = _planned(bits)

    call()
    torch.cuda.synchronize()
    eager = out.clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()

    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
