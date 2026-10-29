# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""`run()` inside a CUDA graph, on the device.

A serving engine captures its decode step, so the call has to be capturable and
has to replay the same answer. Both architectures are checked here with the fast
path a capture needs: a caller-owned `out` buffer, and the planned scale factor
passed in rather than derived, since deriving it is a device synchronisation.
"""
import pytest
import torch

from fosca import (
    PagedDecodeWrapper,
    get_capabilities,
    is_supported,
    pool,
    pool_sm90,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not is_supported(), reason="needs an SM90 or SM100 device"),
]

D, PS, HQ, HKV = 128, 64, 32, 8
BATCH, KV = 2, 8192


def _planned(bits):
    """Build pools for the running architecture and return (call, out, wrapper)."""
    nt = KV // PS
    arch = get_capabilities()["arch"]
    w = PagedDecodeWrapper(torch.empty(256 * 1024 * 1024, dtype=torch.uint8,
                                       device="cuda"))
    indptr = torch.arange(BATCH + 1, device="cuda", dtype=torch.int32) * nt

    if arch == "sm100":
        k = torch.randn(BATCH * nt, PS, HKV, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        k_perm = pool.k_store_perm_sm100_f4(bits)
        v_perm = pool.v_store_perm_sm100_f4(bits)
        mKq, mVq, mScale = pool.build_pools(k, v, bits,
                                                k_store_perm=k_perm,
                                                v_store_perm=v_perm)
        seq = torch.full((BATCH,), KV, device="cuda", dtype=torch.int32)
        q_bf16 = torch.randn(BATCH, HQ, D, device="cuda", dtype=torch.bfloat16)
        q, q_scale = pool.quantize_q_fp8(q_bf16)
        w.plan(indptr, torch.arange(BATCH * nt, device="cuda", dtype=torch.int32),
               seq, num_qo_heads=HQ, num_kv_heads=HKV, head_dim=D, page_size=PS,
               q_data_type=torch.float8_e4m3fn, bits=bits,
               sm_scale=D ** -0.5 / q_scale, max_kv_len=KV)
        lp = w.pscale_lp(mScale)
        out = torch.empty(BATCH, HQ, D, device="cuda", dtype=torch.bfloat16)

        def call():
            w.run(q, mKq, mVq, out=out, scale_pool=mScale, pscale_lp=lp,
                  k_pool_order="permuted")

        return call, out

    k = torch.randn(BATCH, HKV, KV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    seq = torch.full((BATCH,), KV, device="cuda", dtype=torch.int32)
    cache = pool_sm90.build_pools_sm90(k, v, bits, seq_lens=seq)
    q = torch.randn(BATCH, HQ, D, device="cuda", dtype=torch.bfloat16)
    w.plan(indptr, cache.block_table.reshape(-1).to(torch.int32), seq,
           num_qo_heads=HQ, num_kv_heads=HKV, head_dim=D, page_size=PS,
           q_data_type=torch.bfloat16, bits=bits, max_kv_len=KV)
    out = torch.empty(BATCH, HQ, D, device="cuda", dtype=torch.bfloat16)

    def call():
        w.run(q, cache.codes_k, cache.codes_v, out=out, scale_pool=cache.scales)

    return call, out


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
