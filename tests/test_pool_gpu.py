# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""Both pool builders, on the hardware their kernel runs on.

Each test builds the pools from a bf16 cache with the builder this package ships
for that architecture, runs the decode kernel on them, and scores the output
against a storage-matched reference. That makes the pool contract and the kernel
one gate: a builder that writes the wrong order, or a kernel that reads one, shows
up here as a residual rather than as a wrong answer nobody checks.
"""
import pytest
import torch

from cutedsl_decode import (
    PagedDecodeWrapper,
    get_capabilities,
    is_supported,
    pool,
    pool_sm90,
    quant_ref,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not is_supported(), reason="needs an SM90 or SM100 device"),
]

D, PS, HQ, HKV = 128, 64, 32, 8
BATCH, KV = 2, 4096
SM_SCALE = D ** -0.5
TOLERANCE = 5e-3


def _cache(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn(BATCH, HKV, KV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(BATCH, HKV, KV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return k, v


def _reference(q_eff, k_hat, v_hat, block_table, seq_lens, sm_scale):
    """Naive attention over storage-matched values, one page table lookup.

    ``k_hat`` and ``v_hat`` are pools in the layout the kernel reads,
    ``[pages, kv_heads, page_size, head_dim]``.
    """
    group = HQ // HKV
    out = torch.zeros(BATCH, HQ, D, device=q_eff.device, dtype=torch.float32)
    for b in range(BATCH):
        tokens = int(seq_lens[b])
        pages = block_table[b, :(tokens + PS - 1) // PS].long()
        kb = k_hat[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :tokens]
        vb = v_hat[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :tokens]
        qb = q_eff[b].reshape(HKV, group, D)
        p = torch.softmax(torch.einsum("hgd,htd->hgt", qb, kb) * sm_scale, -1)
        out[b] = torch.einsum("hgt,htd->hgd", p, vb).reshape(HQ, D)
    return out


def _worst_rel_l2(got, want):
    got = got.reshape(BATCH, HQ, D).float()
    denom = want.norm(dim=-1).clamp_min(1e-9)
    return ((got - want).norm(dim=-1) / denom).max().item()


def _indptr():
    nt = KV // PS
    return torch.arange(BATCH + 1, device="cuda", dtype=torch.int32) * nt


@pytest.mark.parametrize("bits", [2, 4])
def test_the_sm100_builder_drives_its_kernel_to_the_reference(bits):
    if get_capabilities()["arch"] != "sm100":
        pytest.skip("this is the Blackwell pool builder")

    k, v = _cache()
    k_perm = pool.k_store_perm_sm100_f4(bits)
    v_perm = pool.v_store_perm_sm100_f4(bits)
    # The builder takes physical pages: [pages, page_size, kv_heads, head_dim].
    nt = KV // PS
    mKq, mVq, mScale = pool.build_pools_lut(
        k.permute(0, 2, 1, 3).reshape(BATCH * nt, PS, HKV, D),
        v.permute(0, 2, 1, 3).reshape(BATCH * nt, PS, HKV, D),
        bits, k_store_perm=k_perm, v_store_perm=v_perm)

    block_table = torch.arange(BATCH * nt, device="cuda", dtype=torch.int32).reshape(BATCH, nt)
    seq = torch.full((BATCH,), KV, device="cuda", dtype=torch.int32)

    q_bf16 = torch.randn(BATCH, HQ, D, device="cuda", dtype=torch.bfloat16)
    q_fp8, q_scale = pool.quantize_q_fp8(q_bf16)

    w = PagedDecodeWrapper(torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda"))
    w.plan(_indptr(), block_table.reshape(-1), seq, num_qo_heads=HQ, num_kv_heads=HKV,
           head_dim=D, page_size=PS, q_data_type=torch.float8_e4m3fn, bits=bits,
           sm_scale=SM_SCALE / q_scale, max_kv_len=KV)
    out = w.run(q_fp8, mKq, mVq, scale_pool=mScale, k_pool_order="permuted")
    torch.cuda.synchronize()

    k_hat, v_hat = pool.dequant_pools_lut(mKq, mVq, mScale, bits,
                                          v_store_perm=v_perm, k_store_perm=k_perm)
    want = _reference(q_fp8.float() / q_scale, k_hat, v_hat, block_table, seq, SM_SCALE)
    assert _worst_rel_l2(out, want) < TOLERANCE


@pytest.mark.parametrize("bits", [2, 4])
def test_the_sm90_builder_drives_its_kernel_to_the_reference(bits):
    if get_capabilities()["arch"] != "sm90":
        pytest.skip("this is the Hopper pool builder")

    k, v = _cache()
    seq = torch.full((BATCH,), KV, device="cuda", dtype=torch.int32)
    cache = pool_sm90.build_pools_sm90(k, v, bits, seq_lens=seq)

    q = torch.randn(BATCH, HQ, D, device="cuda", dtype=torch.bfloat16)
    w = PagedDecodeWrapper(torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda"))
    w.plan(_indptr(), cache.block_table.reshape(-1), seq, num_qo_heads=HQ,
           num_kv_heads=HKV, head_dim=D, page_size=PS, q_data_type=torch.bfloat16,
           bits=bits, max_kv_len=KV)
    out = w.run(q, cache.codes_k, cache.codes_v, scale_pool=cache.scales)
    torch.cuda.synchronize()

    qk, sk, zk = quant_ref.quantize_grouped(k, bits, "token", D)
    qv, sv, zv = quant_ref.quantize_grouped(v, bits, "token", D)
    k_hat = quant_ref.dequant_grouped(qk, sk, zk, "token", D).float()
    v_hat = quant_ref.dequant_grouped(qv, sv, zv, "token", D).float()
    # The kernel's K and V are laid out [kv_heads, tokens, head_dim] per page, so the
    # reference reads them through the same page table the kernel used.
    # The values above are in logical order; the pool is a permutation of pages. Scatter
    # them into the physical order the block table names, so the reference reads the pool
    # through the same table the kernel did.
    npages = BATCH * (KV // PS)
    k_logical = k_hat.permute(0, 2, 1, 3).reshape(npages, PS, HKV, D).permute(0, 2, 1, 3)
    v_logical = v_hat.permute(0, 2, 1, 3).reshape(npages, PS, HKV, D).permute(0, 2, 1, 3)
    k_pages = torch.empty_like(k_logical)
    v_pages = torch.empty_like(v_logical)
    k_pages[cache.block_table.reshape(-1).long()] = k_logical
    v_pages[cache.block_table.reshape(-1).long()] = v_logical
    want = _reference(q.float(), k_pages, v_pages, cache.block_table, seq, SM_SCALE)
    assert _worst_rel_l2(out, want) < TOLERANCE
