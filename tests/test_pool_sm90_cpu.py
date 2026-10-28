# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""The Hopper pool builder, on a CPU.

The repacker walks the wgmma A-fragment map on the host, so its correctness is
checkable without a GPU: pack the values, read them back the way the kernel does,
and compare against the values placed through the same map.
"""
import pytest
import torch

from cutedsl_decode import pool_sm90, quant_ref
from cutedsl_decode._sm90.serving_cache import code_row_bytes

D, PAGE = 128, 64


def _codes(batch=2, kv_heads=8, kv_len=256, bits=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(batch, kv_heads, kv_len, D, dtype=torch.bfloat16, generator=g)
    v = torch.randn(batch, kv_heads, kv_len, D, dtype=torch.bfloat16, generator=g)
    qk, _, _ = quant_ref.quantize_grouped(k, bits, "token", D)
    qv, _, _ = quant_ref.quantize_grouped(v, bits, "token", D)
    return qk.permute(0, 2, 1, 3).to(torch.uint8), qv.permute(0, 2, 1, 3).to(torch.uint8)


def test_the_fragment_map_covers_every_k_step():
    maps = pool_sm90.frag_maps(D)
    assert tuple(maps["A_qk"].shape) == (128, D // 16, 8, 2)
    assert tuple(maps["A_pv"].shape) == (128, PAGE // 16, 8, 2)


def test_the_fragment_map_is_independent_of_the_head_dimension_for_v():
    assert torch.equal(pool_sm90.frag_maps(128)["A_pv"], pool_sm90.frag_maps(256)["A_pv"])
    assert not torch.equal(pool_sm90.frag_maps(128)["A_qk"], pool_sm90.frag_maps(256)["A_qk"])


@pytest.mark.parametrize("bits", [2, 4])
def test_the_k_packer_round_trips_through_the_kernel_read(bits):
    kq, _ = _codes(bits=bits)
    maps = pool_sm90.frag_maps(D)
    row = pool_sm90.repack_K_rs(kq, maps["A_qk"], bits, order="row")
    per_k, _ = code_row_bytes(D, PAGE, bits)
    pool_sm90.assert_rs_layout(row, per_k // 4)

    back = pool_sm90._straight_decode(row, bits, D // 16)
    m = maps["A_qk"].long()
    want = pool_sm90._tiles(kq)[:, :, :, m[..., 0], m[..., 1]]
    assert torch.equal(back.to(torch.int32), want.to(torch.int32))


@pytest.mark.parametrize("bits", [2, 4])
def test_the_two_code_pool_orders_hold_the_same_bytes(bits):
    kq, _ = _codes(bits=bits)
    maps = pool_sm90.frag_maps(D)
    row = pool_sm90.repack_K_rs(kq, maps["A_qk"], bits, order="row")
    chunk = pool_sm90.repack_K_rs(kq, maps["A_qk"], bits, order="chunk16")
    assert row.numel() == chunk.numel()
    assert sorted(row.reshape(-1).tolist()) == sorted(chunk.reshape(-1).tolist())


def test_an_unknown_order_is_refused():
    kq, _ = _codes()
    with pytest.raises(ValueError, match="chunk16"):
        pool_sm90.repack_K_rs(kq, pool_sm90.frag_maps(D)["A_qk"], 4, order="tiles")


@pytest.mark.parametrize("bits", [2, 4])
def test_the_whole_builder_lays_pages_out_as_one_global_pool(bits):
    batch, kv_heads, kv_len = 2, 8, 256
    g = torch.Generator().manual_seed(1)
    k = torch.randn(batch, kv_heads, kv_len, D, dtype=torch.bfloat16, generator=g)
    v = torch.randn_like(k)
    seq_lens = torch.full((batch,), kv_len, dtype=torch.int32)

    cache = pool_sm90.build_pools_sm90(k, v, bits, seq_lens=seq_lens)

    pages = batch * (kv_len // PAGE)
    assert cache.codes_k.shape[0] == pages
    assert cache.codes_v.shape[0] == pages
    assert cache.codes_k.shape[1] == kv_heads
    assert tuple(cache.scales.shape) == (pages, kv_heads, PAGE, 4)
    assert tuple(cache.block_table.shape) == (batch, kv_len // PAGE)
    assert cache.scales.dtype == torch.float32


def test_the_builder_refuses_a_length_that_is_not_whole_pages():
    k = torch.randn(1, 8, PAGE + 1, D, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="64-token tile"):
        pool_sm90.build_pools_sm90(k, k, 4, seq_lens=torch.tensor([PAGE + 1], dtype=torch.int32))
