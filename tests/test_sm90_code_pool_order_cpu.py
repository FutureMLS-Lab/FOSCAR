# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The SM90 code-pool orders (`_sm90/serving_cache.py`, CODE POOL ORDERS), on a CPU.

IntPingPongDecodeSm90 reads the chunk16 order, and the row order is the packers'
thread-row layout. Both hold the same bytes per (page, kv-head) block, so a pool in the
wrong order decodes to finite, plausible values and raises nothing unless its SHAPE is
checked. These tests pin the definition of the order, the refusal that names it, and that
the pool builders copy whichever order they are given. The kernel-side read and the refusal inside `__call__`
need an H100, and `scripts/test_pingpong_quant_flex.py` and
`scripts/test_wrapper_quant_sm90_compile.py` gate those.

`serving_cache.py` is loaded from its file with stub parent packages, the way
`test_ours_flex_adapter_cpu.py` loads it, because importing the package imports the DSL.
"""
import ast
import importlib.util
import pathlib
import sys
import types

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sc(monkeypatch):
    for pkg in ("fosca", "fosca._sm90"):
        stub = types.ModuleType(pkg)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, pkg, stub)
    _load(monkeypatch, "fosca.quant_layout",
          ROOT / "src/fosca/quant_layout.py")
    return _load(monkeypatch, "fosca._sm90.serving_cache",
                 ROOT / "src/fosca/_sm90/serving_cache.py")


def _rows(*lead, per, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (*lead, 128, per), generator=g, dtype=torch.uint8)


@pytest.mark.parametrize("per", [16, 32, 64])
def test_chunk_c_of_thread_t_is_row_bytes_16c_to_16c_plus_16(sc, per):
    rows = _rows(2, 3, per=per)
    pool = sc.code_rows_to_order(rows, "chunk16")
    assert pool.shape == (2, 3, per // 16, 128, 16) and pool.is_contiguous()
    for c in range(per // 16):
        assert torch.equal(pool[..., c, :, :], rows[..., 16 * c:16 * c + 16])
    # Thread t's chunk c sits at byte 16t + 2048c of the block, which is the address the
    # consumer's 16-byte load uses.
    flat = pool.reshape(2, 3, -1)
    for t in (0, 1, 31, 127):
        for c in range(per // 16):
            assert torch.equal(flat[..., 16 * t + 2048 * c:16 * t + 2048 * c + 16],
                               rows[..., t, 16 * c:16 * c + 16])


@pytest.mark.parametrize("order", ["row", "chunk16"])
@pytest.mark.parametrize("per", [16, 32, 64])
def test_the_word_strides_say_where_each_row_word_is(sc, order, per):
    """Word 4c + w of thread t's row is at word t*thread + c*chunk + w of the block.

    The pingpong kernel's shared-memory view and the serving writer's stores are both built from
    these strides, so this is the definition both of them read, checked against the pool
    that `code_rows_to_order` writes."""
    rows = _rows(3, per=per)
    block = sc.code_rows_to_order(rows, order).reshape(3, -1).view(torch.int32)
    words = rows.view(torch.int32)                                  # [3, 128, per/4]
    thread, chunk = sc.code_block_word_strides(order, per)
    for t in range(128):
        for w in range(per // 4):
            assert torch.equal(block[:, t * thread + (w // 4) * chunk + w % 4], words[:, t, w])
    assert sc.code_block_word_strides("chunk16", 32) == (4, 512)
    assert sc.code_block_word_strides("row", 32) == (8, 4)


def test_one_chunk_is_the_row_order_byte_for_byte(sc):
    """At D=128 int2 the thread row is one chunk, so both orders are the same bytes."""
    rows = _rows(4, per=16)
    pool = sc.code_rows_to_order(rows, "chunk16")
    assert torch.equal(pool.flatten(), rows.flatten())
    # C-contiguous in every dim, the extent-1 chunk dim included: the kernel checks the
    # exact strides, and a transposed view keeps a stride of 16 there.
    assert pool.stride() == (2048, 2048, 16, 1)
    assert torch.equal(sc.code_rows_to_order(rows, "row"), rows)


def test_block_shapes_and_their_orders(sc):
    assert sc.code_block_shape("row", 32) == (128, 32)
    assert sc.code_block_shape("chunk16", 32) == (2, 128, 16)
    assert sc.code_block_order((128, 32)) == "row"
    assert sc.code_block_order((2, 128, 16)) == "chunk16"
    for order in ("row", "chunk16"):
        for per in (24, 0, -16):
            with pytest.raises(ValueError, match="16-byte chunks"):
                sc.code_block_shape(order, per)
    with pytest.raises(ValueError, match="unknown code pool order"):
        sc.code_block_shape("chunk32", 32)
    for bad in ((64, 32), (128, 0), (0, 128, 16), (2, 128, 8)):
        with pytest.raises(ValueError, match="neither"):
            sc.code_block_order(bad)
    with pytest.raises(ValueError, match=r"\[\.\.\., 128, per\]"):
        sc.code_rows_to_order(torch.zeros(7, 64, 32, dtype=torch.uint8), "row")


@pytest.mark.parametrize("per", [16, 32])
def test_a_row_major_pool_is_refused_and_the_error_names_both_orders(sc, per):
    with pytest.raises(ValueError) as e:
        sc.require_code_pool("mKq", (7, 8, 128, per), "chunk16", per)
    assert "'row'-order pool" in str(e.value)
    assert "reads the 'chunk16'" in str(e.value)
    assert f"{per // 16}, 128, 16" in str(e.value)
    # the launcher's order-only form refuses it too
    with pytest.raises(ValueError, match="'chunk16'"):
        sc.require_code_pool("mKq", (7, 8, 128, per), "chunk16")


def test_the_matching_pool_passes_and_a_wrong_row_length_does_not(sc):
    sc.require_code_pool("mKq", (7, 8, 2, 128, 16), "chunk16", 32)
    sc.require_code_pool("mKq", (7, 8, 2, 128, 16), "chunk16")      # order only
    sc.require_code_pool("mKq", (7, 8, 128, 32), "row", 32)
    with pytest.raises(ValueError, match="2, 128, 16"):
        sc.require_code_pool("mKq", (7, 8, 1, 128, 16), "chunk16", 32)
    for bad in ((7, 8, 2, 64, 16), (7, 8, 0, 128, 16), (7, 8, 64, 32)):
        with pytest.raises(ValueError, match="neither known order"):
            sc.require_code_pool("mKq", bad, "chunk16")
    with pytest.raises(ValueError, match="unknown code pool order"):
        sc.require_code_pool("mKq", (7, 8, 64, 32), None)


def test_a_transposed_pool_of_the_right_shape_is_refused_by_its_strides(sc):
    """The kernel reads each block as one dense run, so the strides are checked too."""
    rows = _rows(3, 8, per=32)                                   # [NP, HKV, 128, 32]
    good = sc.code_rows_to_order(rows, "chunk16")
    sc.require_code_pool("mKq", good.shape, "chunk16", 32, stride=good.stride())
    lying = rows.reshape(3, 8, 128, 2, 16).transpose(2, 3)       # chunk16 SHAPE, row bytes
    assert lying.shape == good.shape
    with pytest.raises(ValueError, match="C-contiguous"):
        sc.require_code_pool("mKq", lying.shape, "chunk16", 32, stride=lying.stride())
    # An extent-1 dim's stride moves no byte: one chunk viewed through a transpose passes.
    one = _rows(3, 8, per=16).reshape(3, 8, 128, 1, 16).transpose(2, 3)
    assert one.stride()[2] == 16
    sc.require_code_pool("mKq", one.shape, "chunk16", 16, stride=one.stride())


def test_the_page_dim_stride_counts_only_when_there_is_more_than_one_page(sc):
    """One page sliced out of a larger pool is dense, and torch agrees; every fifth page
    is not. The page dim follows the same extent-1 rule as the others (review, 2026-09-25)."""
    big = sc.code_rows_to_order(_rows(10, 4, per=32), "chunk16")      # [10, 4, 2, 128, 16]
    one = big[3::10]
    assert one.shape[0] == 1 and one.is_contiguous() and one.stride()[0] != 4 * 2 * 128 * 16
    sc.require_code_pool("mKq", one.shape, "chunk16", 32, stride=one.stride())
    fifth = big[::5]
    assert fifth.shape[0] == 2 and not fifth.is_contiguous()
    with pytest.raises(ValueError, match="C-contiguous"):
        sc.require_code_pool("mKq", fifth.shape, "chunk16", 32, stride=fifth.stride())


def test_the_refusal_warns_against_a_reshape(sc):
    """A shape error that prints the wanted shape invites a `view` to it, which is the one
    mistake nothing can detect, so the message says so."""
    with pytest.raises(ValueError, match="Do not view or reshape"):
        sc.require_code_pool("mKq", (7, 8, 128, 32), "chunk16", 32)


B_, NT_, HKV_, N_, D_ = 2, 3, 2, 64, 128


def _build(sc, builder, codes_k, codes_v):
    """One cache from each builder, pages SHUFFLED, one request ragged (seq 150 -> 3 pages)."""
    sl = torch.tensor([NT_ * N_, 150], dtype=torch.int32)
    if builder == "legacy":
        return sc.make_serving_cache(codes_k, codes_v, torch.rand(B_, HKV_, NT_ * N_, 4), sl,
                                     N=N_, shuffle=True)
    k_sz = (torch.rand(B_, HKV_, NT_ * N_ // 128 + 1, D_), torch.rand(B_, HKV_, NT_ * N_ // 128 + 1, D_))
    v_sz = (torch.rand(B_, HKV_, NT_ * N_), torch.rand(B_, HKV_, NT_ * N_))
    return sc.make_serving_cache_flex(codes_k, codes_v, k_sz, v_sz, sl, k_quant=("channel", 128),
                                      v_quant=("token", 128), N=N_, D=D_, shuffle=True)


@pytest.mark.parametrize("builder", ["legacy", "flex"])
@pytest.mark.parametrize("order", ["row", "chunk16"])
def test_each_page_of_the_pool_is_the_block_it_was_given(sc, builder, order):
    """Every (request, tile) the block table names holds exactly that tile's input block,
    in the order the input was written in."""
    codes_k = sc.code_rows_to_order(_rows(B_, NT_, HKV_, per=32, seed=1), order)
    codes_v = sc.code_rows_to_order(_rows(B_, NT_, HKV_, per=32, seed=2), order)
    cache = _build(sc, builder, codes_k, codes_v)
    assert cache.codes_k.shape[2:] == sc.code_block_shape(order, 32)
    pages = [NT_, 3]
    for b in range(B_):
        for gt in range(pages[b]):
            p = int(cache.block_table[b, gt])
            assert torch.equal(cache.codes_k[p], codes_k[b, gt])
            assert torch.equal(cache.codes_v[p], codes_v[b, gt])


@pytest.mark.parametrize("builder", ["legacy", "flex"])
def test_k_and_v_in_different_orders_are_refused(sc, builder):
    rows = _rows(B_, NT_, HKV_, per=32)
    with pytest.raises(ValueError, match="different orders"):
        _build(sc, builder, rows, sc.code_rows_to_order(rows, "chunk16"))


def test_the_pingpong_kernel_declares_the_chunk16_order():
    """Read from the source, because importing the kernel needs the DSL."""
    tree = ast.parse((ROOT / "src/fosca/_sm90/int_decode_pingpong.py").read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "IntPingPongDecodeSm90")
    values = [n.value.value for n in cls.body
              if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
              and any(isinstance(t, ast.Name) and t.id == "code_pool_order" for t in n.targets)]
    assert values == ["chunk16"]
