# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The SM90 code-pool layout at head dimensions 128 and 256, on a CPU.

The owner asked for D=256 in the unit tests only, not in the paper and not in a benchmark
(2026-09-25). What these tests exercise is everything the LAYOUT defines at both head
dimensions and both widths, for K and for V:

- the one pool geometry in `_sm90/serving_cache.py` (CODE POOL GEOMETRY), which the kernel,
  IntAppend, the packer and the adapter all call, and which must give today's numbers at D=128;
- the host packer `scripts/probes/repack_rs.py`, fed the committed fragment map at D=128 and
  the closed-form map `wgmma_a_frag_map` at D=256, against a reference that reads every
  logical value back out of the chunk16 pool by its own mapping, and at D=128 against the bytes
  the two-half packer wrote before #101;
- the chunk16 load addresses under the suite's shared-memory bank model, `_smem_banks.py`;
- the pool validator, which must refuse a D=256 row-order pool and accept a D=256 chunk16 one.

THE KERNEL'S OWN D=256 PATH IS NOT EXERCISED HERE. Its grouped body is checked on an H100 by
the SASS census, the numerics gate, the output anchor and the serving round trip; nothing in
this file compiles or launches a kernel. The last test only pins, where the DSL is installed,
that the D=256 kernel reads the rows this layout defines, one 128-channel group at a time.
The file also holds IntAppend's scatter maps to the packer, at both head dimensions.

`serving_cache.py` is loaded from its file with stub parent packages, as in
`test_sm90_code_pool_order_cpu.py`, because importing the package imports the DSL.
`repack_rs.py` imports the DSL at module level for its probe kernel, so it is loaded with
permissive stand-ins for those modules. The host packer under test calls none of them: it is
torch and `serving_cache` only.
"""
import importlib.util
import pathlib
import sys
import types

import pytest
import torch

from ._smem_banks import wavefronts

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = 64
HEAD_DIMS = (128, 256)


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sc(monkeypatch):
    for pkg in ("cutedsl_decode", "cutedsl_decode._sm90"):
        stub = types.ModuleType(pkg)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, pkg, stub)
    _load(monkeypatch, "cutedsl_decode.quant_layout",
          ROOT / "src/cutedsl_decode/quant_layout.py")
    return _load(monkeypatch, "cutedsl_decode._sm90.serving_cache",
                 ROOT / "src/cutedsl_decode/_sm90/serving_cache.py")


class _Permissive(types.ModuleType):
    """A stand-in DSL module: every attribute is a submodule or this module, and calling it
    returns the decorated function, so the probe kernel's decorators and constants resolve
    at import. Anything that actually used the DSL would get nonsense, and nothing here does."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return sys.modules.get(f"{self.__name__}.{name}", self)

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return self


_DSL_MODULES = ("cutlass", "cutlass.cute", "cutlass.cute.runtime", "cutlass.cute.nvgpu",
                "cutlass.utils", "cutlass.utils.hopper_helpers", "cutlass.cutlass_dsl",
                "cutlass._mlir", "cutlass._mlir.dialects", "cutedsl_decode._vendor",
                "cutedsl_decode._vendor.quack", "tvm_ffi")


@pytest.fixture
def pack(sc, monkeypatch):
    for name in _DSL_MODULES:
        monkeypatch.setitem(sys.modules, name, _Permissive(name))
    return _load(monkeypatch, "repack_rs", ROOT / "scripts/probes/repack_rs.py")


@pytest.fixture
def committed_maps():
    return torch.load(ROOT / "scripts/frag_maps_wgmma_k16.pt")


# ---------------------------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("bits", [2, 4])
def test_the_row_bytes_follow_the_formula_and_d128_is_unchanged(sc, head_dim, bits):
    """K: (D/16) k-steps x 8 values x bits. V: (D/64) output halves x (page/16) x 8 x bits."""
    per_k, per_v = sc.code_row_bytes(head_dim, PAGE, bits)
    assert per_k == (head_dim // 16) * 8 * bits // 8
    assert per_v == (head_dim // 64) * (PAGE // 16) * 8 * bits // 8
    assert sc.code_v_half_bytes(PAGE, bits) * (head_dim // 64) == per_v
    for per in (per_k, per_v):
        assert sc.code_row_chunks(per) == per // 16
        assert sc.code_block_shape("chunk16", per) == (per // 16, 128, 16)
        assert sc.code_pool_shape("chunk16", 7, 8, per) == (7, 8, per // 16, 128, 16)
        assert sc.code_pool_shape("row", 7, 8, per) == (7, 8, 128, per)
    # The numbers the kernel and IntAppend computed before the helper existed (e4ab066's
    # `(head_dim // 16) * 8 * bits // 8` and `2 * (page_size // 16) * 8 * bits // 8`).
    if head_dim == 128:
        assert (per_k, per_v) == ((head_dim // 16) * 8 * bits // 8,
                                  2 * (PAGE // 16) * 8 * bits // 8)
    assert {(128, 4): (32, 32), (128, 2): (16, 16),
            (256, 4): (64, 64), (256, 2): (32, 32)}[(head_dim, bits)] == (per_k, per_v)


def test_a_geometry_the_layout_does_not_define_is_refused(sc):
    for head_dim, page, bits, match in ((64, 64, 2, "16-byte chunks"),   # an 8-byte int2 K row
                                        (192, 64, 2, "16-byte chunks"),  # a 24-byte int2 row
                                        (96, 64, 4, "multiple of 64"),
                                        (128, 64, 3, "bits must be 2 or 4"),
                                        (128, 48, 2, "multiple of 32")):
        with pytest.raises(ValueError, match=match):
            sc.code_row_bytes(head_dim, page, bits)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("bits", [2, 4])
def test_chunk_c_of_thread_t_is_at_16t_plus_2048c(sc, head_dim, bits):
    for per in sc.code_row_bytes(head_dim, PAGE, bits):
        chunks = sc.code_row_chunks(per)
        seen = set()
        for t in range(128):
            for c in range(chunks):
                off = sc.code_chunk_offset("chunk16", per, t, c)
                assert off == 16 * t + 2048 * c
                assert sc.code_chunk_offset("row", per, t, c) == per * t + 16 * c
                seen.add(off)
        # The chunks tile the block exactly once.
        assert seen == set(range(0, 128 * per, 16))
        with pytest.raises(ValueError, match="no chunk"):
            sc.code_chunk_offset("chunk16", per, 0, chunks)
        with pytest.raises(ValueError, match="no chunk"):
            sc.code_chunk_offset("chunk16", per, 128, 0)


# ---------------------------------------------------------------------------------------------
# The fragment map
# ---------------------------------------------------------------------------------------------

def test_the_closed_form_map_is_the_committed_map(pack, committed_maps):
    """At D=128 the closed form reproduces `scripts/frag_maps_wgmma_k16.pt` exactly, dtype
    included, for K (8 k-steps) and for V^T (4 k-steps)."""
    assert torch.equal(pack.wgmma_a_frag_map(8), committed_maps["A_qk"])
    assert torch.equal(pack.wgmma_a_frag_map(4), committed_maps["A_pv"])
    # `torch.equal` promotes across dtypes, so the dtype is checked on its own.
    assert pack.wgmma_a_frag_map(8).dtype == committed_maps["A_qk"].dtype == torch.int32
    assert pack.wgmma_a_frag_map(4).dtype == committed_maps["A_pv"].dtype == torch.int32
    # Extending in k only moves the column by 16 per k-step, which is the relation the
    # repository recorded for the committed map, so the D=256 map starts with the D=128 one.
    a_qk = committed_maps["A_qk"]
    for kt in range(8):
        assert torch.equal(a_qk[:, kt, :, 1], a_qk[:, 0, :, 1] + 16 * kt)
        assert torch.equal(a_qk[:, kt, :, 0], a_qk[:, 0, :, 0])
    assert torch.equal(pack.wgmma_a_frag_map(16)[:, :8], a_qk)


# ---------------------------------------------------------------------------------------------
# The packer round trip
# ---------------------------------------------------------------------------------------------

#: The m16n8k16 A fragment of one warp (PTX ISA, the register fragment of wgmma .m64nNk16
#: with A in registers, which stacks four of them): element s of lane l sits at row
#: l // 4 + _ROW[s] of its warp's 16-row slab and column 2 * (l % 4) + _COL[s] of its
#: 16-column k-step. Written as a table, independently of the packer's closed form.
_ROW = (0, 0, 8, 8, 0, 0, 8, 8)
_COL = (0, 1, 0, 1, 8, 9, 8, 9)


def _fragment(k_steps):
    """Columns (thread, k-step, element, m, k) of every A-fragment entry, one row each."""
    rows = []
    for t in range(128):
        warp, lane = divmod(t, 32)
        for kt in range(k_steps):
            for s in range(8):
                rows.append((t, kt, s, 16 * warp + lane // 4 + _ROW[s],
                             16 * kt + 2 * (lane % 4) + _COL[s]))
    return torch.tensor(rows).unbind(-1)


def _plane_word(kt, s, bits):
    """(row word, bit) of entry (k-step, element) in a run of plane words, which is the whole K
    row or one V output half: int4 has a word per k-step, int2 a word per k-step pair, and the
    low halfword holds the even elements."""
    j, hi = s // 2, s % 2
    if bits == 4:
        return kt, 4 * j + 16 * hi
    return kt // 2, 2 * (4 * (kt % 2) + j) + 16 * hi


def _read(pool, t, word, bit, bits):
    """The value at (thread t, row word, bit) of a chunk16 pool [..., C, 128, 16], addressed by
    the order's definition: chunk c of thread t is row bytes [16c, 16c+16)."""
    chunk, in_chunk = word // 4, word % 4
    by = pool.long()
    value = sum(by[..., chunk, t, 4 * in_chunk + i] << (8 * i) for i in range(4))
    return (value >> bit) & ((1 << bits) - 1)


def _logical_tiles(x):
    """[B, KV, HKV, D] -> [B, NT, HKV, 64 tokens, D]."""
    B, KV, HKV, D = x.shape
    return x.reshape(B, KV // PAGE, PAGE, HKV, D).permute(0, 1, 3, 2, 4).long()


def _decode_k(pool, head_dim, bits):
    t, kt, s, token, channel = _fragment(head_dim // 16)
    word, bit = _plane_word(kt, s, bits)
    out = torch.full(pool.shape[:3] + (PAGE, head_dim), -1, dtype=torch.long)
    out[..., token, channel] = _read(pool, t, word, bit, bits)
    return out, token, channel


def _decode_v(pool, head_dim, bits):
    t, kt, s, m, token = _fragment(PAGE // 16)
    word, bit = _plane_word(kt, s, bits)
    words_per_half = PAGE // 16 if bits == 4 else PAGE // 32
    out = torch.full(pool.shape[:3] + (PAGE, head_dim), -1, dtype=torch.long)
    tokens, channels = [], []
    for h in range(head_dim // 64):
        out[..., token, 64 * h + m] = _read(pool, t, h * words_per_half + word, bit, bits)
        tokens.append(token)
        channels.append(64 * h + m)
    return out, torch.cat(tokens), torch.cat(channels)


def _covers_every_value_once(token, channel, head_dim):
    seen = torch.zeros(PAGE, head_dim, dtype=torch.long)
    seen.index_put_((token, channel), torch.ones_like(token), accumulate=True)
    return bool((seen == 1).all())


def _chunk_rule_holds(sc, pool, rows, per):
    """The chunk16 bytes are the row-order bytes moved by the chunk rule, one 16-byte chunk at
    a time, at the offsets `code_chunk_offset` gives for each order."""
    lead = pool.shape[:3]
    chunked, row = pool.reshape(*lead, -1), rows.reshape(*lead, -1)
    for t in range(128):
        for c in range(sc.code_row_chunks(per)):
            a = sc.code_chunk_offset("chunk16", per, t, c)
            r = sc.code_chunk_offset("row", per, t, c)
            if not torch.equal(chunked[..., a:a + 16], row[..., r:r + 16]):
                return False
    return True


def _codes(bits, head_dim, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1 << bits, (2, 2 * PAGE, 3, head_dim), generator=g,
                         dtype=torch.uint8)


def _k_map(pack, committed_maps, head_dim):
    """The production map at D=128, the closed form at D=256 (the committed one stops there)."""
    return committed_maps["A_qk"] if head_dim == 128 else pack.wgmma_a_frag_map(head_dim // 16)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("bits", [2, 4])
def test_k_packs_into_chunk16_and_every_value_comes_back(sc, pack, committed_maps, head_dim,
                                                         bits):
    """K at D in {128, 256}: the chunk16 pool holds every logical value where the fragment
    layout says, and its bytes are the row-order bytes permuted by the chunk rule. The
    kernel's D=256 path is exercised on an H100 (the module docstring), not here."""
    k_int = _codes(bits, head_dim, seed=bits + head_dim)
    per_k, _ = sc.code_row_bytes(head_dim, PAGE, bits)
    m = _k_map(pack, committed_maps, head_dim)
    pool = pack.repack_K_rs(k_int, m, bits, order="chunk16")
    rows = pack.repack_K_rs(k_int, m, bits, order="row")
    assert pool.shape == (2, 2, 3) + sc.code_block_shape("chunk16", per_k) and pool.is_contiguous()
    assert rows.shape == (2, 2, 3) + sc.code_block_shape("row", per_k)
    got, token, channel = _decode_k(pool, head_dim, bits)
    assert _covers_every_value_once(token, channel, head_dim)
    assert torch.equal(got, _logical_tiles(k_int))
    assert _chunk_rule_holds(sc, pool, rows, per_k)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("bits", [2, 4])
def test_v_packs_into_chunk16_and_every_value_comes_back(sc, pack, committed_maps, head_dim,
                                                         bits):
    """V^T at D in {128, 256}: D/64 output halves, four at D=256, with the same per-half map.
    The kernel's D=256 path is exercised on an H100 (the module docstring), not here."""
    v_int = _codes(bits, head_dim, seed=10 + bits + head_dim)
    _, per_v = sc.code_row_bytes(head_dim, PAGE, bits)
    pool = pack.repack_V_rs(v_int, committed_maps["A_pv"], bits, order="chunk16")
    rows = pack.repack_V_rs(v_int, committed_maps["A_pv"], bits, order="row")
    assert pool.shape == (2, 2, 3) + sc.code_block_shape("chunk16", per_v) and pool.is_contiguous()
    got, token, channel = _decode_v(pool, head_dim, bits)
    assert _covers_every_value_once(token, channel, head_dim)
    assert torch.equal(got, _logical_tiles(v_int))
    assert _chunk_rule_holds(sc, pool, rows, per_v)


@pytest.mark.parametrize("head_dim,bits", [(128, 4), (256, 2), (256, 4)])
def test_the_round_trip_fails_on_a_row_pool_viewed_as_chunk16(sc, pack, committed_maps,
                                                              head_dim, bits):
    """The negative control: row-order bytes reshaped to the chunk16 shape (the one mistake no
    shape check sees) decode wrong wherever a row has more than one chunk, so the round trip
    above would fail on that layout."""
    k_int = _codes(bits, head_dim, seed=3)
    v_int = _codes(bits, head_dim, seed=4)
    per_k, per_v = sc.code_row_bytes(head_dim, PAGE, bits)
    rows_k = pack.repack_K_rs(k_int, _k_map(pack, committed_maps, head_dim), bits, order="row")
    rows_v = pack.repack_V_rs(v_int, committed_maps["A_pv"], bits, order="row")
    lying_k = rows_k.reshape((2, 2, 3) + sc.code_block_shape("chunk16", per_k))
    lying_v = rows_v.reshape((2, 2, 3) + sc.code_block_shape("chunk16", per_v))
    assert not torch.equal(_decode_k(lying_k, head_dim, bits)[0], _logical_tiles(k_int))
    assert not torch.equal(_decode_v(lying_v, head_dim, bits)[0], _logical_tiles(v_int))
    assert not _chunk_rule_holds(sc, lying_k, rows_k, per_k)
    assert not _chunk_rule_holds(sc, lying_v, rows_v, per_v)


#: These are the sha256 values of the bytes that the packer wrote before the D-generic V packer
#: of #101. They were taken from `6d4a06e^:scripts/probes/repack_rs.py` (dda3f89), whose
#: `repack_V_rs` wrote the two halves `(0, 1)` and, like its `repack_K_rs`, returned the row
#: order. The keys are (bits, tensor), and the fixture is `_arithmetic_codes`. The values were
#: derived on 2026-09-30 with torch 2.13.0 by
#: `tmp/codex_evidence/planB_prB_20260930/derive/derive_packer_hashes.py` in the owner's
#: working copy of the paper repository, which loads both packers as the `pack` fixture does.
PRE_101_SHA256 = {
    (2, "K"): "992192c3b610141c04a90acaf29404f7790b26da0d3346fee778e3996df43e2c",
    (2, "V"): "7157deb1c73ed33cefcac46ed5a50388062a69a207112e9deee4175c6ff92a5d",
    (4, "K"): "5d576c3145f9cdeddf6cad3bb9781601361e1f818053b9987c1a4ffdd1c28f41",
    (4, "V"): "bcd18b2719745fddcf1f73ffe2dd0930b6978eba7ad3f6dde126a290fce9225b",
}


def _arithmetic_codes(bits):
    """Return a [B=2, KV=128, HKV=3, D=128] code tensor built from integer arithmetic alone,
    so that its bytes, and the hashes above, depend on neither a random number generator nor
    the torch version."""
    i = torch.arange(2 * 128 * 3 * 128, dtype=torch.int64)
    return (((i * 2654435761) >> 13) & ((1 << bits) - 1)).to(torch.uint8).reshape(2, 128, 3, 128)


@pytest.mark.parametrize("bits", [2, 4])
def test_the_packer_writes_the_pre_101_bytes_at_d128(sc, pack, committed_maps, bits):
    """At D=128 the D-generic packer writes, in the row order, the bytes the two-half packer
    wrote before #101, for K and for V, and its chunk16 output is that row output moved by the
    chunk rule. The reference is the old implementation, not the code under test."""
    import hashlib

    x = _arithmetic_codes(bits)
    per_k, per_v = sc.code_row_bytes(128, PAGE, bits)
    written = {
        "K": (pack.repack_K_rs(x, committed_maps["A_qk"], bits, order="row"),
              pack.repack_K_rs(x, committed_maps["A_qk"], bits, order="chunk16"), per_k),
        "V": (pack.repack_V_rs(x, committed_maps["A_pv"], bits, order="row"),
              pack.repack_V_rs(x, committed_maps["A_pv"], bits, order="chunk16"), per_v),
    }
    for tensor, (rows, chunked, per) in written.items():
        assert tuple(rows.shape) == (2, 2, 3, 128, per), (tensor, tuple(rows.shape))
        digest = hashlib.sha256(rows.contiguous().numpy().tobytes()).hexdigest()
        assert digest == PRE_101_SHA256[(bits, tensor)], (tensor, digest)
        assert torch.equal(chunked, sc.code_rows_to_order(rows, "chunk16")), tensor


def test_a_map_that_does_not_cover_the_head_dimension_is_refused(pack, committed_maps):
    """The committed A_qk covers 8 k-steps. At D=256 it would pack channels 0-127 only, so the
    packer raises rather than asserts (`python -O` drops an assert)."""
    k_int = _codes(4, 256, seed=5)
    with pytest.raises(ValueError, match="all 16 k-steps"):
        pack.repack_K_rs(k_int, committed_maps["A_qk"], 4, order="chunk16")
    with pytest.raises(ValueError, match="all 4 k-steps"):
        pack.repack_V_rs(_codes(4, 256, seed=6), pack.wgmma_a_frag_map(8), 4, order="chunk16")


# ---------------------------------------------------------------------------------------------
# Shared-memory banks
# ---------------------------------------------------------------------------------------------

#: Row-order wavefronts per load against an ideal of 4. D=128 INT4 is the 8/4 that ncu
#: measured on e4ab066 (profile/h100_chunk16_bankconf_20260925/), where chunk16 measured 4/4.
ROW_ORDER_ACTUAL = {(128, 2): 4, (128, 4): 8, (256, 2): 8, (256, 4): 16}


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("bits", [2, 4])
def test_every_chunk16_load_is_conflict_free_and_the_row_order_is_not(sc, head_dim, bits):
    """Every 16-byte code-ring load the helper's offsets give, for K and V, both warpgroup
    halves of the ring's first two stages, costed with every lane active by the suite's bank
    model, which serves a 16-byte load one 8-lane quarter warp at a time: 4 wavefronts against
    4 ideal in chunk16, and the row order conflicted wherever a row is longer than one chunk,
    so this test fails if the layout regresses. A stage is one block and starts on a 128-byte
    boundary (the ring is 128-byte aligned and a block is a multiple of 2048 bytes), so it
    moves no bank."""
    for tensor, per in zip("KV", sc.code_row_bytes(head_dim, PAGE, bits)):
        for order, want in (("chunk16", 4), ("row", ROW_ORDER_ACTUAL[(head_dim, bits)])):
            for stage in (0, 1):
                for warp in range(4):
                    for c in range(sc.code_row_chunks(per)):
                        addresses = [stage * 128 * per
                                     + sc.code_chunk_offset(order, per, 32 * warp + lane, c)
                                     for lane in range(32)]
                        got = wavefronts(list(enumerate(addresses)), 16, "ld")
                        assert got == (want, 4), (tensor, order, stage, warp, c, got)


# ---------------------------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------------------------

def _pools(sc, pack, head_dim, bits, order):
    """A shuffled two-request ServingCache of packed K and V at `head_dim`, in `order`."""
    k_int, v_int = _codes(bits, head_dim, seed=7), _codes(bits, head_dim, seed=8)
    kq = pack.repack_K_rs(k_int, pack.wgmma_a_frag_map(head_dim // 16), bits, order=order)
    vq = pack.repack_V_rs(v_int, pack.wgmma_a_frag_map(PAGE // 16), bits, order=order)
    scales = torch.rand(2, 3, 2 * PAGE, 4)
    return sc.make_serving_cache(kq, vq, scales, torch.tensor([2 * PAGE, PAGE + 5]), N=PAGE)


@pytest.mark.parametrize("bits", [2, 4])
def test_a_d256_row_pool_is_refused_and_a_d256_chunk16_pool_is_accepted(sc, pack, bits):
    """The validator the kernel's `__call__` and `launch_pingpong` call, at D=256, without
    compiling or launching the kernel, whose D=256 path is exercised on an H100 (the module
    docstring)."""
    per_k, per_v = sc.code_row_bytes(256, PAGE, bits)
    good = _pools(sc, pack, 256, bits, "chunk16")
    row = _pools(sc, pack, 256, bits, "row")
    for name, pool, per in (("mKq", good.codes_k, per_k), ("mVq", good.codes_v, per_v)):
        assert tuple(pool.shape) == sc.code_pool_shape("chunk16", pool.shape[0], 3, per)
        sc.require_code_pool(name, pool.shape, "chunk16", per, stride=pool.stride())
        sc.require_code_pool(name, pool.shape, "chunk16", stride=pool.stride())
    for name, pool, per in (("mKq", row.codes_k, per_k), ("mVq", row.codes_v, per_v)):
        assert tuple(pool.shape) == sc.code_pool_shape("row", pool.shape[0], 3, per)
        with pytest.raises(ValueError) as e:
            sc.require_code_pool(name, pool.shape, "chunk16", per, stride=pool.stride())
        assert str(e.value).startswith(name) and "'row'-order pool" in str(e.value)
        assert f"{per // 16}, 128, 16" in str(e.value)
        with pytest.raises(ValueError, match="reads the 'chunk16'"):
            sc.require_code_pool(name, pool.shape, "chunk16", stride=pool.stride())
    # A D=256 chunk16 pool is not a D=128 one: the row bytes the D=128 reader expects differ.
    per_k128, _ = sc.code_row_bytes(128, PAGE, bits)
    with pytest.raises(ValueError, match=f"{per_k128 // 16}, 128, 16"):
        sc.require_code_pool("mKq", good.codes_k.shape, "chunk16", per_k128,
                             stride=good.codes_k.stride())


# ---------------------------------------------------------------------------------------------
# IntAppend's scatter maps
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def scatter(pack, monkeypatch):
    """`scripts/probes/append_scatter.py`, which imports the packer (`pack`) and the pool
    geometry (`sc`) and nothing of the DSL itself."""
    return _load(monkeypatch, "append_scatter", ROOT / "scripts/probes/append_scatter.py")


def test_each_head_dimension_names_its_own_scatter_file(sc):
    assert sc.scatter_map_file(128) == "frag_scatter_k16.pt"
    assert sc.scatter_map_file(256) == "frag_scatter_k16_d256.pt"
    for bad in (0, 96):
        with pytest.raises(ValueError, match="no scatter map"):
            sc.scatter_map_file(bad)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_the_committed_scatter_map_is_the_builders_output(scatter, head_dim):
    """The committed file at each head dimension is what `append_scatter.build_scatter_maps`
    builds from `frag_maps(head_dim)`, so the D=128 file is unchanged by the builder's
    generalization and the D=256 file is the builder's, not a hand edit."""
    saved = torch.load(scatter.out_path(head_dim))
    scK, scV = scatter.build_scatter_maps(head_dim)
    assert set(saved["scatter_K"]) == set(saved["scatter_V"]) == {2, 4}
    for bits in (2, 4):
        assert saved["scatter_K"][bits].shape == (64, head_dim, 3)
        assert torch.equal(saved["scatter_K"][bits], scK[bits]), bits
        assert torch.equal(saved["scatter_V"][bits], scV[bits]), bits


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("bits", [2, 4])
def test_the_committed_scatter_map_inverts_the_packer(sc, pack, scatter, head_dim, bits):
    """Scattering every token's codes with the committed map, by the host model of IntAppend's
    RMW-OR (`scatter_pool`), rebuilds the packer's thread rows byte for byte, K and every V^T
    half, at D=128 and D=256."""
    saved = torch.load(scatter.out_path(head_dim))
    maps = pack.frag_maps(head_dim)
    per_k, per_v = sc.code_row_bytes(head_dim, PAGE, bits)
    k_int, v_int = _codes(bits, head_dim, seed=21), _codes(bits, head_dim, seed=22)
    want_k = pack.repack_K_rs(k_int, maps["A_qk"], bits, order="row")
    want_v = pack.repack_V_rs(v_int, maps["A_pv"], bits, order="row")
    got_k = scatter.scatter_pool(k_int, saved["scatter_K"][bits], bits, per_k, per_k // 4)
    got_v = scatter.scatter_pool(v_int, saved["scatter_V"][bits], bits, per_v, per_v // 4)
    assert torch.equal(got_k, want_k)
    assert torch.equal(got_v, want_v)


def test_frag_maps_is_the_committed_file_at_128_and_the_closed_form_elsewhere(pack,
                                                                           committed_maps):
    at128 = pack.frag_maps(128)
    assert torch.equal(at128["A_qk"], committed_maps["A_qk"])
    assert torch.equal(at128["A_pv"], committed_maps["A_pv"])
    at256 = pack.frag_maps(256)
    assert torch.equal(at256["A_qk"], pack.wgmma_a_frag_map(16))
    assert torch.equal(at256["A_pv"], committed_maps["A_pv"])


needs_dsl = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None,
    reason="constructing the kernel imports the CuTe DSL (cutlass), absent here")


@needs_dsl
@pytest.mark.parametrize("bits", [2, 4])
def test_the_d256_kernel_reads_these_rows_one_group_at_a_time(sc, bits):
    """The D=256 kernel's row bytes are the layout's, and each 128-channel wgmma group of its
    grouped body reads one or two whole chunks of a row: two at int4 (8 words), one at int2."""
    from cutedsl_decode._sm90.int_decode_pingpong import IntPingPongDecodeSm90
    kern = IntPingPongDecodeSm90(256, bits=bits, heads_per_kv=4)
    per_k, per_v = sc.code_row_bytes(256, PAGE, bits)
    assert (kern.k_code_bytes_per_thread, kern.v_code_bytes_per_thread) == (per_k, per_v)
    assert kern.group_channels == 128
    for per in (per_k, per_v):
        assert kern._group_word_count(per) == 2 * bits == sc.code_row_chunks(per) * 4 // 2
