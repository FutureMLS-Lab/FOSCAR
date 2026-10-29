# tests/test_public_api.py
"""The public surface: exports, and the cache layout and cache objects.

`QuantKVLayout` is resolved against the reader of the device's architecture, so most of
these tests name the architecture through the private `_create` and need no GPU; the ones
that exercise the device resolution itself stand in for `torch.cuda`. Building a layout
imports the reader classes, so the DSL must be installed (no GPU is used).
"""
import contextlib
import dataclasses

import pytest
import torch

import fosca as cd
from fosca import _utils, kv_cache
from fosca.kv_cache import QuantKVCache, QuantKVLayout

D = 128


def test_public_surface():
    assert hasattr(cd, "PagedDecodeWrapper")
    assert hasattr(cd, "get_capabilities")
    assert hasattr(cd, "is_supported")
    assert cd.QuantKVLayout is QuantKVLayout
    assert cd.QuantKVCache is QuantKVCache
    assert isinstance(cd.__version__, str)
    # arch-specific names must NOT be exported
    assert not hasattr(cd, "GroupedQueryAttentionDecodePagedSm90")
    for name in getattr(cd, "__all__", []):
        assert "Sm90" not in name and "Sm100" not in name, name


def _layout(arch, bits=4, **kw):
    kw.setdefault("head_dim", D)
    kw.setdefault("num_kv_heads", 8)
    return QuantKVLayout._create(arch, bits=bits, **kw)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Stand in for `torch.cuda`: `caps[i]` is device i's capability, `current[0]` the current
    device. The capability cache is cleared before and after."""
    caps, current = {}, [0]
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: caps[i])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    _utils._cc.cache_clear()
    yield caps, current
    _utils._cc.cache_clear()


# -- the layout ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 4])
def test_production_orders_are_the_readers_answers(bits):
    from fosca._sm100.int_fold_decode import IntDecodeSm100
    from fosca._sm90.int_decode_pingpong import IntPingPongDecodeSm90

    h = _layout("sm90", bits)
    assert (h.reader, h.code_pool_order) == ("IntPingPongDecodeSm90",
                                             IntPingPongDecodeSm90.code_pool_order)
    assert (h.k_pool_permuted, h.v_pool_permuted) == (None, None)
    b = _layout("sm100", bits)
    assert (b.reader, b.code_pool_order) == ("IntDecodeSm100", None)
    assert b.k_pool_permuted is IntDecodeSm100.k_pool_is_permuted(bits) is True
    assert b.v_pool_permuted is IntDecodeSm100.v_pool_token_rows(bits) is True
    for layout in (h, b):
        assert layout.k_quant == layout.v_quant == ("token", D)
        assert layout.scale_dtype is torch.float32
        assert layout._kernel_kwargs == ()


@pytest.mark.parametrize("kernel_kwargs", [{"dequant_int2_to_f4": 0},
                                           {"fold_scale_zero": 0}])
def test_an_int2_ablation_layout_carries_the_order_its_feed_reads(kernel_kwargs):
    layout = _layout("sm100", 2, kernel_kwargs=kernel_kwargs)
    assert (layout.k_pool_permuted, layout.v_pool_permuted) == (False, False)
    assert layout != _layout("sm100", 2)
    assert dict(layout._kernel_kwargs) == kernel_kwargs


def test_the_orders_cannot_be_passed_in():
    good = _layout("sm100", 2)
    for field, value in (("k_pool_permuted", False), ("v_pool_permuted", False),
                         ("reader", "IntPingPongDecodeSm90"), ("code_pool_order", "row")):
        with pytest.raises(ValueError, match="built by QuantKVLayout.create"):
            dataclasses.replace(good, **{field: value})
    with pytest.raises(dataclasses.FrozenInstanceError):
        good.k_pool_permuted = False
    with pytest.raises(TypeError):
        QuantKVLayout.create(bits=2, head_dim=D, num_kv_heads=8,
                             kernel_kwargs={"dequant_int2_to_f4": 0})


def test_identical_pool_shapes_still_compare_unequal():
    a = _layout("sm100", 2, k_quant=("channel", 64))
    b = _layout("sm100", 2, k_quant=("channel", 128))
    assert a.pool_shapes(10) == b.pool_shapes(10)
    assert a != b
    assert _layout("sm100", 2, k_quant=["channel", 64]) == a, "a list spelling normalizes"


@pytest.mark.parametrize("arch,bits,kw,match", [
    ("sm100", 4, {"head_dim": 384}, "head_dim 384 is not accepted on sm100"),
    ("sm90", 4, {"head_dim": 64}, "head_dim 64 is not accepted on sm90"),
    ("sm90", 4, {"head_dim": 256, "scale_dtype": torch.bfloat16}, "f32 scale pool"),
    ("sm90", 4, {"head_dim": 256, "k_quant": ("channel", 64)}, "token-wise K and V"),
    ("sm90", 2, {"k_quant": ("token", 64), "scale_dtype": torch.float32}, "bfloat16 scale"),
    ("sm100", 2, {"k_quant": ("token", 64), "scale_dtype": torch.float32}, "bfloat16"),
    ("sm100", 2, {"scale_dtype": torch.bfloat16}, "float32"),
    ("sm100", 2, {"k_quant": ("channel", 32)}, "outside the decode's envelope"),
    ("sm90", 2, {"v_quant": ("token", 64)}, "outside the decode's envelope"),
    ("sm100", 3, {}, "bits must be one of"),
    ("sm100", 2.0, {}, "bits must be one of"),
    ("sm90", True, {}, "bits must be one of"),
    ("sm100", 2, {"num_kv_heads": 0}, "num_kv_heads must be a positive int"),
    ("sm100", 2, {"scale_dtype": torch.float16}, "scale_dtype must be one of"),
    ("sm89", 2, {}, "arch must be one of"),
])
def test_create_refuses_what_the_reader_refuses(arch, bits, kw, match):
    with pytest.raises(ValueError, match=match):
        _layout(arch, bits, **kw)


def test_hopper_takes_either_scale_dtype_at_128_and_blackwell_one_per_pair():
    assert _layout("sm90", 4, scale_dtype=torch.bfloat16).scale_dtype is torch.bfloat16
    assert _layout("sm90", 2, k_quant=("token", 64)).scale_dtype is torch.bfloat16
    assert _layout("sm100", 2, k_quant=("token", 64)).scale_dtype is torch.bfloat16
    assert _layout("sm100", 2, v_quant=("channel", 64)).scale_dtype is torch.float32
    assert _layout("sm90", 4, head_dim=256).scale_dtype is torch.float32


@pytest.mark.parametrize("bits", [2, 4])
def test_blackwell_d256_has_only_the_whole_head_eager_layout(bits):
    from fosca._append_core import envelope_refusal
    layout = _layout("sm100", bits, head_dim=256)
    assert layout.k_quant == layout.v_quant == ("token", 256)
    assert layout.scale_dtype is torch.float32
    assert layout.pool_shapes(2)["k_codes"] == (2, 8, 64, 256 * bits // 8)
    # Decode admission does not expand append/dequantize's separate envelope.
    assert envelope_refusal(layout, layout.arch) is not None
    for kw in ({"k_quant": ("token", 128)}, {"v_quant": ("token", 128)},
               {"k_quant": ("channel", 64)}, {"scale_dtype": torch.bfloat16},
               {"kernel_kwargs": {"fold_scale_zero": 0}}):
        with pytest.raises(ValueError, match="whole-head"):
            _layout("sm100", bits, head_dim=256, **kw)


@pytest.mark.parametrize("arch,kernel_kwargs,error,match", [
    ("sm100", {"k_quant": ("token", D)}, ValueError, "layout fields"),
    ("sm100", {"sdtype": "f32"}, ValueError, "layout fields"),
    ("sm100", {"kernel_cls": "pingpong"}, ValueError, "IntDecodeSm100 refuses"),
    ("sm100", {"identity_dequant": False}, ValueError, "IntDecodeSm100 refuses"),
    ("sm100", {"dequant_warpgroups": 1}, ValueError, "IntDecodeSm100 refuses"),
    ("sm100", {"o_stages": [2]}, ValueError, "hashable"),
    # The reader's own constructor refuses these, and the layout asks it.
    ("sm100", {"o_stages": 4}, ValueError, "IntDecodeSm100 refuses"),
    ("sm100", {"code_stages": 2}, ValueError, "IntDecodeSm100 refuses"),
    ("sm100", {"prediction_tile": 2}, ValueError, "IntDecodeSm100 refuses"),
    ("sm100", {"fused_correction": False}, ValueError, "IntDecodeSm100 refuses"),
    ("sm100", {"n_tile": 64}, ValueError, "IntDecodeSm100 refuses"),
    ("sm100", {"not_a_knob": 0}, ValueError, "IntDecodeSm100 refuses"),
    ("sm90", {"fused_combine": "yes"}, ValueError, "must be a bool"),
    ("sm90", {"fused_combine": 1}, ValueError, "must be a bool"),
    ("sm90", {"fused_combine": 0.0}, ValueError, "must be a bool"),
    ("sm90", {"sdtype": torch.bfloat16}, ValueError, "fused_combine"),
    ("sm90", {"fold_scale_zero": False}, ValueError, "fused_combine"),
])
def test_harness_kernel_kwargs_are_checked(arch, kernel_kwargs, error, match):
    with pytest.raises(error, match=match):
        _layout(arch, 2, kernel_kwargs=kernel_kwargs)


@pytest.mark.parametrize("bits,k_quant,kernel_kwargs", [
    (4, None, {"fold_scale_zero": 0}),
    (4, None, {"dequant_int2_to_f4": 1}),
    (2, ("channel", 64), {"fold_scale_zero": "0"}),
])
def test_a_feed_the_configuration_does_not_take_is_refused(bits, k_quant, kernel_kwargs):
    with pytest.raises(ValueError, match="IntDecodeSm100 refuses"):
        _layout("sm100", bits, k_quant=k_quant, kernel_kwargs=kernel_kwargs)


def test_int4_scale_stages_16_is_refused_on_blackwell():
    with pytest.raises(ValueError, match="scale_stages=16 at INT4"):
        _layout("sm100", 4, kernel_kwargs={"scale_stages": 16})
    assert dict(_layout("sm100", 2, kernel_kwargs={"scale_stages": 16})._kernel_kwargs) == {
        "scale_stages": 16}


@pytest.mark.parametrize("arch,bits,want", [
    # D=128, HKV=8: a 4-bit K and V slab is 4096 bytes each over the 64-token page, the
    # scale block 1024 bytes, and Blackwell's page maxima 4 bytes per KV head.
    ("sm90", 4, 8 * (4096 + 4096 + 1024)),
    ("sm90", 2, 40_960),
    ("sm100", 4, 8 * (4096 + 4096 + 1024 + 4)),
    ("sm100", 2, 40_992),
])
def test_bytes_per_page(arch, bits, want):
    assert _layout(arch, bits).bytes_per_page() == want


def test_pool_shapes():
    from fosca._sm90.serving_cache import code_row_bytes

    h = _layout("sm90", 4, num_kv_heads=3).pool_shapes(5)
    per_k, per_v = code_row_bytes(D, 64, 4)
    assert h == {"k_codes": (5, 3, per_k // 16, 128, 16), "v_codes": (5, 3, per_v // 16, 128, 16),
                 "scales": (5, 3, 64, 4), "v_page_max": None}
    # Blackwell stores V exactly like K, in token rows; only the INT2 ablation layouts
    # keep the channel-row V pool their feed reads.
    b = _layout("sm100", 2, num_kv_heads=3).pool_shapes(5)
    assert b == {"k_codes": (5, 3, 64, D * 2 // 8), "v_codes": (5, 3, 64, D * 2 // 8),
                 "scales": (5, 3, 64, 4), "v_page_max": (5, 3)}
    assert _layout("sm100", 4, num_kv_heads=3).pool_shapes(5)["v_codes"] == (5, 3, 64, D * 4 // 8)
    for kernel_kwargs in ({"dequant_int2_to_f4": 0}, {"fold_scale_zero": 0}):
        ablation = _layout("sm100", 2, num_kv_heads=3, kernel_kwargs=kernel_kwargs)
        assert ablation.pool_shapes(5)["v_codes"] == (5, 3, D, 64 * 2 // 8)
    flex = _layout("sm100", 2, k_quant=("channel", 128), v_quant=("channel", 64))
    assert flex.pool_shapes(2)["scales"] == (2, 8, 4 * D) and not flex.has_v_page_max
    tok64 = _layout("sm100", 2, k_quant=("token", 64))
    assert tok64.pool_shapes(2)["scales"] == (2, 8, 4 * 64 + 2 * 64)
    assert tok64.pool_dtypes()["scales"] is torch.bfloat16


# -- device resolution --------------------------------------------------------------------

def test_the_architecture_is_a_whitelist(fake_cuda):
    caps, current = fake_cuda
    caps.update({0: (9, 0), 1: (10, 0), 2: (8, 9), 3: (10, 3), 4: (12, 0)})
    assert _utils.device_arch("cuda:0") == "sm90"
    assert _utils.device_arch("cuda:1") == "sm100"
    for index in (2, 3, 4):
        assert _utils.device_arch(f"cuda:{index}") is None
        with pytest.raises(RuntimeError, match="no kernel for compute capability"):
            QuantKVLayout.create(bits=2, head_dim=D, num_kv_heads=8, device=f"cuda:{index}")


def test_the_architecture_follows_the_current_device(fake_cuda):
    caps, current = fake_cuda
    caps.update({0: (9, 0), 1: (10, 0)})
    assert QuantKVLayout.create(bits=2, head_dim=D, num_kv_heads=8).arch == "sm90"
    current[0] = 1
    # The capability cache is keyed by the resolved ordinal, so a change of current device
    # is seen; keyed by `None` it would answer with device 0 for the rest of the process.
    layout = QuantKVLayout.create(bits=2, head_dim=D, num_kv_heads=8)
    assert layout.arch == "sm100" and layout.k_pool_permuted is True
    assert QuantKVLayout.create(bits=2, head_dim=D, num_kv_heads=8, device="cuda:0").arch == \
        "sm90"
    assert _utils.cuda_index(torch.device("cuda")) == 1
    with pytest.raises(ValueError, match="not a CUDA device"):
        _utils.cuda_index("cpu")


@pytest.fixture
def stub_dsl(monkeypatch):
    """Stand in for the CuTe DSL's compile target: returns the stub, whose `envar.arch` a
    test sets, with the per-process cache of the target cleared before, after every change
    (`retarget`) and after the test, so no stub value outlives it."""
    from cutlass.cutlass_dsl import CuTeDSL

    class Dsl:
        class envar:
            arch = "sm_100a"

    def retarget(arch):
        Dsl.envar.arch = arch
        _utils._dsl_target.cache_clear()

    Dsl.retarget = staticmethod(retarget)
    monkeypatch.setattr(CuTeDSL, "_get_dsl", staticmethod(lambda: Dsl))
    _utils._dsl_target.cache_clear()
    yield Dsl
    _utils._dsl_target.cache_clear()


def test_the_dsl_compile_target_must_be_the_architecture(stub_dsl):
    _utils.require_dsl_target("sm100")
    with pytest.raises(RuntimeError, match="compiles for sm_100a"):
        _utils.require_dsl_target("sm90")
    stub_dsl.retarget("sm_90a")
    _utils.require_dsl_target("sm90")
    with pytest.raises(RuntimeError, match="cannot build the sm100 kernel"):
        _utils.require_dsl_target("sm100")


# -- the cache ----------------------------------------------------------------------------

@pytest.fixture
def any_device(monkeypatch):
    """Let CPU tensors stand in for a device of the layout's architecture."""
    monkeypatch.setattr(kv_cache, "_require_device", lambda layout, device: None)


@pytest.mark.parametrize("arch,bits,kw", [
    ("sm90", 4, {}), ("sm90", 2, {"k_quant": ("token", 64)}),
    ("sm100", 4, {}), ("sm100", 2, {"v_quant": ("channel", 64)}),
])
def test_bind_lays_the_pools_out_region_by_region(any_device, arch, bits, kw):
    layout = _layout(arch, bits, num_kv_heads=3, **kw)
    pages = 5
    buffer = torch.full((pages * layout.bytes_per_page() + 7,), 0xA5, dtype=torch.uint8)
    cache = layout.bind(buffer, pages)
    assert (cache.layout, cache.num_pages, cache.device) == (layout, pages, buffer.device)
    start = buffer.data_ptr()
    names = ["k_codes", "v_codes", "scales"] + (["v_page_max"] if layout.has_v_page_max else [])
    shapes, dtypes = layout.pool_shapes(pages), layout.pool_dtypes()
    for name in names:
        pool = getattr(cache, name)
        assert tuple(pool.shape) == shapes[name] and pool.dtype == dtypes[name], name
        assert pool.is_contiguous() and pool.data_ptr() == start, name
        assert (pool.data_ptr() - buffer.data_ptr()) % 16 == 0, name
        start += pool.numel() * pool.element_size()
    assert start - buffer.data_ptr() == pages * layout.bytes_per_page()
    assert (cache.v_page_max is None) is (not layout.has_v_page_max)
    assert (cache.v_max_hist is None) is (not layout.has_v_page_max)
    # bind zeroes the page maxima, creates the cache's own empty count table outside the
    # buffer, and writes nothing else of the engine-owned buffer.
    maxima = 0
    if layout.has_v_page_max:
        maxima = cache.v_page_max.numel() * 4
        assert bool((cache.v_page_max.view(torch.int32) == 0).all())
        assert cache.v_max_hist.dtype == torch.uint32 and tuple(cache.v_max_hist.shape) == (256,)
        assert bool((cache.v_max_hist.view(torch.int32) == 0).all())
        assert cache.v_max_hist.data_ptr() % 16 == 0
        end = buffer.data_ptr() + buffer.numel()
        assert not buffer.data_ptr() <= cache.v_max_hist.data_ptr() < end
    used = pages * layout.bytes_per_page()
    assert bool((buffer[:used - maxima] == 0xA5).all()) and bool((buffer[used:] == 0xA5).all())
    # A 2-D engine tensor of the same bytes binds the same way.
    rows = buffer[:pages * layout.bytes_per_page()].view(pages, layout.bytes_per_page())
    assert layout.bind(rows, pages).scales.data_ptr() == cache.scales.data_ptr()


def test_bind_refusals(any_device):
    layout = _layout("sm100", 2)
    need = 4 * layout.bytes_per_page()
    with pytest.raises(ValueError, match="uint8"):
        layout.bind(torch.zeros(need // 4, dtype=torch.float32), 4)
    with pytest.raises(ValueError, match=f"needs {need} bytes"):
        layout.bind(torch.zeros(need - 1, dtype=torch.uint8), 4)
    with pytest.raises(ValueError, match="C-contiguous"):
        layout.bind(torch.zeros(2 * need, dtype=torch.uint8)[::2], 4)
    with pytest.raises(ValueError, match="16-byte aligned"):
        layout.bind(torch.zeros(need + 16, dtype=torch.uint8)[1:], 4)
    with pytest.raises(ValueError, match="positive int"):
        layout.bind(torch.zeros(need, dtype=torch.uint8), 0)
    with pytest.raises(TypeError, match="torch.Tensor"):
        layout.bind(bytearray(need), 4)


def test_bind_refuses_a_device_of_another_architecture(fake_cuda):
    caps, _ = fake_cuda
    caps.update({0: (9, 0)})
    with pytest.raises(ValueError, match="resolved for sm100"):
        kv_cache._require_device(_layout("sm100", 2), torch.device("cuda", 0))
    kv_cache._require_device(_layout("sm90", 2), torch.device("cuda", 0))
    with pytest.raises(ValueError, match="not a CUDA device"):
        _layout("sm90", 2).bind(torch.zeros(1 << 20, dtype=torch.uint8), 1)


def test_a_cache_is_not_built_directly():
    with pytest.raises(TypeError, match="allocate"):
        QuantKVCache()


def _host_pools(layout, pages, seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes, dtypes = layout.pool_shapes(pages), layout.pool_dtypes()
    k = torch.randint(0, 256, shapes["k_codes"], dtype=torch.uint8, generator=g)
    v = torch.randint(0, 256, shapes["v_codes"], dtype=torch.uint8, generator=g)
    s = torch.rand(shapes["scales"], generator=g).to(dtypes["scales"])
    return k, v, s


@pytest.mark.parametrize("kw", [{}, {"k_quant": ("token", 64)}, {"k_quant": ("channel", 64)}])
def test_adopt_derives_the_page_maxima_from_the_scales(any_device, kw):
    from fosca.quant_layout import quant_block_offsets

    layout = _layout("sm100", 2, num_kv_heads=2, **kw)
    k, v, s = _host_pools(layout, 3)
    cache = QuantKVCache._adopt(layout, k, v, s)
    assert cache.k_codes is k and cache.v_codes is v and cache.scales is s
    off = quant_block_offsets(layout.k_quant, layout.v_quant, N=64, D=D)
    s_v = (s[..., 2] if off["legacy"] else
           s[..., off["v_ring"]:off["v_ring"] + 128].reshape(3, 2, 64, 2)[..., 0])
    assert cache.v_page_max.dtype == torch.uint32
    assert cache.v_page_max.shape == (3, 2)
    got = cache.v_page_max.view(torch.int32).view(torch.float32)
    assert torch.equal(got, s_v.float().amax(-1))
    from fosca._sm100.fold_runtime import v_max_hist_of
    assert torch.equal(cache.v_max_hist, v_max_hist_of(cache.v_page_max))
    channel = QuantKVCache._adopt(_layout("sm100", 2, v_quant=("channel", 64)),
                                  *_host_pools(_layout("sm100", 2, v_quant=("channel", 64)), 3))
    assert channel.v_page_max is None and channel.v_max_hist is None


def test_adopt_takes_the_page_maxima_and_their_table_together(any_device):
    from fosca._sm100.fold_runtime import v_max_hist_of, v_page_max_for_pool

    layout = _layout("sm100", 4, num_kv_heads=2)
    k, v, s = _host_pools(layout, 3)
    vpm = v_page_max_for_pool(s)
    hist = v_max_hist_of(vpm)
    cache = QuantKVCache._adopt(layout, k, v, s, v_page_max=vpm, v_max_hist=hist)
    assert cache.v_page_max is vpm and cache.v_max_hist is hist
    with pytest.raises(ValueError, match="together"):
        QuantKVCache._adopt(layout, k, v, s, v_page_max=vpm)
    with pytest.raises(ValueError, match="together"):
        QuantKVCache._adopt(layout, k, v, s, v_max_hist=hist)
    with pytest.raises(ValueError, match="v_max_hist is"):
        QuantKVCache._adopt(layout, k, v, s, v_page_max=vpm, v_max_hist=hist[:128].clone())
    channel = _layout("sm100", 4, v_quant=("channel", 64))
    with pytest.raises(ValueError, match="keeps no v_page_max"):
        QuantKVCache._adopt(channel, *_host_pools(channel, 3), v_page_max=vpm, v_max_hist=hist)


def test_recount_v_max_rebuilds_the_table_in_place(any_device):
    """An engine-style copy of one page's maximum from another cache bypasses the table;
    `recount_v_max` makes the table the histogram of the maxima again, in the same tensor."""
    from fosca._sm100.fold_runtime import v_max_hist_of

    layout = _layout("sm100", 2, num_kv_heads=2)
    a = QuantKVCache._adopt(layout, *_host_pools(layout, 4, seed=1))
    b = QuantKVCache._adopt(layout, *_host_pools(layout, 4, seed=2))
    a.v_page_max.view(torch.int32)[0] = torch.tensor([0x7E000000, 0], dtype=torch.int32)
    b.v_page_max[3].copy_(a.v_page_max[0])
    table = b.v_max_hist
    assert not torch.equal(table, v_max_hist_of(b.v_page_max))
    b.recount_v_max()
    assert b.v_max_hist is table and torch.equal(table, v_max_hist_of(b.v_page_max))
    channel = _layout("sm100", 2, v_quant=("channel", 64))
    QuantKVCache._adopt(channel, *_host_pools(channel, 2)).recount_v_max()


def test_adopt_refusals(any_device):
    layout = _layout("sm100", 4)
    k, v, s = _host_pools(layout, 3)
    with pytest.raises(ValueError, match="scales is"):
        QuantKVCache._adopt(layout, k, v, s.to(torch.bfloat16))
    with pytest.raises(ValueError, match="v_codes is"):
        QuantKVCache._adopt(layout, k, v[:2].clone(), s)
    # A V pool of the other Blackwell layout holds the same bytes in the other slab shape: a
    # channel-row V is refused by a production layout, token-row V by an ablation one.
    channel_rows = v.view(*v.shape[:2], D, 64 * 4 // 8)
    with pytest.raises(ValueError, match="channel-row layout must be rebuilt"):
        QuantKVCache._adopt(layout, k, channel_rows, s)
    ablation = _layout("sm100", 2, kernel_kwargs={"dequant_int2_to_f4": 0})
    ka, va, sa = _host_pools(ablation, 3)
    QuantKVCache._adopt(ablation, ka, va, sa)
    with pytest.raises(ValueError, match="stored like K must be rebuilt"):
        QuantKVCache._adopt(ablation, ka, va.view(*va.shape[:2], 64, D * 2 // 8), sa)
    with pytest.raises(ValueError, match="k_codes is"):
        QuantKVCache._adopt(_layout("sm100", 2), k, v, s)
    with pytest.raises(ValueError, match="C-contiguous"):
        QuantKVCache._adopt(layout, k, v.transpose(0, 1).contiguous().transpose(0, 1), s)
    unaligned = torch.zeros(k.numel() + 1, dtype=torch.uint8)[1:].view(k.shape)
    with pytest.raises(ValueError, match="16-byte aligned"):
        QuantKVCache._adopt(layout, unaligned, v, s)
    with pytest.raises(TypeError, match="QuantKVLayout"):
        QuantKVCache._adopt(dataclasses.asdict(layout), k, v, s)


def test_append_checks_its_inputs_before_any_compile(any_device, monkeypatch):
    layout = _layout("sm100", 2, num_kv_heads=2)
    cache = QuantKVCache._adopt(layout, *_host_pools(layout, 2))
    monkeypatch.setattr(kv_cache, "_writer_for",
                        lambda layout: pytest.fail("compiled before the inputs were checked"))
    T = 5
    k = torch.zeros(T, 2, D, dtype=torch.bfloat16)
    slots = torch.arange(T, dtype=torch.int64)
    with pytest.raises(ValueError, match="bfloat16"):
        cache.append(k.float(), k, slots)
    with pytest.raises(ValueError, match="bfloat16"):
        cache.append(torch.zeros(T, 3, D, dtype=torch.bfloat16), k, slots)
    with pytest.raises(ValueError, match="contiguous"):
        cache.append(torch.zeros(T, D, 2, dtype=torch.bfloat16).transpose(1, 2), k, slots)
    with pytest.raises(ValueError, match="16-byte aligned"):
        cache.append(torch.zeros(T * 2 * D + 1, dtype=torch.bfloat16)[1:].view(T, 2, D),
                     k, slots)
    with pytest.raises(ValueError, match="tokens"):
        cache.append(k, k[:4], slots)
    with pytest.raises(ValueError, match="int64"):
        cache.append(k, k, slots.int())
    # A fused-QKV view: a free, aligned token stride is accepted.
    fused = torch.zeros(T, 6, D, dtype=torch.bfloat16)
    monkeypatch.setattr(kv_cache, "_writer_for", lambda layout: ("compiled", _record))
    monkeypatch.setattr(_utils, "on_device", lambda device: contextlib.nullcontext())
    _calls.clear()
    cache.append(fused[:, 0:2], fused[:, 2:4], slots)
    assert _calls == [("compiled", 2, cache.v_page_max, cache.v_max_hist)]
    channel = _layout("sm100", 2, v_quant=("channel", 64))
    with pytest.raises(NotImplementedError, match="append_groups"):
        QuantKVCache._adopt(channel, *_host_pools(channel, 2)).append(k, k, slots)


_calls = []


def _record(compiled, *, inputs, slot_mapping, k_codes, v_codes, scales, v_page_max,
            v_max_hist):
    _calls.append((compiled, len(inputs), v_page_max, v_max_hist))


@pytest.mark.parametrize("arch, k_quant", [("sm100", ("token", 128)), ("sm100", ("token", 64)),
                                           ("sm90", ("token", 128))])
def test_append_quantized_checks_its_inputs_before_any_compile(any_device, monkeypatch, arch,
                                                               k_quant):
    layout = _layout(arch, 2, num_kv_heads=2, k_quant=k_quant)
    cache = QuantKVCache._adopt(layout, *_host_pools(layout, 2))
    monkeypatch.setattr(kv_cache, "_writer_for",
                        lambda layout, prequantized=False: pytest.fail(
                            "compiled before the inputs were checked"))
    T = 5
    k_group = (T, 2, 2) if k_quant == ("token", D // 2) else (T, 2)
    good = [torch.zeros(T, 2, D, dtype=torch.uint8), torch.ones(k_group),
            torch.zeros(k_group), torch.zeros(T, 2, D, dtype=torch.uint8), torch.ones(T, 2),
            torch.zeros(T, 2)]
    slots = torch.arange(T, dtype=torch.int64)
    other_k = torch.ones((T, 2) if len(k_group) == 3 else (T, 2, 2))
    unaligned = torch.zeros(T * 2 * D + 1, dtype=torch.uint8)[1:].view(T, 2, D)
    for index, bad, match in [
            (0, good[0].view(torch.int8), "k_codes must be a torch.uint8"),
            (3, torch.zeros(T, 2, D // 2, dtype=torch.uint8), "v_codes must be"),
            (1, good[1].double(), "k_scale must be a torch.float32"),
            (2, other_k, "k_zero must be"),
            (4, torch.ones(T - 1, 2), "v_scale must be"),
            (5, torch.zeros(T, 2, 1), "v_zero must be"),
            (0, good[0].transpose(1, 2).contiguous().transpose(1, 2), "C-contiguous"),
            (4, torch.ones(2, T).t(), "C-contiguous"),
            (3, unaligned, "16-byte aligned")]:
        args = list(good)
        args[index] = bad
        with pytest.raises(ValueError, match=match):
            cache.append_quantized(*args, slots)
    with pytest.raises(ValueError, match="int64"):
        cache.append_quantized(*good, slots.int())
    with pytest.raises(ValueError, match="slot_mapping must be"):
        cache.append_quantized(*good, slots[:4])
    monkeypatch.setattr(kv_cache, "_writer_for", lambda layout, prequantized=False: (
        "given" if prequantized else "compiled", _record))
    monkeypatch.setattr(_utils, "on_device", lambda device: contextlib.nullcontext())
    _calls.clear()
    cache.append_quantized(*good, slots)
    cache.append_quantized(*good, slots)
    keeps = (cache.v_page_max, cache.v_max_hist)
    assert _calls == [("given", 6, *keeps)] * 2 and cache._writer is None
    channel = _layout(arch, 2, v_quant=("channel", 64))
    with pytest.raises(NotImplementedError, match="append_quantized.*append_groups"):
        QuantKVCache._adopt(channel, *_host_pools(channel, 2)).append_quantized(*good, slots)


@pytest.mark.parametrize("prequantized", [False, True])
def test_both_writers_refuse_more_than_max_append_tokens_before_any_compile(
        any_device, monkeypatch, prequantized):
    """With the limit lowered to 4 tokens, a call of 5 is refused with nothing compiled or
    launched, and a call of exactly 4 compiles and launches once, through `append` and through
    `append_quantized` alike."""
    layout = _layout("sm100", 2, num_kv_heads=2)
    cache = QuantKVCache._adopt(layout, *_host_pools(layout, 2))
    compiled = []
    monkeypatch.setattr(kv_cache, "MAX_APPEND_TOKENS", 4)
    monkeypatch.setattr(kv_cache, "_writer_for", lambda layout, prequantized=False: (
        compiled.append(prequantized) or ("compiled", _record)))
    monkeypatch.setattr(_utils, "on_device", lambda device: contextlib.nullcontext())

    def write(T):
        slots = torch.arange(T, dtype=torch.int64)
        if prequantized:
            codes = torch.zeros(T, 2, D, dtype=torch.uint8)
            cache.append_quantized(codes, torch.ones(T, 2), torch.zeros(T, 2), codes,
                                   torch.ones(T, 2), torch.zeros(T, 2), slots)
        else:
            k = torch.zeros(T, 2, D, dtype=torch.bfloat16)
            cache.append(k, k, slots)

    _calls.clear()
    with pytest.raises(ValueError, match="at most 4 tokens"):
        write(5)
    assert compiled == [] and _calls == []
    write(4)
    assert compiled == [prequantized] and len(_calls) == 1


def _oscar_forms(mn, mx, q):
    """OSCAR's quantization parameters of a row with minimum `mn` and maximum `mx`, the mapping
    `append_quantized`'s docstring gives (s = scale, z = fl(-zero * scale)), and the two
    dequantizations of code q, each in f32 one operation at a time: (s, zero, z, s * q + z,
    (q - zero) * scale, x), where x is (q - zero) * scale exactly, in f64, from the stored
    f32 zero and scale."""
    f32 = torch.float32
    mn, mx = torch.tensor([mn], dtype=f32), torch.tensor([mx], dtype=f32)
    scale = torch.clamp(mx - mn, min=1e-8) / 3
    zero = -mn / scale
    z = -zero * scale
    code = torch.tensor([float(q)], dtype=f32)
    ours = code * scale
    ours = ours + z
    oscar = code - zero
    oscar = oscar * scale
    x = (q - zero.double()) * scale.double()
    return [float(t) for t in (scale, zero, z, ours, oscar, x)]


def test_the_oscar_mapping_rounds_as_append_quantized_documents():
    """The rounding statement of `append_quantized`'s docstring (and docs/integration.md):
    |z - (-zero * scale)| <= 2^-24 |z|; the two f32 forms differ by at most 6 * 2^-24 *
    max(|x|, |z|) to first order, over random rows of every scale and offset; and the stated
    cancellation case, where that bound exceeds the step and the forms differ by 3 s."""
    u = 2.0 ** -24
    g = torch.Generator().manual_seed(3)
    offsets = torch.randn(4000, generator=g, dtype=torch.float64) * 10.0 ** torch.randint(
        -6, 5, (4000,), generator=g)
    ranges = 10.0 ** (torch.rand(4000, generator=g, dtype=torch.float64) * 12 - 9)
    for mn, rng in zip(offsets.tolist(), ranges.tolist()):
        for q in range(4):
            s, zero, z, ours, oscar, x = _oscar_forms(mn, mn + rng, q)
            assert abs(z + zero * s) <= u * abs(z)
            assert abs(ours - oscar) <= 6 * u * max(abs(x), abs(z)) * (1 + 2.0 ** -20)
    s, zero, z, ours, oscar, x = _oscar_forms(1000.0, 1000.00006103515625, 3)
    assert f"{s:.3g}" == "2.03e-05" and f"{abs(z) / s:.2g}" == "4.9e+07"
    assert abs(z) / s > 2.0 ** 24 / 6
    assert 2.99 < abs(ours - oscar) / s < 3.01
    assert abs(ours - oscar) <= 6 * u * max(abs(x), abs(z))

    def ulp(v):
        t = torch.tensor([abs(v)], dtype=torch.float32)
        return float(torch.nextafter(t, torch.tensor([float("inf")])) - t)

    assert f"{ulp(z):.2g}" == "6.1e-05" and f"{ulp(zero) * s:.2g}" == "8.1e-05"
    assert ulp(z) > s and ulp(zero) * s > s


@pytest.mark.parametrize("bits", [2, 4])
def test_the_prequantized_saturation_range_is_the_quantizing_writers(bits):
    """`scale_range(qmax)` is the f32 scale `append` stores for a constant row, div.rn(1e-8,
    qmax), and for a row spanning the whole input envelope, div.rn(2^65, qmax), each as torch's
    correctly rounded f32 division gives it: the bounds the prequantized write saturates to."""
    from fosca._append_core import INPUT_ENVELOPE, scale_range
    qmax = (1 << bits) - 1
    lo, hi = scale_range(qmax)
    f32 = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    assert lo == (f32(1e-8) / qmax).item() and hi == (f32(2 * INPUT_ENVELOPE) / qmax).item()
    assert lo == f32(lo).item() and hi == f32(hi).item() and 0 < lo < hi


# -- the wrapper ----------------------------------------------------------------------------
# Planned on the host: the device's architecture, the DSL target, the SM count, the compile
# and the launch are stood in for, so these check what plan() and run() decide and refuse,
# and what they hand the launch, without a GPU.

from fosca import wrapper as W  # noqa: E402


class _Launched(Exception):
    pass


@pytest.fixture
def host_wrapper(monkeypatch):
    """`make(arch, layout=None, **plan_kwargs)` -> (wrapper, layout, cache), planned on the CPU
    for a 2-request batch of 512 tokens; `launches` records each launch's keyword arguments."""
    from fosca._sm100 import fold_runtime
    from fosca._sm90 import pingpong_runtime

    launches = []
    targets = []
    monkeypatch.setattr(_utils, "require_arch", lambda device=None: arch_box[0])
    monkeypatch.setattr(_utils, "require_dsl_target", lambda arch: targets.append(arch))
    monkeypatch.setattr(_utils, "on_device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(kv_cache, "_require_device", lambda layout, device: None)
    monkeypatch.setattr(W, "get_num_sm", lambda device: 132)
    monkeypatch.setattr(W, "_get_compiled_fold_kernel",
                        lambda *a, has_q_descale=False, has_lse=False: (
                            "fold" + ("_qd" if has_q_descale else "")
                            + ("_lse" if has_lse else ""),
                            dict(fold_runtime.BITS_TUNING[a[5]])))
    monkeypatch.setattr(W, "_get_compiled_pingpong_kernel",
                        lambda *a: "pingpong_lse" if a[-1] is True else "pingpong")
    monkeypatch.setattr(W, "_get_compiled_splitk_combine",
                        lambda *a: "combine_lse" if a[-1] is True else "combine")
    monkeypatch.setattr(fold_runtime, "launch_fold",
                        lambda compiled, **kw: launches.append(("fold", compiled, kw)))
    monkeypatch.setattr(pingpong_runtime, "launch_pingpong",
                        lambda compiled, combine, **kw: launches.append(
                            ("pingpong", compiled, {**kw, "combine": combine})))
    arch_box = ["sm100"]

    def make(arch, layout=None, num_qo_heads=8, **plan_kwargs):
        arch_box[0] = arch
        layout = layout or _layout(arch, 2, num_kv_heads=2)
        cache = QuantKVCache._adopt(layout, *_host_pools(layout, 16))
        w = W.PagedDecodeWrapper(torch.zeros(1 << 20, dtype=torch.float32))
        indptr = torch.tensor([0, 8, 16], dtype=torch.int32)
        indices = torch.arange(16, dtype=torch.int32)
        seq_lens = torch.tensor([512, 500], dtype=plan_kwargs.pop("seq_dtype", torch.int32))
        w.plan(indptr, indices, seq_lens, num_qo_heads, layout, max_kv_len=512, **plan_kwargs)
        return w, layout, cache

    make.launches, make.targets = launches, targets
    return make


def _q(w, *, three_d=True, dtype=None):
    shape = ((w._batch_size, w._num_qo_heads, w._head_dim) if three_d
             else (w._batch_size, 1, w._num_qo_heads, w._head_dim))
    return torch.zeros(shape, dtype=dtype or w._q_data_type)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("splits", [1, 2])
def test_blackwell_plan_forwards_d256_groups_and_preserves_d128_compile_keys(host_wrapper, bits, dim, splits):
    layout = _layout("sm100", bits, head_dim=dim, num_kv_heads=2)
    w, _, _ = host_wrapper("sm100", layout=layout, kv_splits=splits)
    kwargs = dict(w._fold_key[-1])
    if dim == 256:
        assert kwargs["k_quant"] == kwargs["v_quant"] == ("token", 256)
    else:
        assert "k_quant" not in kwargs and "v_quant" not in kwargs
    assert (kwargs.get("persistent") is False) == (splits == 1)


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_run_launches_the_planned_kernel_over_the_cache(host_wrapper, arch):
    w, layout, cache = host_wrapper(arch)
    assert host_wrapper.targets == [arch], "the DSL target is checked before the compile"
    out = torch.zeros(2, 8, D, dtype=torch.bfloat16)
    assert w.run(_q(w), cache, out=out) is out
    kind, compiled, kw = host_wrapper.launches[-1]
    if arch == "sm90":
        assert (kind, compiled) == ("pingpong", "pingpong")
        assert kw["mKq"] is cache.k_codes and kw["mScale"] is cache.scales
        assert kw["out_bhd"].data_ptr() == out.data_ptr()
    else:
        assert (kind, compiled) == ("fold", "fold")
        assert kw["mKq"] is cache.k_codes and kw["mVq"] is cache.v_codes
        assert kw["o_bshd"].data_ptr() == out.data_ptr() and kw["q_bshd"].dim() == 4
    # Without `out`, the output takes the query's form.
    assert w.run(_q(w, three_d=False), cache).shape == (2, 1, 8, D)
    assert w.run(_q(w), cache).shape == (2, 8, D)


def test_plan_takes_int64_lengths_and_refuses_the_rest(host_wrapper):
    w, _, _ = host_wrapper("sm100", seq_dtype=torch.int64)
    assert w._seqlens.dtype == torch.int32
    with pytest.raises(ValueError, match="int32 or int64"):
        host_wrapper("sm100", seq_dtype=torch.float32)


@pytest.mark.parametrize("call,match", [
    # The per-argument plan(), positionally and by keyword.
    (lambda w, l, a: w.plan(*a, 32, 8, 128, 64), "extra positional"),
    (lambda w, l, a: w.plan(*a, 32, 8, 128, 64, bits=2), "retired call form"),
    (lambda w, l, a: w.plan(*a, 32, l, bits=2), "'bits'"),
    (lambda w, l, a: w.plan(*a, 32, l, k_quant=("token", 128)), "'k_quant'"),
    (lambda w, l, a: w.plan(*a, 32, l, head_dim=128, page_size=64), "'head_dim'"),
    (lambda w, l, a: w.plan(*a, 32, num_kv_heads=8), "'num_kv_heads'"),
    (lambda w, l, a: w.plan(*a, 32, 8), "layout must be a QuantKVLayout"),
    # The public kernel_kwargs, whatever it holds.
    (lambda w, l, a: w.plan(*a, 32, l, kernel_kwargs={}), "kernel_kwargs is retired"),
    (lambda w, l, a: w.plan(*a, 32, l, kernel_kwargs={"sdtype": "f32"}),
     "kernel_kwargs is retired"),
    (lambda w, l, a: w.plan(*a, 32, l, not_an_argument=1), "unexpected keyword"),
])
def test_the_retired_plan_arguments_raise(call, match):
    w = W.PagedDecodeWrapper(torch.zeros(4))
    args = (torch.zeros(3, dtype=torch.int32),) * 3
    with pytest.raises(TypeError, match=match):
        call(w, _layout("sm100", 2), args)
    assert not w._planned


def test_the_page_size_has_one_value():
    assert W._QUANT_PAGE_SIZE == kv_cache.PAGE_SIZE == 64


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_the_retired_run_arguments_raise(host_wrapper, arch):
    w, _, cache = host_wrapper(arch)
    q = _q(w)
    k, v, s = cache.k_codes, cache.v_codes, cache.scales
    for call, match in (
            (lambda: w.run(q, k, v, scale_pool=s), "'scale_pool'"),
            (lambda: w.run(q, k, v, out=q), "extra positional"),
            (lambda: w.run(q, k), "kv_cache must be a QuantKVCache"),
            (lambda: w.run(q, cache, pscale_lp=1.0), "'pscale_lp'"),
            (lambda: w.run(q, cache, k_pool_order="permuted"), "'k_pool_order'"),
            (lambda: w.run(q, cache, kernel_kwargs={}), "kernel_kwargs is retired")):
        with pytest.raises(TypeError, match=match):
            call()
    with pytest.raises(TypeError, match="retired"):
        w.pscale_lp(s)
    assert not host_wrapper.launches


def test_a_cache_of_another_layout_is_refused_even_with_identical_pools(host_wrapper):
    planned = _layout("sm100", 2, num_kv_heads=2, k_quant=("channel", 64))
    other = _layout("sm100", 2, num_kv_heads=2, k_quant=("channel", 128))
    assert planned.pool_shapes(16) == other.pool_shapes(16)
    w, _, _ = host_wrapper("sm100", layout=planned)
    cache = QuantKVCache._adopt(other, *_host_pools(other, 16))
    with pytest.raises(ValueError, match="another layout"):
        w.run(_q(w), cache)
    # An equal layout built separately is accepted: the comparison is by value.
    same = _layout("sm100", 2, num_kv_heads=2, k_quant=("channel", 64))
    w.run(_q(w), QuantKVCache._adopt(same, *_host_pools(same, 16)))
    # And so is an ablation layout only for a plan of that same ablation.
    ablation = _layout("sm100", 2, num_kv_heads=2, kernel_kwargs={"dequant_int2_to_f4": 0})
    w2, _, _ = host_wrapper("sm100")
    with pytest.raises(ValueError, match="another layout"):
        w2.run(_q(w2), QuantKVCache._adopt(ablation, *_host_pools(ablation, 16)))


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_query_and_output_are_checked_exactly(host_wrapper, arch):
    w, _, cache = host_wrapper(arch)
    b, hq, d = 2, 8, D
    qd, od = w._q_data_type, w._o_data_type
    bad_q = [
        torch.zeros(1, 16, 128, dtype=qd).expand(1, 16, 128),   # same elements, wrong split
        torch.zeros(b, 2 * hq, d // 2, dtype=qd),
        torch.zeros(b, 2, hq, d, dtype=qd),                    # two query tokens
        torch.zeros(b, hq, d, dtype=torch.float16),
        torch.zeros(b, d, hq, dtype=qd).transpose(1, 2),         # right shape, not contiguous
        torch.zeros(b * hq * d + 1, dtype=qd)[1:].view(b, hq, d),  # misaligned
        torch.zeros(b, hq, d, dtype=qd, device="meta"),          # another device
    ]
    for q in bad_q:
        with pytest.raises(ValueError):
            w.run(q, cache)
    for out in (torch.zeros(b, hq * d, dtype=od), torch.zeros(b, hq, d, dtype=torch.float32),
                torch.zeros(b, d, hq, dtype=od).transpose(1, 2)):
        with pytest.raises(ValueError, match="out"):
            w.run(_q(w), cache, out=out)
    assert not host_wrapper.launches


def test_lse_is_checked_and_reaches_the_variant_each_plan_needs(host_wrapper):
    """`run(lse=)`: an fp32 [B, HQ] C-contiguous, 16-byte aligned tensor on the device, or a
    refusal before any launch; then SM100's `has_lse` variant with the LSE viewed [B, 1, HQ],
    SM90's one-split plan's own kernel with the LSE as its split LSE, and a split plan's LSE
    merge (the fused decode or the external combine) with the LSE as `lse_out`."""
    for arch in ("sm90", "sm100"):
        w, _, cache = host_wrapper(arch)
        for bad in (torch.zeros(2, 8, dtype=torch.float16), torch.zeros(2, 4),
                    torch.zeros(8, 2).t(), torch.zeros(17)[1:].view(2, 8),
                    torch.zeros(2, 8, device="meta")):
            with pytest.raises(ValueError, match="lse must be"):
                w.run(_q(w), cache, lse=bad)
        assert not host_wrapper.launches
    lse = torch.zeros(2, 8)
    w, _, cache = host_wrapper("sm100")
    w.run(_q(w), cache, lse=lse)
    (kind, compiled, kw), = host_wrapper.launches
    assert (kind, compiled) == ("fold", "fold_lse")
    assert kw["lse"].shape == (2, 1, 8) and kw["lse"].data_ptr() == lse.data_ptr()
    w.run(_q(w), cache, lse=lse, q_descale=torch.ones(2, 8))
    assert host_wrapper.launches[-1][1] == "fold_qd_lse"
    w.run(_q(w), cache)
    assert host_wrapper.launches[-1][1] == "fold" and host_wrapper.launches[-1][2]["lse"] is None
    host_wrapper.launches.clear()
    w, _, cache = host_wrapper("sm90")                     # 512 tokens: one split
    assert w.kv_splits == 1
    w.run(_q(w), cache, lse=lse)
    (kind, compiled, kw), = host_wrapper.launches
    assert (compiled, kw["combine"], kw["lse_out"]) == ("pingpong", None, None)
    assert kw["lse"].shape == (1, 2, 8) and kw["lse"].data_ptr() == lse.data_ptr()
    for fused, decode, combine in ((True, "pingpong_lse", None),
                                   (False, "pingpong", "combine_lse")):
        layout = _layout("sm90", 2, num_kv_heads=2, kernel_kwargs={"fused_combine": fused})
        w, _, cache = host_wrapper("sm90", layout=layout, kv_splits=4)
        w.run(_q(w), cache, lse=lse)
        _, compiled, kw = host_wrapper.launches[-1]
        assert (compiled, kw["combine"], kw["lse_out"]) == (decode, combine, lse)
        assert kw["lse"] is w._sm90_ws["lse"]
        w.run(_q(w), cache)
        _, compiled, kw = host_wrapper.launches[-1]
        assert (compiled, kw["lse_out"]) == ("pingpong", None)
        assert kw["combine"] == (None if fused else "combine")


def test_lse_and_q_descale(host_wrapper):
    w, _, cache = host_wrapper("sm90")
    with pytest.raises(ValueError, match="SM100's fp8 query"):
        w.run(_q(w), cache, q_descale=torch.ones(2, 8))
    w, _, cache = host_wrapper("sm100")
    with pytest.raises(ValueError, match="q_descale must be an fp32"):
        w.run(_q(w), cache, q_descale=torch.ones(2, 4))
    with pytest.raises(ValueError, match="q_descale must be an fp32"):
        w.run(_q(w), cache, q_descale=torch.ones(2, 8, dtype=torch.float16))
    assert not host_wrapper.launches
    # A stride-0 per-tensor expand is a valid descale: it reaches the launch, through the
    # plan's kernel compiled with the per-row factor, and the cache's count table of page
    # maxima rides along.
    qd = torch.ones(1, 1).expand(2, 8)
    w.run(_q(w), cache, q_descale=qd)
    (kind, compiled, kw), = host_wrapper.launches
    assert (kind, compiled) == ("fold", "fold_qd")
    assert kw["q_descale"] is qd and kw["v_max_hist"] is cache.v_max_hist
    w.run(_q(w), cache)
    assert host_wrapper.launches[-1][1] == "fold" and host_wrapper.launches[-1][2]["q_descale"] is None


def test_only_a_build_that_reads_the_table_is_handed_it(host_wrapper):
    """The fold with token-wise V reads the count table; the fold-off ablation and
    channel-wise V take none (`fold_runtime.reads_v_max_hist`)."""
    for layout, reads in (
            (_layout("sm100", 4, num_kv_heads=2), True),
            (_layout("sm100", 2, num_kv_heads=2,
                     kernel_kwargs={"fold_scale_zero": 0, "dequant_int2_to_f4": 0}), False),
            (_layout("sm100", 4, num_kv_heads=2, v_quant=("channel", 64)), False)):
        w, _, cache = host_wrapper("sm100", layout=layout)
        w.run(_q(w), cache)
        got = host_wrapper.launches[-1][2]["v_max_hist"]
        assert (got is cache.v_max_hist and got is not None) if reads else got is None


@pytest.mark.parametrize("arch,kw,match", [
    ("sm90", {"o_data_type": torch.float16}, "writes a"),
    ("sm90", {"q_data_type": torch.float8_e4m3fn}, "reads a"),
    ("sm100", {"q_data_type": torch.bfloat16}, "reads a"),
    ("sm100", {"o_data_type": torch.float64}, "writes a"),
    ("sm100", {"q_len_per_req": 2}, "plain decode"),
])
def test_plan_refuses_dtypes_and_query_lengths_the_reader_does_not_build(host_wrapper, arch,
                                                                         kw, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        host_wrapper(arch, **kw)


@pytest.mark.parametrize("num_qo_heads", [2, 4, 6, 8, 16, 18, 32, 64, 128, 256])
def test_blackwell_eager_pads_and_tiles_gqa_heads(host_wrapper, monkeypatch, num_qo_heads):
    compiled = []
    monkeypatch.setattr(W, "_get_compiled_fold_kernel",
                        lambda *a: compiled.append(a) or ("fold", {"code_stages": 16,
                                                                   "scale_stages": 16}))
    w, _, _ = host_wrapper("sm100", num_qo_heads=num_qo_heads)
    assert w._planned and compiled


@pytest.mark.parametrize("num_qo_heads", [2, 4, 16, 32, 64, 128, 256])
def test_hopper_eager_pads_and_tiles_gqa_heads(host_wrapper, num_qo_heads):
    w, _, _ = host_wrapper("sm90", num_qo_heads=num_qo_heads)
    assert w._planned


def test_hopper_refuses_a_partial_high_ratio_tile(host_wrapper):
    with pytest.raises(NotImplementedError, match="multiples of eight"):
        host_wrapper("sm90", num_qo_heads=18)


def test_plan_refuses_an_empty_batch_and_a_bad_head_count(host_wrapper):
    w, layout, _ = host_wrapper("sm100")
    empty = torch.zeros(0, dtype=torch.int32)
    with pytest.raises(ValueError, match="at least one request"):
        w.plan(empty, empty, empty, 8, layout)
    one = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="multiple of the layout's num_kv_heads"):
        w.plan(torch.zeros(2, dtype=torch.int32), one, one, 3, layout)
    assert not w._planned


def test_plan_refuses_a_layout_of_another_architecture(host_wrapper):
    w, _, _ = host_wrapper("sm100")
    with pytest.raises(ValueError, match="resolved for sm90"):
        host_wrapper("sm100", layout=_layout("sm90", 2, num_kv_heads=2))


def test_a_failed_replan_leaves_nothing_runnable(host_wrapper):
    w, layout, cache = host_wrapper("sm100")
    one = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError):
        w.plan(torch.zeros(2, dtype=torch.int32), one, one, 3, layout)
    assert not w._planned and w._compiled_fold is None and w._layout is None
    with pytest.raises(RuntimeError, match=r"Call plan\(\) before run"):
        w.run(_q(host_wrapper("sm100")[0]), cache)


def test_the_wrapper_resolves_its_own_device_and_compile_target(fake_cuda, monkeypatch,
                                                                 stub_dsl):
    """Two wrappers on a Hopper and a Blackwell device in one process: each resolves the
    architecture of ITS device whatever device is current, and each refuses to compile
    when the DSL's single compile target (`envar.arch`, which the kernel cache keys every
    object by) is the other architecture's."""
    caps, current = fake_cuda
    caps.update({0: (9, 0), 1: (10, 0)})

    class _Planned(Exception):
        pass

    real = _utils.require_dsl_target

    def checked(arch):
        # The real check, then stop: what follows is device work this host cannot do.
        real(arch)
        raise _Planned(arch)

    monkeypatch.setattr(_utils, "require_dsl_target", checked)
    lens = torch.tensor([64], dtype=torch.int32)
    iptr = torch.tensor([0, 1], dtype=torch.int32)
    hopper = W.PagedDecodeWrapper(types_ns(torch.device("cuda", 0)))
    blackwell = W.PagedDecodeWrapper(types_ns(torch.device("cuda", 1)))
    for cur in (0, 1, 0):
        current[0] = cur
        # The Blackwell wrapper passes the target check whatever device is current ...
        with pytest.raises(_Planned):
            blackwell.plan(iptr, lens, lens, 8, _layout("sm100", 2), max_kv_len=64)
        # ... and the Hopper one is refused by the compile target, not mis-dispatched.
        with pytest.raises(RuntimeError, match="compiles for sm_100a"):
            hopper.plan(iptr, lens, lens, 8, _layout("sm90", 2), max_kv_len=64)
        with pytest.raises(ValueError, match="resolved for sm90"):
            blackwell.plan(iptr, lens, lens, 8, _layout("sm90", 2), max_kv_len=64)
    stub_dsl.retarget("sm_90a")
    with pytest.raises(_Planned):
        hopper.plan(iptr, lens, lens, 8, _layout("sm90", 2), max_kv_len=64)
    from fosca import kernel_cache
    with pytest.raises(RuntimeError, match="cannot build the sm100 kernel"):
        blackwell.plan(iptr, lens, lens, 8, _layout("sm100", 2), max_kv_len=64)
    # The kernel cache keys every object by that same target, so a refusal here is what keeps
    # an object compiled for one architecture from being filed for the other.
    assert "envar.arch" in kernel_cache.dsl_identity.__doc__


def types_ns(device):
    """A stand-in workspace that carries only a device (these plans stop before using it)."""
    import types
    return types.SimpleNamespace(device=device)
