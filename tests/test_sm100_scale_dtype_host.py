"""CPU-only regression tests for the SM100 quant-flex scale-pool dtype contract."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytest.importorskip("cuda", reason="needs the CUDA Python bindings")


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src/cutedsl_decode"
SM100_DIR = PACKAGE_DIR / "_sm100"
HOST_PACKAGE = "_cutedsl_decode_sm100_host_test"
HOST_PARENT = "_cutedsl_decode_host_test_parent"
#: Every synthetic sys.modules name this fixture installs, so it can undo them.
_INSTALLED: list[str] = []
WRAPPER_HOST_PACKAGE = "_cutedsl_decode_wrapper_host_test"


def _load_fold_runtime():
    """Load the torch-only runtime helpers without importing CUDA/CuTe modules.

    `fold_runtime` reaches the shared scale-block layout as `..quant_layout`, so the
    synthetic package needs a PARENT for that relative import to resolve; without one
    it fails with "attempted relative import beyond top-level package". The parent is
    built here and the layout module preloaded into it, which keeps the fixture
    torch-only: `cutedsl_decode/quant_layout.py` imports nothing at all.

    Every synthetic name is recorded in `_INSTALLED` so `_unload_fold_runtime` can take
    them back out; without that, whichever test ran first would fix the module identity
    for the rest of the session. This covers the names THIS fixture installs and no
    others: `_load_wrapper_without_cutlass` builds its own `fold_runtime` under a
    separate synthetic package, and the two identities coexist by design.
    """
    module_name = f"{HOST_PARENT}.{HOST_PACKAGE}.fold_runtime"
    if module_name in sys.modules:
        return sys.modules[module_name]

    def _install(name, mod):
        _INSTALLED.append(name)
        sys.modules[name] = mod

    parent = types.ModuleType(HOST_PARENT)
    parent.__path__ = [str(PACKAGE_DIR)]
    _install(HOST_PARENT, parent)

    ql_name = f"{HOST_PARENT}.quant_layout"
    ql_spec = importlib.util.spec_from_file_location(
        ql_name, PACKAGE_DIR / "quant_layout.py")
    ql = importlib.util.module_from_spec(ql_spec)
    _install(ql_name, ql)
    assert ql_spec.loader is not None
    ql_spec.loader.exec_module(ql)

    package = types.ModuleType(f"{HOST_PARENT}.{HOST_PACKAGE}")
    package.__path__ = [str(SM100_DIR)]
    _install(f"{HOST_PARENT}.{HOST_PACKAGE}", package)

    spec = importlib.util.spec_from_file_location(
        module_name, SM100_DIR / "fold_runtime.py")
    module = importlib.util.module_from_spec(spec)
    _install(module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _unload_fold_runtime():
    """Remove every synthetic module this fixture installed."""
    for name in reversed(_INSTALLED):
        sys.modules.pop(name, None)
    _INSTALLED.clear()


@pytest.fixture
def fold_runtime():
    """The torch-only `fold_runtime`, with its synthetic package torn down after."""
    try:
        yield _load_fold_runtime()
    finally:
        _unload_fold_runtime()


def _load_wrapper_without_cutlass():
    """Source-load the public wrapper against inert Cutlass/CuTe stubs."""
    module_name = f"{WRAPPER_HOST_PACKAGE}.wrapper"
    if module_name in sys.modules:
        return sys.modules[module_name]

    package = types.ModuleType(WRAPPER_HOST_PACKAGE)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[WRAPPER_HOST_PACKAGE] = package

    utils_name = f"{WRAPPER_HOST_PACKAGE}._utils"
    utils = types.ModuleType(utils_name)
    utils.api = lambda function: function
    utils.get_num_sm = lambda device: 120
    utils.is_sm90 = lambda: False
    utils.supported_page_sizes = lambda device: (8, 16, 32, 64)
    sys.modules[utils_name] = utils

    sm100_name = f"{WRAPPER_HOST_PACKAGE}._sm100"
    sm100_package = types.ModuleType(sm100_name)
    sm100_package.__path__ = [str(SM100_DIR)]
    sys.modules[sm100_name] = sm100_package

    gqa_name = f"{sm100_name}.gqa_decode_paged"
    gqa_module = types.ModuleType(gqa_name)
    gqa_module.GroupedQueryAttentionDecodePaged = type(
        "GroupedQueryAttentionDecodePaged", (), {})
    sys.modules[gqa_name] = gqa_module

    fake_cutlass = types.ModuleType("cutlass")
    fake_cutlass.__path__ = []
    fake_cutlass.Float16 = types.SimpleNamespace(width=16)
    fake_cutlass.BFloat16 = types.SimpleNamespace(width=16)
    fake_cutlass.Float8E5M2 = types.SimpleNamespace(width=8)
    fake_cutlass.Float8E4M3FN = types.SimpleNamespace(width=8)
    fake_cutlass.Float32 = types.SimpleNamespace(width=32)
    fake_cutlass.dtype = object
    fake_cute = types.ModuleType("cutlass.cute")
    fake_cute.__path__ = []
    fake_typing = types.ModuleType("cutlass.cute.typing")
    fake_typing.Float32 = lambda value: value
    fake_typing.Int32 = lambda value: value
    fake_cutlass.cute = fake_cute

    stub_names = {
        "cutlass": fake_cutlass,
        "cutlass.cute": fake_cute,
        "cutlass.cute.typing": fake_typing,
    }
    missing = object()
    previous = {
        name: sys.modules.get(name, missing)
        for name in stub_names
    }
    sys.modules.update(stub_names)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, PACKAGE_DIR / "wrapper.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module


def test_k_token64_defaults_to_bf16_and_accepts_explicit_f32(fold_runtime):
    runtime = fold_runtime
    kwargs = {
        "k_quant": ("token", 64),
        "v_quant": ("token", 128),
    }
    kq, vq, sdtype, offsets = runtime.quant_cfg_of(64, 128, kwargs)
    assert (kq, vq, sdtype) == (
        ("token", 64), ("token", 128), "bf16")
    assert offsets["ring_tot"] == 384
    assert runtime.scale_pool_dtype_of(64, 128, kwargs) is torch.bfloat16

    kwargs["sdtype"] = "f32"
    kq, vq, sdtype, offsets = runtime.quant_cfg_of(64, 128, kwargs)
    assert (kq, vq, sdtype) == (
        ("token", 64), ("token", 128), "f32")
    assert offsets["ring_tot"] == 384
    assert runtime.scale_pool_dtype_of(64, 128, kwargs) is torch.float32


@pytest.mark.parametrize("sdtype", ["f16", "float32", torch.float32])
def test_quant_cfg_rejects_unknown_scale_dtype(sdtype, fold_runtime):
    runtime = fold_runtime
    with pytest.raises(ValueError, match="sdtype must be"):
        runtime.quant_cfg_of(
            64,
            128,
            {
                "k_quant": ("token", 64),
                "v_quant": ("token", 128),
                "sdtype": sdtype,
            },
        )


def test_bf16_remains_illegal_outside_k_token64(fold_runtime):
    runtime = fold_runtime
    with pytest.raises(ValueError, match="only legal with K token-64"):
        runtime.quant_cfg_of(
            64,
            128,
            {
                "k_quant": ("channel", 128),
                "v_quant": ("token", 128),
                "sdtype": "bf16",
            },
        )


@pytest.mark.parametrize(
    ("expected", "wrong"),
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_scale_pool_dtype_check_fails_closed(expected, wrong, fold_runtime):
    runtime = fold_runtime
    runtime.require_scale_pool_dtype(
        torch.empty(1, dtype=expected), expected, expected_sdtype=str(expected))
    with pytest.raises(ValueError, match="scale_pool.dtype"):
        runtime.require_scale_pool_dtype(
            torch.empty(1, dtype=wrong), expected, expected_sdtype=str(expected))


def _uninitialized_wrapper(expected, sdtype):
    """Build only the state needed to reach the pre-launch dtype guard."""
    PagedDecodeWrapper = _load_wrapper_without_cutlass().PagedDecodeWrapper
    wrapper = PagedDecodeWrapper.__new__(PagedDecodeWrapper)
    wrapper._planned = True
    wrapper._bits = 2
    wrapper._is_sm90 = False
    wrapper._scale_pool_dtype = expected
    wrapper._scale_pool_sdtype = sdtype
    wrapper._k_quant = ("token", 64)
    wrapper._v_quant = ("token", 128)
    wrapper._page_size = 64
    wrapper._head_dim = 128
    return wrapper


@pytest.mark.parametrize(
    ("expected", "sdtype", "wrong"),
    [
        (torch.float32, "f32", torch.bfloat16),
        (torch.bfloat16, "bf16", torch.float32),
    ],
)
def test_wrapper_pscale_rejects_mismatched_pool_before_sync(
    expected, sdtype, wrong
):
    wrapper = _uninitialized_wrapper(expected, sdtype)
    assert wrapper.scale_pool_dtype is expected
    with pytest.raises(ValueError, match="scale_pool.dtype"):
        wrapper.pscale_lp(torch.empty(1, 1, 384, dtype=wrong))


@pytest.mark.parametrize(
    ("expected", "sdtype", "wrong"),
    [
        (torch.float32, "f32", torch.bfloat16),
        (torch.bfloat16, "bf16", torch.float32),
    ],
)
def test_wrapper_run_rejects_mismatched_pool_before_launch(
    expected, sdtype, wrong
):
    wrapper = _uninitialized_wrapper(expected, sdtype)
    with pytest.raises(ValueError, match="scale_pool.dtype"):
        wrapper._run_quantized(
            torch.empty(0),
            torch.empty(0, dtype=torch.uint8),
            torch.empty(0, dtype=torch.uint8),
            torch.empty(1, 1, 384, dtype=wrong),
            None,
            None,
            None,
            None,
            False,
        )


def test_invalid_sdtype_replan_discards_prior_kernel_and_fails_closed(monkeypatch, fold_runtime):
    wrapper_module = _load_wrapper_without_cutlass()
    runtime = fold_runtime

    def fake_plan_sm100(self, kv_splits, reduction, kernel_kwargs):
        kwargs = dict(kernel_kwargs or {})
        if (
            self._k_quant != ("token", self._head_dim)
            or self._v_quant != ("token", self._head_dim)
        ):
            kwargs["k_quant"] = self._k_quant
            kwargs["v_quant"] = self._v_quant
        _, _, sdtype, offsets = runtime.quant_cfg_of(
            self._page_size, self._head_dim, kwargs)
        self._scale_pool_sdtype = sdtype
        self._scale_pool_dtype = (
            torch.bfloat16 if sdtype == "bf16" else torch.float32)
        self._quant_offsets = offsets
        self._quant_kernel_kwargs = kwargs
        self._planned_splits = self._kv_splits = 1
        self._planned_reduction = self._reduction = "none"
        self._has_workspace = False
        self._compiled_fold = object()
        self._compiled_combine = object()

    monkeypatch.setattr(
        wrapper_module.PagedDecodeWrapper,
        "_plan_quantized_sm100",
        fake_plan_sm100,
    )
    wrapper = wrapper_module.PagedDecodeWrapper(torch.empty(4096))
    plan_args = (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([64], dtype=torch.int32),
        32,
        8,
        128,
        64,
    )

    wrapper.plan(
        *plan_args,
        q_data_type=torch.float8_e4m3fn,
        bits=2,
        k_quant=("token", 64),
        v_quant=("token", 128),
        kernel_kwargs={"sdtype": "f32"},
        max_kv_len=64,
    )
    prior_kernel = wrapper._compiled_fold
    assert wrapper._planned
    assert prior_kernel is not None
    assert wrapper.scale_pool_dtype is torch.float32

    with pytest.raises(ValueError, match="sdtype must be"):
        wrapper.plan(
            *plan_args,
            q_data_type=torch.float8_e4m3fn,
            bits=2,
            k_quant=("token", 64),
            v_quant=("token", 128),
            kernel_kwargs={"sdtype": "float32"},
            max_kv_len=64,
        )

    assert not wrapper._planned
    assert wrapper._compiled_fold is None
    assert wrapper._compiled_fold is not prior_kernel
    assert wrapper._compiled_combine is None
    assert wrapper.bits is None
    assert wrapper._scale_pool_dtype is None
    assert wrapper._scale_pool_sdtype is None
    assert wrapper._quant_kernel_kwargs is None
    assert wrapper.kv_splits is None
    assert wrapper.reduction is None
    assert wrapper._q_data_type is None
    with pytest.raises(RuntimeError, match=r"Call plan\(\) before run"):
        wrapper.run(torch.empty(0), torch.empty(0), torch.empty(0))
    with pytest.raises(RuntimeError, match=r"Call plan\(\) before pscale_lp"):
        wrapper.pscale_lp(torch.empty(1, 1, 384, dtype=torch.float32))


# --- the K pool order must be DECLARED ---------------------------------------
# `run()` cannot tell a permuted K pool from a logical one: same dtype, same
# shape, same byte count, and a wrong pairing decodes to finite values with every
# K channel taken from the wrong slot inside its 16-channel group. Every caller
# that reads a permuted pool therefore declares its order, and the public API
# refuses to launch on an undeclared one, which is the door an external cache
# writer arrives at.

def _wrapper_for_k_order(k_quant=("token", 128), bits=2, kkw=None):
    PagedDecodeWrapper = _load_wrapper_without_cutlass().PagedDecodeWrapper
    wrapper = PagedDecodeWrapper.__new__(PagedDecodeWrapper)
    wrapper._planned = True
    wrapper._bits = bits
    wrapper._is_sm90 = False
    wrapper._k_quant = k_quant
    wrapper._v_quant = ("token", 128)
    wrapper._head_dim = 128
    wrapper._quant_kernel_kwargs = dict(kkw or {})
    return wrapper


@pytest.mark.parametrize("k_quant,bits", [
    (("token", 128), 2), (("token", 64), 2),
    (("channel", 64), 2), (("channel", 128), 2), (("channel", 256), 2),
    (("token", 128), 4), (("token", 64), 4),
    (("channel", 64), 4), (("channel", 128), 4), (("channel", 256), 4),
])
def test_run_requires_the_k_pool_order_where_the_pool_is_permuted(k_quant, bits):
    """Every configuration that reads a permuted pool must refuse to launch on
    an undeclared one.

    The channel-wise entries are new as of 2026-08-12: they read the same two-mask
    deposit as the token-wise ones now, so they carry the same pool contract and
    owe the same declaration. The bits=4 rows are new as of 2026-08-15: the
    raw-feed store shuffle made every INT4 pool permuted, so INT4 callers owe
    the declaration too."""
    wrapper = _wrapper_for_k_order(k_quant=k_quant, bits=bits)
    with pytest.raises(ValueError, match="PERMUTED K pool"):
        wrapper._require_declared_k_pool_order(None)
    with pytest.raises(ValueError, match="contradicts this plan"):
        wrapper._require_declared_k_pool_order("logical")
    wrapper._require_declared_k_pool_order("permuted")


@pytest.mark.parametrize("k_quant,bits,kkw", [
    (("token", 128), 2, {"dequant_int2_to_f4": 0}),   # the E4M3 feed
])
def test_run_needs_no_declaration_where_the_pool_is_logical(k_quant, bits, kkw):
    """Every configuration that reads `pack_rows_lut` order keeps working with no
    new argument, so no existing dense caller changes.

    CHANNEL-WISE K WAS IN THIS LIST UNTIL 2026-08-12 and moved to the permuted
    test below when the axis term left `k_pool_is_permuted`; INT4 followed on
    2026-08-15 when the raw-feed store shuffle made its pool permuted at every
    configuration. That this test failed on the day the predicate widened is
    the behaviour wanted: the declaration a caller owes depends on the
    predicate, so the two must be changed together or a caller is told the
    wrong thing."""
    wrapper = _wrapper_for_k_order(k_quant=k_quant, bits=bits, kkw=kkw)
    wrapper._require_declared_k_pool_order(None)
    wrapper._require_declared_k_pool_order("logical")
    with pytest.raises(ValueError, match="LOGICAL"):
        wrapper._require_declared_k_pool_order("permuted")


def test_an_unknown_k_pool_order_is_refused(fold_runtime):
    wrapper = _wrapper_for_k_order()
    with pytest.raises(ValueError, match="must be 'permuted'"):
        wrapper._require_declared_k_pool_order("dim-major")


def test_the_sm90_path_takes_no_declaration(fold_runtime):
    """SM90 has no store permutation, so the argument must be inert there rather
    than raising on a caller that never had a K order to declare."""
    wrapper = _wrapper_for_k_order()
    wrapper._is_sm90 = True
    wrapper._require_declared_k_pool_order(None)


# --- `kernel_cls` is retired, so the public doors refuse it ---------------------
# Until 2026-10-07 `plan(kernel_kwargs={"kernel_cls": "pingpong"})` compiled the SM100
# ping-pong fork, and the K pool order guard asked that fork's predicate. The fork is
# deleted, so both doors must REFUSE a dict that still names the selector, for every
# value it took, rather than plan and check the main kernel while the caller believes
# it asked for the fork. The match is each door's own prefix, because a generic
# TypeError naming `kernel_cls` would also come from the constructor splat deep inside
# a compile, and would pass a test that matched only the name.

_KERNEL_CLS_VALUES = [None, "pingpong"]


def _retired_prefix(where, name="kernel_cls"):
    return rf"{where}: '{name}' was retired"


@pytest.mark.parametrize("value", _KERNEL_CLS_VALUES)
def test_plan_refuses_the_retired_kernel_cls(value):
    """The real `plan()`, on the host-loaded wrapper: the refusal comes before any
    planning or compile, so it needs no device and no DSL, and it leaves the
    wrapper unplanned."""
    wrapper = _load_wrapper_without_cutlass().PagedDecodeWrapper(torch.empty(4096))
    with pytest.raises(TypeError, match=_retired_prefix("PagedDecodeWrapper.plan")):
        wrapper.plan(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([64], dtype=torch.int32),
            32, 8, 128, 64,
            q_data_type=torch.float8_e4m3fn,
            bits=2,
            kernel_kwargs={"kernel_cls": value},
            max_kv_len=64,
        )
    assert not wrapper._planned


@pytest.mark.parametrize("value", _KERNEL_CLS_VALUES)
def test_run_refuses_a_planned_dict_that_names_kernel_cls(value):
    """The K pool order guard reads the planned dict BY NAME, so it consults the
    retired table too: a dict naming the fork must not be answered for the main
    kernel, whose K order is the opposite of the fork's at this configuration."""
    wrapper = _wrapper_for_k_order(kkw={"dequant_int2_to_f4": 1, "kernel_cls": value})
    for order in (None, "logical", "permuted"):
        with pytest.raises(TypeError, match=_retired_prefix("PagedDecodeWrapper.run")):
            wrapper._require_declared_k_pool_order(order)
