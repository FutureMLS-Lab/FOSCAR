# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The head_dim guards are a memory-safety boundary, so pin them BEHAVIOURALLY.

Before these guards, `PagedDecodeWrapper.plan(head_dim=64)` reached the SM90 decode
kernel: the wrapper accepted any multiple of 64, resolved `("token", head_dim)` and
passed it on explicitly, `validate_quant_flex` accepted g == D for any D, and the
kernel had no head_dim check. The epilogue then stored channels `output_row` and
`64 + output_row`, and the fused split-K tail indexed the output channel by
warpgroup thread index over 128 lanes, so D=64 wrote past its own row.

THE TWO ARCHITECTURES DIFFER AND THE TESTS MUST NOT FLATTEN THAT. SM90 builds 128 and
256, and 256 in one layout only: its grouped body tiles the output accumulators and the
fused combine by D, and `quant_layout.sm90_wide_refusal` refuses everything else there.
SM100 is any positive multiple of 128, because it tiles D at 128 and masks no partial
tile; demanding exactly 128 there would reject the D=256 its legacy token-D path is
tiled for. On SM100 the wrapper is what narrows the shipped surface to 128, on the ground
that nothing measures anything else there.

WHY THESE CONSTRUCT RATHER THAN GREP. The first version of this file tested the
SOURCE TEXT: it asserted `"head_dim != 128" in txt` and that a `raise` followed. A
dead comment plus an unrelated nearby `raise` satisfied that, one of its cases was
vacuous (it counted asserts that occur before the guard either way), and it missed
two reachable constructors entirely. Constructing the object is what pins behaviour.
The source-text check that remains is only the one a constructed object cannot
show: that the guard is a `raise` rather than an `assert`, since `python -O` is not
what pytest runs under. (codex, 2026-09-04)
"""
import ast
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "cutedsl_decode"

#: Every decode constructor that fixes D, and the rule each one enforces.
D_128_OR_256 = (
    ("_sm90.int_decode_pingpong", "IntPingPongDecodeSm90"),
)
MULTIPLE_OF_128 = (
    ("_sm100.int_fold_decode", "IntDecodeSm100"),
)
ALL_FIXED_D = D_128_OR_256 + MULTIPLE_OF_128

torch = pytest.importorskip("torch", reason="the kernel constructors import torch")

#: CONSTRUCTING a kernel imports the CuTe DSL, which is absent from the plain CPU
#: environment this suite is named for. So the behavioural tests are gated and the
#: source-text floor below is not: on a box without the DSL the floor still fails if
#: a guard is deleted, and on a box with it the constructors are exercised for real.
needs_dsl = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None,
    reason="the kernel constructors import the CuTe DSL (cutlass), absent here")


#: How to construct each kernel with a given head_dim, because the two signatures
#: differ: SM90 takes head_dim first and requires the keyword-only `heads_per_kv`,
#: and SM100 takes (page_size, head_dim, q_head_tile). A single positional helper
#: raised TypeError wherever its call did not fit the signature, and so tested nothing.
#: The quant granularity travels WITH head_dim, because `plan()` resolves
#: `("token", head_dim)` and that is the route the hole was reachable by. For the SM100
#: builder it must: leaving `IntDecodeSm100`'s default ("token", 128) at D=64 makes
#: `quant_layout` reject the pair first, so the test would pass for the wrong reason and
#: prove nothing about the head_dim guard. `IntPingPongDecodeSm90` resolves the same pair
#: itself when it is given none, which the tests at the end of this file pin.
BUILDERS = {
    "IntPingPongDecodeSm90": lambda c, d: c(
        d, heads_per_kv=4, k_quant=("token", d), v_quant=("token", d)),
    "IntDecodeSm100": lambda c, d: c(
        64, d, 8, k_quant=("token", d), v_quant=("token", d)),
}


def _cls(module, name):
    import importlib
    return getattr(importlib.import_module(f"cutedsl_decode.{module}"), name)


def _build(module, name, head_dim):
    return BUILDERS[name](_cls(module, name), head_dim)


@needs_dsl
@pytest.mark.parametrize("module,name", ALL_FIXED_D)
#: Multiples of 64 only. 0, a negative, and 100 are rejected earlier by other shape
#: rules, so parametrizing them would test those rules rather than this guard. 64 is
#: the value that actually reached the SM90 epilogue and wrote past its row.
@pytest.mark.parametrize("bad", [64, 192])
def test_every_fixed_d_kernel_refuses_a_bad_head_dim(module, name, bad):
    """A rejected D must raise ValueError, not sail through and not AssertionError."""
    with pytest.raises(ValueError) as e:
        _build(module, name, bad)
    assert "head_dim" in str(e.value), (
        f"{name} rejected head_dim={bad} without naming head_dim: {e.value}")


@needs_dsl
@pytest.mark.parametrize("module,name", MULTIPLE_OF_128)
def test_sm100_kernels_accept_a_larger_multiple_of_128(module, name):
    """256 is tiled, not rejected. Guarding SM100 at exactly 128 was an over-constraint."""
    try:
        _build(module, name, 256)
    except ValueError as exc:  # pragma: no cover - the regression this pins
        pytest.fail(f"{name} rejected head_dim=256, which its 128-tiling admits: {exc}")
    except Exception:
        # Any other failure is some later constructor rule, not the head_dim guard,
        # and this test is only about the guard.
        pass


@pytest.mark.parametrize("module,name", ALL_FIXED_D)
def test_the_guard_raises_rather_than_asserts(module, name):
    """`python -O` drops `assert`, and this is a memory-safety boundary.

    Source text is the only way to see this: under pytest an `assert` and a `raise`
    are indistinguishable from the outside.
    """
    rel = module.replace(".", "/") + ".py"
    txt = (SRC / rel).read_text()
    guards = [m.start() for m in re.finditer(r"head_dim\s*(?:!=|%)\s*128", txt)]
    assert guards, f"{rel} has no head_dim guard"
    assert any("raise ValueError" in txt[g:g + 600] for g in guards), (
        f"{rel} guards head_dim but no guard raises; an `assert` is removed by "
        f"python -O and this is a memory-safety boundary")


def _accepted_head_dims_literal():
    """`_ACCEPTED_HEAD_DIMS` as the one literal `wrapper.py` states, read from its syntax tree,
    the way `baselines/bench_sweep.py` reads it."""
    tree = ast.parse((SRC / "wrapper.py").read_text())
    found = [node.value for node in tree.body if isinstance(node, ast.Assign)
             and [getattr(t, "id", None) for t in node.targets] == ["_ACCEPTED_HEAD_DIMS"]]
    assert len(found) == 1, "wrapper must state _ACCEPTED_HEAD_DIMS once, as a module constant"
    return ast.literal_eval(found[0])


def test_accepted_head_dims_is_explicit():
    accepted = _accepted_head_dims_literal()
    assert isinstance(accepted, dict)
    assert {arch: set(dims) for arch, dims in accepted.items()} == {
        "sm90": {128, 256}, "sm100": {128}}, (
        f"accepted head dims are {accepted}; add a value here only once a kernel "
        f"implements it AND a run measures it, and update this test in the same commit")
    # SM90's set is the kernel's own envelope, not a second statement of it.
    assert tuple(accepted["sm90"]) == _quant_layout().SM90_HEAD_DIMS
    txt = (SRC / "wrapper.py").read_text()
    assert "head_dim % 64" not in txt, (
        "wrapper must not gate head_dim on a divisibility rule: a multiple of 64 is not "
        "sufficient, and 64 itself writes out of range")


@needs_dsl
def test_plan_refuses_an_unaccepted_head_dim():
    """Behavioural, through the public entry point rather than through its source.

    The head_dim check runs before anything touches the paging tensors, so the empty
    ones below never get read; passing real ones would need a device.
    """
    wrapper = pytest.importorskip(
        "cutedsl_decode.wrapper", reason="the wrapper imports torch and the DSL")
    empty = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError) as e:
        wrapper.PagedDecodeWrapper(torch.zeros(1)).plan(
            indptr=empty, indices=empty, seq_lens=empty,
            num_qo_heads=32, num_kv_heads=8, head_dim=64, page_size=64, bits=4,
        )
    assert "head_dim" in str(e.value)


def _plan_on_cpu(wrapper, head_dim=256, **kw):
    empty = torch.zeros(1, dtype=torch.int32)
    return wrapper.PagedDecodeWrapper(torch.zeros(1)).plan(
        indptr=empty, indices=empty, seq_lens=empty, num_qo_heads=32, num_kv_heads=8,
        head_dim=head_dim, page_size=64, bits=4, **kw)


@needs_dsl
def test_plan_refuses_head_dim_256_on_sm100(monkeypatch):
    """SM100's kernel builds 256 but nothing measures it there, so plan() refuses it once the
    architecture is known, naming SM100."""
    wrapper = pytest.importorskip(
        "cutedsl_decode.wrapper", reason="the wrapper imports torch and the DSL")
    monkeypatch.setattr(wrapper, "_is_sm90", lambda *a, **k: False)
    with pytest.raises(ValueError, match="plan\\(\\) accepts 128 only on SM100"):
        _plan_on_cpu(wrapper)


@needs_dsl
def test_plan_passes_the_head_dim_checks_at_256_on_sm90(monkeypatch):
    """On SM90 both head-dimension checks admit 256, so the call reaches the SM90 planning path.
    An unknown kernel keyword stops it there with that path's own refusal, before anything is
    compiled or any placeholder paging tensor is read on a device."""
    wrapper = pytest.importorskip(
        "cutedsl_decode.wrapper", reason="the wrapper imports torch and the DSL")
    monkeypatch.setattr(wrapper, "_is_sm90", lambda *a, **k: True)
    with pytest.raises(NotImplementedError, match="the SM90 quantized path takes its knobs"):
        _plan_on_cpu(wrapper, kernel_kwargs={"not_a_knob": 1})


class _Planned(Exception):
    """Raised in place of the SM count query, so a plan that passes every layout check stops
    there, before anything is compiled or read on a device."""


def _stop_at_the_planner(*args, **kwargs):
    raise _Planned()


@needs_dsl
@pytest.mark.parametrize("k_quant,v_quant,bf16_pool", [
    (("channel", 128), None, False),
    (None, ("channel", 128), False),
    # V token-D/2: `validate_quant_flex` refuses it with an AssertionError at every head dim.
    (None, ("token", 128), False),
    # K token-D/2: at 128 the path advises a bfloat16 pool, but no pool builds it at 256.
    (("token", 128), None, False),
    (("token", 128), None, True),
    # The built pair with a bfloat16 pool.
    (None, None, True),
])
def test_plan_at_256_refuses_every_layout_but_the_built_one_on_sm90(monkeypatch, k_quant,
                                                                     v_quant, bf16_pool):
    """At 256 the SM90 path refuses every layout but the one the kernel builds with the
    constructor's own reason, ahead of the D=128 envelope's assert and pool advice."""
    wrapper = pytest.importorskip(
        "cutedsl_decode.wrapper", reason="the wrapper imports torch and the DSL")
    monkeypatch.setattr(wrapper, "_is_sm90", lambda *a, **k: True)
    monkeypatch.setattr(wrapper, "get_num_sm", _stop_at_the_planner)
    kernel_kwargs = {"sdtype": torch.bfloat16} if bf16_pool else None
    with pytest.raises(ValueError, match="^at head_dim=256 the SM90 decode builds only"):
        _plan_on_cpu(wrapper, k_quant=k_quant, v_quant=v_quant, kernel_kwargs=kernel_kwargs)


@needs_dsl
@pytest.mark.parametrize("k_quant,v_quant", [(None, None), (("token", 256), ("token", 256))])
def test_plan_at_256_admits_the_built_layout_up_to_the_planner_on_sm90(monkeypatch, k_quant,
                                                                        v_quant):
    """The layout the kernel builds at 256 passes every layout check and reaches the planner."""
    wrapper = pytest.importorskip(
        "cutedsl_decode.wrapper", reason="the wrapper imports torch and the DSL")
    monkeypatch.setattr(wrapper, "_is_sm90", lambda *a, **k: True)
    monkeypatch.setattr(wrapper, "get_num_sm", _stop_at_the_planner)
    with pytest.raises(_Planned):
        _plan_on_cpu(wrapper, k_quant=k_quant, v_quant=v_quant)


@needs_dsl
def test_plan_at_128_still_names_the_pool_k_token_half_needs_on_sm90(monkeypatch):
    """The one-layout refusal is scoped to head dimensions other than 128: at 128, K token-64
    with a float32 pool still gets the advice to build a bfloat16 pool."""
    wrapper = pytest.importorskip(
        "cutedsl_decode.wrapper", reason="the wrapper imports torch and the DSL")
    monkeypatch.setattr(wrapper, "_is_sm90", lambda *a, **k: True)
    monkeypatch.setattr(wrapper, "get_num_sm", _stop_at_the_planner)
    with pytest.raises(ValueError, match="token-64 requires a bfloat16 scale pool"):
        _plan_on_cpu(wrapper, head_dim=128, k_quant=("token", 64))


def test_compile_constructed_refuses_a_mismatched_head_dim():
    """A D=128 kernel compiled against D=64 tensors is the bypass the guard misses."""
    runtime = pytest.importorskip(
        "cutedsl_decode._sm90.pingpong_runtime", reason="imports the DSL")
    src = Path(runtime.__file__).read_text()
    assert "kern.head_dim" in src, (
        "compile_constructed must tie the fake tensor extents to the kernel's own "
        "head_dim; both values are individually legal, so only the pairing is wrong")


def test_the_shared_contract_floor_is_128_not_64():
    contract = pytest.importorskip(
        "cutedsl_decode.kernel_contract", reason="imports torch")
    contract.validate_problem(page_size=64, head_dim=128, q_head_tile=8, bits=2)
    with pytest.raises(AssertionError):
        contract.validate_problem(page_size=64, head_dim=64, q_head_tile=8, bits=2)


@pytest.mark.parametrize("module,name", ALL_FIXED_D)
def test_every_fixed_d_kernel_carries_a_guard_in_source(module, name):
    """The always-runnable floor, for a box without the DSL.

    Weaker than constructing the object, and deliberately kept anyway: without it a
    machine that cannot import cutlass would report a green suite while a guard had
    been deleted outright.
    """
    rel = module.replace(".", "/") + ".py"
    txt = (SRC / rel).read_text()
    assert re.search(r"head_dim\s*(?:!=|%)\s*128", txt), (
        f"{rel} has no head_dim guard at all")


# ---------------------------------------------------------------------------------------------
# No head-dimension default on the Hopper host path
# ---------------------------------------------------------------------------------------------

#: Each entry is the (file under src/ or baselines/, qualified name) of a function on the
#: Hopper host path whose head-dimension default was removed or resolved from the head
#: dimension, and the list holds every such function.
NO_HEAD_DIM_DEFAULT = (
    ("src/cutedsl_decode/_sm90/int_decode_pingpong.py", "IntPingPongDecodeSm90.__init__"),
    ("src/cutedsl_decode/_sm90/pingpong_runtime.py", "compile_pingpong"),
    ("src/cutedsl_decode/_sm90/pingpong_runtime.py", "plan_pingpong"),
    ("src/cutedsl_decode/_sm90/serving_cache.py", "make_serving_cache_flex"),
    ("src/cutedsl_decode/quant_layout.py", "quant_block_offsets"),
    ("src/cutedsl_decode/_sm90/splitk_combine.py", "SplitKCombine.__init__"),
    ("src/cutedsl_decode/_sm90/scheduler.py", "plan_splits"),
    ("src/cutedsl_decode/_sm90/scheduler.py", "_plan_pingpong_model"),
    ("baselines/adapters/ours.py", "_storage_bits_per_element"),
    ("baselines/adapters/ours.py", "OursSm90.__init__"),
)


def _defaults(path, qualname):
    """Return the {parameter: default expression} map of the function `qualname` in `path`,
    and the set of its parameter names."""
    tree = ast.parse((REPO / path).read_text())
    *owner, name = qualname.split(".")
    scope = tree
    for cls in owner:
        scope = next(n for n in scope.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name)
    positional = fn.args.posonlyargs + fn.args.args
    found = dict(zip([a.arg for a in positional][len(positional) - len(fn.args.defaults):],
                     fn.args.defaults))
    found.update({a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
                  if d is not None})
    return found, {a.arg for a in positional + fn.args.kwonlyargs}


@pytest.mark.parametrize("path,qualname", NO_HEAD_DIM_DEFAULT)
def test_no_hopper_host_function_defaults_the_head_dimension(path, qualname):
    """A default of 128 is how a D=256 caller silently gets a D=128 value, whether it is the
    head dimension itself or a ("token", 128) pair. The page size `N` is held to the same
    rule only in `quant_block_offsets`, the one of these that lost its default."""
    defaults, params = _defaults(path, qualname)
    guarded = {"head_dim", "D"} | ({"N"} if qualname == "quant_block_offsets" else set())
    assert (guarded | {"k_quant", "v_quant"}) & params, (
        f"{qualname} takes neither a head dimension nor a quantization pair; the list is stale")
    assert not guarded & set(defaults), (
        f"{qualname} defaults {sorted(guarded & set(defaults))}")
    literal = [p for p, d in defaults.items()
               if isinstance(d, ast.Tuple) and ast.unparse(d) == "('token', 128)"]
    assert not literal, f"{qualname} defaults {literal} to ('token', 128)"


@needs_dsl
@pytest.mark.parametrize("head_dim", [64, 192])
def test_the_hopper_kernel_refuses_a_bad_head_dim_before_the_shared_validator(head_dim):
    """The head-dimension raise comes before `validate_quant_flex`, so it is the first refusal,
    a ValueError under `python -O` too, whether the pair is resolved from the head dimension or
    is the D=128 pair, which fails the shared validator's group asserts at these head
    dimensions."""
    cls = _cls("_sm90.int_decode_pingpong", "IntPingPongDecodeSm90")
    for kw in ({}, dict(k_quant=("token", 128), v_quant=("token", 128))):
        with pytest.raises(ValueError, match="head_dim"):
            cls(head_dim, heads_per_kv=4, **kw)


@needs_dsl
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("fused_combine", [False, True])
@pytest.mark.parametrize("fold", [True, False])
def test_the_hopper_default_pair_builds_the_d128_kernel(bits, fused_combine, fold):
    """The attributes the kernel cache keys on, and every value the host launch reads, are
    those of the kernel built with ("token", 128) for both tensors."""
    from cutedsl_decode import kernel_cache

    cls = _cls("_sm90.int_decode_pingpong", "IntPingPongDecodeSm90")
    common = dict(bits=bits, fused_combine=fused_combine, fold_scale_zero=fold,
                  heads_per_kv=4)
    implicit = cls(128, **common)
    explicit = cls(128, k_quant=("token", 128), v_quant=("token", 128), **common)
    assert kernel_cache.simple_attributes(implicit) == kernel_cache.simple_attributes(explicit)


def _split_parity(monkeypatch):
    """The split-parity script loaded by path with its interpreter re-exec disarmed and no
    `CUTE_DSL_ARCH`, so `main()` returns at the first refusal it reaches."""
    monkeypatch.setenv("BASELINES_ENV_APPLIED", "1")
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    spec = importlib.util.spec_from_file_location(
        "_split_parity_under_test", REPO / "baselines" / "check_sm90_split_parity.py")
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    return parity


@pytest.mark.parametrize("argv,refusal", [
    (["--head-dim", "192"], "the SM90 kernel builds head_dim in (128, 256) only"),
    (["--head-dim", "256", "--flex"], "--flex at --head-dim 256"),
])
def test_the_split_parity_script_refuses_before_the_device(monkeypatch, capsys, argv,
                                                           refusal):
    """A head dimension the kernel does not build, and `--flex` at 256, where no second layout
    is admitted, are refused with status 2 before the architecture and device checks."""
    import sys

    parity = _split_parity(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["check", *argv])
    assert parity.main() == 2
    out = capsys.readouterr().out
    assert refusal in out
    assert "CUTE_DSL_ARCH" not in out, "the head dimension must be refused first"


def test_the_split_parity_script_admits_256_up_to_the_architecture_check(monkeypatch,
                                                                         capsys):
    import sys

    parity = _split_parity(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["check", "--head-dim", "256"])
    assert parity.main() == 2
    assert "CUTE_DSL_ARCH must be sm_90a" in capsys.readouterr().out


def test_the_split_parity_rows_follow_the_head_dimension(monkeypatch):
    parity = _split_parity(monkeypatch)
    full = ("token", 256)
    assert parity.configs_for(256, False) == [(full, full)]
    assert parity.configs_for(128, False) == [parity.FLEX_CONFIGS[0]]
    assert parity.configs_for(128, True) == list(parity.FLEX_CONFIGS)
    assert parity.FLEX_CONFIGS[0] == (("token", 128), ("token", 128))
    with pytest.raises(ValueError):
        parity.configs_for(256, True)


# ---------------------------------------------------------------------------------------------
# The Hopper head-dimension envelope at a head dimension other than 128
# ---------------------------------------------------------------------------------------------

def _quant_layout():
    """`quant_layout.py` loaded by path: it imports nothing, so this needs neither the DSL nor
    the package's `__init__`."""
    spec = importlib.util.spec_from_file_location(
        "_head_dim_guard_quant_layout", SRC / "quant_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Every quantization pair the shared envelope admits at D=256, from the candidate groups the
#: gates draw from, so the refusal below is checked over the pairs a caller can actually pass.
def _d256_pairs(ql):
    groups = (64, 128, 192, 256, 512)
    pairs = []
    for kq in [(a, g) for a in ("token", "channel") for g in groups]:
        for vq in [(a, g) for a in ("token", "channel") for g in groups]:
            try:
                ql.validate_quant_flex(kq, vq, 64, 256)
            except AssertionError:
                continue
            pairs.append((kq, vq))
    return pairs


def test_the_d256_envelope_admits_one_layout_at_two_depths():
    ql = _quant_layout()
    assert ql.SM90_HEAD_DIMS == (128, 256) and ql.SM90_WIDE_STAGES == (2, 4)
    full = ("token", 256)
    assert ql.sm90_wide_layouts(256) == [
        dict(k_quant=full, v_quant=full, sdtype="f32", fold_scale_zero=True, num_stages=s)
        for s in (2, 4)]
    for layout in ql.sm90_wide_layouts(256):
        assert ql.sm90_wide_refusal(256, **layout) is None
        # the list spelling of an admitted pair is the same pair
        assert ql.sm90_wide_refusal(256, ["token", 256], ["token", "256"],
                                    **{k: layout[k] for k in ("sdtype", "fold_scale_zero",
                                                              "num_stages")}) is None


def test_the_d256_envelope_refuses_every_other_pair_fold_off_bf16_and_other_depths():
    ql = _quant_layout()
    full = ("token", 256)
    ok = dict(sdtype="f32", fold_scale_zero=True, num_stages=2)
    pairs = _d256_pairs(ql)
    # the shared envelope admits token-wise K at D and D/2, and channel-wise K and V
    assert (full, full) in pairs and (("token", 128), full) in pairs
    assert (("channel", 64), full) in pairs and (full, ("channel", 256)) in pairs
    for kq, vq in pairs:
        reason = ql.sm90_wide_refusal(256, kq, vq, **ok)
        assert (reason is None) == ((kq, vq) == (full, full)), (kq, vq, reason)
    # The constructor checks the envelope BEFORE `validate_quant_flex`, so the envelope must also
    # refuse every pair the shared validator refuses: it, not the validator's assert, is then the
    # first refusal of such a pair at D=256.
    groups = (64, 128, 192, 256, 512)
    candidates = [(a, g) for a in ("token", "channel") for g in groups]
    refused_by_validator = [(kq, vq) for kq in candidates for vq in candidates
                            if (kq, vq) not in pairs]
    assert ((full, ("token", 128)) in refused_by_validator
            and (("channel", 192), full) in refused_by_validator)
    for kq, vq in refused_by_validator:
        reason = ql.sm90_wide_refusal(256, kq, vq, **ok)
        assert reason is not None and reason.startswith(
            "at head_dim=256 the SM90 decode builds only"), (kq, vq, reason)
    for change in (dict(fold_scale_zero=False), dict(sdtype="bf16"), dict(num_stages=6),
                   dict(num_stages=8), dict(num_stages=0)):
        reason = ql.sm90_wide_refusal(256, full, full, **{**ok, **change})
        assert reason is not None, change
        # the constructor test below matches this prefix, so every refusal must carry it
        assert reason.startswith("at head_dim=256 the SM90 decode builds only"), (change, reason)
    for kq, vq in pairs:
        if (kq, vq) != (full, full):
            reason = ql.sm90_wide_refusal(256, kq, vq, **ok)
            assert reason.startswith("at head_dim=256 the SM90 decode builds only"), (kq, vq)


#: Every D=256 configuration the constructor must refuse, one keyword set per mode, each with
#: the token-wise full-D pair unless the mode is the pair. `sdtype=bf16` funds the half-width
#: K group, so that row is refused by the envelope and not by the half-group sdtype assert. The
#: last three pairs fail `validate_quant_flex`'s group asserts, so they pin that the envelope
#: is checked before the shared validator.
D256_REFUSED = {
    "K channel-wise": dict(k_quant=("channel", 64)),
    "V channel-wise": dict(v_quant=("channel", 128)),
    "both channel-wise": dict(k_quant=("channel", 256), v_quant=("channel", 256)),
    "K half group": dict(k_quant=("token", 128), sdtype=torch.bfloat16),
    "K half group, f32 pool": dict(k_quant=("token", 128)),
    "V half group": dict(v_quant=("token", 128)),
    "K token group past D": dict(k_quant=("token", 512)),
    "K channel group off the grid": dict(k_quant=("channel", 192)),
    "fold off": dict(fold_scale_zero=False),
    "bf16 pool": dict(sdtype=torch.bfloat16),
    "six stages": dict(num_stages=6),
    "eight stages": dict(num_stages=8),
}


@needs_dsl
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("mode", sorted(D256_REFUSED))
def test_every_refused_d256_mode_raises_value_error(mode, bits):
    """Each mode the D=256 envelope refuses is a ValueError from the envelope, never an assert
    and never a kernel built for a layout the grouped body does not implement."""
    cls = _cls("_sm90.int_decode_pingpong", "IntPingPongDecodeSm90")
    kw = dict(k_quant=("token", 256), v_quant=("token", 256), bits=bits, heads_per_kv=4)
    kw.update(D256_REFUSED[mode])
    with pytest.raises(ValueError, match="^at head_dim=256 the SM90 decode builds only"):
        cls(256, **kw)


@needs_dsl
@pytest.mark.parametrize("bits", [2, 4])
def test_the_hopper_default_ring_depth_is_resolved_per_head_dimension(bits):
    """D=128 keeps the inherited bit-aware default (4 at int2, 2 at int4); D=256 resolves 2 at
    both widths, the depth its stage comparison kept; a depth given explicitly is
    kept at either head dimension."""
    cls = _cls("_sm90.int_decode_pingpong", "IntPingPongDecodeSm90")
    assert cls(128, bits=bits, heads_per_kv=4).num_stages == (4 if bits == 2 else 2)
    assert cls(256, bits=bits, heads_per_kv=4).num_stages == 2
    for depth in (2, 4):
        assert cls(128, bits=bits, num_stages=depth, heads_per_kv=4).num_stages == depth
        assert cls(256, bits=bits, num_stages=depth, heads_per_kv=4).num_stages == depth


@needs_dsl
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("fused_combine", [False, True])
@pytest.mark.parametrize("num_stages", [None, 2, 4])
def test_the_hopper_kernel_builds_d256_in_its_one_layout(bits, fused_combine, num_stages):
    """256 constructs in the admitted layout, with the pair given or resolved, and selects the
    grouped body; 128 keeps the whole-D body."""
    cls = _cls("_sm90.int_decode_pingpong", "IntPingPongDecodeSm90")
    for kw in ({}, dict(k_quant=("token", 256), v_quant=("token", 256))):
        kern = cls(256, bits=bits, fused_combine=fused_combine, num_stages=num_stages,
                   heads_per_kv=4, **kw)
        assert kern.head_dim == 256 and kern.group_channels == 128
        assert kern.num_stages in (2, 4)
    assert cls(128, bits=bits, heads_per_kv=4).group_channels is None
