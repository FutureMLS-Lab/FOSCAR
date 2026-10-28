#!/usr/bin/env python3
"""CPU gates for the sm100 K store permutations (`k_store_perm_sm100_f4`).

A permutation is a LAYOUT CONTRACT between whoever writes the cache and the
kernel's K deposit: a pool built without it and read by the mask deposit is
wrong at every K channel, and nothing at runtime can detect that. These gates are
what make the contract checkable without a GPU. There is no flag to set: the
order follows from the stored width and the resolved feed, and
``k_pool_is_permuted`` is the one predicate a POOL WRITER asks. Two orders
exist: int2's self-inverse 16-slot sigma (either K axis, since 2026-08-13) and
int4's 8-slot raw-feed shuffle (every configuration, since 2026-08-15), which
is NOT self-inverse, so its gates pin the inverse separately.

Run: pytest tests/test_kperm_sm100_cpu.py
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts" / "probes", ROOT / "scripts", ROOT / "src"):
    sys.path.insert(0, str(p))

from repack_sm100 import (D, PS, _T8_I4, _check_k_store_perm,  # noqa: E402
                          _k_store_perm_inverse, build_pools_flex,
                          build_pools_lut, check_k_store_perm_f4,
                          check_k_store_perm_i4, dequant_pools_flex,
                          dequant_pools_lut, k_store_perm_sm100_f4,
                          v_store_perm_sm100, v_store_perm_sm100_f4)


def test_default_resolution_oracle():
    """Keep the standalone resolver-agreement gate in the regular CPU suite."""
    result = subprocess.run([sys.executable, str(ROOT / "scripts/test_default_resolution.py")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bits", [2, 4])
def test_k_and_v_shuffle_only_the_reduction_axis(bits):
    """Pin every logical coordinate, not merely a pack/unpack round trip."""
    slots = ([0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
             if bits == 2 else [0, 4, 1, 5, 2, 6, 3, 7])
    group = len(slots)
    kp = k_store_perm_sm100_f4(bits)
    vp = v_store_perm_sm100_f4(bits).reshape(D, PS)
    for d in range(D):
        assert kp[d] == (d // group) * group + slots[d % group]
        for t in range(PS):
            assert vp[d, t] == d * PS + (t // group) * group + slots[t % group]
    # In particular, no V value leaves its channel row or packed-word group.
    assert torch.equal(vp // PS, torch.arange(D)[:, None].expand(D, PS))


@pytest.mark.parametrize("bits", [2, 4])
def test_v_two_mask_expansion_and_pool_shape(bits):
    from repack_sm100 import _gate_f4_v_perm, _gate_i4_v_perm
    gate = _gate_f4_v_perm if bits == 2 else _gate_i4_v_perm
    assert gate(seed=17)
    torch.manual_seed(31)
    k = torch.randn(2, PS, 3, D)
    v = torch.randn_like(k)
    pools = build_pools_lut(k, v, bits,
                            k_store_perm=k_store_perm_sm100_f4(bits),
                            v_store_perm=v_store_perm_sm100_f4(bits))
    assert pools[0].shape == (2, 3, PS, D * bits // 8)
    assert pools[1].shape == (2, 3, D, PS * bits // 8)
    assert pools[0].is_contiguous() and pools[1].is_contiguous()


def test_old_int4_whole_page_order_is_not_the_main_pool_contract(monkeypatch):
    """Old pools have the same shape but different bytes; require rebuilding."""
    old = v_store_perm_sm100(4)
    index = torch.arange(D * PS)
    old = old[8 * (index // 8) + torch.tensor(_T8_I4)[index % 8]]
    new = v_store_perm_sm100_f4(4)
    assert not torch.equal(old, new)
    assert torch.equal(torch.sort(new).values, torch.arange(D * PS))
    # Sensitivity control: the expansion oracle must reject an old-layout pool,
    # not simply agree with whichever packing convention it is handed.
    import repack_sm100
    monkeypatch.setattr(repack_sm100, "v_store_perm_sm100_f4", lambda bits: old)
    assert not repack_sm100._gate_i4_v_perm(seed=17)


def k_pool_perm(bits, kernel_kwargs=None):
    """`_fold_harness.k_pool_perm`, imported LAZILY.

    That module pulls in `cutedsl_decode`, hence cutlass, and several gates in
    this file are pure Python over the permutation itself. A module-level import
    would make the whole file uncollectable on a host without the DSL installed,
    which is where the cheap gates are most useful."""
    from _fold_harness import k_pool_perm as _impl
    return _impl(bits, kernel_kwargs)

#: First 16 values and the full-perm sha256, recorded when the perm was added.
#: A later edit that perturbs it changes which channel every stored slot holds,
#: so it must trip a gate rather than silently move every K operand.
_K_PERM_FIRST16 = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
_K_PERM_SHA256 = (
    "7fe67b1559789c021bd65775bd5b160c2567b447ff5cf19a14fe7a799c95e136")


def test_operand_words_match_the_gather():
    """The permuted pool read with two masks equals the unpermuted pool read
    with the production gather, as OPERAND WORDS.

    This is the whole justification for the permutation: it is not "close" to
    the gather, it is the same bytes, so no downstream stage can tell them
    apart and the kernel's output must be bitwise identical."""
    assert check_k_store_perm_f4() is True


def test_perm_is_a_self_inverse_permutation_of_D():
    perm = k_store_perm_sm100_f4(2)
    assert perm.numel() == D
    assert sorted(perm.tolist()) == list(range(D)), "not a permutation"
    assert torch.equal(perm[perm], torch.arange(D)), "not self-inverse"


def test_perm_is_the_same_sigma_the_v_order_composes():
    """K reuses V's 16-slot sigma. If they ever diverge, the appendix claim
    that one shuffle serves both tensors stops being true."""
    sigma = [(i & 3) | ((i & 4) << 1) | ((i & 8) >> 1) for i in range(16)]
    perm = k_store_perm_sm100_f4(2)
    for group in range(D // 16):
        block = perm[16 * group:16 * (group + 1)] - 16 * group
        assert block.tolist() == sigma, f"group {group} is not sigma"


def test_perm_acts_within_each_16_channel_group():
    """It must not move a code across a 16-channel group, or it would cross a
    packed-word boundary and the two-mask read would gather foreign channels."""
    perm = k_store_perm_sm100_f4(2)
    for d in range(D):
        assert perm[d].item() // 16 == d // 16


def test_perm_values_are_pinned():
    perm = k_store_perm_sm100_f4(2)
    assert perm[:16].tolist() == _K_PERM_FIRST16
    digest = hashlib.sha256(perm.to(torch.int64).numpy().tobytes()).hexdigest()
    assert digest == _K_PERM_SHA256


#: First 16 values and the full-perm sha256 of the INT4 shuffle, recorded when
#: it was added (2026-08-15). Same rationale as the int2 pin above.
_K_PERM_I4_FIRST16 = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
_K_PERM_I4_SHA256 = (
    "5e30506735c33024c524a22c93fb526a5d8f0f1810e1744c75105367d6e77f1f")


def test_int4_operand_words_match_the_prmt_compaction():
    """The int4-shuffled pool read with two masks equals the logical pool read
    with the 5-op PRMT compaction, as OPERAND WORDS. Same standing as the int2
    oracle above: bitwise identity, not closeness."""
    assert check_k_store_perm_i4() is True


def test_int4_perm_is_a_permutation_and_not_an_involution():
    """Unlike int2's sigma, t8 is NOT self-inverse, so store and reference must
    use the perm and its argsort respectively. This test pins that asymmetry:
    if someone "simplifies" a reference to reuse the forward perm, the
    round-trip below is what breaks."""
    perm = k_store_perm_sm100_f4(4)
    assert perm.numel() == D
    assert sorted(perm.tolist()) == list(range(D)), "not a permutation"
    assert not torch.equal(perm[perm], torch.arange(D)), (
        "t8 became self-inverse; the P/P^-1 split in every reference is then "
        "masking a different perm than the one documented")
    inv = _k_store_perm_inverse(perm, bits=4)
    assert torch.equal(perm[inv], torch.arange(D))
    assert torch.equal(inv[perm], torch.arange(D))


def test_int4_perm_acts_within_each_8_channel_group():
    """A code moved across an 8-channel group crosses a packed int4 word
    boundary, and the two-mask read would gather foreign channels."""
    perm = k_store_perm_sm100_f4(4)
    for d in range(D):
        assert perm[d].item() // 8 == d // 8
    _check_k_store_perm(perm, bits=4)


def test_int4_perm_is_t8_tiled():
    perm = k_store_perm_sm100_f4(4)
    for group in range(D // 8):
        block = perm[8 * group:8 * (group + 1)] - 8 * group
        assert block.tolist() == list(_T8_I4), f"group {group} is not t8"


def test_int4_perm_values_are_pinned():
    perm = k_store_perm_sm100_f4(4)
    assert perm[:16].tolist() == _K_PERM_I4_FIRST16
    digest = hashlib.sha256(perm.to(torch.int64).numpy().tobytes()).hexdigest()
    assert digest == _K_PERM_I4_SHA256


def test_reference_inverts_the_perm():
    """`dequant_pools_flex` must undo the K order, or a permuted pool would
    read as a different cache and every gate built on it would compare the
    kernel against the wrong numbers."""
    torch.manual_seed(0)
    k_log = torch.randn(1, 2, PS, 1, D, dtype=torch.bfloat16)
    v_log = torch.randn(1, 2, PS, 1, D, dtype=torch.bfloat16)
    kq, vq = ("token", 128), ("token", 128)
    perm = torch.arange(2)
    vperm = v_store_perm_sm100_f4(2)
    kperm = k_store_perm_sm100_f4(2)
    plain = build_pools_flex(k_log, v_log, 2, kq, vq, v_store_perm=vperm,
                             perm=perm)
    permd = build_pools_flex(k_log, v_log, 2, kq, vq, v_store_perm=vperm,
                             perm=perm, k_store_perm=kperm)
    assert not torch.equal(plain[0], permd[0]), "perm did not change the pool"
    # k-order: logical -- this is the UNPERMUTED arm of the paired
    # comparison, and inverting a perm that was never applied is the exact
    # false green the assertion below exists to catch.
    k_plain, v_plain = dequant_pools_flex(*plain[:3], 2, kq, vq,
                                          v_store_perm=vperm)
    k_permd, v_permd = dequant_pools_flex(*permd[:3], 2, kq, vq,
                                          v_store_perm=vperm,
                                          k_store_perm=kperm)
    assert torch.equal(k_plain, k_permd), "reference did not invert the K order"
    assert torch.equal(v_plain, v_permd)


def test_int4_reference_inverts_the_perm():
    """The int4 twin of the round-trip above, and it is the one gate a
    forward-index regression cannot pass: t8 is NOT self-inverse, so a
    reference that indexes an int4 unpack with the FORWARD perm (correct for
    int2's sigma) reads six of every eight channels from the wrong slot, and
    the equality below breaks. `tests/test_pool_call_sites_cpu.py` states it
    cannot distinguish forward from inverse syntactically; this test is the
    value-level check it defers to."""
    torch.manual_seed(0)
    k_log = torch.randn(1, 2, PS, 1, D, dtype=torch.bfloat16)
    v_log = torch.randn(1, 2, PS, 1, D, dtype=torch.bfloat16)
    kq, vq = ("token", 128), ("token", 128)
    perm = torch.arange(2)
    vperm = v_store_perm_sm100_f4(4)
    kperm = k_store_perm_sm100_f4(4)
    plain = build_pools_flex(k_log, v_log, 4, kq, vq, perm=perm)
    permd = build_pools_flex(k_log, v_log, 4, kq, vq, v_store_perm=vperm,
                             perm=perm, k_store_perm=kperm)
    assert not torch.equal(plain[0], permd[0]), "K perm did not change the pool"
    assert not torch.equal(plain[1], permd[1]), "V perm did not change the pool"
    # k-order: logical -- the UNPERMUTED arm of the paired comparison.
    k_plain, v_plain = dequant_pools_flex(*plain[:3], 4, kq, vq)
    k_permd, v_permd = dequant_pools_flex(*permd[:3], 4, kq, vq,
                                          v_store_perm=vperm,
                                          k_store_perm=kperm)
    assert torch.equal(k_plain, k_permd), "reference did not invert the K order"
    assert torch.equal(v_plain, v_permd), "reference did not invert the V order"
    # The regression this test exists for: the forward perm as its own inverse.
    bad = unpack(permd[0], kperm)
    good = unpack(permd[0], _k_store_perm_inverse(kperm, bits=4))
    assert not torch.equal(bad, good), (
        "forward and inverse indexing agree; t8 stopped being the documented "
        "non-involution and this gate lost its teeth")


def unpack(mKq, k_store_perm):
    """Raw-code unpack of a K pool indexed along channels by `k_store_perm` —
    the minimal expression of the forward-vs-inverse regression above. The
    caller passes the FORWARD perm to model the regression and the inverse to
    model the correct reference; the parameter carries the contract name so
    `test_pool_call_sites_cpu` sees the order is taken, not assumed."""
    from repack_sm100 import unpack_rows_lut
    return unpack_rows_lut(mKq, 4, D).float()[..., k_store_perm.long()]


def test_only_the_int2_e4m3_feed_is_logical():
    """The int2 e4m3 feed reads `pack_rows_lut` spacing, so the predicate must
    answer False for it and the pool writer must leave it logical. int4 left
    this list on 2026-08-15: its raw-code deposit reads the 8-slot shuffle at
    every configuration, so the predicate answers True with no feed term.

    This used to be a ctor REFUSAL of an explicit flag. With the flag gone the
    same property is a POSITIVE statement about the predicate, which is the
    stronger form: a refusal only fires when a caller asks, while the predicate
    is what every caller gets."""
    from cutedsl_decode._sm100.int_fold_decode import IntDecodeSm100
    assert IntDecodeSm100.k_pool_is_permuted(2, dequant_int2_to_f4=0) is False
    assert k_pool_perm(2, {"dequant_int2_to_f4": 0}) is None
    assert IntDecodeSm100.k_pool_is_permuted(4) is True
    assert torch.equal(k_pool_perm(4, {}), k_store_perm_sm100_f4(4))
    assert IntDecodeSm100(64, D, 4, bits=4).k_pool_permuted is True


def test_validator_rejects_a_group_crossing_perm():
    """The one wrong perm that would look plausible: a permutation of range(D)
    that moves a code across a 16-channel group, hence across a packed word."""
    bad = torch.arange(D)
    bad[0], bad[16] = 16, 0
    assert sorted(bad.tolist()) == list(range(D))
    with pytest.raises(AssertionError):
        _check_k_store_perm(bad)


def test_validator_rejects_non_permutations_and_cross_width_perms():
    with pytest.raises(AssertionError):
        _check_k_store_perm(torch.zeros(D, dtype=torch.long))
    # An int2 sigma perm crosses int4's 8-channel word groups, so checking it
    # at bits=4 must refuse; the int4 perm at bits=2 loose-passes by design
    # (within-8 implies within-16), which _k_store_perm_inverse relies on.
    with pytest.raises(AssertionError):
        _check_k_store_perm(k_store_perm_sm100_f4(2), bits=4)
    _check_k_store_perm(k_store_perm_sm100_f4(4), bits=2)


def test_legacy_pool_builder_applies_and_inverts_the_perm():
    """`build_pools_lut` is the legacy path `make_pool_problem` and the e2e gate
    use. Before this change it silently ignored the K order, so a flag-on kernel
    could be handed an unpermuted pool."""
    torch.manual_seed(0)
    # build_pools_lut takes [NP, PS, HKV, D]
    k = torch.randn(2, PS, 1, D, dtype=torch.bfloat16)
    v = torch.randn(2, PS, 1, D, dtype=torch.bfloat16)
    kperm = k_store_perm_sm100_f4(2)
    plain = build_pools_lut(k, v, 2)
    permd = build_pools_lut(k, v, 2, k_store_perm=kperm)
    assert not torch.equal(plain[0], permd[0]), "perm did not reach the K pool"
    assert torch.equal(plain[1], permd[1]), "perm must not touch the V pool"
    k_plain, _ = dequant_pools_lut(*plain, 2)
    k_permd, _ = dequant_pools_lut(*permd, 2, k_store_perm=kperm)
    assert torch.allclose(k_plain, k_permd), "reference did not invert the order"


def test_channel_wise_k_is_permuted_too():
    """Channel-wise K carries the store permutation, at every group size.

    This assertion has been inverted twice and the history is the point. It first
    asserted the ctor REFUSED channel-wise K while the flag was on; then that the
    predicate returned False for it, which routed it to the 7-op gather. Both
    encoded a missing reference rather than a property of the machine:
    `check_k_store_perm_f4` proves the two deposits emit byte-identical operand
    words for arbitrary codes with no scale involved, so a channel-wise fold sees
    the same operands either way. `ref_channel_k_flex` now inverts the order, so
    the term is gone and these cells read the 3-op deposit.

    WHAT MAKES IT SAFE is that the deposit restores logical order before anything
    downstream reads an operand, which `check_k_store_perm_f4` proves exhaustively
    on a basis rather than by sampling. Group locality is checked below as a
    separate, weaker property: sigma never moves a code across a 16-channel group,
    so it never crosses a packed word. An earlier version of this test also
    asserted that 64, 128 and 256 are multiples of 16, which codex correctly
    called vacuous: those are token-group sizes, while channel-wise metadata
    carries one scale per logical channel, so the divisibility said nothing about
    this pairing."""
    from cutedsl_decode._sm100.int_fold_decode import IntDecodeSm100
    for k_quant in (("channel", 64), ("channel", 128), ("channel", 256)):
        kwargs = {"dequant_int2_to_f4": 1, "k_quant": k_quant}
        assert IntDecodeSm100.k_pool_is_permuted(2, **kwargs) is True
        perm = k_pool_perm(2, kwargs)
        assert perm is not None
        # sigma stays inside each 16-channel group, so no code crosses a packed
        # word and no scale group is split by the reorder.
        for d in range(perm.numel()):
            assert int(perm[d]) // 16 == d // 16, (d, int(perm[d]))
    # And the property the safety actually rests on, stated where it is used.
    from repack_sm100 import check_k_store_perm_f4
    assert check_k_store_perm_f4() is True


def test_token_wise_k_is_permuted_with_no_kwarg():
    """The shipped configuration, built the way every caller now builds it.

    No kwarg appears here, which is the property under test: the layout follows
    from bits, feed and axis alone."""
    from cutedsl_decode._sm100.int_fold_decode import IntDecodeSm100
    k = IntDecodeSm100(64, D, 4, bits=2, dequant_int2_to_f4=1,
                        k_quant=("token", 128), v_quant=("token", 128))
    assert k.k_pool_permuted is True
    assert IntDecodeSm100(64, D, 4, bits=2).k_pool_permuted is True


def test_the_removed_kwarg_is_gone():
    """Passing the retired flag must FAIL LOUDLY rather than be ignored.

    A kwarg that is silently accepted and does nothing is worse than one that
    raises: a caller who still passes `k_f4_prepermuted=0` believes it is getting
    a logical pool from a kernel that reads a permuted one."""
    from cutedsl_decode._sm100.int_fold_decode import IntDecodeSm100
    with pytest.raises(TypeError):
        IntDecodeSm100(64, D, 4, bits=2, k_f4_prepermuted=0)


#: The retirement tests below construct kernels or call `compile_fold`, both of
#: which import the CuTe DSL. It is absent from the plain CPU environment this suite
#: is named for, so they are gated the way `tests/test_head_dim_guard_cpu.py` gates
#: its behavioural half. The pool-helper test is gated for the same reason even
#: though its refusal itself needs no DSL: importing `_fold_harness` at all pulls
#: `cutedsl_decode._sm100.fold_runtime` in at module level. (codex, 2026-09-08)
_needs_dsl = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None,
    reason="compile_fold and the kernel constructors import the CuTe DSL (cutlass)")

_FOLD_RUNTIME_PATH = ROOT / "src" / "cutedsl_decode" / "_sm100" / "fold_runtime.py"


def _fold_runtime_by_path():
    """`fold_runtime`, loaded WITHOUT importing the package.

    The parametrizations below are built at collection time, and importing
    `cutedsl_decode._sm100.fold_runtime` through the package pulls `wrapper.py`
    and the DSL in, which would make this whole file uncollectable without
    cutlass. `fold_runtime.py` itself imports only `os` and `torch` at module
    level, so it is loaded by path here as a private module. The table is
    read off it, and the guard-level test below calls its
    `refuse_retired_kernel_kwargs` directly, which needs nothing else in the
    module and so runs where the DSL is absent. The door tests then exercise
    the SAME table through the package (DSL present), so a key that one load
    has and the other lacks shows up as a test that names it."""
    spec = importlib.util.spec_from_file_location(
        "_kperm_fold_runtime_by_path", _FOLD_RUNTIME_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _retired_kernel_kwargs():
    """`fold_runtime._RETIRED_KERNEL_KWARGS`: the names refused at every door."""
    return dict(_fold_runtime_by_path()._RETIRED_KERNEL_KWARGS)


#: EVERY entry of the table, so a knob retired tomorrow is covered by these tests
#: the moment its entry lands, with no test to remember to write (2026-09-14; the
#: previous shape was one hand-written name tuple per retirement, which is how the
#: 2026-09-08 entry and the 2026-09-14 entries came to be two near-copies of the
#: same door tests). `identity_dequant`, the seven consolidation entries,
#: `dequant_warpgroups` and `kernel_cls` are all here. The two renamed depth knobs
#: are refused with a "was renamed" message rather than "was retired", which is
#: why the prefix below stops at either verb.
_RETIRED = sorted(_retired_kernel_kwargs())
for _required in ("identity_dequant", "dequant_warpgroups", "kernel_cls"):
    assert _required in _RETIRED, (_required, _RETIRED)

#: Every value `kernel_cls` was ever accepted with: None selected `IntDecodeSm100`
#: and "pingpong" the retired fork. Callers passed both explicitly, so both are
#: what a stale caller still sends, and both must be refused.
_KERNEL_CLS_VALUES = [None, "pingpong"]


def _guard_prefix(name, where="compile_fold"):
    """The guard's own message, matched as a PREFIX.

    Matching only the knob name made the identity-relay test vacuous: with the
    `compile_fold` guard deleted, the constructor splat still raises `TypeError:
    ... unexpected keyword argument 'identity_dequant'`, which satisfies that
    regex, so the test passed while the refusal it exists to protect was gone.
    codex proved that by mutation on 2026-09-08. Matching `<door>: '<name>' was`
    pins the guard (`refuse_retired_kernel_kwargs` formats `f"{where}: {why}"`,
    and every `why` in the table opens with the quoted name and `was retired`
    or `was renamed`) rather than any TypeError that happens to name the knob.

    `where` is the door's OWN `where=`, and it differs per door: `compile_fold`
    (the default), `quant_cfg_of` (which `scale_pool_dtype_of` resolves
    through), `_fold_harness.k_pool_perm` / `_fold_harness.v_pool_perm`, and
    `PagedDecodeWrapper.plan` / `PagedDecodeWrapper.run` (tested in
    `tests/test_sm100_scale_dtype_host.py`). The K7 rewrite pinned the default
    on the quant-config and pool-helper tests, a prefix neither door raises
    with, so all 32 of their cases failed wherever the DSL was present
    (reviewer B, 2026-09-14, fixed in K8)."""
    return rf"{re.escape(where)}: '{re.escape(name)}' was (retired|renamed)"


_DOORS = ["compile_fold", "quant_cfg_of", "_fold_harness.k_pool_perm",
          "_fold_harness.v_pool_perm", "PagedDecodeWrapper.plan",
          "PagedDecodeWrapper.run"]


@pytest.mark.parametrize("name", _RETIRED)
@pytest.mark.parametrize("where", _DOORS)
def test_the_guard_itself_refuses_every_retired_name(name, where):
    """The guard, called by path so this runs without the DSL.

    Every name in the table is refused with the door's own prefix, for a falsy
    value as well as a truthy one, because a retired knob has no value left to
    pass. `kernel_cls` is also refused for each value it was ever accepted with.
    The guard takes no class selector any more: there is one kernel class, so
    the table is not class-sensitive. The door tests below prove each door
    actually consults it."""
    guard = _fold_runtime_by_path().refuse_retired_kernel_kwargs
    values = [False, 2] + (_KERNEL_CLS_VALUES if name == "kernel_cls" else [])
    for value in values:
        with pytest.raises(TypeError, match=_guard_prefix(name, where=where)):
            guard({name: value}, where=where)
    assert guard({}, where=where) is None
    assert guard(None, where=where) is None
    assert guard({"code_stages": 16}, where=where) is None


@_needs_dsl
@pytest.mark.parametrize("name", _RETIRED)
def test_the_retired_kwargs_raise_from_the_constructor(name):
    """The constructor refuses each retired keyword because it no longer exists.

    `dequant_warpgroups` is here as well: the kernel's signature is its door, and
    until K9 it was the only one. So is `kernel_cls`, which was never a
    constructor argument at all."""
    from cutedsl_decode._sm100.int_fold_decode import IntDecodeSm100
    with pytest.raises(TypeError, match=re.escape(name)):
        IntDecodeSm100(64, D, 4, bits=2, **{name: False})


@_needs_dsl
@pytest.mark.parametrize("name", _RETIRED)
@pytest.mark.parametrize("route", ["kernel_kwargs", "bits_tuning"])
def test_compile_folds_own_guard_refuses_the_retired_kwargs(name, route):
    """`compile_fold` must refuse the name on BOTH routes in, with its own message.

    The constructor splat would raise TypeError on its own, but a generic one
    that a test matching only the knob name cannot tell apart from the guard,
    which is why the prefix is the guard's. `bits_tuning` is the second route in,
    because `compile_fold` merges that caller-supplied dict into `kernel_kwargs`
    before construction.

    Every value is refused, including False, because there is no mode left to
    select: eight callers used to pass `identity_dequant=False` explicitly, and
    the B200 A/B harnesses (773717d:baselines/profile_b200_fold_ab.py and
    773717d:baselines/profile_b200_knob_ab.py) pinned `operands_in_tmem=False` and
    `dump_stage=False`."""
    from cutedsl_decode._sm100.fold_runtime import compile_fold
    common = dict(page_size=PS, num_qo_heads=32, num_kv_heads=8,
                  head_dim=D, reduction="kernel", bits=2)
    extra = ({name: False} if route == "kernel_kwargs"
             else {"bits_tuning": {2: {name: False}}})
    with pytest.raises(TypeError, match=_guard_prefix(name)):
        compile_fold(**common, **extra)


@_needs_dsl
@pytest.mark.parametrize("value", _KERNEL_CLS_VALUES)
def test_compile_fold_refuses_kernel_cls_as_it_was_passed(value):
    """`kernel_cls` was a named parameter of `compile_fold` until 2026-10-07, so a
    stale caller passes it by keyword, as the knob matrix, the e2e check and the
    profiler did. With the parameter gone the keyword lands in `**kernel_kwargs`,
    and the guard must refuse it there with its own message, for None as well as
    for "pingpong". Without the table entry the constructor would still raise a
    generic TypeError naming the keyword, which is why the match is the guard's
    prefix and not the name."""
    from cutedsl_decode._sm100.fold_runtime import compile_fold
    with pytest.raises(TypeError, match=_guard_prefix("kernel_cls")):
        compile_fold(page_size=PS, num_qo_heads=32, num_kv_heads=8, head_dim=D,
                     reduction="kernel", bits=2, kernel_cls=value)


@_needs_dsl
@pytest.mark.parametrize("name", _RETIRED)
@pytest.mark.parametrize("value", [True, False])
def test_the_quant_config_helpers_refuse_the_retired_kwargs(name, value):
    """The quant-flex config helpers read the same dict BY NAME, so they refuse too.

    `quant_cfg_of` and `scale_pool_dtype_of` resolve `(k_quant, v_quant, sdtype,
    qoff)` out of `kernel_kwargs`. Before this they returned the DEFAULT
    configuration for a dict carrying a retired key, which is the fifth door
    codex found (2026-09-08) after the pool guard's docstring had claimed the pool
    helpers were the only one. Every door that consumes the dict has to consult
    the shared table."""
    from cutedsl_decode._sm100.fold_runtime import (
        quant_cfg_of, scale_pool_dtype_of)
    # Both raise with `quant_cfg_of`'s prefix: `scale_pool_dtype_of` resolves
    # through `quant_cfg_of`, which is the door that consults the table.
    for helper in (quant_cfg_of, scale_pool_dtype_of):
        with pytest.raises(TypeError, match=_guard_prefix(name, where="quant_cfg_of")):
            helper(PS, D, {name: value})


@_needs_dsl
@pytest.mark.parametrize("name", _RETIRED)
@pytest.mark.parametrize("value", [True, False])
def test_the_pool_helpers_refuse_the_retired_kwargs(name, value):
    """The POOL helpers must refuse too.

    `k_pool_perm` and `v_pool_perm` read their arguments out of a `kernel_kwargs`
    dict BY NAME, so an unrecognised key is discarded rather than rejected the way
    a keyword argument would be. That made them the one remaining place a retired
    knob could be passed and believed: the caller would get the default pool order
    and no warning, which is the accepted-and-ignored failure the owner's rule
    forbids. Found by codex on 2026-09-08. The refusal itself lives in
    `fold_runtime.refuse_retired_kernel_kwargs`, the table every dict-consuming
    door consults."""
    from _fold_harness import k_pool_perm, v_pool_perm
    # Each helper raises with its own door prefix, the `where=` it passes to
    # the guard: `_fold_harness.k_pool_perm` and `_fold_harness.v_pool_perm`.
    for helper in (k_pool_perm, v_pool_perm):
        with pytest.raises(
                TypeError,
                match=_guard_prefix(name, where=f"_fold_harness.{helper.__name__}")):
            helper(2, {name: value})


@_needs_dsl
@pytest.mark.parametrize("value", _KERNEL_CLS_VALUES)
@pytest.mark.parametrize("bits", [2, 4])
def test_the_dict_doors_refuse_kernel_cls_for_every_value_it_took(value, bits):
    """A dict naming `kernel_cls="pingpong"` used to make the pool helpers answer
    for the fork (logical K, token-major V) and the quant-config helpers accept
    the fork's own knobs. With the fork gone, a dict that still names it must be
    refused rather than quietly answered for `IntDecodeSm100`, whose K and V
    orders are the opposite at most configurations; None is refused too, since
    the selector itself is retired."""
    from cutedsl_decode._sm100.fold_runtime import quant_cfg_of, scale_pool_dtype_of
    from _fold_harness import k_pool_perm, v_pool_perm
    kkw = {"kernel_cls": value}
    for helper in (k_pool_perm, v_pool_perm):
        with pytest.raises(
                TypeError, match=_guard_prefix(
                    "kernel_cls", where=f"_fold_harness.{helper.__name__}")):
            helper(bits, kkw)
    for helper in (quant_cfg_of, scale_pool_dtype_of):
        with pytest.raises(TypeError,
                           match=_guard_prefix("kernel_cls", where="quant_cfg_of")):
            helper(PS, D, kkw)


def test_the_knob_matrix_asserts_a_refusal_for_every_retired_kwarg():
    """`scripts/test_knob_matrix_compile.py` must carry an `_expect_refused` row
    whose `_refusal_match` names every key of `_RETIRED_KERNEL_KWARGS`.

    The matrix is the compile-time gate that runs on the node, and a refusal
    nobody asserts there can vanish in a refactor with nothing noticing. The
    script imports `_fold_harness` at module level, so its CONFIGS are read from
    the source by AST rather than imported. The second loop checks that every
    such row also CARRIES the kwarg it names, because the runner counts a
    TypeError as the refusal only when `_refusal_match in cfg`."""
    src = (ROOT / "scripts" / "test_knob_matrix_compile.py").read_text()
    configs = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CONFIGS" for t in node.targets):
            configs = ast.literal_eval(node.value)
    assert configs is not None, "CONFIGS not found in test_knob_matrix_compile.py"
    matched = {cfg["_refusal_match"] for cfg in configs
               if cfg.get("_expect_refused") and "_refusal_match" in cfg}
    missing = sorted(set(_RETIRED) - matched)
    assert not missing, (
        f"retired kwargs with no `_refusal_match` row in the knob matrix: {missing}")
    for cfg in configs:
        if cfg.get("_expect_refused") and "_refusal_match" in cfg:
            assert cfg["_refusal_match"] in cfg, (
                f"row {cfg} names a `_refusal_match` it does not carry, so the "
                f"runner could never count its TypeError as the refusal")


# --- the derived layout (kwarg removed 2026-08-12) -------------------------
# These are the gates that matter most: the order follows from the configuration,
# so the predicate and the pool keying must agree at every knob combination or a
# pool is built one way and read the other with no symptom.

def _predicate():
    from cutedsl_decode._sm100.int_fold_decode import IntDecodeSm100
    return IntDecodeSm100.k_pool_is_permuted


@pytest.mark.parametrize("bits,kw,want", [
    (2, {}, True),                                    # the shipped int2 path
    (2, {"dequant_int2_to_f4": 1}, True),             # explicit feed, same
    (2, {"k_quant": ("token", 64)}, True),            # the other token axis
    (4, {}, True),                                    # int4 raw feed, 2026-08-15
    (2, {"dequant_int2_to_f4": 0}, False),            # e4m3 feed keeps its own
    (2, {"k_quant": ("channel", 128)}, True),         # axis term removed 2026-08-12
    (2, {"k_quant": ("channel", 64)}, True),
    (2, {"k_quant": ("channel", 256)}, True),
    (2, {"fold_scale_zero": 0}, False),               # the ablation arm
])
def test_the_predicate_answers_for_every_knob(bits, kw, want):
    assert _predicate()(bits, **kw) is want


@pytest.mark.parametrize("bits,kw", [
    (2, {}), (4, {}), (2, {"dequant_int2_to_f4": 0}),
    (2, {"k_quant": ("token", 64)}), (2, {"k_quant": ("channel", 128)}),
    (2, {"fold_scale_zero": 0}),
])
def test_pool_keying_cannot_disagree_with_the_kernel(bits, kw):
    """The one property everything else rests on. `k_pool_perm` must return a
    permutation exactly when the kernel reads one, for every knob combination."""
    from cutedsl_decode._sm100.int_fold_decode import IntDecodeSm100 as cls
    kernel_on = cls.k_pool_is_permuted(
        bits,
        dequant_int2_to_f4=kw.get("dequant_int2_to_f4"),
        k_quant=tuple(kw.get("k_quant", ("token", 128))),
        fold_scale_zero=bool(int(kw.get("fold_scale_zero", 1))))
    pool_on = k_pool_perm(bits, kw) is not None
    assert pool_on is kernel_on, (
        f"pool says {pool_on}, kernel says {kernel_on}, for bits={bits} {kw}")


def test_the_shipped_int2_config_is_permuted_with_no_kwarg():
    """Stated as its own gate because it is what the removal asserts: an int2
    kernel built with nothing but its width reads a permuted pool."""
    from cutedsl_decode._sm100.int_fold_decode import IntDecodeSm100
    k = IntDecodeSm100(64, D, 4, bits=2)
    assert k.dequant_int2_to_f4 is True, "precondition: f4 is the int2 default"
    assert k.k_pool_permuted is True
    assert IntDecodeSm100(64, D, 4, bits=4).k_pool_permuted is True
