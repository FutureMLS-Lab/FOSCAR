#!/usr/bin/env python3
"""CPU gates for the sm100 K store permutations (`k_store_perm_sm100_f4`).

A permutation is a LAYOUT CONTRACT between whoever writes the cache and the
kernel's K deposit: a pool built without it and read by the mask deposit is
wrong at every K channel, and nothing at runtime can detect that. These gates are
what make the contract checkable without a GPU. There is no flag to set: the
order follows from the stored width and the resolved feed, and
``k_pool_is_permuted`` is the one predicate a POOL WRITER asks. Two orders
exist: int2's self-inverse 16-slot sigma, which applies to either K axis, and
int4's 8-slot raw-feed shuffle, which applies at every configuration and is
NOT self-inverse, so its gates pin the inverse separately.

Run: pytest tests/test_kperm_sm100_cpu.py
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

import pytest
import torch

pytest.importorskip("cuda", reason="needs the CUDA Python bindings")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fosca.pool import (D, PS, _T8_I4, _check_k_store_perm,  # noqa: E402
                                 _k_store_perm_inverse, build_pools_flex,
                                 build_pools, check_k_store_perm_f4,
                                 check_k_store_perm_i4, dequant_pools_flex,
                                 dequant_pools, k_store_perm_sm100_f4,
                                 v_store_perm_sm100, v_store_perm_sm100_f4)


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
    from fosca.pool import _gate_f4_v_perm, _gate_i4_v_perm
    gate = _gate_f4_v_perm if bits == 2 else _gate_i4_v_perm
    assert gate(seed=17)
    torch.manual_seed(31)
    k = torch.randn(2, PS, 3, D)
    v = torch.randn_like(k)
    pools = build_pools(k, v, bits,
                            k_store_perm=k_store_perm_sm100_f4(bits),
                            v_store_perm=v_store_perm_sm100_f4(bits))
    assert pools[0].shape == (2, 3, PS, D * bits // 8)
    assert pools[1].shape == (2, 3, D, PS * bits // 8)
    assert pools[0].is_contiguous() and pools[1].is_contiguous()


def test_old_int4_whole_page_order_is_not_the_main_pool_contract(monkeypatch):
    """The whole-page int4 order and the main pool contract have the same shape
    but different bytes, so a pool built with one cannot be read as the other."""
    old = v_store_perm_sm100(4)
    index = torch.arange(D * PS)
    old = old[8 * (index // 8) + torch.tensor(_T8_I4)[index % 8]]
    new = v_store_perm_sm100_f4(4)
    assert not torch.equal(old, new)
    assert torch.equal(torch.sort(new).values, torch.arange(D * PS))
    # Sensitivity control: the expansion oracle must reject an old-layout pool,
    # not simply agree with whichever packing convention it is handed.
    from fosca import pool as pool_mod
    monkeypatch.setattr(pool_mod, "v_store_perm_sm100_f4", lambda bits: old)
    assert not pool_mod._gate_i4_v_perm(seed=17)


def k_pool_perm(bits, kernel_kwargs=None):
    """The K pool permutation for this configuration, or None when it has none.

    Which order a K pool must be written in is a property of the configuration,
    so this asks the kernel's own predicate rather than reading a flag out of the
    dict. The import is lazy because the predicate lives beside the kernel, which
    pulls in the CuTe DSL; the gates above are pure Python and must stay
    collectable without it."""
    from fosca._sm100.int_fold_decode import IntDecodeSm100

    kkw = dict(kernel_kwargs or {})
    if not IntDecodeSm100.k_pool_is_permuted(
            bits,
            dequant_int2_to_f4=kkw.get("dequant_int2_to_f4"),
            k_quant=tuple(kkw.get("k_quant", ("token", 128))),
            fold_scale_zero=bool(int(kkw.get("fold_scale_zero", 1)))):
        return None
    return k_store_perm_sm100_f4(bits)

#: First 16 values and the full-perm sha256.
#: An edit that perturbs either changes which channel every stored slot holds,
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
    """K reuses V's 16-slot sigma. If they diverge, one shuffle cannot serve
    both tensors."""
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


#: First 16 values and the full-perm sha256 of the INT4 shuffle. Same rationale
#: as the int2 pin above.
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
    from fosca.pool import unpack_rows
    return unpack_rows(mKq, 4, D).float()[..., k_store_perm.long()]


def test_only_the_int2_e4m3_feed_is_logical():
    """The int2 e4m3 feed reads `pack_rows` spacing, so the predicate must
    answer False for it and the pool writer must leave it logical. int4's
    raw-code deposit reads the 8-slot shuffle at every configuration, so the
    predicate answers True for it with no feed term.

    The property is a positive statement about the predicate, which is the
    stronger form: a refusal only fires when a caller asks, while the predicate
    is what every caller gets."""
    from fosca._sm100.int_fold_decode import IntDecodeSm100
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
    """`build_pools` is the legacy path `make_pool_problem` and the e2e gate
    use. It applies the K store permutation to the K pool only, and its
    reference dequantization inverts it."""
    torch.manual_seed(0)
    # build_pools takes [NP, PS, HKV, D]
    k = torch.randn(2, PS, 1, D, dtype=torch.bfloat16)
    v = torch.randn(2, PS, 1, D, dtype=torch.bfloat16)
    kperm = k_store_perm_sm100_f4(2)
    plain = build_pools(k, v, 2)
    permd = build_pools(k, v, 2, k_store_perm=kperm)
    assert not torch.equal(plain[0], permd[0]), "perm did not reach the K pool"
    assert torch.equal(plain[1], permd[1]), "perm must not touch the V pool"
    k_plain, _ = dequant_pools(*plain, 2)
    k_permd, _ = dequant_pools(*permd, 2, k_store_perm=kperm)
    assert torch.allclose(k_plain, k_permd), "reference did not invert the order"


def test_channel_wise_k_is_permuted_too():
    """Channel-wise K carries the store permutation, at every group size.

    `check_k_store_perm_f4` proves the two deposits emit byte-identical operand
    words for arbitrary values with no scale involved, so a channel-wise fold
    sees the same operands either way. `ref_channel_k_flex` inverts the order
    before anything downstream reads an operand, so these cells read the 3-op
    deposit.

    WHAT MAKES IT SAFE is that the deposit restores logical order before
    anything downstream reads an operand, which `check_k_store_perm_f4` proves
    exhaustively on a basis rather than by sampling. Group locality is checked
    below as a separate, weaker property: sigma never moves a value across a
    16-channel group, so it never crosses a packed word."""
    from fosca._sm100.int_fold_decode import IntDecodeSm100
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
    from fosca.pool import check_k_store_perm_f4
    assert check_k_store_perm_f4() is True


def test_token_wise_k_is_permuted_with_no_kwarg():
    """The shipped configuration, built the way every caller builds it.

    No kwarg appears here, which is the property under test: the layout follows
    from bits, feed and axis alone."""
    from fosca._sm100.int_fold_decode import IntDecodeSm100
    k = IntDecodeSm100(64, D, 4, bits=2, dequant_int2_to_f4=1,
                        k_quant=("token", 128), v_quant=("token", 128))
    assert k.k_pool_permuted is True
    assert IntDecodeSm100(64, D, 4, bits=2).k_pool_permuted is True


def test_the_removed_kwarg_is_gone():
    """A kwarg the constructor does not accept must FAIL LOUDLY rather than be
    ignored.

    A kwarg that is silently accepted and does nothing is worse than one that
    raises: a caller who passes `k_f4_prepermuted=0` believes it is getting a
    logical pool from a kernel that reads a permuted one."""
    from fosca._sm100.int_fold_decode import IntDecodeSm100
    with pytest.raises(TypeError):
        IntDecodeSm100(64, D, 4, bits=2, k_f4_prepermuted=0)


# --- the derived layout -----------------------------------------------------
# These are the gates that matter most: the order follows from the configuration,
# so the predicate and the pool keying must agree at every knob combination or a
# pool is built one way and read the other with no symptom.

def _predicate():
    from fosca._sm100.int_fold_decode import IntDecodeSm100
    return IntDecodeSm100.k_pool_is_permuted


@pytest.mark.parametrize("bits,kw,want", [
    (2, {}, True),                                    # the shipped int2 path
    (2, {"dequant_int2_to_f4": 1}, True),             # explicit feed, same
    (2, {"k_quant": ("token", 64)}, True),            # the other token axis
    (4, {}, True),                                    # int4 raw feed
    (2, {"dequant_int2_to_f4": 0}, False),            # e4m3 feed keeps its own
    (2, {"k_quant": ("channel", 128)}, True),         # channel-wise K carries it too
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
    from fosca._sm100.int_fold_decode import IntDecodeSm100 as cls
    kernel_on = cls.k_pool_is_permuted(
        bits,
        dequant_int2_to_f4=kw.get("dequant_int2_to_f4"),
        k_quant=tuple(kw.get("k_quant", ("token", 128))),
        fold_scale_zero=bool(int(kw.get("fold_scale_zero", 1))))
    pool_on = k_pool_perm(bits, kw) is not None
    assert pool_on is kernel_on, (
        f"pool says {pool_on}, kernel says {kernel_on}, for bits={bits} {kw}")


def test_the_shipped_int2_config_is_permuted_with_no_kwarg():
    """Stated as its own gate because it is the shipped configuration: an int2
    kernel built with nothing but its width reads a permuted pool."""
    from fosca._sm100.int_fold_decode import IntDecodeSm100
    k = IntDecodeSm100(64, D, 4, bits=2)
    assert k.dequant_int2_to_f4 is True, "precondition: f4 is the int2 default"
    assert k.k_pool_permuted is True
    assert IntDecodeSm100(64, D, 4, bits=4).k_pool_permuted is True
