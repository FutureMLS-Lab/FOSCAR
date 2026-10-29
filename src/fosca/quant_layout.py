# src/fosca/quant_layout.py
"""The scale-block layout and quantization envelope, shared by BOTH decode kernels.

Torch-free, so the kernel modules can import it.  Per (page, kv-head) the scale
block is SECTIONED, sdtype, contiguous:

    [K ring section | V ring section | K channel section | V channel section]

ring sections (token-wise, staged through the TMA scale ring):
    ("token", D):    [N, 2] = (s, z) per token                    (2N elems)
    ("token", D/2):  [N, 4] = (s_lo, z_lo, s_hi, z_hi) per token  (4N elems)
channel sections (read per token-GROUP; ride the same per-page block):
    ("channel", Gt): [2, D] = row 0 s_d, row 1 z_d — the page's group slice,
                     duplicated per page when Gt > N (pages of one group carry
                     identical copies)

The legacy interleaved [N, 4] = (s_k, z_k, s_v, z_v) layout is preserved
EXACTLY when both tensors are ("token", D) — the shipped fast path keeps its
[NP, HKV, N, 4] f32 mScale contract and compiles byte-identical.

THIS FILE IS THE ONE COPY, AND IT MUST STAY THAT WAY.  One implementation of
`quant_block_offsets` is better than two that agree, because two copies are free
to drift apart.

`validate_quant_flex` is the same rule for the envelope, and duplicating it is
worse than duplicating a layout, because the copies need not even disagree
visibly: two inline constructor asserts can admit in one place what they refuse
in another.  This function refuses a configuration once, where a reader looking
for the envelope will find it.

`sm90_wide_refusal` and `sm90_wide_layouts` are the SM90 decode's rule for a head
dimension other than 128, an SM90 rule like SM100's own head-dimension rule.  They live
here for the same reason: the constructor refuses with the first, and the admitted
layouts are enumerated by the second, so a reader can see both from one place.
"""


def _norm_pair(q, which):
    """Normalize one (axis, group) config to a plain `(str, int)` tuple.

    The point of normalizing HERE, once, is consistency: a config compared as
    `k_quant == ("token", D)` would let a LIST `["token", 128]` fall through to the
    sectioned path and take a DIFFERENT scale-pool layout for the same configuration.
    Normalizing first is the whole of the input contract: `int()` on the group, exactly
    as `validate_quant_flex` does, so callers that already pre-convert and callers that
    do not agree on what a config means.

    It reads `q[0]` and `q[1]` and converts, and adds no validation of its own: no
    caller can produce an input that a shape or axis check would trip, so such a check
    would only narrow the set of configs this accepts.
    """
    return (q[0], int(q[1]))


def quant_block_offsets(k_quant, v_quant, N, D):
    """Return the element offsets and sizes of the per-(page, hk) scale block for a
    (k_quant, v_quant) pair at page size N and head dimension D, as dict(k_ring, v_ring,
    ring_tot, k_ch, v_ch, block_tot, legacy). The offsets count elements, and *_ring and
    *_ch are None when that tensor does not use the section. N and D have no defaults,
    because the layout depends on both and a defaulted head dimension would be silently
    wrong at any other."""
    k_quant = _norm_pair(k_quant, "k_quant")
    v_quant = _norm_pair(v_quant, "v_quant")

    def ring_elems(qc):
        axis, g = qc
        if axis == "channel":
            return 0
        return 2 * N if g == D else 4 * N

    def ch_elems(qc):
        return 2 * D if qc[0] == "channel" else 0

    legacy = k_quant == ("token", D) and v_quant == ("token", D)
    if legacy:
        return dict(k_ring=0, v_ring=0, ring_tot=4 * N, k_ch=None, v_ch=None,
                    block_tot=4 * N, legacy=True)
    kr, vr = ring_elems(k_quant), ring_elems(v_quant)
    kc, vc = ch_elems(k_quant), ch_elems(v_quant)
    return dict(k_ring=0 if kr else None, v_ring=kr if vr else None,
                ring_tot=kr + vr,
                k_ch=kr + vr if kc else None, v_ch=kr + vr + kc if vc else None,
                block_tot=kr + vr + kc + vc, legacy=False)


#: The head dimensions `IntPingPongDecodeSm90` builds. At 128 it builds every configuration
#: `validate_quant_flex` admits. At any other it runs its grouped body, which builds one layout at
#: the ring depths in `SM90_WIDE_STAGES` (`sm90_wide_layouts`), and refuses everything else.
SM90_HEAD_DIMS = (128, 256)
#: The ring stage depths admitted at a head dimension other than 128. Both fit two CTAs per SM at
#: D=256 (84.7 KiB of shared memory per CTA at int4 with four stages); int4 at six would not.
SM90_WIDE_STAGES = (2, 4)


def sm90_wide_layouts(head_dim):
    """Every layout the SM90 decode admits at `head_dim` other than 128, as constructor keywords.

    The grouped body implements one layout: token-wise K and V over all `head_dim` channels, the
    fold, and an f32 scale pool, at each depth of `SM90_WIDE_STAGES`. `sdtype` is the normalized
    name `kernel_contract.normalize_sdtype` returns. The layout leaves the bit width,
    `fused_combine`, `phase_lock` and `int2_fast_mantissa_injection` free, and the coverage
    matrix at this head dimension is generated from this list, crossed with the first three. The
    GQA ratio is free too, and it is sampled at 2, 4 and 8, as at head dimension 128."""
    full = ("token", head_dim)
    return [dict(k_quant=full, v_quant=full, sdtype="f32", fold_scale_zero=True,
                 num_stages=stages) for stages in SM90_WIDE_STAGES]


def sm90_wide_refusal(head_dim, k_quant, v_quant, *, sdtype, fold_scale_zero, num_stages):
    """Why the SM90 decode refuses this configuration at `head_dim` other than 128, or None when
    `sm90_wide_layouts` admits it.

    Every refusal is a reason string rather than an assert, so the constructor raises
    `ValueError` with it under `python -O` too. The constructor calls this before
    `validate_quant_flex`, so it refuses a pair the shared validator would also refuse, and every
    refusal starts "at head_dim=D the SM90 decode builds only". The pairs are compared after
    `_norm_pair`, so a list spelling of an admitted pair is admitted."""
    pairs = (_norm_pair(k_quant, "k_quant"), _norm_pair(v_quant, "v_quant"))
    full = ("token", head_dim)
    if pairs != (full, full):
        return (f"at head_dim={head_dim} the SM90 decode builds only token-wise K and V over all "
                f"{head_dim} channels, k_quant=v_quant={full}; got k_quant={pairs[0]}, "
                f"v_quant={pairs[1]}")
    if not fold_scale_zero:
        return (f"at head_dim={head_dim} the SM90 decode builds only the fold path; the "
                f"fold_scale_zero=False ablation is built at head_dim=128 only")
    if sdtype != "f32":
        return (f"at head_dim={head_dim} the SM90 decode builds only an f32 scale pool; "
                f"got sdtype={sdtype!r}")
    if num_stages not in SM90_WIDE_STAGES:
        return (f"at head_dim={head_dim} the SM90 decode builds only num_stages in "
                f"{SM90_WIDE_STAGES}; got {num_stages}")
    return None


def validate_quant_flex(k_quant, v_quant, page_size, head_dim):
    """Normalize + validate the per-tensor (axis, group) configs against the
    SM100 fold kernel's envelope (SM90 IntPingPongDecodeSm90 parity):
      K: ("token", D), ("token", D//2), ("channel", Gt in {64, 128, 256})
      V: ("token", D), ("channel", Gt in {64, 128, 256})
    Returns (k_quant, v_quant) as plain tuples."""
    k_quant = _norm_pair(k_quant, "k_quant")
    v_quant = _norm_pair(v_quant, "v_quant")
    k_axis, k_group = k_quant
    v_axis, v_group = v_quant
    assert k_axis in ("token", "channel"), k_quant
    assert v_axis in ("token", "channel"), v_quant
    if k_axis == "token":
        assert k_group in (head_dim, head_dim // 2), (
            f"K token-wise group must be D or D/2 ({head_dim}/{head_dim // 2}); "
            f"got {k_group}")
    else:
        assert k_group in (64, 128, 256) and k_group % page_size == 0, (
            f"K channel-wise token-group must be in {{64, 128, 256}} and a "
            f"multiple of page_size={page_size}; got {k_group}")
    if v_axis == "token":
        assert v_group == head_dim, (
            "V token-wise supports group == D only (token-64 V deferred: "
            "dual scaled-sP planes — SM90 parity)")
    else:
        assert v_group in (64, 128, 256) and v_group % page_size == 0, (
            f"V channel-wise token-group must be in {{64, 128, 256}} and a "
            f"multiple of page_size={page_size}; got {v_group}")
    return k_quant, v_quant
