# src/fosca/kernel_contract.py
"""The constructor contract both decode kernels share: names, meanings, validation.

NO KERNEL BODY LIVES HERE, and that restriction is the point rather than an oversight.
`IntPingPongDecodeSm90` and `IntDecodeSm100` share an interface and almost nothing else --
one is a warp-specialized pingpong on SM90, the other a tcgen05 fold on SM100 -- and a
base class that reached into either body would be the third mechanism in this repository
whose duplication has already cost a bug. What is genuinely common is the vocabulary a
CALLER has to speak, so that is what this file owns.

Torch-free, like `quant_layout.py`, so both kernel modules can import it.

WHY THE CONTRACT IS ONE FILE. No argument name may mean different things across the two
kernels, and `sdtype` is the one that would otherwise drift, because the two kernels spell
its vocabulary differently:

    SM100: ("f32", "bf16")                  SM90: (torch.float32, torch.bfloat16)

A caller holding one configuration must be able to hand it to both, so both spellings are
accepted everywhere, and `normalize_sdtype` is the one place that decides.

`page_size == 64` and `bits in (2, 4)` are each checked once for the same reason: two
copies that agree today are free to drift, and a single copy cannot drift against itself.

WHAT IS DELIBERATELY NOT UNIFIED, because unifying it would be wrong rather than merely
hard. This file normalizes the SPELLING of `sdtype`; it does not decide which sdtype a
configuration may use, and the two kernels genuinely disagree there:

    SM100   at D=128, bf16 REQUIRED when K is token-(D/2), because token-64 doubles its
            ring section and the halved scale element is what funds it; elsewhere bf16
            is permitted alongside the f32 default, matching SM90. At D=256 SM100
            builds only token-wise K and V over all D channels with an f32 pool.
    SM90    at D=128, bf16 REQUIRED when K is token-(D/2), because token-64 doubles its
            ring section and the halved scale element is what funds it; elsewhere bf16
            is merely PERMITTED, there being no assert against it. At any other head
            dimension SM90 builds only token-wise K and V over all D channels with an
            f32 pool, so it refuses a half-width K group and a bf16 pool alike
            (`quant_layout.sm90_wide_refusal`).

    So the asymmetry is about what each kernel FORBIDS, not about V token-64: that
    granularity is deferred on both arches, `validate_quant_flex` refusing it in one
    place for both.

That is not drift, and the difference is narrower than it looks. BOTH rules key on K
alone -- V token-64 is deferred on both arches, refused in one place by
`validate_quant_flex` -- so neither kernel needs a V term. What they differ about is
what happens OUTSIDE K token-(D/2): SM100 asserts f32 there and SM90 at D=128 asserts
nothing, so bf16 is legal on SM90 at D=128 and refused on SM100. That is a difference in
what each kernel FORBIDS, not in what either supports, and it is why each keeps its own
rule next to the envelope that explains it.
"""
from __future__ import annotations

#: The canonical spelling of the scale-pool element type. STRINGS, because this file is
#: torch-free by the same rule `quant_layout.py` follows, and because a name survives
#: being written into a CSV, a cell name or a PTX fingerprint label, which a torch dtype
#: object does not.
SDTYPE_NAMES = ("f32", "bf16")

#: Torch spellings a caller may pass instead. Matched on the OBJECT'S TYPE -- a
#: `torch.dtype` -- and only then by its name, so that importing torch is not required
#: and so that the literal string "torch.float32" is NOT quietly accepted as a dtype.
#: Matching on `str(x)` alone did both.
_TORCH_SDTYPE = {
    "torch.float32": "f32",
    "torch.float": "f32",
    "torch.bfloat16": "bf16",
}


def _is_torch_dtype(x):
    t = type(x)
    # `torch` EXACTLY, not a prefix: `t.__module__.split(".")[0] == "torch"` admitted a
    # class declaring itself in "torch.fake". Class metadata is writable, so this is a
    # convenience check and not a security boundary; it exists to keep an honest caller
    # from being surprised, and the docstring says which spellings are meant.
    return t.__name__ == "dtype" and t.__module__ == "torch"

#: Every `q_head_tile` the SM100 kernel can build. SM90 narrows this to exactly 8, and
#: says why at its own assert: its inline RS wgmma is m64n8k16.
Q_HEAD_TILES = (1, 2, 4, 8, 16, 32)


def normalize_sdtype(sdtype):
    """The canonical name for a scale-pool element type, from either spelling.

    Accepts "f32"/"bf16" and `torch.float32`/`torch.bfloat16`, and raises on anything
    else with a message that names what it got and which vocabularies it accepts, because
    an `assert sdtype in (...)` says neither.
    """
    if isinstance(sdtype, str):
        if sdtype in SDTYPE_NAMES:
            return sdtype
    elif _is_torch_dtype(sdtype):
        name = _TORCH_SDTYPE.get(str(sdtype))
        if name is not None:
            return name
    raise AssertionError(
        f"sdtype must be one of {SDTYPE_NAMES} or the torch spelling of the same "
        f"(torch.float32, torch.bfloat16); got {sdtype!r}")


def validate_problem(page_size, head_dim, q_head_tile, bits):
    """The four shape arguments both kernels take, checked once.

    Each kernel narrows this further from its own hardware, and those narrower asserts
    stay where they are: SM90 requires `q_head_tile == 8`, and SM100 requires
    `q_head_tile >= 4` in real mode because 2 makes the compiler emit an unaligned
    `cvt_fptrunc`. What is HERE is only what is true of both.
    """
    assert page_size == 64, (
        f"both kernels tile KV at 64: SM90's single-WG QK atom assumes it and SM100's TMA "
        f"box does; got page_size={page_size}")
    # The 2-D-half PV loop is a head_dim assumption, not a page_size one: the PV mma emits 64
    # channels, so the number of output accumulators is head_dim//64.
    # THIS IS THE SHARED FLOOR, AND IT IS 128, not 64. Both consumers refuse a smaller D --
    # SM90 any D outside {128, 256}, the head dimensions its output tiling and its
    # thread-indexed fused tail are built for, SM100 any D that is not a multiple of 128,
    # because it tiles D at 128 and masks no partial tile. A contract that admitted 64
    # would describe a configuration neither kernel implements, and each constructor
    # narrows this floor further from its own hardware.
    assert head_dim > 0 and head_dim % 128 == 0, (
        f"head_dim must be a positive multiple of 128; got {head_dim}")
    assert q_head_tile in Q_HEAD_TILES, (
        f"q_head_tile must be one of {Q_HEAD_TILES}; got {q_head_tile}")
    assert bits in (2, 4), f"token-wise quantized KV: bits must be 2 or 4; got {bits}"


class GQAPagedDecode:
    """Base of both decode kernels. Carries the contract, and no kernel body.

    Subclasses call `_validate_shape` from their own `__init__` and then add whatever
    their hardware narrows. Nothing here is traced, and nothing here touches state a
    kernel builds.

    THE TWO NAMES THAT MEAN DIFFERENT THINGS, spelled out because they read alike:

      `q_head_tile`   how many Q head ROWS one threadblock owns. On SM100 it is the MMA
                      N tile factor; on SM90 it is fixed at 8 by the wgmma shape. It is
                      a property of the KERNEL.
      `heads_per_kv`  the GQA ratio, q_heads // kv_heads. It is a property of the
                      PROBLEM.

    Both are named here because they read alike, and the relation between them is NOT the
    same on the two kernels, so no rule about it lives in this file.
    SM100 packs the grouped heads on the tiled mode and launches
    `ceil_div(q_heads // kv_heads, q_head_tile)` head tiles per KV head, so a ratio
    larger than the tile is SPREAD over several threadblocks. SM90's inline RS wgmma
    produces exactly `q_head_tile` head columns and refuses a larger ratio, because
    there is nowhere else for the rest to go. SM90's bound is its own and must not be
    restated here as if it applied to both.

    SM90 also takes `heads_per_kv` at construction, it being trace-time there, while
    SM100 does not take it at all and derives it at launch from the head counts. That
    asymmetry is real: forcing a parameter into SM100 that its launch path already
    computes would add a way for a caller to disagree with the tensors it passes.
    """

    def _validate_shape(self, page_size, head_dim, q_head_tile, bits):
        """The four shape arguments, checked where a kernel's `__init__` begins.

        SDTYPE IS DELIBERATELY NOT PART OF THIS. SM100's default is `None`, meaning
        "resolve it from the K axis", and that resolution needs `k_axis`/`k_group`,
        which are not known until `validate_quant_flex` has run. Folding the two
        together would make the contract reject `None`. Each kernel calls
        `normalize_sdtype` at the point it has a value to normalize.
        """
        validate_problem(page_size, head_dim, q_head_tile, bits)
