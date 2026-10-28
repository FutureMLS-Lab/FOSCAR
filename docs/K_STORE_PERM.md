# The sm100 K pool layout specification

This file specifies the CHANNEL ORDER in which an int2 or int4 K pool is written
for the sm100 fold decode kernel. It is not the documentation of a flag any more:
`k_f4_prepermuted` was removed from `IntDecodeSm100` (owner decision,
2026-08-12), so the layout is derived from the configuration and no caller can
select it. Whoever writes a K pool reads this file to learn which order to write,
and calls the resolver below rather than reproducing the predicate.

## Which configurations carry the permutation

`k_pool_is_permuted` is the SINGLE resolver, a classmethod on `IntDecodeSm100`. A
pool writer asks it through `_fold_harness.k_pool_perm`, and `PagedDecodeWrapper`
asks the same classmethod to check the order a caller declares.

The main kernel no longer consults it to PICK a deposit, as of 2026-08-13: its f4
deposit is unconditional two masks, so there is no branch for the two sides to
disagree about, which is the point of the change. The resolver still has two jobs,
not one: telling a pool writer which order to store, and letting
`PagedDecodeWrapper` check a caller's `k_pool_order` declaration before it launches.
The kernel object also keeps `self.k_pool_permuted`, which nothing in the kernel
reads any more; it is retained as the object's answer about its own pool contract
and is asserted by `tests/test_kperm_sm100_cpu.py`:

    IntDecodeSm100.k_pool_is_permuted(
        bits, *, dequant_int2_to_f4=None, k_quant=("token", 128),
        fold_scale_zero=True) -> bool

The predicate is: **permuted iff `bits == 4` (every configuration), or
`bits == 2` AND the resolved f4 feed is on.**

The int4 term has no feed condition, added 2026-08-15 with the raw-code
two-mask deposit: no flag resolves that deposit off (`fold_scale_zero=False` is
ctor-refused at this width), so the pool order is unconditional with it. The
int4 order is `k_store_perm_sm100_f4(4)`: the 8-slot shuffle
`_T8_I4 = (0, 4, 1, 5, 2, 6, 3, 7)` as position -> logical within each
8-channel group (one packed u32), so the stored nibble sequence is
[q0, q4, q1, q5, q2, q6, q3, q7] and the kernel's deposit is
`w & 0x0F0F0F0F` / `(w >> 4) & 0x0F0F0F0F` with no PRMT. UNLIKE int2's sigma
it is NOT an involution: a writer applies the perm, a reference inverts with
`_k_store_perm_inverse(perm, bits=4)` (its argsort), and reusing the forward
array to invert is silently wrong at six of every eight channels.
`check_k_store_perm_i4` is the enforcing oracle, and the V pool carries the
SAME shuffle along tokens within its dim-major rows
(`v_store_perm_sm100_f4(4)`, gate G1i4).

The int2 terms are both load-bearing, and the feed term is RESOLVED rather than
read, through `resolve_dequant_int2_to_f4`, because the f4 feed is the int2
default and an absent `dequant_int2_to_f4` therefore means on.

A `k_quant[0] == "token"` term stood here until 2026-08-12, so a writer who read
an earlier revision of this file would build a LOGICAL channel-wise K pool and be
silently wrong at every channel. It never described the hardware:
`check_k_store_perm_f4` shows the two deposits emit the same operand words, so the
permutation is absorbed between the pool and the operand stage and a channel-wise
fold sees the same operands either way. The term encoded a missing reference,
namely that `ref_channel_k_flex` unpacked K without inverting the order, and that
reference now inverts it. Channel-wise K is therefore permuted like everything
else on this feed, gated at all twelve channel-wise flex configs on B200 with a
worst relative error of 1.82e-3 against a 5e-3 bar.

| bits | resolved feed | K axis | stored K order |
|---|---|---|---|
| 2 | f4 (E2M1) | token or channel | **permuted**, `k_store_perm_sm100_f4(2)` (sigma) |
| 2 | e4m3 | token or channel | logical |
| 4 | raw-code (always) | token or channel | **permuted**, `k_store_perm_sm100_f4(4)` (t8) |

At int2 the f4 feed is permuted at either axis and the e4m3 feed is logical; the
int2 feed resolves OFF under the fold-off ablation arm (`fold_scale_zero=False`),
which therefore reads logical order at int2. Three other knobs resolved the feed
off in the same way until they were retired, `identity_dequant` on 2026-09-08 and
`operands_in_tmem` and `dump_stage` on 2026-09-14; unlike the `k_quant` term
discussed above, dropping them changes no answer this resolver gives, because no
configuration that still builds can set them. At int4 the width alone DOES decide the layout, as of
2026-08-15: there is no feed term and no knob that resolves the raw-code deposit
off, so every int4 pool is permuted.

A pool writer calls `_fold_harness.k_pool_perm(bits, kernel_kwargs)`, which
returns the permutation array or `None` by asking `k_pool_is_permuted` about the
same kwargs the kernel is built with.

## Nothing detects a pool written in the wrong order

**A permuted pool read by the gather, or a logical pool read by the two masks, is
wrong at every K channel and raises nothing.** Both orders are valid packings of
valid codes; row byte counts, pages, TMA boxes and block-table entries are
identical between them. There is no shape to check, no sentinel to compare and no
assertion that can fire. The fault surfaces only as attention output that is
wrong by an amount a reader will mistake for quantization noise.

This matters more here than it would elsewhere, because **no serving append path
emits this order.** Pool construction lives entirely in test and benchmark
fixtures, `repack_sm100.build_pools_flex` and `build_pools_lut` as reached
through `_fold_harness`, and those thread the order from the one resolver. An
external writer, a serving append among them, that writes int2 token-wise K
without consulting `k_pool_is_permuted` is silently wrong, and this specification
is the only thing standing between it and that outcome. Treat any new pool writer
as unvalidated until it has been paired against `check_k_store_perm_f4` below.

## The permutations

### Shared reduction-axis packing for K and V (2026-09-24)

The main SM100 kernel now uses one orientation rule at both widths: pack along
the tensor-core reduction dimension, with a shuffle confined to each 32-bit
word. K packs channels within token rows; V packs tokens within channel rows.
`_reduction_store_perm` defines the within-word order for both tensor helpers.

For page size 64 and head dimension 128, the physical shapes are:

| Tensor | Logical quantized codes before packing | INT2 uint8 pool | INT4 uint8 pool |
|---|---|---|---|
| K | `[NP, HKV, 64, 128]` | `[NP, HKV, 64, 32]` | `[NP, HKV, 64, 64]` |
| V (transposed) | `[NP, HKV, 128, 64]` | `[NP, HKV, 128, 16]` | `[NP, HKV, 128, 32]` |

Both V operands are K-major (token-contiguous) at the tensor core. INT2 expands
to packed E2M1 panels; INT4 expands to E4M3 bytes. These operand images and their
shared-memory stores remain format-specific. The change adds no runtime page
transpose, no extra code traffic, and no alternate-layout configuration knob.
It does not change Hopper or the INT2/E4M3 ablation.

INT4's V deposit partition is the same at every head tile: stream thread
`t < 64` owns the 16-byte code chunks `c = t + 64*r`, `r=0..3`, each at
`(channel=c//2, token_start=32*(c%2))`, and deposits them into the
64-byte-swizzled V operand stage. The pool layout does not depend on the
partition or on the operand swizzle. See `docs/kernel/int_fold_decode.md`,
"Shared-memory bank conflicts in the dequant streams".

**Compatibility:** INT4 V used to apply t8 to a token-major whole-page
permutation. Its shape and dtype have not changed, but its bytes have.
Rebuild previously packed INT4 pools with `_fold_harness.v_pool_perm` for the
same kernel configuration; do not reuse old caches. INT2 main pools and both K
orders are byte-for-byte unchanged. Pool dimensions cannot detect an old layout.

Validation entry points:

```sh
pytest tests/test_kperm_sm100_cpu.py tests/test_pool_call_sites_cpu.py
python scripts/probes/probe_sm100_e4m3a_subnormal.py --only VP
python baselines/accuracy_gate.py --bits 2,4 --out data/kmajor_accuracy.csv
```

The paired old/new latency comparison that validated this layout was measured by
`183286a:baselines/profile_b200_kmajor_v.py`, which has since been deleted.
[The validation record](SM100_KMAJOR_V_VALIDATION.md) holds its method, its
allocation controls and its results. Its INT2 runs are an unchanged-layout
control, not evidence of an INT2 optimization.

### Within-word orders

`k_store_perm_sm100_f4(bits)` returns one of two orders, keyed by width.

At int2 it is the SAME self-inverse 16-slot shuffle sigma that the V
order composes, applied along D within each 16-channel group:

    sigma(i) = (i & 3) | ((i & 4) << 1) | ((i & 8) >> 1)

extended over the head dimension as `16 * (d // 16) + sigma[d % 16]`, so channel
`d` is stored in the slot that index names.

At int4 it is the 8-slot shuffle t8 = (0, 4, 1, 5, 2, 6, 3, 7) as
position -> logical within each 8-channel group (one packed u32), extended as
`8 * (d // 8) + t8[d % 8]`. The V order composes the same t8 onto its
dim-major base (`v_store_perm_sm100_f4(4)`). UNLIKE sigma it is NOT
self-inverse: a writer applies t8 and a reference inverts with
`_k_store_perm_inverse(perm, bits=4)`; reusing the forward array to invert is
silently wrong at six of every eight channels, and
`test_int4_reference_inverts_the_perm` is the gate that catches it. The
derivation below is int2's; int4's is one line, stated at
`v_store_perm_sm100_f4`: the stored nibble sequence becomes
[q0, q4, q1, q5, q2, q6, q3, q7], so the even/odd nibble masks yield the two
operand words directly.

**Why it works.** `pack_rows_lut` puts code `i` of a u32 at
`b(i) = 16*(i//8) + 4*(i%4) + 2*((i%8)//4)`, a bijection onto the even bits
0..30, so the sixteen codes a word carries are interleaved. The b4x16_p64 K image
wants out0 nibble `j` = code `j` and out1 nibble `j` = code `8+j`. Reading
`w & 0x33333333` leaves each kept pair in place, so out0 nibble `j` sources bit
`4j` and out1 nibble `j` sources bit `4j+2`; sending logical channel `j` to the
slot with `b = 4j` and channel `8+j` to the slot with `b = 4j+2` is exactly
sigma.

**Why it is self-inverse.** Sigma keeps bits 0 and 1 of `i` and exchanges bits 2
and 3, and it never leaves the 16-channel group, so applying it twice exchanges
those two bits back and returns `i`. One array is therefore both the store order
and its own inverse, which is why a reference can undo the stored order by
indexing with the same permutation.

**Why it is the same permutation V already uses.** `v_store_perm_sm100_f4`
composes this sigma along its own axis for the same reason: the packer's
interleave has to cancel on the read, and the cancellation condition is a
property of `pack_rows_lut`, not of which tensor is being packed. One shuffle
serves both tensors, and a gate holds them equal group by group so the claim
cannot quietly stop being true.

K rows stay token-major and every row keeps its byte count, so pages, TMA boxes
and the block table are untouched. Only which channel each packed slot holds
changes.

## Why the order exists: the deposit asymmetry

At int2 with the f4 feed, the two deposits cost differently per stored element,
under the counting rule that charges one shift, one three-input logic operation,
or one byte permute as one:

| deposit | ops per 16 codes | per element |
|---|---|---|
| `map_int2_f4` (K, permuted, two masks) | 3 | 0.19 |
| `map_int2_f4v` (V, permuted, two masks) | 3 | 0.19 |
| the gather this replaced, kept for the arithmetic only | 7 | 0.44 |

The first two rows are now the SAME CODE: `map_int2_f4` and `map_int2_f4v` are
byte-for-byte identical, because K and V stopped being different code when they
stopped being different layouts. The third row is what the main kernel no longer
contains, and it is retained here because the paper's appendix cites the 0.44 as the
counterfactual cost of writing a pool unpermuted. No GPU kernel deposit contains that
form since the SM100 ping-pong fork was retired on 2026-10-07 (its last kernel copy is
`map_int2_f4` in `72a801c:src/cutedsl_decode/_sm100/int_fold_decode_pingpong.py`); the
CPU oracle in `scripts/probes/repack_sm100.py` still implements the two-part gather.

THE MAIN KERNEL NO LONGER HAS A GATHER ROW. `map_int2_f4` carried a `k_pre`
`const_expr` selecting between the two-mask read and a 7-op two-part gather until
2026-08-13, and the gather went dead the moment channel-wise K adopted the
permutation: that function is reached only under `const_expr(f4_feed)`, and the
predicate is true wherever it runs. It is deleted, so the main kernel's f4 K deposit
is two masks with no branch.

Two earlier revisions of this file were wrong about this row in opposite directions.
One said the gather is the live deposit for channel-wise K, which it stopped being
when the axis term went. The next said it is the live deposit for the int2 e4m3
feed, which is also wrong: that feed uses `map_int2_k`, a different function
entirely.

Where the permutation applies it removes **57% of the K construction ALU work**,
7 ops per 16 codes becoming 3.

## Channel-wise K, which carries the permutation too

Channel-wise K is permuted, like everything else on the f4 feed, and has been
since 2026-08-13. Because the two-mask read of a permuted pool reproduces the
logical-order panel image byte for byte, a channel-wise arm sees exactly the
operands it saw before, which is why widening the predicate needed no change to
the fold, the scales or the Q'-plane machinery.

What channel-wise K lacked, until 2026-08-13, was validation rather than
correctness. `test_fold_quant_flex_sm100.ref_channel_k_flex` is the
storage-matched reference for those cells, and it unpacked the K codes itself
without inverting the stored order, so a permuted channel-wise pool would have
been compared against the wrong numbers rather than caught. It now takes
`k_store_perm` and inverts it, and the twelve channel-wise cells are gated on
B200 at a worst relative error of 1.82e-3 against a 5e-3 bar.

TWO EARLIER REVISIONS OF THIS FILE GAVE A CORRECTNESS REASON INSTEAD, and it was
wrong: that a channel-wise K scale block is indexed by logical channel while the
permutation moves which channel each slot holds, so the fold would pair one
channel's parameters with another's codes. It would not. The deposit reproduces
the logical-order operand words byte for byte, which `check_k_store_perm_f4` now
proves exhaustively on a basis rather than by sampling, so the permutation is
absorbed entirely between the pool and the operand stage and everything
downstream, a channel-wise fold included, sees the operands it always saw. The
wrong reason is recorded here because it survived three reviews before codex
caught it, and because it is the kind of argument that sounds like a proof.

## Correctness

`repack_sm100.check_k_store_perm_f4` is the enforcing oracle, and it compares the
two paths' OPERAND WORDS: logical pool plus gather against permuted pool plus two
masks. They are byte-identical, which is why the two arms' kernel outputs are
**bitwise equal at every measured cell** rather than merely close. That is a
stronger statement than any relative-L2 bar. The oracle is host-side and needs
neither a GPU nor any kernel kwarg, so it survives the removal and is what a new
pool writer pairs itself against.

`tests/test_kperm_sm100_cpu.py` carries the CPU gates. **This file states the
PROPERTIES those gates hold and deliberately does not state how many tests the
module collects.** An earlier revision claimed 14 while the module collected 32,
and the number has now had to be corrected twice, because parametrization moves
it without changing a single guarantee. Run `pytest
tests/test_kperm_sm100_cpu.py --collect-only -q` if you want today's count.

The gated properties are:

* the operand-word oracle above, as the whole justification for the order;
* sigma is a permutation of the D channels and is self-inverse;
* its first sixteen values and a sha256 of the whole array are pinned, PER
  WIDTH, so an
  edit that perturbs either order trips a gate instead of silently moving
  every K operand;
* the int2 order is the same sigma the V order composes, group by group, and
  the int4 order is t8 tiled per 8-channel group;
* no code crosses its packed-word channel group (16 at int2, 8 at int4),
  which is the one wrong permutation that
  would otherwise look plausible, since crossing a group crosses a packed word
  and the mask read would gather foreign channels;
* the validator refuses non-permutations and group-crossing permutations,
  per width (an int2 sigma checked at bits=4 is refused because it crosses
  8-channel groups);
* both pool builders apply the order and both references invert it, and the
  int4 round-trip gate (`test_int4_reference_inverts_the_perm`) fails on a
  reference that reuses the forward t8 as its own inverse, which sigma's
  self-inverseness never could catch;
* the resolver answers True for int4 everywhere and for the int2 f4 feed at
  either K axis, and False elsewhere, checked knob by knob across the e4m3 feed
  and the fold-off arm;
* pool keying cannot disagree with the kernel: `k_pool_perm` returns an array
  exactly when `k_pool_is_permuted` returns True, over that same knob set;
* the shipped int2 configuration, built with no kwargs at all, is permuted.

## What the permutation is worth

Cutting operand-construction ALU work need not buy a proportional decode-latency
gain, because that work can be hidden behind what the other warpgroups are doing.
So price a deposit change with a paired A/B timing rather than by extrapolating
from operation counts, and publish that timing only through `baselines/bench_sweep.py`,
which times every case of one YAML in one process.

Full record: git show 3f7f180:docs/K_STORE_PERM.md (lines 267-309).

## Reproducing

The correctness statement reproduces on any machine, with no GPU:

    pytest tests/test_kperm_sm100_cpu.py

    PYTHONPATH=src python -c "import sys; sys.path.insert(0, 'scripts/probes'); \
        from repack_sm100 import check_k_store_perm_f4; print(check_k_store_perm_f4())"

A timing of this axis is one `baselines/bench_sweep.py` YAML, run once in each tree and
compared with `baselines/compare_bench_sweeps.py`. One process times every case of the
YAML. Where CUPTI loses a device record, `_bench_core` leaves that iteration out of the
median, and it refuses the capture when too few whole iterations remain.
