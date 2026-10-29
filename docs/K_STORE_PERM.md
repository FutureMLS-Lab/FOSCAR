# The SM100 K and V pool store order

This page specifies the byte order in which the Blackwell (SM100) decode kernel
expects its quantized K and V pools to be written. It is for anyone who fills a
pool outside `fosca.pool`: a serving-cache append path, a converter from
another cache format, or a fixture that packs its own codes. The builder API is
summarized in [Building pools](../README.md#building-pools); this page is the
normative statement of the order itself. Hopper pools use a different layout and
are out of scope here.

## Pool shapes

Page size is 64 and the head dimension is 128. A pool is indexed by physical
page; the block table matters only when the kernel reads it.

| Tensor | Logical quantized values | INT2 pool (`uint8`) | INT4 pool (`uint8`) |
|---|---|---|---|
| K | `[NP, HKV, 64, 128]` | `[NP, HKV, 64, 32]` | `[NP, HKV, 64, 64]` |
| V | `[NP, HKV, 128, 64]` | `[NP, HKV, 128, 16]` | `[NP, HKV, 128, 32]` |

K rows are token-major, so one token's 128 channels occupy one row and a row is
`128 * bits / 8` bytes. V rows are channel-major, so one channel's 64 tokens
occupy one row and a row is `64 * bits / 8` bytes. Both are multiples of 16, and
the kernel requires contiguous pools with 16-byte-aligned code rows.

The scale pool is unaffected by the store order. At the legacy granularity, where
both K and V are token-wise with group 128, it is `f32 [NP, HKV, 64, 4]` holding
`(s_k, z_k, s_v, z_v)` per token. At any other granularity it is the sectioned
block `[NP, HKV, block_tot]` in the pool's storage dtype, whose section offsets
come from `fosca.quant_layout.quant_block_offsets`.

## The two store permutations

Both orders are within-word shuffles. A 32-bit word holds `32 / bits` codes, 16
at INT2 and 8 at INT4, and the shuffle moves codes only inside that word. No code
changes row, page or packed word, so pages, TMA boxes and block-table entries are
identical between a permuted pool and a logical one.

The order is an index array whose element `i` is the logical code that stored
slot `i` holds. One word's stored codes are therefore `q[perm[0]], q[perm[1]], …`
in slot order.

**INT2: the sigma shuffle.** `sigma(i) = (i & 3) | ((i & 4) << 1) | ((i & 8) >> 1)`,
applied within each 16-channel group:

    sigma = (0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15)

so the stored sequence is
`[q0, q1, q2, q3, q8, q9, q10, q11, q4, q5, q6, q7, q12, q13, q14, q15]`.

**INT4: the t8 shuffle.** `t8 = (0, 4, 1, 5, 2, 6, 3, 7)`, applied within each
8-channel group, so the stored sequence is
`[q0, q4, q1, q5, q2, q6, q3, q7]`.

Sigma is its own inverse. It keeps bits 0 and 1 of the index and exchanges bits 2
and 3, so applying it twice returns the index, and one array serves as both the
store order and the inverse. t8 is not its own inverse:
`t8[t8] = (0, 2, 4, 6, 1, 3, 5, 7)`. A reference that undoes an INT4 store order
must index with the argsort, and reusing the forward array reads six of every
eight channels from the wrong slot.

### Why the masks need these orders

`pack_rows` is the packer both pools go through. At INT4 it writes sequential
nibbles, so byte `i` holds code `2i` in the low nibble and code `2i + 1` in the
high nibble. At INT2 it puts code `i` of a 16-code group at bit
`b(i) = 16 * (i // 8) + 4 * (i % 4) + 2 * ((i % 8) // 4)`, a bijection onto the
even bits 0..30.

The kernel builds its tensor-core operand words with two masks per word:

- **INT4.** `w & 0x0F0F0F0F` yields codes 0..3 of the word and
  `(w >> 4) & 0x0F0F0F0F` yields codes 4..7, with no byte permute. That is
  exactly the sequence t8 stores.
- **INT2.** `w & 0x33333333` leaves one code per nibble at bit `4j`, and
  `(w >> 2) & 0x33333333` leaves the other at bit `4j + 2`. Requiring nibble `j`
  of the first word to be logical channel `j` and nibble `j` of the second to be
  channel `8 + j` fixes the slot at bit `4j` to hold channel `j` and the slot at
  bit `4j + 2` to hold channel `8 + j`, which is sigma.

The masks cost three ALU operations per word, one shift and two ands. The
alternative on a logical pool is a two-part gather at INT2, seven operations per
16 codes, and a mask-plus-two-byte-permutes compaction at INT4, five operations
per 8 codes. At INT2 the order removes 57% of the K operand-construction work.
Less ALU work does not translate proportionally into decode latency, because it
can hide behind what the other warpgroups are doing, so a layout change is priced
by timing rather than by the operation count.

## Which configurations carry which order

The order follows from the configuration. It is not a flag: there is no argument
that selects it, because a caller who could choose the order independently of the
kernel could only choose a wrong one.

| `bits` | resolved INT2 feed | K axis | K stored order | V stored order |
|---|---|---|---|---|
| 2 | E2M1, the default | token or channel | sigma | sigma along tokens, channel-major rows |
| 2 | E4M3 | token or channel | logical | `v_store_perm_sm100(2)`, token-major |
| 4 | raw code, always | token or channel | t8 | t8 along tokens, channel-major rows |

Three things about the table. INT4 has no feed term, because the raw-code
two-mask deposit is unconditional and `fold_scale_zero=False` is refused at that
width, so every INT4 pool carries t8. The INT2 feed is resolved rather than read:
an absent `dequant_int2_to_f4` means on, and both an explicit `0` and the
`fold_scale_zero=False` arm turn it off. The K axis is not a condition, because
the two-mask deposit reproduces the logical-order operand words for arbitrary
codes, which means a channel-wise fold sees the same operands as a token-wise
one.

The predicate is a classmethod on the kernel:

    IntDecodeSm100.k_pool_is_permuted(
        bits, *, dequant_int2_to_f4=None, k_quant=("token", 128),
        fold_scale_zero=True) -> bool

It answers `True` for INT4 at every configuration and for INT2 with the E2M1 feed
at either K axis, and `False` for the INT2 E4M3 feed. The kernel object stores the
same answer as `k_pool_permuted`.

## Resolver entry points

`fosca.pool` owns the orders and the builders that apply them.

| Entry point | Returns |
|---|---|
| `k_store_perm_sm100_f4(bits)` | `(128,)` int tensor, the K order for `bits` |
| `v_store_perm_sm100_f4(bits)` | `(8192,)` int tensor, the V order over the flat channel-major page, position `d * 64 + t` |
| `v_store_perm_sm100(bits)` | `(8192,)` int tensor, the token-major V order the INT2 E4M3 feed reads |
| `_k_store_perm_inverse(perm, bits=4)` | the argsort of a K order, validated first |
| `_check_k_store_perm(perm, bits=2)` | raises unless `perm` is a permutation of `range(128)` that stays inside each packed-word group |
| `_check_v_store_perm(perm)` | raises unless `perm` is a permutation of its own index range |
| `check_k_store_perm_f4()`, `check_k_store_perm_i4()` | the operand-word oracles described under Validation |

`build_pools`, `pack_pools` and `build_pools_flex` take the arrays as
`k_store_perm=` and `v_store_perm=` and apply them before packing, with `None`
meaning logical order. `dequant_pools` and `dequant_pools_flex` take the same
arrays to invert the order. `PagedDecodeWrapper.run(k_pool_order=...)` takes
`"permuted"` or `"logical"` where the kernel reads a permuted pool and accepts
`None` only where it reads logical order. The declaration is checked against the
planned configuration, not against the pool, so a declaration that contradicts
the bytes cannot be caught there.

A writer that packs codes itself, rather than calling a builder, applies the
arrays directly:

```python
import torch
from fosca import pool

NP, HKV, bits = 4, 2, 2
k_codes = torch.randint(0, 2**bits, (NP, HKV, 64, 128), dtype=torch.uint8)
v_codes = torch.randint(0, 2**bits, (NP, HKV, 64, 128), dtype=torch.uint8)

k_store = pool.k_store_perm_sm100_f4(bits)   # (128,): stored slot -> logical channel
v_store = pool.v_store_perm_sm100_f4(bits)   # (8192,): flat channel-major position

k_pool = pool.pack_rows(k_codes[..., k_store], bits)          # [NP, HKV, 64, 32]

v_t = v_codes.transpose(-1, -2).contiguous()                      # [NP, HKV, 128, 64]
lead = v_t.shape[:-2]
v_t = v_t.reshape(*lead, 128 * 64)[..., v_store].reshape(*lead, 128, 64)
v_pool = pool.pack_rows(v_t, bits)                            # [NP, HKV, 128, 16]

assert pool.check_k_store_perm_f4() is True
assert pool.check_k_store_perm_i4() is True
```

The V array indexes the flat channel-major view of one page, so it is applied to
`v_codes.transpose(-1, -2).reshape(..., 128 * 64)` and the result is reshaped back
to `[..., 128, 64]` before packing.

## Why the order cannot be detected from the pool bytes

Both orders are valid packings of valid integer values. A permuted pool and a
logical pool have the same dtype, the same shape, the same byte count, and the
same pages, TMA boxes and block-table entries. There is no sentinel, no spare bit
and no shape difference to check, so no assertion can fire.

Reading one as the other is wrong at every K channel, and it fails quietly: each
channel comes from the wrong slot inside its packed-word group, which yields
finite values of the right magnitude. The error appears only in attention output,
at a size a reader will attribute to quantization noise. A pool writer that does
not consult the predicate is unvalidated until it is paired against the oracles
below.

## Validation

`check_k_store_perm_f4` and `check_k_store_perm_i4` compare the two paths as
operand words: a logical pool read with the gather, against a permuted pool read
with the two masks. Both are host-side and need no GPU. The comparison is
exhaustive rather than sampled, because every output bit of both deposits depends
on exactly one input code, so agreement on a basis of one-hot inputs implies
agreement everywhere; random vectors are kept as a cross-check that the basis
argument was applied to the right functions. Byte-identical operand words are why
the two arms of a paired kernel run are bitwise equal rather than merely close.

The CPU gates in `tests/test_kperm_sm100_cpu.py` hold these properties:

- the operand-word identity above, at both widths;
- each order is a permutation of the head dimension that stays inside its
  packed-word group, 16 channels at INT2 and 8 at INT4, and its first values and
  full digest are pinned per width;
- sigma is self-inverse and t8 is not, so an INT4 reference that reuses the
  forward array is caught;
- the INT2 K order is the same sigma the V order composes, group by group, and
  the INT4 K order is t8 tiled per 8-channel group;
- the validators refuse non-permutations and group-crossing permutations, and
  refuse an INT2 order checked at INT4 width;
- both builders apply the order and both references invert it, at INT2 and at
  INT4;
- the predicate answers `True` for INT4 everywhere and for the INT2 E2M1 feed at
  either K axis, and `False` for the E4M3 feed;
- the shipped INT2 configuration, built with no arguments at all, is permuted.

`pytest -q` runs them with the rest of the CPU suite.
