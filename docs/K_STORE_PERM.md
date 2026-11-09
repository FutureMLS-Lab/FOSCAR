# The SM100 K and V pool store order

The Blackwell decode reads its K and V pools in a fixed order within each
32-bit word. This page states that order for anyone who fills a pool's bytes
without `QuantKVCache.append`: an engine that swaps a cache in, converts one from
another format, or copies pages between devices. The cache API is described in
[Writing the cache](../README.md#writing-the-cache); this page is the statement
of the order itself. Hopper stores its pools in another order.

## Pools

Each pool is uint8 and indexed by physical page and KV head. Both have one
shape, one token per row, packing that token's channels:

| Pool | Shape at `head_dim` 128 |
|---|---|
| K | `[num_pages, num_kv_heads, 64, 128 * bits / 8]` |
| V | `[num_pages, num_kv_heads, 64, 128 * bits / 8]` |

At `head_dim` 128 a row is 32 bytes at 2 bits and 64 bytes at 4 bits, so every
row is a multiple of 16 bytes, which the reader requires.
`layout.pool_shapes(num_pages)` returns both shapes for a layout and
`layout.pool_dtypes()` their dtype.

V is stored exactly like K, so identical K and V values give identical pool
bytes, and the reader takes its V operand from the same image with rows and
columns exchanged. `layout.v_pool_permuted` records this, and it is true in every
layout `QuantKVLayout.create()` builds. The order concerns the two value pools
only; the scale pool is unaffected by it.

## Where a value sits

A 32-bit word holds `32 / bits` values: 16 at 2 bits and 8 at 4 bits. Write
`W = 32 / bits` and `H = 16 / bits`. Field `f` of word `w`, which occupies bits
`[bits * f, bits * f + bits)`, holds the row's channel

    channel = W * w + f // 2 + H * (f % 2)

The even fields therefore hold the word's first `W / 2` channels in order and
the odd fields its last `W / 2`. The channels of one word, in field order:

- 2 bits: `0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15`
- 4 bits: `0, 4, 1, 5, 2, 6, 3, 7`

A writer needs the inverse, the bit a channel occupies:

```python
def stored_bit(bits, channel):
    per_word = 32 // bits          # 16 at 2 bits, 8 at 4 bits
    half = per_word // 2
    within = channel % per_word
    field = 2 * (within % half) + within // half
    return (channel // per_word) * 32 + bits * field


def pack_row(values, bits):
    row = 0
    for channel, q in enumerate(values):
        row |= q << stored_bit(bits, channel)
    return row


assert pack_row([i % 4 for i in range(16)], 2) == 0xFA50FA50
assert pack_row(list(range(8)), 4) == 0x73625140
```

## Why the bytes cannot reveal the order

A permuted row and a logical row have the same dtype, shape and byte count, and
every field holds a value in range either way. There is no sentinel and no spare
bit, so no assertion can tell the two apart.

Reading one as the other is wrong at 14 of every 16 channels at 2 bits and 6 of
every 8 at 4 bits; the remaining two are the group's first and last channel,
which the order leaves in place. Every value stays finite and in range, so the
output stays plausible and the error reads as quantization noise. A writer that
guesses wrong is unvalidated until its bytes are decoded through the cache's own
reader.

## The layout carries the orders

`QuantKVLayout.create(bits=, head_dim=, num_kv_heads=, k_quant=, v_quant=,
scale_dtype=, device=)` resolves both orders against the reader of the device
the cache will live on, and records them as `layout.k_pool_permuted` and
`layout.v_pool_permuted`. Both are true in every layout `create()` builds.

`layout.allocate(num_pages)` or `layout.bind(buffer, num_pages)` then makes a
cache of that layout, and `cache.append`, `cache.append_quantized` and
`PagedDecodeWrapper.run(q, cache)` take the order from it. No argument selects
an order and none declares one: the `k_pool_order` argument of `run()` is
retired and raises. A layout is a frozen value that derives both orders when it
is built, so a hand-built one whose fields disagree with the reader's answers is
refused rather than used.

## Filling the bytes yourself

A bound cache's buffer is region-major: all K pages, then all V pages, then the
scale pool, then the per-page V-scale maxima. `cache.k_codes` and
`cache.v_codes` are views of the first two regions in the shapes above, so a
writer can fill them in place. Keep each pool C-contiguous and 16-byte aligned,
and after a copy into the buffer that bypasses `append`, call
`cache.recount_v_max()` before the next decode of that layer.

To check the order, read the bytes back with `cache.dequantize(...)` and compare
against the values you packed. It decodes through the reader, so a wrong order
shows up as channels taken from the wrong slot of their word.
