# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""Where a quantized value sits in the Blackwell code pools that `IntDecodeSm100` reads.

V is stored exactly like K, so both pools have one layout in global memory, one row per token
packing its D channels, `32 // bits` values to a little-endian 32-bit word:

    K  uint8 [NP, HKV, 64, D * bits / 8]
    V  uint8 [NP, HKV, 64, D * bits / 8]

The order inside a word is the production one, which the reader's predicates name
(`IntDecodeSm100.k_pool_is_permuted` and `v_pool_token_rows`) and the serving writer
`QuantAppendSm100` stores: the row packing composed with the within-word store shuffle, sigma
at int2 and t8 at int4. Composed, channel e of a row sits in word e // W at bit
4 (e % 8) + 2 ((e % 16) // 8) at int2 and 8 (e % 4) + 4 ((e % 8) // 4) at int4, W = 32 // bits.
`code_field` is that placement read backwards, one field at a time.

Torch-free and DSL-free, so the serving writer (`_sm100/quant_append.py`) traces both functions
with Int32 and Int64, and the CPU tests call them with integers. `QuantDequantSm100.__init__`
checks its byte-lane extraction against `code_field` for every channel of every unit, and
`code_row_index` widens the row index to 64 bits so a pool past the 32-bit range stays correct.
"""


def code_field(bits, word, field):
    """The row element whose quantized value field `field` of word `word` holds.

    Field f of a word is its bits [bits*f, bits*f + bits). At both widths field f holds element
    W * word + f // 2 + (W // 2) * (f % 2), W = 32 // bits: the even fields hold the word's first
    W / 2 elements in order and the odd fields its last W / 2. The element is a channel of the
    row's token, in a K row and in a V row alike."""
    if bits not in (2, 4):
        raise ValueError(f"code_field takes bits 2 or 4; got {bits}")
    per_word = 32 // bits
    return per_word * word + field // 2 + (per_word // 2) * (field % 2)


def code_row_index(rows_per_block, num_kv_heads, page, kv_head, row, *, wide):
    """Row index, in a whole C-contiguous pool [NP, HKV, rows_per_block, row bytes], of row `row`
    of the (page, kv_head) block: one of its 64 token rows, in K and in V. Multiplied by the row
    bytes it is the row's byte offset, which passes 2^31 at page 65,536 of an INT4 K pool with
    HKV = 8, so `wide` converts the page to a 64-bit integer BEFORE it is multiplied (Int64 in
    the writer's kernel, an int64 conversion in the host tests), and the product stays 64-bit."""
    return (wide(page) * num_kv_heads + kv_head) * rows_per_block + row
