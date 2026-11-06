# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""Which SM100 unsplit builds the wrapper compiles without the item loop.

The rule and its measurements are `wrapper.py`'s ONE WORK ITEM PER CTA. It lives here, on
resolved scalars and with no imports, so that the rule can be evaluated without torch or the
DSL, from the scalars a caller already holds.
"""


def reads_exponent_table(v_axis, fold_scale_zero):
    """Whether a build reads `v_max_hist`: the fold with token-wise V. Channel-wise V and the
    fold-off ablation take a constant exponent as a launch scalar instead. The flag is read
    as an integer only for token-wise V, so channel-wise V never inspects it."""
    return v_axis == "token" and bool(int(fold_scale_zero))


def drops_item_loop(reduction, reads_table, names_persistent, items, cap):
    """Whether the wrapper compiles an SM100 decode without the item loop: an unsplit plan
    (`reduction == "none"`), a build that reads the exponent table, no `persistent` among
    the caller's kernel arguments, and no more (request, KV head) `items` than the CTA
    `cap`, so that no CTA gets a second item."""
    return reduction == "none" and reads_table and not names_persistent and items <= cap
