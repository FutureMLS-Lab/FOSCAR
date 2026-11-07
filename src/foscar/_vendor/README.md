# Vendored dependencies

Minimal vendored subset (16 modules) of two upstream projects, snapshotted from a local
`third_party/` checkout (2026-06-10). Only the transitive import closure of the SM90 research
kernels is included; package `__init__.py` files are intentionally emptied (the upstream ones
import each project's entire kernel surface).

- `quack/` — from Dao-AILab "QuACK" (CuTe-DSL kernels), BSD-3-Clause.
  Modules: activation, copy_utils, cute_dsl_utils, layout_utils, rounding, sm90_utils, sm100_utils, utils.
  https://github.com/Dao-AILab/quack
- `flash_attn/cute/` — from Dao-AILab "flash-attention" (cute submodule), BSD-3-Clause.
  Modules: ampere_helpers, blackwell_helpers, copy_utils, mma_sm100_desc, pipeline, seqlen_info, softmax, utils.
  https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute

All internal imports are rewritten to `cutedsl_decode._vendor.*`. Do not edit these files except
to re-snapshot from upstream; local consumers import them via `from cutedsl_decode._vendor... import`.
