# Vendored dependencies

A minimal subset of two upstream projects, both BSD-3-Clause. Only the
transitive import closure of the SM90 kernel is included, and the package
`__init__.py` files are deliberately empty because the upstream ones import each
project's entire kernel surface.

- `quack/` — Dao-AILab QuACK (CuTe DSL kernels), BSD-3-Clause.
  Modules: activation, copy_utils, cute_dsl_utils, layout_utils, rounding,
  sm90_utils, sm100_utils, utils.
  https://github.com/Dao-AILab/quack
- `flash_attn/cute/` — Dao-AILab flash-attention (cute submodule), BSD-3-Clause.
  Modules: ampere_helpers, blackwell_helpers, copy_utils, mma_sm100_desc,
  pipeline, seqlen_info, softmax, utils.
  https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute

All internal imports are rewritten to the `fosca._vendor` package path.
Do not edit these modules except to refresh them from upstream. See
[LICENSE](LICENSE) for the license text and the `NOTICE` file at the repository
root for the attribution and the list of changes.
