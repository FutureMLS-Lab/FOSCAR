# Contributing

## Setup

The package is pure Python and JIT-compiles its kernels, so a source checkout is
all you need:

```sh
git clone https://github.com/donglinz/fosca.git
cd fosca
python -m venv .venv && . .venv/bin/activate
pip install -e ".[cu13,test]"     # cu12 on a CUDA 12 driver
```

`cu12` and `cu13` select the CuTe DSL backend and are mutually exclusive.
Blackwell requires `cu13`.

## Running the tests

```sh
pytest -q                    # CPU suite: no GPU and no nvidia-cutlass-dsl needed
pytest -q -m gpu             # GPU tests, on an SM90 or SM100 device
```

The `_cpu` suffix on a test module means it must pass with no GPU. A module
without the suffix needs hardware.

## What a change must satisfy

- **CPU suite green.** Every change, including documentation-only ones, runs the
  CPU suite.
- **Kernel changes are measured.** A change to either decode kernel needs a
  timing comparison on the affected architecture, and a change to a split
  planner needs its self-test re-pinned rather than loosened.
- **Comments state what the code does and why an invariant holds.** Present
  tense, no dates, no commit hashes, no pointers to private measurement records.
- **No dead code.** A module, function or flag that nothing reaches is removed
  rather than kept for later.

## Style

`ruff check` and `ruff format --check` at the configured line length. One source
line per sentence in prose and docstrings.

## Pull requests

Open a pull request against `main` with a description of what changed and what
was measured. Keep unrelated changes in separate pull requests.
