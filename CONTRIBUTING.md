# Contributing

## Setup

FOSCAR is pure Python and JIT-compiles its kernels, so a source checkout is all
you need:

```sh
git clone https://github.com/FutureMLS-Lab/FOSCAR.git
cd foscar
python -m venv .venv && . .venv/bin/activate
pip install -e ".[cu13,test]"     # cu12 on a CUDA 12 driver
```

`cu12` and `cu13` select the CuTe DSL backend and are mutually exclusive.
Blackwell requires `cu13`.

## Running the tests

```sh
pytest -q
```

Most modules need no device and run anywhere; the ones that do skip when there is
none. `pytest -m gpu` selects only the device tests.

The `_cpu` suffix on a test module means it must pass with no GPU. A module
without the suffix needs hardware, and is where a change to either kernel or to a
pool builder is gated.

## What a change must satisfy

- **The suite is green.** Every change, including documentation-only ones, runs
  `pytest -q` on a machine with a supported GPU.
- **Kernel changes are measured.** A change to either decode kernel needs a
  timing comparison on the affected architecture, and a change to a split
  planner needs its self-test re-pinned.
- **Comments state what the code does and why an invariant holds.** Present
  tense, no dates, no commit hashes, no pointers to private measurement records.
- **No dead code.** A module, function or flag that nothing reaches is removed.

## Style

`ruff check` and `ruff format --check` at the configured line length. One source
line per sentence in prose and docstrings.

## Pull requests

Open a pull request against `main` with a description of what changed and what
was measured. Keep unrelated changes in separate pull requests.
