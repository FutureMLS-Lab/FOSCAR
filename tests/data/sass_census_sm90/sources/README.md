# Sources of the census test fixtures

`tests/test_sass_census_sm90_cpu.py` tests the parsers of `scripts/_sass_census_sm90.py` on the
files in the parent directory, which are real output of CUDA 12.9 tools. This directory holds
what regenerates them: four CUDA sources whose kernels were built only to produce those listings
and are never launched, and `make_fixtures.py`, which builds the four sources, runs the tools and
cuts the excerpts. The line ranges of every excerpt are in `make_fixtures.py`.

## How each fixture was made

The tools were those of `/usr/local/cuda-12.9` on the H100 nodes: `nvcc` V12.9.86, `nvdisasm`
V12.9.88 and `cuobjdump` V12.9.82, with GNU `readelf` 2.38. Every `nvcc` build passes
`-arch=sm_90a -cubin`. The first build of `regions.cu` also passed `-Xptxas -v`, which only makes
`ptxas` print its resource report, and the build that `make_fixtures.py` makes without it is
byte-identical to that first build. The census compiled the decode kernel with `nvidia-cutlass-dsl`
4.5.2 and its vendored CUDA 12.9, at `CUTE_DSL_ARCH=sm_90a`.

| Fixture | Tool output | Of | Captured |
|---|---|---|---|
| `decode_i2.excerpt.sass` | `nvdisasm -c`, an excerpt | the `i2_default` decode cubin that the census carved | research-secure-26, 2026-09-29 UTC |
| `decode_i2.excerpt.ptx.norm` | the census's normalized PTX, an excerpt | the same compile | research-secure-26, 2026-09-29 UTC |
| `decode_i2.res-usage.txt` | `cuobjdump -res-usage` | the same cubin | research-secure-26, 2026-09-29 UTC |
| `regions.excerpt.sass` | `nvdisasm -c`, an excerpt | `regions.cu` | research-secure-26, 2026-09-29 UTC |
| `regions.res-usage.txt` | `cuobjdump -res-usage` | `regions.cu` | research-secure-26, 2026-09-29 UTC |
| `spill.sass` | `nvdisasm -c` | `spill.cu` | research-secure-27, 2026-09-26 |
| `spill.res-usage.txt` | `cuobjdump -res-usage` | `spill.cu` | research-secure-27, 2026-09-26 |
| `call.sass` | `nvdisasm -c` | `call.cu` | research-secure-27, 2026-09-26 |
| `big.excerpt.sass` | `nvdisasm -c`, an excerpt | `big.cu` | research-secure-27, 2026-09-26 |
| `combine.res-usage.txt` | `cuobjdump -res-usage` | `tests/data/sm90_splitk_combine.cubin` | research-secure-27, 2026-09-26 |
| `combine.readelf.txt` | `readelf -S -W` | `tests/data/sm90_splitk_combine.cubin` | research-secure-27, 2026-09-26 |
| `nvdisasm.version.txt` | `nvdisasm --version` | | research-secure-26, 2026-09-29 UTC |
| `cuobjdump.version.txt` | `cuobjdump --version` | | research-secure-26, 2026-09-29 UTC |

The three `decode_i2` fixtures come from an exploration run on GPU 6 of research-secure-26, which
ran the fingerprint and the census on main `6bf0be2` with a first draft of the census scripts
that #109 added, before that pull request merged. The two `--version` outputs and
`decode_i2.res-usage.txt` were also captured on research-secure-27 on 2026-09-26, and those
captures are byte-identical to the ones committed.

## Regenerating

The census run needs an SM90 GPU and the environment that the docstring of
`scripts/_sass_census_sm90.py` gives. From the repository root:

```sh
# 1. The fingerprint and the census of the one configuration, on one GPU in one session.
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=<gpu> CUTE_DSL_ARCH=sm_90a CUDA_HOME=<cuda-12.9> \
  <python-with-cutlass-dsl> scripts/_ptx_fingerprint_sm90.py --out <dir>/fp.json --only i2_default
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=<gpu> CUTE_DSL_ARCH=sm_90a CUDA_HOME=<cuda-12.9> \
  <python-with-cutlass-dsl> scripts/_sass_census_sm90.py --save <dir>/census.json \
  --fingerprint <dir>/fp.json --only i2_default
# 2. The builds, the tool outputs and the fixtures. This step needs no GPU.
CUDA_HOME=<cuda-12.9> python3 tests/data/sass_census_sm90/sources/make_fixtures.py \
  --decode <dir>/census.json.artifacts/i2_default --work <dir>/work \
  --out tests/data/sass_census_sm90
git status --short tests/data/sass_census_sm90
```

With the toolchain and DSL above, and a decode kernel whose `i2_default` cubin is the one the
`decode_i2` fixtures were cut from, the last command prints nothing. The script refuses to cut
an excerpt from a file whose sha256 is not the one its line ranges were chosen on. So after a
change to the decode kernel's binary, the `decode_i2` ranges have to be chosen again on the new
listing, and the tests that read those excerpts edited with them.

These steps ran on research-secure-26 GPU 7 on 2026-09-30 at 20:06 UTC, at main `0dc8be4` with
this directory's sources as committed. The census ran twice and wrote the same `i2_default`
cubin both times, the one the `decode_i2` fixtures were cut from (sha256 `174d85f2...268c`).
`make_fixtures.py` ran three times: once on the artifacts of each census run, each time writing
all 13 fixtures byte-identical to the committed ones, and once in place as above, which left
`git status` empty. Each of the four `nvcc` builds was byte-identical to the build its fixtures
were first taken from. No step showed nondeterminism in these runs. A different toolkit, DSL,
`readelf` or decode kernel may produce different output. If the source file of an excerpt
changed, the script stops before it writes any fixture; otherwise the last command above shows
which whole-file fixtures changed.
