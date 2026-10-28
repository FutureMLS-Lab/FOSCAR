"""Regenerate the CPU-test fixtures of `scripts/_sass_census_sm90.py` from real tool output.

The fixtures in the parent directory are files a CUDA tool wrote, whole or cut to an excerpt
that keeps chosen line ranges of the tool's file in their original order and changes nothing
else. This script builds the four CUDA sources beside it with `nvcc -arch=sm_90a -cubin`,
runs `nvdisasm -c` and `cuobjdump -res-usage` on each build, reads the census's own
`i2_default` artifacts, runs `cuobjdump -res-usage` and `readelf -S -W` on
`tests/data/sm90_splitk_combine.cubin`, records both tools' `--version`, and writes every
fixture into `--out`. The builds and the whole tool outputs go to `--work`. It needs a CUDA
toolkit at `$CUDA_HOME` and GNU `readelf` on the PATH, and no GPU; the census run that writes
the `i2_default` artifacts does need one (README.md).

The line ranges of each excerpt were chosen on one listing, so the file an excerpt is cut
from must have the sha256 recorded with its ranges, and the script refuses a file that does
not. A listing of another build needs its ranges chosen again, and the tests that read the
excerpt edited with them.

Usage:
    CUDA_HOME=<toolkit> python3 make_fixtures.py --decode <save>.artifacts/i2_default \\
        --work <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMBINE_CUBIN = HERE.parents[1] / "sm90_splitk_combine.cubin"
CUDA_SOURCES = ("regions", "spill", "big", "call")

#: Each excerpt maps to the file in --work it is cut from, that file's sha256 and the 1-based
#: inclusive line ranges it keeps.
EXCERPTS = {
    "decode_i2.excerpt.sass": (
        "decode_i2.sass",
        "206a371ee13a6b2d6f151864bc189a2f15533dedd0c7b84c8bd1137cc4d50d23",
        [(1, 14), (125, 131), (290, 296), (636, 666), (775, 790), (846, 919), (1185, 1192),
         (1505, 1515), (1533, 1545), (2411, 2429)]),
    # The last line of the normalized PTX, a NUL, is left out.
    "decode_i2.excerpt.ptx.norm": (
        "decode_i2.ptx.norm",
        "4a933a774e06294c980f77a69aac3a4c9088e030bc29459e35c9722088af9fff",
        [(1, 31), (946, 1104), (2580, 2594)]),
    # These ranges keep the STL, LDL and LDL.LU of a spill before the first USETMAXREG and in
    # each of the three regions that follow it.
    "regions.excerpt.sass": (
        "regions.sass",
        "e8b59007b9ec23d11aca58f2c650ffccea016eb5b4936562559e5bab0ca3b7af",
        [(1, 37), (175, 180), (298, 301), (593, 601), (613, 614), (776, 780), (1224, 1230),
         (1238, 1241), (1300, 1306), (1876, 1896)]),
    # These ranges keep instruction addresses of four and of five hexadecimal digits.
    "big.excerpt.sass": (
        "big.sass",
        "d26c79b444201c02358ac439ac0755ab9b7c598e42ec0ee5447dac341b17f7b2",
        [(1, 14), (4105, 4115), (6040, 6045)]),
}
#: These fixtures are whole files of --work, copied under the same name.
WHOLE = ("decode_i2.res-usage.txt", "regions.res-usage.txt", "spill.sass",
         "spill.res-usage.txt", "call.sass", "combine.res-usage.txt", "combine.readelf.txt",
         "nvdisasm.version.txt", "cuobjdump.version.txt")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool_output(argv: list[str], dest: Path | None = None) -> bytes:
    """Run `argv` and return its standard output as the bytes it printed, written to `dest`
    unchanged when one is given. Exit on a nonzero status, and on an empty output when
    `dest` is given."""
    result = subprocess.run(argv, capture_output=True)
    if result.returncode != 0:
        sys.exit(f"{' '.join(argv)} exited {result.returncode}: "
                 f"{result.stderr.decode(errors='replace').strip()[:400]}")
    if dest is not None:
        if not result.stdout:
            sys.exit(f"{' '.join(argv)} printed nothing")
        dest.write_bytes(result.stdout)
    return result.stdout


def capture(decode: Path, work: Path) -> None:
    """Write every whole tool output the fixtures are made from into `work`."""
    if not os.environ.get("CUDA_HOME"):
        sys.exit("CUDA_HOME is unset; set it to the CUDA toolkit whose tools make the fixtures")
    bin_dir = Path(os.environ["CUDA_HOME"]) / "bin"
    nvcc, nvdisasm, cuobjdump = (str(bin_dir / name)
                                 for name in ("nvcc", "nvdisasm", "cuobjdump"))
    readelf = shutil.which("readelf")
    if readelf is None:
        sys.exit("no readelf on the PATH; combine.readelf.txt is GNU readelf output")
    for tool in (nvcc, nvdisasm, cuobjdump):
        if not Path(tool).is_file():
            sys.exit(f"no {tool}")
    for tool in (nvcc, readelf):
        version = tool_output([tool, "--version"]).decode(errors="replace").splitlines()
        print(f"{tool}: " + " / ".join(line for line in version if "release" in line
                                       or line.startswith("GNU readelf")))

    for name in CUDA_SOURCES:
        cubin = work / f"{name}.cubin"
        tool_output([nvcc, "-arch=sm_90a", "-cubin", "-o", str(cubin), str(HERE / f"{name}.cu")])
        tool_output([nvdisasm, "-c", str(cubin)], work / f"{name}.sass")
        tool_output([cuobjdump, "-res-usage", str(cubin)], work / f"{name}.res-usage.txt")

    # The listing the census read and the normalized PTX it hashed are copied as written from
    # its own `i2_default` artifacts, and the resource usage is that of the cubin it carved.
    norms = sorted((decode / "dsl_dump").glob("*.ptx.norm"))
    if len(norms) != 1:
        sys.exit(f"{decode / 'dsl_dump'} holds {len(norms)} *.ptx.norm files, not one")
    shutil.copyfile(decode / "i2_default.0.sass", work / "decode_i2.sass")
    shutil.copyfile(norms[0], work / "decode_i2.ptx.norm")
    tool_output([cuobjdump, "-res-usage", str(decode / "i2_default.0.cubin")],
                work / "decode_i2.res-usage.txt")

    tool_output([cuobjdump, "-res-usage", str(COMBINE_CUBIN)], work / "combine.res-usage.txt")
    tool_output([readelf, "-S", "-W", str(COMBINE_CUBIN)], work / "combine.readelf.txt")
    tool_output([nvdisasm, "--version"], work / "nvdisasm.version.txt")
    tool_output([cuobjdump, "--version"], work / "cuobjdump.version.txt")


def cut(work: Path, out: Path) -> None:
    """Write every fixture into `out` from the tool outputs in `work`."""
    for name, (source, digest, _) in EXCERPTS.items():
        found = sha256(work / source)
        if found != digest:
            sys.exit(f"{work / source} has sha256 {found}, and the line ranges of {name} were "
                     f"chosen on the file with sha256 {digest}")
    out.mkdir(parents=True, exist_ok=True)
    for name, (source, digest, ranges) in EXCERPTS.items():
        # The file is read and written as bytes, so that no newline or encoding translation
        # changes a line.
        lines = (work / source).read_bytes().split(b"\n")
        picked = [line for first, last in ranges for line in lines[first - 1:last]]
        (out / name).write_bytes(b"\n".join(picked) + b"\n")
        print(f"{name}: {len(picked)} lines of {source} (sha256 {digest}), lines {ranges}")
    for name in WHOLE:
        shutil.copyfile(work / name, out / name)
        print(f"{name}: all of it (sha256 {sha256(work / name)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--decode", required=True, type=Path,
                    help="the census's i2_default artifacts, <save>.artifacts/i2_default")
    ap.add_argument("--work", required=True, type=Path,
                    help="an empty or new directory for the builds and whole tool outputs")
    ap.add_argument("--out", required=True, type=Path, help="where the fixtures are written")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if any(args.work.iterdir()):
        sys.exit(f"--work {args.work} is not empty")
    capture(args.decode, args.work)
    cut(args.work, args.out)


if __name__ == "__main__":
    main()
