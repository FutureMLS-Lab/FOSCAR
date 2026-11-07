"""The NVVM options `IntPingPongDecodeSm90` compiles with, and how they reach `cute.compile`.

`pingpong_runtime.nvvm_options` derives them from the constructed kernel,
`pingpong_runtime.dsl_takes_nvvm_options` decides from the installed DSL release whether they
are passed at all, and `pingpong_runtime.pingpong_compiler` is the one compile path of the
kernel. These tests pin the derivation, the release table (and its refusal of unlisted
releases), what the INSTALLED release must select, and the refusal of every keyword argument,
which `cute.compile` would let replace the derived options.
"""
import importlib.metadata
from types import SimpleNamespace

import pytest

runtime = pytest.importorskip(
    "foscar._sm90.pingpong_runtime", reason="imports torch")


def _kern(head_dim, fused_combine, bits=2, k=("token", 128), v=("token", 128)):
    return SimpleNamespace(head_dim=head_dim, fused_combine=fused_combine, bits=bits,
                           k_axis=k[0], k_group=k[1], v_axis=v[0], v_group=v[1], page_size=64)


@pytest.mark.parametrize("head_dim,fused,remat,mcse", [
    (128, False, "false", "false"), (128, True, "false", "false"),
    (256, False, "true", "false"), (256, True, "true", "true")])
def test_options_follow_the_kernel(head_dim, fused, remat, mcse):
    """Every option is stated, both ways: libNVVM keeps a value set by an earlier compile in
    the same process, so an option left out would be inherited."""
    assert runtime.nvvm_options(_kern(head_dim, fused, v=("token", head_dim),
                                      k=("token", head_dim))) == " ".join([
        f"-Xllc -enable-new-nvvm-remat={remat}", "-Xllc -cost-threshold-remat=6",
        f"-Xllc -rp-aware-mcse={mcse}", "-Xllc -disable-lsr=false"])


# (bits, K, V, pressure-blind CSE) at head_dim 128
_MCSE_CLASSES = [
    (4, ("token", 64), ("channel", 64), False),
    (4, ("token", 64), ("token", 128), False),
    (4, ("channel", 128), ("token", 128), False),
    (4, ("channel", 256), ("token", 128), False),
    (4, ("channel", 64), ("token", 128), True),
    (4, ("channel", 128), ("channel", 64), True),
    (4, ("token", 128), ("token", 128), True),
    (2, ("token", 64), ("channel", 64), True),
    (2, ("channel", 128), ("token", 128), True),
    (2, ("token", 128), ("token", 128), True),
]


@pytest.mark.parametrize("bits,k,v,off", _MCSE_CLASSES)
def test_cse_classes(bits, k, v, off):
    kern = _kern(128, False, bits, k, v)
    assert runtime.mcse_off(kern) is off
    assert f"-Xllc -rp-aware-mcse={'false' if off else 'true'}" in runtime.nvvm_options(kern)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("fused", [False, True])
def test_cse_at_head_dim_256_follows_the_fused_combine(bits, fused):
    """At head_dim 256, where the remat pass stays on, a fused-combine kernel keeps the
    pressure-aware CSE and any other kernel turns it off."""
    kern = _kern(256, fused, bits, ("token", 256), ("token", 256))
    assert runtime.mcse_off(kern) is (not fused)
    # at head_dim 128 the fused combine does not change it
    assert runtime.mcse_off(_kern(128, fused, bits)) is True


# (bits, K, V, strength reduction off) for kernels without the fused combine
_LSR_CLASSES = [
    (2, ("channel", 128), ("token", 128), True),
    (2, ("channel", 256), ("token", 128), True),
    (2, ("channel", 128), ("channel", 64), False),
    (2, ("channel", 256), ("channel", 128), False),
    (2, ("channel", 64), ("token", 128), False),
    (2, ("token", 128), ("token", 128), False),
    (2, ("token", 64), ("channel", 64), False),
    (4, ("channel", 128), ("channel", 64), True),
    (4, ("channel", 256), ("channel", 64), True),
    (4, ("channel", 128), ("token", 128), False),
    (4, ("channel", 256), ("channel", 128), False),
    (4, ("channel", 64), ("channel", 64), False),
    (4, ("token", 64), ("channel", 64), False),
    (4, ("token", 128), ("token", 128), False),
]


@pytest.mark.parametrize("bits,k,v,off", _LSR_CLASSES)
def test_strength_reduction_classes(bits, k, v, off):
    kern = _kern(128, False, bits, k, v)
    assert runtime.lsr_off(kern) is off
    assert runtime.nvvm_options(kern).endswith(f"-Xllc -disable-lsr={'true' if off else 'false'}")
    assert runtime.lsr_off(_kern(128, True, bits, k, v)) is False
    assert runtime.lsr_off(_kern(256, False, bits, k, v)) is False


@pytest.mark.parametrize("version,takes", [
    ("4.5.2", False), ("4.5.0", False), ("4.5", False), ("4.5+cu129", False),
    ("4.8.0", True), ("4.8.1", True), ("4.8", True), ("4.8+cu129", True), ("4.8.0.post1", True)])
def test_listed_releases(version, takes):
    assert runtime.dsl_takes_nvvm_options(version) is takes


@pytest.mark.parametrize("version,release", [
    ("4.8.0", (4, 8)), ("4.8", (4, 8)), ("4.8+cu129", (4, 8)), ("4.8.0.post1", (4, 8)),
    ("4.8rc1", (4, 8)), ("4.6.0.dev0", (4, 6)), ("4.10.2", (4, 10)), (" 4.5.2\n", (4, 5)),
    ("nightly", None), ("48", None), ("4.8oops", None), ("4.8.0+", None), ("V4.8.0", None),
    ("0!4.8.0", None), ("1!4.8.0", None), ("04.08.00", None), ("4.8a01", None),
    ("4.8+01", None), ("\u0664.\u0668.\u0660", None), ("4.8+cu129.0", (4, 8))])
def test_release_segment(version, release):
    assert runtime.dsl_release(version) == release


@pytest.mark.parametrize("version", ["4.6.0.dev0", "4.6.1", "4.7.0", "4.9.0", "5.0.0", "nightly"])
def test_unlisted_releases_are_refused(version):
    with pytest.raises(RuntimeError, match="the SM90 decode compiles under"):
        runtime.dsl_takes_nvvm_options(version)


class _RecordingCompile:
    """Stands in for `cute.compile`: records the bracket options and the call."""

    def __init__(self):
        self.bracket = None
        self.calls = []

    def __getitem__(self, bracket):
        self.bracket = bracket
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "compiled"


@pytest.fixture
def recording_compile(monkeypatch):
    cute = pytest.importorskip("cutlass.cute", reason="needs the DSL")
    recorder = _RecordingCompile()
    monkeypatch.setattr(cute, "compile", recorder)
    return recorder


@pytest.mark.parametrize("keyword", ["options", "pipeline", "no_cache", "anything"])
def test_every_keyword_is_refused(recording_compile, keyword):
    compile_fn, _ = runtime.pingpong_compiler(_kern(128, False), 3)
    with pytest.raises(TypeError, match=f"refused keyword\\(s\\): {keyword}"):
        compile_fn("kernel", 1, **{keyword: "x"})
    assert recording_compile.calls == []


def test_the_installed_release_selects_its_path(recording_compile):
    """Under the installed DSL, the compile carries exactly what its release must carry: the
    derived NVVM text as an option object under 4.8, the plain options string under 4.5."""
    version = importlib.metadata.version("nvidia-cutlass-dsl")
    major_minor = runtime.dsl_release(version)
    assert major_minor in runtime.DSL_TAKES_NVVM_OPTIONS, (
        f"the test environment's DSL {version} is one the SM90 decode refuses")
    kern = _kern(256, True)
    compile_fn, options = runtime.pingpong_compiler(kern, 3)
    assert compile_fn("kernel", 1) == "compiled"
    if major_minor == (4, 8):
        nvvm = runtime.nvvm_options(kern)
        assert options == f"--enable-tvm-ffi --opt-level 3 --nvvm-options '{nvvm}'"
        assert recording_compile.calls == [(("kernel", 1), {})]
        assert f'nvvm-options="{nvvm}"' in [o.serialize() for o in recording_compile.bracket]
    else:
        assert options == "--enable-tvm-ffi --opt-level 3"
        assert recording_compile.bracket is None
        assert recording_compile.calls == [(("kernel", 1), {"options": options})]


def test_an_unlisted_release_does_not_compile(recording_compile, monkeypatch):
    monkeypatch.setattr(runtime, "installed_dsl_version", lambda: "4.6.1")
    with pytest.raises(RuntimeError, match="4.6.1"):
        runtime.pingpong_compiler(_kern(128, False), 3)
    assert recording_compile.calls == []


def test_the_split_combine_compile_refuses_an_unlisted_release(monkeypatch):
    monkeypatch.setattr(runtime, "installed_dsl_version", lambda: "4.7.0")
    with pytest.raises(RuntimeError, match="4.7.0"):
        runtime.compile_splitk_combine(head_dim=128)


def _option_names(text):
    return {part.split("=")[0] for part in text.split() if part != "-Xllc"}


def _every_configuration_class():
    """One kernel per class `nvvm_options` can tell apart: both widths, both head dimensions,
    every K and V grouping the SM90 decode builds at each, with and without the fused combine."""
    for head_dim in (128, 256):
        ks = [("token", head_dim), ("token", head_dim // 2), ("channel", 64), ("channel", 128),
              ("channel", 256)]
        vs = [("token", head_dim), ("channel", 64), ("channel", 128), ("channel", 256)]
        for bits in (2, 4):
            for k in ks:
                for v in vs:
                    for fused in (False, True):
                        yield _kern(head_dim, fused, bits, k, v)


def test_other_kernels_restate_every_decode_option():
    """libNVVM keeps a value an earlier compile set, so a kernel other than the decode must state
    every option ANY decode compile sets, at a reset value. The names are collected over every
    configuration class, so an option added for one class only is caught."""
    names = set()
    for kern in _every_configuration_class():
        names |= _option_names(runtime.nvvm_options(kern))
    assert names == _option_names(runtime.NVVM_DEFAULTS)
    assert runtime.NVVM_DEFAULTS == (
        "-Xllc -enable-new-nvvm-remat=true -Xllc -cost-threshold-remat=10 "
        "-Xllc -rp-aware-mcse=true -Xllc -disable-lsr=false")


def test_other_kernels_compile_with_the_defaults_under_the_installed_release(recording_compile):
    major_minor = runtime.dsl_release(importlib.metadata.version("nvidia-cutlass-dsl"))
    compile_fn, options = runtime.sm90_compiler(3)
    assert compile_fn("kernel", 1) == "compiled"
    if major_minor == (4, 8):
        assert options == f"--enable-tvm-ffi --opt-level 3 --nvvm-options '{runtime.NVVM_DEFAULTS}'"
        assert recording_compile.calls == [(("kernel", 1), {})]
        assert (f'nvvm-options="{runtime.NVVM_DEFAULTS}"'
                in [o.serialize() for o in recording_compile.bracket])
    else:
        assert options == "--enable-tvm-ffi --opt-level 3"
        assert recording_compile.calls == [(("kernel", 1), {"options": options})]


@pytest.mark.parametrize("keyword", ["options", "pipeline"])
def test_other_kernels_refuse_keywords(recording_compile, keyword):
    compile_fn, _ = runtime.sm90_compiler(3)
    with pytest.raises(TypeError, match=f"refused keyword\\(s\\): {keyword}"):
        compile_fn("kernel", **{keyword: "x"})
    assert recording_compile.calls == []


def test_other_kernels_refuse_an_unlisted_release(recording_compile, monkeypatch):
    monkeypatch.setattr(runtime, "installed_dsl_version", lambda: "4.6.1")
    with pytest.raises(RuntimeError, match="4.6.1"):
        runtime.sm90_compiler(3)
