"""CPU-only regression tests for the SM100 quant-flex scale-pool dtype contract."""

import importlib.util
import math
import random
import struct
import sys
import types
from pathlib import Path

import pytest
import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src/fosca"
SM100_DIR = PACKAGE_DIR / "_sm100"
HOST_PACKAGE = "_fosca_sm100_host_test"
HOST_PARENT = "_fosca_host_test_parent"
#: Every synthetic sys.modules name this fixture installs, so it can undo them.
_INSTALLED: list[str] = []


def _load_fold_runtime():
    """Load the torch-only runtime helpers without importing CUDA/CuTe modules.

    `fold_runtime` reaches the shared scale-block layout as `..quant_layout`, so the
    synthetic package needs a PARENT for that relative import to resolve; without one
    it fails with "attempted relative import beyond top-level package". The parent is
    built here and the layout module preloaded into it, which keeps the fixture
    torch-only: `fosca/quant_layout.py` imports nothing at all.

    Every synthetic name is recorded in `_INSTALLED` so `_unload_fold_runtime` can take
    them back out; without that, whichever test ran first would fix the module identity
    for the rest of the session. This covers the names THIS fixture installs and no
    others.
    """
    module_name = f"{HOST_PARENT}.{HOST_PACKAGE}.fold_runtime"
    if module_name in sys.modules:
        return sys.modules[module_name]

    def _install(name, mod):
        _INSTALLED.append(name)
        sys.modules[name] = mod

    parent = types.ModuleType(HOST_PARENT)
    parent.__path__ = [str(PACKAGE_DIR)]
    _install(HOST_PARENT, parent)

    ql_name = f"{HOST_PARENT}.quant_layout"
    ql_spec = importlib.util.spec_from_file_location(
        ql_name, PACKAGE_DIR / "quant_layout.py")
    ql = importlib.util.module_from_spec(ql_spec)
    _install(ql_name, ql)
    assert ql_spec.loader is not None
    ql_spec.loader.exec_module(ql)

    package = types.ModuleType(f"{HOST_PARENT}.{HOST_PACKAGE}")
    package.__path__ = [str(SM100_DIR)]
    _install(f"{HOST_PARENT}.{HOST_PACKAGE}", package)

    spec = importlib.util.spec_from_file_location(
        module_name, SM100_DIR / "fold_runtime.py")
    module = importlib.util.module_from_spec(spec)
    _install(module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _unload_fold_runtime():
    """Remove every synthetic module this fixture installed."""
    for name in reversed(_INSTALLED):
        sys.modules.pop(name, None)
    _INSTALLED.clear()


@pytest.fixture
def fold_runtime():
    """The torch-only `fold_runtime`, with its synthetic package torn down after."""
    try:
        yield _load_fold_runtime()
    finally:
        _unload_fold_runtime()


def test_k_token64_defaults_to_bf16_and_accepts_explicit_f32(fold_runtime):
    runtime = fold_runtime
    kwargs = {
        "k_quant": ("token", 64),
        "v_quant": ("token", 128),
    }
    kq, vq, sdtype, offsets = runtime.quant_cfg_of(64, 128, kwargs)
    assert (kq, vq, sdtype) == (
        ("token", 64), ("token", 128), "bf16")
    assert offsets["ring_tot"] == 384
    assert runtime.scale_pool_dtype_of(64, 128, kwargs) is torch.bfloat16

    kwargs["sdtype"] = "f32"
    kq, vq, sdtype, offsets = runtime.quant_cfg_of(64, 128, kwargs)
    assert (kq, vq, sdtype) == (
        ("token", 64), ("token", 128), "f32")
    assert offsets["ring_tot"] == 384
    assert runtime.scale_pool_dtype_of(64, 128, kwargs) is torch.float32


@pytest.mark.parametrize("sdtype", ["f16", "float32", torch.float32])
def test_quant_cfg_rejects_unknown_scale_dtype(sdtype, fold_runtime):
    runtime = fold_runtime
    with pytest.raises(ValueError, match="sdtype must be"):
        runtime.quant_cfg_of(
            64,
            128,
            {
                "k_quant": ("token", 64),
                "v_quant": ("token", 128),
                "sdtype": sdtype,
            },
        )


def test_bf16_remains_illegal_outside_k_token64(fold_runtime):
    runtime = fold_runtime
    with pytest.raises(ValueError, match="only legal with K token-64"):
        runtime.quant_cfg_of(
            64,
            128,
            {
                "k_quant": ("channel", 128),
                "v_quant": ("token", 128),
                "sdtype": "bf16",
            },
        )


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _f32_of_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_pscale_lp_integer_formula_equals_the_float_formula(fold_runtime):
    """The writer, the Blackwell decode and `v_max_hist_of` bin a page maximum with
    `vmax_bucket`, and the decode turns the highest bin into `pscale_lp_of_bucket`; for
    every x >= 1e-30 that must equal floor(log2(448 / x)), main's host formula.

    Every binade that holds values >= 1e-30, at the mantissa fields where the floor
    can change (0x600000 is 1.75, where 448 / x crosses a power of two) and at
    seeded random fields in between.
    """
    runtime = fold_runtime
    assert runtime.PSCALE_VMAX_FLOOR_BITS == _f32_bits(1e-30)
    assert runtime.PSCALE_FLOOR_BUCKET == runtime.vmax_bucket(runtime.PSCALE_VMAX_FLOOR_BITS)
    floor_x = _f32_of_bits(runtime.PSCALE_VMAX_FLOOR_BITS)
    assert runtime.pscale_lp_of_bucket(runtime.PSCALE_FLOOR_BUCKET) == 108
    rng = random.Random(0)
    edges = (0, 1, 2, 0x5FFFFF, 0x600000, 0x600001, 0x600002, 0x7FFFFE, 0x7FFFFF)
    checked = 0
    for biased in range(1, 255):            # every normal binade, max finite included
        for mant in edges + tuple(rng.randrange(1 << 23) for _ in range(40)):
            bits = (biased << 23) | mant
            x = _f32_of_bits(bits)
            if x < floor_x:
                continue
            want = math.floor(math.log2(448.0 / max(x, 1e-30)))
            assert runtime.pscale_lp_of_bucket(runtime.vmax_bucket(bits)) == want, (hex(bits), x)
            checked += 1
    assert checked > 10_000


def test_vmax_bucket_lands_every_pattern_in_the_table(fold_runtime):
    """Every 32-bit pattern, sign-set, infinite and NaN included, lands in [0, VMAX_BINS),
    because the writer indexes the table with it; the sign bit does not move the bin; and the
    integer form, the clamped tensor form and the 32-bit signed arithmetic a kernel does agree.
    """
    runtime = fold_runtime
    patterns = [0, 0x80000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x0DA24260, 0x3F800000,
                0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0xFF800000,
                0xFFFFFFFF]
    rng = random.Random(1)
    patterns += [rng.getrandbits(32) for _ in range(2000)]
    for bits in patterns:
        b = runtime.vmax_bucket(bits)
        assert 0 <= b < runtime.VMAX_BINS, hex(bits)
        assert b == runtime.vmax_bucket(bits ^ 0x80000000), hex(bits)
        # what a kernel computes on a signed Int32: the masked pattern is non-negative and the
        # capped sum stays below 2^31
        assert min(bits & 0x7FFFFFFF, 0x7F800000) + 0x1FFFFF < 2 ** 31
    t = torch.tensor(patterns, dtype=torch.int64)
    got = runtime.vmax_bucket(t, torch.clamp_max)
    assert got.tolist() == [runtime.vmax_bucket(b) for b in patterns]


def _hist_by_hand(runtime, cells):
    counts = [0] * runtime.VMAX_BINS
    for bits in cells:
        if bits & 0x7FFFFFFF:
            counts[runtime.vmax_bucket(bits)] += 1
    return counts


def _cells(t):
    return [int(x) & 0xFFFFFFFF for x in t.view(torch.int32).reshape(-1).tolist()]


def test_v_max_hist_of_counts_every_cell_that_holds_a_maximum(fold_runtime):
    """`v_max_hist_of` is the per-bin count of the cells other than +0.0 and -0.0, written into
    a uint32 table, in place when given one."""
    runtime = fold_runtime
    rng = random.Random(2)
    raw = [0, 0x80000000, 0x0DA24260, 0x3F800000, 0x3F800000, 0x3FE00001, 0x7F7FFFFF,
           0x00000010] + [rng.getrandbits(31) for _ in range(500)]
    cells = torch.tensor([b - (1 << 32) if b >> 31 else b for b in raw],
                         dtype=torch.int32).view(torch.uint32).reshape(-1, 4)
    hist = runtime.v_max_hist_of(cells)
    assert hist.dtype == torch.uint32 and tuple(hist.shape) == (runtime.VMAX_BINS,)
    assert hist.view(torch.int32).tolist() == _hist_by_hand(runtime, raw)
    out = torch.full((runtime.VMAX_BINS,), 7, dtype=torch.int32).view(torch.uint32)
    assert runtime.v_max_hist_of(cells, out=out) is out
    assert out.view(torch.int32).tolist() == _hist_by_hand(runtime, raw)


def test_v_page_max_for_pool_is_each_pages_largest_stored_v_scale(fold_runtime):
    """Legacy and sectioned pools give the largest s_v of each (page, KV head) as
    uint32 bits, a bf16 pool's value widened first; channel-wise V gives None; and a
    page whose maximum is not positive (-0.0, a negative record) gives +0.0, the one
    "no maximum" pattern, so no adopted cell is sign-set."""
    runtime = fold_runtime
    gen = torch.Generator().manual_seed(0)
    legacy = torch.rand(3, 2, 64, 4, generator=gen) + 0.1
    got = runtime.v_page_max_for_pool(legacy)
    assert got.dtype == torch.uint32 and tuple(got.shape) == (3, 2)
    assert torch.equal(got.view(torch.float32), legacy[..., 2].amax(-1))

    signed = legacy.clone()
    signed[0, 0, :, 2] = -0.0
    signed[1, 1, :, 2] = -torch.rand(64, generator=gen) - 0.5
    got = runtime.v_page_max_for_pool(signed).view(torch.int32)
    assert got[0, 0].item() == 0 and got[1, 1].item() == 0
    assert (got >= 0).all()

    kw = {"k_quant": ("token", 64), "v_quant": ("token", 128)}
    _, _, _, off = runtime.quant_cfg_of(64, 128, kw)
    flex = (torch.rand(3, 2, off["block_tot"], generator=gen) + 0.1).to(torch.bfloat16)
    got = runtime.v_page_max_for_pool(flex, kw["k_quant"], kw["v_quant"])
    ring = flex[..., off["v_ring"]:off["v_ring"] + 128].float().reshape(3, 2, 64, 2)
    assert torch.equal(got.view(torch.float32), ring[..., 0].amax(-1))
    with pytest.raises(ValueError, match="k_quant/v_quant"):
        runtime.v_page_max_for_pool(flex)

    kw = {"k_quant": ("token", 128), "v_quant": ("channel", 128)}
    _, _, _, off = runtime.quant_cfg_of(64, 128, kw)
    chan = torch.rand(3, 2, off["block_tot"], generator=gen)
    assert runtime.v_page_max_for_pool(chan, kw["k_quant"], kw["v_quant"]) is None


def test_reads_v_max_hist_is_the_fold_with_token_wise_v(fold_runtime):
    """The table is a kernel input exactly where the P operand carries s_v."""
    runtime = fold_runtime
    assert runtime.reads_v_max_hist(64, 128) is True
    assert runtime.reads_v_max_hist(64, 128, {"fold_scale_zero": 0}) is False
    assert runtime.reads_v_max_hist(64, 128, {"v_quant": ("channel", 128)}) is False
    assert runtime.reads_v_max_hist(
        64, 128, {"k_quant": ("channel", 64), "v_quant": ("token", 128)}) is True
    legacy = torch.rand(3, 2, 64, 4) + 0.1
    assert runtime.v_max_hist_for_pool(legacy, {"fold_scale_zero": 0}) is None
    assert torch.equal(runtime.v_max_hist_for_pool(legacy),
                       runtime.v_max_hist_of(runtime.v_page_max_for_pool(legacy)))


def _hw_umax(a, b):
    """What `atom.max.u32` stores: the larger of two 32-bit words read as unsigned."""
    return a if (a & 0xFFFFFFFF) > (b & 0xFFFFFFFF) else b


def _settle_moves(runtime, old, best, exch):
    """`_append_core._settle_v_max`, transcribed: the (bin, delta) adds one settle issues, in
    the kernel's program order, from the word the atomic returned and the word it wrote."""
    new = best
    if not exch and old > best:            # Uint32 compare in the kernel
        new = old
    b_old, b_new = runtime.vmax_bucket(old), runtime.vmax_bucket(new)
    counts_old, counts_new = (old & 0x7FFFFFFF) != 0, (new & 0x7FFFFFFF) != 0
    if counts_old and counts_new and b_old == b_new:
        return []
    return ([(b_new, +1)] if counts_new else []) + ([(b_old, -1)] if counts_old else [])


#: Cell words outside what `bind`, `_adopt` and the writer produce, which a table recounted over
#: an engine's raw copy could still meet: -0.0, sign-set values, +inf, NaN patterns.
RAW_WORDS = (0x80000000, 0xBF800000, 0xC0A00000, 0x7F800000, 0x7FC00000, 0x7FFFFFFF,
             0xFFFFFFFF, 0x3F800000, 0x00000001)


def test_writer_protocol_keeps_the_table_the_histogram_of_the_cells(fold_runtime):
    """A model of the cache writer's protocol (`_append_core`, "V_PAGE_MAX AND V_MAX_HIST").

    Each update of a cell is one atomic read-modify-write that returns the old word: an exchange
    for a run at position 0, an unsigned max (`_hw_umax`, the hardware's rule, kept apart from
    the settle's own recomputation) otherwise. Its issuer later settles it with
    `_settle_v_max`'s rule (`_settle_moves`), whose adds are separate events: they land in any
    order relative to every other atomic and add, the issuer's own next update included (wider
    than the kernel, which settles each update within its run, so the table is checked under
    every order the hardware could give the adds), and only a settle's own +1 stays before its
    -1. The cells start from what `bind` makes (zeros), from what `_adopt` makes
    (`v_page_max_for_pool` of a pool with a -0.0 page and a negative page), or from raw words
    (`RAW_WORDS`, sign-set and non-finite ones included, with the table recounted over them).
    Random cells, values, duplicates and interleavings: once every update has settled, the table
    equals `v_max_hist_of` of the cells, word for word. On the states the writer and `_adopt`
    produce, the decode's exponent is then main's.
    """
    runtime = fold_runtime
    rng = random.Random(3)
    values = [_f32_bits(x) for x in (1e-9, 3e-6, 0.01, 0.0123, 0.5, 1.0, 1.74, 1.76, 7.0,
                                     2.0 ** 20, 2.0 ** 64 / 3)]
    for trial in range(600):
        ncell = rng.randint(1, 5)
        if trial % 3 == 0:
            cells = [0] * ncell
        elif trial % 3 == 1:
            # an adopted pool: one page of -0.0, one negative, the rest positive
            pool = torch.rand(ncell, 1, 64, 4, generator=torch.Generator().manual_seed(trial))
            pool[0, 0, :, 2] = -0.0
            if ncell > 1:
                pool[1, 0, :, 2] = -pool[1, 0, :, 2] - 0.1
            cells = _cells(runtime.v_page_max_for_pool(pool))
        else:
            cells = [rng.choice(RAW_WORDS + tuple(values)) for _ in range(ncell)]
        hist = runtime.v_max_hist_of(
            torch.tensor([c - (1 << 32) if c >> 31 else c for c in cells],
                         dtype=torch.int32).view(torch.uint32)).view(torch.int32).tolist()
        # The updates, as (cell, exchange?, best); a duplicate is two updates of one cell.
        ops = [(rng.randrange(ncell), rng.random() < 0.4, rng.choice(values))
               for _ in range(rng.randint(1, 12))]
        issuer = [rng.randrange(3) for _ in ops]
        # Event streams: the atomics of one issuer in program order; each settle's adds after
        # its own atomic, the +1 before the -1, interleaved with everything else.
        rmw_queue = {w: [i for i in range(len(ops)) if issuer[i] == w] for w in range(3)}
        pending_adds = {}
        returned = {}
        while any(rmw_queue.values()) or any(pending_adds.values()):
            choices = [("rmw", w) for w, q in rmw_queue.items() if q]
            choices += [("add", i) for i, adds in pending_adds.items() if adds]
            kind, key = rng.choice(choices)
            if kind == "rmw":
                i = rmw_queue[key].pop(0)
                c, exch, best = ops[i]
                old = cells[c]
                cells[c] = best if exch else _hw_umax(old, best)
                returned[i] = old
                pending_adds[i] = _settle_moves(runtime, old, best, exch)
            else:
                b, delta = pending_adds[key].pop(0)
                hist[b] = (hist[b] + delta) % (1 << 32)
        want = _hist_by_hand(runtime, cells)
        assert [h % (1 << 32) for h in hist] == want, (trial, cells, ops)
        # and on writer and adopted states the decode's exponent is main's: 135 - max(top bin,
        # floor bin) against floor(log2(448 / max(max s_v, 1e-30)))
        if trial % 3 != 2:
            top = max([b for b, h in enumerate(want) if h] + [runtime.PSCALE_FLOOR_BUCKET])
            live = [_f32_of_bits(c) for c in cells if c & 0x7FFFFFFF]
            sv_max = max(live) if live else 0.0
            assert runtime.pscale_lp_of_bucket(top) == math.floor(
                math.log2(448.0 / max(sv_max, 1e-30)))


def test_a_settle_never_leaves_its_cell_uncounted(fold_runtime):
    """`_settle_v_max` adds to the new bin before it subtracts from the old one, so after each
    of its adds the moved cell is still counted in some bin; the final histogram cannot tell the
    two orders apart, so this checks the transient states of one bin-changing update."""
    runtime = fold_runtime
    for old_x, new_x, exch in ((1.0, 300.0, False), (300.0, 0.01, True), (0.5, 0.9, True)):
        old, best = _f32_bits(old_x), _f32_bits(new_x)
        hist = [0] * runtime.VMAX_BINS
        hist[runtime.vmax_bucket(old)] = 1
        moves = _settle_moves(runtime, old, best, exch)
        assert moves, (old_x, new_x)
        for b, delta in moves:
            hist[b] += delta
            assert sum(hist) >= 1, (old_x, new_x, moves)
        assert hist[runtime.vmax_bucket(best)] == 1 and sum(hist) == 1


def _launch_runs(rng, npage, values, duplicates):
    """The runs of one Blackwell writer launch, as (page, exchange?, best). Each page the launch
    writes gets one of: a run from position 0 shorter than the page, then runs that continue it;
    a run that fills the page from position 0; or runs that continue an earlier use. With
    `duplicates`, a run that fills its page may share it with other runs, and a short
    position-0 run may be repeated. A short position-0 run's best scans the launch, so it is
    the largest s_v of every token of its page here; any other run's is its own tokens'."""
    runs = []
    for page in rng.sample(range(npage), rng.randint(1, npage)):
        kind = rng.choice(["start", "fill", "continue"])
        if kind == "fill":
            runs.append([page, True, False, [rng.choice(values) for _ in range(3)]])
            if duplicates and rng.random() < 0.7:
                runs += [[page, rng.random() < 0.3, True, [rng.choice(values)]]
                         for _ in range(rng.randint(1, 2))]
            continue
        for i in range(rng.randint(1, 3)):
            runs.append([page, kind == "start" and i == 0, True,
                         [rng.choice(values) for _ in range(rng.randint(1, 2))]])
        if duplicates and kind == "start" and rng.random() < 0.5:
            runs.append([page, True, True, [rng.choice(values)]])
    rng.shuffle(runs)
    out = []
    for page, exch, scans, tokens in runs:
        best = max(tokens)
        if exch and scans:
            best = max(max(r[3]) for r in runs if r[0] == page)
        out.append((page, exch, best, max(tokens)))
    return out


def test_skip_test_keeps_the_table_over_launches(fold_runtime):
    """A model of the Blackwell writer's page-maximum protocol over a sequence of launches
    (`_append_core`, "THE SKIP TEST AND THE SETTLE").

    Each CTA takes its runs in program order. A run that continues its page copies the cell
    early, as any value the cell has held so far in this launch (the copy may be stale), and
    skips its update when best <= that copy as unsigned words. Every other update is an atomic
    that returns the old word, and its settle adds land in any order later, interleaved with
    every other CTA. After every launch the table equals the histogram of the cells, with or
    without duplicate slots and from raw cell words. Without duplicates, each cell also equals
    what the protocol without the skip test leaves (every run's atomic applied, from the same
    starting cells, zeroed or raw), and from zeroed cells it is the largest s_v of its page's
    current use, as the round-trip gate checks on the device.
    """
    runtime = fold_runtime
    rng = random.Random(11)
    # pairs in one bin (0.01 and 0.0123, 0.5 and 0.6, 6.0 and 7.0) and pairs either side of a
    # bin edge (1.74 and 1.76)
    values = [_f32_bits(x) for x in (1e-9, 3e-6, 0.01, 0.0123, 0.5, 0.6, 1.0, 1.74, 1.76, 6.0,
                                     7.0, 2.0 ** 20)]
    skipped = 0
    for trial in range(1500):
        duplicates = trial % 2 == 1
        raw = trial % 4 >= 2
        npage = rng.randint(1, 4)
        cells = ([rng.choice(RAW_WORDS + tuple(values)) for _ in range(npage)] if raw
                 else [0] * npage)
        hist = _hist_by_hand(runtime, cells)
        reference = list(cells)
        truth = [None] * npage
        for _launch in range(rng.randint(1, 5)):
            runs = _launch_runs(rng, npage, values, duplicates)
            ctas = rng.randint(1, 3)
            owner = [rng.randrange(ctas) for _ in runs]
            todo = {c: [i for i in range(len(runs)) if owner[i] == c] for c in range(ctas)}
            history = [[c] for c in cells]
            copy, adds = {}, []
            while any(todo.values()) or adds:
                choices = [("run", c) for c, q in todo.items() if q]
                choices += [("add", None)] if adds else []
                kind, c = rng.choice(choices)
                if kind == "add":
                    b, delta = adds.pop(rng.randrange(len(adds)))
                    hist[b] = (hist[b] + delta) % (1 << 32)
                    continue
                i = todo[c][0]
                page, exch, best, _ = runs[i]
                if not exch and i not in copy:
                    copy[i] = rng.choice(history[page])
                    continue
                todo[c].pop(0)
                if not exch and best <= copy[i]:
                    skipped += 1
                    continue
                old = cells[page]
                cells[page] = best if exch else _hw_umax(old, best)
                if cells[page] != old:
                    history[page].append(cells[page])
                adds.extend(_settle_moves(runtime, old, best, exch))
            want = _hist_by_hand(runtime, cells)
            assert [h % (1 << 32) for h in hist] == want, (trial, cells, runs)
            for page, exch, best, _ in runs:
                reference[page] = best if exch else _hw_umax(reference[page], best)
            if not duplicates:
                assert cells == reference, (trial, cells, reference, runs)
            else:
                reference = list(cells)
            for page in {r[0] for r in runs}:
                mine = [r for r in runs if r[0] == page]
                top = max(r[3] for r in mine)
                if any(r[1] for r in mine):
                    truth[page] = top
                elif truth[page] is not None:
                    truth[page] = max(truth[page], top)
            if not duplicates and not raw:
                for page in range(npage):
                    if truth[page] is not None:
                        assert cells[page] == truth[page], (trial, page, cells, truth, runs)
    assert skipped > 1000
