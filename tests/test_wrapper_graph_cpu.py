# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""Graph mode's host side (`PagedDecodeWrapper(..., use_cuda_graph=True)`), on the CPU.

The compiles, launches and the CUDA calls `plan()` and `update()` make are replaced by stand-ins
that record what they are given, so these tests hold what the host decides: which build a bucket
launches with which buffers, what `plan()`, `update()` and `run()` refuse, and that `update()`
reuses a pinned buffer only after that buffer's previous copy. What the kernels compute under a
captured graph is checked on a GPU, by running the captured launches against the eager path.
Building a layout imports the reader classes, so the DSL must be installed (no GPU is used).
"""
import contextlib
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("cutlass", reason="requires nvidia-cutlass-dsl")

from foscar import _utils, graph_plan, kv_cache
from foscar import wrapper as W
from foscar.kv_cache import QuantKVCache, QuantKVLayout

D, HQ, HKV, MAX_PAGES = 128, 32, 8, 64


class _Event:
    def __init__(self, log):
        self.log = log

    def record(self, stream):
        self.log.append(("record", id(self), stream))

    def synchronize(self):
        self.log.append(("sync", id(self)))


@pytest.fixture
def graph_env(monkeypatch):
    """`make(arch, batch, max_kv_len=4096, **plan_kwargs)` -> (wrapper, layout, cache, page
    table, seq_lens), planned in graph mode on the CPU at 132 SMs; `launches` records each
    launch, `log` the CUDA events and syncs, `stream` the current stream, `capturing` whether
    the stream is capturing."""
    from foscar._sm100 import fold_runtime
    from foscar._sm90 import pingpong_runtime

    launches, log, stream, capturing, arch_box = [], [], ["s0"], [False], ["sm90"]
    monkeypatch.setattr(_utils, "require_arch", lambda device=None: arch_box[0])
    monkeypatch.setattr(_utils, "require_dsl_target", lambda arch: None)
    monkeypatch.setattr(_utils, "on_device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(_utils, "cuda_index", lambda device=None: 0)
    monkeypatch.setattr(kv_cache, "_require_device", lambda layout, device: None)
    monkeypatch.setattr(W, "get_num_sm", lambda device: 132)
    monkeypatch.setattr(W, "_pinned_words",
                        lambda n: log.append("pinned") or torch.zeros(n, dtype=torch.int32))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: log.append("device-sync"))
    monkeypatch.setattr(torch.cuda, "Event", lambda: _Event(log))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream[0])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])

    def variant(stem, has_q_descale, has_lse):
        log.append("compile")
        return stem + ("_qd" if has_q_descale else "") + ("_lse" if has_lse else "")

    # eager's getters, with eager's own signatures, and graph mode's own
    def fold(*a, has_q_descale=False, has_lse=False):
        return (variant("fold", has_q_descale, has_lse), a), dict(fold_runtime.BITS_TUNING[a[5]])

    def fold_graph(*a):
        return ((variant("fold_graph", *a[7:]), a),
                dict(fold_runtime.BITS_TUNING[a[4]]))

    def pingpong(*a):
        log.append("compile")
        return ("pingpong", a)

    def pingpong_graph(*a):
        log.append("compile")
        return ("pingpong_graph", a)

    def combine(head_dim, out_dtype=torch.bfloat16, has_lse=False):
        log.append("compile")
        return ("combine", has_lse)

    def combine_graph(head_dim, out_dtype, has_lse):
        log.append("compile")
        return ("combine_graph", has_lse)

    monkeypatch.setattr(W, "_get_compiled_fold_kernel", fold)
    monkeypatch.setattr(W, "_get_compiled_fold_graph_kernel", fold_graph)
    monkeypatch.setattr(W, "_get_compiled_pingpong_kernel", pingpong)
    monkeypatch.setattr(W, "_get_compiled_pingpong_graph_kernel", pingpong_graph)
    monkeypatch.setattr(W, "_get_compiled_splitk_combine", combine)
    monkeypatch.setattr(W, "_get_compiled_splitk_combine_graph", combine_graph)
    monkeypatch.setattr(fold_runtime, "launch_fold",
                        lambda compiled, **kw: launches.append(("fold", compiled, kw)))
    monkeypatch.setattr(fold_runtime, "launch_fold_graph",
                        lambda compiled, **kw: launches.append(("fold_graph", compiled, kw)))
    monkeypatch.setattr(pingpong_runtime, "launch_pingpong",
                        lambda compiled, combine, **kw: launches.append(
                            ("pingpong", compiled, {**kw, "combine": combine})))
    monkeypatch.setattr(pingpong_runtime, "launch_pingpong_graph",
                        lambda compiled, combine, **kw: launches.append(
                            ("pingpong_graph", compiled, {**kw, "combine": combine})))

    def make(arch, batch, max_kv_len=4096, bits=2, **plan_kwargs):
        arch_box[0] = arch
        layout = QuantKVLayout._create(arch, bits=bits, head_dim=D, num_kv_heads=HKV)
        g = torch.Generator().manual_seed(0)
        shapes, dtypes = layout.pool_shapes(16), layout.pool_dtypes()
        cache = QuantKVCache._adopt(
            layout, torch.randint(0, 256, shapes["k_codes"], dtype=torch.uint8, generator=g),
            torch.randint(0, 256, shapes["v_codes"], dtype=torch.uint8, generator=g),
            torch.rand(shapes["scales"], generator=g).to(dtypes["scales"]))
        ws = torch.zeros(W.graph_workspace_bytes(batch, HQ, layout, max_kv_len=max_kv_len)
                         // 4 + 4, dtype=torch.float32)
        w = W.PagedDecodeWrapper(ws, use_cuda_graph=True)
        pt = torch.zeros(batch, MAX_PAGES, dtype=torch.int32)
        sl = torch.ones(batch, dtype=torch.int32)
        w.plan(page_table=pt, seq_lens=sl, num_qo_heads=HQ, layout=layout,
               max_kv_len=max_kv_len, **plan_kwargs)
        return w, layout, cache, pt, sl

    make.launches, make.log, make.stream, make.capturing = launches, log, stream, capturing
    return make


def _io(w, lse=False, q_descale=False):
    b = w._batch_size
    q = torch.zeros(b, HQ, D, dtype=w._q_data_type)
    out = torch.zeros(b, HQ, D, dtype=w._o_data_type)
    extra = {}
    if lse:
        extra["lse"] = torch.zeros(b, HQ, dtype=torch.float32)
    if q_descale:
        extra["q_descale"] = torch.ones(b, HQ, dtype=torch.float32)
    return q, out, extra


def test_hopper_split_bucket_launches_the_graph_build_over_its_work_list(graph_env):
    w, _, cache, pt, _ = graph_env("sm90", 2)
    g = w._graph
    assert g.bucket.split_capable and w._sm90_fused_combine
    assert g.decode == ("pingpong_graph", g.decode[1]) and g.combine is None
    assert graph_env.log[-1] == "device-sync"                 # set-up waits once
    # the one-chunk plan at length 1, before any update
    slots = g.work_views()[0].numpy()
    assert [tuple(r) for r in slots[:2 * HKV]] == [(i, 0, 1, 1) for i in range(2 * HKV)]
    assert (slots[2 * HKV:, 0] == graph_plan.IDLE_ITEM).all()
    q, out, _ = _io(w)
    assert w.run(q, cache, out=out) is out
    kind, compiled, kw = graph_env.launches[-1]
    assert kind == "pingpong_graph" and kw["block_table"] is pt
    slots, splits = g.work_views()
    assert kw["work"].data_ptr() == slots.data_ptr() == g.work.data_ptr()
    assert kw["splits"].data_ptr() == splits.data_ptr() and kw["lse_out"] is None
    assert kw["o_partial"].shape == (g.bucket.s_tight, 2, HQ, D)
    assert kw["o_partial"].data_ptr() == w._float_workspace_buffer.data_ptr()


def test_hopper_external_combine_bucket_and_its_lse_form(graph_env):
    w, _, cache, _, _ = graph_env("sm90", 4, lse=True)
    g = w._graph
    assert g.bucket.split_capable and not w._sm90_fused_combine
    assert g.combine == ("combine_graph", True)
    q, out, extra = _io(w, lse=True)
    w.run(q, cache, out=out, **extra)
    _, _, kw = graph_env.launches[-1]
    assert kw["combine"] == ("combine_graph", True) and kw["lse_out"] is extra["lse"]


def test_an_unsplit_bucket_runs_eagers_unsplit_build_on_the_engine_buffers(graph_env):
    w, _, cache, pt, sl = graph_env("sm90", 40, lse=True)       # 40 x 8 items fill 264 CTAs
    g = w._graph
    assert not g.bucket.split_capable and g.work is None and g.decode[0] == "pingpong"
    q, out, extra = _io(w, lse=True)
    w.run(q, cache, out=out, **extra)
    kind, _, kw = graph_env.launches[-1]
    assert kind == "pingpong" and kw["seq_lens"] is sl and kw["block_table"] is pt
    assert kw["outer_splits"] == 1 and kw["combine"] is None
    assert kw["lse"].data_ptr() == extra["lse"].data_ptr()
    w.update([7] * 40)                                         # checks the lengths, copies nothing
    assert not [e for e in graph_env.log if isinstance(e, tuple)]


@pytest.mark.parametrize("batch,split", [(2, True), (19, False)])
def test_blackwell_buckets(graph_env, batch, split):
    w, layout, cache, pt, sl = graph_env("sm100", batch, q_descale=True, lse=True)
    g = w._graph
    assert g.bucket.split_capable is split
    q, out, extra = _io(w, lse=True, q_descale=True)
    w.run(q, cache, out=out, **extra)
    kind, compiled, kw = graph_env.launches[-1]
    assert kw["page_table"].data_ptr() == pt.data_ptr()
    assert kw["q_descale"] is extra["q_descale"] and kw["seqlens"] is sl
    if split:
        # the graph split build, through its own launch
        assert kind == "fold_graph" and compiled[0] == "fold_graph_qd_lse"
        slots, splits = g.work_views()
        assert "table_offsets" not in kw and kw["work"].data_ptr() == slots.data_ptr()
        assert kw["split_counts"].data_ptr() == splits.data_ptr()
        assert kw["max_pages"] == MAX_PAGES and tuple(kw["work"].shape) == (g.bucket.wave, 4)
        assert kw["o_partial"].shape == (g.bucket.s_tight, batch, 1, HQ, D)
    else:
        # eager's unsplit build, planned by eager's own step, through eager's launch
        assert kind == "fold" and compiled[0] == "fold_qd_lse" and compiled[1][4] == "none"
        assert w._kv_splits == 1 and w._reduction == "none"
        assert kw["table_offsets"].tolist() == [b * MAX_PAGES for b in range(batch)]
        assert "work" not in kw and "o_partial" not in kw


def test_update_plans_into_the_next_pinned_buffer_and_waits_only_to_reuse_one(graph_env):
    w, _, _, _, _ = graph_env("sm90", 2, max_kv_len=4096)
    g = w._graph
    graph_env.log.clear()
    for step in range(6):
        w.update([4096, 1 + step])
    events = [e for e in graph_env.log if isinstance(e, tuple)]
    ids = [id(ev) for ev in g.events]
    # four records, then each further update waits on the buffer it reuses before recording
    assert [e[0] for e in events] == ["record"] * 4 + ["sync", "record"] * 2
    assert events[4][1] == ids[0] and events[6][1] == ids[1]
    assert "device-sync" not in graph_env.log
    k = g.bucket.chunks(np.array([4096, 6]))
    assert g.work_views()[1].tolist() == k.tolist()
    assert g.host_words[1][-4:].tolist()[:2] == k.tolist()


class _CopyLog:
    """The device work list, logging which pinned buffer each copy reads."""

    def __init__(self, graph, log):
        self.work, self.host, self.log = graph.work, graph.host, log

    def copy_(self, src, non_blocking=False):
        self.log.append(("copy", next(i for i, h in enumerate(self.host) if h is src)))
        self.work.copy_(src, non_blocking=non_blocking)


def test_each_update_waits_for_its_buffer_before_writing_it_and_copies_the_fresh_plan(
        graph_env, monkeypatch):
    """Per step, on the buffer it takes: wait for that buffer's last copy (from its first
    reuse), write the plan into it, copy it, record the copy's event; and the device list is
    what a fresh write of the step's plan gives, also when a reused buffer's k_b changed."""
    w, _, _, _, _ = graph_env("sm90", 2, max_kv_len=4096)
    g, log = w._graph, graph_env.log
    real_write, real_lengths = graph_plan.Bucket.write, graph_plan.Bucket.write_lengths

    def logged(real):
        def method(self, out, lengths, k):
            log.extend(("write", i) for i, h in enumerate(g.host_words) if h is out)
            return real(self, out, lengths, k)
        return method

    monkeypatch.setattr(graph_plan.Bucket, "write", logged(real_write))
    monkeypatch.setattr(graph_plan.Bucket, "write_lengths", logged(real_lengths))
    device = _CopyLog(g, log)
    g.work = device
    slot_of = {id(ev): i for i, ev in enumerate(g.events)}
    steps = [[4096, 1], [1, 1], [64, 64], [4096, 4096],      # the four buffers' first writes
             [1, 4096], [1, 2], [64, 64], [2048, 2048], [4000, 3]]
    last_k, changed, same = {}, 0, 0
    for i, lengths in enumerate(steps):
        log.clear()
        w.update(lengths)
        slot = i % W._GRAPH_HOST_BUFFERS
        seen = [(e[0], slot_of[e[1]]) if e[0] in ("sync", "record") else e
                for e in log if isinstance(e, tuple)]
        assert seen == ([("sync", slot)] if i >= W._GRAPH_HOST_BUFFERS else []) + \
            [("write", slot), ("copy", slot), ("record", slot)], (i, seen)
        k = g.bucket.chunks(np.array(lengths))
        if slot in last_k:
            changed += not np.array_equal(last_k[slot], k)
            same += np.array_equal(last_k[slot], k)
        last_k[slot] = k
        fresh = np.full_like(g.host_words[0], 7)
        real_write(g.bucket, fresh, np.array(lengths, dtype=np.int64), k)
        assert device.work.numpy().tolist() == fresh.tolist(), i
    assert changed and same            # both kinds of reuse were exercised


def test_update_refuses(graph_env):
    w, _, _, _, _ = graph_env("sm90", 2, max_kv_len=4096)
    for bad, match in (([1.0, 2.0], "integer lengths"), ([True, True], "integer lengths"),
                       (torch.ones(2, dtype=torch.float32), "integer lengths"),
                       (torch.ones(2, dtype=torch.int32, device="meta"), "on the host"),
                       ([5, 5, 5], r"\[2\]"), ([0, 5], r"\[1, max_kv_len"),
                       ([4097, 5], r"\[1, max_kv_len")):
        with pytest.raises(ValueError, match=match):
            w.update(bad)
    w.update(np.array([5, 6], dtype=np.int32))
    w.update(torch.tensor([5, 6]))
    graph_env.stream[0] = "s1"
    with pytest.raises(RuntimeError, match="stream"):
        w.update([5, 6])
    graph_env.capturing[0] = True
    with pytest.raises(RuntimeError, match="capture"):
        w.update([5, 6])


@pytest.mark.parametrize("arch,batch", [("sm90", 40), ("sm100", 19)])
def test_an_unsplit_bucket_refuses_another_stream_too(graph_env, arch, batch):
    w, _, _, _, _ = graph_env(arch, batch)
    assert not w._graph.bucket.split_capable
    w.update([5] * batch)
    graph_env.stream[0] = "s1"
    with pytest.raises(RuntimeError, match="stream"):
        w.update([5] * batch)


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_plan_under_capture_is_refused_before_any_set_up(graph_env, arch):
    graph_env(arch, 2)                         # installs the stand-ins for this architecture
    graph_env.log.clear()
    graph_env.capturing[0] = True
    layout = QuantKVLayout._create(arch, bits=2, head_dim=D, num_kv_heads=HKV)
    w = W.PagedDecodeWrapper(torch.zeros(1 << 20, dtype=torch.float32), use_cuda_graph=True)
    with pytest.raises(RuntimeError, match="before capture"):
        w.plan(page_table=torch.zeros(2, MAX_PAGES, dtype=torch.int32),
               seq_lens=torch.ones(2, dtype=torch.int32), num_qo_heads=HQ, layout=layout,
               max_kv_len=4096)
    # nothing compiled, no pinned buffer, no wait for the device
    assert graph_env.log == []
    assert w._graph is None and not w._planned


def test_plan_is_one_shot_and_a_second_plan_leaves_the_first_intact(graph_env):
    w, layout, cache, pt, sl = graph_env("sm90", 2)
    g = w._graph
    with pytest.raises(RuntimeError, match="plans once"):
        w.plan(page_table=pt, seq_lens=sl, num_qo_heads=HQ, layout=layout, max_kv_len=4096)
    assert w._graph is g and w._planned
    q, out, _ = _io(w)
    w.run(q, cache, out=out)


def test_run_refuses_what_a_capture_cannot_take(graph_env):
    w, _, cache, _, _ = graph_env("sm90", 2, lse=True)
    q, out, extra = _io(w, lse=True)
    with pytest.raises(ValueError, match="needs out="):
        w.run(q, cache, **extra)
    with pytest.raises(ValueError, match="planned with lse=True"):
        w.run(q, cache, out=out)
    n = len(graph_env.launches)
    w.run(q, cache, out=out, **extra)
    assert len(graph_env.launches) == n + 1


@pytest.mark.parametrize("batch", [2, 19])
@pytest.mark.parametrize("planned", [True, False])
def test_run_refuses_a_q_descale_the_plan_did_not_declare_and_a_missing_one(graph_env, batch,
                                                                            planned):
    w, _, cache, _, _ = graph_env("sm100", batch, q_descale=planned)
    q, out, extra = _io(w, q_descale=True)
    with pytest.raises(ValueError, match=f"planned with lse=False and q_descale={planned}"):
        w.run(q, cache, out=out, **({} if planned else extra))
    n = len(graph_env.launches)
    w.run(q, cache, out=out, **(extra if planned else {}))
    assert len(graph_env.launches) == n + 1


class _Shape(torch.Tensor):
    """A small int32 page table that reports another shape, for the bounds plan() checks before it
    reads a page."""
    reported = None

    @property
    def shape(self):
        return torch.Size(type(self).reported)


@pytest.mark.parametrize("rows,cols,max_kv_len,match", [
    (2, 2**30, 4096, "exceed int32 offsets"),                       # 2^31 entries
    (1, 2**25, 2**31 - 63, r"at most 2\^31 - 64"),                  # its page round-up overflows
])
def test_graph_plan_refuses_what_int32_cannot_hold(graph_env, monkeypatch, rows, cols,
                                                    max_kv_len, match):
    graph_env("sm90", 2)
    layout = QuantKVLayout._create("sm90", bits=2, head_dim=D, num_kv_heads=HKV)
    monkeypatch.setattr(_Shape, "reported", (rows, cols))
    table = torch.zeros(rows, 4, dtype=torch.int32).as_subclass(_Shape)
    w = W.PagedDecodeWrapper(torch.zeros(1 << 16, dtype=torch.float32), use_cuda_graph=True)
    with pytest.raises(ValueError, match=match):
        w.plan(page_table=table, seq_lens=torch.ones(rows, dtype=torch.int32), num_qo_heads=HQ,
               layout=layout, max_kv_len=max_kv_len)
    assert w._graph is None and not w._planned


def test_blackwells_graph_tile_is_the_kernels_resolved_tile(graph_env, monkeypatch):
    from foscar._sm100 import fold_runtime

    graph_env("sm100", 2)
    layout = QuantKVLayout._create("sm100", bits=2, head_dim=D, num_kv_heads=HKV)
    for tile in (128, 256):
        monkeypatch.setattr(fold_runtime, "resolve_fold_tiling",
                            lambda k_quant, v_quant, tile=tile, **kw: (tile, 2))
        assert W._graph_tile("sm100", layout) == tile
        assert W._graph_bucket("sm100", layout, 2, 4096, 132).tile == tile


@pytest.mark.parametrize("batch,fused", [(2, True), (4, False)])
def test_hoppers_graph_decode_stores_the_lse_only_with_the_fused_combine(graph_env, batch,
                                                                        fused):
    w, _, _, _, _ = graph_env("sm90", batch, lse=True)
    g = w._graph
    assert w._sm90_fused_combine is fused and g.decode[0] == "pingpong_graph"
    assert g.decode[1][-1] is fused                    # has_lse: the decode's own LSE store
    assert (g.combine == ("combine_graph", True)) is not fused


@pytest.mark.parametrize("kwargs,error,match", [
    ({"indptr": torch.zeros(3, dtype=torch.int32)}, TypeError, "page_table=, not indptr"),
    ({"kv_splits": 2}, TypeError, "kv_splits"),
    ({"reduction": "kernel"}, TypeError, "reduction"),
    ({"lse": 1}, TypeError, "bool"),
    ({"q_descale": True}, ValueError, "bf16"),
    ({"max_kv_len": 64 * MAX_PAGES + 1}, ValueError, "max_kv_len"),
    ({"max_kv_len": None}, ValueError, "max_kv_len"),
    ({"page_table": torch.zeros(2, MAX_PAGES, dtype=torch.int64)}, ValueError, "int32"),
    ({"page_table": torch.zeros(MAX_PAGES, 2, dtype=torch.int32).t()}, ValueError,
     "contiguous"),
    # a column slice of a wider table: its row stride is not max_pages
    ({"page_table": torch.zeros(2, 2 * MAX_PAGES, dtype=torch.int32)[:, :MAX_PAGES]},
     ValueError, "contiguous"),
    ({"seq_lens": torch.ones(4, dtype=torch.int32)[::2]}, ValueError, "contiguous"),
    ({"seq_lens": torch.ones(3, dtype=torch.int32)}, ValueError, "one length per row"),
    ({"bits": 2}, TypeError, "'bits'"),
])
def test_graph_plan_refuses(graph_env, kwargs, error, match):
    graph_env("sm90", 2)                       # installs the stand-ins
    layout = QuantKVLayout._create("sm90", bits=2, head_dim=D, num_kv_heads=HKV)
    w = W.PagedDecodeWrapper(torch.zeros(1 << 16, dtype=torch.float32), use_cuda_graph=True)
    args = dict(page_table=torch.zeros(2, MAX_PAGES, dtype=torch.int32),
                seq_lens=torch.ones(2, dtype=torch.int32), num_qo_heads=HQ, layout=layout,
                max_kv_len=4096)
    args.update(kwargs)
    with pytest.raises(error, match=match):
        w.plan(**args)
    assert w._graph is None and not w._planned


def test_eager_plan_refuses_graph_arguments_and_update(graph_env):
    graph_env("sm90", 2)
    w = W.PagedDecodeWrapper(torch.zeros(1 << 16, dtype=torch.float32))
    layout = QuantKVLayout._create("sm90", bits=2, head_dim=D, num_kv_heads=HKV)
    args = (torch.tensor([0, 1, 2], dtype=torch.int32), torch.arange(2, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32), HQ, layout)
    # eager's plan takes no graph-mode argument: one is an unexpected keyword there
    assert type(w) is W.PagedDecodeWrapper
    for kw in ({"page_table": torch.zeros(2, 4, dtype=torch.int32)}, {"lse": True},
               {"q_descale": True}):
        with pytest.raises(TypeError, match="unexpected keyword"):
            w.plan(*args, max_kv_len=64, **kw)
    with pytest.raises(RuntimeError, match="update\\(\\) is graph mode's"):
        w.update([1, 1])
    assert type(W.PagedDecodeWrapper(torch.zeros(4), True)) is W._GraphDecodeWrapper


def test_a_failed_graph_plan_leaves_nothing_runnable(graph_env, monkeypatch):
    graph_env("sm90", 2)
    layout = QuantKVLayout._create("sm90", bits=2, head_dim=D, num_kv_heads=HKV)
    w = W.PagedDecodeWrapper(torch.zeros(4, dtype=torch.float32), use_cuda_graph=True)
    with pytest.raises(RuntimeError, match="graph_workspace_bytes"):
        w.plan(page_table=torch.zeros(2, MAX_PAGES, dtype=torch.int32),
               seq_lens=torch.ones(2, dtype=torch.int32), num_qo_heads=HQ, layout=layout,
               max_kv_len=4096)
    assert w._graph is None and not w._planned and w._layout is None


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_the_workspace_covers_every_bucket_up_to_the_largest(graph_env, arch):
    graph_env(arch, 2)                         # installs the stand-ins for this architecture
    layout = QuantKVLayout._create(arch, bits=2, head_dim=D, num_kv_heads=HKV)
    # at 300 Hopper's unsplit scratch, which grows with the batch, is the largest
    for max_batch in (64, 300):
        total = W.graph_workspace_bytes(max_batch, HQ, layout, max_kv_len=32768)
        for batch in range(1, max_batch + 1):
            bucket = W._graph_bucket(arch, layout, batch, 32768, 132)
            assert 4 * W._graph_scratch_words(arch, bucket, HQ, D) <= total, (max_batch, batch)
        assert W.graph_workspace_bytes(8, HQ, layout, max_kv_len=32768) <= total


@pytest.mark.parametrize("arch,heads,head_tiles", [
    ("sm90", 128, 2), ("sm100", 128, 2), ("sm100", 8, 1), ("sm100", 16, 1)
])
def test_graph_supports_all_eager_head_shapes(graph_env, monkeypatch, arch, heads, head_tiles):
    monkeypatch.setattr(sys.modules[__name__], "HQ", heads)
    w, _, cache, _, _ = graph_env(arch, 1)
    assert w._graph.bucket.head_tiles == head_tiles
    q, out, _ = _io(w)
    assert w.run(q, cache, out=out) is out


def test_blackwell_d256_is_supported_in_graphs(graph_env, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "D", 256)
    w, layout, cache, _, _ = graph_env("sm100", 2)
    assert layout.head_dim == 256 and w._planned
    q, out, _ = _io(w)
    assert w.run(q, cache, out=out) is out
