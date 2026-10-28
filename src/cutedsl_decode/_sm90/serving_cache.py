"""ServingCache — the batch-free GLOBAL paged quantized-KV pool that the SM90 decode kernel reads.

This is the host fixture (used by the SM90 numerics gates, benches and harnesses) that builds the
serving-readiness contract from per-request inputs: a single pooled set of fragment-order code
pages + ONE interleaved paged scale pool, addressed through a global block table.  Design spec:
docs/superpowers/specs/2026-06-16-int-decode-serving-readiness-design.md.

CONTRACT (the four tensors the kernel reads):
  codes_k / codes_v   uint8     [NP, HKV, <block>]     # NP = total physical pages; page = one
                                                       #   N(=64)-token tile.  NO leading batch
                                                       #   dim — one GLOBAL pool.  <block> is one
                                                       #   (page, kv-head) block of plane-word
                                                       #   A-frag bytes in the READER's order
                                                       #   (CODE POOL ORDERS below).
  scales              <sdtype>  [NP, HKV, N, 4]        # last dim = (s_k, z_k, s_v, z_v); ONE
                                                       #   interleaved pool, PAGED under the SAME
                                                       #   block table.  sdtype in {f32, bf16}.
                                                       #   HKV-major-then-token (NOT [NP,N,HKV,4]):
                                                       #   the per-(hk,page) N*4 block is then fully
                                                       #   contiguous, so the bf16 TMA box (256
                                                       #   elems = 512B) clears the 16B inner-box
                                                       #   minimum (an 8B [N,4] minor box on the
                                                       #   token-major order is an illegal TMA).
  block_table         int32     [B, maxPages]          # logical (request, tile) -> GLOBAL page id.
  seq_lens            int32     [B]                     # per-request kv length (ragged).

The page id `block_table[b, gt]` is a free coordinate on the GLOBAL page (NP) dim for BOTH codes
and scales — two requests whose block tables point at the same page read identical bytes (the
genuine-pooling property the old B-replicated scheme could not express).

CODE POOL GEOMETRY.  Everything about where a code byte sits is computed here and nowhere else:
`code_row_bytes` gives the `per` of the K and V rows, `code_row_chunks` their 16-byte chunk count
C, `code_block_shape` / `code_pool_shape` the block and pool shape in either order, and
`code_block_word_strides` / `code_chunk_offset` where row word w, or chunk c of thread t, sits in
a block, and `code_word_offset` where row word w sits in a whole pool, with the page widened to
64 bits before it is multiplied.  The pingpong kernel, IntAppend, the pool writers (`repack_rs.py`, the adapter) and
this module's own checks call these, so a head dimension or width has one formula.  They are
generic in the head dimension D: the layout is defined for any D whose rows are whole chunks
(D = 128 and 256 at both widths), and the pingpong kernel reads both: at D=256 its grouped body
copies one 128-channel group of a row at a time (`IntPingPongDecodeSm90._code_group_words`).

CODE POOL ORDERS.  Thread t's plane-word A-fragment row holds `per` bytes (`code_row_bytes`:
32 int4 / 16 int2 at D=128, 64 / 32 at D=256; `scripts/probes/repack_rs.py` defines the bytes).
A (page, kv-head) block stores the 128 rows in one of two orders, and the READER decides which:
  "row"      [128, per]           row t contiguous.  No kernel reads this order; it is the
                                  thread-row layout the packers produce and the pool shape
                                  the reader refuses by name.
  "chunk16"  [per/16, 128, 16]    chunk c of thread t = row bytes [16c, 16c+16).
                                  IntPingPongDecodeSm90 reads this (its `code_pool_order`).
Both orders are ONE contiguous 128*per-byte run per block, so the TMA copy of a block is the same
in both.  What chunk16 changes is the shared-memory read: a warp's 16-byte loads of chunk c cover
512 consecutive bytes (16t + 2048c), which is conflict-free, where row order puts thread t at
per*t: two lanes of every 8 on the same banks at a 32-byte row, four at a 64-byte row.  At D=128
int2 there is one chunk and the two orders are the same bytes; at int4 K chunk c is k-steps
4c..4c+3 (channels 64c..64c+63) and V chunk c is output half c, and the same holds for every
chunk at D=256 int4.  At D=256 int2 K chunk c is k-steps 8c..8c+7 (channels 128c..128c+127) and
V chunk c is output halves 2c and 2c+1.  Nothing in the bytes says which order a pool is in,
and a row pool reshaped to the chunk16 shape (a `view`, not `code_rows_to_order`) is accepted
and read wrong; every writer therefore asks the reader's predicate and converts with
`code_rows_to_order`.
`code_block_word_strides` says where each row word sits in a block of either order; the
pingpong kernel's shared-memory view and IntAppend's scatter are both derived from it and
from the kernel's `code_pool_order`, so reader and serving writer cannot disagree.

Per-request inputs to `make_serving_cache`:
  per_req_codes_k : uint8 [B, NT, HKV, <block_k>]     (repack_K_rs output, either order)
  per_req_codes_v : uint8 [B, NT, HKV, <block_v>]     (repack_V_rs output, the same order)
  per_req_scales  : f32   [B, HKV, S_pad, 4]          (s_k, z_k, s_v, z_v) per LOGICAL token,
                                                       S_pad % N == 0 (every staged tile covered;
                                                       tail values masked downstream but FINITE).
"""
import dataclasses

import torch

#: Bytes per chunk of the "chunk16" order: one 16-byte shared-memory load per chunk.
CODE_CHUNK_BYTES = 16

#: Channels of one V^T output half: the PV mma emits 64 output channels per run, so a V row
#: holds head_dim/64 halves stacked along the row (`repack_rs.repack_V_rs`).
CODE_V_HALF_CHANNELS = 64


def _require_order(order):
    if order not in ("row", "chunk16"):
        raise ValueError(f"unknown code pool order {order!r}; expected 'row' or 'chunk16'")


def code_row_chunks(per):
    """C, the number of 16-byte chunks in a `per`-byte thread row (a positive whole number)."""
    if per <= 0 or per % CODE_CHUNK_BYTES:
        raise ValueError(f"a thread row must be a positive whole number of 16-byte chunks; "
                         f"got {per} bytes")
    return per // CODE_CHUNK_BYTES


def code_v_half_bytes(page_size, bits):
    """Bytes of one output half of a V^T thread row: page_size/16 k-steps of 8 fragment values
    at `bits` each, which is 16 at int4 and 8 at int2 for a 64-token page."""
    if bits not in (2, 4):
        raise ValueError(f"bits must be 2 or 4; got {bits}")
    # int2 packs k-step PAIRS into one u32 word, so its k-steps must pair up.
    tokens = 16 if bits == 4 else 32
    if page_size <= 0 or page_size % tokens:
        raise ValueError(f"page_size must be a positive multiple of {tokens} at int{bits}; "
                         f"got {page_size}")
    return (page_size // 16) * 8 * bits // 8


def code_row_bytes(head_dim, page_size, bits):
    """(K, V) bytes of one warpgroup thread's plane-word row: the `per` of the two code pools.

    K: head_dim/16 k-steps x 8 fragment values x bits.  V^T: head_dim/64 output halves of
    `code_v_half_bytes` each.  At a 64-token page that is 32 / 32 bytes at D=128 int4, 16 / 16 at
    D=128 int2, 64 / 64 at D=256 int4 and 32 / 32 at D=256 int2.  A head dimension whose rows are
    not whole 16-byte chunks has no code pool in either order and is refused."""
    half = code_v_half_bytes(page_size, bits)
    if head_dim <= 0 or head_dim % CODE_V_HALF_CHANNELS:
        raise ValueError(f"head_dim must be a positive multiple of {CODE_V_HALF_CHANNELS} "
                         f"(whole V output halves); got {head_dim}")
    per_k = (head_dim // 16) * 8 * bits // 8
    per_v = (head_dim // CODE_V_HALF_CHANNELS) * half
    for per in (per_k, per_v):
        code_row_chunks(per)
    return per_k, per_v


def code_block_shape(order, per):
    """The (page, kv-head) block shape of a code pool whose thread rows hold `per` bytes."""
    _require_order(order)
    chunks = code_row_chunks(per)
    if order == "row":
        return (128, per)
    return (chunks, 128, CODE_CHUNK_BYTES)


def code_pool_shape(order, num_pages, num_kv_heads, per):
    """The shape of a whole code pool, [num_pages, num_kv_heads, <block>], in `order`.  The two
    leading extents are passed through as given, so a symbolic page count works."""
    return (num_pages, num_kv_heads) + code_block_shape(order, per)


def code_block_order(block_shape):
    """Which order a (page, kv-head) code block is in, read from its shape alone."""
    block = tuple(int(x) for x in block_shape)
    if len(block) == 2 and block[0] == 128 and block[1] > 0 and block[1] % CODE_CHUNK_BYTES == 0:
        return "row"
    if len(block) == 3 and block[0] > 0 and block[1:] == (128, CODE_CHUNK_BYTES):
        return "chunk16"
    raise ValueError(f"{block} is neither a 'row' [128, per] nor a 'chunk16' "
                     f"[per/16, 128, 16] code block")


def code_block_word_strides(order, per):
    """(thread, chunk) strides, in 4-byte words, of a (page, kv-head) block of `per`-byte rows.

    Word 4c + w (w < 4) of thread t's row sits at word t*thread + c*chunk + w of the block, in
    either order: both keep a 16-byte chunk's four words together, and they differ only in
    what is adjacent to a chunk.  The pingpong kernel's shared-memory view of a stage and the
    serving append's scatter are both built from this, so each follows whatever order
    `IntPingPongDecodeSm90.code_pool_order` names rather than restating chunk16."""
    code_block_shape(order, per)
    words = CODE_CHUNK_BYTES // 4
    if order == "row":
        return per // 4, words
    return words, 128 * words


def code_chunk_offset(order, per, thread, chunk):
    """Byte offset, inside a (page, kv-head) block, of chunk `chunk` of thread `thread`'s row.

    16t + 2048c in chunk16 and per*t + 16c in the row order.  This is the address of the
    consumer's 16-byte load (a ring stage holds one block verbatim), computed from the same
    `code_block_word_strides` the kernel's shared-memory view is built from."""
    thread_words, chunk_words = code_block_word_strides(order, per)
    if not (0 <= thread < 128 and 0 <= chunk < code_row_chunks(per)):
        raise ValueError(f"no chunk {chunk} of thread {thread} in a block of {per}-byte rows")
    return 4 * (thread * thread_words + chunk * chunk_words)


def code_word_offset(order, per, num_kv_heads, page, kv_head, thread, word, *, wide):
    """Word offset, in a whole C-contiguous code pool [NP, HKV, <block>], of row word `word`
    of thread `thread`'s row in the (page, kv_head) block.

    A pool holds NP * HKV * 32 * per words, which passes 2^31 once the pool passes 8 GiB (page
    262,144 at int4 with HKV = 8), so a 32-bit product of the page wraps negative and the
    address lands before the pool. `wide` converts the caller's page to a 64-bit integer, and
    it is applied to the page BEFORE the page is multiplied: cutlass.Int64 in IntAppend's
    kernel, a torch int64 conversion in the host tests. It is required because nothing else
    here knows the caller's integer type. The in-block part stays in the caller's type, since
    it is below 32 * per words."""
    thread_words, chunk_words = code_block_word_strides(order, per)
    block_words = 128 * per // 4
    in_block = thread * thread_words + (word // 4) * chunk_words + word % 4
    return (wide(page) * num_kv_heads + kv_head) * block_words + in_block


def scatter_map_file(head_dim):
    """The file name, under `scripts/`, of IntAppend's scatter map at `head_dim`.

    The map inverts the packer's fragment gather at one head dimension, so each head dimension
    has its own file: `frag_scatter_k16.pt` at 128, the file every D=128 append has read, and
    `frag_scatter_k16_d<D>.pt` at any other.  `scripts/probes/append_scatter.py --head-dim D`
    writes it under its byte-exact inverse gate, and IntAppend reads it; both take the name
    from here."""
    if head_dim <= 0 or head_dim % CODE_V_HALF_CHANNELS:
        raise ValueError(f"no scatter map at head_dim={head_dim}: the code pool is defined for "
                         f"whole {CODE_V_HALF_CHANNELS}-channel V halves")
    return "frag_scatter_k16.pt" if head_dim == 128 else f"frag_scatter_k16_d{head_dim}.pt"


def code_rows_to_order(rows, order):
    """Thread rows [..., 128, per] -> the same bytes in `order`, C-contiguous.

    Written into a fresh tensor rather than by `.contiguous()`: with one chunk the
    transposed chunk dim has extent 1, torch then calls the view contiguous and keeps its
    stride of 16, which moves no byte but reads as a malformed pool to anything that
    compares strides."""
    if rows.dim() < 2 or rows.shape[-2] != 128:
        raise ValueError(f"thread rows must be [..., 128, per]; got {tuple(rows.shape)}")
    block = code_block_shape(order, rows.shape[-1])
    if order == "row":
        return rows.contiguous()
    lead = rows.shape[:-2]
    out = rows.new_empty((*lead, *block))
    out.copy_(rows.reshape(*lead, 128, block[0], CODE_CHUNK_BYTES).transpose(-3, -2))
    return out


_BLOCK_TEXT = {"row": "128, per", "chunk16": "per/16, 128, 16"}


def require_code_pool(name, shape, order, per=None, stride=None):
    """Refuse a code pool that is not a C-contiguous [NP, HKV, <block>] in `order`.

    With `per`, the block must be exactly `code_block_shape(order, per)`.  Without it only
    the order is checked, for a launcher that knows the reader but not its row bytes; the
    compiled kernel's own argument check still holds the extents.  With `stride`, the pool
    must also be dense in C order (an extent-1 dim's stride moves no byte and is not
    checked), because the kernel reads the block as one contiguous run and a transposed
    tensor of the right shape would be read in the wrong order.  The error names the
    reader's order and the order the pool's shape says it has, because the likely mistake
    is a row-major pool handed to the chunk16 reader.  Plain Python on purpose: the
    kernel's `__call__` calls it at trace time, where the DSL forbids a `raise` of its own
    and `python -O` would drop an assert."""
    _require_order(order)
    got = tuple(shape)
    try:
        found = code_block_order(got[2:])
    except ValueError:
        found = None
    if found == order and (per is None or tuple(got[2:]) == code_block_shape(order, per)):
        if stride is not None:
            dense, acc = [], 1
            for extent in reversed(got[1:]):
                dense.insert(0, acc)
                acc *= int(extent)
            dense.insert(0, acc)
            st = tuple(stride)
            # The page dim is symbolic in `compile_constructed`'s fake tensors, so it counts
            # as extent 1 only when it is a plain int 1; every later dim is static.
            free = [isinstance(got[0], int) and got[0] == 1] + [int(e) == 1 for e in got[1:]]
            if any(not f and s != d for f, s, d in zip(free, st, dense)):
                raise ValueError(
                    f"{name} has strides {st}; a code pool must be C-contiguous, strides "
                    f"{tuple(dense)} (serving_cache contract)")
        return
    want = (", ".join(str(x) for x in code_block_shape(order, per)) if per is not None
            else _BLOCK_TEXT[order])
    what = f"a {found!r}-order pool" if found else "a pool in neither known order"
    raise ValueError(
        f"{name} has shape {got}, which is {what}; this kernel reads the {order!r} "
        f"code-pool order, [num_pages, num_kv_heads, {want}]. Build it with "
        f"repack_rs.repack_K_rs / repack_V_rs(order={order!r}), or convert thread rows "
        f"with serving_cache.code_rows_to_order(rows, {order!r}). Do not view or reshape "
        f"a pool into that shape: the orders hold the same bytes in different places, so "
        f"a reshaped pool passes this check and decodes wrong at int4.")


@dataclasses.dataclass
class ServingCache:
    """The four contract tensors + the page size N (see module docstring)."""
    codes_k: torch.Tensor      # uint8 [NP, HKV, <block_k>], the reader's code pool order
    codes_v: torch.Tensor      # uint8 [NP, HKV, <block_v>], the same order
    scales: torch.Tensor       # sdtype [NP, HKV, N, 4]
    block_table: torch.Tensor  # int32 [B, maxPages]
    seq_lens: torch.Tensor     # int32 [B]
    N: int = 64


def _code_blocks(per_req_codes_k, per_req_codes_v):
    """The per-(page, kv-head) block shapes of both code tensors, which share one order."""
    block_k = tuple(per_req_codes_k.shape[3:])
    block_v = tuple(per_req_codes_v.shape[3:])
    if code_block_order(block_k) != code_block_order(block_v):
        raise ValueError(f"K codes {block_k} and V codes {block_v} are in different orders")
    return block_k, block_v


def make_serving_cache(per_req_codes_k, per_req_codes_v, per_req_scales, seq_lens, N=64,
                       sdtype=torch.float32, shuffle=True, identity_block_table=False):
    """Scatter each request's NT code tiles + matching N-token scale blocks into ONE global pool.

    per_req_codes_k : uint8 [B, NT, HKV, <block_k>]    (repack_K_rs output)
    per_req_codes_v : uint8 [B, NT, HKV, <block_v>]    (repack_V_rs output, the same order)
    per_req_scales  : f32   [B, HKV, S_pad, 4]          (s_k, z_k, s_v, z_v) per logical token
    seq_lens        : int32 [B]                          per-request kv length

    The code blocks are copied as they are, so the pool is in whatever order the per-request
    codes were written in (module docstring, CODE POOL ORDERS).

    Each request b contributes `ceil(seq_lens[b]/N)` pages.  Pages are scattered into ONE pool of
    NP = sum_b pages_per[b] physical pages via torch.randperm (shuffle=True) so a wrong page-id
    read lands in another request's data; block_table[b, gt] records the global page id.

    identity_block_table=True (or shuffle=False) lays pages out as base_b + gt (no permutation) —
    the contiguous-stand-in case (the retired page_size=0 path is exactly an identity block table).
    identity_block_table forces no shuffle.

    Returns a ServingCache.
    """
    dev = per_req_codes_k.device
    B, NT, HKV = per_req_codes_k.shape[0], per_req_codes_k.shape[1], per_req_codes_k.shape[2]
    block_k, block_v = _code_blocks(per_req_codes_k, per_req_codes_v)
    assert per_req_codes_v.shape[:3] == (B, NT, HKV)
    assert per_req_scales.shape[0] == B and per_req_scales.shape[1] == HKV
    assert per_req_scales.shape[3] == 4, "per_req_scales last dim must be 4 (s_k,z_k,s_v,z_v)"
    S_pad = per_req_scales.shape[2]
    assert S_pad % N == 0, f"scale S_pad {S_pad} must be a multiple of N={N}"
    assert S_pad // N >= NT, "scale S_pad must cover every code tile (S_pad/N >= NT)"

    seq_lens = seq_lens.to(device=dev, dtype=torch.int32)
    pages_per = [(int(s) + N - 1) // N for s in seq_lens.tolist()]
    pairs = [(b, gt) for b in range(B) for gt in range(pages_per[b])]
    NP = len(pairs)
    assert NP > 0, "empty cache (all seq_lens == 0)"

    if identity_block_table:
        shuffle = False
    if shuffle:
        perm = torch.randperm(NP).tolist()
    else:
        perm = list(range(NP))   # identity: page i = the i-th (b, gt) in row-major order

    maxPages = max(1, max(pages_per))
    block_table = torch.zeros(B, maxPages, dtype=torch.int32, device=dev)
    codes_k = torch.zeros((NP, HKV) + block_k, dtype=torch.uint8, device=dev)
    codes_v = torch.zeros((NP, HKV) + block_v, dtype=torch.uint8, device=dev)
    # interleaved paged scale pool [NP, HKV, N, 4] (sdtype): per-(hk,page) the N*4 block is fully
    # contiguous (so the bf16 TMA box of 256 elems clears the 16B inner-box minimum).  Reshape the
    # per-req scales [B, HKV, S_pad, 4] into [B, HKV, NTp, N, 4] tiles so a whole N-token block
    # copies in one slice (token-major within the block, the 4 streams innermost).
    scales = torch.zeros(NP, HKV, N, 4, dtype=sdtype, device=dev)
    sc_src = per_req_scales.to(sdtype)                      # [B, HKV, S_pad, 4]
    sc_src = sc_src.reshape(B, HKV, S_pad // N, N, 4)       # [B, HKV, NTp, N, 4]

    for i, (b, gt) in enumerate(pairs):
        p = perm[i]
        codes_k[p] = per_req_codes_k[b, gt]
        codes_v[p] = per_req_codes_v[b, gt]
        scales[p] = sc_src[b, :, gt]                        # [HKV, N, 4] block for tile gt
        block_table[b, gt] = p

    return ServingCache(
        codes_k=codes_k.contiguous(), codes_v=codes_v.contiguous(),
        scales=scales.contiguous(), block_table=block_table.contiguous(),
        seq_lens=seq_lens.contiguous(), N=N)


# ---------------------------------------------------------------------------------------------
# Flexible-granularity pool (per-tensor quant configs) — consumed by IntPingPongDecodeSm90's
# k_quant/v_quant modes.  The scale-block layout is NOT defined here, and must not be
# copied here: it is the same contract both decode kernels read, and a second copy needs
# a drift check to stay honest.  See `cutedsl_decode/quant_layout.py` for the sectioned
# layout it describes.
# ---------------------------------------------------------------------------------------------
from ..quant_layout import quant_block_offsets  # noqa: F401  (re-exported for callers)
def make_serving_cache_flex(per_req_codes_k, per_req_codes_v, k_sz, v_sz, seq_lens,
                            k_quant=None, v_quant=None, N=64, *, D,
                            sdtype=torch.float32, shuffle=True, identity_block_table=False):
    """Build a flexible-granularity ServingCache. k_sz and v_sz are the (scale, zero) tensors
    of quant_ref.quantize_grouped in LOGICAL token order, padded to whole pages, shaped by pair:
      ("token", D):   [B, HKV, S_pad]           per token
      ("token", D/2): [B, HKV, S_pad, 2]        per token x {lo, hi} channel half
      ("channel", Gt):[B, HKV, S_pad//Gt, D]    per channel x token-group
    `D` is the head dimension, and it is required because a stale default would silently select
    the half-group branch for a full-D pool. A `None` pair is ("token", D), the kernel's own
    default. The code tensors are repack_rs outputs in either code pool order, copied as they
    are. The returned `scales` pool is make_serving_cache's [NP, HKV, N, 4] interleave when both
    pairs are ("token", D), and otherwise the sectioned block of quant_block_offsets."""
    if k_quant is None:
        k_quant = ("token", D)
    if v_quant is None:
        v_quant = ("token", D)
    dev = per_req_codes_k.device
    B, NT, HKV = per_req_codes_k.shape[0], per_req_codes_k.shape[1], per_req_codes_k.shape[2]
    block_k, block_v = _code_blocks(per_req_codes_k, per_req_codes_v)
    off = quant_block_offsets(k_quant, v_quant, N=N, D=D)
    if off["legacy"]:
        # (s_k, z_k, s_v, z_v) interleave — delegate to the shipped builder
        sk, zk = k_sz
        sv, zv = v_sz
        S_pad = sk.shape[-1]
        inter = torch.stack([sk, zk, sv, zv], dim=-1).float()   # [B, HKV, S_pad, 4]
        return make_serving_cache(per_req_codes_k, per_req_codes_v, inter, seq_lens, N=N,
                                  sdtype=sdtype, shuffle=shuffle,
                                  identity_block_table=identity_block_table)

    def ring_block(qc, sz):
        s, z = sz
        axis, g = qc
        if axis == "channel":
            return None
        S_pad = s.shape[2]
        if g == D:      # [B,HKV,S_pad] -> [B,HKV,NTp,N,2]
            blk = torch.stack([s, z], dim=-1)
            return blk.reshape(B, HKV, S_pad // N, N * 2).float()
        # g == D//2: [B,HKV,S_pad,2] -> (s_lo,z_lo,s_hi,z_hi) per token
        blk = torch.stack([s[..., 0], z[..., 0], s[..., 1], z[..., 1]], dim=-1)
        return blk.reshape(B, HKV, S_pad // N, N * 4).float()

    def ch_block(qc, sz):
        s, z = sz                                   # [B,HKV,S_pad//Gt,D]
        axis, g = qc
        if axis != "channel":
            return None
        n_grp = s.shape[2]
        blk = torch.stack([s, z], dim=-2).float()   # [B,HKV,n_grp,2,D]
        blk = blk.reshape(B, HKV, n_grp, 2 * D)
        # duplicate per page: page p belongs to group p*N//g
        pages = torch.arange(NT, device=s.device) * N // g
        return blk[:, :, pages.clamp(max=n_grp - 1)]           # [B,HKV,NT,2D]

    parts = []
    for qc, sz, rk in ((k_quant, k_sz, "k"), (v_quant, v_sz, "v")):
        rb = ring_block(qc, sz)
        if rb is not None:
            parts.append(("ring_" + rk, rb))
    for qc, sz, rk in ((k_quant, k_sz, "k"), (v_quant, v_sz, "v")):
        cb = ch_block(qc, sz)
        if cb is not None:
            parts.append(("ch_" + rk, cb))
    NTp = min(p.shape[2] for _, p in parts)
    block = torch.cat([p[:, :, :NTp] for _, p in parts], dim=-1)      # [B,HKV,NTp,block_tot]
    assert block.shape[-1] == off["block_tot"], (block.shape, off)
    assert NTp >= NT, "scale blocks must cover every code tile"

    # scatter pages exactly like make_serving_cache
    seq_lens = seq_lens.to(device=dev, dtype=torch.int32)
    pages_per = [(int(s) + N - 1) // N for s in seq_lens.tolist()]
    pairs = [(b, gt) for b in range(B) for gt in range(pages_per[b])]
    NP = len(pairs)
    if identity_block_table:
        shuffle = False
    perm = torch.randperm(NP).tolist() if shuffle else list(range(NP))
    maxPages = max(1, max(pages_per))
    block_table = torch.zeros(B, maxPages, dtype=torch.int32, device=dev)
    codes_k = torch.zeros((NP, HKV) + block_k, dtype=torch.uint8, device=dev)
    codes_v = torch.zeros((NP, HKV) + block_v, dtype=torch.uint8, device=dev)
    scales = torch.zeros(NP, HKV, off["block_tot"], dtype=sdtype, device=dev)
    blk = block.to(sdtype)
    for i, (b, gt) in enumerate(pairs):
        p = perm[i]
        codes_k[p] = per_req_codes_k[b, gt]
        codes_v[p] = per_req_codes_v[b, gt]
        scales[p] = blk[b, :, gt]
        block_table[b, gt] = p
    return ServingCache(codes_k=codes_k.contiguous(), codes_v=codes_v.contiguous(),
                        scales=scales.contiguous(), block_table=block_table.contiguous(),
                        seq_lens=seq_lens.contiguous(), N=N)
