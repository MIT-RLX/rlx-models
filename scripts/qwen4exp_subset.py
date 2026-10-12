#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# SPDX-License-Identifier: GPL-3.0-only
"""Build a small standalone `qwen4exp` GGUF out of published Qwen3.8-Flash-Next shards.

Qwen3.8-Flash-Next is 512x56B. Its smallest published quantization is tens of
gigabytes, and `per_layer_token_embd.weight` alone — the 51-billion-weight
n-gram table — is ~28 GB of that. Testing `rlx-qwen4exp` against real weights by
downloading the model is not practical. It does not have to be: a GGUF header
carries every tensor's byte offset, so the layers worth testing can be pulled
out with HTTP range requests.

By default this fetches **~90 MB** — `blk.1` (linear attention + the model's one
PLE module) and `blk.3` (gated attention + the QSA indexer), with 8 of each
block's 512 routed experts, plus the three `output_hc_*` tensors. Those two
blocks are the model's two layer *kinds*, and between them they cover every
emitter in the crate except the LM head.

    python3 scripts/qwen4exp_subset.py out.gguf
    RLX_QWEN4EXP_GGUF=out.gguf cargo test -p rlx-qwen4exp --test real_weights

What is deliberately NOT fetched:

* **`per_layer_token_embd.weight`** — the n-gram table. It is never a graph
  parameter (the crate takes the embeddings as a host-gathered input), so a
  weights test does not need it. `--ple-rows N` fetches a prefix anyway if you
  want to check a row read; each row is `embedding_length_per_layer_input`
  values, contiguous, so N rows is a cheap prefix of the tensor.
* **`token_embd.weight` / `output.weight`** — 1.2 GB together at Q4_K, and the
  embedding and LM head are not what this crate's novel code is. `--with-embd`
  adds them.

Pass `--blocks 0,3,7` to pick different blocks, `--experts N` to change how many
expert slabs come along (0 to skip the banks entirely), or `--dry-run` to see
the plan without downloading.

`--mmproj N` switches to the **vision tower** instead: the same trick on
`mmproj-F16.gguf`, keeping its globals (patch embedding, position table,
`v.post_ln`, the merger) and the first `N` of its 27 blocks, with
`clip.vision.block_count` rewritten to match. At `N = 3` that is ~110 MB of
real ViT weights rather than 904 MB.

    python3 scripts/qwen4exp_subset.py --mmproj 3 mmproj-sub.gguf
    RLX_QWEN4EXP_MMPROJ=mmproj-sub.gguf \
      cargo test -p rlx-qwen4exp --features vision --test vision
"""
import argparse
import os
import struct
import urllib.request

BLOCK = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 8: ('Q8_0', 32, 34), 10: ('Q2_K', 256, 84),
         11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176),
         14: ('Q6_K', 256, 210), 16: ('IQ2_XXS', 256, 66), 18: ('IQ3_XXS', 256, 98),
         19: ('IQ1_S', 256, 50), 20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110),
         22: ('IQ2_S', 256, 82), 23: ('IQ4_XS', 256, 136), 29: ('IQ1_M', 256, 56),
         30: ('BF16', 1, 2)}


class RangeReader:
    def __init__(self, url, chunk=8 << 20):
        self.url, self.buf, self.pos, self.chunk = url, b'', 0, chunk

    def _fetch(self, upto):
        while len(self.buf) < upto:
            lo = len(self.buf)
            hi = lo + max(self.chunk, upto - lo) - 1
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={lo}-{hi}'})
            with urllib.request.urlopen(req) as r:
                d = r.read()
            if not d:
                raise EOFError
            self.buf += d

    def read(self, n):
        self._fetch(self.pos + n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b


def u32(r): return struct.unpack('<I', r.read(4))[0]
def u64(r): return struct.unpack('<Q', r.read(8))[0]


def rstr(r):
    n = u64(r)
    return r.read(n).decode('utf-8', 'replace')


T_STR, T_ARR = 8, 9
FMT = {0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2), 4: ('<I', 4), 5: ('<i', 4),
       6: ('<f', 4), 7: ('<?', 1), 10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8)}


def rval(r, t):
    if t == T_STR:
        return ('s', rstr(r))
    if t == T_ARR:
        et = u32(r)
        n = u64(r)
        return ('a', et, [rval(r, et) for _ in range(n)])
    f, sz = FMT[t]
    return ('v', t, struct.unpack(f, r.read(sz))[0])


def wstr(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def wval(v):
    if v[0] == 's':
        return struct.pack('<I', T_STR) + wstr(v[1])
    if v[0] == 'a':
        _, et, items = v
        out = struct.pack('<I', T_ARR) + struct.pack('<I', et) + struct.pack('<Q', len(items))
        for it in items:
            out += wval_inner(it)
        return out
    return struct.pack('<I', v[1]) + struct.pack(FMT[v[1]][0], v[2])


def wval_inner(v):
    if v[0] == 's':
        return wstr(v[1])
    if v[0] == 'a':
        raise ValueError('nested arrays unsupported')
    return struct.pack(FMT[v[1]][0], v[2])


def read_header(url):
    r = RangeReader(url)
    assert r.read(4) == b'GGUF', f'{url} is not a GGUF'
    ver = u32(r)
    tc = u64(r)
    kvc = u64(r)
    kv = {}
    for _ in range(kvc):
        k = rstr(r)
        t = u32(r)
        kv[k] = rval(r, t)
    tensors = []
    for _ in range(tc):
        name = rstr(r)
        nd = u32(r)
        dims = [u64(r) for _ in range(nd)]
        typ = u32(r)
        off = u64(r)
        tensors.append((name, dims, typ, off))
    align = kv.get('general.alignment')
    align = align[2] if align else 32
    data_start = (r.pos + align - 1) // align * align
    return ver, kv, tensors, data_start, align


def nbytes(dims, typ):
    n = 1
    for d in dims:
        n *= d
    _, blk, sz = BLOCK[typ]
    assert n % blk == 0, f'{n} is not a multiple of the {blk}-element block'
    return n // blk * sz




def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('out', nargs='?', default='qwen4exp-blk1-blk3.gguf')
    ap.add_argument('--blocks', default='1,3',
                    help='comma-separated block indices (default 1,3 — the PLE '
                         'linear layer and a QSA attention layer)')
    ap.add_argument('--experts', type=int, default=8, metavar='N',
                    help='keep only the FIRST N experts of each routed bank, '
                         'rewriting the declared expert axis. A quantized bank '
                         'is [in, out, experts] with the expert axis OUTERMOST, '
                         'so each expert is a contiguous byte range — 8 of 512 '
                         'is ~12 MB instead of ~800 MB. The result is a valid '
                         'N-expert bank for block-level tests, not the real '
                         'routing. 0 skips the banks.')
    ap.add_argument('--ple-rows', type=int, default=0, metavar='N',
                    help='also fetch the first N rows of per_layer_token_embd '
                         '(the n-gram table). Not needed to run a block: the '
                         'crate takes the embeddings as a graph input.')
    ap.add_argument('--with-embd', action='store_true',
                    help='also fetch token_embd and output (~1.2 GB at Q4_K)')
    ap.add_argument('--mmproj', type=int, default=0, metavar='N',
                    help='slice the VISION tower instead: keep its globals and '
                         'the first N of 27 blocks, rewriting '
                         'clip.vision.block_count. The tower is one file, so '
                         'this reads it directly rather than a shard set.')
    ap.add_argument('--mmproj-file', default='mmproj-F16.gguf',
                    help='which mmproj to slice (default mmproj-F16.gguf)')
    ap.add_argument('--repo', default='unsloth/Qwen3.8-Flash-Next-GGUF')
    ap.add_argument('--quant', default='UD-IQ1_S')
    ap.add_argument('--shards', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true',
                    help='print what would be fetched and exit (header reads only)')
    args = ap.parse_args()
    blocks = {int(b) for b in args.blocks.split(',') if b.strip()}

    base = f'https://huggingface.co/{args.repo}/resolve/main'
    if args.mmproj > 0:
        return mmproj_subset(args, base)
    stem = f'{base}/{args.quant}/Qwen3.8-Flash-Next-{args.quant}'
    def shard_url(i): return f'{stem}-{i:05d}-of-{args.shards:05d}.gguf'

    print('reading metadata from shard 1 ...', flush=True)
    _, kv_meta, _, _, _ = read_header(shard_url(1))

    def want(n):
        if n == 'per_layer_token_embd.weight':
            return args.ple_rows > 0
        if n in ('token_embd.weight', 'output.weight'):
            return args.with_embd
        if n.startswith('output_hc_'):
            return True
        if not n.startswith('blk.'):
            return False
        try:
            idx = int(n.split('.')[1])
        except (IndexError, ValueError):
            return False
        if idx not in blocks:
            return False
        # The routed banks are ~800 MB for a single block; everything else in a
        # block is small enough to be worth having unconditionally.
        if '_exps.' in n:
            return args.experts > 0
        return True

    # Tensors live in the data shards (2..N); collect whichever shard has each.
    picked = []          # (name, dims, typ, src_off, shard_index)
    layout = {}          # shard index -> (data_start, align)
    for i in range(2, args.shards + 1):
        print(f'reading tensor index from shard {i} ...', flush=True)
        _, _, tensors, data_start, align = read_header(shard_url(i))
        layout[i] = (data_start, align)
        for t in tensors:
            if want(t[0]):
                picked.append((*t, i))
    if not picked:
        raise SystemExit(f'no tensors matched blocks {sorted(blocks)}')
    align = next(iter(layout.values()))[1]
    print(f'{len(picked)} tensors selected')

    # Drop the tokenizer arrays: 248320 tokens + 247587 merges say nothing about
    # the weights and would dominate the file's header.
    drop = ('tokenizer.ggml.tokens', 'tokenizer.ggml.token_type',
            'tokenizer.ggml.merges', 'tokenizer.chat_template')
    kv_out = {k: v for k, v in kv_meta.items()
              if k not in drop and not k.startswith('split.')}
    # The reader falls back to the tokenizer token count when there is no
    # `vocab_size` key, and we just dropped the tokens — so state it.
    if 'qwen4exp.vocab_size' not in kv_out:
        n_tok = len(kv_meta['tokenizer.ggml.tokens'][2]) if 'tokenizer.ggml.tokens' in kv_meta else 0
        kv_out['qwen4exp.vocab_size'] = ('v', 4, n_tok)
    kv_out['general.name'] = (
        's', 'Qwen3.8 Flash Next (blocks %s subset)' % ','.join(map(str, sorted(blocks))))

    hdr = (b'GGUF' + struct.pack('<I', 3)
           + struct.pack('<Q', len(picked)) + struct.pack('<Q', len(kv_out)))
    for k, v in kv_out.items():
        hdr += wstr(k) + wval(v)

    infos = b''
    off = 0
    plan = []   # (name, shard, src_abs, nbytes, dst)
    for name, dims, typ, src_off, shard in sorted(picked):
        if '_exps.' in name and args.experts > 0 and len(dims) == 3:
            full = dims[2]
            keep = min(args.experts, full)
            dims = [dims[0], dims[1], keep]
            print(f'    {name}: {keep}/{full} experts')
        elif name == 'ffn_gate_inp.weight' or name.endswith('.ffn_gate_inp.weight'):
            # The router describes every expert; narrow it to match the banks so
            # a top-k over it cannot select a slab that is not here. ne[1] is the
            # outer dim, so the first N rows are a contiguous prefix.
            if args.experts > 0 and len(dims) == 2:
                keep = min(args.experts, dims[1])
                print(f'    {name}: {keep}/{dims[1]} experts')
                dims = [dims[0], keep]
        elif name == 'per_layer_token_embd.weight' and args.ple_rows > 0:
            keep = min(args.ple_rows, dims[1])
            print(f'    {name}: {keep}/{dims[1]} rows')
            dims = [dims[0], keep]
        sz = nbytes(dims, typ)
        infos += wstr(name) + struct.pack('<I', len(dims))
        for d in dims:
            infos += struct.pack('<Q', d)
        infos += struct.pack('<I', typ) + struct.pack('<Q', off)
        plan.append((name, shard, layout[shard][0] + src_off, sz, off))
        off = (off + sz + align - 1) // align * align

    total = sum(p[3] for p in plan)
    print(f'{total/1e6:.1f} MB of tensor data')
    head = hdr + infos
    head += b'\x00' * ((-len(head)) % align)
    blob_len = off

    # Coalesce adjacent source ranges per shard so this is a handful of big
    # GETs rather than one per tensor.
    runs = {}
    for name, shard, src, sz, dst in sorted(plan, key=lambda p: (p[1], p[2])):
        rs = runs.setdefault(shard, [])
        if rs and src == rs[-1][1]:
            rs[-1][1] = src + sz
            rs[-1][2].append((name, src, sz, dst))
        else:
            rs.append([src, src + sz, [(name, src, sz, dst)]])
    n_runs = sum(len(v) for v in runs.values())
    print(f'{n_runs} contiguous ranges across {len(runs)} shard(s)')

    if args.dry_run:
        for name, _, _, sz, _ in sorted(plan, key=lambda p: -p[3])[:10]:
            print(f'    {name:48s} {sz/1e6:8.2f} MB')
        print(f'dry run: would write {args.out} '
              f'({(len(head) + blob_len)/1e6:.1f} MB)')
        return

    with open(args.out, 'wb') as f:
        f.write(head)
        f.truncate(len(head) + blob_len)
        done = 0
        for shard, rs in sorted(runs.items()):
            url = shard_url(shard)
            for lo, hi, items in rs:
                req = urllib.request.Request(url, headers={'Range': f'bytes={lo}-{hi-1}'})
                with urllib.request.urlopen(req) as r:
                    buf = r.read()
                assert len(buf) == hi - lo, f'short read {len(buf)} != {hi-lo}'
                for name, src, sz, dst in items:
                    f.seek(len(head) + dst)
                    f.write(buf[src - lo: src - lo + sz])
                done += hi - lo
                print(f'  {done/1e6:8.1f} / {total/1e6:.1f} MB', flush=True)
    print(f'wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)')


def mmproj_subset(args, base):
    """Slice the vision tower: globals + the first N blocks, one file in, one out."""
    url = f'{base}/{args.mmproj_file}'
    print(f'reading {args.mmproj_file} header ...', flush=True)
    _, kv, tensors, data_start, align = read_header(url)

    n_layer = kv['clip.vision.block_count'][2]
    keep = min(args.mmproj, n_layer)
    print(f'keeping {keep}/{n_layer} vision blocks')

    def want(name):
        if not name.startswith('v.blk.'):
            # Globals: patch embedding, position table, post_ln, the merger.
            return True
        try:
            return int(name.split('.')[2]) < keep
        except (IndexError, ValueError):
            return False

    sel = [t for t in tensors if want(t[0])]
    if not sel:
        raise SystemExit('no vision tensors matched')
    print(f'{len(sel)} tensors selected')

    kv_out = dict(kv)
    # The consumer sizes its block loop from this, so it has to agree with the
    # tensors actually present.
    kv_out['clip.vision.block_count'] = ('v', 4, keep)
    # `is_deepstack_layers` is per-layer; truncate it so index N-1 is the last.
    ds = kv_out.get('clip.vision.is_deepstack_layers')
    if ds and ds[0] == 'a':
        kv_out['clip.vision.is_deepstack_layers'] = ('a', ds[1], ds[2][:keep])
    kv_out['general.name'] = ('s', f'Qwen3.8 Flash Next mmproj ({keep} blocks)')

    hdr = (b'GGUF' + struct.pack('<I', 3)
           + struct.pack('<Q', len(sel)) + struct.pack('<Q', len(kv_out)))
    for k, v in kv_out.items():
        hdr += wstr(k) + wval(v)

    infos = b''
    off = 0
    plan = []
    for name, dims, typ, src_off in sorted(sel):
        sz = nbytes(dims, typ)
        infos += wstr(name) + struct.pack('<I', len(dims))
        for d in dims:
            infos += struct.pack('<Q', d)
        infos += struct.pack('<I', typ) + struct.pack('<Q', off)
        plan.append((name, data_start + src_off, sz, off))
        off = (off + sz + align - 1) // align * align

    total = sum(p[2] for p in plan)
    print(f'{total/1e6:.1f} MB of tensor data')
    head = hdr + infos
    head += b'\x00' * ((-len(head)) % align)

    runs = []
    for name, src, sz, dst in sorted(plan, key=lambda p: p[1]):
        if runs and src == runs[-1][1]:
            runs[-1][1] = src + sz
            runs[-1][2].append((name, src, sz, dst))
        else:
            runs.append([src, src + sz, [(name, src, sz, dst)]])
    print(f'{len(runs)} contiguous ranges to fetch')

    if args.dry_run:
        for name, _, sz, _ in sorted(plan, key=lambda p: -p[2])[:8]:
            print(f'    {name:44s} {sz/1e6:8.2f} MB')
        print(f'dry run: would write {args.out} ({(len(head) + off)/1e6:.1f} MB)')
        return

    with open(args.out, 'wb') as f:
        f.write(head)
        f.truncate(len(head) + off)
        done = 0
        for lo, hi, items in runs:
            req = urllib.request.Request(url, headers={'Range': f'bytes={lo}-{hi-1}'})
            with urllib.request.urlopen(req) as r:
                buf = r.read()
            assert len(buf) == hi - lo, f'short read {len(buf)} != {hi-lo}'
            for name, src, sz, dst in items:
                f.seek(len(head) + dst)
                f.write(buf[src - lo: src - lo + sz])
            done += hi - lo
            print(f'  {done/1e6:8.1f} / {total/1e6:.1f} MB', flush=True)
    print(f'wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)')


main()
