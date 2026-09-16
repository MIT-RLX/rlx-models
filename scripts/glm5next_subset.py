#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# SPDX-License-Identifier: GPL-3.0-only
"""Build a small standalone `glm5next` GGUF out of a published GLM-5.3-Flash shard.

GLM-5.3-Flash is 93 GB at its smallest quantization and its smallest *text*
shard is 43.5 GB, so testing `rlx-glm5next` against real weights by downloading
the model is not practical. It does not have to be: a GGUF header carries every
tensor's byte offset, so the layers worth testing can be pulled out with HTTP
range requests.

By default this fetches **337 MB** — `blk.0` in full (KDA + dense FFN + mHC) plus
`blk.3`'s attention, indexer and mHC (everything but its 2.2 GB of routed expert
banks) — and writes a valid single-file GGUF carrying the real `glm5next.*`
metadata. Those two blocks are the model's two layer *kinds*, so between them
they cover every block the crate emits.

    python3 scripts/glm5next_subset.py out.gguf
    RLX_GLM5NEXT_GGUF=out.gguf cargo test -p rlx-glm5next --test real_weights

Pass `--blocks 0,3,7` to pick different blocks, or `--with-experts` to include
the routed banks (adds ~2.2 GB per MoE block).
"""
import argparse

import struct, sys, urllib.request, os

BLOCK = {0:('F32',1,4),1:('F16',1,2),8:('Q8_0',32,34),10:('Q2_K',256,84),11:('Q3_K',256,110),
         12:('Q4_K',256,144),13:('Q5_K',256,176),14:('Q6_K',256,210),16:('IQ2_XXS',256,66),
         18:('IQ3_XXS',256,98),19:('IQ1_S',256,50),23:('IQ4_XS',256,136),29:('IQ1_M',256,56),
         30:('BF16',1,2)}

class RangeReader:
    def __init__(self, url, chunk=8 << 20):
        self.url, self.buf, self.pos, self.chunk = url, b'', 0, chunk
    def _fetch(self, upto):
        while len(self.buf) < upto:
            lo = len(self.buf); hi = lo + max(self.chunk, upto - lo) - 1
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={lo}-{hi}'})
            with urllib.request.urlopen(req) as r:
                d = r.read()
            if not d: raise EOFError
            self.buf += d
    def read(self, n):
        self._fetch(self.pos + n)
        b = self.buf[self.pos:self.pos + n]; self.pos += n; return b

def u32(r): return struct.unpack('<I', r.read(4))[0]
def u64(r): return struct.unpack('<Q', r.read(8))[0]
def rstr(r):
    n = u64(r); return r.read(n).decode('utf-8', 'replace')

T_STR, T_ARR = 8, 9
FMT = {0:('<B',1),1:('<b',1),2:('<H',2),3:('<h',2),4:('<I',4),5:('<i',4),
       6:('<f',4),7:('<?',1),10:('<Q',8),11:('<q',8),12:('<d',8)}

def rval(r, t):
    if t == T_STR: return ('s', rstr(r))
    if t == T_ARR:
        et = u32(r); n = u64(r)
        return ('a', et, [rval(r, et) for _ in range(n)])
    f, sz = FMT[t]; return ('v', t, struct.unpack(f, r.read(sz))[0])

def wstr(s):
    b = s.encode(); return struct.pack('<Q', len(b)) + b

def wval(v):
    if v[0] == 's': return struct.pack('<I', T_STR) + wstr(v[1])
    if v[0] == 'a':
        _, et, items = v
        out = struct.pack('<I', T_ARR) + struct.pack('<I', et) + struct.pack('<Q', len(items))
        for it in items:
            out += wval_inner(it)
        return out
    return struct.pack('<I', v[1]) + struct.pack(FMT[v[1]][0], v[2])

def wval_inner(v):
    if v[0] == 's': return wstr(v[1])
    if v[0] == 'a': raise ValueError('nested arrays unsupported')
    return struct.pack(FMT[v[1]][0], v[2])

def read_header(url):
    r = RangeReader(url)
    assert r.read(4) == b'GGUF'
    ver = u32(r); tc = u64(r); kvc = u64(r)
    kv = {}
    for _ in range(kvc):
        k = rstr(r); t = u32(r); kv[k] = rval(r, t)
    tensors = []
    for _ in range(tc):
        name = rstr(r); nd = u32(r); dims = [u64(r) for _ in range(nd)]
        typ = u32(r); off = u64(r)
        tensors.append((name, dims, typ, off))
    align = kv.get('general.alignment')
    align = align[2] if align else 32
    data_start = (r.pos + align - 1) // align * align
    return ver, kv, tensors, data_start, align

def nbytes(dims, typ):
    n = 1
    for d in dims: n *= d
    _, blk, sz = BLOCK[typ]
    assert n % blk == 0, f'{n} not a multiple of block {blk}'
    return n // blk * sz

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('out', nargs='?', default='glm5next-blk0-blk3.gguf')
    ap.add_argument('--blocks', default='0,3',
                    help='comma-separated block indices to extract (default 0,3)')
    ap.add_argument('--with-experts', action='store_true',
                    help='include the routed expert banks (~2.2 GB per MoE block)')
    ap.add_argument('--experts', type=int, default=0, metavar='N',
                    help='include only the FIRST N experts of each routed bank, '
                         'rewriting the bank to declare N experts. A quantized '
                         'bank is [experts, out, in] with the expert axis '
                         'outermost, so each expert is a contiguous byte range '
                         '— 8 of 288 is ~60 MB instead of 2.2 GB. The result is '
                         'a valid N-expert bank for block-level tests, not the '
                         'real routing.')
    ap.add_argument('--repo', default='unsloth/GLM-5.3-Flash-GGUF')
    ap.add_argument('--quant', default='UD-IQ1_S')
    ap.add_argument('--shards', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true',
                    help='print what would be fetched and exit (reads only the header)')
    args = ap.parse_args()
    blocks = {int(b) for b in args.blocks.split(',') if b.strip()}

    base = f"https://huggingface.co/{args.repo}/resolve/main"
    stem = f"{base}/{args.quant}/GLM-5.3-Flash-{args.quant}"
    shard = f"{stem}-00002-of-{args.shards:05d}.gguf"
    meta_shard = f"{stem}-00001-of-{args.shards:05d}.gguf"
    out = args.out

    print('reading metadata from shard 1 ...', flush=True)
    _, kv_meta, _, _, _ = read_header(meta_shard)
    print('reading tensor index from shard 2 ...', flush=True)
    _, _, tensors, data_start, align = read_header(shard)

    def want(n):
        if n == 'output_norm.weight':
            return True
        if not n.startswith('blk.'):
            return False
        try:
            idx = int(n.split('.')[1])
        except (IndexError, ValueError):
            return False
        if idx not in blocks:
            return False
        # The routed expert banks are 2.2 GB for a single MoE block; everything
        # else in a block is small enough to be worth having.
        if '_exps.' in n:
            return args.with_experts or args.experts > 0
        return True

    sel = [t for t in tensors if want(t[0])]
    if not sel:
        raise SystemExit(f'no tensors matched blocks {sorted(blocks)} in this shard')
    print(f'{len(sel)} tensors selected')
    if args.experts:
        print(f'note: --experts {args.experts} slices the routed banks only. '
              'ffn_gate_inp / exp_probs_b still describe every expert; a '
              'consumer must narrow them to match.')

    # Drop the tokenizer arrays: 154880 tokens + 321649 merges are irrelevant to
    # a weights test and would triple the file's header.
    drop = ('tokenizer.ggml.tokens', 'tokenizer.ggml.token_type',
            'tokenizer.ggml.merges', 'tokenizer.chat_template')
    kv_out = {k: v for k, v in kv_meta.items() if k not in drop and not k.startswith('split.')}
    kv_out['general.name'] = ('s', 'GLM 5.3 Flash (blocks %s subset)' % ','.join(map(str, sorted(blocks))))

    # Header: recompute each tensor's offset into our own contiguous data blob.
    hdr = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', len(sel)) + struct.pack('<Q', len(kv_out))
    for k, v in kv_out.items():
        hdr += wstr(k) + wval(v)
    infos = b''
    off = 0
    plan = []
    for name, dims, typ, src_off in sel:
        # `--experts N`: keep the first N slabs and shrink the declared expert
        # axis (GGML's outermost dim is last). Each slab is contiguous, so this
        # is a prefix of the tensor's bytes.
        if '_exps.' in name and args.experts > 0 and len(dims) == 3:
            full = dims[2]
            keep = min(args.experts, full)
            dims = [dims[0], dims[1], keep]
            print(f'    {name}: {keep}/{full} experts')
        sz = nbytes(dims, typ)
        infos += wstr(name) + struct.pack('<I', len(dims))
        for d in dims: infos += struct.pack('<Q', d)
        infos += struct.pack('<I', typ) + struct.pack('<Q', off)
        plan.append((name, data_start + src_off, sz, off))
        off = (off + sz + align - 1) // align * align
    total = sum(sz for _, _, sz, _ in plan)
    print(f'{total/1e6:.1f} MB of tensor data')
    head = hdr + infos
    pad = (-len(head)) % align
    head += b'\x00' * pad
    blob_len = off

    # Coalesce adjacent source ranges so this is a handful of big GETs, not 57.
    plan.sort(key=lambda p: p[1])
    runs = []
    for name, s, sz, dst in plan:
        if runs and s == runs[-1][1]:
            runs[-1][1] = s + sz; runs[-1][2].append((name, s, sz, dst))
        else:
            runs.append([s, s + sz, [(name, s, sz, dst)]])
    print(f'{len(runs)} contiguous ranges to fetch')
    if args.dry_run:
        for name, _, sz, _ in sorted(plan, key=lambda p: -p[2])[:8]:
            print(f'    {name:44s} {sz/1e6:8.2f} MB')
        print(f'dry run: would write {out} '
              f'({(len(head) + blob_len)/1e6:.1f} MB)')
        return

    with open(out, 'wb') as f:
        f.write(head)
        f.truncate(len(head) + blob_len)
        done = 0
        for lo, hi, items in runs:
            req = urllib.request.Request(shard, headers={'Range': f'bytes={lo}-{hi-1}'})
            with urllib.request.urlopen(req) as r:
                buf = r.read()
            assert len(buf) == hi - lo, f'short read {len(buf)} != {hi-lo}'
            for name, s, sz, dst in items:
                f.seek(len(head) + dst)
                f.write(buf[s - lo: s - lo + sz])
            done += hi - lo
            print(f'  {done/1e6:8.1f} / {total/1e6:.1f} MB', flush=True)
    print(f'wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)')

main()
