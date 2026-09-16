#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Transliterate the Ooura split-radix FFT out of TEN-VAD's `src/fftw.c` into
Rust, for `crates/rlx-ten-vad/src/ooura.rs`.

`rlx-ten-vad` is bit-identical to the TEN-VAD C frontend, and that requires the
*same* butterflies applied in the *same* order: `f32` addition is not
associative, so a mathematically-equivalent FFT is not enough. Hand-porting two
thousand lines of unrolled index arithmetic would not survive review, hence this.

The C subset in play is small — `float`/`int` locals, `for`/`while`/`if`, array
indexing, and calls passing `&arr[k]` — so the translation is purely syntactic:

* pointer params become `(slice, offset)` pairs, `a[i]` → `a[(ao + (i)) as usize]`
* `for (I; C; S) { B }` → `I; while C { B S; }` (there is no `continue`/`goto`)
* `x++` → `x += 1`, `if (c)` → `if c`, `wd1r = 1;` → `wd1r = 1.0;`

Only the n = 1024 path is emitted, which is all TEN-VAD uses. The driver
(`rdft`, `r2c`, `c2r`, `power_spectrum`, …) and the `IP`/`W` tables are
hand-written in `ooura.rs` around this output.

Usage::

    python3 scripts/transpile_ooura.py /path/to/ten-vad/src/fftw.c > /tmp/gen.rs
    # splice /tmp/gen.rs between the tables and the driver in ooura.rs
    cargo fmt -p rlx-ten-vad
    cargo test -p rlx-ten-vad --test reference_parity   # must stay bit-identical

Upstream: Takuya Ooura's `fftsg.c`, (C) 1996-2001, free for any use.
"""

import re, sys

SRC = open(sys.argv[1]).read()
NEED = ['cftf1st', 'cftb1st', 'cftrec4', 'cfttree', 'cftleaf', 'cftmdl1', 'cftmdl2',
        'cftf161', 'cftf162', 'cftf081', 'cftf082', 'bitrv2', 'bitrv2conj',
        'rftfsub', 'rftbsub']
PTR = {'a': ('a', '&mut [f32]'), 'w': ('w', '&[f32]'), 'c': ('c', '&[f32]'), 'ip': ('ip', '&[i32]')}

def extract(name):
    pat = re.compile(r'^static (?:void|int) AUP_FFTW_' + name + r'\(([^)]*)\)\s*\{\n(.*?)^\}\n',
                     re.S | re.M)
    m = pat.search(SRC)
    if not m:
        sys.exit(f'{name} not found')
    return m.group(1), m.group(2)

def sig(name, params):
    out, arrays = [], []
    for p in [x.strip() for x in params.split(',')]:
        if p.startswith('float*') or p.startswith('int*'):
            nm = p.split('*')[1].strip()
            arrays.append(nm)
            out.append(f'{nm}: {PTR[nm][1]}')
            out.append(f'{nm}o: i32')
        else:
            out.append(f'{p.split()[1]}: i32')
    ret = ' -> i32' if name == 'cfttree' else ''
    return f'fn {name}({", ".join(out)}){ret}', arrays

def float_locals(body):
    names = []
    for m in re.finditer(r'^[ \t]*float[ \t]+([A-Za-z_][\w,\s]*?);', body, flags=re.M):
        names += [n.strip() for n in m.group(1).split(',')]
    return names

def float_int_literals(body, names):
    """C promotes `wd1r = 1;` silently; Rust does not."""
    for n in names:
        body = re.sub(r'\b' + n + r' = (-?\d+);', lambda m, v=n: f'{v} = {m.group(1)}.0;', body)
    return body

def decls(body):
    def repl(m):
        ty, names = m.group(1), m.group(2)
        rty = 'f32' if ty == 'float' else 'i32'
        return '\n'.join(f'let mut {n.strip()}: {rty};' for n in names.split(','))
    # declaration lists wrap across lines in the wider functions
    return re.sub(r'^[ \t]*(int|float)[ \t]+([A-Za-z_][\w,\s]*?);', repl, body, flags=re.M)

def for_to_while(body):
    """`for (I; C; S) { B }` -> `I; while C { B S; }` (no continue/goto here)."""
    while True:
        m = re.search(r'\bfor \(([^;]*);([^;]*);([^)]*)\)\s*\{', body)
        if not m:
            return body
        init, cond, step = (x.strip() for x in m.groups())
        depth, i = 1, m.end()
        while depth:
            if body[i] == '{': depth += 1
            elif body[i] == '}': depth -= 1
            i += 1
        inner = body[m.end():i - 1]
        head = f'{init};\n' if init else ''
        body = body[:m.start()] + f'{head}while {cond} {{{inner}{step};\n}}' + body[i:]

def calls(body, arrays):
    def repl(m):
        fn, argstr = m.group(1), m.group(2)
        out = []
        for arg in [x.strip() for x in argstr.split(',')]:
            am = re.fullmatch(r'&(\w+)\[(.*)\]', arg)
            if am and am.group(1) in arrays:
                out += [am.group(1), f'{am.group(1)}o + ({am.group(2)})']
            elif arg in arrays:
                out += [arg, f'{arg}o']
            else:
                out.append(arg)
        return f'{fn}({", ".join(out)})'
    # single-line calls only, which is all this code has
    return re.sub(r'AUP_FFTW_(\w+)\(([^;]*?)\)(?=;)', repl, body)

def subscripts(body, arrays):
    for nm in arrays:
        body = re.sub(r'\b' + nm + r'\[([^\]]*)\]',
                      lambda m, n=nm: f'{n}[({n}o + ({m.group(1)})) as usize]', body)
    return body

def strip_cond_parens(body):
    """`if (C) {` -> `if C {` — Rust rejects the redundant parentheses."""
    out, i = [], 0
    while i < len(body):
        m = re.compile(r'\b(if|while) \(').search(body, i)
        if not m:
            out.append(body[i:])
            break
        out.append(body[i:m.start()])
        depth, j = 1, m.end()
        while depth:
            if body[j] == '(': depth += 1
            elif body[j] == ')': depth -= 1
            j += 1
        out.append(f'{m.group(1)} {body[m.end():j - 1]}')
        i = j
    return ''.join(out)

def incdec(body):
    body = re.sub(r'\b(\w+)\+\+', r'\1 += 1', body)
    return re.sub(r'\b(\w+)--', r'\1 -= 1', body)

def literals(body):
    body = re.sub(r'(\d)f\b', r'\1f32', body)                 # 0.5f  -> 0.5f32
    body = re.sub(r'(\d+\.\d+e[-+]?\d+)f32', r'\1_f32', body) # 1.0e-1f32 -> ok
    return body

print('// @generated by scripts/transpile_ooura.py — do not edit by hand.')
for name in NEED:
    params, body = extract(name)
    signature, arrays = sig(name, params)
    b = re.sub(r'//.*', '', body)
    fl = float_locals(b)
    b = decls(b)
    b = float_int_literals(b, fl)
    b = for_to_while(b)
    b = incdec(b)
    b = strip_cond_parens(b)
    b = calls(b, arrays)
    b = subscripts(b, arrays)
    b = literals(b)
    b = b.replace('return isplt;', 'isplt')
    print(f'{signature} {{\n{b}}}\n')
