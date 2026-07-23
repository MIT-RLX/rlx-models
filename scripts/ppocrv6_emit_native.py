#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Decompose PP-OCRv6 ONNX → native Rust HIR builders inside rlx-ppocrv6,
# then parameterize spatial dims (height/width) for variable input sizes.
#
# Example:
#   python3 scripts/ppocrv6_emit_native.py --tier tiny --task det \
#     --onnx .cache/ppocrv6/tiny/det/inference_rlx.onnx --h 96 --w 320

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def staticize(onnx: Path, h: int, w: int, batch: int = 1) -> Path:
    dest = onnx.with_name(f"{onnx.stem}_n{batch}_c3_{h}x{w}.onnx")
    if not dest.is_file():
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "ppocrv6_static_onnx.py"),
                str(onnx),
                str(dest),
                "--n",
                str(batch),
                "--c",
                "3",
                "--h",
                str(h),
                "--w",
                str(w),
            ],
            cwd=str(ROOT),
        )
    return dest


def decompose(static_onnx: Path, out: Path, crate_name: str, seq_len: int) -> None:
    if out.exists():
        shutil.rmtree(out)
    decomp = ROOT / "target" / "release" / "rlx-onnx-decompose"
    if not decomp.is_file():
        subprocess.check_call(
            ["cargo", "build", "-p", "rlx-onnx-decompose", "--release"],
            cwd=str(ROOT),
        )
    subprocess.check_call(
        [
            str(decomp),
            str(static_onnx),
            "-o",
            str(out),
            "--crate-name",
            crate_name,
            "--seq-len",
            str(seq_len),
        ],
        cwd=str(ROOT),
    )


def parameterize_graph_rs(src: Path, dest: Path, ref_h: int, ref_w: int) -> None:
    text = src.read_text()
    # GraphOptions: add height/width.
    text = text.replace(
        """pub struct GraphOptions {
    pub sequence_length: usize,
    pub max_waveform_samples: usize,
}""",
        """pub struct GraphOptions {
    pub sequence_length: usize,
    pub max_waveform_samples: usize,
    pub height: usize,
    pub width: usize,
}""",
    )
    text = text.replace(
        """Self {
            sequence_length: 128,
            max_waveform_samples: 24_000,
        }""",
        f"""Self {{
            sequence_length: 128,
            max_waveform_samples: 24_000,
            height: {ref_h},
            width: {ref_w},
        }}""",
    )
    # Input tensor uses opts.
    text = re.sub(
        rf'm\.input\("x", Shape::new\(&\[1, 3, {ref_h}, {ref_w}\], DType::F32\)\)',
        'm.input("x", Shape::new(&[1, 3, opts.height, opts.width], DType::F32))',
        text,
    )
    # Keep ONNX meta shapes (incl. ConvTranspose). Remap H/W via shape_from_meta
    # so variable sizes still work; do not force conv2d_output_shape over meta.
    # Remap H/W axes inside shape_from_meta.
    helper = f"""
fn map_spatial(v: usize, ref_v: usize, actual: usize) -> usize {{
    if v == 0 || ref_v == 0 {{
        return v;
    }}
    if v == ref_v {{
        return actual;
    }}
    let mut r = ref_v;
    let mut a = actual;
    while r > 1 {{
        if v == r + 1 {{
            return a + 1;
        }}
        if r % 2 != 0 {{
            break;
        }}
        r /= 2;
        a /= 2;
        if r == v {{
            return a;
        }}
    }}
    v
}}

fn shape_from_meta(meta: &serde_json::Value, opts: &GraphOptions) -> Shape {{
    let obj = meta.as_object().expect("shape meta");
    let shape_v = obj.get("shape").expect("shape");
    let dtype_s = obj.get("dtype").and_then(|d| d.as_str()).unwrap_or("f32");
    let mut dims: Vec<usize> = match shape_v {{
        serde_json::Value::Array(a) => a.iter().map(|d| resolve_dim(d, opts)).collect(),
        _ => vec![1],
    }};
    // Remap NCHW spatial axes from the reference staticize size.
    // Rank-3: CTC [N,T,C] (large C), SVTR [N,T,C] (T on width pyramid), or [N,C,W].
    if dims.len() == 4 {{
        dims[2] = map_spatial(dims[2], {ref_h}, opts.height);
        dims[3] = map_spatial(dims[3], {ref_w}, opts.width);
    }} else if dims.len() == 3 {{
        // CTC logits [N,T,C] (large C): remap T.
        // Feature [N,C,W] (C often equals ref width, e.g. 160): remap W only.
        // SVTR [N,T,C] (moderate C): remap T when W-axis does not move.
        if dims[2] > 1000 {{
            dims[1] = map_spatial(dims[1], {ref_w}, opts.width);
        }} else {{
            let last_w = map_spatial(dims[2], {ref_w}, opts.width);
            let mid_w = map_spatial(dims[1], {ref_w}, opts.width);
            if last_w != dims[2] {{
                dims[2] = last_w;
            }} else if mid_w != dims[1] {{
                dims[1] = mid_w;
            }}
        }}
    }}
    let dtype = match dtype_s {{
        "f32" => DType::F32,
        "i64" => DType::I64,
        "i32" => DType::I32,
        "bool" => DType::Bool,
        _ => DType::F32,
    }};
    Shape::new(&dims, dtype)
}}
"""
    # Replace the existing shape_from_meta function body.
    text = re.sub(
        r"fn shape_from_meta\(meta: &serde_json::Value, opts: &GraphOptions\) -> Shape \{.*?\n\}",
        helper.strip(),
        text,
        count=1,
        flags=re.S,
    )
    # Nested module: weights live in super::weights, not crate::weights.
    text = text.replace("crate::weights::", "super::weights::")
    # mean(axis=-1) → last axis
    text = text.replace(
        "m.mean(x, vec![-1], true)",
        "m.mean(x, vec![m.shape(x).rank().saturating_sub(1)], true)",
    )
    # Promote rank-0 scalar binds to [1] (rlx has no rank-0).
    text = text.replace(", &[], data.clone())", ", &[1], data.clone())")
    text = text.replace("bind_param(&key, &[], vec![", "bind_param(&key, &[1], vec![")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)


def copy_weights_rs(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def write_mod_rs(path: Path, mods: list[str]) -> None:
    body = "\n".join(f"pub mod {m};" for m in mods)
    path.write_text(
        "// RLX — versatile ML compiler + runtime.\n"
        "// AUTO-GENERATED native PP-OCRv6 graphs (see scripts/ppocrv6_emit_native.py).\n\n"
        f"{body}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=["tiny", "small"], required=True)
    ap.add_argument("--task", choices=["det", "rec"], required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    args = ap.parse_args()

    static = staticize(args.onnx, args.h, args.w)
    crate = f"ppocrv6_{args.tier}_{args.task}_rlx"
    tmp = Path("/tmp") / "ppocrv6_native" / crate
    decompose(static, tmp, crate, seq_len=max(args.h, args.w))

    out_dir = ROOT / "crates" / "rlx-ppocrv6" / "src" / "native" / f"{args.tier}_{args.task}"
    parameterize_graph_rs(tmp / "src" / "graph.rs", out_dir / "graph.rs", args.h, args.w)
    copy_weights_rs(tmp / "src" / "weights.rs", out_dir / "weights.rs")
    (out_dir / "mod.rs").write_text(
        f"""// RLX — versatile ML compiler + runtime.
// Native {args.tier} {args.task} graph (decomposed + spatial-parameterized).

pub mod graph;
pub mod weights;

pub use graph::{{GraphOptions, build_hir}};
pub use weights::{{LoadedWeights, load_weights}};

pub const REF_HEIGHT: usize = {args.h};
pub const REF_WIDTH: usize = {args.w};
"""
    )
    # Copy safetensors next to model cache expectation is separate; also stash under assets.
    st_src = tmp / "weights" / "model.safetensors"
    st_dst = (
        ROOT
        / ".cache"
        / "ppocrv6"
        / args.tier
        / args.task
        / f"ppocrv6_{args.tier}_{args.task}.safetensors"
    )
    if st_src.is_file():
        shutil.copy2(st_src, st_dst)
        print(f"weights -> {st_dst}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
