#!/usr/bin/env python3
"""Reference outputs for `crates/rlx-d1`, from LiquidAI's own d1 code.

Loads `LiquidAI/d1-3B` with `trust_remote_code=True` (its `modeling_d1.py`,
`runner.py`, `prompt.py`) in fp32 on CPU and, for every case in
`crates/rlx-d1/tests/fixtures/d1_cases.json`, records:

* `system_one` — the official API answer (tree-packed, as shipped);
* per question, the rendered prompt, its token ids and the option scores
  (log-softmax at the answer slot, max-pooled per option), each computed as a
  plain single-row pass;
* for cases with images, the processor outputs (`pixel_values`,
  `spatial_shapes`, `pixel_attention_mask`, `input_ids`) and the projected
  image features, in `<out>/vision_<case>.safetensors`.

    python3 scripts/d1_reference.py --model weights/lm/d1-3b \
        --images /tmp/d1img --out /tmp/d1ref

Missing case pictures are made in `--images` first: the COCO "two cats"
photo (val2017/000000039769) is downloaded, the two PNGs are drawn.

With `--gguf d1-3B-Q4_K_M.gguf` the text model's weights are replaced by
that file's, dequantized (the `gguf` package): LiquidAI's code then computes
the quantized model's function with float activations — the reference for
`tests/gguf_parity.rs`, independent of any GGUF runtime's kernels.

Needs `transformers>=5.14`, torch, torchvision, Pillow (and `gguf` for
`--gguf`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


COCO_CATS = "http://images.cocodataset.org/val2017/000000039769.jpg"


def make_images(d: Path) -> None:
    """The case pictures: a real photo (JPEG, single tile), a 1400x900 PNG
    (capped, then tiled with a thumbnail) and a 200x120 PNG (upscaled)."""
    import urllib.request

    from PIL import Image, ImageDraw

    d.mkdir(parents=True, exist_ok=True)
    if not (d / "cats.jpg").exists():
        urllib.request.urlretrieve(COCO_CATS, d / "cats.jpg")
    if not (d / "shapes_1400x900.png").exists():
        im = Image.new("RGB", (1400, 900))
        px = im.load()
        for y in range(900):
            for x in range(1400):
                px[x, y] = ((x * 255) // 1399, (y * 255) // 899, ((x + y) * 7) % 256)
        draw = ImageDraw.Draw(im)
        draw.ellipse((200, 150, 600, 550), fill=(220, 30, 30))
        draw.rectangle((800, 300, 1200, 700), fill=(30, 30, 220))
        im.save(d / "shapes_1400x900.png")
    if not (d / "small_200x120.png").exists():
        small = Image.new("RGB", (200, 120), (10, 200, 40))
        ImageDraw.Draw(small).rectangle((50, 30, 150, 90), fill=(250, 250, 0))
        small.save(d / "small_200x120.png")


# llama.cpp `lfm2` tensor names -> HF names under `model.language_model.`
# (the reverse of `conversion/lfm2.py` + llama.cpp's tensor map).
GGUF_TO_HF = {
    "attn_norm": "operator_norm",
    "ffn_norm": "ffn_norm",
    "shortconv.in_proj": "conv.in_proj",
    "shortconv.conv": "conv.conv",
    "shortconv.out_proj": "conv.out_proj",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.out_proj",
    "attn_q_norm": "self_attn.q_layernorm",
    "attn_k_norm": "self_attn.k_layernorm",
    "ffn_gate": "feed_forward.w1",
    "ffn_up": "feed_forward.w3",
    "ffn_down": "feed_forward.w2",
}


def load_gguf_text_weights(model, path: str) -> None:
    """Overwrite the text model's parameters with a GGUF's, dequantized."""
    import gguf

    reader = gguf.GGUFReader(path)
    params = dict(model.named_parameters())
    lm = "model.language_model."
    done = set()
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            hf = lm + "embed_tokens.weight"
        elif t.name == "token_embd_norm.weight":
            hf = lm + "embedding_norm.weight"
        else:
            _, i, rest = t.name.split(".", 2)
            part = rest.rsplit(".", 1)[0]
            hf = f"{lm}layers.{i}.{GGUF_TO_HF[part]}.weight"
        w = torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()
        dst = params[hf]
        dst.data.copy_(w.reshape(dst.shape))
        done.add(hf)
    missing = [n for n in params if n.startswith(lm) and n not in done]
    assert not missing, f"not in the GGUF: {missing[:5]}"
    print(f"[d1-ref] text weights from {path} ({len(done)} tensors, dequantized)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=str(Path(__file__).resolve().parent.parent
                                           / "crates/rlx-d1/tests/fixtures/d1_cases.json"))
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--gguf", default="", help="text GGUF whose dequantized weights replace the HF ones")
    args = ap.parse_args()

    from PIL import Image
    from safetensors.torch import save_file
    from transformers import AutoModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    make_images(Path(args.images))
    cases = json.loads(Path(args.cases).read_text())
    only = {s for s in args.only.split(",") if s}

    torch.manual_seed(0)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    if args.gguf:
        load_gguf_text_weights(model, args.gguf)
    engine = model.engine
    # The engine's own modules, so the prompt code is exactly the shipped one.
    prompt = sys.modules[type(engine).__module__.replace("runner", "prompt")]
    runner = sys.modules[type(engine).__module__]

    results = []
    for case in cases:
        if only and case["name"] not in only:
            continue
        print(f"[d1-ref] {case['name']}", flush=True)
        images = [Image.open(Path(args.images) / p).convert("RGB") for p in case.get("images", [])]
        official = model.system_one(case["state"], case["questions"], images=images or None)

        rows = []
        with torch.inference_mode():
            pics = [runner.cap_pixels(im) for im in images]
            markup = engine._image_markup(len(pics)) if pics else ""
            prefix = prompt.prefix_text(engine.tokenizer, case["state"], engine.bos, engine.state_style,
                                        engine.system, markup)
            for name, qd in case["questions"].items():
                q = prompt.as_question(qd)
                suffix = prompt.suffix_text(engine.tokenizer, q, engine.lead, engine.option_style)
                text = prefix + suffix
                if pics:
                    inputs = engine._image_inputs(text, pics)
                    ids = inputs["input_ids"][0].tolist()
                    row = model(**inputs, logits_to_keep=1).logits[0, -1].float()
                else:
                    ids = engine.tokenizer.encode(text, add_special_tokens=False)
                    row = model(input_ids=torch.tensor([ids]), logits_to_keep=1).logits[0, -1].float()
                logz = row - torch.logsumexp(row, dim=-1)
                groups = prompt.readout_ids(engine.tokenizer, q)
                rows.append({
                    "question": name,
                    "text": text,
                    "ids": ids,
                    "groups": groups,
                    "group_logz": [[float(logz[i]) for i in g] for g in groups],
                    "probs": prompt.readout(engine.tokenizer, q, logz),
                    "top_logit_ids": torch.topk(row, 5).indices.tolist(),
                })

            if pics:
                inputs = engine._image_inputs(prefix, pics)
                feats = torch.cat(model.model.get_image_features(
                    pixel_values=inputs["pixel_values"], spatial_shapes=inputs["spatial_shapes"],
                    pixel_attention_mask=inputs["pixel_attention_mask"]).pooler_output)
                save_file({
                    "pixel_values": inputs["pixel_values"].float().contiguous(),
                    "spatial_shapes": inputs["spatial_shapes"].to(torch.int64).contiguous(),
                    "pixel_attention_mask": inputs["pixel_attention_mask"].to(torch.int64).contiguous(),
                    "input_ids": inputs["input_ids"].to(torch.int64).contiguous(),
                    "image_features": feats.float().contiguous(),
                    "capped_rgb": torch.tensor(list(pics[0].tobytes()), dtype=torch.uint8),
                    "capped_size": torch.tensor(list(pics[0].size), dtype=torch.int64),
                }, str(out / f"vision_{case['name']}.safetensors"))

        results.append({"name": case["name"], "system_one": official, "rows": rows})

    (out / "ref.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"[d1-ref] wrote {out / 'ref.json'}")


if __name__ == "__main__":
    main()
