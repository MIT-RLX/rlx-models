#!/usr/bin/env python3
"""HF transformers reference for rlx-nllb (NLLB-200 / M2M100).

Usage: python3 scripts/nllb_reference.py MODEL_DIR OUT.json [--beams 4] [--max-new 64]

Per case it records, from `M2M100ForConditionalGeneration` in f32 on CPU:

* `src_ids`         — tokenizer output (`[src_lang] + pieces + [eos]`)
* `encoder_hidden`  — final encoder output, flattened `[seq * d]`
* `greedy_ids`      — `generate(num_beams=1)` incl. decoder start + forced lang
* `beam_ids`        — `generate(num_beams=B, early_stopping=True)`
* `dec_hidden`      — teacher-forced decoder output over `greedy_ids[:-1]`
* `step_topk`       — per step, top-8 `(id, log_prob)` of the next token
* `greedy_text`     — decoded translation
"""

import argparse
import json

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CASES = [
    ("Hello, how are you?", "eng_Latn", "fra_Latn"),
    ("The weather is nice today, so we are going to the park.", "eng_Latn", "deu_Latn"),
    ("Je ne sais pas où se trouve la gare.", "fra_Latn", "eng_Latn"),
    ("Machine translation has improved a lot.", "eng_Latn", "zho_Hans"),
    ("Please send me the report before Friday.", "eng_Latn", "rus_Cyrl"),
    (
        "No Language Left Behind is a project that open-sources models capable of "
        "delivering high-quality translations directly between any pair of 200+ "
        "languages, including low-resource languages like Asturian, Luganda, Urdu and more.",
        "eng_Latn",
        "spa_Latn",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    model.eval()
    out = {"beams": args.beams, "max_new_tokens": args.max_new, "cases": []}
    for text, src, tgt in CASES:
        tok = AutoTokenizer.from_pretrained(args.model_dir, src_lang=src)
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"]
        tgt_id = tok.convert_tokens_to_ids(tgt)
        common = dict(
            forced_bos_token_id=tgt_id,
            max_new_tokens=args.max_new,
            do_sample=False,
        )
        greedy = model.generate(**enc, num_beams=1, **common)[0]
        beam = model.generate(
            **enc, num_beams=args.beams, early_stopping=True, length_penalty=1.0, **common
        )[0]
        enc_out = model.model.encoder(input_ids=ids).last_hidden_state[0]
        dec_in = greedy[:-1].unsqueeze(0)
        dec = model.model.decoder(
            input_ids=dec_in, encoder_hidden_states=enc_out.unsqueeze(0)
        ).last_hidden_state[0]
        logp = torch.log_softmax(model.lm_head(dec), dim=-1)
        topk = []
        for row in logp:
            v, i = row.topk(8)
            topk.append([[int(a), float(b)] for a, b in zip(i, v)])
        out["cases"].append(
            {
                "text": text,
                "src": src,
                "tgt": tgt,
                "tgt_id": int(tgt_id),
                "src_ids": ids[0].tolist(),
                "encoder_hidden": enc_out.flatten().tolist(),
                "greedy_ids": greedy.tolist(),
                "beam_ids": beam.tolist(),
                "dec_hidden": dec.flatten().tolist(),
                "step_topk": topk,
                "greedy_text": tok.decode(greedy, skip_special_tokens=True),
                "beam_text": tok.decode(beam, skip_special_tokens=True),
            }
        )
        print(f"{src}->{tgt}: {out['cases'][-1]['greedy_text']!r}")
    with open(args.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
