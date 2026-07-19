#!/usr/bin/env python3
"""Export Parler-TTS to ONNX for the RLX port: T5 encoder + decoder step.

The DAC codec is handled natively by `rlx-dac`, so only the T5 encoder and the
autoregressive decoder step are exported. The delay-pattern generation loop lives
in Rust.

True Parler semantics:
  - description  → T5 `text_encoder` (`input_ids`)
  - transcript   → `embed_prompts(prompt_input_ids)` prefix on the decoder

Pass `--with-prompt` to export the decoder with `prompt_input_ids` (preferred).
Without it, the legacy 3-input decoder matches the existing published ONNX
(transcript must be fed through the encoder as a temporary workaround).

  python export_onnx.py weights/tts/parlertts weights/tts/parlertts/onnx
  python export_onnx.py weights/tts/parlertts weights/tts/parlertts/onnx --with-prompt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from parler_tts import ParlerTTSForConditionalGeneration


def main(model_dir: str, out_dir: str, *, with_prompt: bool) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = ParlerTTSForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float32
    ).eval()
    d_model = m.text_encoder.config.d_model
    dec = m.decoder  # ParlerTTSForCausalLM
    H = dec.config.hidden_size
    n_layers = dec.config.num_hidden_layers
    n_heads = dec.config.num_attention_heads
    n_cb = dec.config.num_codebooks
    hd = H // n_heads

    # ---- T5 encoder (description) ----
    class Enc(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = m.text_encoder

        def forward(s, input_ids, attention_mask):
            return s.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    ids = torch.ones(1, 12, dtype=torch.long)
    mask = torch.ones(1, 12, dtype=torch.long)
    torch.onnx.export(
        Enc().eval(),
        (ids, mask),
        str(out / "text_encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "input_ids": {0: "b", 1: "t"},
            "attention_mask": {0: "b", 1: "t"},
            "encoder_hidden_states": {0: "b", 1: "t"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"exported text_encoder.onnx (d_model={d_model})")

    # ---- decoder: delay-pattern codes (+ optional prompt prefix) → logits ----
    if with_prompt:

        class Dec(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.dec = dec
                s.embed_prompts = m.embed_prompts

            def forward(
                s,
                decoder_input_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                prompt_input_ids,
            ):
                prompt_hidden = s.embed_prompts(prompt_input_ids)
                out = s.dec(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    prompt_hidden_states=prompt_hidden,
                    use_cache=False,
                    return_dict=True,
                )
                return out.logits

        dec_ids = torch.zeros(1, n_cb, 4, dtype=torch.long)
        enc_hs = torch.randn(1, 12, d_model)
        enc_mask = torch.ones(1, 12, dtype=torch.long)
        prompt_ids = torch.ones(1, 8, dtype=torch.long)
        torch.onnx.export(
            Dec().eval(),
            (dec_ids, enc_hs, enc_mask, prompt_ids),
            str(out / "decoder.onnx"),
            input_names=[
                "decoder_input_ids",
                "encoder_hidden_states",
                "encoder_attention_mask",
                "prompt_input_ids",
            ],
            output_names=["logits"],
            dynamic_axes={
                "decoder_input_ids": {0: "b", 2: "t"},
                "encoder_hidden_states": {0: "b", 1: "et"},
                "encoder_attention_mask": {0: "b", 1: "et"},
                "prompt_input_ids": {0: "b", 1: "pt"},
                "logits": {0: "b", 2: "t"},
            },
            opset_version=17,
            dynamo=False,
        )
        print(
            f"exported decoder.onnx WITH prompt "
            f"(H={H} layers={n_layers} heads={n_heads} n_cb={n_cb} hd={hd})"
        )
    else:

        class Dec(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.dec = dec

            def forward(s, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
                out = s.dec(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                return out.logits

        dec_ids = torch.zeros(1, n_cb, 4, dtype=torch.long)
        enc_hs = torch.randn(1, 12, d_model)
        enc_mask = torch.ones(1, 12, dtype=torch.long)
        torch.onnx.export(
            Dec().eval(),
            (dec_ids, enc_hs, enc_mask),
            str(out / "decoder.onnx"),
            input_names=[
                "decoder_input_ids",
                "encoder_hidden_states",
                "encoder_attention_mask",
            ],
            output_names=["logits"],
            dynamic_axes={
                "decoder_input_ids": {0: "b", 2: "t"},
                "encoder_hidden_states": {0: "b", 1: "et"},
                "encoder_attention_mask": {0: "b", 1: "et"},
                "logits": {0: "b", 2: "t"},
            },
            opset_version=17,
            dynamo=False,
        )
        print(
            f"exported decoder.onnx (legacy, no prompt) "
            f"(H={H} layers={n_layers} heads={n_heads} n_cb={n_cb} hd={hd})"
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], with_prompt="--with-prompt" in sys.argv[3:])
