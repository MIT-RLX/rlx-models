#!/usr/bin/env python3
"""Reference dump for `rlx-pocket-tts` parity tests.

Runs the upstream `pocket_tts` package (kyutai-labs/pocket-tts) on a local
language directory and writes everything the Rust port compares against, per
text chunk:

    token ids, frames_after_eos, the flow-head noise of every generated frame,
    the generated latents (before de-normalization), the EOS logits, the
    backbone output fed to the flow head, and the decoded waveform.

The generation loop below is `TTSModel._generate_audio_stream_short_text`
without the threads, so the noise each frame consumed can be recorded and fed
back to the Rust side; the decode is one Mimi call over all of a chunk's
latents, which is what the streaming decoder computes frame by frame.

    .venv-pocket/bin/python -I scripts/pocket_tts_reference.py \
        --lang-dir weights/tts/pocket-tts-kyutai/languages/english_2026-10 \
        --language english_2026-10 --voice alba \
        --out weights/tts/pocket-tts-kyutai/reference/english_2026-10-alba

Needs the `pocket-tts` package (`uv pip install pocket-tts`, or from a checkout).
"""

import argparse
import copy
import json
import pathlib
import tempfile

import safetensors.torch
import scipy.io.wavfile
import torch
import yaml

import pocket_tts.utils.config as pt_config
from pocket_tts.default_parameters import MAX_TOKEN_PER_CHUNK
from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states


def local_config(language: str, lang_dir: pathlib.Path) -> pathlib.Path:
    """The upstream config with its weights and tokenizer pointed at `lang_dir`."""
    cfg = yaml.safe_load((pt_config.CONFIGS_DIR / f"{language}.yaml").read_text())
    cfg["weights_path"] = str((lang_dir / "model.safetensors").resolve())
    cfg.pop("weights_path_without_voice_cloning", None)
    tok = cfg["flow_lm"]["lookup_table"]
    # hf://<org>/<repo>/<path>@<rev> → <repo root>/<path>; the drifting model
    # reuses another language's tokenizer.
    rel = tok["tokenizer_path"].removeprefix("hf://").split("@")[0].split("/", 2)[2]
    local = lang_dir.parent.parent / rel
    if not local.exists():
        name = "tokenizer.json" if tok["tokenizer"] in ("tokenizers", "json") else "tokenizer.model"
        local = lang_dir / name
    tok["tokenizer_path"] = str(local.resolve())
    out = pathlib.Path(tempfile.mkdtemp()) / f"{language}.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang-dir", required=True)
    ap.add_argument("--language", required=True)
    ap.add_argument("--voice", default="alba")
    ap.add_argument(
        "--text",
        default="Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs.",
    )
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lang_dir = pathlib.Path(args.lang_dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model = TTSModel.load_model(config=local_config(args.language, lang_dir), temp=args.temp)
    model.eval()
    voice_path = lang_dir / "embeddings" / f"{args.voice}.safetensors"
    state0 = model.get_state_for_audio_prompt(str(voice_path))

    # Record every noise draw and every EOS-head call.
    noises: list[torch.Tensor] = []
    real_normal, real_trunc = torch.nn.init.normal_, torch.nn.init.trunc_normal_

    def normal_(t, *a, **k):
        r = real_normal(t, *a, **k)
        noises.append(r.detach().clone())
        return r

    def trunc_normal_(t, *a, **k):
        r = real_trunc(t, *a, **k)
        noises.append(r.detach().clone())
        return r

    torch.nn.init.normal_, torch.nn.init.trunc_normal_ = normal_, trunc_normal_
    eos_in: list[torch.Tensor] = []
    eos_out: list[torch.Tensor] = []

    def eos_hook(_m, i, o):
        eos_in.append(i[0].detach().clone())
        eos_out.append(o.detach().clone())

    model.flow_lm.out_eos.register_forward_hook(eos_hook)

    tok = model.flow_lm.conditioner.tokenizer
    chunks = split_into_best_sentences(
        tok,
        args.text,
        MAX_TOKEN_PER_CHUNK,
        model.pad_with_spaces_for_short_inputs,
        remove_semicolons=model.remove_semicolons,
        append_terminal_punctuation=model.append_terminal_punctuation,
        capitalize_first_letter=model.capitalize_first_letter,
        replace_characters=model.replace_characters,
    )
    tensors: dict[str, torch.Tensor] = {}
    meta = {
        "language": args.language,
        "voice": args.voice,
        "text": args.text,
        "temp": model.temp,
        "eos_threshold": model.eos_threshold,
        "chunks": [],
    }
    audio_all = []
    with torch.no_grad():
        for ci, chunk in enumerate(chunks):
            text, guess = prepare_text_prompt(
                chunk,
                model.pad_with_spaces_for_short_inputs,
                model.remove_semicolons,
                model.append_terminal_punctuation,
                model.capitalize_first_letter,
                model.replace_characters,
            )
            frames_after_eos = model.model_recommended_frames_after_eos
            if frames_after_eos is None:
                frames_after_eos = guess + 2
            prepared = model.flow_lm.conditioner.prepare(text)
            max_gen_len = model._estimate_max_gen_len(prepared.shape[1])
            st = copy.deepcopy(state0)
            end = model._flow_lm_current_end(st)
            model._expand_kv_cache(st, end + prepared.shape[1] + max_gen_len)

            noises.clear()
            eos_in.clear()
            eos_out.clear()
            model._run_flow_lm_and_increment_step(model_state=st, text_tokens=prepared)
            # The text prompt call also samples (and discards) a latent.
            noises.clear()
            eos_in.clear()
            eos_out.clear()

            backbone = torch.full((1, 1, model.flow_lm.ldim), float("nan"))
            latents, eos_step = [], None
            for step in range(max_gen_len):
                nxt, is_eos = model._run_flow_lm_and_increment_step(
                    model_state=st, backbone_input_latents=backbone
                )
                if is_eos.item() and eos_step is None and step >= model._MIN_FRAMES_BEFORE_EOS:
                    eos_step = step
                if eos_step is not None and step >= eos_step + frames_after_eos:
                    break
                latents.append(nxt)
                backbone = nxt
            n = len(latents)
            lat = torch.cat(latents, dim=1)  # [1, n, ldim]
            mimi_state = init_states(model.mimi, batch_size=1, sequence_length=max_gen_len * 16)
            emb = lat * model.flow_lm.emb_std + model.flow_lm.emb_mean
            audio = model.mimi.decode_from_latent(emb, mimi_state)[0, 0]
            raw = audio.clone()
            fade = torch.linspace(0, 1, model.config.mimi.sample_rate // 200)
            audio[: fade.numel()] *= fade
            audio_all.append(audio)

            p = f"chunk{ci}."
            tensors[p + "tokens"] = prepared[0].to(torch.int64).contiguous()
            # One noise per sampled frame (the frame that hit the stop rule also sampled one).
            tensors[p + "noise"] = torch.cat([z.reshape(1, -1) for z in noises], 0).contiguous()
            tensors[p + "latents"] = lat[0].contiguous()
            tensors[p + "eos_logit"] = torch.cat([e.reshape(1) for e in eos_out]).contiguous()
            tensors[p + "hidden"] = torch.cat([h.reshape(1, -1) for h in eos_in], 0).contiguous()
            tensors[p + "audio_raw"] = raw.contiguous()
            meta["chunks"].append(
                {
                    "text": text,
                    "tokens": prepared.shape[1],
                    "frames": n,
                    "max_gen_len": max_gen_len,
                    "frames_after_eos": frames_after_eos,
                    "eos_step": eos_step,
                    "voice_offset": end,
                }
            )
    tensors["audio"] = torch.cat(audio_all).contiguous()
    safetensors.torch.save_file(tensors, str(out / "reference.safetensors"))
    (out / "reference.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    scipy.io.wavfile.write(str(out / "reference.wav"), model.sample_rate, tensors["audio"].numpy())
    secs = tensors["audio"].numel() / model.sample_rate
    print(f"wrote {out} — {len(chunks)} chunk(s), {secs:.2f} s of audio")


if __name__ == "__main__":
    main()
