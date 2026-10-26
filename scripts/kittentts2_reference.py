#!/usr/bin/env python3
"""Reference dump for `rlx-kittentts2` parity tests.

Runs the upstream `kittenml` package (KittenTTS 2 speech LM + chatterbox-turbo
S3Gen) in float32 and writes every intermediate the Rust port compares against:

    prompt ids, speaker-projection output, greedy LM tokens + first logits,
    S3Gen encoder output, meanflow mel (with the noise it consumed), HiFT f0,
    the deterministic sine source, and the final waveform.

Randomness is pinned so every stage is reproducible from the dump alone: the
flow noise is saved and fed back, and the HiFT sine source is run with zero
phase and zero additive noise (the Rust side exposes the same switch).

    .venv-kitten2/bin/python scripts/kittentts2_reference.py \
        --repo weights/tts/kitten-tts-2 \
        --s3gen weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors \
        --out weights/tts/kitten-tts-2/reference

Needs: torch, torchaudio, transformers<5, safetensors, librosa, diffusers,
einops, omegaconf, s3tokenizer, kittenml==0.9.6 (installed --no-deps),
kitten-text-processing.
"""

import argparse
import json
import pathlib

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--s3gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default="Bruno")
    ap.add_argument("--text", default="Hello there, how are you today?")
    ap.add_argument("--greedy-tokens", type=int, default=48)
    ap.add_argument("--audio-tokens-json", default=None,
                    help="run S3Gen on these codec tokens (first chunk of an "
                         "`rlx-kittentts2 --dump-tokens` file) instead of the greedy ones")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # S3Codec fetches the s3gen weights from the hub; point it at the local file.
    import kittenml.kittentts2.vocoder as vocoder
    vocoder.hf_hub_download = lambda **_: args.s3gen

    from kittenml.kittentts2.model import KittenTTS2
    from kittenml.kittentts2.prompt import build_generation_prompt
    from kittenml.kittentts2.logits import build_logits_processors
    from kittenml.kittentts2 import text as text_utils

    repo = pathlib.Path(args.repo)
    config = json.loads((repo / "config.json").read_text())
    config["dtype"] = "fp32"
    m = KittenTTS2(repo, config, device="cpu", backend="torch")
    # The packed checkpoint installs its bf16 tensors as-is (`assign=True`);
    # the ternary values are exact in f32, so widen for an f32 reference.
    m.model.float()
    tm = m.token_map

    dump = {}
    meta = {"voice": args.voice, "text": args.text}

    # ── prompt ────────────────────────────────────────────────────────────────
    ref = m._voice_reference(args.voice)
    spoken = text_utils.normalize_text(args.text, normalize=False)
    meta["spoken"] = spoken
    target_ids = m.tokenizer.encode(spoken, add_special_tokens=False)
    prompt = build_generation_prompt(tm, target_ids, reference_prefix=ref["prefix"])
    meta["prompt_ids"] = prompt
    meta["target_text_ids"] = target_ids
    meta["transcript_ids"] = m.tokenizer.encode(ref["transcript"].strip(),
                                                add_special_tokens=False)

    emb = ref["embedding"].float()               # [1, 512] raw pyannote embedding
    dump["speaker_embedding"] = emb.float().numpy()
    with torch.no_grad():
        dump["speaker_projected"] = m.model.spk_proj(emb).float().numpy()

    # ── LM: first-step logits and a greedy continuation ───────────────────────
    ids = torch.tensor([prompt])
    m.model._pending_speaker_embedding = emb
    with torch.no_grad():
        logits = m.model(input_ids=ids).logits[0, -1].float()
    dump["first_logits"] = logits.numpy()

    processors = build_logits_processors(
        tm, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
        repetition_penalty=1.1, repetition_window=config["generation"]["repetition_window"],
        token_run_penalty=1.3, token_run_grace=10, prompt_length=len(prompt))
    with torch.no_grad():
        gen = m.model.generate(
            ids, max_new_tokens=args.greedy_tokens, do_sample=False,
            temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
            logits_processor=processors,
            eos_token_id=[tm.speech_end_id, tm.stop_id], pad_token_id=0)
    m.model._pending_speaker_embedding = None
    greedy = gen[0].tolist()[len(prompt):]
    meta["greedy_tokens"] = greedy

    # ── S3Gen on the greedy audio tokens ──────────────────────────────────────
    lo, hi = tm.audio_id_base, tm.audio_id_end
    audio = [t - lo for t in greedy if lo <= t < hi]
    if args.audio_tokens_json:
        audio = json.loads(pathlib.Path(args.audio_tokens_json).read_text())[0]
    meta["audio_tokens"] = audio
    codec = m.codec
    s3 = codec.s3gen
    cond = ref["conditioning"]
    for k in ("prompt_token", "prompt_feat", "embedding"):
        dump["cond_" + k] = cond[k].float().numpy()

    from kittenml.kittentts2.s3gen import S3GEN_SIL
    tokens = torch.tensor(audio + [int(S3GEN_SIL)] * 3).long()
    meta["s3gen_tokens"] = tokens.tolist()
    flow = s3.flow
    with torch.inference_mode():
        speech = tokens[None]
        token_len = torch.tensor([speech.shape[1]])
        e = torch.nn.functional.normalize(torch.atleast_2d(cond["embedding"]), dim=1)
        spks = flow.spk_embed_affine_layer(e)
        dump["flow_spks"] = spks.numpy()
        all_tok = torch.cat([cond["prompt_token"], speech], dim=1)
        all_len = cond["prompt_token_len"] + token_len
        x = flow.input_embedding(all_tok.long())
        h, h_masks = flow.encoder(x, all_len)
        mu = flow.encoder_proj(h).transpose(1, 2)
        dump["flow_mu"] = mu.numpy()
        mel_len1 = cond["prompt_feat"].shape[1]
        conds = torch.zeros_like(mu)
        conds[:, :, :mel_len1] = cond["prompt_feat"].transpose(1, 2)
        z = torch.randn_like(mu)
        dump["flow_noise"] = z.numpy()
        mask = torch.ones(1, 1, mu.shape[2])
        est = flow.decoder.estimator
        t_span = torch.linspace(0, 1, 3)
        xk = z
        for i, (t, r) in enumerate(zip(t_span[:-1], t_span[1:])):
            v = est.forward(xk, mask=mask, mu=mu, t=t[None], spks=spks, cond=conds, r=r[None])
            dump[f"flow_v{i}"] = v.numpy()
            xk = xk + (r - t) * v
        mel = xk[:, :, mel_len1:]
        dump["flow_mel"] = mel.numpy()
        meta["mel_len1"] = int(mel_len1)

        hift = s3.mel2wav
        f0 = hift.f0_predictor(mel)
        dump["hift_f0"] = f0.numpy()
        # Deterministic source: zero random phase, zero additive noise.
        up = hift.f0_upsamp(f0[:, None]).transpose(1, 2)            # [1, L, 1]
        sg = hift.m_source.l_sin_gen
        fm = torch.zeros(1, sg.harmonic_num + 1, up.shape[1])
        for i in range(sg.harmonic_num + 1):
            fm[:, i:i + 1, :] = up.transpose(1, 2) * (i + 1) / sg.sampling_rate
        theta = 2 * np.pi * (torch.cumsum(fm, dim=-1) % 1)
        sine = sg.sine_amp * torch.sin(theta)
        uv = (up.transpose(1, 2) > sg.voiced_threshold).float()
        sine = (sine * uv).transpose(1, 2)                            # [1, L, 9]
        source = hift.m_source.l_tanh(hift.m_source.l_linear(sine)).transpose(1, 2)
        dump["hift_source"] = source.numpy()
        wav = hift.decode(x=mel, s=source)
        dump["hift_wav"] = wav.numpy()
        wav_faded = wav.clone()
        wav_faded[:, :len(s3.trim_fade)] *= s3.trim_fade
        dump["wav"] = wav_faded.numpy()

    # C order: the Rust reader takes row-major arrays only.
    np.savez(out / "reference.npz", **{k: np.ascontiguousarray(v) for k, v in dump.items()})
    (out / "reference.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    import soundfile as sf
    sf.write(str(out / "reference.wav"), dump["wav"].reshape(-1), 24000)
    print(f"wrote {out}: {len(prompt)} prompt ids, {len(greedy)} greedy tokens, "
          f"{len(audio)} audio tokens, mel {dump['flow_mel'].shape}, wav {dump['wav'].size}")


if __name__ == "__main__":
    main()
