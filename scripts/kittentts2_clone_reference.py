#!/usr/bin/env python3
"""Reference dump for `rlx-kittentts2` voice cloning (a new reference clip).

Runs the upstream conditioning path on one wav and writes every stage the
Rust port compares against, with the resampled audio each stage consumed so
the networks can be checked independently of the resamplers:

    LM side      wav16 (librosa) → S3 tokenizer log-mel → encoder → tokens
                 wav16 (torchaudio) → pyannote x-vector (512)
    S3Gen side   wav24 (librosa, ≤ 10 s, tiled to ≥ 6 s) → 24 kHz log-mel
                 wav24 → 16 kHz (torchaudio) → Kaldi fbank → CAM++ (192)
                                              → S3 tokenizer → prompt tokens

    .venv-kitten2/bin/python scripts/kittentts2_clone_reference.py \
        --repo weights/tts/kitten-tts-2 \
        --s3gen weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors \
        --wav weights/tts/kitten-tts-2/voices/<clip>.wav \
        --out weights/tts/kitten-tts-2/clone_reference
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
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import librosa
    from safetensors.torch import load_file

    import kittenml.kittentts2.vocoder as vocoder
    from kittenml.kittentts2.s3gen import S3_SR, S3GEN_SR, S3Gen, get_resampler
    from kittenml.kittentts2.speaker import _load_waveform
    from kittenml.kittentts2.speaker_embedding import load_speaker_embedding_model
    from kittenml.kittentts2.s3gen import extract_feature

    s3 = S3Gen(meanflow=True)
    s3.load_state_dict(load_file(args.s3gen), strict=True)
    s3.eval()
    dump = {}

    with torch.inference_mode():
        # ── LM reference tokens: S3 tokenizer on the whole clip at 16 kHz ─────
        wav16, _ = librosa.load(args.wav, sr=S3_SR, mono=True)
        dump["lm_wav16"] = wav16.astype(np.float32)
        w = torch.from_numpy(wav16).float().unsqueeze(0)
        mel = s3.tokenizer.log_mel_spectrogram(w)                    # [1, 128, F]
        dump["tok_mel"] = mel[0].numpy()
        mel_len = torch.tensor([mel.shape[-1]])
        hidden, code_len = s3.tokenizer.encoder(mel, mel_len)
        dump["tok_hidden"] = hidden[0].numpy()
        codes = s3.tokenizer.quantizer.encode(hidden)
        dump["lm_reference_tokens"] = codes[0, :int(code_len[0])].numpy().astype(np.int32)

        # ── pyannote x-vector (the `spk_proj` input) ──────────────────────────
        wf = _load_waveform(args.wav, "cpu")                           # [1, 1, n] @ 16 kHz
        dump["spk_wav16"] = wf[0, 0].numpy()
        spk = load_speaker_embedding_model(
            pathlib.Path(args.repo) / "speaker" / "model.safetensors")
        raw = spk(wf).squeeze(0).float().numpy()
        dump["speaker_raw"] = raw
        dump["speaker_embedding"] = (raw / np.linalg.norm(raw)).astype(np.float32)

        # ── S3Gen conditioning (embed_ref on ≤ 10 s at 24 kHz) ────────────────
        wav24 = vocoder._load_decoder_reference(args.wav)[:int(10 * S3GEN_SR)]
        dump["s3_wav24"] = wav24
        ref24 = torch.from_numpy(wav24).float().unsqueeze(0)
        dump["prompt_feat"] = s3.mel_extractor(ref24).transpose(1, 2)[0].numpy()
        wav16b = get_resampler(S3GEN_SR, S3_SR, "cpu")(ref24)
        dump["s3_wav16"] = wav16b[0].numpy()
        feat, _, _ = extract_feature(wav16b)
        dump["campp_fbank"] = feat[0].numpy()
        dump["xvector"] = s3.speaker_encoder.forward(feat.float())[0].numpy()
        toks, lens = s3.tokenizer(wav16b.float())
        ref = s3.embed_ref(wav24, S3GEN_SR, device="cpu")
        dump["prompt_token"] = ref["prompt_token"][0].numpy().astype(np.int32)
        dump["embed_ref_xvector"] = ref["embedding"][0].numpy()

        # ── > 30 s clip: the tokenizer's windowed path ────────────────────────
        long16 = np.tile(wav16, int(np.ceil(42 * S3_SR / len(wav16))))[:42 * S3_SR]
        dump["long_wav16"] = long16.astype(np.float32)
        lt, ll = s3.tokenizer(torch.from_numpy(long16).float().unsqueeze(0))
        dump["long_tokens"] = lt[0, :int(ll[0])].numpy().astype(np.int32)

    np.savez(out / "clone_reference.npz",
             **{k: np.ascontiguousarray(v) for k, v in dump.items()})
    (out / "clone_reference.json").write_text(json.dumps(
        {"wav": str(args.wav), "shapes": {k: list(np.shape(v)) for k, v in dump.items()}},
        indent=1))
    print({k: np.shape(v) for k, v in dump.items()})


if __name__ == "__main__":
    main()
