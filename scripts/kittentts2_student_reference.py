#!/usr/bin/env python3
"""Reference dump for `rlx-kittentts2`'s distilled student decoder.

Builds the upstream student conditioning for a built-in voice (≤ 4 s of its
clip, prompt cut to ≤ 112 tokens), runs `meanflow_forward` for one step from
saved noise on fixed codec tokens, and vocodes with a deterministic source:

    .venv-kitten2/bin/python scripts/kittentts2_student_reference.py \
        --repo weights/tts/kitten-tts-2 \
        --s3gen weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors \
        --tokens weights/tts/kitten-tts-2/bruno_tokens.json \
        --out weights/tts/kitten-tts-2/student_reference
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
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--voice", default="Bruno")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    import kittenml.kittentts2.vocoder as vocoder
    vocoder.hf_hub_download = lambda **_: args.s3gen
    from kittenml.kittentts2.s3gen import S3GEN_SIL
    from kittenml.kittentts2.student_flow import meanflow_forward

    repo = pathlib.Path(args.repo)
    codec = vocoder.S3Codec(device="cpu", student_weights=repo / "decoders/student_w4.pt")
    voices = json.loads((repo / "voices/voices.json").read_text())
    wav = repo / "voices" / voices[args.voice]["reference"]
    cond = codec.reference_conditioning(str(wav))
    audio = json.loads(pathlib.Path(args.tokens).read_text())[0]
    tokens = torch.tensor(audio + [int(S3GEN_SIL)] * 3).long()[None]
    dump = {k: cond[k].float().numpy() for k in ("prompt_token", "prompt_feat", "embedding")}
    with torch.inference_mode():
        n_all = cond["prompt_token"].shape[1] + tokens.shape[1]
        z = torch.randn(1, 80, 2 * n_all)
        mel, _, _ = meanflow_forward(codec.student, tokens, torch.tensor([tokens.shape[1]]),
                                     n_steps=1, z=z, **cond)
        dump["noise"] = z.numpy()
        dump["mel"] = mel.numpy()
        # Same pass with activation fake-quant off: isolates the network from
        # int8 rounding flips (round-half boundaries move by a ulp).
        import kittenml.kittentts2.student_flow as sf
        saved = sf.activation_quant
        sf.activation_quant = lambda x, eps=1e-5: x
        mel_noaq, _, _ = meanflow_forward(codec.student, tokens, torch.tensor([tokens.shape[1]]),
                                          n_steps=1, z=z, **cond)
        sf.activation_quant = saved
        dump["mel_no_act_quant"] = mel_noaq.numpy()
        hift = codec.s3gen.mel2wav
        f0 = hift.f0_predictor(mel)
        up = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        sg = hift.m_source.l_sin_gen
        fm = torch.zeros(1, sg.harmonic_num + 1, up.shape[1])
        for i in range(sg.harmonic_num + 1):
            fm[:, i:i + 1, :] = up.transpose(1, 2) * (i + 1) / sg.sampling_rate
        sine = sg.sine_amp * torch.sin(2 * np.pi * (torch.cumsum(fm, dim=-1) % 1))
        sine = (sine * (up.transpose(1, 2) > sg.voiced_threshold).float()).transpose(1, 2)
        source = hift.m_source.l_tanh(hift.m_source.l_linear(sine)).transpose(1, 2)
        dump["source"] = source.numpy()
        dump["wav"] = hift.decode(x=mel, s=source).numpy()
    np.savez(out / "student_reference.npz",
             **{k: np.ascontiguousarray(v) for k, v in dump.items()})
    (out / "student_reference.json").write_text(json.dumps(
        {"voice": args.voice, "tokens": tokens[0].tolist()}))
    print({k: v.shape for k, v in dump.items()})


if __name__ == "__main__":
    main()
