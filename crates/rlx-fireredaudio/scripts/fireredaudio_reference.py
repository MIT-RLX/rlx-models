"""FireRedAudio ASR as the official code runs it, for the crate's checks.

    python -I fireredaudio_reference.py <FireRedAudio source> <model dir> <clip.wav> <out.json> [repeat]

The source is a checkout of github.com/FireRedTeam/FireRedAudio (its
`inference.py` and `fireredaudio/` package); the model directory holds the
Hugging Face `FireRedTeam/FireRedAudio` checkpoint (`config.json`, the
shards, `tokenizer.json`). It writes the ASR prompt's ids, the mel's and the
audio encoder's first and last rows with every row's sum (the encoder run in
f32), and two transcripts: greedy (`num_beams=1`, no repetition penalty)
and the official ASR settings (`num_beams=4`, `repetition_penalty=1.1`).
"""

import json
import sys

import torch


def rows(t: torch.Tensor) -> dict:
    t = t.float()
    return {
        "shape": list(t.shape),
        "first": t[0].tolist(),
        "last": t[-1].tolist(),
        "sums": t.sum(-1).tolist(),
    }


def main() -> None:
    src, model_dir, wav, out = sys.argv[1:5]
    # The clip back to back `repeat` times: past one 30 s chunk.
    repeat = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    sys.path.insert(0, src)
    import inference
    from transformers import GenerationConfig

    torch.manual_seed(0)
    inf = inference.FireRedAudioInference(model_dir, device="cpu")
    model = inf.model
    chatml = inference.build_understand_prompt(
        inference.DEFAULT_ASR_PROMPT, 1, model.config.audio_special_token
    )
    audios = [{
        "feat_type": inference.FEAT_TYPE_UNDERSTAND,
        "audio_understand": inference.read_audio(wav, inference.UNDERSTAND_SAMPLE_RATE)
        .repeat(repeat)
        .numpy(),
        "audio_generation": None,
        "role": "user",
    }]
    batch = inf.encoder.encode(chatml, audios)
    feats = batch["audio_features"]
    mask = batch["audio_feature_attention_mask"]
    n = int(mask.sum())
    mel = feats[0, :, :n].transpose(0, 1)

    # The audio encoder in f32, its weights cast for the check only.
    enc = model.audio_encoder.float()
    with torch.inference_mode():
        audio = model.get_audio_features(input_features=feats.float(), feature_attention_mask=mask)
    model.audio_encoder.to(torch.bfloat16)

    def transcribe(**gen) -> tuple[list[int], str]:
        cfg = inf._gen_config("asr", None)
        for k, v in gen.items():
            setattr(cfg, k, v)
        with torch.inference_mode():
            ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_features=batch["audio_features"],
                audio_feature_attention_mask=mask,
                generation_config=cfg,
            )
        ids = ids[0].tolist()
        return ids, inf.tokenizer.decode(ids, skip_special_tokens=True)

    greedy_ids, greedy = transcribe(num_beams=1, repetition_penalty=1.0)
    beam_ids, beam = transcribe()
    del enc
    json.dump(
        {
            "clip": wav.rsplit("/", 1)[-1],
            "repeat": repeat,
            "chatml": chatml,
            "input_ids": batch["input_ids"][0].tolist(),
            "mel_frames": n,
            "mel": rows(mel),
            "encoder": rows(audio),
            "greedy_ids": greedy_ids,
            "greedy": greedy,
            "beam_ids": beam_ids,
            "beam": beam,
        },
        open(out, "w"),
    )
    print("greedy:", greedy)
    print("beam:  ", beam)


if __name__ == "__main__":
    main()
