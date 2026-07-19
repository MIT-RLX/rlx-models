#!/usr/bin/env python3
"""ChatterBox ONNX inference prototype — validate the T3 AR loop end-to-end."""
import sys, wave
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

D = "weights/tts/chatterbox"
TEXT = "The quick brown fox jumps over the lazy dog."
START_TEXT, STOP_TEXT, START_SPEECH, STOP_SPEECH = 255, 0, 6561, 6562
EOS = {2, 6562}
NLAYERS = 30


def sess(n):
    return ort.InferenceSession(f"{D}/onnx/{n}", providers=["CPUExecutionProvider"])


def resample(x, a, b):
    if a == b:
        return x
    n = int(len(x) * b / a)
    i = np.clip((np.arange(n) * a / b).astype(int), 0, len(x) - 2)
    f = np.arange(n) * a / b - i
    return x[i] * (1 - f) + x[i + 1] * f


embed, lm, spenc, cdec = sess("embed_tokens.onnx"), sess("language_model_q4f16.onnx"), sess("speech_encoder.onnx"), sess("conditional_decoder.onnx")

# 1) reference audio @ 24k
w = wave.open("crates/rlx-luxtts/tests/fixtures/prompt.wav", "rb")
sr, ch = w.getframerate(), w.getnchannels()
ref = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
if ch > 1:
    ref = ref.reshape(-1, ch).mean(1)
ref = resample(ref, sr, 24000).astype(np.float32)
se = spenc.run(None, {"audio_values": ref[None]})
seo = {o.name: v for o, v in zip(spenc.get_outputs(), se)}
audio_features = seo["audio_features"]
speaker_embeddings, speaker_features = seo["speaker_embeddings"], seo["speaker_features"]
print("audio_features", audio_features.shape, "spk_emb", speaker_embeddings.shape, "spk_feat", speaker_features.shape)

# 2) text tokens
tok = Tokenizer.from_file(f"{D}/tokenizer.json")
text_ids = [START_TEXT] + tok.encode(TEXT).ids + [STOP_TEXT]
print("text_ids", len(text_ids), text_ids[:8])


def do_embed(ids, pos0, exag=0.5):
    n = len(ids)
    return embed.run(None, {
        "input_ids": np.array([ids], np.int64),
        "position_ids": np.arange(pos0, pos0 + n, dtype=np.int64)[None],
        "exaggeration": np.array([exag], np.float32),
    })[0]


text_embeds = do_embed(text_ids, 0)
start_embed = do_embed([START_SPEECH], len(text_ids))
inputs_embeds = np.concatenate([audio_features, text_embeds, start_embed], axis=1).astype(np.float32)
seq = inputs_embeds.shape[1]
print("prefill seq", seq)

kv = {}
for i in range(NLAYERS):
    kv[f"past_key_values.{i}.key"] = np.zeros((1, 16, 0, 64), np.float16)
    kv[f"past_key_values.{i}.value"] = np.zeros((1, 16, 0, 64), np.float16)
out = lm.run(None, {"inputs_embeds": inputs_embeds, "attention_mask": np.ones((1, seq), np.int64), **kv})
logits, present = out[0], out[1:]
onames = [o.name for o in lm.get_outputs()]


def sample(row, prev, temp=0.8, top_p=0.95, top_k=1000, rep=1.2, rng=None):
    l = row.astype(np.float64).copy()
    for t in set(prev):
        l[t] = l[t] / rep if l[t] > 0 else l[t] * rep
    l = l / temp
    idx = np.argsort(l)[::-1][:top_k]
    p = np.exp(l[idx] - l[idx].max()); p /= p.sum()
    c = np.cumsum(p); k = np.searchsorted(c, top_p) + 1
    idx, p = idx[:k], p[:k] / p[:k].sum()
    return int(rng.choice(idx, p=p))


rng = np.random.default_rng(0)
gen = []
tot = seq
nxt = sample(logits[0, -1], gen, rng=rng)
for step in range(1000):
    if nxt in EOS:
        break
    gen.append(nxt)
    emb = do_embed([nxt], tot)
    inp = {"inputs_embeds": emb.astype(np.float32), "attention_mask": np.ones((1, tot + 1), np.int64)}
    for i in range(NLAYERS):
        inp[f"past_key_values.{i}.key"] = present[onames.index(f"present.{i}.key") - 1]
        inp[f"past_key_values.{i}.value"] = present[onames.index(f"present.{i}.value") - 1]
    out = lm.run(None, inp)
    logits, present = out[0], out[1:]
    tot += 1
    nxt = sample(logits[0, -1], gen, rng=rng)

print("generated", len(gen), "speech tokens:", gen[:12])
# S3Gen flow: prompt_token (reference) prepended to generated; mel_len = 2*prompt_len
audio_tokens = seo["audio_tokens"].astype(np.int64)  # [1, ref_len]
mel_len = speaker_features.shape[1]
prompt = audio_tokens[:, : mel_len // 2]
print("ref audio_tokens", audio_tokens.shape, "-> prompt", prompt.shape[1], "mel_len", mel_len)
codes = np.concatenate([prompt, np.array([gen], np.int64)], axis=1)
wav = cdec.run(None, {"speech_tokens": codes, "speaker_embeddings": speaker_embeddings, "speaker_features": speaker_features})[0]
wav = wav.reshape(-1)
print("waveform", wav.shape, "dur", len(wav) / 24000)
o = wave.open(sys.argv[1] if len(sys.argv) > 1 else "cb_proto.wav", "wb")
o.setnchannels(1); o.setsampwidth(2); o.setframerate(24000)
o.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes()); o.close()
