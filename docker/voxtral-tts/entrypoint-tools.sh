#!/usr/bin/env bash
# RLX — Voxtral TTS Docker tools (tokenize, convert voice .pt → .f32).
# No Python lives in the Rust crate; this image is the supported host interface.
set -euo pipefail

CMD=${1:-help}
shift || true

MODEL_DIR=${RLX_VOXTRAL_TTS_DIR:-/model}
TEXT=${RLX_VOXTRAL_TTS_TEXT:-Hello}
VOICE=${RLX_VOXTRAL_TTS_VOICE:-neutral_female}
OUT=${RLX_VOXTRAL_TTS_OUT:-/out/prompt_tokens.txt}

case "$CMD" in
  tokenize)
    python3 - "$MODEL_DIR" "$TEXT" "$VOICE" "$OUT" <<'PY'
import sys
from pathlib import Path
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

model_dir, text, voice, out = sys.argv[1:5]
tok_path = Path(model_dir) / "tekken.json"
tok = MistralTokenizer.from_file(str(tok_path)) if tok_path.is_file() else MistralTokenizer.from_hf_hub("mistralai/Voxtral-4B-TTS-2603")
ids = tok.instruct_tokenizer.encode_speech_request(SpeechRequest(input=text, voice=voice)).tokens
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(" ".join(str(i) for i in ids) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(ids)} tokens)")
PY
    ;;
  convert-voices)
    python3 - "$MODEL_DIR" <<'PY'
import struct
import sys
from pathlib import Path
import torch

model_dir = Path(sys.argv[1])
voice_dir = model_dir / "voice_embedding"
if not voice_dir.is_dir():
    raise SystemExit(f"missing {voice_dir}")
for pt in sorted(voice_dir.glob("*.pt")):
    t = torch.load(pt, map_location="cpu", weights_only=True)
    if isinstance(t, torch.Tensor):
        data = t.detach().float().reshape(-1).tolist()
    else:
        raise SystemExit(f"unexpected voice tensor type in {pt}")
    out = pt.with_suffix(".f32")
    blob = struct.pack(f"<{len(data)}f", *data)
    out.write_bytes(blob)
    print(f"{pt.name} -> {out.name} ({len(data)} floats)")
PY
    ;;
  help|*)
    cat <<'EOF'
Usage: entrypoint-tools.sh <command>

  tokenize        RLX_VOXTRAL_TTS_TEXT, RLX_VOXTRAL_TTS_VOICE, RLX_VOXTRAL_TTS_OUT
  convert-voices  write voice_embedding/*.f32 next to *.pt
EOF
    exit 0
    ;;
esac
