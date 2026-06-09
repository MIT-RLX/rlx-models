#!/usr/bin/env bash
# RLX — vLLM-Omni reference for Voxtral-4B-TTS (export codes / synthesize wav).
set -euo pipefail

CMD=${1:-help}
shift || true

MODEL_DIR=${RLX_VOXTRAL_TTS_DIR:-/model}
TEXT=${RLX_VOXTRAL_TTS_TEXT:-Hello}
VOICE=${RLX_VOXTRAL_TTS_VOICE:-neutral_female}
OUT_WAV=${RLX_VOXTRAL_TTS_OUT_WAV:-/out/reference.wav}
OUT_CODES=${RLX_VOXTRAL_TTS_OUT_CODES:-/out/codes.txt}
CFG_ALPHA=${RLX_VOXTRAL_TTS_CFG_ALPHA:-1.2}
SEED=${RLX_VOXTRAL_TTS_SEED:-42}

case "$CMD" in
  export-codes|synthesize)
    python3 - "$CMD" "$MODEL_DIR" "$TEXT" "$VOICE" "$OUT_WAV" "$OUT_CODES" "$CFG_ALPHA" "$SEED" <<'PY'
import sys
from pathlib import Path

cmd, model_dir, text, voice, out_wav, out_codes, cfg_alpha, seed = sys.argv[1:9]
cfg_alpha = float(cfg_alpha)
seed = int(seed)
model_name = model_dir if Path(model_dir).is_dir() else "mistralai/Voxtral-4B-TTS-2603"

from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni

import torch
import random
import numpy as np

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed % (2**32 - 1))

tok_path = Path(model_dir) / "tekken.json"
tok = MistralTokenizer.from_file(str(tok_path)) if tok_path.is_file() else MistralTokenizer.from_hf_hub(model_name)
tokenized = tok.instruct_tokenizer.encode_speech_request(SpeechRequest(input=text, voice=voice))
inputs = {
    "prompt_token_ids": tokenized.tokens,
    "additional_information": {"voice": [voice]},
}

llm = Omni(model=model_name, log_stats=False)
sp = SamplingParams(max_tokens=2500, extra_args={"cfg_alpha": cfg_alpha})

stage0_codes = None
final_audio = None

for out in llm.generate([inputs], [sp, sp], py_generator=True):
    if out.error:
        raise SystemExit(out.error)
    if out.stage_id == 0 and out.finished:
        frames = out.multimodal_output.get("audio") or []
        if frames:
            stage0_codes = torch.cat(frames, dim=-1).flatten().int().tolist()
    if out.stage_id == 1 and out.finished:
        frames = out.multimodal_output.get("audio") or []
        if frames:
            final_audio = torch.cat(frames).float().cpu().numpy()

if cmd == "export-codes":
    if not stage0_codes:
        raise SystemExit("stage 0 produced no codes")
    n_frames = len(stage0_codes) // 37
    Path(out_codes).parent.mkdir(parents=True, exist_ok=True)
    body = " ".join(str(c) for c in stage0_codes)
    Path(out_codes).write_text(f"{n_frames}\n{body}\n", encoding="utf-8")
    print(f"wrote {out_codes} ({n_frames} frames)")
else:
    if final_audio is None:
        raise SystemExit("stage 1 produced no audio")
    import soundfile as sf
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, final_audio, 24000)
    print(f"wrote {out_wav} ({len(final_audio)/24000:.2f}s)")
PY
    ;;
  help|*)
    cat <<'EOF'
Usage: entrypoint-ref.sh <command>

  export-codes   RLX_VOXTRAL_TTS_OUT_CODES (vLLM stage 0 discrete codes)
  synthesize     RLX_VOXTRAL_TTS_OUT_WAV (full vLLM-Omni pipeline)
EOF
    exit 0
    ;;
esac
