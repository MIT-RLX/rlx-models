#!/usr/bin/env bash
# Prepare ASR bench clips under .cache/whisper-bench/ (gitignored — clips are
# regenerated here, never committed; only this recipe lives in git).
#
#   jfk          (auto)  openai/whisper public-domain sample
#   ls-clean     (auto)  LibriSpeech test-clean  utterance (CC BY 4.0, human ref)
#   ls-other     (auto)  LibriSpeech test-other  utterance (CC BY 4.0, human ref)
#   mlk          (local) MLK "I Have a Dream" via MLK_SRC (copyrighted)
#   commonvoice / fleurs / librivox (local) via <NAME>_SRC + <NAME>_REF env
#
# Each clip → <id>_16k.wav + <id>.reference.txt; manifest.json lists all present.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/.cache/whisper-bench"
LS="${ROOT}/.cache/librispeech"
mkdir -p "$OUT" "$LS"
have_ff() { command -v ffmpeg >/dev/null 2>&1; }

# ── jfk (public domain) ────────────────────────────────────────────────────
JFK_REFERENCE=' And so my fellow Americans ask not what your country can do for you ask what you can do for your country.'
if [[ ! -s "$OUT/jfk_16k.wav" ]]; then
  have_ff || { echo "fetch_whisper_bench: ffmpeg required" >&2; exit 1; }
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  git clone --depth 1 --filter=blob:none --sparse https://github.com/openai/whisper.git "$tmp/repo"
  (cd "$tmp/repo" && git sparse-checkout set tests)
  ffmpeg -y -loglevel error -i "$tmp/repo/tests/jfk.flac" -ar 16000 -ac 1 "$OUT/jfk_16k.wav"
fi
printf '%s' "$JFK_REFERENCE" > "$OUT/jfk.reference.txt"
echo "jfk ready"

# ── LibriSpeech test-clean / test-other (CC BY 4.0, official transcripts) ───
ls_clip() { # id split spk/chap utt
  local id=$1 split=$2 sub=$3 utt=$4
  if [[ -s "$OUT/${id}_16k.wav" && -s "$OUT/$id.reference.txt" ]]; then echo "$id cached"; return; fi
  have_ff || { echo "skip $id: ffmpeg required" >&2; return; }
  local tar="$LS/$split.tar.gz"
  [[ -s "$tar" ]] || curl -L --fail -o "$tar" "https://www.openslr.org/resources/12/$split.tar.gz"
  local base="LibriSpeech/$split/$sub" trf="LibriSpeech/$split/$sub/${sub/\//-}.trans.txt"
  ( cd "$LS" && tar xzf "$split.tar.gz" "$trf" "$base/$utt.flac" )
  grep "^$utt " "$LS/$trf" | cut -d' ' -f2- | tr -d '\n' > "$OUT/$id.reference.txt"
  ffmpeg -y -loglevel error -i "$LS/$base/$utt.flac" -ar 16000 -ac 1 "$OUT/${id}_16k.wav"
  echo "$id ready"
}
ls_clip ls-clean test-clean 6930/75918 6930-75918-0004
ls_clip ls-other test-other 7902/96591 7902-96591-0014

# ── optional local clips (copyrighted/gated: bring your own audio + ref) ────
local_clip() { # id src reftext ss t
  local id=$1 src=$2 ref=$3 ss=$4 t=$5
  if [[ -s "$OUT/${id}_16k.wav" ]]; then printf '%s' "$ref" > "$OUT/$id.reference.txt"; echo "$id cached"; return; fi
  if [[ -f "$src" ]] && have_ff; then
    ffmpeg -y -loglevel error -ss "$ss" -t "$t" -i "$src" -ar 16000 -ac 1 "$OUT/${id}_16k.wav"
    printf '%s' "$ref" > "$OUT/$id.reference.txt"; echo "$id ready"
  else
    echo "skip $id (set ${id^^}_SRC to a local audio file)"
  fi
}
local_clip mlk "${MLK_SRC:-/Users/macmini/Downloads/archive/esp32qt-idf copy/data/I Have a Dream, Martin Luther King Jr. [MLKDream].wav}" \
  'One hundred years later the Negro still languishes in the corners of American society' 120 12
# Common Voice (accents), FLEURS (multilingual), LibriVox (long-form public domain):
# gated or transcript-less, so wire via env, e.g.:
#   CV_SRC=clip.mp3 CV_REF='the reference text' scripts/fetch_whisper_bench.sh
[[ -n "${CV_SRC:-}" ]]       && local_clip commonvoice "$CV_SRC"       "${CV_REF:-}"       "${CV_SS:-0}"       "${CV_T:-12}"
[[ -n "${FLEURS_SRC:-}" ]]   && local_clip fleurs      "$FLEURS_SRC"   "${FLEURS_REF:-}"   "${FLEURS_SS:-0}"   "${FLEURS_T:-12}"
[[ -n "${LIBRIVOX_SRC:-}" ]] && local_clip librivox    "$LIBRIVOX_SRC" "${LIBRIVOX_REF:-}" "${LIBRIVOX_SS:-0}" "${LIBRIVOX_T:-30}"

# ── manifest from whatever clips exist ─────────────────────────────────────
{
  echo '{'; echo '  "version": 2,'; echo '  "clips": ['
  first=1
  for wav in "$OUT"/*_16k.wav; do
    [[ -e "$wav" ]] || continue
    id=$(basename "$wav" _16k.wav)
    [[ -s "$OUT/$id.reference.txt" ]] || continue
    [[ $first -eq 1 ]] && first=0 || echo '    ,'
    printf '    { "id": "%s", "language": "en", "sample_rate": 16000, "wav": "%s_16k.wav", "reference": "%s.reference.txt" }\n' "$id" "$id" "$id"
  done
  echo '  ]'; echo '}'
} > "$OUT/manifest.json"
echo "manifest: $(grep -c '"id"' "$OUT/manifest.json") clips under $OUT"
