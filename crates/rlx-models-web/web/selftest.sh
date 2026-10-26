#!/usr/bin/env bash
# Load one model in headless Chrome via selftest.html and print its report.
# usage (repo root): crates/rlx-models-web/web/selftest.sh <model id> [timeout s]
# Exit status: 0 when the page reported ok.
set -u
model=$1; limit=${2:-600}; port=${PORT:-8787}
chrome=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
python3 crates/rlx-models-web/web/serve.py --port $port --once > /tmp/selftest-$model.out 2>/dev/null &
srv=$!
sleep 1
prof=$(mktemp -d /tmp/chrome-rlx.XXXX)
"$chrome" --headless=new --user-data-dir=$prof \
  --no-first-run --disable-extensions "http://127.0.0.1:$port/crates/rlx-models-web/web/selftest.html?model=$model" >/dev/null 2>&1 &
browser=$!
for ((i=0; i<limit; i++)); do kill -0 $srv 2>/dev/null || break; sleep 1; done
kill $browser 2>/dev/null; kill $srv 2>/dev/null; wait 2>/dev/null; rm -rf "$prof"
cat /tmp/selftest-$model.out
grep -q '"ok": *true' /tmp/selftest-$model.out
