#!/usr/bin/env bash
# Open a page of this crate in a headless browser and print what it POSTs to
# /__result. usage (repo root): run_page.sh <chrome|firefox> <page?query> [timeout s]
set -u
browser=$1; page=$2; limit=${3:-600}; port=${PORT:-8787}
out=$(mktemp /tmp/rlx-web-run.XXXX)
python3 crates/rlx-models-web/web/serve.py --port "$port" --once > "$out" 2>/dev/null &
srv=$!
sleep 1
prof=$(mktemp -d /tmp/rlx-web-prof.XXXX)
url="http://127.0.0.1:$port/crates/rlx-models-web/web/$page"
case $browser in
  chrome)
    "${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}" --headless=new \
      --user-data-dir="$prof" --no-first-run --disable-extensions --enable-unsafe-webgpu "$url" >/dev/null 2>&1 & ;;
  firefox)
    "${FIREFOX:-/Applications/Firefox.app/Contents/MacOS/firefox}" --headless --no-remote \
      --profile "$prof" "$url" >/dev/null 2>&1 & ;;
  *) echo "browser: chrome | firefox" >&2; exit 2 ;;
esac
pid=$!
for ((i = 0; i < limit; i++)); do kill -0 $srv 2>/dev/null || break; sleep 1; done
kill $pid $srv 2>/dev/null; pkill -f "$prof" 2>/dev/null; wait 2>/dev/null; rm -rf "$prof"
cat "$out"; rm -f "$out"
