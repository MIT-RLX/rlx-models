// Open Genmoji's prompt pipeline in the browser (genmoji.html): the metaprompt
// rewrite and the calibrated object / animal / person classifier on
// Qwen3.5-0.8B, split by layer across Web Workers (pipeline.js), then the
// FLUX prompt assembled as `rlx-genmoji` assembles it (genmoji_prompt.js).
// Static: the model streams from the Hugging Face Hub and the metaprompt from
// the open-genmoji repository (fetched, not vendored: it publishes no license).
// `?local=1` reads both from this repo's `weights/` instead (`just web-models-serve`).
import { Qwen35Pipeline } from "./pipeline.js";
import * as P from "./genmoji_prompt.js";
import { ImagePipeline, sticker } from "./genmoji_image.js";

const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const LOCAL = q.get("local") === "1";
const REPO = new URL("../../../", location.href).href;

const MODEL = LOCAL
  ? { modelUrl: `${REPO}weights/lm/qwen3.5-0.8b-q4km/model.gguf`, tokenizerUrl: `${REPO}weights/lm/qwen3.5-0.8b-q4km/tokenizer.json` }
  : {
      modelUrl: q.get("model") ?? "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf",
      tokenizerUrl: "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json",
    };
// The image files `rlx-genmoji --fast --export-web` writes: the repo's
// `weights/web/genmoji` with `?local=1`, else `genmoji/` next to this page.
const IMAGE_BASE = (q.get("imageBase") ?? (LOCAL ? `${REPO}weights/web/genmoji` : new URL("genmoji", location.href).href))
  .replace(/\/$/, "");

const METAPROMPT = LOCAL
  ? `${REPO}weights/image/genmoji/metaprompt/open-genmoji`
  : "https://raw.githubusercontent.com/EvanZhouDev/open-genmoji/main/metaprompt/open-genmoji";

// Metaprompt + few-shot history (~880 tokens) + the rewrite (≤160).
const MAX_SEQ = 1280;
const MAX_NEW = 160;
// Prompt tokens per decode step past the prefill: the long metaprompt goes
// through in 8-token chunks instead of one token at a time.
const ROWS = 8;
// The prefill graph's width: 512 of the prompt's ~880 tokens in one pass, the
// rest 8 at a time (~14 s → ~5 s at 128 wide). Its activations grow ~3.4 MB
// per token per stage: 512 needs ~2 GiB, which fits Chrome's WebGPU buffer
// limit on an M4 Pro; 768 (2.8 GiB) does not. `?prefill=256` for smaller GPUs.
const PREFILL = Number(q.get("prefill") ?? 512);

let pipe = null;
let meta = null;
let catIds = null;
let baseline = null;
// A crashed worker (out of GPU memory, a panic) takes the pipeline with it.
let dead = false;

const fail = (e) => {
  const msg = String(e?.message ?? e);
  log(`Error: ${msg}`);
  if (/unreachable|RuntimeError|worker/i.test(msg)) {
    dead = true;
    $("go").disabled = $("desc").disabled = true;
    log(`A worker crashed — reload the page (try fewer workers or a smaller ?prefill=, now ${PREFILL}).`);
  }
};

const log = (msg) => {
  $("log").textContent += `${msg}\n`;
  $("log").scrollTop = $("log").scrollHeight;
};
const ms = (t) => `${(t / 1000).toFixed(2)} s`;

async function loadMetaprompt() {
  const [md, json] = await Promise.all([
    fetch(`${METAPROMPT}.md`).then((r) => (r.ok ? r.text() : Promise.reject(new Error(`metaprompt: HTTP ${r.status}`)))),
    fetch(`${METAPROMPT}.json`).then((r) => (r.ok ? r.json() : { messages: [] })),
  ]);
  return { text: md, history: json.messages ?? [] };
}

$("load").onclick = async () => {
  $("load").disabled = true;
  const workers = Number($("workers").value);
  const backend = $("backend").value;
  const fetched = new Array(workers).fill(0);
  const totals = new Array(workers).fill(1);
  try {
    log(`Loading Qwen3.5-0.8B on ${workers} worker${workers > 1 ? "s" : ""} (${backend})…`);
    meta = await loadMetaprompt();
    pipe = await Qwen35Pipeline.create({
      ...MODEL, workers, backend, maxSeq: MAX_SEQ, rows: ROWS, prefill: PREFILL,
      onProgress: log,
      onDownload: (i, f, t) => {
        fetched[i] = f;
        totals[i] = t;
        const done = fetched.reduce((a, b) => a + b, 0) / totals.reduce((a, b) => a + b, 0);
        $("progress").style.width = `${Math.min(100, done * 100).toFixed(1)}%`;
      },
    });
    $("progress").style.width = "100%";
    catIds = P.categoryTokenIds(pipe.tokenizer);
    // The classifier's calibration: what it answers for a content-free subject.
    baseline = P.categoryScores(
      await pipe.score(P.chatPrompt(P.classifyMessages("N/A")), catIds.flat()), catIds);
    log(`Ready — memory per worker: ${(await pipe.memoryMb()).join(", ")} MB`);
    $("desc").disabled = $("go").disabled = false;
    $("desc").focus();
  } catch (e) {
    fail(e);
    $("load").disabled = dead;
  }
};

async function makePrompt() {
  const desc = $("desc").value.trim();
  if (!desc || !pipe || dead) return;
  $("go").disabled = true;
  for (const id of ["rewrite", "flux", "cmd", "timing"]) $(id).textContent = "…";
  $("category").textContent = "…";
  try {
    const t0 = performance.now();
    let shown = "";
    const raw = await pipe.generate(P.chatPrompt(P.rewriteMessages(meta.text, meta.history, desc)), MAX_NEW, {
      onToken: (piece) => ($("rewrite").textContent = shown += piece),
    });
    const rewrite = P.cleanReply(raw);
    const rw = pipe.stats;
    $("rewrite").textContent = rewrite;
    const t1 = performance.now();
    const scores = P.categoryScores(await pipe.score(P.chatPrompt(P.classifyMessages(desc)), catIds.flat()), catIds);
    const category = P.pickCategory(scores, baseline);
    const t2 = performance.now();
    $("category").innerHTML = P.CATEGORIES.map((c) => `<span class="${c === category ? "on" : ""}">${c}</span>`).join("");
    const flux = P.assemble(rewrite, category, desc);
    $("flux").textContent = flux;
    const quoted = `'${flux.replaceAll("'", "'\\''")}'`;
    $("cmd").textContent = `target/release/rlx-genmoji --fast --direct --device metal ${quoted} -o genmoji.png`;
    $("timing").textContent =
      `rewrite ${ms(t1 - t0)} (prompt ${rw.prompt_tokens} tokens in ${ms(rw.prefill_ms)}, ` +
      `${rw.tokens} tokens at ${rw.tok_per_s} tok/s) · classify ${ms(t2 - t1)}`;
    await render(flux);
  } catch (e) {
    fail(e);
  } finally {
    $("go").disabled = dead;
  }
}

let image = null;
const imgLog = (msg) => {
  $("imgLog").textContent += `${msg}\n`;
  $("imgLog").scrollTop = $("imgLog").scrollHeight;
};

$("loadImg").onclick = async () => {
  $("loadImg").disabled = true;
  try {
    imgLog("Loading the image models: CLIP + VAE, T5 (4 workers), FLUX.1 denoiser (8 workers)…");
    image = await ImagePipeline.create({
      base: IMAGE_BASE,
      backend: $("backend").value,
      onProgress: imgLog,
      onDownload: (f) => ($("imgProgress").style.width = `${Math.min(100, f * 100).toFixed(1)}%`),
    });
    $("imgProgress").style.width = "100%";
    imgLog(`Ready — ${image.workers} image workers. The next prompt also renders.`);
  } catch (e) {
    imgLog(`Error: ${e.message ?? e}`);
    $("loadImg").disabled = false;
  }
};

/** Render the FLUX prompt on screen; resolves when the sticker is shown. */
async function render(flux) {
  if (!image) return;
  $("imgTiming").textContent = "rendering…";
  let rgb;
  try {
    rgb = await image.generate(flux, {
      onStep: (i, n) => ($("imgTiming").textContent = `denoising step ${i}/${n}…`),
    });
  } catch (e) {
    $("imgTiming").textContent = "—";
    imgLog(`Error: ${e.message ?? e}`);
    return;
  }
  const c = sticker(rgb, 160, 512);
  c.style.width = c.style.height = "256px";
  $("sticker").replaceChildren(c);
  const s = image.stats;
  $("imgTiming").textContent = `${(s.total_ms / 1000).toFixed(1)} s: text encoders ${(s.text_ms / 1000).toFixed(1)} s, ` +
    `denoising ${(s.denoise_ms / 1000).toFixed(1)} s (${s.step_ms.length} steps: ${s.step_ms.map((m) => (m / 1000).toFixed(1)).join(", ")} s), ` +
    `VAE ${(s.decode_ms / 1000).toFixed(1)} s`;
  return c;
}

$("go").onclick = makePrompt;
$("desc").onkeydown = (e) => e.key === "Enter" && makePrompt();

// `?auto=1[&workers=N&backend=…]`: load, run a few descriptions, POST a JSON
// report to `/__result` (the dev server prints it; see run_page.sh).
if (q.get("auto") === "1") {
  (async () => {
    const report = { ok: false, workers: Number(q.get("workers") ?? 2), backend: q.get("backend") ?? "webgpu", results: [] };
    try {
      $("workers").value = String(report.workers);
      $("backend").value = report.backend;
      await $("load").onclick();
      if (q.get("image") === "1") {
        await $("loadImg").onclick();
        if (!image) throw new Error($("imgLog").textContent.trim().split("\n").pop());
      }
      if (!pipe) throw new Error($("log").textContent.trim().split("\n").pop());
      report.memory_mb = await pipe.memoryMb();
      for (const d of (q.get("descs") ?? "flying pig|a wizard cat|a firefighter|taco").split("|")) {
        $("desc").value = d;
        await makePrompt();
        const r = { desc: d, rewrite: $("rewrite").textContent, category: $("category").textContent,
          flux: $("flux").textContent, timing: $("timing").textContent };
        if (image) {
          r.image = $("imgTiming").textContent;
          r.stats = image.stats;
          r.png = $("sticker").querySelector("canvas")?.toDataURL("image/png");
        }
        report.results.push(r);
      }
      report.ok = report.results.every((r) => (r.flux.startsWith("emoji of") || r.flux.startsWith("Emoji of"))
        && (!image || r.png));
    } catch (e) {
      report.error = String(e?.message ?? e);
    }
    report.log = $("log").textContent;
    report.image_log = $("imgLog").textContent;
    await fetch("/__result", { method: "POST", body: JSON.stringify(report) });
  })();
}
