// Chat demo for Qwen3.5-0.8B on Web Workers, in two modes:
//
// - "layers": one `Qwen35Pipeline` (pipeline.js) — the 24 layers split across
//   N workers, one conversation. Less memory per worker; not faster, since
//   every token still passes through every worker in turn.
// - "copies": N single-worker pipelines, each holding the whole model, taking
//   questions in parallel (one per input line). Each answer is a little slower,
//   but they overlap on the GPU, so total throughput grows (measured 1.5× with
//   2 workers, 1.8× with 3). ~1.9 GB of wasm memory per worker.
//
// URL options: ?model=<id in models.js | hf:owner/repo/file.gguf>&hf=<custom Hub GGUF>
// &mode=layers|copies&workers=N&eos=0|1&max=N&greedy=0|1&temp=T&ask=<q>
// (`ask` loads and asks on its own — repeat it for several questions; the page
// sets `document.body.dataset.done` when it finishes).
import { MODELS } from "./models.js";
import { Qwen35Pipeline } from "./pipeline.js";
import { renderMarkdown } from "./md.js";

// Prompt + answer tokens a sequence may hold. Prompts are prefilled in one
// 128-wide graph; the rest is room for the answer.
// Context of the loaded model(s), from the Context select at load.
let MAX_SEQ = 2048;

const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
// Models: the local checkpoints (models.js), Qwen3.5 GGUFs streamed straight
// from the Hugging Face Hub, or any Hub GGUF you name.
const HF_TOKENIZER = "Qwen/Qwen3.5-0.8B";
const HF_PRESETS = [
  ["unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf", "Qwen3.5-0.8B Q4_K_M (533 MB)"],
  ["unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_S.gguf", "Qwen3.5-0.8B Q4_K_S (508 MB)"],
  ["unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q5_K_M.gguf", "Qwen3.5-0.8B Q5_K_M (590 MB)"],
  ["unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q6_K.gguf", "Qwen3.5-0.8B Q6_K (639 MB)"],
  ["unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf", "Qwen3.5-0.8B Q8_0 (812 MB)"],
  ["unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf", "Qwen3.5-0.8B Q4_0 (507 MB)"],
  ["unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-IQ4_XS.gguf", "Qwen3.5-0.8B IQ4_XS (493 MB)"],
  ["unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-IQ4_NL.gguf", "Qwen3.5-0.8B IQ4_NL (507 MB)"],
  ["unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf", "Qwen3.5-0.8B BF16 (1.5 GB)"],
  ["unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf", "Qwen3.5-2B Q4_K_M (1.3 GB)"],
  ["unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5-4B Q4_K_M (2.7 GB, 2+ workers)", 2],
];
// Workers a preset needs at least (each wasm instance has 4 GiB).
const MIN_WORKERS = Object.fromEntries(HF_PRESETS.filter((p) => p[2]).map(([path, , n]) => [`hf:${path}`, n]));
const localBase = (id) => new URL(`../../../weights/${MODELS[id].dir}`, location.href).href;
{
  // Local checkpoints only where this server has them (`just web-models-serve`);
  // a static host (GitHub Pages) has no weights/, so the Hub is the default.
  const ids = Object.entries(MODELS).filter(([, m]) => m.kind === "qwen35").map(([id]) => id);
  const found = await Promise.all(
    ids.map((id) => fetch(`${localBase(id)}/model.gguf`, { method: "HEAD" }).then((r) => r.ok, () => false)),
  );
  const local = document.createElement("optgroup");
  local.label = "This server";
  ids.forEach((id, i) => found[i] && local.append(new Option(MODELS[id].label, id)));
  const hub = document.createElement("optgroup");
  hub.label = "Hugging Face Hub (streamed)";
  for (const [path, label] of HF_PRESETS) hub.append(new Option(label, `hf:${path}`));
  hub.append(new Option("Other GGUF on the Hub…", "custom"));
  if (local.children.length) $("model").append(local);
  $("model").append(hub);
}
if (q.get("model")) $("model").value = q.get("model");
if (q.get("hf")) {
  $("model").value = "custom";
  $("hfPath").value = q.get("hf");
}
const syncModel = () => ($("customBox").hidden = $("model").value !== "custom");
$("model").onchange = () => {
  syncModel();
  // A bigger model needs its layers spread over more workers.
  const need = MIN_WORKERS[$("model").value];
  if (need && Number($("workers").value) < need) {
    $("workers").value = need;
    $("workersVal").textContent = need;
  }
};
syncModel();

// "owner/repo/path/file.gguf" (or a URL) → a URL the Hub resolves to the file.
const hubUrl = (spec, file) => {
  if (/^https?:\/\//.test(spec)) return spec;
  const [owner, repo, ...rest] = spec.replace(/^\/+/, "").split("/");
  return `https://huggingface.co/${owner}/${repo}/resolve/main/${file ?? rest.join("/")}`;
};
function modelSource() {
  const v = $("model").value;
  if (v === "custom" || v.startsWith("hf:")) {
    const spec = v === "custom" ? $("hfPath").value.trim() : v.slice(3);
    const tok = (v === "custom" ? $("hfTok").value.trim() : "") || HF_TOKENIZER;
    return {
      modelUrl: hubUrl(spec),
      tokenizerUrl: /^https?:/.test(tok) ? tok : hubUrl(tok, "tokenizer.json"),
      hub: true,
      label: spec.split("/").pop(),
    };
  }
  return { base: localBase(v), hub: false, label: MODELS[v].label };
}
// Default: one chat, its layers split across 3 workers.
$("mode").value = q.get("mode") ?? "layers";
$("workers").value = q.get("workers") ?? "3";
if (q.get("eos")) $("untilEos").checked = q.get("eos") !== "0";
if (q.get("max")) $("maxTokens").value = q.get("max");
if (q.get("greedy")) $("greedy").checked = q.get("greedy") !== "0";
if (q.get("think")) $("think").checked = q.get("think") !== "0";
if (q.get("budget")) $("thinkBudget").value = q.get("budget");
const syncBudget = () => ($("thinkBudgetVal").textContent = $("thinkBudget").value);
$("thinkBudget").oninput = syncBudget;
syncBudget();
if (q.get("spec")) $("spec").checked = q.get("spec") !== "0";
if (q.get("ctx")) $("ctx").value = q.get("ctx");
if (q.get("temp")) $("temp").value = q.get("temp");
const syncGreedy = () => $("samplingBox").classList.toggle("off", $("greedy").checked);
$("greedy").onchange = syncGreedy;
syncGreedy();
const sampling = () =>
  $("greedy").checked
    ? null
    : {
        temperature: Number($("temp").value),
        topP: Number($("topP").value),
        topK: Number($("topK").value),
        presencePenalty: Number($("presence").value),
      };
// "Until end of answer" makes the length slider moot.
const syncEos = () => $("maxTokensBox").classList.toggle("off", $("untilEos").checked);
$("untilEos").onchange = syncEos;
syncEos();
for (const id of ["workers", "maxTokens", "temp", "topP", "topK", "presence"]) {
  const show = () => ($(`${id}Val`).textContent = $(id).value);
  $(id).oninput = show;
  show();
}
const HINTS = {
  layers: "Each worker loads only its slice of the layers into its own wasm instance and WebGPU device, " +
    "fetching just those tensors. Less memory per worker; one answer at a time.",
  copies: "Each worker holds the whole model (~1.9 GB) and answers its own question. Type one question per " +
    "line: they are answered at the same time, sharing the GPU.",
};
const showHint = () => {
  $("modeHint").textContent = HINTS[$("mode").value];
  $("input").placeholder = $("mode").value === "copies" ? "Questions, one per line…" : "Ask Qwen3.5…";
};
$("mode").onchange = showHint;
showHint();

// The loaded model: one pipeline (layers) or several (copies).
let pipelines = [];
// Aborts the generation in flight (the Stop button, Esc).
let stopper = null;
let mode = "layers";
const status = (s, err = false) => (($("status").textContent = s), $("status").classList.toggle("err", err));
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
// One user turn in Qwen's chat template. `follow`: a later turn of the same
// conversation (closes the previous answer first). `think`: let the model
// reason before answering (an open <think> block) instead of skipping it.
const chatPrompt = (text, { follow = false, think = false } = {}) =>
  (follow ? "<|im_end|>\n" : "") +
  `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n` +
  (think ? "<think>\n" : "<think>\n\n</think>\n\n");

// An answer with reasoning: the text before </think> is the reasoning (shown
// folded), the rest the answer.
function renderAnswer(text, think, streaming) {
  if (!think) return renderMarkdown(text);
  const cut = text.indexOf("</think>");
  const reasoning = (cut < 0 ? text : text.slice(0, cut)).trim();
  const reply = cut < 0 ? "" : text.slice(cut + "</think>".length).trim();
  const open = streaming && cut < 0 ? " open" : "";
  const label = cut < 0 ? "Thinking…" : "Thought";
  return `<details class="think"${open}><summary>${label}</summary>${renderMarkdown(reasoning)}</details>` +
    renderMarkdown(reply);
}
// The answer alone (after the reasoning), for Copy.
const answerOnly = (text, think) => {
  if (!think) return text;
  const cut = text.indexOf("</think>");
  return cut < 0 ? "" : text.slice(cut + "</think>".length).trim();
};

// Follow new text only while the reader is at the bottom: scrolling up to read
// an earlier answer is not undone by the next streamed token.
let follow = true;
$("chat").addEventListener("scroll", () => {
  const c = $("chat");
  follow = c.scrollHeight - c.scrollTop - c.clientHeight < 60;
});
const toBottom = (force = false) => {
  if (force || follow) $("chat").scrollTop = $("chat").scrollHeight;
};

function addMsg(kind, text = "", parent = $("chat")) {
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  el.textContent = text;
  parent.append(el);
  toBottom(kind === "user");
  return el;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Pages served without a secure context have no async clipboard.
    const ta = Object.assign(document.createElement("textarea"), { value: text });
    document.body.append(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

// A Copy button on `el` that copies whatever `get()` returns at click time
// (an answer's raw Markdown, a prompt's text).
function addCopy(el, get) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "copy";
  b.textContent = "Copy";
  b.title = "Copy to clipboard";
  b.onclick = async () => {
    await copyText(get());
    b.textContent = "Copied";
    b.classList.add("done");
    setTimeout(() => ((b.textContent = "Copy"), b.classList.remove("done")), 1200);
  };
  el.append(b);
}

function workerCards(n, layersOf) {
  $("workerCards").replaceChildren(...Array.from({ length: n }, (_, i) => {
    const el = document.createElement("div");
    el.className = "worker";
    el.id = `w${i}`;
    el.innerHTML = `<b>Worker ${i + 1}</b><span class="k">layers</span><span class="layers">${layersOf(i)}</span>
      <span class="k">download</span><span class="dl">waiting</span>
      <span class="k">wasm memory</span><span class="mem">–</span><div class="bar"><i></i></div>`;
    return el;
  }));
}

// Fill worker card `i` from a pipeline progress line
// ("stage … ready (… fetched X MB …, wasm memory Y MB)").
function cardReady(i, msg) {
  const card = $(`w${i}`);
  if (!card) return;
  const mb = /fetched (\d+) MB/.exec(msg)?.[1];
  const cachedMb = /(\d+) MB from cache/.exec(msg)?.[1];
  const lay = /layers (\d+–\d+)/.exec(msg)?.[1];
  if (lay && mode === "layers") card.querySelector(".layers").textContent = lay;
  const mem = /wasm memory (\d+) MB/.exec(msg)?.[1];
  card.querySelector(".dl").textContent = mb ? `${mb} MB${cachedMb ? ` (${cachedMb} cached)` : ""}` : "done";
  card.querySelector(".mem").textContent = mem ? `${mem} MB` : "–";
  card.classList.add("ready");
  const following = $(`w${i + 1}`);
  if (following) following.querySelector(".dl").textContent = "loading…";
}

function setLoaded(on) {
  $("load").disabled = on;
  $("unload").disabled = !on;
  $("model").disabled = $("workers").disabled = $("mode").disabled = on;
  $("hfPath").disabled = $("hfTok").disabled = on;
  $("input").disabled = $("send").disabled = !on;
}

// Positions per decode step: 4 with "Speculate", else one.
const rowsPerStep = () => ($("spec").checked ? 4 : 1);
async function load() {
  MAX_SEQ = Number($("ctx").value);
  const workers = Number($("workers").value);
  mode = $("mode").value;
  const src = modelSource();
  setLoaded(true);
  $("input").disabled = $("send").disabled = true;
  // Filled in from each worker's header as it loads.
  workerCards(workers, () => (mode === "copies" ? "all" : "…"));
  $("w0").querySelector(".dl").textContent = "loading…";
  status(src.hub ? `Streaming ${src.label} from the Hugging Face Hub…` : "Loading…");
  // Live download progress on each worker's card.
  const onDownload = (i, fetched, total) => {
    const card = $(`w${i}`);
    if (!card || card.classList.contains("ready")) return;
    card.querySelector(".dl").textContent = `${Math.round(fetched / 1e6)} of ${Math.round(total / 1e6)} MB`;
    card.querySelector(".bar > i").style.width = `${Math.round((100 * fetched) / total)}%`;
  };
  const t0 = performance.now();
  try {
    if (mode === "layers") {
      let next = 0;
      pipelines = [await Qwen35Pipeline.create({
        ...src, workers, backend: "webgpu", maxSeq: MAX_SEQ, onDownload, rows: rowsPerStep(),
        onProgress: (msg) => cardReady(next++, msg),
      })];
    } else {
      // One after another: each copy's download peaks while it loads.
      pipelines = [];
      for (let i = 0; i < workers; i++) {
        pipelines.push(await Qwen35Pipeline.create({
          ...src, workers: 1, backend: "webgpu", maxSeq: MAX_SEQ, rows: rowsPerStep(),
          onDownload: (_, f, t) => onDownload(i, f, t),
          onProgress: (msg) => cardReady(i, msg),
        }));
        status(`Loaded ${i + 1} of ${plural(workers, "worker")}…`);
      }
    }
    $("sLoad").textContent = fmtMs(performance.now() - t0);
    const mem = (await Promise.all(pipelines.map((p) => p.memoryMb()))).flat();
    $("sMem").textContent = `${Math.max(...mem)} MB`;
    const sum = (key) => pipelines.flatMap((p) => p[key]).reduce((a, b) => a + b, 0);
    const fetched = sum("fetchedMb"), cached = sum("cachedMb");
    const source = cached >= fetched ? `${fetched} MB read from this browser's cache`
      : cached ? `${fetched - cached} MB downloaded + ${cached} MB from cache` : `${fetched} MB downloaded`;
    status(`Ready: ${plural(workers, "worker")}, ${mode === "layers" ? "layers split" : "a full model each"}; ` +
      `${source}, ${mem.reduce((a, b) => a + b, 0)} MB of wasm memory.`);
    $("input").disabled = $("send").disabled = false;
    $("input").focus();
  } catch (e) {
    status(String(e?.message ?? e), true);
    for (const p of pipelines) p.terminate();
    pipelines = [];
    setLoaded(false);
  }
}

// Stream one answer from `pipeline` into a message element.
async function answer(pipeline, text, el, who, { conversation = false } = {}) {
  const body = document.createElement("div");
  body.className = "md cursor";
  el.append(body);
  const think = $("think").checked;
  let raw = "";
  addCopy(el, () => answerOnly(raw, think) || raw);
  // In a conversation, later turns continue the workers' caches.
  const follow = conversation && pipeline.position > 0;
  const prompt = chatPrompt(text, { follow, think });
  // Until end of answer: everything the context has left (the pipeline
  // clamps to it).
  const untilEos = $("untilEos").checked;
  const limit = untilEos ? MAX_SEQ : Number($("maxTokens").value);
  let shown = "";
  try {
    const samp = sampling();
    const run = (p, max, opts) => pipeline.generate(p, max, {
      signal: stopper?.signal,
      sampling: samp,
      speculate: pipeline.rows > 1,
      onToken: (piece) => {
        shown += piece;
        raw = shown;
        body.innerHTML = renderAnswer(shown, think, true);
        toBottom();
      },
      ...opts,
    });
    let st;
    if (think) {
      // Reasoning first, up to the budget or the model's own </think>; then
      // the answer, continuing the same context. Past the budget the
      // reasoning is closed for the model, as Qwen suggests for a budget.
      const thinkEnd = pipeline.tokenizer.token_id("</think>");
      await run(prompt, Math.min(Number($("thinkBudget").value), limit),
        { append: follow, stopAt: thinkEnd === undefined ? [] : [thinkEnd] });
      const reasoning = pipeline.stats;
      const closed = pipeline.endToken === thinkEnd;
      st = reasoning;
      if (!reasoning.stopped && (closed || !reasoning.ended) && limit > reasoning.tokens) {
        shown += closed ? "</think>" : "\n\n*(thinking budget reached)*\n</think>";
        const close = closed ? "</think>\n\n"
          : "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n</think>\n\n";
        await run(close, limit - reasoning.tokens, { append: true });
        st = combineStats(reasoning, pipeline.stats);
      }
    } else {
      await run(prompt, limit, { append: follow });
      st = pipeline.stats;
    }
    raw = shown;
    window.__lastRaw = shown; // for scripted runs
    body.innerHTML = renderAnswer(shown, think, false);
    const meta = document.createElement("div");
    meta.className = "meta";
    // The pipeline stops at an end-of-turn token or at the limit.
    const why = st.stopped
      ? "stopped by you"
      : st.ended
      ? "end of answer"
      : st.position >= MAX_SEQ - 2
        ? `stopped: the ${MAX_SEQ}-token context is full`
        : `stopped at the ${limit}-token limit`;
    const how = (samp ? `sampled (T ${samp.temperature})` : "greedy") +
      (st.guessed ? ` · ${st.kept} of ${st.guessed} guesses kept, ${st.steps} steps` : "");
    meta.textContent = `${st.tokens} tokens · ${why} · ${how} · prompt ${st.prompt_tokens} tokens in ${fmtMs(st.prefill_ms)} · ${st.tok_per_s.toFixed(1)} tok/s · ${who}`;
    if (conversation) showContext(pipeline);
    el.append(meta);
    return st;
  } catch (e) {
    const err = document.createElement("div");
    err.className = "err";
    err.textContent = `[error: ${e?.message ?? e}]`;
    el.append(err);
    return null;
  } finally {
    body.classList.remove("cursor");
  }
}

// One answer's stats from its reasoning and answer phases.
function combineStats(a, b) {
  const tokens = a.tokens + b.tokens;
  const ms = a.decode_ms + b.decode_ms;
  return { ...b, tokens, prompt_tokens: a.prompt_tokens, prefill_ms: a.prefill_ms, decode_ms: ms,
    tok_per_s: ms ? +((tokens - 2) / (ms / 1e3)).toFixed(2) : 0,
    guessed: (a.guessed ?? 0) + (b.guessed ?? 0), kept: (a.kept ?? 0) + (b.kept ?? 0), steps: (a.steps ?? 0) + (b.steps ?? 0) };
}

// How much of the context the conversation uses.
function showContext(pipeline) {
  $("sCtx").textContent = `${pipeline.position} / ${MAX_SEQ}`;
}

// Start over: the next question opens a new conversation.
function newChat() {
  for (const p of pipelines) p.reset();
  if (pipelines.length) showContext(pipelines[0]);
  addMsg("sys", "New conversation.");
}

async function ask(text) {
  if (!pipelines.length || !text.trim()) return;
  $("input").value = "";
  $("input").disabled = $("send").disabled = true;
  stopper = new AbortController();
  $("send").hidden = true;
  $("stop").hidden = false;
  try {
    if (mode === "layers") {
      addCopy(addMsg("user", text), () => text);
      const st = await answer(pipelines[0], text, addMsg("bot"), plural(pipelines[0].workers, "worker"), {
        conversation: true,
      });
      if (st) {
        $("sTok").textContent = st.tok_per_s.toFixed(1);
        $("sPre").textContent = fmtMs(st.prefill_ms);
      }
    } else {
      await askParallel(text.split("\n").map((s) => s.trim()).filter(Boolean));
    }
  } finally {
    stopper = null;
    $("stop").hidden = true;
    $("send").hidden = false;
    $("input").disabled = $("send").disabled = false;
    $("input").focus();
  }
}

// Stop: the answer(s) in flight end after their current token.
const stopNow = () => stopper?.abort();
$("stop").onclick = stopNow;
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") stopNow();
});

// Several questions at once: each free worker takes the next one.
async function askParallel(questions) {
  const grid = document.createElement("div");
  grid.className = "batch";
  $("chat").append(grid);
  const cards = questions.map((question) => {
    const el = addMsg("bot", "", grid);
    el.innerHTML = `<div class="q"></div>`;
    el.querySelector(".q").textContent = question;
    // The question's own Copy, beside the answer's.
    const qc = document.createElement("button");
    qc.type = "button";
    qc.className = "copy";
    qc.style.right = "62px";
    qc.textContent = "Copy Q";
    qc.title = "Copy the question";
    qc.onclick = async () => {
      await copyText(question);
      qc.textContent = "Copied";
      setTimeout(() => (qc.textContent = "Copy Q"), 1200);
    };
    el.append(qc);
    return el;
  });
  const t0 = performance.now();
  let next = 0;
  let tokens = 0;
  const prefill = [];
  await Promise.all(pipelines.map(async (p, w) => {
    while (next < questions.length && !stopper?.signal.aborted) {
      const i = next++;
      const who = document.createElement("span");
      who.className = "who";
      who.textContent = `worker ${w + 1}`;
      cards[i].querySelector(".q").prepend(who);
      const st = await answer(p, questions[i], cards[i], `worker ${w + 1}`);
      if (st) {
        tokens += st.tokens;
        prefill.push(st.prefill_ms);
      }
    }
  }));
  cards.forEach((card) => {
    if (!card.querySelector(".md")) addMsg("meta", "not started — stopped", card);
  });
  const secs = (performance.now() - t0) / 1e3;
  const total = tokens / secs;
  $("sTok").textContent = total.toFixed(1);
  // The slowest: answers that start together wait on each other's prefill.
  $("sPre").textContent = prefill.length ? `≤ ${fmtMs(Math.max(...prefill))}` : "–";
  addMsg("total", `${plural(questions.length, "answer")}, ${tokens} tokens in ${secs.toFixed(2)} s — ` +
    `${total.toFixed(1)} tok/s in total across ${plural(pipelines.length, "worker")} (prefill included)`);
}

$("load").onclick = load;
$("newChat").onclick = newChat;
$("clearCache").onclick = async () => {
  const gone = self.caches ? await caches.delete("rlx-gguf-v1") : false;
  status(gone ? "Cleared the downloaded models from this browser." : "Nothing cached.");
};
$("unload").onclick = () => {
  for (const p of pipelines) p.terminate();
  pipelines = [];
  setLoaded(false);
  $("workerCards").replaceChildren();
  status("Unloaded.");
};
$("form").onsubmit = (e) => {
  e.preventDefault();
  ask($("input").value);
};
$("input").onkeydown = (e) => {
  // Enter sends; Shift+Enter adds a line (one question per line in parallel mode).
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask($("input").value);
  }
};

// Scripted run: `reload=1` loads, unloads and loads again (the second load
// reads the browser's cache); then ask, mark done.
if (q.get("reload")) {
  const t = performance.now();
  await load();
  const first = performance.now() - t;
  $("unload").onclick();
  const t2 = performance.now();
  await load();
  window.__loadTimes = { first_ms: Math.round(first), second_ms: Math.round(performance.now() - t2) };
  console.log("LOADTIMES", JSON.stringify(window.__loadTimes));
}
// Scripted run: load, ask, mark done. In parallel mode every `ask` goes at once.
if (q.get("ask")) {
  if (!pipelines.length) await load();
  if (mode === "copies") await ask(q.getAll("ask").join("\n"));
  else for (const question of q.getAll("ask")) await ask(question);
  document.body.dataset.done = "1";
}
