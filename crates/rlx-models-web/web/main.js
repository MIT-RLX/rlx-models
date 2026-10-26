import { MODELS } from "./models.js";

// Served from the repo root (`just web-models-serve`): weights live at /weights.
const WEIGHTS = new URL("../../../weights", location.href).href.replace(/\/$/, "");
const SAMPLE = new URL("../../../assets/harness/jfk_16k.wav", location.href).href;

const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let kind = null;

for (const [id, m] of Object.entries(MODELS)) $("model").add(new Option(m.label, id));

worker.onmessage = ({ data }) => {
  if (data.type === "progress" || data.type === "ready" || data.type === "error") {
    $("status").textContent = data.type === "error" ? `error: ${data.text}` : data.text;
    if (data.type === "ready") for (const k of ["asr", "embed", "gen"]) $(k).classList.toggle("hidden", k !== kind);
  } else if (data.type === "result") {
    $("status").textContent = data.stats;
    if (data.sims) {
      const rows = data.sims.map((r, i) => `<tr><td>${i}</td>${r.map((v) => `<td>${v.toFixed(3)}</td>`).join("")}</tr>`);
      $("out").innerHTML = `<table>${rows.join("")}</table>`;
    } else {
      $("out").textContent = data.text;
    }
  }
};

$("load").onclick = () => {
  const id = $("model").value;
  kind = MODELS[id].kind === "qwen35" ? "gen" : MODELS[id].kind; // same panel
  for (const k of ["asr", "embed", "gen"]) $(k).classList.add("hidden");
  $("out").textContent = "";
  worker.postMessage({ type: "load", id, weightsBase: WEIGHTS, backend: $("backend").value });
};

// Any audio the browser can decode → 16 kHz mono Float32Array.
async function to16k(arrayBuffer) {
  const decoded = await new AudioContext().decodeAudioData(arrayBuffer);
  const frames = Math.ceil(decoded.duration * 16000);
  const ctx = new OfflineAudioContext(1, frames, 16000);
  const src = ctx.createBufferSource();
  src.buffer = decoded;
  src.connect(ctx.destination);
  src.start();
  return (await ctx.startRendering()).getChannelData(0);
}

async function transcribe(buf) {
  $("status").textContent = "transcribing…";
  const pcm = await to16k(buf);
  worker.postMessage({ type: "asr", pcm }, [pcm.buffer]);
}
$("wav").onchange = async (e) => transcribe(await e.target.files[0].arrayBuffer());
$("sample").onclick = async () => transcribe(await (await fetch(SAMPLE)).arrayBuffer());

$("run-embed").onclick = () => {
  const texts = $("texts").value.split("\n").map((s) => s.trim()).filter(Boolean);
  $("status").textContent = "embedding…";
  worker.postMessage({ type: "embed", texts, prompt: $("prompt").value });
};

$("run-gen").onclick = () => {
  // LFM2.5 chat template, as text; the tokenizer parses the special tokens.
  const prompt = `<|startoftext|><|im_start|>user\n${$("question").value}<|im_end|>\n<|im_start|>assistant\n`;
  $("status").textContent = "generating…";
  worker.postMessage({ type: "gen", prompt, maxTokens: Number($("max").value) });
};
