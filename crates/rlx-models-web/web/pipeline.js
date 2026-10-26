// Qwen3.5 split across Web Workers by layer: worker i holds only stage i of N
// (a contiguous run of trunk layers), in its own wasm32 address space and on
// its own WebGPU device. Prompts are tokenized here; token ids go to the first
// worker, hidden rows (a Float32Array, 4 KB per decode step for 0.8B) from
// each worker to the next, and the last worker answers with the next token.
//
//   import { Qwen35Pipeline } from "./pipeline.js";
//   const p = await Qwen35Pipeline.create({ base: "/weights/lm/qwen3.5-0.8b-q4km", workers: 3 });
//   const text = await p.generate(prompt, 64, { onToken: (piece) => out.append(piece) });
//   p.terminate();
//
// `base` serves `model.gguf` and `tokenizer.json` — or pass `modelUrl` and
// `tokenizerUrl` directly, e.g. a GGUF on the Hugging Face Hub
// (`https://huggingface.co/<repo>/resolve/main/<file>.gguf`), which answers
// Range requests with CORS, so each worker streams only its own tensors from
// the Hub. `workers` is 1 (the whole
// model in one worker) up to the number of layer groups (6 for Qwen3.5-0.8B:
// 24 layers, a full-attention layer every 4). Each worker downloads only the
// header and its own tensors with HTTP Range requests (falling back to the
// whole file where the server ignores Range), so it never holds the rest.
// Decoding one sequence is still sequential — every token passes through
// every stage — so more workers buy memory headroom per worker, not speed.
import init, { TextTokenizer } from "./pkg/rlx_models_web.js";

// Requests in flight across all workers, by id: a pipeline step is sent to
// the first worker but answered by the last (or by whichever one fails).
class Requests {
  constructor() {
    this.pending = new Map();
    this.next = 0;
  }

  open() {
    const id = this.next++;
    const done = new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
    return { id, done };
  }

  settle(data) {
    const p = this.pending.get(data.id);
    if (!p) return;
    this.pending.delete(data.id);
    data.ok ? p.resolve(data) : p.reject(new Error(data.error));
  }

  failAll(err) {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }
}

class StageWorker {
  constructor(url, requests) {
    this.requests = requests;
    this.onDownload = () => {};
    this.worker = new Worker(url, { type: "module" });
    this.worker.onmessage = ({ data }) => {
      if (data.progress) this.onDownload(data.progress);
      else requests.settle(data);
    };
    this.worker.onerror = (e) => requests.failAll(new Error(e.message ?? "worker error"));
  }

  call(op, args = {}, transfer = []) {
    const { id, done } = this.requests.open();
    this.worker.postMessage({ id, op, ...args }, transfer);
    return done;
  }
}

export class Qwen35Pipeline {
  /**
   * @param {object} o
   * @param {string} [o.base]       URL of a directory holding model.gguf + tokenizer.json
   * @param {string} [o.modelUrl]   the GGUF itself (instead of `base`), e.g. on the Hugging Face Hub
   * @param {string} [o.tokenizerUrl]  its HF `tokenizer.json` (instead of `base`)
   * @param {number} [o.workers=2]  how many Web Workers (= pipeline stages) to engage
   * @param {string} [o.backend="webgpu"]  "webgpu" or "cpu"
   * @param {number} [o.maxSeq=256] tokens a conversation may hold (prompts + answers)
   * @param {boolean} [o.parallelLoad=false]  load every stage at once rather than one
   *   after another; faster, at the cost of every stage's load-time peak at once
   * @param {number} [o.rows=1]     positions per decode step. Above 1 every step
   *   runs that many rows (at somewhat more than one row's cost): prompts past
   *   the prefill go in `rows`-token chunks, and `generate({ speculate })`
   *   can check up to `rows - 1` guessed tokens per step
   * @param {number} [o.prefill=128]  prompt tokens the workers' prefill graph
   *   takes in one pass; the rest of a prompt goes a decode step (or `rows`) at
   *   a time. Wider costs more memory per worker but runs a long prompt (a
   *   metaprompt) in one pass
   * @param {object} [o.settings]   RLX_* settings applied in every worker
   * @param {(msg: string) => void} [o.onProgress]
   * @param {(index: number, fetched: number, total: number) => void} [o.onDownload]
   *   bytes of model.gguf each worker has downloaded so far, of what it needs
   */
  static async create({ base, modelUrl, tokenizerUrl, workers = 2, backend = "webgpu", maxSeq = 256, parallelLoad = false,
    rows = 1, prefill = 128, settings = {}, onProgress = () => {}, onDownload = () => {}, workerUrl = new URL("./pipeline_worker.js", import.meta.url) }) {
    if (!(Number.isInteger(workers) && workers >= 1)) throw new Error(`workers must be a positive integer, got ${workers}`);
    await init();
    modelUrl ??= `${base}/model.gguf`;
    tokenizerUrl ??= `${base}/tokenizer.json`;
    const tokRes = await fetch(tokenizerUrl);
    if (!tokRes.ok) throw new Error(`${tokenizerUrl}: HTTP ${tokRes.status}`);
    const tokenizer = new TextTokenizer(new Uint8Array(await tokRes.arrayBuffer()));
    const requests = new Requests();
    const stages = Array.from({ length: workers }, () => new StageWorker(workerUrl, requests));
    stages.forEach((s, i) => (s.onDownload = ({ fetched, total }) => onDownload(i, fetched, total)));
    const p = new Qwen35Pipeline(tokenizer, stages, backend);
    p.maxSeq = maxSeq;
    p.rows = rows;
    p.prefill = prefill;
    const load = async (index) => {
      const t0 = performance.now();
      const r = await stages[index].call("load", { url: modelUrl, index, count: workers, maxSeq, settings, backend, rows, prefill });
      p.loadMs[index] = Math.round(performance.now() - t0);
      p.fetchedMb[index] = r.fetched_mb;
      p.cachedMb[index] = r.cached_mb ?? 0;
      p.layers[index] = r.layers;
      onProgress(`stage ${index + 1}/${workers} ready (${r.layers ? `layers ${r.layers[0]}–${r.layers[1] - 1}, ` : ""}${p.loadMs[index]} ms, build ${r.build_ms} ms, ` +
        `fetched ${r.fetched_mb} MB${r.ranged ? " by Range" : " (whole file)"}` +
        `${r.cached_mb ? ` — ${r.cached_mb} MB from cache, ${r.downloaded_mb} MB downloaded` : ""}, wasm memory ${r.memory_mb} MB)`);
    };
    try {
      if (parallelLoad) await Promise.all(stages.map((_, i) => load(i)));
      else for (let i = 0; i < workers; i++) await load(i);
      // Chain the stages: stage i sends to stage i+1 over its own channel.
      const channels = Array.from({ length: workers - 1 }, () => new MessageChannel());
      await Promise.all(stages.map((s, i) => {
        const prev = i > 0 ? channels[i - 1].port2 : undefined;
        const next = i < workers - 1 ? channels[i].port1 : undefined;
        return s.call("link", { prev, next }, [prev, next].filter(Boolean));
      }));
    } catch (e) {
      p.terminate();
      throw e;
    }
    return p;
  }

  constructor(tokenizer, stages, backend) {
    this.tokenizer = tokenizer;
    this.stages = stages;
    this.backend = backend;
    this.maxSeq = 256;
    /** Tokens the workers' caches hold: the conversation so far (0 = none). */
    this.position = 0;
    // The last answer's final token when the answer was cut short (limit or
    // Stop): generated but never fed back, so the next turn feeds it first.
    this.pending = null;
    this.loadMs = [];
    /** MB of model.gguf each worker downloaded (only its own tensors where the server supports Range). */
    this.fetchedMb = [];
    /** Of that, MB each worker read from the browser's cache. */
    this.cachedMb = [];
    /** Each worker's trunk layers, [start, end), when it fetched by Range. */
    this.layers = [];
    this.eos = ["<|im_end|>", "<|endoftext|>"].map((t) => tokenizer.token_id(t)).filter((t) => t !== undefined);
    /** Positions per decode step (`create({ rows })`). */
    this.rows = 1;
    /** The prefill graph's width (`create({ prefill })`). */
    this.prefill = 128;
    /** Every token of the conversation so far, for drafting guesses. */
    this.context = [];
    /** Timings of the last `generate`: prefill_ms, decode_ms, tokens, tok_per_s. */
    this.stats = null;
  }

  /** Each worker's wasm32 linear memory in MB (it never shrinks: the peak so far). */
  async memoryMb() {
    return Promise.all(this.stages.map(async (s) => (await s.call("memory")).memory_mb));
  }

  /** Number of workers engaged. */
  get workers() {
    return this.stages.length;
  }

  // One pass through every stage; resolves to the last stage's token id.
  // One step through every stage: in at the first worker, along the chain,
  // out of the last. Resolves to the last stage's token id.
  async #pass(kind, first) {
    const { out } = await this.stages[0].call("run", { kind, input: first });
    if (typeof out !== "number") throw new Error("the last stage returned no token");
    return out;
  }

  // One multi-row step: `ids` (at most `rows`) enter the context; resolves to
  // the next token after each. `undo` lets `#undo` cut it back.
  async #passRows(ids, undo, guesses) {
    const { out } = await this.stages[0].call("run", { kind: "rows", input: Uint32Array.from(ids), undo, guesses });
    if (!(out instanceof Uint32Array)) throw new Error("the last stage returned no tokens");
    return Array.from(out);
  }

  // Keep only the first `keep` rows of the last undoable step, in every stage.
  #undo(keep) {
    return Promise.all(this.stages.map((s) => s.call("rowsUndo", { keep })));
  }

  // Up to `n` tokens guessed to come next: what followed the most recent
  // earlier place the conversation's last 3 (else 2) tokens occurred. A
  // single repeated token predicts too little: its guesses mostly fail, and a
  // failed guess costs a step's re-run.
  #draft(n) {
    const c = this.context;
    for (let k = Math.min(3, c.length - 1); k >= 2; k--) {
      const tail = c.slice(c.length - k);
      for (let i = c.length - k - 1; i >= 0; i--) {
        let hit = true;
        for (let m = 0; m < k && hit; m++) hit = c[i + m] === tail[m];
        if (hit) return c.slice(i + k, i + k + n);
      }
    }
    return [];
  }

  /**
   * Continue `prompt` (special tokens as written; no chat template is
   * applied) and return the answer. Stops after `maxTokens`, at an
   * end-of-turn token unless `stopAtEos` is false, or when `signal` (an
   * AbortSignal) fires — checked between tokens, so what was generated so far
   * is returned and `stats.stopped` is set.
   *
   * `append`: continue the conversation already in the workers' caches
   * (`position` tokens) instead of starting a new one — only the new turn's
   * tokens are processed. Prompts of any length (up to the context) work: the
   * first `prefill` tokens of a new conversation go through the prefill graph, the
   * rest one decode step each.
   *
   * `sampling`: null (default) takes the most likely token every step;
   * `{ temperature, topP, topK, presencePenalty, seed? }` samples instead (a
   * random seed when none is given). Greedy decoding makes a small model
   * repeat itself and often never end its turn; Qwen suggests 0.7 / 0.8 / 20 /
   * 1.5 for non-thinking chat. Prompt tokens are never sampled, and the
   * sampler's penalties see only the answer's tokens.
   *
   * `stopAt`: more token ids that end the answer like end of turn (not shown,
   * not fed), e.g. `</think>` to stop after the reasoning; `endToken` says
   * which one ended it.
   */
  async generate(prompt, maxTokens = 64, { onToken = () => {}, stopAtEos = true, signal, sampling = null, append = false,
    speculate = false, stopAt = [] } = {}) {
    if (this.busy) throw new Error("a generation is already in flight");
    this.busy = true;
    try {
      const last = this.stages[this.stages.length - 1];
      const setSampling = (opts) =>
        last.call("sampling", {
          opts: opts && {
            temperature: opts.temperature ?? 0.7,
            topP: opts.topP ?? 1,
            topK: opts.topK ?? 0,
            presencePenalty: opts.presencePenalty ?? 0,
            seed: opts.seed ?? crypto.getRandomValues(new Uint32Array(1))[0],
          },
        });
      let prompt_ids = Array.from(this.tokenizer.encode(prompt));
      if (!append || this.position === 0) {
        this.position = 0;
        this.pending = null;
        this.context = [];
      } else if (this.pending !== null) {
        // Already in `context`: it was shown.
        this.context.push(...prompt_ids);
        prompt_ids.unshift(this.pending);
      }
      if (this.pending === null) this.context.push(...prompt_ids);
      this.pending = null;
      /** The token that ended the last answer (end of turn or a `stopAt` id); null otherwise. */
      this.endToken = null;
      if (!prompt_ids.length) throw new Error("empty prompt");
      const room = this.maxSeq - 1 - this.position - prompt_ids.length;
      if (room < 1) {
        throw new Error(`the conversation is full (${this.position} of ${this.maxSeq} tokens): start a new one`);
      }
      maxTokens = Math.min(maxTokens, room);
      // A multi-row step needs room for all its rows, padding included.
      if (this.rows > 1) {
        const rowsRoom = this.maxSeq - this.rows - this.position - prompt_ids.length + 1;
        if (rowsRoom < 1) {
          throw new Error(`the conversation is full (${this.position} of ${this.maxSeq} tokens): start a new one`);
        }
        maxTokens = Math.min(maxTokens, rowsRoom);
      }

      const t0 = performance.now();
      // Every prompt token but the last is only fed (greedy, its output
      // unused); sampling starts at the last, whose output is the answer's
      // first token.
      await setSampling(null);
      let rest = prompt_ids;
      let tok;
      if (this.position === 0) {
        const head = rest.slice(0, Math.min(this.prefill, rest.length - 1 || 1));
        rest = rest.slice(head.length);
        if (rest.length === 0) await setSampling(sampling);
        tok = await this.#pass("prefill", Uint32Array.from(head));
        this.position += head.length;
      }
      if (this.rows > 1) {
        // The rest of the prompt `rows` tokens per step.
        for (let i = 0; i < rest.length; i += this.rows) {
          const chunk = rest.slice(i, i + this.rows);
          if (i + this.rows >= rest.length) await setSampling(sampling);
          const out = await this.#passRows(chunk, false);
          tok = out[out.length - 1];
          this.position += chunk.length;
        }
      } else {
        for (let i = 0; i < rest.length; i++) {
          if (i === rest.length - 1) await setSampling(sampling);
          tok = await this.#pass("decode", Uint32Array.of(rest[i]));
          this.position += 1;
        }
      }
      const t1 = performance.now();
      // Tokens that end the answer, unshown and unfed: end of turn, and any
      // `stopAt` ids (e.g. `</think>` to end a reasoning phase).
      const stops = [...(stopAtEos ? this.eos : []), ...stopAt];
      if (this.rows > 1) {
        return this.#answerRows(tok, prompt_ids.length, maxTokens, t0, t1,
          { onToken, stops, signal, speculate });
      }

      const ids = [];
      // The few slowest decode steps as [ms, position] — first use of a
      // context rung shows up here.
      const slowest = [];
      let shown = "";
      let stopped = false;
      let ended = false;
      for (;;) {
        if (stops.includes(tok)) {
          ended = true;
          this.endToken = tok;
          break;
        }
        if (signal?.aborted) {
          stopped = true;
          break;
        }
        ids.push(tok);
        const text = this.tokenizer.decode(Uint32Array.from(ids));
        onToken(text.slice(shown.length), tok);
        shown = text;
        if (ids.length >= maxTokens) break;
        const ts = performance.now();
        tok = await this.#pass("decode", Uint32Array.of(tok));
        slowest.push([Math.round(performance.now() - ts), this.position]);
        if (slowest.length > 6) slowest.sort((a, b) => b[0] - a[0]).pop();
        this.position += 1;
      }
      // At the token limit the last shown token was never fed back: the next
      // turn feeds it first. (Stop exits before showing the newest token, so
      // everything shown was fed; an end-of-turn token is not fed either —
      // the next turn's text starts with <|im_end|>.)
      if (!ended && !stopped && ids.length) this.pending = ids[ids.length - 1];
      const t2 = performance.now();
      const steps = Math.max(ids.length - 1, 0);
      this.stats = { tokens: ids.length, prompt_tokens: prompt_ids.length, prefill_ms: Math.round(t1 - t0),
        decode_ms: Math.round(t2 - t1), tok_per_s: steps ? +(steps / ((t2 - t1) / 1e3)).toFixed(2) : 0,
        stopped, ended, position: this.position, slowest: slowest.sort((a, b) => b[0] - a[0]).slice(0, 5) };
      return this.tokenizer.decode(Uint32Array.from(ids));
    } finally {
      this.busy = false;
    }
  }

  // The answer loop of a multi-row pipeline: each step feeds the newest token
  // and, with `speculate`, up to `rows - 1` guesses after it (`#draft`); the
  // guesses the model agrees with are kept, and the step is cut back to them.
  // As in `generate`, the last token shown is not fed until the next step (or
  // turn), and an end-of-turn token is never fed.
  async #answerRows(tok, promptTokens, maxTokens, t0, t1, { onToken, stops, signal, speculate }) {
    const ids = [];
    let shown = "";
    let stopped = false;
    let ended = false;
    let steps = 0;
    let guessed = 0;
    let kept = 0;
    const show = (t) => {
      ids.push(t);
      this.context.push(t);
      const text = this.tokenizer.decode(Uint32Array.from(ids));
      onToken(text.slice(shown.length), t);
      shown = text;
    };
    // `next`: tokens produced but not yet shown, in order. `fedRows`: rows
    // the last step fed (every one kept so far; 0 before the first step).
    let next = [tok];
    let fedRows = 0;
    for (;;) {
      let count = next.length;
      const e = next.findIndex((t) => stops.includes(t));
      if (e >= 0) {
        count = e;
        ended = true;
        this.endToken = next[e];
      }
      const room = maxTokens - ids.length;
      const limited = count > room;
      if (limited) count = room;
      // Rows the last step keeps: the token it fed first and every shown
      // guess but the last (fed next) — or, when the answer ends here, through
      // the last shown. One undo covers rejected guesses, the end and the limit.
      if (fedRows) {
        const keep = ended ? count + 1 : Math.max(count, 1);
        if (keep < fedRows) await this.#undo(keep);
        this.position += Math.min(keep, fedRows);
      }
      next.slice(0, count).forEach(show);
      if (ended) break;
      if (limited || ids.length >= maxTokens) {
        this.pending = ids[ids.length - 1];
        break;
      }
      if (signal?.aborted) {
        stopped = true;
        this.pending = ids[ids.length - 1];
        break;
      }
      tok = ids[ids.length - 1];
      const drafts = speculate ? this.#draft(Math.min(this.rows - 1, maxTokens - ids.length - 1,
        this.maxSeq - 1 - this.position - this.rows)) : [];
      // The last stage judges the guesses (it sees hidden rows, not ids).
      const out = await this.#passRows([tok, ...drafts], drafts.length > 0, drafts);
      steps += 1;
      let j = 0;
      while (j < drafts.length && drafts[j] === out[j]) j++;
      guessed += drafts.length;
      kept += j;
      next = [...drafts.slice(0, j), out[j]];
      fedRows = 1 + drafts.length;
    }
    const t2 = performance.now();
    const decoded = Math.max(ids.length - 1, 0);
    this.stats = { tokens: ids.length, prompt_tokens: promptTokens, prefill_ms: Math.round(t1 - t0),
      decode_ms: Math.round(t2 - t1), tok_per_s: decoded ? +(decoded / ((t2 - t1) / 1e3)).toFixed(2) : 0,
      stopped, ended, position: this.position, steps, guessed, kept };
    return this.tokenizer.decode(Uint32Array.from(ids));
  }

  /**
   * Logits of each of `candidates` (token ids) as the next token after
   * `prompt` (special tokens as written; no chat template is applied). The
   * whole prompt goes through the prefill pass — a decode step's logits are
   * not scaled like the full model's — so it must fit the prefill graph
   * (`create({ prefill })` tokens). Starts a new conversation and leaves none behind.
   */
  async score(prompt, candidates) {
    if (this.busy) throw new Error("a generation is already in flight");
    this.busy = true;
    try {
      const ids = this.tokenizer.encode(prompt);
      if (!ids.length || ids.length > this.prefill) {
        throw new Error(`score takes 1–${this.prefill} prompt tokens, got ${ids.length}`);
      }
      this.reset();
      const last = this.stages[this.stages.length - 1];
      await last.call("sampling", { opts: null });
      await this.#pass("prefill", Uint32Array.from(ids));
      const { scores } = await last.call("scores", { ids: Uint32Array.from(candidates) });
      return Array.from(scores);
    } finally {
      this.reset();
      this.busy = false;
    }
  }

  /** Forget the conversation: the next `generate` starts a new one. */
  reset() {
    this.position = 0;
    this.pending = null;
    this.context = [];
  }

  terminate() {
    for (const s of this.stages) s.worker.terminate();
    this.stages = [];
  }
}
