#!/usr/bin/env node
// Run after: wasm-pack build crates/rlx-wakeword-wasm --target nodejs --release
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

const pkg = require(join(__dirname, "pkg", "rlx_wakeword_wasm.js"));

const score = pkg.smoke_score();
if (!Number.isFinite(score) || score < 0 || score > 1) {
  console.error(`smoke_score failed: ${score}`);
  process.exit(1);
}
console.log(`smoke_score=${score.toFixed(4)} (ok)\n`);

const hop = Number(process.env.HOP_MS || 40);
const nMin = Number(process.env.N_MIN || 2);
const nMax = Number(process.env.N_MAX || 10);
const modes = process.env.MODES || "f32,tern-fc,tern-all";
const table = pkg.bench_multi_phrase(hop, nMin, nMax, modes);
process.stdout.write(table);
