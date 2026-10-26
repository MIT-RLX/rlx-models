// Open Genmoji's prompt side, as `crates/rlx-genmoji/src/assistant.rs` does it:
// the metaprompt conversation for the rewrite, the object / animal / person
// classifier with contextual calibration, and Genmoji-style assembly of the
// FLUX prompt. Kept in step with that file (`node genmoji_prompt_check.mjs`
// compares `assemble` against the Rust output).

/** Qwen3.5's chat template with thinking off (contents trimmed, as it does). */
export function chatPrompt(messages) {
  return (
    messages.map((m) => `<|im_start|>${m.role}\n${m.content.trim()}<|im_end|>\n`).join("") +
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
  );
}

/** The rewrite conversation: few-shot history, then the metaprompt with the description. */
export function rewriteMessages(metaprompt, history, userPrompt) {
  return [...history, { role: "user", content: `${metaprompt}\n\nUSER PROMPT: "${userPrompt}"` }];
}

/** Drop code fences, newlines and any `<think>…</think>` block. */
export function cleanReply(reply) {
  let s = reply;
  for (;;) {
    const a = s.indexOf("<think>");
    const b = s.indexOf("</think>");
    if (a < 0 || b < 0 || b < a) break;
    s = s.slice(0, a) + s.slice(b + "</think>".length);
  }
  return s.replaceAll("```", "").replaceAll("\n", "").trim();
}

export const CATEGORIES = ["object", "animal", "person"];

/** The classifier's few-shot question about `subject`. */
export function classifyMessages(subject) {
  return [
    {
      role: "user",
      content:
        "Classify the main subject of an emoji. Answer with one word: object, animal, or person.\n" +
        "A person is a human being in any job or role; an animal is a living creature or a " +
        "character based on one; an object is anything else.\n" +
        "taco -> object\nfireplace -> object\nrobber -> person\nteddy bear -> animal\n" +
        "dog chef -> animal\nastronaut -> person\nrainbow -> object\n" +
        `${subject} ->`,
    },
  ];
}

/**
 * Every first token each category word may open the reply with (lower case,
 * Capitalized, space-prefixed); a category scores its best spelling.
 */
export function categoryTokenIds(tokenizer) {
  return CATEGORIES.map((c) => {
    const spellings = [c, c[0].toUpperCase() + c.slice(1), ` ${c}`];
    return [...new Set(spellings.map((v) => tokenizer.encode(v)[0]).filter((t) => t !== undefined))];
  });
}

/** Log-softmax over the categories of each one's best-spelling logit. */
export function categoryScores(logits, idsPerCategory) {
  let k = 0;
  const best = idsPerCategory.map((ids) => {
    let m = -Infinity;
    for (let i = 0; i < ids.length; i++) m = Math.max(m, logits[k++]);
    return m;
  });
  const top = Math.max(...best);
  const lse = top + Math.log(best.reduce((s, l) => s + Math.exp(l - top), 0));
  return best.map((l) => l - lse);
}

/**
 * Contextual calibration: subtract what the model answers for a content-free
 * subject ("N/A"), which removes its prior toward one label.
 */
export function pickCategory(scores, baseline) {
  const cal = scores.map((s, i) => s - baseline[i]);
  return CATEGORIES[cal.indexOf(Math.max(...cal))];
}

const ADDONS = {
  object: ["detailed texture."],
  animal: ["cute.", "enlarged head in cartoon style.", "head is turned towards viewer."],
  person: ["head and shoulders portrait.", "head is turned towards viewer."],
};

const MANAGED = [
  "cute",
  "enlarged head in cartoon style",
  "head is turned towards viewer",
  "head turned towards viewer",
  "detailed texture",
  "3d lighting",
  "no cast shadows",
  "head and shoulders portrait",
];

const SETTINGS = [
  "background", "forest", "woods", "jungle", "ocean", "sea", "water", "river", "lake", "sky",
  "night", "room", "field", "meadow", "garden", "park", "desert", "beach", "city", "street",
  "mountains", "snow", "grass", "branch", "tree", "landscape", "scene",
];
const SETTING_PREPS = ["in", "on", "at", "under", "inside", "against", "among", "over"];

const bare = (w) => w.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "").toLowerCase();

/** Drop "in/on/… [up to two words] SETTING" from `phrase` unless the user asked for that setting. */
export function stripSettings(phrase, userPrompt) {
  const user = userPrompt.toLowerCase();
  const words = phrase.split(/\s+/).filter(Boolean);
  // The setting must end the clause: "in space suit" is not a place.
  const ends = (j) => j + 1 >= words.length || ["with", "and"].includes(bare(words[j + 1])) || words[j].endsWith(",");
  const out = [];
  let i = 0;
  while (i < words.length) {
    if (SETTING_PREPS.includes(bare(words[i]))) {
      let hit = -1;
      for (let j = i + 1; j < Math.min(i + 4, words.length); j++) {
        const w = bare(words[j]);
        if (SETTINGS.includes(w) && !user.includes(w) && ends(j)) {
          hit = j;
          break;
        }
      }
      if (hit >= 0) {
        i = hit + 1;
        continue;
      }
    }
    out.push(words[i]);
    i += 1;
  }
  return out.join(" ").replace(/,+$/, "");
}

/** The rewrite's description sentences, then the category's add-ons, then the style suffix. */
export function assemble(rewrite, category, userPrompt) {
  const parts = rewrite
    .split(".")
    .map((p) => p.trim())
    .filter((p) => p && !MANAGED.includes(p.toLowerCase()))
    .map((p) => stripSettings(p, userPrompt))
    .filter(Boolean)
    .map((p) => `${p}.`);
  if (!parts.length || !parts[0].toLowerCase().startsWith("emoji of")) parts.unshift("emoji of");
  parts.push(...ADDONS[category], "3D lighting.", "no cast shadows.");
  return parts.join(" ");
}
