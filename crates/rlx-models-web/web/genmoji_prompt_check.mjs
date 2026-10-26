// `node genmoji_prompt_check.mjs`: genmoji_prompt.js's `assemble` against the
// output of the Rust `rlx_genmoji::assistant::assemble` for the same inputs.
import { assemble } from "./genmoji_prompt.js";

const cases = [
  ["emoji of flying pink pig. enlarged head in cartoon style. cute. white wings. head is turned towards viewer. 3D lighting. no cast shadows.", "animal", "flying pig",
    "emoji of flying pink pig. white wings. cute. enlarged head in cartoon style. head is turned towards viewer. 3D lighting. no cast shadows."],
  ["emoji of an owl in a dark forest. cute. 3D lighting.", "animal", "owl in dark forest",
    "emoji of an owl in a dark forest. cute. enlarged head in cartoon style. head is turned towards viewer. 3D lighting. no cast shadows."],
  ["emoji of fish swimming in the water. 3D lighting.", "animal", "fish",
    "emoji of fish swimming. cute. enlarged head in cartoon style. head is turned towards viewer. 3D lighting. no cast shadows."],
  ["emoji of a boat on the water, with sails. detailed texture.", "object", "a boat on the water",
    "emoji of a boat on the water, with sails. detailed texture. 3D lighting. no cast shadows."],
  ["emoji of teddy bear in space suit. cute.", "animal", "teddy bear in space suit",
    "emoji of teddy bear in space suit. cute. enlarged head in cartoon style. head is turned towards viewer. 3D lighting. no cast shadows."],
  ["a nurse with a stethoscope. head and shoulders portrait.", "person", "a nurse",
    "emoji of a nurse with a stethoscope. head and shoulders portrait. head is turned towards viewer. 3D lighting. no cast shadows."],
  ["emoji of a cake on a table in the garden at night, and candles.", "object", "cake",
    "emoji of a cake on a table in the garden and candles. detailed texture. 3D lighting. no cast shadows."],
  ["Emoji of rainbow popsicle. detailed texture.", "object", "rainbow popsicle",
    "Emoji of rainbow popsicle. detailed texture. 3D lighting. no cast shadows."],
];

let bad = 0;
for (const [rewrite, cat, user, want] of cases) {
  const got = assemble(rewrite, cat, user);
  if (got !== want) {
    bad++;
    console.log(`MISMATCH for ${JSON.stringify(user)}\n  want ${want}\n  got  ${got}`);
  }
}
console.log(bad ? `${bad} of ${cases.length} differ` : `all ${cases.length} match the Rust assemble`);
process.exit(bad ? 1 : 0);
