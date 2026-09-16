# rlx-translate

the OS's on-device machine translation, natively on RLX.

macOS 15+ ships `Translation.framework`. Its low-latency path is a classic NMT
stack — a sed-script normalizer, SentencePiece, a multilingual Espresso
encoder-decoder under beam search, and a stack of pre/post-processing blocks —
all described by a JSON **Quasar** config that the OS installs *separately from
the weights*.

```
text ──normalizer (sed script)──▶ tokenizer ──▶ phrasebook lookup
     ──SentencePiece encode────▶ [control tokens] + ids
     ──Espresso NMT, beam 3────▶ target ids
     ──SentencePiece decode────▶ structured prediction ──▶ case map ──▶ text
```

This crate ships and redistributes **no model data**. Everything is read from
what the operating system installed.


## Contents

- **[Using it](#using-it)** — [Status](#status), [The weights are a second download](#the-weights-are-a-second-download), [Installing without the UI](#installing-without-the-ui), [Variants matter](#variants-matter), …
- **[How it works](#how-it-works)** — [`pyespresso.mdl.bin` is a manifest, not a weight container](#pyespressomdlbin-is-a-manifest-not-a-weight-container), [Two engines](#two-engines), [Direction is a token, not a model](#direction-is-a-token-not-a-model), [The token protocol](#the-token-protocol), …
- **[What it scores](#what-it-scores)** — [Measured against the OS's live output](#measured-against-the-oss-live-output), [The NMT works](#the-nmt-works), [How good is it, not just how faithful?](#how-good-is-it-not-just-how-faithful), [The metric was not chrF](#the-metric-was-not-chrf), …
- **[Defects found, and how](#defects-found-and-how)** — [Fourteen of the twenty shortlists were dead](#fourteen-of-the-twenty-shortlists-were-dead), [What the shortlist fix was worth](#what-the-shortlist-fix-was-worth), [`input_<lang>` follows the source, not the target](#inputlang-follows-the-source-not-the-target), [The source was truncated at 64 tokens, silently](#the-source-was-truncated-at-64-tokens-silently), …
- **[Speed](#speed)** — [What decoding actually costs](#what-decoding-actually-costs), [Where the remaining time goes](#where-the-remaining-time-goes), [The beam was wider than the OS's, and it bought nothing](#the-beam-was-wider-than-the-oss-and-it-bought-nothing), [`rs-beam` finally wired up](#rs-beam-finally-wired-up)
- **[License](#license)** — 

## Using it

### Status

| Area | State |
| --- | --- |
| Quasar config reader (`quasar`) | done — all 416 shipped configs parse |
| Decode parameters (`pdec`) | done — 3520 translator blocks |
| Pipeline plan / stage DAG (`pipeline`) | done — 2176 pair graphs ordered |
| Graph executor (`execute`) | done — `translate` runs the shipped 30-stage DAG |
| Normalizer sed engine (`normalizer`) | done — runs the shipped `.pat` scripts |
| Phrasebook (`phrasebook`) | done |
| Sentence casing (`casemap`) | done |
| Quality estimator (`quality`) | done — repeat regex + hallucinated-profanity check |
| Do-not-translate (`dnt`) | done — identifier spans, learned by probing the OS |
| Output punctuation (`postproc`) | done |
| SentencePiece vocabulary (`spm`) | done — 168 000 pieces |
| Espresso graph reader (`net`) | done — all 9 installed graphs |
| CPU evaluator (`exec`) | done — all 11 layer types |
| Export to safetensors (`export`) | done — verified at cosine 0.999993 |
| Convert to safetensors + GGUF + `.rlxp` (`convert`) | done — 7 bundles, 96 graphs, 6215 tensors |
| Asset discovery (`assets`) | done |
| SentencePiece encoder (`spm::encode`) | done — unigram Viterbi + byte fallback |
| NMT decode driver (`decode`) | done — 0.002 chrF behind the OS on human references |
| chrF (`score::chrf`) | verified against sacreBLEU 2.6.0 |
| Tuning surface (`tuning`) | done — one struct, env or `key=value`, `rlx-translate tune` |
| Examples | 12, all runnable with no arguments |

Not executed, and passing their input through: `PDecForceAlign`,
`StructuredPrediction`, `AlignmentProcessor`, `LinkAlternatives`. All four are
annotators that add metadata rather than change text; `rlx-translate plan
<pair>` marks them `todo`.

**Coverage.** `tests/all_pairs.rs` drives every ordered direction the machine
can feed: **397 of 397 produce a translation, none fails** (the other 19 are
`en_GB` pairings the OS itself rejects as `unsupportedLanguagePairing`). On the
114 of those with reference text, 78 are identical to the OS. The sweep takes
297 s, against 4334 s before the decoding work in
[Speed](#speed).

**Requirements.** macOS 15 or later, and at least one language pair installed
(see [The weights are a second download](#the-weights-are-a-second-download)).
No model data ships with this crate. Every test that needs an installed pair
skips with a message when one is absent, so `cargo test -p rlx-translate` is
green on a machine with nothing installed — it just proves less.

**Against the OS's live output: 653 of 855 sentences byte-identical (76%), mean
chrF 0.943, over 57 directions** — every direction the reference corpus covers,
including the 14 that pivot through English. `vi_VN-es_ES` is 15 of 15. Two defects found late account for most of
that — the shortlist tables were being misread, and `input_<lang>` was keyed by
the wrong language:

(pre-metric-fix figures)

| | mean chrF | identical / 645 |
| --- | --- | --- |
| before both | 0.788 | 206 |
| + shortlist fix | 0.818 | 232 |
| + input-graph fix | 0.959 | 509 |
| + config beam and `rs-beam` | **0.960** | 506, in a quarter of the time |

(That table is the 43 single-hop directions, held fixed for comparison. Adding
the 14 pivots brings it to 855 sentences at chrF 0.955.)

**230 tests green** (160 unit, 70 integration), clippy and fmt clean.

### The weights are a second download

The config asset (`com.apple.MobileAsset.UAF.Translation.Assets`, ~33 MB)
arrives on its own and contains **no weights** — only the per-pair configs, a
language identifier and an endpointer. The model files live in separately named
assets that appear only once a language is installed:

* `MT-bi-en-es-de-it-fr-pt-nl-20` — one multilingual bidirectional model covering
  en/es/de/it/fr/pt/nl, holding `MT/spm.model`, `MT/pyespresso.mdl.bin`, the
  Espresso graphs, `MT/normalizer.pat`, `MT/tokenizer.pat`,
  `MT/gender_defaults_list` and the quality-estimator tables.
* `MT-bi-…-partial-<lang>-20` — that language's decoder, handover and input graphs.
* `PB-<lang>` — per-source-language phrasebooks.

### Installing without the UI

`tools/install-language.m` asks the OS to fetch its own assets, so a machine can
be provisioned from a script:

```sh
clang -fobjc-arc -framework Foundation -o install-language \
    crates/rlx-translate/tools/install-language.m
./install-language                # list locales and their state
./install-language en_US fr_FR    # install (this REPLACES the current set)
```

Expect ~700 MB per locale: a locale pulls its MT models plus the much larger ASR
and TTS assets for that language.

MobileAsset's own API will not do this. `MAAssetQuery`/`MAAsset` refuse
`com.apple.MobileAsset.UAF.Translation.Assets` unless the caller holds the private
entitlement `com.apple.private.assets.accessible-asset-types` —
`queryMetaDataSync` returns 5 and `startCatalogDownload:` returns 12 — and that
cannot be self-granted under SIP. `_LTDLanguageAssetService` in
`TranslationDaemon.framework` sits above that gate and works from an ordinary
unsigned process.

System Settings → General → Language & Region → *Translation Languages* does the
same thing by hand.

### Variants matter

A pair ships several config variants and **each points at a different model
asset**: variant `20` at `MT-bi-…-20`, variant `0` at `MT-bi-…-0`. Only one is
normally installed, so `Assets::best_config` picks the variant whose files
actually resolve rather than the lowest number. `Assets::resolve` deliberately
has no "search every root for the tail of the path" fallback — variants share
inner paths (`MT/spm.model`), so that fallback silently returns the wrong model.

On disk an asset is an opaque `<sha1>.asset/AssetData/`; the logical name lives
in its `Info.plist` as `AssetSpecifier` (`com.apple.sequoia.asset.mt.bi-…-fr.20`)
and is matched against the config's spelling (`MT-bi-…-partial-fr-20`) through
`normalize_asset_name`.

### Usage

```sh
rlx-translate status                     # what this machine has installed
rlx-translate pairs                      # pairs with a config present
rlx-translate directions                 # every ordered direction (416 here)
rlx-translate plan en_US-fr_FR           # resolved 30-stage pipeline
rlx-translate config en_US-fr_FR         # dump the pair's blocks
rlx-translate translate en_US-fr_FR "I love the summer"
rlx-translate bench                      # phrasebook + NMT on the OS's own gold
rlx-translate nbest <src-tgt> <text> [n]  # the n best translations, with scores
rlx-translate parity <ref-dir>           # score against a live-output dump
rlx-translate export <out-dir> fr        # safetensors + model.json
rlx-translate normalize <file> "text"
rlx-translate probe <pyespresso.mdl.bin>
rlx-translate tune                       # every search setting, and its value
rlx-translate convert <out-dir> [f16|f32|q8_0]   # every bundle, all three formats
```

Any subcommand also takes search settings as trailing `key=value` arguments —
`rlx-translate nbest en_US-fr_FR "the sea is warm" 3 beam=16`. Set
`RLX_TRANSLATE_ASSETS` to a colon-separated list of extra asset roots to read a
copied asset tree instead of the live system one.

#### From Rust

The model on its own — encode, search, readout:

```rust,no_run
use rlx_translate::{assets::Assets, decode::Nmt, pdec::PDecParams,
                    quasar::LangPair, spm::Vocab};

let assets = Assets::discover();
let pair = LangPair::parse("en_US-fr_FR")?;
let (_, config) = assets.best_config(&pair)?;
let params = PDecParams::from_block(config.mt_app()?.translator(&pair)?)?;
let home = assets.model_home(&pair).expect("weights installed");
let vocab = Vocab::load(home.join("spm.model"))?;

// `input_<lang>` follows the *source* language, the other two the target.
let nmt = Nmt::load_for_pair(&home, &[home.as_path()], "en", "fr",
                             &params.shortlist.lang_pair)?;
for v in nmt.translate_nbest(&vocab, &params, "the sea is warm", 3)? {
    println!("{:>8.3}  {}", v.normalized_score, v.text);
}
# Ok::<(), anyhow::Error>(())
```

The whole pipeline — phrasebook, casing, do-not-translate, quality flags — is a
[`pipeline::TranslationPlan`] run by [`execute::run`]; `src/bin/rlx_translate.rs`
is the worked example, and `tests/graph_agreement.rs` shows the same wiring with
per-stage models so pivot directions work.

### Examples

Twelve, and none of them is a toy: each one exists because a question came up
that the code could not answer by inspection. Several found defects that are
now fixed, and the example is what would catch a regression.

```sh
cargo run --release -p rlx-translate --example <name> [-- <args>]
```

**Measurement**

| example | question it answers |
| --- | --- |
| `bench_human` | How good is it, not just how faithful? Scores ours *and* the OS's against human references. Needs `RLX_TRANSLATE_FLORES` or `RLX_TRANSLATE_HUMAN`, plus `RLX_TRANSLATE_REFERENCE`. |
| `sweep_all` | Every direction the machine supports — 378 of them — against human references. No reference output needed. |
| `why_missed` | For each disagreement, does our own model prefer the OS's answer? If yes the search failed; if no, searching harder cannot help. |
| `profile_decode` | Where decoding time goes, phase by phase. `RLX_TRANSLATE_PROFILE=1` adds a per-op and per-graph table. |
| `dot_bench` | How fast the int8 dot product and GEMV actually are. Read this before writing intrinsics — a NEON version lost to the plain loop. |

**Probes** — these ask the model or the graph a question directly

| example | question it answers |
| --- | --- |
| `graph_langs` | Does each per-language graph follow the source or the target? Found that `input_<lang>` follows the **source**, worth chrF 0.818 → 0.959. |
| `tag_placement` | Where do the direction tags belong in the source sequence? |
| `control_tokens` | What a direction's control tokens resolve to, whether the vocabulary has them, and whether its shortlist loads. |
| `alignment_probe` | What the cross-attention looks like, and that it is shifted by one step. |
| `fallback_scan` | Which languages fall back to bytes, and how often. Thai 56.5% of sentences, French 39%. |
| `batch_probe` | Whether the decoder graph tolerates more than one row. It does not: `reshape_2` is `[8, 1, 64]`. |
| `pick_nonlexicon` | Picks candidate sentences the phrasebook does not cover, for building a corpus that exercises the NMT. |

### Tuning the search

Every setting that changes *what the search does* lives in one struct,
`tuning::Tuning`, and is reachable three ways: a `key=value` argument on any
subcommand, the matching `RLX_TRANSLATE_*` variable, or `Nmt::tuning_mut`.

```sh
rlx-translate tune                                   # what is in effect
rlx-translate nbest en_US-fr_FR "the sea is warm" 3 beam=16
RLX_TRANSLATE_NO_REPEAT=0 rlx-translate bench en_US-fr_FR
```

A value prints as `config` when the OS's own block for the pair supplies it and
nothing has overridden it, and `auto` when it is derived from how many results
were asked for. An unrecognised name is an error rather than a no-op: a typo in
a sweep script would otherwise read as "this lever does nothing", which is a
conclusion this port has already drawn wrongly once.

`rlx-translate tune` also lists the settings that change how an answer is
*produced* rather than what it is — `RLX_TRANSLATE_PROFILE` for the per-op
table, `RLX_TRANSLATE_GEMM_LANES`, `RLX_TRANSLATE_INCREMENTAL` — so one command
answers "what can I set?".

The two that were hardest to find are `beam_multiple` and `stop_after`, because
the search's `nbest` means two things at once — how many results to return, and
how many finished hypotheses to collect before stopping. Over-generating for
deduplication therefore also made every translation run to its full 80-step
length budget. Splitting them is most of the speedup below.

### Converting the whole installation

`rlx-translate convert <out-dir> [f16|f32|q8_0]` walks every installed bundle
and writes all three formats plus every configuration:

```
converting 7 bundles as F16 -> /Volumes/FOUR/rlx-translate-export
  mt-ar-en                 9 graphs,  583 tensors | safetensors  640 MB, gguf  320 MB, rlxp  530 MB
  mt-de-en-es-fr-it-nl-pt 24 graphs, 1553 tensors | safetensors 1857 MB, gguf  929 MB, rlxp 1780 MB
  mt-en-hi                 9 graphs,  583 tensors | ...
  mt-en-id-th-vi          15 graphs,  971 tensors | ...
  mt-en-ja-ko-zh          15 graphs,  971 tensors | ...
  mt-en-pl-ru-uk          15 graphs,  971 tensors | ...
  mt-en-tr                 9 graphs,  583 tensors | ...
  configs ... 416 files
```

**7 bundles · 96 graphs · 6215 tensors · 1.79 G parameters · 0 missing**, over
20 languages. safetensors 7.16 GB, GGUF (F16) 3.58 GB, `.rlxp` 6.38 GB.

| format | granularity | holds |
| --- | --- | --- |
| `.safetensors` | one per graph, f32 | what `rlx_core::weight_map::WeightMap` reads |
| `.gguf` | one per bundle | every tensor, plus the graph description as the `quasar.model_json` metadata key — so a `.gguf` alone is enough to rebuild the model |
| `.rlxp` | one per bundle | both of the above plus `spm.model`, the normalizer and tokenizer sed scripts, the shortlists and the manifest |

All three come from one [`export::stage_graph`] staging, so they cannot drift
apart in content — only in encoding. `quasar-configs.rlxp` holds all 416 Quasar
configs.

### Two defects the export carried, and the tests that now catch them

Writing a file is not evidence that it holds the right numbers, and a release is
where that stops being a slogan. Producing the full export and reading it back
found two:

- **`decoder_layers` was `null` in every descriptor ever written.** The count is
  *inferred* from `StateStrings`; the converter asked the manifest for a
  `DecoderLayers` key that does not exist. A consumer reading `model.json` could
  not have built the decoder.
- **`all-zh_TW.shortlist` was never exported.** The converter built the filename
  as `all-<lang>` — the same assumption that had already been fixed in the
  *loader* and not here. Its absence does not fail: it translates Traditional
  Chinese into Simplified.

Copying every table in sight fixed the second and cost 2.3 GB of tables no
bundle can use, so the rule is the bundle's own languages *including regional
variants*: `all-zh` **and** `all-zh_TW`, not `all-fr`.

`tests/convert_roundtrip.rs` now asserts that every descriptor field is present
and non-zero, that every installed shortlist reaches the export, and that every
`.rlxp` unpacks with byte-identical safetensors inside — not just the first one.
Point `RLX_TRANSLATE_EXPORT` at the output of `rlx-translate convert`:

```
416 configs exported, 416 installed
20 shortlist tables, all exported
7 descriptors complete
7 packs verified, 6215 tensors compared, worst relative difference 4.88e-4
```

### F16 is not lossy here, and that is measured

Every linear weight is `W_int8 / w_quantization_scale` — at most 255 distinct
values over one scalar range. F16 carries ~11 significant bits, so its relative
error sits an order of magnitude below the int8 step the weights already had.
`tests/convert_roundtrip.rs` reads the GGUF back and compares it against the
safetensors it was staged from:

```
compared 6215 tensors; worst relative difference 4.88e-4
```

against an int8 step of ~7.9e-3. Storing f32 would record precision the shipped
model never had, which is why GGUF defaults to F16 and is half the size. Pass
`f32` if you want the staging verbatim, or `q8_0` for block quantization (rows
whose length is not a multiple of the block size fall back to F32 rather than
being silently mis-encoded).

The same tests check that the `.rlxp` unpacks to the directory it packed and
that all 416 configs survived the round trip.

## How it works

### `pyespresso.mdl.bin` is a manifest, not a weight container

Despite the name and the config's `"model-type": "espresso"`, the file holds no
weights — it is ~77 KB of typed key/value records naming graphs and tensors. The
weights sit beside it in the **classic Espresso triple** (`*.espresso.net` /
`.shape` / `.weights`), the same container `rlx-neuralhash` already reads.

Framing: one leading `0x00`, then `<key><space><value>` where a bool is a single
`T`/`F` byte with no separator, an int is tag `0x04` plus a little-endian u32,
and a string runs to the next space. Because bools run straight into the
following key (`…IsEspresso TSourceInputStr…`) the stream can only be walked with
a schema — `espresso::KEYS` — and an unknown key is an error rather than a guess.

`rlx-translate probe <file>` prints the decoded manifest. For the shipped model:

```
engine: CPU
decoder layers: 3
target languages: ["de", "en", "es", "fr", "it", "nl", "pt"]
EncoderGraph encoder.espresso.net   EmbeddingGraph embedding.espresso.net
ReadoutGraph readout.espresso.net
SourceInputStr src_tokens           TargetInputStr prev_output_tokens
EncoderValuesStr encoder.15.output  ScoresStr final_layer_output
HandoverStrings decoder.{0,1,2}.encoder_attn.{key,value}_transpose
StateStrings    decoder.{0,1,2}.self_attn.accum
StateWidth=512  AlignmentHeads=1  AddSrcEos=true  ApplyLog=true
```

So: a shared encoder (396 layers = 12 int8 attention blocks) and embedding
(`quantized_gather` of tokens and positions), a per-target-language **3-layer**
decoder whose cross-attention K/V are precomputed once per sentence
("handover") with incremental self-attention state (`accum`), and a readout that
gathers the tied embedding and applies softmax.

The trailing `<InputSymbolTable>` framing is not decoded yet;
`Manifest::symbol_table_offset` reports where it begins.

### Two engines

`TranslationSession.Strategy` (macOS 26.4) selects between them:

* `.lowLatency` — the Espresso NMT this crate targets.
* `.highFidelity` — **not an NMT**. It prompts the on-device foundation model
  (`com.apple.fm.language.instruct_3b.machine_translation.generic`) with a fixed
  "You are an expert translator…" instruction. Out of scope here.

### Direction is a token, not a model

One model serves every pair in its bundle. Direction comes from control tokens,
NLLB-style. the OS pre-joins two target tags with `> <`:

```
source-token = "en_US"
target-token = "fr_FR> <en_US-fr_FR-optimal"   ← target locale + variant selector
```

`PDecParams::target_tokens` splits them back out.

Gender alternatives ride in-band as `<STRUCT_FEMALE_BEG>`, `<STRUCT_MALE_BEG>`,
`<STRUCT_MID>`, `<STRUCT_END>`, suppressed per-direction via
`shortlist-suppress-tokens`.

### The token protocol

The config's raw value is wrapped as `<src-{v}>` / `<tar-{v}>`, and the `> <`
join exists precisely so that wrapping yields **two valid pieces**:

```
source-token "en_US"                       -> <src-en_US>            (718)
target-token "fr_FR> <en_US-fr_FR-optimal" -> <tar-fr_FR>            (729)
                                            + <en_US-fr_FR-optimal>  (252)
```

`<en_US>` and `<fr_FR>` do not exist; `<src-*>` and `<tar-*>` do. All three tags
go on the **source**. Then the part that is not written down anywhere:

> **`<s>` (id 1) is the sequence terminator, on *both* sides, and also the
> decoder's BOS.** `</s>` (id 2) is essentially unused by this model.

So the decoder opens with `<s>` and stops on `<s>`, and the source — which
`AddSrcEos T` says to terminate — is terminated with `<s>` too. Both halves were
worth a factor of 20 or more, measured as mean log-rank of the OS's next piece:

```
decoder opens with <s>              0.554   vs  2.251 for the language tag
source terminated with <s>          0.014   vs  0.574 for </s>   (20/21 top-1)
```

Suppressing `<s>` as a control token — the obvious thing to do — leaves the model
unable to stop, and it loops forever emitting a correct translation followed by
garbage. That single mistake accounted for most of the remaining gap.

### Running the graph, not a hand-wired order

`translate` used to call the phrasebook and then the NMT in an order written
into a `match`. It now walks the config's own DAG through
[`execute::run`], so the fan-in and the precedence are the OS's:

```
$ rlx-translate translate en_US-fr_FR "my neighbour bought seventeen blue umbrellas"
graph      : 30 stages, phrasebook 63486 sources
produced by: do_not_translate
translation: Mon voisin a acheté dix-sept parapluies bleus
```

A stage evaluates to `Option<String>`, which is enough because `MergerBlock` is
a coalesce. Two details had to be right, and each produced a wrong answer first:

**Named inputs.** `DoNotTranslateBlock` receives
`{"target": ..., "source": "graph-input"}`. Roles sort before values, so "first
input" reads *source* and returns the input untranslated. `Stage::roles` now
carries the binding name.

**Output ports.** The phrasebook publishes more than one result, which is why
`pb:final` appears 597 times and `pb:out` six. Its **default** port is the text
carried onwards — the NMT chain hangs off it (`pb_feature <- ["pb"]`), so a miss
must not starve it — while `:final` is the hit that merges read. Getting that
backwards makes a merge prefer the untranslated source over the NMT's answer.
And a port that exists but holds `None` must *not* fall back to the default:
that fallback is exactly how the source sneaks past a merge again.

Both were caught by unit tests on synthetic graphs before they reached real
output, which is the argument for the executor having tests of its own rather
than only being exercised end to end.

Scored at scale rather than on hand-picked sentences —
`tests/graph_agreement.rs` runs the whole graph over the reference dump with the
phrasebook stubbed out, so it measures the graph and not the lexicon:

```
graph over 30 sentences: 14 identical to the OS, chrF 0.671
```

Nothing fell through every stage, and the figure matches what the NMT scores on
its own in the bench, so the refactor cost nothing.

**Edges carry tokens.** the shipped graph passes *token arrays* between the
SentencePiece, alignment and merge stages — which is why `spm_encode` and
`spm_decode` are separate stages at all. An edge here is a
[`execute::Val`]: `Text` or `Tokens`. `SentencePieceBlock` reads its direction
from the block's `action` attribute (the same block type both encodes and
decodes, so `Stage::spm` carries it) and does real work; blocks that want a
surface form render tokens on demand.

Round-tripping `"La femme de mes rêves"` through `encode -> decode` returns it
through 5 pieces, and re-scoring the graph at scale gives **14 of 30 identical
to the OS, chrF 0.671 — byte-identical to before the change**, which is what a
correct round trip should cost.

Still pass-through: `AlignmentProcessor`, `PDecForceAlign` and
`LinkAlternatives`. They project spans between source and target, and the token
edges are the precondition for them rather than the whole of it — the alignment
itself still has to be computed.

### How `Q_meta` is read

`quantized_gather` stores **four f32 per column** (2048 = 4 x 512; the layout is
`[col][4]`, confirmed by 512/512 groups being strictly increasing). Those four
are the **0/25/75/100th percentiles**, and the byte is a *uniform quantile
index*, so the knots sit at bytes **0, 64, 192, 255**.

That was fitted, not guessed. Byte codes are uniform over 0..255, so the knot
placement alone sets the value distribution, and only this placement makes a
column Gaussian (**kurtosis 3.02**; the equal-thirds reading gives 1.96, far too
flat, and inflates the embedding norm 1.6x). Note that *nearest-neighbour
quality cannot choose between curves* — every monotone curve gives identical
neighbours — which is why an earlier reading survived so long.

### Suppression follows the config

Decoding used to drop every angle-bracketed piece from the candidate set. The
config disagrees: `shortlist-suppress-tokens` is **empty** for en->fr, so the OS
suppresses nothing, and the blanket rule was also removing the `<STRUCT_*>`
markers the model uses to offer gender alternatives.

It now suppresses what the config names, plus the direction tags — emitting
`<src-en_US>` or `<en_US-fr_FR-optimal>` mid-sentence is garbage — and leaves
`<s>` scoreable because it terminates the sequence. Measured on the bench the
change is **exactly neutral** (112 identical, chrF 0.808, byte-identical to
before), so it is a correctness fix rather than a trade.

### Do-not-translate, learned by probing

The block takes the source and the final target and the config gives it no
parameters, so the rule set is not written down anywhere. Rather than guess it,
`scratchpad/TRDnt` asks the live framework. Across en->fr, en->de and en->ja it
preserves these verbatim:

```
https://www.example.com   john.smith@example.com   @SomeHandle   #WWDC2026
README.md               +1 415 555 0123          AA123           ABC-123-XYZ
```

The two that did *not* survive are the informative ones, and both are correct
behaviour: `cargo build --release` is ordinary words and gets translated, and
`$99.99` becomes `99,99 $`, which is French currency formatting rather than a
lost span. So the rule is about **identifiers**, not about anything that merely
looks unusual — and a module that protected `$99.99` would make output wrong.

`dnt::spans` finds them and `dnt::lost` reports which failed to survive.
`translate` prints a warning when one does. Product names (`iPad Pro`,
`iPhone 15 Pro Max`) survive too but are not pattern-matchable — the model just
keeps them — so they are out of scope.

**Detection, not repair.** Putting a span *back* means knowing where it belongs
in the target, which is exactly `AlignmentProcessorBlock`, and that is not
implemented. Guessing an insertion point would be worse than reporting.

On the two probes run end to end, our own output matches the OS verbatim:

```
"the file is README.md in the repo"  -> Le fichier est README.md dans le dépôt
"call me on +1 415 555 0123 ..."     -> Appelez-moi au +1 415 555 0123 ...
```

### The quality estimator, and what it reveals

Implemented (`quality`), because it is small, entirely data-driven, and aimed at
precisely the degeneracy this port kept hitting. Two shipped checks:

```
qualityEstimator/repeat/repeatRegex.txt
    source  ([^ ].*?)( \1){1}      <- a run repeated once
    target  ([^ ].*?)( \1){3}      <- a run repeated three times
qualityEstimator/ovs/ovs.<lang>.{src,tgt}   <- offensive / vulgar term lists
```

Run against the shipped framework's *own* output, both fire:

```
"Avocado platypus i love the summer" -> "J'adore l'été merde"
    [UnlicensedProfanity("merde")]
"...rêves rêves rêves rêves rêves"
    [RepeatedTarget("rêves")]
"i love the summer" -> "J'adore l'été"
    []
```

Two things fall out of this. The repeat regex is what the hand-rolled n-gram
constraints above were badly reinventing — the OS's version works on the decoded
surface, on whitespace runs, at a threshold of four occurrences. And the `ovs`
lists explain the profanity in the OS's degenerate outputs: `merde`, `salope`
and `alboche` are all on the French list, so those were *hallucinations the
quality estimator exists to catch*, not translations of anything in the input.
The vulgar-term check only fires when the source licenses nothing — translating
profanity the user actually wrote is correct.

This block scores; it does not rewrite. The shipped graph feeds it into `merger_final`,
which is one of the six merges still to do.

#### What actually fixes it: finish the pipeline

The chase ended somewhere better than a decoding constraint. **The phrasebook
already contains `platypus|||ornithorynque`.** the OS's architecture puts exactly
this kind of rare lexical item in the lexicon so the model never has to spell
it — and `rlx-translate translate` was running the phrasebook stage *only*,
stopping with "this input needs the NMT" on a miss.

It now falls through to the NMT, with the same sentence-casing and punctuation
post-processing applied to both paths:

```
"platypus"                                    -> Ornithorynque          (phrasebook)
"i love the summer"                           -> J'adore l'été          (phrasebook)
"where can i get a hamburger"                 -> Où est-ce que je peux trouver un hamburger ?
"my neighbour bought seventeen blue umbrellas" -> Mon voisin a acheté dix-sept parapluies bleus   (NMT)
"the platypus is swimming in the river today"  -> L'ornitharyque nage dans la rivière aujourd'hui (NMT)
```

So the word is now correct wherever the lexicon covers it, which was the whole
point of the lexicon. In free text the model still misspells it
(`ornitharyque`) — the phrasebook is whole-string exact match, by the OS's
design, not a span substituter. Worth noting that **the OS does no better**: on
the same word in a sentence the OS simply drops it
(`Avocado platypus i love the summer` -> `J'adore l'été merde`).

#### What the decoding levers could and could not do

Two of the three defects are not ours. The remaining one — our
`ornithornithaque`, a mangled `ornithorynque` — was worth chasing, and the chase
is more interesting than the result.

The model **knows the word**: asked to translate `platypus` alone it offers
`ornithorynque` at rank 2. The failure is contextual. And **no piece in the
vocabulary contains `ornith`** — the model spells that rare word out of generic
fragments, so the stutter is *character*-level, not token-level.

Two candidate fixes, both measured:

*No-repeat n-gram* (`Nmt::set_no_repeat_ngram`, `SearchOptions::no_repeat_ngram`,
default 0). Implemented and working — at `n = 1` the output changes completely —
but at 2 or 3 it does not fire here, precisely because the repetition is below
the token level. Kept as an opt-in, since it is the standard tool and the OS's
own looping (`Anhänger` five times) is the kind of thing it catches. It is not a
fix for this case and is not presented as one.

*No-repeat character run* (`Nmt::set_no_repeat_char_ngram`, default 0) sees what
the token-level check cannot. At 6 it does fire, and unlike the length-penalty
lever it costs **nothing** measurable — agreement with the OS stays at 112/200
and chrF 0.808, identical to baseline, at both 6 and 8.

But it does not fix the word, and that distinction matters. French for platypus
is `ornithorynque`; the constraint merely replaces a *malformed* word with an
*untranslated* one:

```
off        1. Avocat ornithornithaque j'adore l'été     <- mangled
chars=6    1. Avocat platypus j'adore l'été             <- English left in place
           3. Avocat ornithopus j'aime l'été            <- still mangled
           5. Avocat ornithique j'aime l'été
```

`ornithorynque` appears nowhere in the beam for that sentence, though it is rank
2 for `platypus` on its own. And rank 1 there is `Platypus` — **the model's own
preference for this word is to copy it**, not to translate it. So this is not a
search failure that a constraint can repair; the constraint only changes which
wrong answer surfaces.

*Dropping `norm-costs`* looked compelling: the clean candidate already wins on
raw score (-4.333 against -5.142) and only loses after length normalization.
Turning it off does produce `Avocat platypus j'adore l'été`, and on ordinary
sentences changes nothing at all. But across the bench it is **worse**:

| | `norm-costs` on (config) | off |
| --- | --- | --- |
| agreement with the OS, identical | **112 (56.0%)** | 107 (53.5%) |
| agreement chrF | **0.808** | 0.792 |
| NMT chrF on gold | **0.520** | 0.513 |

It repairs one adversarial sentence and costs five matches elsewhere, so the
config's setting stays. `RLX_TRANSLATE_NORM_COSTS=0` and
`RLX_TRANSLATE_NO_REPEAT=<n>` exist for A/B, both defaulting to the OS's
behaviour.

The honest conclusion is that this is a model limitation on out-of-distribution
input, not a decoding bug. Of the two levers, one costs more than it returns and
one is free but does not recover the right word. Both stay off by default.

#### It found a bug in `rlx-embed`

Every multilingual checkpoint in that registry — `multilingual-e5-*`,
`paraphrase-multilingual-*` — is **XLM-RoBERTa**, and the RoBERTa family starts
position ids at `pad_token_id + 1`, not 0. That is why their
`max_position_embeddings` is 514 rather than 512. `embed_with_rlx` built
positions as `0..seq`, so every token read the wrong row of the position table.
The output stayed plausible and merely got quietly worse — the kind of defect
that survives a long time. Fixed in `rlx-embed` (`RlxBertModel::position_offset`,
derived from `model_type` and `pad_token_id` in the checkpoint's own
`config.json`), with the numbers to justify it:

| | 0-based | `pad_token_id + 1` |
| --- | --- | --- |
| worst equivalent | 0.457 | **0.629** |
| best unrelated | 0.262 | **0.019** |
| margin | 0.195 | **0.610** |

A 3.1x wider margin, and every language improved — `ja` equivalent 0.472 ->
0.922, `zh` unrelated 0.262 -> -0.037. Unit tests pin the offset for bert,
xlm-roberta and malformed configs.

### The roadmap is in the config

`rlx-translate plan <pair>` resolves the shipped stage graph, and it says
exactly what is left:

```
en_US-fr_FR · task mt_app · 30 stages
10 [have]   data present, executor written
10 [ok  ]   trivial (Select / Null / Tokenizer)
10 [todo]   not implemented
```

The ten `todo` stages are **six `MergerBlock`s**, plus `AlignmentProcessorBlock`,
`LinkAlternativesBlock`, `DoNotTranslateBlock` and the alignment wiring. That is
not an accident of naming — those blocks *are* the mechanism for the cases this
port still gets wrong:

* `MergerBlock` is a **coalesce**, not a splicer. Every one of the 752 mergers
  across 60 configs has `merge-style: "any"` and reads named output ports
  (`pb:final`, `select_target:match`, `pb:out`), so it means "take whichever
  branch produced a value". The phrasebook-then-NMT fallback this crate already
  does *is* that semantics. An earlier draft of this section claimed the mergers
  splice a lexicon entry into a sentence; they do not, and there is no
  span-substitution style in the format — which also explains why the OS itself
  cannot place `ornithorynque` mid-sentence.
* `do_not_translate` is identifier preservation — implemented below.
* `force_align` + `alignment` give word alignment, which is what a *repair*
  would need.

### Why the gender block is not implemented

`StructuredPredictionBlock` has its data installed (`gender_defaults_list`, 1033
occupation nouns — the words where French must choose *le comptable* or *la
comptable*), so it looks like the obvious next block. It is not implementable
against evidence:

* The model does not emit `<STRUCT_*>` markers unprompted. Asked for
  `the accountant is here` it simply picks a gender —
  `le comptable est ici`, `L'infirmière est arrivée`, `mon médecin m'a appelé`.
  The markers evidently need the source annotated first, which is
  `AmbiguityAnnotator`'s job.
* **Zero `<STRUCT_*>` markers appear in any of the 416 reference files.** Those
  alternatives are internal — consumed by `StructuredPrediction` and
  `LinkAlternatives` — and never reach the public `TranslationSession` output.

So there is nothing to validate an implementation against through the harness
that built the rest of this port. Guessing at it would produce code no
measurement could check, which is how the `72/184` knot search went wrong
earlier.

### Gender alternatives are reachable, and correctly not taken

The same probe settles something recorded here as unverifiable. Decoding
greedily over the whole vocabulary, the model emits

> `<STRUCT_MALE_BEG>` le petit chat gris `<STRUCT_MID>` la petite chatte grise
> `<STRUCT_END>`

— both variants inline, exactly what `StructuredPredictionBlock` is for. So the
markers are real and this pair does not suppress them. But under the shortlist
and beam search neither we nor the OS emits them: `the doctor is tired` gives
`Le médecin est fatigué` from both, and no `<STRUCT_*>` appears in any of the
4175 benchmark sentences or 416 reference files. Reachable, not taken, and
matching the OS — so there is nothing to implement here, which is a different
answer from "cannot tell".

### The alignment is there, and it is shifted

Those four blocks all want a word alignment, and the manifest says where it is:
`AlignmentLayerStr: decoder.1.encoder_attn.attn_probs`, `AlignmentHeads: 1`,
`ShiftedAlignments: true`. `examples/alignment_probe.rs` reads it out — `[8, 1,
9]`, heads by one query by source length — and it aligns, one step late, which
is what "shifted" means:

```
  step  3  emit "le"        peaks on source[3] "▁the"
  step  4  emit "▁petit"    peaks on source[3] "▁the"
  step  5  emit "▁chat"     peaks on source[4] "▁small"
  step  6  emit "▁gris"     peaks on source[6] "▁cat"
```

Read one step late — the peak at step *t+1* belongs to the token emitted at *t*
— `le`->`the`, `petit`->`small`, `chat`->`cat`, `gris`->`grey`. Every one right.

### What is actually implemented, and the status display that lied

`rlx-translate plan` reported seven `MergerBlock` stages as `todo` while
`execute` had been running them all along, and reported `PDecForceAlignBlock` as
`have` — which read as implemented — when nothing runs it. The classification
lived in `pipeline` and had gone stale as the executor grew, and its own doc
comment still said "no block executor is implemented yet".

`execute::handles` owns it now, beside the match it describes, and
`with_asset_status` no longer promotes a stage to "files present" unless
something runs it. A missing file is still reported either way, because that is
an incomplete *install* rather than a missing feature. Four blocks are genuinely
unimplemented and now say so: `PDecForceAlign`, `StructuredPrediction`,
`AlignmentProcessor`, `LinkAlternatives`. All four are annotators — they add
metadata rather than change text — so passing their input through is the right
default.

## What it scores

### Measured against the OS's live output

`rlx-translate parity <ref-dir>` scores the native pipeline against a dump of the
real framework (75 542 translations over 416 ordered directions, produced by the
Swift harness described below):

```
366 pairs · 73 360 phrasebook-covered · 182 no entry (need the NMT)
exact 73 355 (99.993%) · case-only 0 · space 0 · substantive 5
361 of 366 directions at exactly 100.0%
```

Throughput of the reference framework itself, for comparison: median 48 tr/s
batched (min 5, max 668, p90 214).

### The NMT works

It translates, in every language the machine has installed. Greedy decode over
the shortlist, scored as **chrF against the OS's own output** on 120 directions
(3 reference sentences each):

```
120 direct directions   mean chrF 0.713   median 0.720
  >= 0.95   18 (15%)     <- byte-identical to the OS
  >= 0.90   34 (28%)
  >= 0.70   67 (56%)
  >= 0.50  103 (86%)
```

Byte-identical directions span every script that ships: `en->ar`, `en->ru`,
`en->tr`, `en->id`, `en->nl`, `en->pt`, `en->es`, `de->en`, `es->pt`, `fr->pt`,
`it->pt`, `vi->en`.

```
en_GB -> ar_AE   "i love the summer"  -> "أحب الصيف"                     1.000
en_US -> ru_RU   "i love the summer"  -> "Я люблю лето"                  1.000
en_GB -> es_ES   "i love the summer"  -> "Me encanta el verano"          1.000
de_DE -> en_US   "Mein Hals schmerzt" -> "My throat hurts"               1.000
en_GB -> fr_FR   "i love the summer"  -> "J'adore l'été"                 0.938
en_GB -> ja_JP   "i love the summer"  -> "夏が大好きです"                 0.676
en_GB -> ko_KR   "i love the summer"  -> "저는 여름을 사랑합니다"          0.589
```

The Japanese and Korean outputs differ from the OS's only by a trailing full
stop; chrF is unforgiving at these lengths.

Teacher-forced against the OS's next piece on en->fr, mean log10(rank+1) is
**0.014** with **20 of 21** positions exactly top-1. Beam search (beam 3,
`norm-costs`) is wired through `Nmt::beam_decoder`.

**What that number is not.** It was measured by an earlier `all_pairs` that
drove the NMT *alone*, skipping the surrounding pipeline blocks and the 288
pivot directions. Sentence casing (`"meninggalkan"` vs `"Meninggalkan"`),
do-not-translate spans (`en->hi` transliterating "iPad Pro" into Devanagari) and
Simplified-to-Traditional for `zh_TW` all sat outside it. And it scored against
the phrasebook-derived dump, so it compared our NMT to the OS's *lexicon*
answers.

`tests/all_pairs.rs` now runs **every** installed direction through the stage
graph, and separates two things that were previously conflated:

* **coverage** — did the direction produce a translation at all. Needs no
  reference, and is what the pivot work changed.
* **chrF against the OS** — only where a reference exists, defaulting to the
  non-lexicon dump, and always reported alongside *which producer answered*
  (phrasebook or NMT).

Two things had to be fixed for a sweep of this size to be runnable at all. The
model cache was per-direction, so every direction reloaded its decoders; it is
now keyed on the resolved bundle, since every direction into French wants the
same one. And it was unbounded — each model materialises a tied readout table of
up to 168 000 x 512 f32, ~344 MB, which across 400 directions would exhaust a
shared machine. It is capped at six, enough for a pivot's two hops plus
neighbours.

**Pivot directions now work too.** the OS ships one multilingual model per
language *group* and routes everything else through English. Those pairs have no
translator block *for the pair* — but the graph carries several for the hops:
`ar_AE-de_DE` has **eight `PDecTranslatorBlock`s across two model files**
(`MT-bi-en-ar` then the seven-language bundle). The graph already described the
pivot; the executor was using one set of decode parameters for every translator
stage, so the second hop translated the wrong direction.

Each stage now carries its own `PDecParams` and the model is resolved from the
block's own `model-file` and `target-locale` — the first hop targets English
regardless of where the pair ends up:

```
ar_AE-de_DE  "أحب الصيف"      -> Ich liebe den Sommer
ar_AE-fr_FR  "أحب الصيف"      -> J'adore l'été
ja_JP-fr_FR  "夏が大好きです"   -> J'adore l'été
ru_RU-it_IT  "Я люблю лето"   -> Amo l'estate
```

One wrinkle worth knowing: the config names a variant that is often not the
installed one (blocks say `MT-bi-en-ar-0`, the machine has `-20`), so
[`Assets::model_home_for`] tries an exact asset match first and then ignores the
trailing variant number. Exact first matters — variants are genuinely different
models.

Trying to *score* the pivot turned up something about the measurement rather
than the model, and it is worth stating because it qualifies other numbers here.

`tests/pivot_agreement.rs` first ran with the phrasebook stubbed out, and
reported chrF 0.748 with visible entity duplication
(`10.5" iPad Pro` -> `10,5" iPad Pro 10,5"`). Putting the phrasebook back gave
**18/18 identical, chrF 1.000** — which is not a triumph, it is the same mistake
inverted: every sampled sentence was a lexicon entity, so the NMT never ran.
Splitting the score by producer says so outright:

```
6 pivot directions, 18 sentences: 18 identical, chrF 1.000
of those, 18 were phrasebook hits and 0 went through the pivot NMT
```

Filtering to sentences the phrasebook *misses* then leaves **nothing at all**,
and that is the real finding: **the reference dump was generated by sampling the
phrasebook `.dict` files**, so every sentence in it is a lexicon entry. It
cannot exercise the NMT for any direction.

That also explains `graph_agreement`'s 14/30: with the phrasebook stubbed, it
was scoring our *NMT* against the OS's *phrasebook* answers. Not a wrong
measurement, but not the one it looked like.

So a reference dump of **non-lexicon** sentences was generated
(`scratchpad/TRCorpus`): twenty English sentences verified against the shipped
phrasebook (all twenty miss), translated `en->X` by the OS to give natural
source text in each language, then run through eleven directions. With that:

```
6 pivot directions, 21 sentences: 3 identical, chrF 0.693
of those, 0 were phrasebook hits and 21 went through the pivot NMT
```

That is the first honest measurement of the pivot NMT, and it immediately
showed the repetition defect on real text:

```
ar->fr  ours   "J'ai publié le message avant de fermer le bureau avant de fermer le bureau"
        os     "J'ai posté le message avant de fermer le bureau"
```

#### Which reversed an earlier decision

`no_repeat_ngram` was rejected earlier as "a trade": it changed nothing on the
bench and diverged from a config that names no such constraint. That conclusion
came from the phrasebook-derived dump, **where the decoder never repeats and the
lever has nothing to do**. Re-measured where repetition happens:

| | pivot NMT (non-lexicon) | lexicon bench |
| --- | --- | --- |
| off | chrF 0.693, 3 identical | 112 identical, chrF 0.808 |
| **n = 3** | **chrF 0.720, 4 identical** | 112 identical, chrF 0.808 |

It moves output *closer* to the OS at no measured cost, so it is now the default.
n = 3 and n = 4 score identically. This is worth remembering as a pattern: a
lever that measures as neutral may simply have been tested on data that cannot
exercise it.

> **Run the slow tests with `--release`.** `cargo test` defaults to a debug
> build and NMT decode is ~13x slower there: this suite took 190 s in release
> against 40+ minutes in debug.

The architecture, read off the graphs rather than guessed:

* **20 encoder blocks** — 4 in `input_<lang>`, 12 in the shared `encoder`, and
  **4 more inside `handover_<lang>`** (which is not merely the cross-attention
  K/V projection) — and **3 decoder blocks**.
* **DeepNet / DeepNorm post-LN**: the residual branch is scaled *before* the add.
  Encoder `alpha = 1.8346896`, decoder `alpha = sqrt(3) = 1.7320508`. Both match
  DeepNet exactly for N=20, M=3: `0.81*(N^4*M)^(1/16)` and `(3M)^(1/4)`.
* Embedding = `gather(token) * sqrt(512) + gather(position)`.
* A simplified **average attention network** decoder, with no gate:
  `accum.next = x + accum`, `y_avg = accum.next * (1/position)`, a 512-512-512
  FFN, then `+ sqrt(3)*x`, then LayerNorm.
* `readout.espresso.net` is *only* a gather of the tied table; the logits are a
  host-side dot product, then log-softmax (`ApplyLog T`).

Three things are easy to get wrong because they are not where you would look:

1. **The FFN nonlinearity hides in `dynamic_dequantize`'s `has_relu: 1`**, not in
   a layer of its own.
2. **The `1/sqrt(head_dim)` attention scale is folded into the query's
   `w_quantization_scale`** — it is exactly `127 * 8 = 1016` on layers whose
   weights needed no extra range. There is no scale layer, and adding one is
   *wrong*: measured, it drives encoder layers 1-3 to nearly uniform attention.
3. **Positions are 1-based.** The manifest says `PositionZeroBased F`, and the
   decoder graph proves it independently — `elementwise operation: 10` is a
   *reciprocal* of `position`, so 0 would divide by zero.

### How good is it, not just how faithful?

Every other number here scores our output against *the OS's*, which says whether
the port is right and nothing about whether the model is any good. FLORES is
line-aligned across all 200 languages, so the same sentences carry a human
reference in every direction. `examples/bench_human.rs` reports three chrF
figures; over an early 45 sentences:

Over 1425 sentences in all 57 directions:

| | chrF vs human |
| --- | --- |
| ours | 0.595 |
| the OS | 0.597 |
| ours vs the OS | 0.913 |

**We are 0.002 chrF behind the OS, and ahead of it in 22 of 57 directions.** No
direction trails by more than 0.018. That settles what the remaining
disagreements are worth: they are lexical coin-flips (`Flur` against `Halle`),
not quality, and chasing them would be fitting the OS's noise.

FLORES is news prose, which is not the domain a translation app serves, so the
same measurement runs on Tatoeba — short everyday sentences with human
references, 55 of the 57 directions:

| corpus | sentences | ours | the OS | agreement | delta |
| --- | --- | --- | --- | --- | --- |
| FLORES (news prose) | 1425 | 0.595 | 0.597 | 0.913 | -0.002 |
| Tatoeba (conversational) | 2750 | 0.645 | 0.651 | 0.920 | -0.006 |

The two agree, which is the point of running both: the port is not tuned to one
kind of text. On Tatoeba we are ahead of the OS in 25 of 55 directions and the
worst deficit is -0.031. The lowest *agreement* figures are all CJK targets
(`it_IT-ja_JP` 0.780, `en_US-zh_CN` 0.796), where both we and the OS score ~0.35
against the human reference — short Japanese and Chinese have few character
n-grams to overlap, so chrF is unstable there rather than the translation being
bad.

It also says what the FLORES corpus was for. Every one of these numbers is on
Wikipedia prose the port was never built against, three to five times longer
than the hand-built corpus, and it found two defects — a silent source
truncation and a byte-fallback round-trip — that the old corpus could not
express.

Two notes on using these datasets. FLORES+ on Hugging Face is gated, but
`Muennighoff/flores200` is only a loader pointing at Meta's CDN, so the tarball
comes down without an account. And use **devtest**: the OS's own corpus tags name
`floresp_dev.split_dev` and `.split_train`, so the dev split is in their training
data. The tags are worth reading in general — the models name the public corpora
they were built from, `flores101`, `newstest2016`, OPUS Bianet, Tatoeba, beside
vendor-internal ones like `keyboard_misspelling` and `dev_web_spring`.

FLORES is news prose, which is not what a translation app sees, so
`scratchpad/tatoeba` holds a matching conversational pool — 100 short sentences
with human references for 55 of the 57 directions, from
`Helsinki-NLP/tatoeba_mt`.

### The metric was not chrF

Every figure above is a chrF number, and for most of this port's life
`score::chrf` was not computing chrF. It used character n-grams of order 1..=4,
combined them with `F1`, and lowercased both sides. chrF is orders 1..=6 with
`beta = 2` — recall weighted four times precision — and is case-sensitive. The
numbers were internally consistent, so every comparison drawn from them holds,
but they were comparable to nothing anyone else publishes.

It now averages precision and recall over the orders both strings populate and
combines them with `beta = 2`, matching sacreBLEU's effective-order smoothing,
and a test pins six values against **sacreBLEU 2.6.0** to 5e-4. Re-baselined:

| | old metric | chrF |
| --- | --- | --- |
| agreement with the OS, 855 sentences | 0.955 | **0.943** |
| FLORES, ours vs human | 0.677 | **0.595** |
| FLORES, OS vs human | 0.679 | **0.597** |
| Tatoeba, ours vs human | 0.705 | **0.645** |
| Tatoeba, OS vs human | 0.709 | **0.651** |

Every conclusion survives — the gap to the OS is -0.002 on FLORES and -0.006
conversationally, against -0.002 and -0.004 before — because both arms of every
comparison always used the same function. **Tables elsewhere in this file that
were measured before this fix are marked; they are comparable within themselves
and not with the headline.** The byte-count exact-match figures are unaffected.

One old test asserted `chrf("un livre", "Un livre") == 1.0`, which is the
lowercasing bug written down as a requirement. It now asserts the opposite.

### The bench

`rlx-translate bench` scores both stages against the OS's own `featureTestDicts`
gold pairs, and falls back to the NMT for whatever the phrasebook does not
cover — so it reports end-to-end coverage rather than one stage's:

```
188 pairs · gold 200 · covered by phrasebook 98 · exact 98
accuracy on covered gold pairs: 100.0%
NMT on the 102 gold pairs the phrasebook misses: chrF 0.520, 12 exact (11.8%)
end to end: 200 of 200 gold pairs translated, 110 exact (55.0%)
agreement with the OS's live output on 200 of those sources: 112 identical (56.0%), chrF 0.808
```

Phrasebook lookup runs at 4–11 M/s.

**`featureTestDicts` is a specification, not a transcript.** Some of its entries
list *several* acceptable answers — `the plant is open` gives both
`La fábrica está abierta` and `La planta está abierta`, and both `工厂开着` and
`植物开着` — so reading them as independent rows scores one of the two wrong
whichever is produced. Accepting any listed reference is what took the
phrasebook stage from 94.2% to **100%**; nothing about the translation changed.

More importantly, **the OS's own shipped framework scores 51.1% exact against
that gold** (measured over the 139 entries checkable here). `Er ist blau` is
wanted as "He's drunk"; the OS says "He is blue". So the gold columns have a
ceiling well under 100%, and for a *port* the question worth asking is not "do
we match the spec" but "do we reproduce what the OS produces" — which is the
`agreement with the OS` line, and what `RLX_TRANSLATE_REFERENCE` enables.

The NMT column covers what the phrasebook misses: dictionary items like
`जोड़ा-जामा` -> "Suit of clothes", so exact match is harsh there and chrF is the
comparable number. It is blank for pairs the OS routes through English.

### Every direction, not just the 57 with a reference

The agreement corpus covers 57 directions; the machine supports around 400. Both
single-direction defects this port has found — `en_US-zh_TW`'s shortlist and
`en_US-tr_TR`'s input graph — were found because those directions happened to be
in the corpus. FLORES is aligned across every locale, so a human reference
exists for every ordered pair without any harness run, which makes a sweep of
the lot possible: `examples/sweep_all.rs`, **378 directions**, mean chrF 0.626.

**Every one produced output**, and nothing is broken. Reading the result needs
care, because chrF has a different floor in every language — the median across
19 source languages runs from 0.444 into Korean to 0.751 into English — so the
raw ranking is a list of scripts, not of defects. Normalised against each
target's own median, the furthest-below entries are overwhelmingly *Korean as a
source*: 7 of the worst 12, at -0.06 to -0.10.

That is the language, not the port. Where the OS's output exists for those
directions we are level with it:

| | ours | the OS |
| --- | --- | --- |
| `ko_KR-en_US` | 0.649 | 0.647 |
| `ko_KR-es_ES` | 0.590 | 0.594 |
| `en_US-ko_KR` | 0.518 | 0.513 |

So the sweep's finding is a negative one, which is what a smoke test over 378
directions is for: there is no third single-direction defect waiting.

### More than one answer is often right

```sh
rlx-translate nbest en_US-fr_FR "where can i get a hamburger" 3
```
```
1. Où puis-je obtenir un hamburger      score -1.365  normalized -0.195
2. Où puis-je avoir un hamburger        score -1.554  normalized -0.222
3. Où puis-je obtenir un hamburger ?    score -2.508  normalized -0.314
```

Beam search keeps the runners-up alive anyway, so they are free, and they carry
real information: for `i love the summer` both `J'aime l'été` and `J'adore
l'été` are correct, and the second is what the OS returns. Variants are
deduplicated case-insensitively — several token paths decode to the same string,
and a list repeating one answer three times says nothing. A confident input
yields fewer than `n`, which is honest rather than padded.

Beam is not cosmetic: for `i love the summer` into Japanese, greedy gives
`夏が大好きです` and beam gives `夏が好きです`, which is the reference answer. The
bench decodes with beam for that reason.

### Cosine, used carefully

chrF is a surface metric: it scores `我爱夏天` against `我喜欢夏季` at 0.11 even
though both are correct. A semantic second opinion comes from the model itself —
[`Nmt::sentence_embedding`] mean-pools the encoder output, and
[`score::cosine_centered`] compares two of them.

**Centre them first.** Post-norm encoder states are strongly anisotropic: one
common direction dominates every pair, so raw cosine is squashed into a narrow
band and *unrelated sentences read as 0.94*. Subtracting the mean of a handful
of sentences in the same language removes it:

| pair | raw | centred |
| --- | --- | --- |
| `J'aime l'été` / `J'adore l'été` | 0.9950 | 0.8840 |
| `La femme de mes rêves` / `La femme dont je rêve` | 0.9828 | 0.5957 |
| `J'adore l'été` / `Le train arrive à huit heures` | 0.9416 | **-0.2635** |
| `La femme de mes rêves` / `Il pleut des cordes` | 0.9402 | **-0.1657** |

The ordering survives either way, but the margin goes from 0.04 to 0.76 — 19x —
and only the centred number can be *reported* without misleading. A test pins
both the separation and that centring widens it.

This is deliberately a second opinion, not a verdict. Mean-pooled states ignore
word order, so a sentence and its reversal score identically. It tells you
whether an answer is about the same thing, not whether it is right — and it is
honest about that: on the `featureTestDicts` misses it reads **0.488**, close to
chrF's 0.520, confirming those are genuine meaning failures (`Ich habe einen
Kater` rendered "I have a cat" rather than "a hangover") and not paraphrase the
surface metric was punishing unfairly.

Earlier in this port, cosine between *pipeline stages* produced a phantom
"representation collapse" and misled the NMT diagnosis three times. That failure
mode is specific to comparing intermediate activations against each other;
comparing two sentences in one space is a different use, and it earns its place
by measurement above rather than by assumption.

### BERTScore

chrF compares characters; pooled cosine compares one vector per sentence and
throws away word order. The BERT-family answer is
[`score::bertscore`] — greedily match every candidate token to its closest
reference token and back, then take the harmonic mean. It takes embedding
vectors rather than text, so the library keeps no embedding-model dependency;
the validation test supplies them from `rlx-embed`, using per-token contextual
states and dropping `[CLS]`/`[SEP]`, which match perfectly in every pair and
would inflate every score.

```
identical   1.0000
paraphrase  0.9399   "La femme dont je rêve"
replacement 0.8972   "La femme de mes cauchemars"   (dreams -> nightmares)
unrelated   0.7333
```

It separates a one-word *semantic* replacement from a paraphrase, which neither
chrF nor pooled cosine does reliably. Its absolute range is compressed — 0.73
for unrelated text — which is a known property of the metric, so read the
ordering rather than the number.

### Validated against an outside model

Judging our own translations with the translation model's own encoder is
circular — a systematic bias in that encoder would flatter every answer equally
and the metric would never notice. `tests/embed_validation.rs` cross-checks it
against `rlx-embed` (a **dev-dependency**; nothing in the library or the binary
depends on it), which had no part in producing the text:

```
equivalent  "My throat hurts" / "My neck hurts"              ->  0.5866
equivalent  "The woman of my dreams" / "The woman I dream of" ->  0.8398
unrelated   "My throat hurts" / "The train arrives at eight"  -> -0.1497
unrelated   "We meet at the bank" / "It is raining heavily"   -> -0.2436
```

The separation matches what the in-house metric gives after centring, and on
three ranking cases the two embedders agree 3/3 on which candidate is closer —
which is the property that matters, since a metric is used to order candidates
rather than to report an absolute number. Where they differ is in confidence:
for `We meet at the bank` the outside model scores the paraphrase at 0.362 and
the in-house encoder at -0.132. Same ordering, weaker signal, so the in-house
number should not be read too finely.

The default embedder is `intfloat/multilingual-e5-base` at
`/Volumes/FOUR/weights/embed/multilingual-e5-base`, falling back to
`all-MiniLM-L6-v2` (English-only) if it is absent; `RLX_EMBED_MODEL` overrides
both. With the multilingual model the check runs in every language the
translator covers:

```
fr  equivalent +0.9357  unrelated -0.1766
de  equivalent +0.8417  unrelated +0.0189
es  equivalent +0.7579  unrelated -0.2219
ru  equivalent +0.7042  unrelated -0.2469
ja  equivalent +0.9221  unrelated -0.1318
zh  equivalent +0.6287  unrelated -0.0373
```

#### Do the translations actually mean the same thing?

The similarity numbers above are on hand-picked pairs, which proves the metric
can separate *something*. The question that matters is whether it separates our
real output — and whether that output is equivalent to the OS's. Both need the
same control: if every sentence in a language scored high against every other,
"equivalent" would be vacuous.

So each of our translations is compared against the OS's translation of the
**same** source (matched) and of the **other** sources (mismatched):

```
en_US-fr_FR: matched 0.646  mismatched -0.136    ours "J'aime l'été"
                                                os    "J'adore l'été"
en_US-de_DE: matched 0.620  mismatched -0.131    ours "Sie sind eine freundliche Person"
                                                os    "Du bist eine freundliche Person"
en_US-ja_JP: matched 0.805  mismatched -0.134    ours "夏が好きです"
                                                os    "夏が大好きです。"

matched 0.691 vs mismatched -0.134 (gap 0.824)
retrieval@1: 21/24
```

**Cosine works.** A gap of 0.824, and in 21 of 24 cases the OS's own translation
is the nearest of all eight candidates. That is a discrimination result, not a
similarity score, so it cannot be inflated by a shared language component.

**The translations are equivalent, not identical**, and 0.691 rather than ~1.0
is the honest reading of that: `J'aime` against `J'adore`, formal `Sie` against
informal `Du`, "like" against "love".

**All three retrieval misses are cases where the OS's own output is degenerate.**
Every one is a synthetic `Avocado platypus …` probe, and in each the OS either
dropped content and appended profanity, or looped:

```
"Avocado platypus i love the summer"
  ours   "Avocat ornithornithaque j'aime l'été"
  os     "J'adore l'été merde"                      <- prefix dropped, profanity added
  near   "J'adore l'été"          (the clean sentence)

"Avocado platypus they are my followers"
  ours   "Avocado Platypus sie sind meine Anhänger"
  os     "Sie sind meine Anhänger Anhänger Anhänger Anhänger Anhänger"   <- looped
  near   "Sie sind meine Anhänger"
```

So the metric preferred the *clean* reference over the OS's broken one, which is
the correct call — and on these inputs our output is the more faithful of the
two. Japanese missed nothing. Two caveats keep this honest: these probes are
deliberately adversarial (nonsense prefix plus profanity, built to stress the
decoder), so they say nothing about ordinary use; and our own rendering is not
clean either — `ornithornithaque` is a malformed attempt at `ornithorynque`.

### What the sibling crates were worth

The workspace has two other translation crates. `rlx-translategemma` is the
prompted-LLM paradigm (Gemma 3 backbone) — the OS's `.highFidelity` equivalent,
not comparable machinery. `rlx-nllb` is a real encoder-decoder NMT with its own
beam search, and reading it produced two things.

**A tunable length penalty.** `rlx-nllb` scores `score / len^alpha`
(`GenerateConfig::length_penalty`) where this crate had only the config's boolean
`norm-costs`. Those are the two endpoints of the same knob — true is
`alpha = 1`, false is `alpha = 0` — and both had already measured badly: 1.0
promotes a longer malformed hypothesis, 0.0 loses five matches. So the middle
was worth searching. Adopting the convention and sweeping it:

| alpha | 0.0 | 0.5 | **1.0** | 1.3 | 1.6 |
| --- | --- | --- | --- | --- | --- |
| identical to the OS | 107 | 110 | **112** | 112 | 112 |
| chrF | 0.792 | 0.798 | **0.808** | 0.807 | 0.808 |

It plateaus at 1.0. **the OS's `norm-costs: true` is already the optimum** — the
sweep validates the config rather than beating it, which is a result worth
having explicitly rather than assuming either way. `Nmt::set_length_penalty`
keeps it tunable.

**Duplication worth noting:** `rlx-nllb/generate.rs` and this crate's `beam.rs`
are two independent beam searches, and `rlx-models-core` has no shared one.
Both now carry `no_repeat_ngram`; a common module would be the right refactor.

Both NLLB and TranslateGemma would also make useful *independent* translation
oracles — a second system to adjudicate the cases where we differ from the OS.
Their HuggingFace caches on this machine are empty stubs, though, so that needs
a download first.

### Measuring it

Free-running output, teacher-forced rank and cosine between pipeline stages all
**disagree**, and only one of them tracked correctness: **teacher-forced mean
log10(rank+1) of the OS's own next piece**, over the reference dump. Cosine in
particular is worthless here — an earlier diagnosis of "representation collapse
at `input_<lang>` (cosine 0.927)" was a phantom, since ~0.9 cosine after a
post-norm encoder is ordinary transformer anisotropy.

One trap is worth naming. Searching the `Q_meta` knots against end-to-end
quality, *while the source was terminated with the wrong piece*, picked 72/184
over 64/192 — consistently, across four held-out slices. It was silently
compensating for an unrelated bug. Once the terminator was right, 64/192 scored
a perfect 21/21 and 72/184 did not. A tuned parameter will absorb a defect
elsewhere and look like evidence.

Verified *not* to be the fault, each by measurement, so they need not be retried:
the reshape+transpose head split is elementwise exact (0 of 3584 mismatches);
`batch_matmul` matches a hand-computed `q.k^T` to 0.0; cross-attention is
properly peaked; int8 quantization is faithful (the `RLX_TRANSLATE_F32=1` bypass
agrees); activation quantization is genuinely per-tensor (`.espresso.shape`
declares `q_scale` rank 1); and the AAN `accum` really does accumulate.

**Open:** the remaining errors are short ambiguous inputs where the model copies
the source (`"a chair"` -> `"A chair"`). The config also asks for
`lm-mode: partial_bias` at `lm-weight: 0.25`, which is not implemented — no LM
asset ships, so this most likely refers to the OS's *partial* (streaming) mode,
the one selected by `<src-partial>`.

## Defects found, and how

### Fourteen of the twenty shortlists were dead

The 43-direction sweep put `en_US-zh_TW` last by a distance: chrF 0.348 and
**zero** exact matches, while `zh_TW-en_US` scored 0.833. Our output was
Simplified Chinese where the OS's was Traditional — but mixed, which is the
tell. `en_US-zh_TW` and `en_US-zh_CN` share one model, one vocabulary and one
`decoder_zh`; they differ only in a direction token and a shortlist table.

Both differences were broken, and both silently:

- `Nmt::load` built the table path from the two-letter language, `all-zh`.
  Every other shipped table is named that way, so it worked everywhere except
  the one direction where it mattered — `all-zh_TW` is the sole exception. Now
  `load_with_shortlist` takes the config's `shortlist-lang-pair`, and
  `translate_nbest` **errors** if the loaded table is not the one the direction
  asks for, because a model with the wrong table still translates, just into
  the wrong variant.
- The reader then failed on the table anyway. It solved the table length by
  probing three hard-coded candidates around 167 969, a number from the French
  bundle. The length is in the header all along — the fourth word is the offset
  count, and `n` is exactly the vocabulary size: 168 000 for the French-family
  bundle, 96 000 for en-zh-ja-ko, 48 000 for `all-en`. The probe **failed
  outright on every vocabulary that was not 168 000** — and on the ones that
  were, it was still wrong: the entries array begins at
  `16 + (n + 1) * 4`, so a length short by 31 put its base 124 bytes early and
  **shifted every candidate list by 31 entries**.

So all twenty tables were broken, in two different ways: fourteen never loaded,
and the six that did returned shifted candidates.

The shift was invisible because a shifted candidate list is still a list of
plausible target pieces. The absence was invisible because `Nmt::load` took the
table with
`.ok()`, and `candidates` falls back to the whole vocabulary when there is no
shortlist — so every CJK and into-English direction scored 48 000 or 96 000
candidates instead of ~700, and still produced plausible text. `tests/shortlists.rs` now parses
every installed table, and a table that is present but unreadable is an error
rather than a fallback.

Three sentences of `en_US-zh_TW`, before and after:

| | |
| --- | --- |
| the OS | 她在辦公室關門前把信寄出了 |
| before | 她在办公室关门前发布了信 |
| after | 她在公司關門前把信寄了 |

The unit tests did not catch any of this: their fixtures left the header count
at zero and padded to 167 969 entries so the *probe* would land, which asserted
the parser's mistake rather than the format. They build real headers now.

### What the shortlist fix was worth

Re-running the 43-direction sweep: **chrF 0.788 -> 0.818, 206 -> 232 exact
matches over 645 sentences** (the input-graph fix below takes it to 0.959 and
509), and the sweep itself went from 3193 s to 798 s
because the affected directions had been scoring 48 000 or 96 000 candidates a
step instead of ~700.

28 directions improved, 7 were unchanged, and 8 lost ground — all of them by
0.021 or less, which is what a shortlist that now genuinely *restricts* the
search should look like.

| | before | after |
| --- | --- | --- |
| `en_US-zh_TW` | 0.348 | 0.533 |
| `en_US-tr_TR` | 0.401 | 0.526 |
| `nl_NL-en_US` | 0.621 | 0.745 |
| `de_DE-en_US` | 0.696 | 0.799 |
| `es_ES-en_US` | 0.811 | 0.910 |
| `en_US-es_ES` | 0.892 | 0.969 |
| `en_US-fr_FR` | 0.952 | 0.966 |
| `tr_TR-en_US` | 0.659 | 0.638 |
| `en_US-ar_AE` | 0.911 | 0.898 |

### `input_<lang>` follows the source, not the target

A bundle ships three per-language graphs — `input_<lang>`, `handover_<lang>`,
`decoder_<lang>` — and this port keyed all three by the **target**. That is
wrong for one of them. `input_<lang>` is the first four blocks of the *encoder*;
it reads the source, and it follows the source language. `handover_<lang>`
builds the decoder's cross-attention and `decoder_<lang>` generates, so those do
follow the target — taking the handover from the source instead produces noise
in every direction, which is the control that says only `input_` moves.

The mistake survived because the directions it was checked against tolerate it.
`en_US-tr_TR` does not, and its failure was distinctive enough to chase: with
the shortlist off it reproduced the English source verbatim, and with it on,
Turkish words in English word order.

| direction | all-target | input=source |
| --- | --- | --- |
| `en_US-tr_TR` | `Biz yürüdük boyunca nehir boyunca başladı yağmur yağıyor` | `Yağmur yağmaya başlayana kadar nehir boyunca yürüdük` |
| `en_US-zh_TW` | `一直沿河走，直到下雨` | `我們沿著河邊走，直到開始下雨` |
| `tr_TR-en_US` | `The rain started until we walked the river` | `we walked along the river until it started raining` |

The right-hand tr and zh_TW cells are the OS's reference verbatim.

Over 120 sentences in 8 directions, every direction improved:

| direction | all-target | input=source |
| --- | --- | --- |
| `en_US-tr_TR` | 0.526 | **0.966** |
| `tr_TR-en_US` | 0.638 | **0.967** |
| `en_US-hi_IN` | 0.767 | **0.986** |
| `de_DE-en_US` | 0.799 | **0.971** |
| `en_US-zh_TW` | 0.533 | **0.877** |
| `en_US-es_ES` | 0.969 | **0.980** |
| `en_US-fr_FR` | 0.966 | **0.991** |
| `en_US-ja_JP` | 0.862 | **0.888** |
| **overall** | 0.757, 39 exact | **0.953, 92 exact** |

`Nmt::load_for_pair` is now the loader to call; `Parts` makes the three graphs
independently selectable and `examples/graph_langs.rs` is the experiment.
`RLX_TRANSLATE_INPUT_GRAPH=target` restores the old keying.

Over the full sweep — 645 sentences, 43 directions — **chrF 0.818 -> 0.959 and
232 -> 509 exact matches. Every direction improved and none regressed**, from
`en_US-zh_TW` at 0.533 -> 0.877 to `en_US-it_IT` at 0.941 -> 0.997. The worst
direction is now `en_US-zh_CN` at 0.848, where it was 0.348 before either fix.

Note what it took to find: the direction that exposed it was the *worst* one,
and the two preceding attempts to fix that direction — dropping a tag, moving a
tag — both looked convincing on a single sentence and were both wrong. The
structural question ("which language does this graph belong to?") was the one
worth asking.

### The source was truncated at 64 tokens, silently

FLORES-200 devtest is human-written Wikipedia prose, three to five times longer
than anything the hand-built corpus contained. The first long sentence put
through the port came back as this:

> Cependant, cependant, ces plans ont été rendus rendus, quand plus de plus de
> plans, plus d'en plus de l'armée rouge est entrée et créée...

`take(self.source_len)` cut the source at 64 tokens — the length the embedding
graph happens to have been *traced* at — and did it **after** the terminator was
appended, so a longer source lost both its tail and its `<s>`. The encoder then
sees an unfinished sentence, and the decoder answers an unfinished sentence with
repetition, which is why the symptom pointed nowhere near the cause.

Uncapped, the same input:

> Cependant, ces plans sont devenus obsolètes presque du jour au lendemain,
> lorsque plus de 800 000 soldats de l'Armée rouge de l'Union soviétique sont
> entrés et ont créé les fronts biélorusse et ukrainien...

The graphs are length-polymorphic — 501 source pieces run without complaint — so
64 was never a model limit. Nor is it the framework's behaviour: over the 100 longest
FLORES sentences, the shipped framework's output covers the whole source, mean
length ratio 1.17 with a minimum of 0.99 and not one output short enough to
suggest a cut. Removing the cap moves *towards* the OS, not away from it.

There is no cap by default now;
`RLX_TRANSLATE_MAX_SOURCE_TOKENS` sets one, and a cut keeps room for the
terminator. The 855-sentence corpus is byte-identical afterwards (653 exact,
chrF 0.955), because none of its sentences was long enough to be cut — which is
the whole argument for measuring on text the port was not built against.

### Byte fallback was reaching the model as literal text

The FLORES benchmark put one direction far outside the pack: `th_TH-en_US`
scored 0.529 against the human reference where the OS scored 0.683, and agreed
with the OS only 0.651. On the hand-built corpus the same direction scored 0.945
— because that corpus's Thai was the OS's own output fed back. Human Thai broke
it.

The raw NMT was fine. `nbest` gave `Gosling and Stone were nominated for Best
Actor and Best Actress.`; the stage graph gave `...Attoraged, ADecasting,
ORTHESSORTING,ADSORTHSORTINGDEIED`. The `spm_encode` trace says why:

```
น<0xE0><0xB8><0xB3>ชาย
```

Thai SARA AM has no piece in the vocabulary, so SentencePiece spells it in
bytes. That is normal and the model is trained for it. What was not normal is
that four places in this crate rendered tokens back to text by concatenating
`piece()` strings, leaving the `<0xNN>` markers *in the string* — and the stage
graph re-renders `spm_encode`'s tokens and hands the result to the translator.
The NMT was being asked to translate a string containing `<0xE0><0xB8><0xB3>`.

`Vocab::decode` collects the bytes and decodes them together — one fallback
character is several pieces and none of them is valid UTF-8 alone — and is now
used everywhere tokens become text, including output rendering, which had the
same latent bug for any target containing fallback.

| `th_TH-en_US` | before | after |
| --- | --- | --- |
| vs human | 0.529 | **0.672** |
| vs the OS | 0.651 | **0.871** |

That is -0.154 behind the OS becoming -0.011. Re-running all 57 directions, the
fix helps everywhere and **regresses nothing**:

| | before | after |
| --- | --- | --- |
| ours vs human | 0.672 | **0.677** |
| agreement with the OS | 0.924 | **0.932** |
| mean delta vs the OS | -0.0074 | **-0.0022** |
| directions where we beat the OS | 19 / 57 | **22 / 57** |
| worst direction | -0.154 | **-0.018** |

Biggest gains after Thai: `fr_FR-en_US` +0.032, `ar_AE-en_US` +0.026,
`fr_FR-de_DE` +0.025, `ko_KR-en_US` +0.017, `zh_TW-en_US` +0.016. `examples/fallback_scan.rs` measures the blast radius — **every**
language falls back somewhere:

| locale | sentences with fallback | characters |
| --- | --- | --- |
| th_TH | 56.5% | `ำ` x184 |
| fr_FR | 39.0% | `’` x107, nbsp x64, thin space x15 |
| hi_IN | 9.0% | nukta forms `फ़` `ड़` `ज़` |
| zh_TW | 8.0% | `．` `…` `姊` |
| ar_AE | 6.0% | Arabic diacritics |

French's are typographic variants, which looked like a missing normalizer step —
but the shipped `normalizer.pat` is 40 bytes and only trims whitespace, so the OS
is not mapping them either. Byte fallback is the intended path; only the
round-trip through text was wrong.

### Input that is not a sentence

Every measurement in this file is sentences, because both corpora are
sentences. A translation app receives text boxes. Fifteen edge cases put
through `TREdge` — a harness that asks the shipped framework directly rather
than guessing — found four defects:

| input | the OS | before | now |
| --- | --- | --- | --- |
| `""` | error `nothingToTranslate` | `'` | `""` |
| `"   "` | `"   "` | `'` | `"   "` |
| `"🎉🎉🎉"` | `"🎉🎉🎉"` | `"🎉"` | `"🎉🎉🎉"` |
| `"a\nb"` | `"A\n\nB"` | `"Une"` | `"A\n\nB"` |
| `"one\ntwo\nthree"` | `"Un\n\nDeux\n\nTrois"` | `"Un"` | matches |

Two rules, both read off the framework rather than invented:

- **A line break is a paragraph boundary.** Each line is translated on its own
  and the results are joined with a blank line. Handing the whole string to the
  model lost every line after the first — `"a\nb"` came back as `"Une"`.
- **Input with no letter in it is returned unchanged.** The model was being
  asked to translate `""` and answering `'`; `"🎉🎉🎉"` and answering `"🎉"`.
  `"123"`, `"!!!"` and `"OK"` were already right by luck.

Widening that to 49 cases found five more, all one family — what counts as a
line break — plus one that matters more than the rest:

| input | the OS | before | now |
| --- | --- | --- | --- |
| `"a\tb"` | `"A b"` | **`"A A"`** | `"A b"` |
| `"a\rb"` | `"A\n\nB"` | **`"Une té"`** | `"A\n\nB"` |
| `"a\n\n\nb"` | `"A\n\nB"` | `"A\n\n\n\n\n\nB"` | `"A\n\nB"` |
| `"a\u{200b}b"` | `"A b"` | `"A\u{200b}b"` | `"A b"` |

A run of line breaks — `\n`, `\r`, `\r\n`, any number — is **one** paragraph
boundary; every other whitespace, tab and zero-width space included, is a space
within a line.

### Text the model cannot carry, it alters

The one that matters. `en_US-fr_FR` on `"السلام hello"` returned
`"Bonjour السلنم"` — one Arabic letter changed — and with the words the other
way round it produced `"السل٧م"`, containing an Arabic-Indic *digit*. The
framework returns the run untouched. Chinese, Japanese and Cyrillic came through
intact, so this had gone unnoticed; Arabic is where the French-family bundle's
coverage runs out.

Silently altering text the user typed is worse than leaving it untranslated, so
`dnt::restore_foreign_scripts` puts back any run in a script neither language
writes. It is deliberately conservative — it fires only when the output holds
exactly as many runs of that script as the source did, so a genuine translation
*into* that script is never overwritten. Both orders are now returned intact.

A detail that cost a debugging round: the corrupted run contained U+0667, which
is not `char::is_alphabetic`, so it split one run into two, the counts stopped
matching and the repair declined to fire. Script-specific digits belong to their
script.

Also fixed: `dnt::lost` compared case-sensitively, so the sentence caser turning
`#hashtag` into `#Hashtag` was reported as a span lost in translation on every
sentence beginning with one.

Of the 49, eighteen still differ and the interesting ones are not defects: `MiXeD CaSe TEXT` is a
word-order choice; on `See https://example.com/a?b=1 for details` **the OS
mangles the URL** to `https://example.com/a ? B=1` while this port leaves it
intact; and on 56 `a`s the OS returns 57. Two real ones remain, both cosmetic:
`-40°C` gains a space, and `<b>bold</b>` has its tag capitalised to `<B>`.
the OS also localises `5 km` to `5 kilomètres` and `1,000,000` to `1 000 000`,
which this port does not attempt.

The lesson is the ratio: four defects in fifteen non-sentence inputs, against
none in 4175 sentences. The corpora say the model is right; they say nothing
about the code around it.

### Pivot directions are measured now, not skipped

A seventh of what this machine can translate has no single-hop model: the OS
routes `ar_AE-de_DE`, `it_IT-ja_JP` and twelve others through English, so their
plans hold two `PDecTranslatorBlock`s with different bundles. The agreement gate
resolved one model per *direction* and skipped all fourteen.

It resolves one per translator *stage* now — keyed on bundle, both locales and
the shortlist table, since `input_<lang>` follows the source and zh_CN/zh_TW
share a bundle but not a table. The 43 single-hop directions come out
byte-identical, 506 exact and chrF 0.960, which is what a coverage change should
do. The fourteen new ones average chrF 0.940 with 147 of 210 exact, topped by
`tr_TR-fr_FR` at 0.993 and `vi_VN-es_ES` at a clean 15 of 15.

Two-hop pivoting was already implemented; what was missing was any measurement
of it.

### A weak direction that was not weak

`en_US-uk_UA` looked like the worst direction on the conversational corpus:
0.693 against the human reference where the OS scored 0.723. Two plausible causes
were checked and both refuted by data before any code was written.

*Punctuation.* the OS renders `Авраам Лінкольн — відома людина.` with an em dash
where we write `-`, which looks like a Ukrainian typographic rule. Counting
across the OS's own output: 7 spaced hyphens against 2 em dashes in the
conversational corpus, 5 against 1 in FLORES. the OS is not applying a rule; the
em dash is its model's occasional choice.

*The phrasebook.* the OS scored a perfect 1.000 against the human reference on
`A car is outside.`, which is phrasebook-shaped, and the agreement harness
disables the phrasebook. But the CLI with it enabled produces the same output —
these are not lexicon entries.

What it actually was: **sampling noise.** Of 50 sentences, 39 were exact ties,
we won 1 and lost 10, and four paraphrase coin-flips carried the entire gap
(deltas -0.43, -0.38, -0.25, -0.23; excluding them the mean is -0.005). At
n=100 the gap disappears:

| direction | n=50 | n=100 |
| --- | --- | --- |
| `en_US-uk_UA` (conversational) | -0.030 | **-0.004** |

Measured at full depth across both corpora, French and Ukrainian are at parity:
1100 FLORES sentences give ours 0.696 against the OS's 0.700, and 600
conversational sentences give ours 0.745 against 0.744. We are ahead on
`en_US-fr_FR` (+0.011 conversational), `uk_UA-fr_FR` (+0.014) and
`en_US-uk_UA` (+0.001 on FLORES). The one French direction still measurably
behind is `th_TH-fr_FR` at -0.018 — and that is the Thai side of a pivot, the
language with 56.5% byte-fallback, not the French side.

The lesson is the same one this port keeps relearning, now with the variance to
quantify it: at n=50 with a per-sentence standard deviation of 0.093, the
standard error is 0.013, so anything under about 0.03 is indistinguishable from
noise. Several "worst direction" figures earlier in this file sit inside that
band.

### A convincing result that did not survive the corpus

`en_US-tr_TR` and `en_US-zh_TW` were still the two worst directions afterwards,
and Turkish failed in a specific way: word-for-word output in *English* word
order, and with the shortlist off, the English source reproduced verbatim.
`en_US-hi_IN` and `en_US-ar_AE` — same bundle shape, identical manifest flags,
identical tag inventory — translate fluently either way, so it was not the
bundle family.

`examples/tag_placement.rs` walks the ways the direction tags can sit around the
sentence. A direction's `target-token` is often two tags: the language,
`<tar-tr_TR>`, and a corpus tag, `<en_US-tr_TR-optimal>`. Greedy, one sentence:

| prefix | tr | fr | hi |
| --- | --- | --- | --- |
| `src tar opt` + text | word salad | correct | correct |
| `src tar` + text (no opt) | **exact** | correct | garbage |
| `src tar` + text + `opt` | good | correct | correct |

Trailing the corpus tag looked like the answer — it is the only column that
works everywhere, and under beam search it renders that Turkish sentence
*exactly* as the OS does. Over 120 sentences in 8 directions it is a small net
loss: chrF 0.757 -> 0.753, 39 exact -> 34, and `en_US-tr_TR` itself drops to
0.455. The effect is real but direction-dependent, so it stays off, behind
`RLX_TRANSLATE_DOMAIN_TAG_LAST`, with the numbers recorded on the field.

It is the cleanest example so far of something this port has now been caught by
twice: a single sentence, or a single direction, is not evidence about a format.

## Speed

### What decoding actually costs

`examples/profile_decode.rs` times each phase; `RLX_TRANSLATE_PROFILE=1` adds a
per-op breakdown from inside the executor. For `en_US-fr_FR` on a 9-word
sentence, the parts summed to ~430 ms while the whole took 10.9 s — which
localised the cost to the search rather than the model.

**Absolute timings on this machine are not stable.** It is shared, and it sat
at load 78 on 14 cores throughout; the same binary on the same sentence
measured 87 ms and 589 ms of encoding within the hour. So every claim below is
an *interleaved A/B* — arms alternated within one loop, so contention hits both
— rather than a before-and-after.

Two candidate causes were tested, both mine, each behind a switch:

| | replay prefixes | cache prefix state |
| --- | --- | --- |
| **wait for 8 completions** | 1779 / 2176 ms | 1023 / 1075 ms |
| **wait for 2** | 1754 / 1729 ms | 1010 / 887 ms |

- `BeamAdapter::log_probs` took an arbitrary prefix and replayed it from a fresh
  state, so a beam of 8 over 15 steps ran ~960 decoder steps to perform ~120.
  Beam search only ever extends a prefix by one token, so caching the state each
  prefix leaves behind makes the replay a single step. **This is the win: ~2x.**
  `RLX_TRANSLATE_INCREMENTAL=0` restores the replay, and
  `tests/real_decode.rs` runs both and requires identical text and scores — a
  stale cache would change translations, not crash.
- The search's `nbest` means both "results to return" and "completions to wait
  for", so over-generating for deduplication also made a 1-best request wait for
  eight. Splitting them (`stop_after`) is the *right* shape and it is what the
  defaults now do, but measured against the cache it is **within noise** on this
  sentence. A first sequential reading credited it with the second half of the
  speedup; that was machine load, not the change.

Output is unchanged by both: same translations, same scores.

### Where the remaining time goes

With `RLX_TRANSLATE_PROFILE=1` the executor tallies every op, by kind and by
graph. Over one whole translation:

| op | ms | calls | share | | graph | ms | calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `inner_product` | 332.8 | 2826 | 78.2% | | `decoder_fr` | 732.6 | 14416 |
| `dynamic_dequantize` | 29.1 | 2826 | 6.8% | | `encoder` | 526.0 | 1188 |
| `transpose` | 17.3 | 258 | 4.1% | | `handover_fr` | 270.3 | 474 |
| `batch_matmul` | 15.7 | 936 | 3.7% | | `input_fr` | 90.2 | 399 |

One int8 GEMM is four fifths of the time, and the decoder — thousands of
single-row steps — is about half of it. Both readings point at the same
optimisation, and **both are misleading**. Three were tried:

- **Hand-vectorising the dot product.** `examples/dot_bench.rs` measures the
  existing loop at ~75 GMAC/s standalone and ~50 GMAC/s in the GEMV shape the
  decoder runs. LLVM vectorises the zip perfectly well. A NEON version
  (`vmull_s8` + `vpadalq_s16`; `vdotq_s32` is still unstable) managed 34 GMAC/s
  at n=512 and 5 GMAC/s at n=2048 — 2x to 14x *slower*, and was deleted.
- **Threading the GEMM.** Its output rows are independent, so
  `exec::for_each_output` can split them. Measured: 926/945 ms at one lane,
  1509/1252 at four, 1550/1147 at fourteen. Kept as
  `RLX_TRANSLATE_GEMM_LANES`, defaulting to 1.
- **Batching the beam into one decoder call.** Eight hypotheses advance through
  the same position, so they could share one call and read the weights once.
  `examples/batch_probe.rs` asks the graph directly and it refuses:
  `reshape_2` is declared `[8, 1, 64]`, heads by *one query* by head-dim. Making
  it `[8, n, 64]` needs a transpose inserted around it, because a stacked
  `[n, 512]` is row-major and the reshape wants head-major. `Nmt::step_batch`
  is written and the probe validates row 0 against a single step, so the
  remaining work is a graph rewrite — in the attention layout, which is where
  this port's worst bugs have lived. Not attempted on that basis.

The reason all three miss is in the numbers themselves. The profiler reports
118-218 us per `inner_product` call; the benchmark measures 21 us of arithmetic
for the same shape. The gap is not code. This machine is shared and sat at
**load 78 on 14 cores** throughout — the same GEMV benchmark, unchanged,
measured 21 us and 141 us on two runs an hour apart. The benchmark's
best-of-five filters the preemption that the profiler's every-call mean
includes.

So the honest summary is that the executor's arithmetic is already close to
what the core can deliver, the one real win available was algorithmic (the
prefix-state cache, ~2x), and further micro-optimisation here needs a quiet
machine to even be measurable.

### The beam was wider than the OS's, and it bought nothing

`examples/why_missed.rs` asks, for every sentence where our answer differs from
the OS's, which of two things is true: does our own model score *the OS's* string
higher than the one we emitted — a search failure, recoverable by searching
harder — or lower, in which case searching harder cannot help and the
disagreement is in the weights or in a stage this port does not implement.

Over 22 misses in three directions: **21 times the model prefers our answer.**
So the remaining gap is not something a wider beam reaches. Which raised the
opposite question, because this port widened the beam to 8 while every shipped
config asks for 3:

| | exact | chrF | time |
| --- | --- | --- | --- |
| beam 8 (was) | 509 / 645 | 0.959 | 843 s |
| beam 3 (config) | 507 / 645 | 0.959 | **376 s** |

Two fewer exact matches out of 645, identical chrF, **2.24x faster**, and what
the OS actually does. A 1-best request now uses the config's beam; asking for
several variants still widens it, because distinct variants have to come from
somewhere.

### `rs-beam` finally wired up

Every config sets `rs-beam: 0.66` and this port never used it. `PruningPolicy`
was written with the note that guessing the framework's semantics "would corrupt output
in a way that is hard to attribute", and to wire it up once there was a
reference decode to diff against. 645 sentences is that reference.

Reading it as `cutoff = best - factor * |best|`:

| | exact / 645 | chrF | time |
| --- | --- | --- | --- |
| off | 507 | 0.959 | 376 s |
| on | 506 | **0.960** | **212 s** |

One exact match either way, which is *not* evidence the semantics are right —
only that they do not corrupt. It is on now because the config asks for it and
it nearly halves the time. `RLX_TRANSLATE_RS_BEAM_PRUNE=0` is the way back.

Between this and the beam width, a full 43-direction sweep went from 843 s to
212 s at the same quality.

## License

GPL-3.0-only, as with the rest of the workspace.
