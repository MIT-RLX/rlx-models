#!/usr/bin/env python3
"""Prepare weights/* model dirs for separate Hugging Face repos.

Writes/updates per-dir:
  - README.md   (HF model card)
  - LICENSE     (when missing and license is MIT/Apache-2.0)
  - .gitattributes (Git LFS patterns)

Does not upload. Re-run after fetching or packing new trees.

Usage:
  python3 scripts/prepare_weights_hf.py --cleanup
  python3 scripts/prepare_weights_hf.py --only moss-nano,soprano,tiny-tts-rlx
  python3 scripts/prepare_weights_hf.py --cleanup --only rlx-asr

Native RLXP models document nested graphs/*.rlxp (no Hub ONNX). Upload with
scripts/publish_weights_hf.py afterward.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

# Shared blurb for official flat `.rlxp` (RLXPFLAT) packs.
RLXP_CONTAINER_LINES = [
    "Official RLX package format (`RLXPFLAT`, container v2).",
    "",
    "```text",
    "[0..8)   magic          RLXPFLAT",
    "[8..12)  version        u32 LE (= 2)",
    "[12..16) flags          u32 LE (hybrid hot/warm/cold)",
    "[16..24) toc_len        u64 LE",
    "[24..)   TOC            JSON table of contents",
    "         data region    64-byte aligned payloads",
    "```",
    "",
    "The TOC lists **tensors** (named weight blobs) and/or **sidecars** (files: ONNX,",
    "tokenizers, manifests, …). Sidecars are usually **cold + zstd**; model weights in",
    "tensor packs are **hot + uncompressed** for mmap. Runtime crates open the pack",
    "directly (or materialize sidecars to a temp dir for asset-only packs).",
]

# Suggested Hub id under the account that already hosts kitten-tts-mini-0.8-rlx.
ORG = "eugenehp"

APACHE2 = """                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
"""

MIT = """MIT License

Copyright (c) the original authors and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

GITATTRIBUTES = """\
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.gguf filter=lfs diff=lfs merge=lfs -text
*.rlxp filter=lfs diff=lfs merge=lfs -text
*.rlxpack filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.onnx_data filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.f32 filter=lfs diff=lfs merge=lfs -text
*.data filter=lfs diff=lfs merge=lfs -text
*.wav filter=lfs diff=lfs merge=lfs -text
*.mp3 filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
"""

# path relative to weights/ → metadata
# kind: rlx-native | redistrib | converted
MODELS: dict[str, dict] = {
    # vision
    "vision/bioclip-2": {
        "title": "BioCLIP-2 (RLX staging)",
        "kind": "redistrib",
        "license": "mit",
        "pipeline": "zero-shot-image-classification",
        "upstream": "https://huggingface.co/imageomics/bioclip-2",
        "crate": "rlx-bioclip2",
        "run": 'cargo run -p rlx-bioclip2 --release -- --model-dir . --image photo.jpg --labels "cat,dog"',
        "summary": "OpenCLIP ViT-L/14 biology foundation weights for RLX.",
        "tags": ["biology", "clip", "vision", "rlx"],
    },
    "vision/dinov2": {
        "title": "DINOv2 ViT-L/14 (RLX meta layout)",
        "kind": "converted",
        "license": "apache-2.0",
        "pipeline": "image-feature-extraction",
        "upstream": "https://huggingface.co/facebook/dinov2-large",
        "crate": "rlx-dinov2",
        "run": "cargo run -p rlx-dinov2 --release -- --weights dinov2_vitl14.meta.safetensors --image photo.jpg",
        "summary": "DINOv2 ViT-L/14 with HF + RLX meta-key safetensors.",
        "tags": ["dinov2", "vision", "rlx"],
        "files_note": "`dinov2_vitl14.safetensors` (HF keys) and `dinov2_vitl14.meta.safetensors` (RLX meta layout).",
    },
    "vision/sam-v1": {
        "title": "SAM ViT-B + InsectSAM (RLX)",
        "kind": "converted",
        "license": "apache-2.0",
        "pipeline": "mask-generation",
        "upstream": "https://github.com/facebookresearch/segment-anything",
        "crate": "rlx-sam",
        "run": "just sam1 -- --weights sam_vit_b_meta.safetensors --device cpu --point 512,512",
        "summary": "SAM ViT-B meta-layout weights plus InsectSAM safetensors for RLX.",
        "tags": ["sam", "segmentation", "rlx"],
        "files_note": "`sam_vit_b_meta.safetensors`, `insectsam.safetensors`.",
    },
    "vision/siglip2-base-224": {
        "title": "SigLIP 2 Base patch16-224 (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "zero-shot-image-classification",
        "upstream": "https://huggingface.co/google/siglip2-base-patch16-224",
        "crate": "rlx-siglip2",
        "run": 'cargo run -p rlx-siglip2 --release -- --model-dir . --image photo.jpg --labels "a cat, a dog"',
        "summary": "Google SigLIP 2 fixed-resolution base checkpoint for RLX.",
        "tags": ["siglip2", "vision", "rlx"],
    },
    "vision/siglip2-base-naflex": {
        "title": "SigLIP 2 Base NaFlex (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "zero-shot-image-classification",
        "upstream": "https://huggingface.co/google/siglip2-base-patch16-naflex",
        "crate": "rlx-siglip2",
        "run": 'cargo run -p rlx-siglip2 --release -- --model-dir . --image doc.png --labels "a document, a photo"',
        "summary": "Google SigLIP 2 NaFlex (variable aspect) checkpoint for RLX.",
        "tags": ["siglip2", "naflex", "vision", "rlx"],
    },
    "vision/uni2": {
        "title": "UNI2-h (RLX staging)",
        "kind": "redistrib",
        "license": "cc-by-nc-nd-4.0",
        "pipeline": "image-feature-extraction",
        "upstream": "https://huggingface.co/MahmoodLab/UNI2-h",
        "crate": "rlx-uni2",
        "run": "cargo run -p rlx-uni2 --release -- --weights uni2h.safetensors",
        "summary": "MahmoodLab UNI2-h pathology ViT-H/14. Gated CC-BY-NC-ND — respect upstream terms.",
        "tags": ["pathology", "uni2", "vision", "rlx"],
        "write_license_file": False,
        "files_note": "`uni2h.safetensors` + `config.json` (PyTorch `.bin` removed as duplicate).",
    },
    # lm
    "lm/bonsai-27b-gguf": {
        "title": "Bonsai-27B Q1_0 GGUF (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "text-generation",
        "upstream": "https://huggingface.co/prism-ml/Bonsai-27B-gguf",
        "crate": "rlx-qwen35",
        "run": "cargo run -p rlx-qwen35 --release -- --weights Bonsai-27B-Q1_0.gguf --device metal",
        "summary": "prism-ml Bonsai-27B packed Q1_0 GGUF for RLX Qwen3.5 runner.",
        "tags": ["gguf", "qwen35", "bonsai", "rlx"],
    },
    "lm/qwen3-0.6b": {
        "title": "Qwen3-0.6B (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "text-generation",
        "upstream": "https://huggingface.co/Qwen/Qwen3-0.6B",
        "crate": "rlx-qwen3",
        "run": "cargo run -p rlx-qwen3 --release -- --weights . --device metal",
        "summary": "Qwen3-0.6B safetensors + tokenizer for RLX.",
        "tags": ["qwen3", "rlx"],
        "write_license_file": False,  # keep upstream LICENSE
        "keep_existing_readme": False,
    },
    "lm/qwen3-0.6b-gguf": {
        "title": "Qwen3-0.6B Q4_K_M GGUF (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "text-generation",
        "upstream": "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF",
        "crate": "rlx-qwen3",
        "run": "cargo run -p rlx-qwen3 --release -- --weights Qwen3-0.6B-Q4_K_M.gguf --packed --device metal",
        "summary": "Qwen3-0.6B Q4_K_M GGUF + tokenizer.json for packed RLX prefill.",
        "tags": ["qwen3", "gguf", "rlx"],
    },
    # audio
    "audio/w2v-bert-2.0": {
        "title": "Wav2Vec2-BERT 2.0 (RLX staging)",
        "kind": "redistrib",
        "license": "mit",
        "pipeline": "feature-extraction",
        "upstream": "https://huggingface.co/facebook/w2v-bert-2.0",
        "crate": "rlx-wav2vec2-bert",
        "run": "export RLX_W2V_BERT_DIR=.; cargo test -p rlx-wav2vec2-bert --release",
        "summary": "Seamless W2V-BERT 2.0 Conformer encoder weights for RLX.",
        "tags": ["wav2vec2-bert", "speech", "rlx"],
    },
    # asr
    "asr": {
        "title": "RLX streaming Conformer ASR",
        "kind": "rlx-native",
        "license": "apache-2.0",
        "pipeline": "automatic-speech-recognition",
        "upstream": None,
        "crate": "rlx-asr",
        "run": "just fetch-rlx-asr && cargo run -p rlx-asr --release -- transcribe --wav clip.wav",
        "summary": "Single-file RLX streaming ASR pack (frontend, VAD, Conformer, CTC, AED) as `model.rlxp`.",
        "tags": ["asr", "rlxp", "conformer", "rlx"],
        "primary": ["model.rlxp"],
        "avoid": [],
        "fetch": "just fetch-rlx-asr   # or: hf download eugenehp/rlx-asr model.rlxp --local-dir weights/asr",
        "files_note": "Hub ships `model.rlxp` only. Pack locally with `just asr-pack-rlxp` (can convert from a local `model.gguf`).",
        "notes": [
            "Hub ships `.rlxp` only. A local legacy `model.gguf` still loads if present."
        ],
        "rlxp": {
            "file": "model.rlxp",
            "intro": (
                "Tensor pack for streaming Conformer ASR: named weights plus CTC unit list "
                "and etiquette metadata as sidecars. No ONNX graphs — the RLX graph is built at runtime."
            ),
            "architecture": [
                "**Pipeline:** frontend/encoder → Conformer → CTC/AED decode.",
                "",
                "| Prefix | Role | dtype |",
                "|---|---|---|",
                "| `encoder.*` | Conformer stack | mostly f32 |",
                "| `decoder.*` | AED / CTC head | f32 |",
                "| `codebook.*` | discrete tables | f32 / i8 |",
                "| `ls.*` / `tp.*` | layer-scale / projections | f32 |",
            ],
            "tensors_note": (
                "Weights come from the former `model.gguf` (`rlx-asr` layout). Most are `f32`; "
                "a few codebook / index tensors are `i8`."
            ),
            "modules": {
                "encoder": "Conformer encoder stack",
                "decoder": "AED / CTC projection (`effective_We`, …)",
                "ls": "Layer-scale / auxiliary layer tensors",
                "codebook": "Discrete codebook tables",
                "tp": "Token / projection helpers",
                "silence_fbank": "Silence filterbank reference",
            },
            "sidecars_note": "Text metadata only — no neural graphs in sidecars.",
            "roles": {
                "units.txt": "CTC / unit vocabulary (one symbol per line)",
                "etiquette.json": "Pack etiquette / runtime metadata",
            },
            "tree": [
                "model.rlxp",
                "├── tensors/          # hot mmap region",
                "│   ├── encoder.*     # Conformer",
                "│   ├── decoder.*     # AED head",
                "│   ├── ls.* / codebook.* / tp.*",
                "│   └── silence_fbank",
                "└── sidecars/         # cold zstd",
                "    ├── units.txt",
                "    └── etiquette.json",
            ],
            "pack": (
                "`just asr-pack-rlxp` → `rlx-asr-pack-gguf --rlxp`. Prefers converting an existing "
                "local `model.gguf`; otherwise packs from loose sources under `.cache/asr` / `weights/asr`."
            ),
        },
        "repo_name": "rlx-asr",
    },
    # tts — RLX native / converted first
    "tts/rlx-tts": {
        "title": "RLX TTS (FastSpeech2 + WaveRNN)",
        "kind": "rlx-native",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": None,
        "crate": "rlx-tts",
        "run": 'just fetch-rlx-tts && just tts-demo',
        "summary": "Single-file RLX FastSpeech2 + WaveRNN TTS as `rlx-tts.rlxp`.",
        "tags": ["tts", "rlxp", "rlx"],
        "primary": ["rlx-tts.rlxp"],
        "avoid": [],
        "fetch": "just fetch-rlx-tts   # or: hf download eugenehp/rlx-tts rlx-tts.rlxp --local-dir weights/tts/rlx-tts",
        "files_note": "Hub ships `rlx-tts.rlxp` only. Pack locally with `just export-rlx-tts-rlxp`.",
        "notes": [
            "Hub ships `.rlxp` only. A local legacy `rlx-tts.gguf` still loads if present."
        ],
        "rlxp": {
            "file": "rlx-tts.rlxp",
            "intro": (
                "Tensor + frontend pack: FastSpeech2 encoder/decoder and WaveRNN weights in the hot "
                "region; G2P / lexicon / neural frontend files as cold sidecars."
            ),
            "architecture": [
                "**Pipeline:** text → G2P frontend → FastSpeech2 encoder → mel decoder → WaveRNN → WAV.",
                "",
                "| Prefix | Role | dtype |",
                "|---|---|---|",
                "| `encoder.*` | FastSpeech2 + variance adaptor | f32 |",
                "| `decoder.*` | mel decoder | f32 |",
                "| `wavernn.*` | vocoder | f32 |",
            ],
            "tensors_note": "All acoustic weights are `f32` row-major, hot (uncompressed).",
            "modules": {
                "encoder": "FastSpeech2 encoder + variance adaptor",
                "decoder": "FastSpeech2 mel decoder",
                "wavernn": "WaveRNN vocoder",
            },
            "sidecars_note": "Text-normalization / G2P assets used before the acoustic model.",
            "roles": {
                "manifest.json": "Bundle manifest (sample rate, paths, versions)",
                "neural_fe_config.json": "Neural frontend config",
                "post.cfg": "Post-filter / WaveRNN post config",
                "symmap.json": "Symbol map",
                "lexicon.txt": "Lexicon overrides",
                "frontend/g2p_seq2seq.safetensors": "Seq2seq G2P checkpoint",
                "frontend/g2p_bpe.json": "G2P BPE vocab",
                "frontend/g2p_post_rule.dat": "G2P post rules",
                "frontend/g2p_lhp_rule.dat": "G2P LHP rules",
                "frontend/rewrite_rule.dat": "Rewrite transducer rules",
                "frontend/tn_prefix_rule.dat": "Text-norm prefix rules",
                "frontend/gprm_index.json": "GPRM index",
                "frontend/rewrite_map.json": "Rewrite map",
                "frontend/nashville_isym_phones.json": "Phone inventory",
                "frontend/phonetic/to_lhp.json": "Phone → LHP map",
            },
            "tree": [
                "rlx-tts.rlxp",
                "├── tensors/",
                "│   ├── encoder.*     # FastSpeech2",
                "│   ├── decoder.*",
                "│   └── wavernn.*     # vocoder",
                "└── sidecars/",
                "    ├── manifest.json / neural_fe_config.json / post.cfg / symmap.json",
                "    ├── lexicon.txt",
                "    └── frontend/     # G2P + TN rules + safetensors",
            ],
            "pack": (
                "`just export-rlx-tts-rlxp` → `rlx-tts --pack-rlxp`. Converts local `rlx-tts.gguf` "
                "or packs encoder/decoder/wavernn safetensors + frontend files from the bundle dir."
            ),
        },
        "repo_name": "rlx-tts",
    },
    "tts/tiny-tts-rlx": {
        "title": "TinyTTS / MeloTTS RLX bundle",
        "kind": "rlx-native",
        "license": "mit",
        "pipeline": "text-to-speech",
        "upstream": "https://github.com/tronghieuit/tiny-tts",
        "crate": "rlx-tiny-tts",
        "run": 'just fetch-tiny-tts && cargo run -p rlx-tiny-tts --release --features apple-silicon -- --data weights/tts/tiny-tts-rlx --text "Hi." --device metal --out /tmp/tiny.wav',
        "summary": "MeloTTS/VITS2 English nested `.rlxp` subgraphs + frontend for RLX TinyTTS and MeloTTS. CPU / Metal / MLX / CUDA / wgpu.",
        "tags": ["tts", "melotts", "vits2", "rlx", "rlxp"],
        "primary": ["tiny-tts.rlxp"],
        "fetch": "just fetch-tiny-tts   # or: hf download eugenehp/tiny-tts-rlx tiny-tts.rlxp --local-dir weights/tts/tiny-tts-rlx",
        "files_note": "Hub ships `tiny-tts.rlxp` only (nested graph packs + frontend). Also used by `rlx-melotts` / `just fetch-melotts`. Runtime does **not** load `.onnx` from Hub.",
        "avoid": [],
        "notes": [
            "MeloTTS (`rlx-melotts`) loads this same bundle — locally `weights/tts/melotts` is a symlink.",
            "Hub ships **no ONNX**. Nested `graphs/*.rlxp` hold hot f32/i64 tensors + `graph.json`; the crate lowers to HIR per utterance length."
        ],
        "rlxp": {
            "file": "tiny-tts.rlxp",
            "intro": (
                "Outer RLXPFLAT bundle: nested native subgraph packs under `graphs/` "
                "(hot weight tensors + graph IR sidecars) plus English frontend assets. "
                "No `.onnx` on Hub. `rlx-tiny-tts` materializes the outer pack, then lowers "
                "each nested pack to HIR for the utterance length."
            ),
            "architecture": [
                "**Pipeline:** text → English frontend (G2P) → `text_encoder` → `duration_predictor` → "
                "monotonic alignment + latent sample (Rust) → `flow` → `decoder` → 44.1 kHz mono WAV.",
                "",
                "| Module | Role | Typical dtype | Notes |",
                "|---|---|---|---|",
                "| `text_encoder` | phone/tone/lang → latent | f32 | length = phoneme count `T` |",
                "| `duration_predictor` | per-phone duration | f32 | |",
                "| `flow` | prior / flow | f32 | |",
                "| `decoder` | HiFi-GAN-style vocoder | f32 | ConvTranspose upsample ×512 |",
                "| frontend | G2P + BERT tokenizer + CMUdict | — | file sidecars |",
            ],
            "sidecars_note": (
                "Outer TOC is file sidecars only. Neural weights live **inside** each "
                "`graphs/<name>.rlxp` as hot mmap tensors (not safetensors)."
            ),
            "skip_sidecars": [".gitattributes", "LICENSE", "README.md"],
            "roles": {
                "config.json": "Model / sample-rate config (44.1 kHz)",
                "graphs/text_encoder.rlxp": "Nested pack: hot tensors + graph.json",
                "graphs/duration_predictor.rlxp": "Nested pack: hot tensors + graph.json",
                "graphs/flow.rlxp": "Nested pack: hot tensors + graph.json",
                "graphs/decoder.rlxp": "Nested pack: hot tensors + graph.json",
                "frontend/bert/tokenizer.json": "BERT WordPiece tokenizer",
                "frontend/bert/tokenizer_config.json": "Tokenizer config",
                "frontend/g2p_checkpoint.safetensors": "G2P checkpoint",
                "frontend/g2p_cmudict.txt": "CMUdict for G2P",
                "frontend/cmudict_rep.txt": "CMUdict repair / variants",
                "frontend/homographs.en": "Homograph list",
                "frontend/perceptron_tagger.json": "POS tagger",
                "frontend/symbols.json": "Symbol table",
            },
            "tree": [
                "tiny-tts.rlxp",
                "├── graphs/",
                "│   ├── text_encoder.rlxp      # hot f32/i64 + graph.json",
                "│   ├── duration_predictor.rlxp",
                "│   ├── flow.rlxp",
                "│   └── decoder.rlxp",
                "├── config.json",
                "└── frontend/                 # G2P / tokenizer (not neural)",
            ],
            "pack": (
                "`just export-tiny-tts-rlxp` / `pack_rlxp` example: ONNX pack-time source → "
                "nested `graphs/*.rlxp` via `rlx-assets` native-pack, then outer bundle. "
                "Hub artifact has zero `.onnx`."
            ),
        },
        "repo_name": "tiny-tts-rlx",
    },
    "tts/melotts": {
        "title": "MeloTTS → use tiny-tts-rlx",
        "kind": "rlx-native",
        "license": "mit",
        "pipeline": "text-to-speech",
        "upstream": "https://github.com/myshell-ai/MeloTTS",
        "crate": "rlx-melotts",
        "run": 'ln -sfn tiny-tts-rlx weights/tts/melotts && just melotts-demo "Hi." metal',
        "summary": "Alias card only — MeloTTS shares the weight tree with [`eugenehp/tiny-tts-rlx`](https://huggingface.co/eugenehp/tiny-tts-rlx). There is no separate weight dump in this repo.",
        "tags": ["tts", "melotts", "vits2", "rlx"],
        "primary": [],
        "files_note": "Download weights from [`eugenehp/tiny-tts-rlx`](https://huggingface.co/eugenehp/tiny-tts-rlx) (`tiny-tts.rlxp`). See that card’s **Pack layout** for the `.rlxp` tree. Locally, `weights/tts/melotts` can be a symlink to `tiny-tts-rlx`.",
        "fetch": "just fetch-melotts   # downloads eugenehp/tiny-tts-rlx tiny-tts.rlxp + local melotts symlink",
        "notes": [
            "Do not look for MeloTTS-specific GGUFs here; use the TinyTTS RLX bundle and `rlx-melotts`."
        ],
        "repo_name": "melotts",
        "write_license_file": False,
        "skip_upload": True,
        "card_only": True,
    },
    "tts/inflect-nano-rlx": {
        "title": "Inflect-Nano RLX bundle",
        "kind": "rlx-native",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/owensong/Inflect-Nano-v1",
        "crate": "rlx-inflect-nano",
        "run": 'cargo run -p rlx-inflect-nano --release -- --data . --text "Hello!" --out out.wav',
        "summary": "Exported Inflect-Nano acoustic + vocoder + frontend for RLX (optional parity fixtures).",
        "tags": ["tts", "inflect-nano", "rlx"],
        "repo_name": "inflect-nano-rlx",
    },
    "tts/snac-24khz": {
        "title": "SNAC 24 kHz decoder (RLX export)",
        "kind": "converted",
        "license": "mit",
        "pipeline": "audio-to-audio",
        "upstream": "https://huggingface.co/hubertsiuzdak/snac_24khz",
        "crate": "rlx-orpheus",
        "run": "export ORPHEUS_SNAC_PATH=snac_24khz_decoder.safetensors",
        "summary": "SNAC 24 kHz conv decoder exported to safetensors for RLX/Orpheus, plus small parity refs.",
        "tags": ["snac", "codec", "rlx"],
        "files_note": "`snac_24khz_decoder.safetensors`, `snac_24khz_decoder_config.json`, optional `ref_*.npy`.",
        "repo_name": "snac-24khz-rlx",
    },
    # tts redistribs
    "tts/chatterbox": {
        "title": "ChatterBox ONNX + RLX native (staging)",
        "kind": "converted",
        "license": "mit",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/synath/chatterbox-ONNX",
        "crate": "rlx-chatterbox",
        "run": "cargo run -p rlx-chatterbox --release -- --model-dir .",
        "summary": "ChatterBox ONNX tree plus RLX `native/` T3 extract (parity bins removed).",
        "tags": ["chatterbox", "tts", "onnx", "rlx"],
    },
    "tts/encodec24": {
        "title": "EnCodec 24 kHz (RLX staging)",
        "kind": "converted",
        "license": "mit",
        "pipeline": "audio-to-audio",
        "upstream": "https://huggingface.co/facebook/encodec_24khz",
        "crate": "rlx-encodec",
        "run": "cargo test -p rlx-encodec --release",
        "summary": "EnCodec 24 kHz safetensors for MetaVoice / RLX encodec path.",
        "tags": ["encodec", "codec", "rlx"],
    },
    "tts/f5tts": {
        "title": "F5-TTS ONNX (RLX staging)",
        "kind": "redistrib",
        "license": "mit",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/huggingfacess/F5-TTS-ONNX",
        "crate": "rlx-f5tts",
        "run": "cargo run -p rlx-f5tts --release -- --model-dir .",
        "summary": "F5-TTS ONNX graphs + vocab for RLX.",
        "tags": ["f5-tts", "onnx", "rlx"],
    },
    "tts/gepard": {
        "title": "Gepard-1.0 (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/nineninesix/gepard-1.0",
        "crate": "rlx-gepard",
        "run": 'cargo run -p rlx-gepard --release -- --weights . --text "Hello." --out out.wav',
        "summary": "Gepard TTS + NanoCodec decoder weights for RLX.",
        "tags": ["gepard", "tts", "rlx"],
    },
    "tts/kokoro-82m": {
        "title": "Kokoro-82M ONNX (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX",
        "crate": "rlx-kokoro",
        "run": "cargo run -p rlx-kokoro --release -- --model . --voice af_heart",
        "summary": "Kokoro-82M ONNX + voice embeddings for RLX.",
        "tags": ["kokoro", "tts", "onnx", "rlx"],
    },
    "tts/luxtts": {
        "title": "LuxTTS ONNX (RLX staging)",
        "kind": "converted",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/YatharthS/LuxTTS",
        "crate": "rlx-luxtts",
        "run": "cargo run -p rlx-luxtts --release -- --model-dir .",
        "summary": "LuxTTS ONNX encoder/decoder + Vocos vocoder assets for RLX.",
        "tags": ["luxtts", "tts", "onnx", "rlx"],
    },
    "tts/maya1": {
        "title": "Maya1 Q4_K_M GGUF (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/mradermacher/maya1-GGUF",
        "crate": "rlx-maya1",
        "run": "cargo run -p rlx-maya1 --release -- --weights maya1.Q4_K_M.gguf",
        "summary": "Maya1 speech LM GGUF + tokenizer for RLX (needs SNAC separately).",
        "tags": ["maya1", "gguf", "tts", "rlx"],
    },
    "tts/metavoice": {
        "title": "MetaVoice-1B (RLX safetensors)",
        "kind": "converted",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/metavoiceio/metavoice-1B-v0.1",
        "crate": "rlx-metavoice",
        "run": "cargo run -p rlx-metavoice --release -- --model-dir .",
        "summary": "MetaVoice stages converted to safetensors (PyTorch `.pt` removed).",
        "tags": ["metavoice", "tts", "rlx"],
    },
    "tts/miocodec": {
        "title": "MioCodec 25Hz/24kHz (RLX staging)",
        "kind": "redistrib",
        "license": "mit",
        "pipeline": "audio-to-audio",
        "upstream": "https://huggingface.co/Aratako/MioCodec-25Hz-24kHz",
        "crate": "rlx-miotts",
        "run": "cargo run -p rlx-miotts --release -- --codec-dir .",
        "summary": "MioCodec weights + ONNX decoder body (local parity fixtures removed).",
        "tags": ["miocodec", "codec", "rlx"],
        "keep_existing_readme": False,
    },
    "tts/miotts": {
        "title": "MioTTS-0.6B (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/Aratako/MioTTS-0.6B",
        "crate": "rlx-miotts",
        "run": "cargo run -p rlx-miotts --release -- --model-dir . --codec-dir ../miocodec",
        "summary": "MioTTS-0.6B speech LM + presets/samples for RLX.",
        "tags": ["miotts", "tts", "rlx"],
        "keep_existing_readme": False,
    },
    "tts/miratts": {
        "title": "MiraTTS (RLX staging)",
        "kind": "redistrib",
        "license": "cc-by-nc-sa-4.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/YatharthS/MiraTTS",
        "crate": "rlx-miratts",
        "run": "cargo run -p rlx-miratts --release -- --model-dir .",
        "summary": "MiraTTS LM + ONNX decoders for RLX. Non-commercial license — see upstream.",
        "tags": ["miratts", "tts", "rlx"],
        "write_license_file": False,
    },
    "tts/moss-nano": {
        "title": "MOSS-TTS-Nano (RLX)",
        "kind": "rlx-native",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        "crate": "rlx-moss-nano",
        "run": 'just fetch-moss-nano && just moss-nano',
        "summary": "Single runnable `moss-nano.rlxp`: nested native graphs (prefill/local/codec) + tokenizer + voices. No ONNX on Hub.",
        "tags": ["moss-tts", "rlxp", "tts", "rlx"],
        "primary": ["moss-nano.rlxp"],
        "avoid": [],
        "fetch": "just fetch-moss-nano   # or: hf download eugenehp/moss-nano moss-nano.rlxp --local-dir weights/tts/moss-nano",
        "files_note": "Hub ships `moss-nano.rlxp` only (nested `graphs/*.rlxp`). Pack locally with `just export-moss-nano-rlxp`. CPU / Metal / MLX / CUDA / wgpu.",
        "notes": [
            "Hub ships `.rlxp` only — **no ONNX / `.data`**. Nested packs hold hot tensors + graph.json."
        ],
        "rlxp": {
            "file": "moss-nano.rlxp",
            "intro": (
                "Outer RLXPFLAT: nested native subgraph packs (prefill, local frame, codec) plus "
                "tokenizer and voice manifest. **No `.onnx` / `.data` on Hub.** Weights are hot "
                "tensors inside each nested `.rlxp`; graph structure is `graph.json`."
            ),
            "architecture": [
                "**Pipeline:** text → tokenizer → global prefill (12-layer) → local frame sampler "
                "(16 codebook tokens/frame, CPU-pinned) → MOSS audio tokenizer decode → 48 kHz stereo.",
                "",
                "| Module | Role | Sample rate | Notes |",
                "|---|---|---|---|",
                "| `moss_tts_prefill` | global AR transformer | — | growing padded seq |",
                "| `moss_tts_local_fixed_sampled_frame` | local codebook sampler | — | 16 tokens/frame |",
                "| `moss_audio_tokenizer_decode_full` | codec → waveform | 48 kHz stereo | |",
                "| `browser_poc_manifest.json` | builtin voices | — | reference codes |",
            ],
            "sidecars_note": "Outer TOC: tokenizer/manifest + nested `graphs/*.rlxp` (neural).",
            "roles": {
                "browser_poc_manifest.json": "Voice / style manifest (builtin prompt codes)",
                "tokenizer.json": "Text tokenizer",
                "graphs/moss_tts_prefill.rlxp": "Nested: hot tensors + graph.json",
                "graphs/moss_tts_local_fixed_sampled_frame.rlxp": "Nested: hot tensors + graph.json",
                "graphs/moss_audio_tokenizer_decode_full.rlxp": "Nested codec pack",
            },
            "tree": [
                "moss-nano.rlxp",
                "├── graphs/",
                "│   ├── moss_tts_prefill.rlxp",
                "│   ├── moss_tts_local_fixed_sampled_frame.rlxp",
                "│   └── moss_audio_tokenizer_decode_full.rlxp",
                "├── browser_poc_manifest.json",
                "└── tokenizer.json",
            ],
            "pack": (
                "`just export-moss-nano-rlxp` — pack-time ONNX+`.data` → nested `graphs/*.rlxp` "
                "(external data inlined as tensors). Hub has zero ONNX."
            ),
        },
        "repo_name": "moss-nano",
    },
    "tts/neutts": {
        "title": "NeuCodec encoder (RLX staging)",
        "kind": "converted",
        "license": "apache-2.0",
        "pipeline": "audio-to-audio",
        "upstream": "https://huggingface.co/neuphonic/neucodec",
        "crate": "rlx-neutts",
        "run": "export NEUTTS_ENCODER_PATH=neucodec_encoder.safetensors",
        "summary": "NeuCodec encoder safetensors used by RLX NeuTTS voice clone.",
        "tags": ["neucodec", "neutts", "rlx"],
    },
    "tts/openvoice": {
        "title": "OpenVoice ONNX v2 (RLX staging)",
        "kind": "redistrib",
        "license": "mit",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/Hinotsuba/OpenVoice-ONNX-v2",
        "crate": "rlx-openvoice",
        "run": "cargo run -p rlx-openvoice --release -- --model-dir .",
        "summary": "OpenVoice tone-color ONNX assets for RLX (pairs with TinyTTS).",
        "tags": ["openvoice", "onnx", "rlx"],
    },
    "tts/parler-dac": {
        "title": "Descript DAC 44 kHz (RLX staging)",
        "kind": "redistrib",
        "license": "mit",
        "pipeline": "audio-to-audio",
        "upstream": "https://huggingface.co/descript/dac_44khz",
        "crate": "rlx-parlertts",
        "run": "cargo run -p rlx-parlertts --release -- --dac-dir .",
        "summary": "DAC 44 kHz vocoder weights paired with Parler-TTS on RLX.",
        "tags": ["dac", "codec", "rlx"],
    },
    "tts/parlertts": {
        "title": "Parler-TTS Mini v1 (RLX staging)",
        "kind": "converted",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/parler-tts/parler-tts-mini-v1",
        "crate": "rlx-parlertts",
        "run": "cargo run -p rlx-parlertts --release -- --model-dir .",
        "summary": "Parler-TTS Mini safetensors + ONNX exports for RLX (probe ONNX removed).",
        "tags": ["parler-tts", "tts", "rlx"],
    },
    "tts/piper": {
        "title": "Piper en_US-lessac-medium (RLX staging)",
        "kind": "converted",
        "license": "mit",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/rhasspy/piper-voices",
        "crate": "rlx-piper",
        "run": "cargo run -p rlx-piper --release -- --dir .",
        "summary": "Piper ONNX voice + RLX `rlx-split/` subgraphs.",
        "tags": ["piper", "tts", "onnx", "rlx"],
    },
    "tts/sesame": {
        "title": "Sesame CSM-1B (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/unsloth/csm-1b",
        "crate": "rlx-sesame",
        "run": "cargo run -p rlx-sesame --release -- --model-dir .",
        "summary": "CSM-1B conversational TTS weights (ungated mirror) for RLX.",
        "tags": ["sesame", "csm", "tts", "rlx"],
    },
    "tts/soprano": {
        "title": "Soprano 1.1 (RLX)",
        "kind": "rlx-native",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/KevinAHM/soprano-1.1-onnx",
        "crate": "rlx-soprano",
        "run": 'just fetch-soprano && just soprano-demo',
        "summary": "Single runnable `soprano.rlxp`: nested native backbone + Vocos packs (no ONNX on Hub).",
        "tags": ["soprano", "rlxp", "tts", "rlx"],
        "primary": ["soprano.rlxp"],
        "avoid": [],
        "fetch": "just fetch-soprano   # or: hf download eugenehp/soprano soprano.rlxp --local-dir weights/tts/soprano",
        "files_note": "Hub ships `soprano.rlxp` only (nested `graphs/*.rlxp` + tokenizer). Pack locally with `just export-soprano-rlxp`. CPU / Metal / MLX / CUDA / wgpu.",
        "notes": [
            "Hub ships `.rlxp` only — **no ONNX**. Nested packs hold hot tensors + graph.json; runtime lowers per KV/seq bucket."
        ],
        "rlxp": {
            "file": "soprano.rlxp",
            "intro": (
                "Outer RLXPFLAT with nested native subgraph packs for the Qwen3-style KV backbone "
                "and Vocos decoder. **No `.onnx` on Hub.**"
            ),
            "architecture": [
                "**Pipeline:** text → tokenizer → AR backbone (KV cache) → Vocos decoder → 32 kHz mono.",
                "",
                "| Module | Role | Dims | dtype |",
                "|---|---|---|---|",
                "| `soprano_backbone_kv_fp32` | 17-layer AR LM + KV | hidden 512, head_dim 128, vocab 8192 | f32 |",
                "| `soprano_decoder_fp32` | Vocos vocoder | TOKEN_SIZE 2048 | f32 |",
                "| `tokenizer.json` | text tokenizer | — | — |",
            ],
            "sidecars_note": "Neural weights live inside nested `graphs/*.rlxp` (hot mmap).",
            "roles": {
                "tokenizer.json": "Text tokenizer",
                "graphs/soprano_backbone_kv_fp32.rlxp": "Nested backbone pack",
                "graphs/soprano_decoder_fp32.rlxp": "Nested Vocos pack",
            },
            "tree": [
                "soprano.rlxp",
                "├── graphs/",
                "│   ├── soprano_backbone_kv_fp32.rlxp",
                "│   └── soprano_decoder_fp32.rlxp",
                "└── tokenizer.json (+ HF tokenizer sidecars)",
            ],
            "pack": (
                "`just export-soprano-rlxp` — pack-time ONNX → nested `graphs/*.rlxp`. Hub has zero ONNX."
            ),
        },
        "repo_name": "soprano",
        "write_license_file": False,
        "keep_existing_readme": False,
    },
    "tts/supertonic-3": {
        "title": "Supertonic-3 (RLX staging)",
        "kind": "redistrib",
        "license": "openrail",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/Supertone/supertonic-3",
        "crate": "rlx-supertonic",
        "run": "cargo run -p rlx-supertonic --release -- --data .",
        "summary": "Supertonic-3 ONNX + voice styles for RLX. OpenRAIL — see upstream.",
        "tags": ["supertonic", "tts", "rlx"],
        "write_license_file": False,
    },
    "tts/zipvoice": {
        "title": "ZipVoice ONNX (RLX staging)",
        "kind": "converted",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/k2-fsa/ZipVoice",
        "crate": "rlx-zipvoice",
        "run": "cargo run -p rlx-zipvoice --release -- --model-dir .",
        "summary": "ZipVoice ONNX text encoder + flow decoder for RLX.",
        "tags": ["zipvoice", "onnx", "tts", "rlx"],
    },
    "tts/zipvoice-distill": {
        "title": "ZipVoice-Distill ONNX (RLX staging)",
        "kind": "converted",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/k2-fsa/ZipVoice",
        "crate": "rlx-zipvoice",
        "run": "cargo run -p rlx-zipvoice --release -- --model-dir .",
        "summary": "ZipVoice distill ONNX graphs for RLX.",
        "tags": ["zipvoice", "onnx", "tts", "rlx"],
    },
    "tts/zonos": {
        "title": "Zonos-v0.1 transformer (RLX staging)",
        "kind": "redistrib",
        "license": "apache-2.0",
        "pipeline": "text-to-speech",
        "upstream": "https://huggingface.co/Zyphra/Zonos-v0.1-transformer",
        "crate": "rlx-zonos",
        "run": "cargo run -p rlx-zonos --release -- --model-dir .",
        "summary": "Zyphra Zonos-v0.1 transformer weights for RLX.",
        "tags": ["zonos", "tts", "rlx"],
        "keep_existing_readme": False,
    },
}


# Prefer these names when auto-picking a primary runtime file.
_PRIMARY_CANDIDATES = (
    "rlx-tts.rlxp",
    "rlx-tts.gguf",
    "moss-nano.rlxp",
    "moss-nano.gguf",
    "soprano.rlxp",
    "soprano.gguf",
    "model.rlxp",
    "model.gguf",
    "tiny-tts.rlxp",
    "tiny-tts.rlxpack",
    "snac_24khz_decoder.safetensors",
)

# Never advertise these as the main download (wrong format / codec-only / LM-only).
_AVOID_DEFAULT = (
    "codec-q4_k_m.gguf",
    "Soprano-1.1-80M.Q4_K_M.gguf",
    "*.f16.gguf",
)


def _fmt_size(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{int(x)} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} TiB"


def discover_files(path: Path) -> dict:
    """Scan a model dir for card sections (primary / listing / avoid)."""
    if not path.is_dir() or path.is_symlink():
        return {"primary": [], "listing": [], "avoid": []}
    listing: list[tuple[str, int]] = []
    avoid: list[str] = []
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(path).as_posix()
        if any(part.startswith(".") for part in p.relative_to(path).parts):
            continue
        if rel in {"README.md", "LICENSE", ".gitattributes"}:
            continue
        if "fixtures" in p.parts or "__pycache__" in p.parts:
            continue
        sz = p.stat().st_size
        listing.append((rel, sz))
        name = p.name
        if name.endswith(".f16.gguf") or name in {
            "codec-q4_k_m.gguf",
            "Soprano-1.1-80M.Q4_K_M.gguf",
        }:
            avoid.append(rel)
    primary: list[str] = []
    names = {n for n, _ in listing}
    for cand in _PRIMARY_CANDIDATES:
        if cand in names:
            primary.append(cand)
    # Prefer RLX runnable packs / .rlxp over community Q4 when both exist.
    for n, _ in listing:
        base = Path(n).name
        if base.endswith((".rlxp", ".rlxpack")) and n not in primary:
            primary.append(n)
        if base in {"moss-nano.gguf", "soprano.gguf", "rlx-tts.gguf", "model.gguf"} and n not in primary:
            primary.append(n)
    # Cap listing for huge trees — show top-level + important suffixes.
    important_ext = {".gguf", ".rlxp", ".rlxpack", ".safetensors", ".onnx", ".json", ".data"}
    top = []
    for n, sz in listing:
        depth = n.count("/")
        if depth == 0 or Path(n).suffix.lower() in important_ext:
            top.append((n, sz))
    top.sort(key=lambda t: (-t[1], t[0]))
    top = top[:24]
    return {"primary": primary, "listing": top, "avoid": sorted(set(avoid))}


def _parse_rlxp_toc(path: Path) -> dict | None:
    """Parse RLXPFLAT JSON TOC (header + length-prefixed JSON)."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 24 or data[:8] != b"RLXPFLAT":
        return None
    toc_len = struct.unpack_from("<Q", data, 16)[0]
    toc_bytes = data[24 : 24 + toc_len]
    try:
        return json.loads(toc_bytes)
    except json.JSONDecodeError:
        # Fallback: brace-scan (older / odd writers).
        j0 = data.find(b'{"manifest"', 0, min(len(data), 1 << 20))
        if j0 < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        end = None
        for i in range(j0, len(data)):
            c = data[i]
            if in_str:
                if esc:
                    esc = False
                elif c == ord("\\"):
                    esc = True
                elif c == ord('"'):
                    in_str = False
                continue
            if c == ord('"'):
                in_str = True
            elif c == ord("{"):
                depth += 1
            elif c == ord("}"):
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return None
        try:
            return json.loads(data[j0:end])
        except json.JSONDecodeError:
            return None


def _tensor_group(name: str) -> str:
    if name.startswith(("encoder.", "decoder.", "wavernn.")):
        return name.split(".", 1)[0]
    if name.startswith(("ls.", "codebook.", "tp.", "silence_")):
        return name.split(".", 1)[0] if "." in name else name
    parts = name.split(".")
    return parts[0] if parts else name


def _sidecar_role(sidecar_id: str, roles: dict[str, str]) -> str:
    if sidecar_id in roles:
        return roles[sidecar_id]
    best_key = ""
    best_role = ""
    for key, role in roles.items():
        if sidecar_id.startswith(key) and len(key) > len(best_key):
            best_key = key
            best_role = role
    return best_role


def render_rlxp_section(meta: dict, path: Path) -> list[str]:
    """Build markdown outlining `.rlxp` container + inner assets/tensors."""
    guide = meta.get("rlxp")
    if not guide:
        return []
    pack_name = guide.get("file")
    if not pack_name:
        for p in meta.get("primary") or []:
            if str(p).endswith(".rlxp"):
                pack_name = p
                break
    if not pack_name:
        return []
    pack_path = path / pack_name
    toc = _parse_rlxp_toc(pack_path) if pack_path.is_file() else None

    lines = ["## Pack layout (`.rlxp`)", ""]
    if guide.get("intro"):
        lines += [guide["intro"], ""]
    lines += RLXP_CONTAINER_LINES + [""]

    if toc is None:
        # Static fallback when the pack file is missing locally.
        if guide.get("tree"):
            lines += ["### Logical tree", "", "```text"]
            lines += list(guide["tree"])
            lines += ["```", ""]
        if guide.get("pack"):
            lines += ["### How it is packed", "", guide["pack"], ""]
        return lines

    man = toc.get("manifest") or {}
    tensors = toc.get("tensors") or []
    sidecars = toc.get("sidecars") or []
    strings = toc.get("strings") or []
    version = struct.unpack_from("<I", pack_path.read_bytes(), 8)[0]
    flags = struct.unpack_from("<I", pack_path.read_bytes(), 12)[0]

    lines += [
        "### This pack",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| **File** | `{pack_name}` ({_fmt_size(pack_path.stat().st_size)}) |",
        f"| **Manifest name** | `{man.get('name', '')}` |",
        f"| **Producer** | `{man.get('producer', '')}` |",
        f"| **Container** | RLXPFLAT v{version}, flags=0x{flags:x} |",
        f"| **Tensors** | {len(tensors)} |",
        f"| **Sidecars** | {len(sidecars)} |",
        "",
    ]

    roles: dict[str, str] = dict(guide.get("roles") or {})
    skip_sidecars = set(guide.get("skip_sidecars") or [])

    if tensors:
        lines += ["### Tensors (hot weight region)", ""]
        if guide.get("tensors_note"):
            lines += [guide["tensors_note"], ""]
        group_n: Counter[str] = Counter()
        group_bytes: Counter[str] = Counter()
        schemes: Counter[str] = Counter()
        for t in tensors:
            if "name_i" in t and strings:
                name = strings[int(t["name_i"])]
            else:
                name = t.get("name") or "?"
            g = _tensor_group(name)
            group_n[g] += 1
            group_bytes[g] += int(t.get("length") or 0)
            schemes[str(t.get("scheme") or "?")] += 1
        lines += [
            f"All tensors are mmap'd from the hot region "
            f"(schemes: {', '.join(f'`{k}`×{v}' for k, v in schemes.most_common())}).",
            "",
            "| Prefix | Tensors | Stored | Role |",
            "|---|---:|---:|---|",
        ]
        module_roles = dict(guide.get("modules") or {})
        for g, n in group_n.most_common():
            role = module_roles.get(g, module_roles.get(g + ".", ""))
            lines.append(f"| `{g}.*` | {n} | {_fmt_size(group_bytes[g])} | {role} |")
        lines.append("")

    if sidecars:
        lines += ["### Sidecars (file assets)", ""]
        if guide.get("sidecars_note"):
            lines += [guide["sidecars_note"], ""]
        lines += [
            "Paths below are logical ids inside the pack (`__flat__/sidecar/<id>`).",
            "Cold sidecars are zstd-compressed; sizes show raw → stored.",
            "",
            "| Sidecar | Raw | Stored | Role |",
            "|---|---:|---:|---|",
        ]
        shown = 0
        for s in sidecars:
            sid = s.get("id") or "?"
            if sid in skip_sidecars:
                continue
            raw = int(s.get("raw_length") or s.get("length") or 0)
            stored = int(s.get("length") or 0)
            role = _sidecar_role(sid, roles)
            lines.append(
                f"| `{sid}` | {_fmt_size(raw)} | {_fmt_size(stored)} | {role} |"
            )
            shown += 1
        if shown == 0:
            lines.append("| *(none listed)* | | | |")
        lines.append("")

    if guide.get("architecture"):
        lines += ["### Architecture", ""]
        lines.extend(guide["architecture"])
        lines.append("")

    if guide.get("tree"):
        lines += ["### Logical tree", "", "```text"]
        lines += list(guide["tree"])
        lines += ["```", ""]

    if guide.get("pack"):
        lines += ["### How it is packed", "", guide["pack"], ""]

    return lines


def render_readme(rel: str, meta: dict, path: Path | None = None) -> str:
    repo = meta.get("repo_name") or Path(rel).name
    hub = f"{ORG}/{repo}"
    license_id = meta["license"]
    tags = meta.get("tags") or []
    tag_yaml = "\n".join(f"- {t}" for t in tags)
    upstream = meta.get("upstream")
    kind = meta["kind"]
    kind_blurb = {
        "rlx-native": "RLX-native weight bundle (graphs + sidecars ready for `rlx-*` crates).",
        "converted": "Converted / re-laid-out for RLX from an upstream checkpoint.",
        "redistrib": "Staging redistrib of an upstream checkpoint for RLX runners.",
    }[kind]

    path = path or (WEIGHTS / rel)
    disc = discover_files(path) if path.is_dir() and not path.is_symlink() else {
        "primary": [],
        "listing": [],
        "avoid": [],
    }
    primary = list(meta.get("primary") or disc["primary"])
    avoid = list(meta.get("avoid") or disc["avoid"])
    fetch = meta.get("fetch") or f"hf download {hub} --local-dir ."
    notes = meta.get("notes") or []

    lines = [
        "---",
        f"license: {license_id}",
        f"pipeline_tag: {meta['pipeline']}",
        "library_name: rlx",
        "tags:",
        tag_yaml,
        "---",
        "",
        f"# {meta['title']}",
        "",
        meta["summary"],
        "",
        "| Field | Value |",
        "|---|---|",
        f"| **Hub id** | [`{hub}`](https://huggingface.co/{hub}) |",
        f"| **Kind** | {kind_blurb} |",
        f"| **RLX crate** | [`{meta['crate']}`](https://github.com/MIT-RLX/rlx-models/tree/main/crates/{meta['crate']}) |",
    ]
    if upstream:
        lines.append(f"| **Upstream** | {upstream} |")
    lines += ["", "## Quick start", "", "```bash", fetch, meta["run"], "```", ""]

    if primary:
        lines += ["## Primary files (use these)", ""]
        for p in primary:
            size = ""
            fp = path / p
            if fp.is_file():
                size = f" — {_fmt_size(fp.stat().st_size)}"
            lines.append(f"- `{p}`{size}")
        lines.append("")
    if meta.get("files_note"):
        lines += ["## Contents", "", meta["files_note"], ""]
    elif disc["listing"]:
        lines += ["## File highlights", ""]
        for n, sz in disc["listing"][:16]:
            mark = " ← primary" if n in primary or Path(n).name in primary else ""
            lines.append(f"- `{n}` ({_fmt_size(sz)}){mark}")
        lines.append("")

    lines += render_rlxp_section(meta, path)

    if avoid:
        lines += [
            "## Do not use as the main runtime pack",
            "",
            "These files may be present for historical / staging reasons but are **not** the "
            "supported RLX load path:",
            "",
        ]
        for a in avoid:
            lines.append(f"- `{a}`")
        lines += [
            "",
            "Community `*.f16.gguf` / LM-only Q4 packs are format wraps — they are not drop-in "
            "replacements for the RLX primary file above.",
            "",
        ]

    for note in notes:
        lines += ["## Note", "", note, ""]

    lines += [
        "## Run with RLX",
        "",
        "Clone [rlx-models](https://github.com/MIT-RLX/rlx-models), place this repo under "
        f"`weights/{rel}` (or pass the path explicitly), then:",
        "",
        "```bash",
        meta["run"],
        "```",
        "",
        "## License",
        "",
    ]
    if license_id == "apache-2.0":
        lines.append("Apache License 2.0 — see `LICENSE`. Inherit upstream terms when redistributing.")
    elif license_id == "mit":
        lines.append("MIT — see `LICENSE`. Inherit upstream terms when redistributing.")
    elif license_id == "cc-by-nc-nd-4.0":
        lines.append(
            "CC-BY-NC-ND-4.0 (non-commercial, no derivatives). Do not re-upload without upstream permission; gated on the original Hub card."
        )
    elif license_id == "cc-by-nc-sa-4.0":
        lines.append("CC-BY-NC-SA-4.0 — see upstream license; non-commercial share-alike.")
    elif license_id == "openrail":
        lines.append("OpenRAIL — see upstream model card for use restrictions.")
    else:
        lines.append(f"License `{license_id}` — see upstream.")
    if upstream:
        lines += ["", f"Original weights and authorship: {upstream}"]
    if kind == "redistrib":
        lines += [
            "",
            "## Redistrib note",
            "",
            "This Hub repo exists so RLX recipes have a stable fetch target. "
            "When you only need the upstream checkpoint, prefer the Upstream link above.",
            "",
        ]
    else:
        lines += [
            "",
            "## Maintenance",
            "",
            "Cards and LFS attrs are regenerated from the local `weights/` tree in "
            "[rlx-models](https://github.com/MIT-RLX/rlx-models) via "
            "`python3 scripts/prepare_weights_hf.py`.",
            "",
        ]
    return "\n".join(lines)


def ensure_license(path: Path, meta: dict) -> None:
    if meta.get("write_license_file", True) is False:
        return
    lic = meta["license"]
    dest = path / "LICENSE"
    if dest.exists():
        return
    if lic == "apache-2.0":
        dest.write_text(APACHE2)
    elif lic == "mit":
        dest.write_text(MIT)
    # other licenses: card-only


def prepare_one(rel: str, meta: dict) -> None:
    path = WEIGHTS / rel
    # Alias / card-only entries (e.g. melotts symlink) still get a README next to
    # the symlink parent if we materialize a tiny card dir — skip local write when
    # the path is only a symlink.
    if path.is_symlink():
        print(f"skip symlink {rel} (Hub card via publish --cards-only)")
        return
    if not path.is_dir():
        print(f"skip missing {rel}")
        return
    (path / ".gitattributes").write_text(GITATTRIBUTES)
    ensure_license(path, meta)
    (path / "README.md").write_text(render_readme(rel, meta, path))
    print(f"ok {rel} -> {ORG}/{meta.get('repo_name') or Path(rel).name}")


def cleanup_weights_tree() -> None:
    """Remove local junk that should never ship to Hub."""
    removed = 0
    for pat in (".DS_Store",):
        for p in WEIGHTS.rglob(pat):
            p.unlink(missing_ok=True)
            removed += 1
    for p in WEIGHTS.rglob(".cache"):
        if p.is_dir():
            import shutil

            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    # Misleading community packs superseded by RLX runnable GGUFs.
    for rel in (
        "tts/moss-nano/codec-q4_k_m.gguf",
        "tts/soprano/Soprano-1.1-80M.Q4_K_M.gguf",
    ):
        p = WEIGHTS / rel
        if p.is_file():
            p.unlink()
            removed += 1
            print(f"removed misleading {rel}")
    # Stale status file from the one-shot F16 wrap experiment.
    stale = WEIGHTS / "GGUF_STATUS.json"
    if stale.is_file():
        stale.unlink()
        removed += 1
    # Placeholder manifests from fetch-tts-validation-bundles.
    for p in WEIGHTS.rglob("manifest.json"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "placeholder" in text and "use real HF weights" in text:
            p.unlink(missing_ok=True)
            removed += 1
    print(f"cleanup: touched {removed} paths")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleanup", action="store_true", help="remove junk before writing cards")
    ap.add_argument("--only", type=str, default="", help="comma-separated rel paths or leaf names")
    args = ap.parse_args()
    if args.cleanup:
        cleanup_weights_tree()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    for rel, meta in sorted(MODELS.items()):
        leaf = Path(rel).name
        repo = meta.get("repo_name") or leaf
        if only and rel not in only and leaf not in only and repo not in only:
            continue
        prepare_one(rel, meta)
    # index
    rows = []
    for rel, meta in sorted(MODELS.items()):
        repo = meta.get("repo_name") or Path(rel).name
        rows.append(
            f"| `{rel}` | `{ORG}/{repo}` | {meta['kind']} | `{meta['license']}` | {meta.get('upstream') or '—'} |"
        )
    index = f"""# RLX weights staging

Local staging tree for **separate** Hugging Face model repos (one directory ≈ one Hub repo).
Gitignored — never commit blobs into `rlx-models`.

```bash
# Clean junk + regenerate every model card / LICENSE / .gitattributes
python3 scripts/prepare_weights_hf.py --cleanup

# Push README cards only (fast)
python3 scripts/publish_weights_hf.py --cards-only

# Push full trees for selected repos
python3 scripts/publish_weights_hf.py --only moss-nano,soprano,tiny-tts-rlx,rlx-tts,rlx-asr
```

## Layout

```text
weights/
  vision/   lm/   tts/   asr/   audio/
```

Compat symlinks at the old top-level paths still point here for local `just` recipes.
`weights/tts/melotts` → `tiny-tts-rlx` (alias; Hub card redirects).

## Publish map

| Local path | Suggested Hub id | Kind | License | Upstream |
|------------|------------------|------|---------|----------|
{chr(10).join(rows)}

## Primary RLX-native packs

| Hub | Primary file |
|-----|----------------|
| `eugenehp/rlx-tts` | `rlx-tts.rlxp` |
| `eugenehp/rlx-asr` | `model.rlxp` |
| `eugenehp/moss-nano` | `moss-nano.rlxp` |
| `eugenehp/soprano` | `soprano.rlxp` |
| `eugenehp/tiny-tts-rlx` | `tiny-tts.rlxp` |
"""
    (WEIGHTS / "README.md").write_text(index)
    print("ok weights/README.md")


if __name__ == "__main__":
    main()
