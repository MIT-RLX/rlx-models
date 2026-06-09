// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Verify mmap embedding row load matches full-table fuse (env-gated).

use rlx_locateanything::config::LocateAnythingConfig;
use rlx_locateanything::embed::{fuse_inputs_embeds, fuse_inputs_embeds_from_store};
use rlx_locateanything::fixtures::require_model_dir;
use rlx_locateanything::load::LocateAnythingWeightStore;
use rlx_locateanything::weights::LocateAnythingWeightPrefix;

#[test]
fn embed_mmap_rows_match_full_table() {
    let Some(dir) = require_model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR or run just fetch-locateanything");
        return;
    };
    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let Ok(store) = LocateAnythingWeightStore::open(&dir) else {
        eprintln!("skip: no safetensors weights in {dir:?}");
        return;
    };
    let token_ids: Vec<u32> = vec![1, 2, 3, cfg.image_token_index, 10, 11];
    let h = cfg.text_config.hidden_size;
    let vision = vec![0.1f32; h];

    let wm = store
        .load_keys(&[LocateAnythingWeightPrefix::lm_embed_tokens()])
        .expect("embed table");
    let full = fuse_inputs_embeds(&cfg, &wm, &token_ids, &vision).expect("full fuse");
    let mmap = fuse_inputs_embeds_from_store(&cfg, &store, &token_ids, &vision).expect("mmap fuse");
    assert_eq!(full.len(), mmap.len());
    for (a, b) in full.iter().zip(mmap.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 1e-5, "embed row mismatch: {diff}");
    }
}
