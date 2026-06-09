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

//! Processor prompt image-slot count on the bundled sample.

use rlx_locateanything::fixtures::{require_model_dir, require_probe_image};
use rlx_locateanything::{
    LocateAnythingConfig, preprocess::preprocess_path,
    processor_prompt::ground_single_with_image_placeholder, tokenizer::load_tokenizer,
};

#[test]
fn processor_prompt_includes_image_slots() {
    let Some(dir) = require_model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    let Some(img) = require_probe_image() else {
        eprintln!("skip: missing fixtures/sample.jpg");
        return;
    };

    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    let prep = preprocess_path(&img, &cfg).expect("preprocess");
    let kh = cfg.vision_config.merge_kernel_size[0];
    let kw = cfg.vision_config.merge_kernel_size[1];
    let n_image = (prep.grid_h / kh) * (prep.grid_w / kw);

    let Ok(tok) = load_tokenizer(&dir) else {
        eprintln!("skip: tokenizer files missing in {dir:?}");
        return;
    };
    let user = ground_single_with_image_placeholder("person");
    let ids = rlx_locateanything::processor_prompt::build_processor_prompt_ids(
        &dir, &cfg, &tok, &user, n_image,
    )
    .expect("ids");
    let slots = ids.iter().filter(|&&t| t == cfg.image_token_index).count();
    assert_eq!(slots, n_image, "prompt image slots");
}
