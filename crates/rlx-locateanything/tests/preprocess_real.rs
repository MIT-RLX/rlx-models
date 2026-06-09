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

//! Native-resolution preprocess on the bundled `fixtures/sample.jpg`.

use rlx_locateanything::fixtures::{require_model_dir, require_probe_image};
use rlx_locateanything::{LocateAnythingConfig, preprocess::preprocess_path};

#[test]
fn preprocess_real_fixture_native_resolution() {
    let Some(dir) = require_model_dir() else {
        eprintln!("skip: set RLX_LOCATEANYTHING_DIR");
        return;
    };
    let Some(path) = require_probe_image() else {
        eprintln!("skip: missing bundled sample (fixtures/sample.jpg)");
        return;
    };
    let cfg = LocateAnythingConfig::from_file(&dir.join("config.json")).expect("config");
    assert_eq!(cfg.preprocessor.in_token_limit, 25_600);
    let prep = preprocess_path(&path, &cfg).expect("preprocess");
    eprintln!(
        "fixture {} -> grid {}x{} patches={}",
        path.display(),
        prep.grid_h,
        prep.grid_w,
        prep.num_patches()
    );
    assert!(prep.grid_h > 4 && prep.grid_w > 4);
}
