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

//! The CATMuS Medieval release decodes to the documented network, when it
//! is present (`just fetch-kraken-catmus`).

use rlx_kraken::{KrakenRunner, graph};
use rlx_runtime::Device;
use std::path::Path;

#[test]
fn catmus_model_contract() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../weights/ocr/kraken-catmus-medieval/catmus-medieval.mlmodel");
    if !p.exists() {
        eprintln!("skipping: {} not present", p.display());
        return;
    }
    let r = KrakenRunner::open(&p, Device::Cpu).unwrap();
    let spec = r.spec();
    assert_eq!((spec.input.height, spec.input.channels), (120, 1));
    assert_eq!(spec.blocks.len(), 19);
    assert_eq!(r.padding(), 16);
    // Three 2×2 pools: 1/8 of the width, the height folded into 960 features.
    assert_eq!(
        graph::output_shape(spec, [1, 1, 120, 1030]).unwrap(),
        [1, 225, 1, 128]
    );
}
