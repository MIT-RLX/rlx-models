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

//! The released RIMES checkpoint loads with its charset and the fixed
//! architecture's shapes, when present (`just fetch-dan rimes`).

use rlx_dan::{DanCheckpoint, graph, model};
use std::path::Path;

#[test]
fn rimes_checkpoint_contract() {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/ocr/dan-rimes/dan_rimes_page.pt");
    if !p.exists() {
        eprintln!("skipping: {} not present", p.display());
        return;
    }
    let ck = DanCheckpoint::open(&p).unwrap();
    assert_eq!(ck.vocab(), 122);
    assert_eq!(ck.charset[0], "\n");
    assert!(ck.charset.iter().any(|c| c == "é"));
    // Every tensor both graphs read is present with its expected shape.
    graph::build_encoder(&ck.encoder, &ck.decoder, 64, 64).unwrap();
    graph::build_step(&ck.decoder, 4 * 8, ck.vocab()).unwrap();
    assert_eq!(graph::feature_size(64, 64), [2, 8]);
    assert_eq!(model::WINDOW, 100);
}
