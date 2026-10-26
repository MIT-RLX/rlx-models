// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! The two modules of `rlx-models-core` the decider's `protocol.rs` and
//! `vision.rs` read: `ordered_json` and `image_preprocess`.

#[allow(dead_code)]
#[path = "../../../../rlx-models-core/src/image_preprocess.rs"]
pub mod image_preprocess;
#[path = "../../../../rlx-models-core/src/ordered_json.rs"]
pub mod ordered_json;
