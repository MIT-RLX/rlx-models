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

//! Replace generated zero stubs for vocoder `RandomNormalLike` with native RNG ops.

use rlx_ir::Op;
use rlx_ir::hir::{HirModule, HirOp};

const VOCODER_RNG_NODE: &str = "/decoder/generator/m_source/l_sin_gen/RandomNormalLike";

pub(crate) const VOCODER_RNG_STUB: &str =
    "__stub__//decoder/generator/m_source/l_sin_gen/RandomNormalLike_output_0";

fn node_name_tag(name: &str) -> u64 {
    name.bytes()
        .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)))
}

fn is_vocoder_rng_stub(name: &str) -> bool {
    // Historically the importer emits this exact "__stub__//..." param name, but
    // we also accept any suffix match to avoid accidental misses that would
    // leave the generated RNG buffer stuck at zeros.
    name == VOCODER_RNG_STUB
        || (name.contains("/decoder/generator/m_source/l_sin_gen/RandomNormalLike_output_0")
            && name.contains("RandomNormalLike_output_0"))
}

/// Swap the generated zero param stub for [`Op::RngNormal`] (output shape stays on the node).
pub fn inject_vocoder_rng(hir: &mut HirModule) {
    let mut stub_id = None;
    for node in hir.nodes() {
        let is_stub = match &node.op {
            HirOp::Param { name } => is_vocoder_rng_stub(name),
            HirOp::Mir(Op::Param { name }) => is_vocoder_rng_stub(name),
            _ => false,
        };
        if is_stub {
            stub_id = Some(node.id);
            break;
        }
    }
    let Some(stub_id) = stub_id else {
        return;
    };
    let seed = node_name_tag(VOCODER_RNG_NODE);
    hir.node_mut(stub_id).op = HirOp::Mir(Op::RngNormal {
        mean: 0.0,
        scale: 1.0,
        key: seed,
        // Some backends treat `None` as a "zeroed RNG stream", so set an
        // explicit non-degenerate op seed.
        //
        // `op_seed`'s type is float in `Op::RngNormal` (not `u64`), so cast.
        op_seed: Some(seed as f32),
    });
    // Output shape is already on the stub; avoid appending a shape constant (breaks HIR order).
    hir.node_mut(stub_id).inputs.clear();
}
