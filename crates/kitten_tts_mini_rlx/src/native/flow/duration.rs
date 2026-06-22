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

//! Break the duration feedback cycle for native weights graphs (ORT stale carry).

use rlx_ir::hir::{HirModule, HirMut, HirNodeId, HirOp};
use rlx_ir::{DType, Op, Shape};

use crate::opts::DURATION_CARRY;

/// Rewire `/Expand_1` and `/Where_1` to read [`DURATION_CARRY`] instead of the
/// freshly computed `duration` tensor (same rewrite as bundle export).
pub fn inject_duration_carry(hir: &mut HirModule, sequence_length: usize) -> Vec<u8> {
    let Some(duration_id) = hir.outputs.get(1).copied() else {
        return crate::compile_profile::duration_carry_seed_bytes(sequence_length);
    };

    let carry_shape = Shape::new(&[sequence_length], DType::I64);
    let carry_bytes = crate::compile_profile::duration_carry_seed_bytes(sequence_length);
    let mut m = HirMut::new(hir);
    let carry_id = m.param(DURATION_CARRY, carry_shape);
    drop(m);

    for idx in 0..hir.len() {
        let id = HirNodeId(idx as u32);
        let node = hir.node(id);
        let duration_input = node.inputs.first() == Some(&duration_id);
        let duration_where = node.inputs.get(2) == Some(&duration_id);
        let is_expand = matches!(node.op, HirOp::Mir(Op::Expand { .. }));
        let is_where = matches!(node.op, HirOp::Mir(Op::Where));
        if is_expand && duration_input {
            hir.node_mut(id).inputs[0] = carry_id;
        } else if is_where && duration_where {
            hir.node_mut(id).inputs[2] = carry_id;
        }
    }

    carry_bytes
}
