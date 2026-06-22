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

//! Dump memory-plan slots for `input_ids` and overlapping buffers.

use rlx_compile::memory::{MemoryPlan, plan_memory_aligned};
use rlx_ir::DType;
use rlx_onnx_import::{ImportOptions, build_hir_from_bundle, load_bundle};
use rlx_runtime::{CompileOptions, Device, Session, stages};

fn exec_graph(g: rlx_ir::Graph) -> rlx_ir::Graph {
    let needs = g
        .nodes()
        .iter()
        .any(|n| matches!(n.shape.dtype(), DType::F16 | DType::BF16));
    if !needs {
        return g;
    }
    let mut out = rlx_ir::Graph::new(format!("{}_f32_exec", g.name));
    let mut id_map = std::collections::HashMap::new();
    for node in g.nodes() {
        let inputs: Vec<_> = node.inputs.iter().map(|i| id_map[i]).collect();
        let mut shape = node.shape.clone();
        if matches!(shape.dtype(), DType::F16 | DType::BF16) {
            shape = shape.with_dtype(DType::F32);
        }
        let new_id = out.add_node(node.op.clone(), inputs, shape);
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|o| id_map[o]).collect());
    out
}

fn report_overlaps(label: &str, exec: &rlx_ir::Graph, plan: &MemoryPlan) {
    let input_ids = exec
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == "input_ids"))
        .expect("input_ids");
    eprintln!(
        "{label} input_ids: id={} dtype={:?} shape={:?} slot={:?}",
        input_ids.id,
        input_ids.shape.dtype(),
        input_ids.shape.dims(),
        plan.assignments.get(&input_ids.id)
    );
    if let Some(islot) = plan.assignments.get(&input_ids.id) {
        eprintln!("nodes at input_ids offset {}:", islot.offset);
        for (nid, slot) in &plan.assignments {
            if slot.offset == islot.offset {
                let n = exec.node(*nid);
                eprintln!(
                    "  id={nid} size={} op={:?} name={:?}",
                    slot.size, n.op, n.name
                );
            }
        }
        eprintln!("byte-range overlaps (partial):");
        let i_end = islot.offset + islot.size;
        for (nid, slot) in &plan.assignments {
            if nid == &input_ids.id {
                continue;
            }
            let p_end = slot.offset + slot.size;
            if islot.offset < p_end && i_end > slot.offset {
                let n = exec.node(*nid);
                eprintln!(
                    "  id={nid} [{}, {}) size={} op={:?} name={:?}",
                    slot.offset, p_end, slot.size, n.op, n.name
                );
            }
        }
        eprintln!("slots within 4KiB of input_ids:");
        for (nid, slot) in &plan.assignments {
            let p_end = slot.offset + slot.size;
            if p_end >= islot.offset.saturating_sub(4096)
                && slot.offset <= i_end.saturating_add(4096)
            {
                let n = exec.node(*nid);
                eprintln!(
                    "  id={nid} [{}, {}) size={} op={:?} name={:?}",
                    slot.offset, p_end, slot.size, n.op, n.name
                );
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let bundle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let seq = 8usize;
    kitten_tts_mini_rlx::opts::set_compile_sequence_length(seq);
    kitten_tts_mini_rlx::kernels::register_native_kernels();
    let mut bundle = load_bundle(&bundle_dir)?;
    kitten_tts_mini_rlx::bundle_patches::patch_bundle_nodes(&mut bundle.nodes, seq, 12_000);
    let opts = ImportOptions {
        sequence_length: seq,
        max_waveform_samples: 12_000,
        ..ImportOptions::quant_bundle()
    };
    let (hir, _, _, _) = build_hir_from_bundle(&bundle, opts.clone())?;
    let mut compile_opts = CompileOptions::default();
    compile_opts.fusion_opts.skip_fusion = true;

    let result = stages::compile_hir_stages(Device::Cpu, hir.clone(), &compile_opts)?;
    let g = stages::graph_from_lir(result.lir);
    let exec = exec_graph(g);
    let plan = plan_memory_aligned(&exec, 64);
    report_overlaps("LIR", &exec, &plan);

    let input_ids = exec
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == "input_ids"))
        .expect("input_ids")
        .id;

    let mut by_off: std::collections::HashMap<usize, Vec<rlx_ir::NodeId>> =
        std::collections::HashMap::new();
    for (nid, slot) in &plan.assignments {
        by_off.entry(slot.offset).or_default().push(*nid);
    }
    for (off, ids) in by_off {
        if ids.len() > 1 {
            eprintln!("SHARED OFFSET {off}:");
            for id in ids {
                let n = exec.node(id);
                eprintln!("  id={id} op={:?} name={:?}", n.op, n.name);
            }
        }
    }

    let iid = input_ids;
    eprintln!("=== consumers of input_ids %0 ===");
    for node in exec.nodes() {
        if node.inputs.contains(&iid) {
            eprintln!(
                "  user id={} op={:?} name={:?} dtype={:?}",
                node.id,
                node.op,
                node.name,
                node.shape.dtype()
            );
        }
    }
    for node in exec.nodes() {
        if matches!(&node.op, rlx_ir::Op::Gather { .. }) {
            let idx = node.inputs.get(1).copied();
            let idx_node = idx.map(|id| exec.node(id));
            if idx == Some(iid) || idx_node.is_some_and(|n| n.inputs.contains(&iid)) {
                eprintln!(
                    "Gather id={} name={:?} idx={idx:?} idx_op={:?} idx_dtype={:?}",
                    node.id,
                    node.name,
                    idx_node.map(|n| &n.op),
                    idx_node.map(|n| n.shape.dtype())
                );
            }
        }
    }

    for key in [
        kitten_tts_mini_rlx::opts::DURATION_CARRY,
        kitten_tts_mini_rlx::opts::RUNTIME_INPUT_IDS_SHAPE,
    ] {
        if let Some(n) = exec
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, rlx_ir::Op::Param { name } if name == key))
        {
            eprintln!(
                "param {key}: id={} slot={:?} shape={:?} dtype={:?}",
                n.id,
                plan.assignments.get(&n.id),
                n.shape.dims(),
                n.shape.dtype()
            );
        }
    }

    for node in exec.nodes() {
        if let rlx_ir::Op::Input { name } = &node.op {
            if name == "style" || name == "speed" || name == "input_ids" {
                eprintln!(
                    "input {name}: id={} slot={:?} shape={:?}",
                    node.id,
                    plan.assignments.get(&node.id),
                    node.shape.dims()
                );
            }
        }
    }

    let compile_opts2 = compile_opts.clone();
    let compiled = Session::new(Device::Cpu).compile_hir_with(hir, &compile_opts2)?;
    drop(compiled);

    Ok(())
}
