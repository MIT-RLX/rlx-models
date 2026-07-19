use rlx_runtime::Op;
use rlx_tiny_tts::model::import_graph_named;
use std::path::PathBuf;
fn main() -> anyhow::Result<()> {
    let path = PathBuf::from("weights/tts/f5tts/F5_Preprocess.onnx");
    let named = [
        ("audio_len", 87546usize),
        ("text_ids_len", 107),
        ("max_duration", 580),
        ("text_embed_len", 612),
    ];
    let (hir, _, _) = import_graph_named(&path, "F5_Preprocess", 580, true, &named)?;
    let g = hir.lower_to_mir()?.into_graph();
    let mut nw = 0;
    let mut nc = 0;
    for n in g.nodes() {
        match &n.op {
            Op::Where => {
                nw += 1;
                eprint!("Where#{nw} out={:?} n_in={}", n.shape, n.inputs.len());
                for (i, &inp) in n.inputs.iter().enumerate() {
                    let p = g.node(inp);
                    eprint!(
                        " | in{i}={:?} ne={}",
                        p.shape,
                        p.shape.num_elements().unwrap_or(0)
                    );
                }
                eprintln!();
            }
            Op::Compare(c) => {
                nc += 1;
                let l = g.node(n.inputs[0]);
                let r = g.node(n.inputs[1]);
                let ln = l.shape.num_elements().unwrap_or(0);
                let rn = r.shape.num_elements().unwrap_or(0);
                if ln != rn {
                    eprintln!(
                        "Compare#{nc} {c:?} MISMATCH out={:?} lhs_n={ln} rhs_n={rn}",
                        n.shape
                    );
                }
            }
            Op::ElementwiseRegion {
                chain,
                num_inputs,
                scalar_input_mask,
                input_modulus,
                ..
            } => {
                if chain
                    .iter()
                    .any(|s| matches!(s, rlx_runtime::op::ChainStep::Compare(..)))
                {
                    eprintln!(
                        "Region with Compare: nin={num_inputs} mask={scalar_input_mask:#x} modulus={:?}",
                        &input_modulus[..*num_inputs as usize]
                    );
                }
            }
            _ => {}
        }
    }
    eprintln!("total Where={nw} Compare={nc}");
    Ok(())
}
