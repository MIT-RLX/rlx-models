// RLX — GPLv3. Localize the non-first-stage divergence.
//   SAVE:  RLX_DSV4_SAVE=1 ... example dsv4_split_check   (writes h_a = hidden after layer 2)
//   PROBE: RLX_DSV4_DBG=attn|moe ... example dsv4_split_check  (runs stage 3..4 on h_a, non-first)
use anyhow::{Context, Result};
use rlx_ir::DType;
use rlx_ir::quant::QuantScheme;
use rlx_models_core::standard_decoder::{DeepseekV4Spec, build_deepseek_v4_stage};
use rlx_models_core::weight_loader::MlxLoader;
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;

type Packed = HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>;
const HA: &str = "/private/tmp/claude-501/-Users-Shared-rlx-models/b3447056-e916-4a44-8d5c-f126828edfd6/scratchpad/h_a.f32";

fn run_stage(
    dir: &str,
    spec: &DeepseekV4Spec,
    seq: usize,
    a: usize,
    b: usize,
    first: bool,
    last: bool,
    input: (&str, &[f32]),
) -> Result<Vec<f32>> {
    let mut loader = MlxLoader::open_lazy(dir)?;
    let mut packed: Packed = HashMap::new();
    let (graph, params) =
        build_deepseek_v4_stage(spec, &mut loader, seq, a..b, first, last, &mut packed)?;
    let mut c = Session::new(Device::Cpu).compile_with(graph, &CompileOptions::default());
    for (n, d) in &params {
        c.set_param(n, d);
    }
    for (n, (bytes, _, _)) in &packed {
        let dt = if n.ends_with(".scales") || n.ends_with(".biases") {
            DType::BF16
        } else {
            DType::U8
        };
        c.set_param_typed(n, bytes, dt);
    }
    Ok(c.run(&[input]).into_iter().next().unwrap())
}
fn mag(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0f32, f32::max)
}

fn main() -> Result<()> {
    let dir = std::env::var("RLX_DSV4_DIR").context("RLX_DSV4_DIR")?;
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{dir}/config.json"))?)?;
    let spec = DeepseekV4Spec::from_config(&cfg)?;
    let seq = 6usize;
    let ids: Vec<f32> = [0u32, 671, 6102, 294, 8760, 344]
        .iter()
        .map(|&x| x as f32)
        .collect();

    if std::env::var("RLX_DSV4_WHOLE").is_ok() {
        // Run layers 0..4 as ONE graph; with RLX_DSV4_DBGLAYER=3 + DBG=o this
        // outputs layer 3's attention `o` computed IN-GRAPH (first=true).
        let out = run_stage(&dir, &spec, seq, 0, 4, true, false, ("input_ids", &ids))?;
        println!(
            "WHOLE stage(0..4) [DBG={:?} layer3]: max|abs| {:.3e} sample {:?}",
            std::env::var("RLX_DSV4_DBG").ok(),
            mag(&out),
            &out[..4]
        );
        if let Ok(p) = std::env::var("RLX_DSV4_OUT") {
            std::fs::write(
                p,
                out.iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )?;
        }
        return Ok(());
    }
    if std::env::var("RLX_DSV4_SAVE").is_ok() {
        let h_a = run_stage(&dir, &spec, seq, 0, 3, true, false, ("input_ids", &ids))?;
        let bytes: Vec<u8> = h_a.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(HA, &bytes)?;
        println!(
            "saved h_a: {} floats, max|abs| {:.3e}",
            h_a.len(),
            mag(&h_a)
        );
        return Ok(());
    }
    // load h_a, run stage 3..4 (non-first). RLX_DSV4_DBG=attn/moe/routed localizes.
    let raw = std::fs::read(HA)?;
    let h_a: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let x = run_stage(&dir, &spec, seq, 3, 4, false, false, ("hidden_in", &h_a))?;
    println!(
        "stage(3..4|h_a) [DBG={:?}]: max|abs| {:.3e} sample {:?}",
        std::env::var("RLX_DSV4_DBG").ok(),
        mag(&x),
        &x[..4]
    );
    if let Ok(p) = std::env::var("RLX_DSV4_OUT") {
        std::fs::write(
            p,
            x.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(),
        )?;
    }
    Ok(())
}
