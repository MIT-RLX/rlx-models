// Generic native-run parity harness for any Supertonic graph.
// Reads a JSON manifest (written by the python driver) describing the component,
// its inputs, and optional distinct named lengths; runs the graph natively via
// `compile_named`; writes the first output for comparison against onnxruntime.
//
// Machine-independent: work dir defaults under the OS temp dir; onnx dir is
// resolved relative to this crate. Both overridable via RLX_ST_WORK / RLX_ST_ONNX_DIR.
//
// Manifest (`{work}/manifest.json`):
//   {"component":"text_encoder","length":40,
//    "named":[["text_length",40]],
//    "inputs":[{"name":"text_ids","file":"text_ids","dtype":"i64","numel":40}, ...]}
use rlx_runtime::DType;
use rlx_tiny_tts::model::TinyModel;
use rlx_tiny_tts::{BundleConfig, Device};
use std::path::PathBuf;

fn main() {
    let work = std::env::var("RLX_ST_WORK")
        .unwrap_or_else(|_| format!("{}/st_graph", std::env::temp_dir().display()));
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{work}/manifest.json")).unwrap()).unwrap();
    let component: String = man["component"].as_str().unwrap().to_string();
    let length = man["length"].as_u64().unwrap() as usize;
    let named: Vec<(String, usize)> = man["named"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|p| {
                    (
                        p[0].as_str().unwrap().to_string(),
                        p[1].as_u64().unwrap() as usize,
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let named_ref: Vec<(&str, usize)> = named.iter().map(|(k, v)| (k.as_str(), *v)).collect();

    let onnx_dir = PathBuf::from(std::env::var("RLX_ST_ONNX_DIR").unwrap_or_else(|_| {
        format!(
            "{}/../../weights/tts/supertonic-3/onnx",
            env!("CARGO_MANIFEST_DIR")
        )
    }));
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 44100,
        add_blank: true,
        language: "EN".into(),
        speakers: Default::default(),
        default_speaker: None,
        noise_scale: 0.667,
        noise_scale_w: 0.8,
        length_scale: 1.0,
        inter_channels: 80,
        gin_channels: 80,
    };
    let device = match std::env::var("RLX_DEV").as_deref() {
        Ok("mlx") => Device::Mlx,
        Ok("metal") => Device::Metal,
        _ => Device::Cpu,
    };
    let m = TinyModel::new(onnx_dir, cfg);
    // Leak the raw bytes so the (&str,&[u8],DType) slice can borrow them.
    let inputs_owned: Vec<(String, Vec<u8>, DType)> = man["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            let name = i["name"].as_str().unwrap().to_string();
            let dt = match i["dtype"].as_str().unwrap() {
                "i64" => DType::I64,
                "i32" => DType::I32,
                _ => DType::F32,
            };
            let bytes = std::fs::read(format!("{work}/{}", i["file"].as_str().unwrap())).unwrap();
            (name, bytes, dt)
        })
        .collect();
    let inputs: Vec<(&str, &[u8], DType)> = inputs_owned
        .iter()
        .map(|(n, b, d)| (n.as_str(), b.as_slice(), *d))
        .collect();

    let comp: &'static str = Box::leak(component.clone().into_boxed_str());
    let t_c = std::time::Instant::now();
    let mut g = m
        .compile_named(comp, device, length, &named_ref)
        .unwrap_or_else(|e| panic!("compile {component}: {e:#}"));
    let compile_ms = t_c.elapsed().as_millis();
    let t_r = std::time::Instant::now();
    let out = g.run_typed(&inputs);
    let run_ms = t_r.elapsed().as_millis();
    // Second run: warm (compiled graph reused) — the per-inference cost.
    let t_r2 = std::time::Instant::now();
    let _ = g.run_typed(&inputs);
    let run2_ms = t_r2.elapsed().as_millis();
    eprintln!(
        "[st_graph-timing] compile={compile_ms}ms run(cold)={run_ms}ms run(warm)={run2_ms}ms"
    );
    eprintln!(
        "[st_graph] {component} device={device:?} -> {} outputs",
        out.len()
    );
    for (i, (bytes, dt)) in out.iter().enumerate() {
        std::fs::write(format!("{work}/rlx_out_{i}.f32"), bytes).unwrap();
        eprintln!("  out[{i}] {} f32 dtype={dt:?}", bytes.len() / 4);
    }
}
