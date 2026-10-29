//! `pyannote` VAD against whisperX 3.8.6's internals (`tribev2-ref/asr/dump_vad.py`):
//! raw window activations, aggregated frame scores, Binarize, merge_chunks.
//! usage: pyannote_parity CKPT STAGES_DIR NAME
use rlx_vad::pyannote::{
    PyannoteSegmentation, Region, STEP, WINDOW, aggregate, binarize, merge_chunks,
};

fn f32s(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (ckpt, dir, name) = (&a[1], &a[2], &a[3]);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{name}.vad.json"))?)?;
    let audio = f32s(&format!("{dir}/{name}.audio.f32"));
    let raw_ref = f32s(&format!("{dir}/{name}.vad_raw.f32"));
    let agg_ref = f32s(&format!("{dir}/{name}.vad_agg.f32"));
    let regions = |v: &serde_json::Value| -> Vec<Region> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|r| Region {
                start: r[0].as_f64().unwrap(),
                end: r[1].as_f64().unwrap(),
            })
            .collect()
    };
    let (bin_ref, merged_ref) = (regions(&meta["binarized"]), regions(&meta["merged"]));
    let n = audio.len();
    let num_chunks = if n >= WINDOW {
        (n - WINDOW) / STEP + 1
    } else {
        0
    };
    let has_last = n < WINDOW || (n - WINDOW) % STEP > 0;
    // 1. host-side stages from the REFERENCE raw activations
    let agg_from_ref = aggregate(&raw_ref, num_chunks + has_last as usize, n, has_last);
    let agg_exact = agg_from_ref == agg_ref;
    let bin_from_ref = binarize(&agg_ref, 0.5, 0.363, 30.0);
    let merged_from_ref = merge_chunks(&bin_from_ref, 30.0);
    // 2. the model
    let m = PyannoteSegmentation::open(
        std::path::Path::new(ckpt),
        if std::env::var_os("VAD_METAL").is_some() {
            rlx_runtime::Device::Metal
        } else {
            rlx_runtime::Device::Cpu
        },
    )?;
    let t = std::time::Instant::now();
    let scores = m.frame_scores(&audio)?;
    let el = t.elapsed();
    let dmax = scores
        .iter()
        .zip(&agg_ref)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let bin = binarize(&scores, 0.5, 0.363, 30.0);
    let merged = merge_chunks(&bin, 30.0);
    let max_dt = |a: &[Region], b: &[Region]| -> String {
        if a.len() != b.len() {
            return format!("COUNT {} vs {}", a.len(), b.len());
        }
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x.start - y.start).abs().max((x.end - y.end).abs()))
            .fold(0f64, f64::max);
        if d == 0.0 {
            "exact".into()
        } else {
            format!("max |dt| {d:.4}s")
        }
    };
    if std::env::var_os("VAD_DEBUG").is_some() {
        eprintln!("ref {:?}\nrs  {:?}", bin_ref, bin_from_ref);
    }
    println!(
        "{name}: agg(ref raw) {} | binarize(ref agg) {} | merge(ref) {} || model: frame max |d| {dmax:.2e}, binarized {}, merged {} [{:.0} ms, {} frames vs {}]",
        if agg_exact { "exact" } else { "DIFF" },
        max_dt(&bin_from_ref, &bin_ref),
        max_dt(&merged_from_ref, &merged_ref),
        max_dt(&bin, &bin_ref),
        max_dt(&merged, &merged_ref),
        el.as_secs_f64() * 1e3,
        scores.len(),
        agg_ref.len()
    );
    Ok(())
}
