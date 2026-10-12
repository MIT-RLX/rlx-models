//! Per-trunk-layer hidden at the last position of row 0, for bisecting a
//! parity failure to the first diverging layer.
use anyhow::Result;
use rlx_kev::api::to_record;
use rlx_kev::encode::{EncodeOpts, Encoder, HfTokenizer, rows_of};
use rlx_runtime::{Device, Session};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (run, base, reqp) = (&a[0], &a[1], &a[2]);

    let req = rlx_kev::api::SystemOneRequest::parse(&std::fs::read_to_string(reqp)?)?;
    let (rec, _) = to_record(&req)?;
    let ck = rlx_kev::Checkpoint::open(std::path::Path::new(run))?;
    let enc = Encoder::new(HfTokenizer::from_file(&ck.tokenizer_path())?)?.encode(
        &rec,
        EncodeOpts {
            max_state: 8192,
            max_branch: 8192,
            strict: false,
            option_isolation: false,
        },
    )?;
    let (state_ids, _, rows) = rows_of(&enc)?;
    let mut ids: Vec<f32> = state_ids.iter().map(|x| *x as f32).collect();
    ids.extend(rows[0].ids.iter().map(|x| *x as f32));
    let seq = ids.len();
    eprintln!("row0 seq={seq} last={}", seq - 1);

    let adapter = rlx_kev::LoraAdapter::open(std::path::Path::new(run))?;
    let kw = rlx_kev::load_merged(std::path::Path::new(base), adapter)?;

    let (hir, params, _packed) = rlx_qwen35::build_qwen35_trunk_export_graph(
        &kw.cfg, kw.weights, 1, seq, false, true, false, false, true,
    )?;
    let mut c = Session::new(Device::Cpu).compile(hir);
    for (n, d) in params {
        c.set_param(&n, &d);
    }
    // `last_token_idx` is declared I32 in this graph, and `run` feeds f32
    // slices — so pass the integer's BIT PATTERN, not its value. Feeding
    // 67.0f32 here reads as a nonsense integer and the gather silently
    // returns position 0 (see rlx-qwen35 tests/last_token_idx_gather.rs).
    let last = [f32::from_bits((seq - 1) as u32)];
    let out = c.run(&[("input_ids", &ids[..]), ("last_token_idx", &last[..])]);
    eprintln!("outputs: {}", out.len());
    let mut dump = Vec::new();
    for (i, o) in out.iter().enumerate() {
        let rms = (o.iter().map(|x| x * x).sum::<f32>() / o.len() as f32).sqrt();
        eprintln!(
            "  out {i:2} (len {}): rms={rms:.5} first4={:?}",
            o.len(),
            &o[..4.min(o.len())]
        );
        dump.push(o.clone());
    }
    if let Some(p) = a.get(3) {
        std::fs::write(p, serde_json::to_string(&dump)?)?;
    }
    Ok(())
}
