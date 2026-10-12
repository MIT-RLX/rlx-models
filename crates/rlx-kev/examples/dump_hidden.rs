//! Dump row-0 hidden states at the readout positions, for parity bisecting
//! against `scripts`-side reference dumps.
//!
//! For a memory profile run it under `/usr/bin/time -l` — the phase split
//! (load vs forward) is what shows whether the weight bundle or the compiled
//! arena dominates.
//!
//! ```text
//! cargo run --release -p rlx-kev --example dump_hidden -- <run> <base> <request.json> [out.json]
//! ```
use anyhow::Result;
use rlx_kev::api::to_record;
use rlx_kev::model::ShapePolicy;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        anyhow::bail!("usage: dump_hidden <run_dir> <base_dir> <request.json> [out.json]");
    }
    let req = rlx_kev::api::SystemOneRequest::parse(&std::fs::read_to_string(&a[2])?)?;
    let (rec, _meta) = to_record(&req)?;

    let mut model = rlx_kev::load(
        std::path::Path::new(&a[0]),
        std::path::Path::new(&a[1]),
        rlx_runtime::Device::Cpu,
        ShapePolicy::default(),
    )?;
    let h = model.row_hidden(&rec)?;
    let r0 = &h.readouts[0];
    eprintln!(
        "rows={} seq={} hidden={} | row0 decide={} opts={:?}",
        h.rows, h.seq, h.hidden_size, r0.decide, r0.opts
    );

    let out = serde_json::json!({
        "rows": h.rows, "seq": h.seq, "hidden_size": h.hidden_size,
        "decide": r0.decide, "opts": r0.opts,
        "final_hidden_last": h.at(0, r0.decide),
        "final_hidden_opts": r0.opts.iter().map(|o| h.at(0, *o)).collect::<Vec<_>>(),
        "hidden_first3": (0..3).map(|i| h.at(0, i)).collect::<Vec<_>>(),
        "final_all": (0..h.seq).map(|i| h.at(0, i)).collect::<Vec<_>>(),
    });
    if let Some(p) = a.get(3) {
        std::fs::write(p, serde_json::to_string(&out)?)?;
        eprintln!("wrote {p}");
    }
    let v = h.at(0, r0.decide);
    let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    eprintln!("final hidden @decide: rms={rms:.5} first4={:?}", &v[..4]);
    Ok(())
}
