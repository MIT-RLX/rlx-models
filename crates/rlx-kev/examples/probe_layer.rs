//! Feed a reference layer input through ONE rlx trunk layer and compare the
//! output to the reference's own layer output. Isolates a single block, so
//! nothing accumulates.
//!
//! ```text
//! probe_layer <run> <base> <ref_hidden.json> <layer>
//! ```
use anyhow::Result;
use rlx_runtime::{Device, Session};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (run, base, refp) = (&a[0], &a[1], &a[2]);
    let il: usize = a[3].parse()?;

    let rf: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refp)?)?;
    let pair = &rf["layer_in_out"][il.to_string()];
    let grab = |v: &serde_json::Value| -> Vec<Vec<f32>> {
        v.as_array()
            .expect("rows")
            .iter()
            .map(|r| {
                r.as_array()
                    .expect("row")
                    .iter()
                    .map(|x| x.as_f64().expect("f") as f32)
                    .collect()
            })
            .collect()
    };
    let h_in = grab(&pair["in"]);
    let h_out = grab(&pair["out"]);
    let seq = h_in.len();
    let d = h_in[0].len();
    eprintln!("layer {il}: seq={seq} hidden={d}");

    let adapter = rlx_kev::LoraAdapter::open(std::path::Path::new(run))?;
    let kw = rlx_kev::load_merged(std::path::Path::new(base), adapter)?;
    let (g, params, _packed) =
        rlx_qwen35::build_qwen35_layer_probe_graph(&kw.cfg, kw.weights, il, 1, seq, false)?;
    let mut c = Session::new(Device::Cpu).compile(g);
    for (n, v) in params {
        c.set_param(&n, &v);
    }
    let flat: Vec<f32> = h_in.iter().flatten().copied().collect();
    let out = c.run(&[("trunk_h", &flat[..])]);
    let got = &out[0];
    assert_eq!(got.len(), seq * d, "probe output shape");

    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    println!("pos   cos        max|diff|");
    for i in [0usize, 1, 2, 3, 4, 8, 16, 32, seq - 1] {
        let want = &h_out[i];
        let mine = &got[i * d..(i + 1) * d];
        let dot: f32 = want.iter().zip(mine).map(|(x, y)| x * y).sum();
        let cs = dot / (rms(want) * rms(mine) * d as f32);
        let md = want
            .iter()
            .zip(mine)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!("{i:4}  {cs:.8}  {md:.6}");
    }
    Ok(())
}
