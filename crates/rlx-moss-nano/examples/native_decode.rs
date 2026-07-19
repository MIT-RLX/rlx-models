//! De-risk `moss_tts_decode_step` natively: prefill(seq=P) → decode_step(1 new row,
//! KV, valid_len=P) → compare `global_hidden` vs onnxruntime. Env: RLX_MOSS_DIR.
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;
use std::path::PathBuf;

const RW: usize = 17;
const L: usize = 12; // layers
fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn asf(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let dir =
        PathBuf::from(std::env::var("RLX_MOSS_DIR").unwrap_or("weights/tts/moss-nano".into()));
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 48000,
        add_blank: false,
        language: "EN".into(),
        speakers: Default::default(),
        default_speaker: None,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        length_scale: 1.0,
        inter_channels: 0,
        gin_channels: 0,
    };
    let model = TinyModel::new(dir.clone(), cfg);
    let p = 8usize; // prompt seq

    // prompt rows: text col + 16 pad
    let mut ids = vec![1024i32; p * RW];
    for s in 0..p {
        ids[s * RW] = (100 + s) as i32;
    }
    let mask = vec![1i32; p];

    // ---- KV: from ORT dump (KVDIR) to isolate decode_step, else native prefill ----
    let mut kv: Vec<Vec<u8>> = Vec::new();
    if let Ok(kvdir) = std::env::var("KVDIR") {
        for i in 0..2 * L {
            kv.push(std::fs::read(format!("{kvdir}/kv_{i}.f32"))?);
        }
    } else {
        let mut pg = model
            .compile_named(
                "moss_tts_prefill",
                Device::Cpu,
                p,
                &[("batch", 1), ("prefill_seq", p)],
            )
            .map_err(|e| anyhow::anyhow!("compile prefill: {e:#}"))?;
        let pout = pg.run_typed(&[
            ("input_ids", &i32b(&ids), DType::I32),
            ("attention_mask", &i32b(&mask), DType::I32),
        ]);
        for i in 1..=2 * L {
            kv.push(pout[i].0.clone());
        } // [1,p,12,64] each
    }

    // one new assistant row
    let mut row = vec![1024i32; RW];
    row[0] = 9;

    // ---- decode_step compiled at past_seq=PAST_SEQ (>= p → padded KV) ----
    let past_seq: usize = std::env::var("PAST_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(p);
    if past_seq > p {
        // pad each KV layer from [1,p,768] to [1,past_seq,768] with trailing zeros.
        for k in kv.iter_mut() {
            let mut buf = vec![0u8; past_seq * 768 * 4];
            buf[..k.len()].copy_from_slice(k);
            *k = buf;
        }
    }
    let mut dg = model
        .compile_named(
            "moss_tts_decode_step",
            Device::Cpu,
            1,
            &[("batch", 1), ("step_seq", 1), ("past_seq", past_seq)],
        )
        .map_err(|e| anyhow::anyhow!("compile decode_step: {e:#}"))?;
    let mut ins: Vec<(&str, &[u8], DType)> = vec![
        ("input_ids", &[], DType::I32), // placeholder, fixed below
    ];
    let rowb = i32b(&row);
    let pvl = i32b(&[p as i32]);
    ins.clear();
    ins.push(("input_ids", &rowb, DType::I32));
    ins.push(("past_valid_lengths", &pvl, DType::I32));
    let names: Vec<String> = (0..L)
        .flat_map(|i| [format!("past_key_{i}"), format!("past_value_{i}")])
        .collect();
    for (n, k) in names.iter().zip(kv.iter()) {
        ins.push((n.as_str(), k, DType::F32));
    }
    let dout = dg.run_typed(&ins);
    let nh = asf(&dout[0].0); // [1,1,768]
    let peak = nh
        .iter()
        .fold(0f32, |m, &x| if x.is_nan() { m } else { m.max(x.abs()) });
    let nan = nh.iter().filter(|x| x.is_nan()).count();
    eprintln!(
        "native decode_step hidden: n={} peak={peak:.4} nans={nan} first={:?}",
        nh.len(),
        &nh[..4.min(nh.len())]
    );
    if let Ok(sp) = std::env::var("SP") {
        let _ = std::fs::write(format!("{sp}/dec_native.f32"), &dout[0].0);
        let _ = std::fs::write(format!("{sp}/dec_row.i32"), &rowb);
    }
    Ok(())
}
