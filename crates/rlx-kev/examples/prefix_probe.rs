//! Compare the prefix-cached forward against the uncached one at several
//! state lengths, to separate padding effects from the rest.
use anyhow::Result;
use rlx_kev::api::to_record;
use rlx_kev::model::ShapePolicy;
use rlx_kev::prefix::PrefixCacheConfig;
use rlx_runtime::Device;

fn req_with(state: &str) -> String {
    serde_json::to_string(&serde_json::json!({
        "state": state,
        "questions": {
            "d": {"type": "choice", "instructions": "Which team?",
                  "criteria": {"returns": "a", "shipping": "b", "billing": serde_json::Value::Null}},
            "e": {"type": "noul", "instructions": "Urgent?"}
        }
    }))
    .expect("json")
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (run, base) = (&a[0], &a[1]);
    let word = "delayed parcel and a duplicate charge ";

    for reps in [1usize, 4, 16, 40] {
        let state = word.repeat(reps);
        let text = req_with(&state);
        let req = rlx_kev::api::SystemOneRequest::parse(&text)?;
        let (rec, _) = to_record(&req)?;

        let mut plain = rlx_kev::load(
            std::path::Path::new(run),
            std::path::Path::new(base),
            Device::Cpu,
            ShapePolicy::default(),
        )?;
        let h_plain = plain.row_hidden(&rec)?;
        let ls = h_plain.seq;
        let want = plain.probs(&rec)?;
        drop(plain);

        let mut cached = rlx_kev::load(
            std::path::Path::new(run),
            std::path::Path::new(base),
            Device::Cpu,
            ShapePolicy::default(),
        )?
        .with_prefix_cache(PrefixCacheConfig {
            capacity: 2,
            min_state_tokens: 1,
        });
        let h_cached = cached.row_hidden(&rec)?;
        let got = cached.probs(&rec)?;

        // The packed readout is at `ls_state + local`; the cached one at
        // `local`. Recover ls_state from the difference.
        let r_p = &h_plain.readouts[0];
        let r_c = &h_cached.readouts[0];
        let ls_state = r_p.decide - r_c.decide;
        let d = h_plain.hidden_size;
        let cos = |a: &[f32], b: &[f32]| {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        };
        let _ = d;
        let mut line = String::new();
        for local in [0usize, 1, 2, r_c.decide] {
            let a = h_plain.at(0, ls_state + local);
            let b = h_cached.at(0, local);
            line.push_str(&format!(" p{local}:{:.5}", cos(a, b)));
        }
        println!("    state_len={ls_state} hidden cos ->{line}");

        let worst = want
            .iter()
            .zip(&got)
            .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
            .fold(0.0f32, f32::max);
        println!(
            "reps {reps:3}  row_seq {ls:4}  max|dp| {worst:.6}   cached {:?}",
            got[1]
        );
    }
    Ok(())
}
