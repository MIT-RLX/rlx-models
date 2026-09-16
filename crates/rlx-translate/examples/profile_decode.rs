//! Where does decoding time go?
//!
//! Times the four phases separately — model load, encode, decoder steps, and
//! readout scoring — because "translation is slow" is not actionable and the
//! obvious suspect (the 20-block encoder) is not obviously the expensive one.

use std::time::Instant;

use rlx_translate::assets::Assets;
use rlx_translate::decode::{DecoderState, Nmt};
use rlx_translate::pdec::PDecParams;
use rlx_translate::quasar::LangPair;
use rlx_translate::spm::Vocab;

fn main() -> anyhow::Result<()> {
    let assets = Assets::discover();
    let pair = LangPair::parse(
        &std::env::args()
            .nth(1)
            .unwrap_or_else(|| "en_US-fr_FR".to_string()),
    )?;
    let (_, config) = assets.best_config(&pair)?;
    let params = PDecParams::from_block(config.mt_app()?.translator(&pair)?)?;
    let home = assets
        .model_home_for(&pair, &params.model_file)
        .ok_or_else(|| anyhow::anyhow!("model not installed"))?;

    let t = Instant::now();
    let vocab = Vocab::load(home.join("spm.model"))?;
    let t_vocab = t.elapsed();

    let t = Instant::now();
    let tgt: String = pair
        .target
        .chars()
        .take(2)
        .collect::<String>()
        .to_lowercase();
    let src_lang: String = pair
        .source
        .chars()
        .take(2)
        .collect::<String>()
        .to_lowercase();
    let nmt = Nmt::load_for_pair(
        &home,
        &[home.as_path()],
        &src_lang,
        &tgt,
        &params.shortlist.lang_pair,
    )?;
    let t_load = t.elapsed();

    let text = "my neighbour bought seventeen blue umbrellas last spring";
    let bos = vocab.id("<s>").expect("bos");
    let mut src = Vec::new();
    for p in params
        .source_token_pieces()
        .iter()
        .chain(params.target_token_pieces().iter())
    {
        src.push(vocab.id(p).expect("control token"));
    }
    src.extend(vocab.encode(text));
    src.push(bos);

    // Encode twice: the first pass pays for any lazily-built tables.
    let t = Instant::now();
    let ho = nmt.encode(&src)?;
    let t_encode1 = t.elapsed();
    let t = Instant::now();
    let ho2 = nmt.encode(&src)?;
    let t_encode2 = t.elapsed();
    drop(ho2);

    let t = Instant::now();
    let cands = nmt.candidates(&src, &[bos], vocab.len());
    let t_cands = t.elapsed();

    // Greedy walk, timing the decoder and the readout apart.
    let mut state = DecoderState::new(nmt.layers(), nmt.width);
    let mut prev = bos;
    let (mut d_step, mut d_logits) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
    let mut steps = 0usize;
    for pos in 1..=16usize {
        let t = Instant::now();
        let (hidden, next) = nmt.step(prev, pos, &state, &ho)?;
        d_step += t.elapsed();
        let t = Instant::now();
        let scores = nmt.logits(&hidden, &cands)?;
        d_logits += t.elapsed();
        steps += 1;
        let (best, _) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("best");
        state = next;
        prev = cands[best];
        if prev == bos {
            break;
        }
    }

    let t = Instant::now();
    let _ = nmt.translate_nbest(&vocab, &params, text, 1)?;
    let t_beam = t.elapsed();

    println!(
        "  vocab load        {:>9.1} ms",
        t_vocab.as_secs_f64() * 1e3
    );
    println!("  model load        {:>9.1} ms", t_load.as_secs_f64() * 1e3);
    println!(
        "  shortlist union   {:>9.1} ms  ({} candidates)",
        t_cands.as_secs_f64() * 1e3,
        cands.len()
    );
    println!(
        "  encode (1st)      {:>9.1} ms",
        t_encode1.as_secs_f64() * 1e3
    );
    println!(
        "  encode (2nd)      {:>9.1} ms",
        t_encode2.as_secs_f64() * 1e3
    );
    println!(
        "  decoder steps     {:>9.1} ms  ({steps} steps, {:.1} ms each)",
        d_step.as_secs_f64() * 1e3,
        d_step.as_secs_f64() * 1e3 / steps as f64
    );
    println!(
        "  readout scoring   {:>9.1} ms  ({:.1} ms each)",
        d_logits.as_secs_f64() * 1e3,
        d_logits.as_secs_f64() * 1e3 / steps as f64
    );
    println!("  full beam search  {:>9.1} ms", t_beam.as_secs_f64() * 1e3);
    // Timings mean nothing without the settings that produced them, and those
    // are now overridable from the environment.
    if rlx_translate::profile::enabled() {
        println!(
            "\n  by op (whole run)\n{}",
            rlx_translate::profile::report()
        );
    } else {
        println!("\n  (set RLX_TRANSLATE_PROFILE=1 for a per-op breakdown)");
    }
    println!("\n  settings");
    for (k, v, _) in nmt.tuning().describe() {
        println!("    {k:<18} {v}");
    }
    Ok(())
}
