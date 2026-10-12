//! Where does the training-shape padding actually go?
//!
//! The training log reports one utilisation number, which says nothing about
//! which of the two independent wastes to attack:
//!
//!   * **unfilled rows** — the packer pads every batch in a `(seq, opts)` group
//!     to a constant row count so a short tail batch does not become a new
//!     shape (each shape costs a graph rebuild). The tail batch of each group
//!     therefore computes rows that hold nothing.
//!   * **sequence padding** — every row in a batch is padded to the bucket-
//!     rounded longest row in that batch.
//!
//! Packing fixes the first; tighter length grouping fixes the second. This runs
//! the real packer over the real corpus on CPU with no model weights, so the
//! answer covers all 12,576 records rather than whatever a short GPU run
//! happened to sample.
use anyhow::Result;
use rlx_kev::encode::{EncodeOpts, Encoder, HfTokenizer, rows_of};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let suite =
        std::path::PathBuf::from(args.next().expect("usage: pad_audit <suite> <tokenizer>"));
    let tok = std::path::PathBuf::from(args.next().expect("usage: pad_audit <suite> <tokenizer>"));
    let budget: usize = std::env::var("KEV_TRAIN_TOKEN_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let max_rows: usize = std::env::var("KEV_TRAIN_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let records = rlx_kev::recipe::load_split(&suite.join("train.jsonl"))?;
    let cfg = rlx_kev::train::RecipeConfig::default();
    let examples = rlx_kev::train::epoch_examples(&records, &cfg, 0)?;
    let encoder = Encoder::new(HfTokenizer::from_file(&tok)?)?;
    let opts = EncodeOpts {
        max_state: rlx_kev::encode::MAX_STATE,
        max_branch: rlx_kev::encode::MAX_BRANCH,
        strict: false,
        option_isolation: false,
    };

    let batches = rlx_kev::train::shape_batches(&encoder, &examples, opts, max_rows, budget)?;

    let (mut slot, mut real, mut idle, mut pad) = (0usize, 0usize, 0usize, 0usize);
    // Seq padding splits again, and the two halves have opposite fixes:
    //   * bucket rounding — the batch's longest row rounded up to SHAPE_BUCKET.
    //     Fixed by a finer bucket, at the price of more distinct shapes.
    //   * within-batch spread — short rows padded to the batch's longest row.
    //     Fixed by grouping on length, and a finer bucket makes it WORSE by
    //     splitting the pool each group draws from.
    let (mut bucket_w, mut spread_w) = (0usize, 0usize);
    // Per-shape, so a single pathological bucket cannot hide inside the total.
    let mut per_shape: std::collections::BTreeMap<(usize, usize), (usize, usize, usize, usize)> =
        Default::default();
    for (shape, idxs) in &batches {
        let mut active = 0usize;
        let mut r = 0usize;
        let mut true_max = 0usize;
        for &i in idxs {
            let enc = encoder.encode(&examples[i].record, opts)?;
            let (state_ids, _, rows) = rows_of(&enc)?;
            active += rows.len();
            for x in &rows {
                let len = state_ids.len() + x.ids.len();
                r += len;
                true_max = true_max.max(len);
            }
        }
        let act = active.min(shape.rows);
        bucket_w += act * shape.seq.saturating_sub(true_max);
        spread_w += (act * true_max).saturating_sub(r);
        let s = shape.rows * shape.seq;
        slot += s;
        real += r;
        idle += (shape.rows - active.min(shape.rows)) * shape.seq;
        pad += active.min(shape.rows) * shape.seq - r.min(active.min(shape.rows) * shape.seq);
        let e = per_shape.entry((shape.seq, shape.max_options)).or_default();
        e.0 += s;
        e.1 += r;
        e.2 += (shape.rows - active.min(shape.rows)) * shape.seq;
        e.3 += 1;
    }

    let pct = |x: usize| 100.0 * x as f64 / slot.max(1) as f64;
    println!(
        "{} batches over {} examples, budget {budget}, max_rows {max_rows}",
        batches.len(),
        examples.len()
    );
    println!(
        "slot {slot} positions: real {:.1}%, unfilled rows {:.1}%, seq padding {:.1}%",
        pct(real),
        pct(idle),
        pct(pad)
    );
    println!(
        "  seq padding splits: bucket rounding {:.1}%, within-batch spread {:.1}%",
        pct(bucket_w),
        pct(spread_w)
    );
    println!("\n  seq  opts  batches   slots      real%   idle%");
    for ((seq, o), (s, r, i, n)) in per_shape {
        println!(
            "{seq:5} {o:5} {n:8} {s:9}   {:5.1}%  {:5.1}%",
            100.0 * r as f64 / s.max(1) as f64,
            100.0 * i as f64 / s.max(1) as f64
        );
    }
    Ok(())
}
