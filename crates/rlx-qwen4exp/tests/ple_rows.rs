// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Reading n-gram rows out of a GGUF without holding the table.
//!
//! The published table is 51 billion parameters and ~28 GB, and a decode step
//! needs nine rows of it, so [`GgufPleRows`] reads the quantization blocks a
//! row overlaps and nothing else. The oracle here is the crate's own
//! whole-tensor dequantization: a lazily read row has to equal the
//! corresponding slice of it **exactly**, since both end up in the same
//! `dequant_typed`. What the lazy path can get wrong is the arithmetic from a
//! row index to a byte offset — which is what these fixtures vary.

use rlx_gguf::{GgmlType, GgufFile, GgufWriter};
use rlx_qwen4exp::ple::{GgufPleRows, PLE_TABLE_TENSOR, PleRowSource, block_elements};

/// Deterministic table values, spread so quantization error is visible.
fn values(rows: usize, head_dim: usize) -> Vec<f32> {
    (0..rows * head_dim)
        .map(|i| ((i as f32) * 0.37).sin() * 0.8)
        .collect()
}

fn write_table(
    dir: &std::path::Path,
    name: &str,
    data: &[f32],
    head_dim: usize,
    dtype: GgmlType,
) -> std::path::PathBuf {
    let rows = data.len() / head_dim;
    let bytes = match dtype {
        GgmlType::F32 => data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
        GgmlType::IQ4NL => {
            let blk = block_elements(GgmlType::IQ4NL).expect("iq4_nl block geometry");
            let blk_bytes = rlx_gguf::bytes_for_public(GgmlType::IQ4NL, blk).unwrap();
            assert_eq!(data.len() % blk, 0, "the fixture must be block-aligned");
            let mut out = vec![0u8; data.len() / blk * blk_bytes];
            for (b, chunk) in data.chunks(blk).enumerate() {
                rlx_gguf::iq_quantize::quantize_iq4_nl_block(
                    chunk,
                    &mut out[b * blk_bytes..(b + 1) * blk_bytes],
                );
            }
            out
        }
        other => panic!("fixture does not write {other:?}"),
    };
    let mut w = GgufWriter::new();
    w.set_arch("qwen4exp");
    // GGML order: innermost dimension first, so the row width leads.
    w.add_tensor_bytes(PLE_TABLE_TENSOR, vec![head_dim, rows], dtype, bytes)
        .expect("add table");
    let path = dir.join(name);
    w.write_to_path(&path).expect("write gguf");
    path
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rlx-qwen4exp-ple-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("tmpdir");
    d
}

/// An f32 table: rows come back exactly as written.
#[test]
fn f32_rows_are_read_verbatim() {
    let (rows, hd) = (64usize, 160usize);
    let data = values(rows, hd);
    let dir = tmpdir("f32");
    let path = write_table(&dir, "table-f32.gguf", &data, hd, GgmlType::F32);

    let mut src = GgufPleRows::open(&path, hd).expect("open table");
    assert_eq!(src.rows(), rows as u64);
    for r in [0usize, 1, 17, rows - 1] {
        let got = src.row(r as u32).expect("row");
        assert_eq!(got, &data[r * hd..(r + 1) * hd], "row {r}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The published type, against the whole-tensor dequantization.
#[test]
fn iq4_nl_rows_match_the_whole_tensor_dequant() {
    let (rows, hd) = (64usize, 160usize);
    let data = values(rows, hd);
    let dir = tmpdir("iq4nl");
    let path = write_table(&dir, "table-iq4nl.gguf", &data, hd, GgmlType::IQ4NL);

    let whole = GgufFile::from_path(&path).expect("open whole");
    let (reference, shape) = whole.dequant_f32(PLE_TABLE_TENSOR).expect("dequant all");
    assert_eq!(shape, vec![hd, rows]);

    let mut src = GgufPleRows::open(&path, hd).expect("open table");
    for r in 0..rows {
        let got = src.row(r as u32).expect("row");
        assert_eq!(
            got,
            &reference[r * hd..(r + 1) * hd],
            "row {r} read lazily differs from the same row dequantized whole"
        );
    }
    // 160 elements is 5 IQ4_NL blocks of 18 bytes, and nothing was cached
    // twice, so the whole sweep costs 90 bytes a row.
    assert_eq!(src.bytes_read(), (rows * 5 * 18) as u64);
    std::fs::remove_dir_all(&dir).ok();
}

/// A row width that does not divide the block, so rows straddle boundaries.
#[test]
fn rows_that_straddle_a_block_boundary_still_match() {
    // 40 is not a multiple of IQ4_NL's 32: row 1 starts 8 elements into block 1
    // and ends in block 2. Reading one block, or rounding the offset, breaks.
    let (rows, hd) = (32usize, 40usize);
    let data = values(rows, hd);
    let dir = tmpdir("straddle");
    let path = write_table(&dir, "table-straddle.gguf", &data, hd, GgmlType::IQ4NL);

    let whole = GgufFile::from_path(&path).expect("open whole");
    let (reference, _) = whole.dequant_f32(PLE_TABLE_TENSOR).expect("dequant all");

    let mut src = GgufPleRows::open(&path, hd)
        .expect("open table")
        .with_cache_blocks(0);
    for r in 0..rows {
        let got = src.row(r as u32).expect("row");
        assert_eq!(got, &reference[r * hd..(r + 1) * hd], "row {r}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The window cache serves repeats without reading, and changes no value.
#[test]
fn the_window_cache_is_transparent() {
    let (rows, hd) = (16usize, 160usize);
    let data = values(rows, hd);
    let dir = tmpdir("cache");
    let path = write_table(&dir, "table-cache.gguf", &data, hd, GgmlType::IQ4NL);

    let mut src = GgufPleRows::open(&path, hd).expect("open table");
    let first = src.row(3).expect("row");
    let after_one = src.bytes_read();
    for _ in 0..9 {
        assert_eq!(
            src.row(3).expect("row"),
            first,
            "a cached row must not drift"
        );
    }
    assert_eq!(src.bytes_read(), after_one, "repeats must not re-read");
    assert_eq!(src.cache_hits(), 9);

    let mut cold = GgufPleRows::open(&path, hd)
        .expect("open table")
        .with_cache_blocks(0);
    assert_eq!(
        cold.row(3).expect("row"),
        first,
        "the cache changes nothing"
    );
    for _ in 0..9 {
        cold.row(3).expect("row");
    }
    assert_eq!(cold.bytes_read(), after_one * 10, "no cache, ten reads");
    assert_eq!(cold.cache_hits(), 0);
    std::fs::remove_dir_all(&dir).ok();
}

/// A row past the end, and a table whose width is not the configured one.
#[test]
fn bad_rows_and_bad_shapes_are_refused() {
    let (rows, hd) = (8usize, 160usize);
    let data = values(rows, hd);
    let dir = tmpdir("bad");
    let path = write_table(&dir, "table-bad.gguf", &data, hd, GgmlType::F32);

    let mut src = GgufPleRows::open(&path, hd).expect("open table");
    let err = src.row(rows as u32).expect_err("row 8 of 8");
    assert!(format!("{err:#}").contains("past the table"), "{err:#}");

    let err = GgufPleRows::open(&path, 64)
        .err()
        .expect("a 64-wide read of a 160-wide table must be rejected");
    let msg = format!("{err:#}");
    assert!(msg.contains("expected [64, rows]"), "{msg}");

    let err = GgufPleRows::open_tensor(&path, "nope.weight", hd)
        .err()
        .expect("a missing tensor must be rejected");
    assert!(format!("{err:#}").contains("no tensor"), "{err:#}");
    std::fs::remove_dir_all(&dir).ok();
}

/// The runner decodes identically whether its rows come from memory or the file.
///
/// This is the end of the chain the lazy source exists for: same values, two
/// delivery paths, so the logits must match bit for bit. The table here is the
/// published IQ4_NL and the tiny config's row width is 4, so eight rows share
/// every block — the case the window cache is for.
#[test]
fn a_runner_decodes_the_same_from_memory_or_from_the_file() {
    use rlx_core::weight_map::WeightMap;
    use rlx_qwen4exp::config::PleHash;
    use rlx_qwen4exp::ple::DensePleTable;
    use rlx_qwen4exp::runner::{Qwen4ExpRunner, RunnerOptions};
    use rlx_qwen4exp::synth;
    use rlx_runtime::Device;

    let cfg = synth::tiny_config(2048);
    let hd = cfg.ple_head_dim;
    let rows = PleHash::derive(&cfg, 0).padded_rows as usize;
    let data = values(rows, hd);
    let dir = tmpdir("runner");
    let path = write_table(&dir, "table-runner.gguf", &data, hd, GgmlType::IQ4NL);

    // What the file actually holds after quantization — the in-memory table has
    // to carry the same values, or this would compare two different models.
    let whole = GgufFile::from_path(&path).expect("open whole");
    let (dequantized, _) = whole.dequant_f32(PLE_TABLE_TENSOR).expect("dequant all");

    let tokens = [11u32, 2048, 7, 909];
    let mut logits = Vec::new();
    for from_file in [false, true] {
        let mut w = synth::synth_tensors(&cfg, 13);
        w.remove("per_layer_token_embd.weight");
        let mut map = WeightMap::from_tensors(w);
        let run = Qwen4ExpRunner::from_source(
            cfg.clone(),
            &mut rlx_core::flow_util::WeightMapSource(&mut map),
            RunnerOptions::new(Device::Cpu),
        )
        .expect("build runner");
        let mut run = if from_file {
            run.with_ple_rows(Box::new(GgufPleRows::open(&path, hd).expect("open rows")))
        } else {
            run.with_ple_rows(Box::new(DensePleTable {
                data: dequantized.clone(),
                rows,
                head_dim: hd,
            }))
        };
        let mut out = run.prefill_logits(&tokens[..2]).expect("prefill");
        for &t in &tokens[2..] {
            out = run.decode_logits(t).expect("decode");
        }
        logits.push(out);
    }
    assert_eq!(
        logits[0], logits[1],
        "the file-backed row source changed the model's output"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The published table's geometry, from the real checkpoint's header.
///
/// Everything the lazy reader claims about cost rests on this: the table is
/// IQ4_NL, its rows are 160 wide, and IQ4_NL blocks 32 elements — so a row is
/// exactly 5 blocks and costs 90 bytes. If a future conversion changes the type
/// or the width, the reader still works (the geometry is derived, not assumed)
/// but the numbers in its documentation stop being true, and this says so.
#[test]
fn the_published_tables_geometry_is_the_one_documented() {
    const MANIFEST: &str = include_str!("fixtures/qwen38_flash_next_tensors.tsv");
    let line = MANIFEST
        .lines()
        .find(|l| l.starts_with(PLE_TABLE_TENSOR))
        .expect("the published checkpoint carries the n-gram table");
    let mut f = line.split('\t');
    f.next();
    let dims: Vec<u64> = f
        .next()
        .expect("dims")
        .split(',')
        .map(|d| d.parse().expect("dim"))
        .collect();
    let ty = f.next().expect("type");

    assert_eq!(ty, "IQ4_NL");
    assert_eq!(dims, vec![160, 320_001_536], "[head_dim, rows]");

    let blk = block_elements(GgmlType::IQ4NL).expect("iq4_nl block geometry");
    let blk_bytes = rlx_gguf::bytes_for_public(GgmlType::IQ4NL, blk).expect("iq4_nl block");
    assert_eq!(
        (blk, blk_bytes),
        (32, 18),
        "IQ4_NL packs 32 elements in 18 bytes"
    );
    let row_w = dims[0] as usize;
    assert_eq!(row_w % blk, 0, "a row is a whole number of blocks");
    assert_eq!(row_w / blk * blk_bytes, 90, "bytes per row");

    // 51.2e9 parameters at 0.5625 bytes each: the ~28 GB the docs quote.
    let total = dims[0] * dims[1] / blk as u64 * blk_bytes as u64;
    let gb = total as f64 / 1e9;
    assert!((28.0..29.0).contains(&gb), "table is {gb:.1} GB");
}
