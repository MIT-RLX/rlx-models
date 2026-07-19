//! Reconstruct a whole document from a directory of scrolled frame captures.
//!
//! Frames are read in filename order (= scroll order), each cleaned of chrome
//! (`fastclean`) then stitched by scroll-overlap into one deduplicated document.
//!
//! Usage: `rlx-termclean-stitch <dir-of-frames>`

use std::fs;

use rlx_termclean::{fastclean, stitch};

fn main() {
    let dir = match std::env::args().nth(1) {
        Some(d) => d,
        None => {
            eprintln!("usage: rlx-termclean-stitch <dir-of-frames>");
            std::process::exit(2);
        }
    };
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read dir {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false))
        .collect();
    paths.sort();

    let raws: Vec<String> = paths
        .iter()
        .map(|p| fs::read_to_string(p).unwrap_or_default())
        .collect();
    let cleaned: Vec<Vec<String>> = raws
        .iter()
        .map(|r| {
            fastclean::clean_frame(r)
                .lines()
                .map(|l| l.to_string())
                .collect()
        })
        .collect();

    let (doc, st) = stitch::stitch_with_stats(&cleaned);
    println!("=== reconstructed document ({} lines) ===", doc.len());
    for l in &doc {
        println!("{l}");
    }
    println!(
        "\n=== {} frames, {} input lines → {} unique ({:.0}% overlap removed) ===",
        st.frames,
        st.input_lines,
        st.output_lines,
        100.0 * st.dedup_ratio()
    );
}
