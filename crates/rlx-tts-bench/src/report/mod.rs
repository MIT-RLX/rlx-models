//! JSON + HTML + Markdown report writers.

mod html;
mod json;
mod markdown;

pub use html::write_html;
pub use json::{
    BenchRow, Summary, append_results_jsonl, read_results_jsonl, write_results_jsonl,
    write_summary_json,
};
pub use markdown::write_markdown;
