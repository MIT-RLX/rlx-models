use anyhow::Result;
use clap::Parser;
use rlx_tts_bench::cli::{Cli, entry};

fn main() -> Result<()> {
    entry(Cli::parse())
}
