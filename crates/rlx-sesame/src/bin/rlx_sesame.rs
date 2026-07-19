fn main() {
    if let Err(e) = rlx_sesame::cli::run(rlx_sesame::cli::Args::parse()) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

use clap::Parser;
