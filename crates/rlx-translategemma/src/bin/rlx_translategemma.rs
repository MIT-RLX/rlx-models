// RLX — versatile ML compiler + runtime.
use rlx_translategemma::cli_run;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match cli_run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rlx-translategemma: {e:#}");
            ExitCode::FAILURE
        }
    }
}
