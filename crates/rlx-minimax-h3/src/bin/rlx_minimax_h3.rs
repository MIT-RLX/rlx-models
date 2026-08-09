fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    rlx_minimax_h3::cli_run(&args)
}
