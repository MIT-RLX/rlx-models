fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    rlx_nanowakeword::cli::run(&args)
}
