fn main() -> anyhow::Result<()> {
    rlx_tsac::run(&std::env::args().skip(1).collect::<Vec<_>>())
}
