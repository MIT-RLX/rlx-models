fn main() -> anyhow::Result<()> {
    rlx_mimi::run(&std::env::args().skip(1).collect::<Vec<_>>())
}
