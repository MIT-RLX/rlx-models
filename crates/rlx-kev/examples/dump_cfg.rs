//! Print the `Qwen35Config` rlx parses from an HF base directory.
fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).expect("usage: dump_cfg <base_dir>");
    let p = std::path::Path::new(&dir).join("config.json");
    let c = rlx_qwen35::Qwen35Config::from_hf_config_json(&p)?;
    println!("{c:#?}");
    Ok(())
}
