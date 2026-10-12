//! List every `Op::Input` the prefill graph declares, so a caller can prove
//! it binds all of them. An unbound input reads whatever the arena holds.
use anyhow::Result;
use rlx_ir::Op;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (run, base) = (&a[0], &a[1]);
    let rows: usize = a.get(2).map_or(4, |v| v.parse().unwrap());
    let seq: usize = a.get(3).map_or(128, |v| v.parse().unwrap());

    let adapter = rlx_kev::LoraAdapter::open(std::path::Path::new(run))?;
    let kw = rlx_kev::load_merged(std::path::Path::new(base), adapter)?;
    let (g, _params, _packed) = rlx_qwen35::build_qwen35_graph_sized_ext(
        &kw.cfg, kw.weights, rows, seq, false, false, false, false, None, true, false, true, false,
    )?;
    let mut names = Vec::new();
    for n in g.nodes() {
        if let Op::Input { name } = &n.op {
            names.push((name.clone(), format!("{:?}", n.shape)));
        }
    }
    names.sort();
    println!("{} declared inputs:", names.len());
    for (n, sh) in &names {
        println!("  {n:32} {sh}");
    }
    Ok(())
}
