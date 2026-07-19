// Probe: can rlx-onnx-import import a given ONNX graph natively (no ORT)?
// Usage: cargo run -p rlx-tiny-tts --example onnx_import_check -- <file.onnx> [len] [name=len ...]
//
// Extra `name=len` args bind distinct ONNX dim_param names to distinct concrete
// lengths (e.g. `text_length=40 latent_length=120`) so a single cross-attention
// graph can carry two+ dynamic lengths instead of collapsing them to one.
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: onnx_import_check <file.onnx> [len] [name=len ...]");
    let len: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let named: Vec<(String, usize)> = std::env::args()
        .skip(3)
        .filter_map(|a| {
            let (k, v) = a.split_once('=')?;
            Some((k.to_string(), v.parse().ok()?))
        })
        .collect();
    let named_ref: Vec<(&str, usize)> = named.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let p = std::path::PathBuf::from(&path);
    match rlx_tiny_tts::model::import_graph_named(&p, "probe", len, true, &named_ref) {
        Ok((_hir, params, _report)) => {
            let n = params.len();
            println!("OK  {path}  params={n}  named={named:?}");
        }
        Err(e) => println!("ERR {path}  {e:#}"),
    }
}
