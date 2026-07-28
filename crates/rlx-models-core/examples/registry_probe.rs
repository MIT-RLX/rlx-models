//! Manual probe for the GGUF/HF model registry (built-ins + optional live GGUF).
fn main() {
    use rlx_models_core::gguf_architecture_from_path;
    use rlx_models_core::model_registry::*;

    ensure_builtin_gguf_models();
    for (arch, want) in [
        ("laguna", Some("laguna")),
        ("phi3", Some("phi")),
        ("mistral3", Some("mistral")),
        ("qwen35moe", Some("qwen35")),
        ("flux", Some("flux2")),
        ("bert", None),
        ("clip", None),
    ] {
        let got = runner_for_gguf_arch(arch);
        println!(
            "arch={arch:12} runner={got:?} fam={:?} hint={}",
            family_for_gguf_arch(arch),
            hint_for_gguf_arch(arch).unwrap_or("<none>")
        );
        assert_eq!(got, want, "{arch}");
    }
    assert_eq!(runner_for_hf_model_type("gemma4"), Some("gemma"));
    assert_eq!(runner_for_hf_model_type("gemma4moe"), None);
    assert_eq!(runner_for_hf_model_type("whisper"), Some("whisper"));
    assert_eq!(runner_for_hf_model_type("ministral3"), Some("mistral"));
    let models = registered_gguf_models();
    println!("registered={}", models.len());
    for m in &models {
        println!(
            "  id={:16} runner={:?} arches={:?}",
            m.id, m.runner, m.arches
        );
    }
    if let Ok(p) = std::env::var("RLX_REGISTRY_PROBE_GGUF") {
        if !p.is_empty() {
            let path = std::path::Path::new(&p);
            let arch = gguf_architecture_from_path(path).expect("arch");
            let runner = runner_for_gguf_arch(&arch);
            println!(
                "live gguf={} arch={} runner={:?}",
                path.display(),
                arch,
                runner
            );
            assert_eq!(arch, "laguna");
            assert_eq!(runner, Some("laguna"));
        }
    }
    println!("registry_probe ok");
}
