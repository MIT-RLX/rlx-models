// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

#[cfg(feature = "native")]
#[test]
fn native_weights_available_when_present() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
    if dir.join("model.safetensors").is_file() {
        assert!(kitten_tts_mini_rlx::native_weights_available(&dir));
        assert!(kitten_tts_mini_rlx::resolve_weights_file(&dir).is_some());
    }
}

#[cfg(feature = "native")]
#[test]
fn module_map_covers_all_kinds() {
    use kitten_tts_mini_rlx::native::config::ModuleKind;
    use kitten_tts_mini_rlx::native::flow::MODULE_INDEX;
    assert_eq!(MODULE_INDEX.len(), ModuleKind::ALL.len());
}

#[cfg(feature = "native")]
#[test]
fn gguf_weights_load_when_present() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
    let gguf = dir.join("model.gguf");
    if !gguf.is_file() {
        eprintln!("skip gguf_weights_load: run `just export-kitten-gguf`");
        return;
    }
    let w = kitten_tts_mini_rlx::load_weights(&dir).expect("load gguf");
    assert!(!w.f32.is_empty());
}
