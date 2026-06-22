use rlx_tsac::{
    TSAC_TARBALL_URL, default_tsac_dir, platform_note, tsac_binary_supported, tsac_supported,
    weights_available,
};

#[test]
fn platform_detection_is_stable() {
    assert_eq!(tsac_supported(), tsac_binary_supported());
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        assert!(tsac_binary_supported());
    } else {
        assert!(!tsac_binary_supported());
    }
    assert!(!platform_note().is_empty());
}

#[test]
fn tarball_url_points_at_bellard() {
    assert!(TSAC_TARBALL_URL.starts_with("https://bellard.org/tsac/"));
}

#[test]
fn default_dir_is_cache_relative() {
    let dir = default_tsac_dir();
    assert!(dir.to_string_lossy().contains("tsac"));
}

#[test]
fn weights_missing_in_empty_dir() {
    let dir = std::env::temp_dir().join(format!("rlx-tsac-empty-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    assert!(!weights_available(&dir));
    let _ = std::fs::remove_dir_all(&dir);
}
