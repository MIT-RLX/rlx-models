/// Whether the official Bellard `tsac` **binary** can run on this host.
pub fn tsac_binary_supported() -> bool {
    cfg!(all(target_os = "linux", target_arch = "x86_64"))
}

/// Deprecated alias.
pub fn tsac_supported() -> bool {
    tsac_binary_supported()
}

pub fn platform_note() -> &'static str {
    if tsac_binary_supported() {
        "Linux x86_64 Bellard binary available; all platforms use native codec with RLX device tags"
    } else {
        "Use native codec (`--device cpu|metal|cuda|…`) — Bellard binary is Linux x86_64 only"
    }
}
