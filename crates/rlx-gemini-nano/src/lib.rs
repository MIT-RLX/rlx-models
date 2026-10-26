//! Gemini Nano (Chrome's on-device model) for RLX.
//!
//! Google has not published Gemini Nano. The only copy is the TFLite file
//! Chrome downloads when its built-in AI is enabled:
//! `<Chrome user data>/OptGuideOnDeviceModel/<version>/weights.bin`.
//! [`locate_chrome_weights`] finds it on macOS, Linux and Windows.
//!
//! This crate currently provides the container layer: a bounds-checked TFLite
//! reader ([`tflite`]), a schema-less protobuf decoder for the architecture
//! metadata MediaPipe embeds ([`proto`]) and a manifest dump ([`inspect`]).
//! The transformer graph is wired once the tensor manifest of a real
//! `weights.bin` has been read.

pub mod inspect;
pub mod proto;
pub mod tflite;

use std::path::{Path, PathBuf};

pub use tflite::{TensorInfo, TensorType, TfliteModel};

/// Chrome user-data directories to search, most specific channel first.
fn chrome_user_data_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if let Some(home) = &home {
        let mac = home.join("Library/Application Support");
        for name in [
            "Google/Chrome",
            "Google/Chrome Canary",
            "Google/Chrome Beta",
            "Google/Chrome Dev",
            "Chromium",
        ] {
            dirs.push(mac.join(name));
        }
        let linux = home.join(".config");
        for name in [
            "google-chrome",
            "google-chrome-beta",
            "google-chrome-unstable",
            "chromium",
        ] {
            dirs.push(linux.join(name));
        }
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) {
        for name in [
            "Google/Chrome/User Data",
            "Google/Chrome SxS/User Data",
            "Google/Chrome Beta/User Data",
            "Google/Chrome Dev/User Data",
            "Chromium/User Data",
        ] {
            dirs.push(local.join(name));
        }
    }
    dirs
}

/// `weights.bin` files under `<user data>/OptGuideOnDeviceModel/*/`, newest
/// version directory first within each user-data directory.
pub fn weights_in(user_data: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(user_data.join("OptGuideOnDeviceModel")) else {
        return Vec::new();
    };
    let mut found: Vec<(Vec<u64>, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let w = e.path().join("weights.bin");
            w.is_file().then(|| {
                let version = e
                    .file_name()
                    .to_string_lossy()
                    .split('.')
                    .map(|p| p.parse().unwrap_or(0))
                    .collect();
                (version, w)
            })
        })
        .collect();
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.into_iter().map(|(_, p)| p).collect()
}

/// Every Gemini Nano `weights.bin` Chrome has downloaded on this machine.
pub fn locate_chrome_weights() -> Vec<PathBuf> {
    chrome_user_data_dirs()
        .iter()
        .flat_map(|d| weights_in(d))
        .collect()
}
