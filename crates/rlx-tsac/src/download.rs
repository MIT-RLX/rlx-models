use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
#[cfg(feature = "fetch")]
use std::process::Command;

pub const TSAC_TARBALL_URL: &str = "https://bellard.org/tsac/tsac-2024-04-08.tar.gz";
pub const TSAC_VERSION_DIR: &str = "tsac-2024-04-08";

const WEIGHT_FILES: &[&str] = &[
    "dac_mono_q8.bin",
    "dac_stereo_q8.bin",
    "tsac_mono_q8.bin",
    "tsac_stereo_q8.bin",
];

pub fn default_tsac_dir() -> PathBuf {
    std::env::var("RLX_TSAC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/tsac"))
}

pub fn resolve_install_dir(explicit: Option<&Path>) -> PathBuf {
    explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(default_tsac_dir)
}

pub fn resolve_tsac_bin(install_dir: &Path) -> PathBuf {
    std::env::var("RLX_TSAC_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| install_dir.join("tsac"))
}

pub fn weights_available(dir: &Path) -> bool {
    WEIGHT_FILES.iter().all(|name| dir.join(name).is_file())
}

pub fn install_complete(dir: &Path) -> bool {
    weights_available(dir) && resolve_tsac_bin(dir).is_file()
}

pub fn dac_model_path(install_dir: &Path, stereo: bool) -> PathBuf {
    if stereo {
        install_dir.join("dac_stereo_q8.bin")
    } else {
        install_dir.join("dac_mono_q8.bin")
    }
}

#[cfg(feature = "fetch")]
pub fn fetch_tsac(out_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let tarball = out_dir.join("tsac.tar.gz");
    eprintln!("downloading {TSAC_TARBALL_URL} …");
    download_url(TSAC_TARBALL_URL, &tarball)?;
    extract_tarball(&tarball, out_dir)?;
    flatten_extracted_dir(out_dir)?;
    std::fs::remove_file(&tarball).ok();
    ensure_tsac(out_dir)?;
    eprintln!("installed TSAC under {}", out_dir.display());
    Ok(out_dir.to_path_buf())
}

#[cfg(not(feature = "fetch"))]
pub fn fetch_tsac(_out_dir: &Path) -> Result<PathBuf> {
    bail!("rebuild with feature `fetch` to download TSAC")
}

pub fn ensure_tsac(install_dir: &Path) -> Result<()> {
    if weights_available(install_dir) {
        #[cfg(unix)]
        if resolve_tsac_bin(install_dir).is_file() {
            make_executable(&resolve_tsac_bin(install_dir))?;
        }
        return Ok(());
    }
    #[cfg(feature = "fetch")]
    {
        fetch_tsac(install_dir)?;
        return Ok(());
    }
    #[cfg(not(feature = "fetch"))]
    {
        bail!(
            "missing TSAC weights under {} — set RLX_TSAC_DIR or run with --fetch (needs `fetch` feature)",
            install_dir.display()
        )
    }
}

#[cfg(feature = "fetch")]
fn download_url(url: &str, dest: &Path) -> Result<()> {
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let mut reader = resp.into_reader();
    let mut file =
        std::fs::File::create(dest).with_context(|| format!("create {}", dest.display()))?;
    std::io::copy(&mut reader, &mut file).context("write tarball")?;
    Ok(())
}

#[cfg(feature = "fetch")]
fn extract_tarball(tarball: &Path, out_dir: &Path) -> Result<()> {
    let status = Command::new("tar")
        .arg("xzf")
        .arg(tarball)
        .arg("-C")
        .arg(out_dir)
        .status()
        .context("run tar (is `tar` installed?)")?;
    if !status.success() {
        bail!("tar extract failed with status {status}");
    }
    Ok(())
}

#[cfg(feature = "fetch")]
fn flatten_extracted_dir(out_dir: &Path) -> Result<()> {
    let inner = out_dir.join(TSAC_VERSION_DIR);
    if !inner.is_dir() {
        bail!(
            "expected `{}/` in tarball — got unexpected layout",
            TSAC_VERSION_DIR
        );
    }
    for entry in std::fs::read_dir(&inner)? {
        let entry = entry?;
        let src = entry.path();
        let name = entry.file_name();
        let dest = out_dir.join(name);
        if dest.exists() {
            if dest.is_dir() {
                std::fs::remove_dir_all(&dest)?;
            } else {
                std::fs::remove_file(&dest)?;
            }
        }
        std::fs::rename(&src, &dest)
            .with_context(|| format!("move {} -> {}", src.display(), dest.display()))?;
    }
    std::fs::remove_dir(&inner).ok();
    Ok(())
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let mut perms = meta.permissions();
    perms.set_mode(perms.mode() | 0o111);
    std::fs::set_permissions(path, perms).with_context(|| format!("chmod {}", path.display()))?;
    Ok(())
}
