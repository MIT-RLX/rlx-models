//! Run Bellard `tsac` and standalone `tsac-ng` inside a Docker reference image.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const DEFAULT_REF_IMAGE: &str = "rlx-tsac-ref";
pub const DEFAULT_REF_PLATFORM: &str = "linux/amd64";

#[derive(Debug, Clone)]
pub struct DockerRefOptions {
    pub image: String,
    pub platform: String,
}

impl Default for DockerRefOptions {
    fn default() -> Self {
        Self {
            image: std::env::var("RLX_TSAC_REF_IMAGE").unwrap_or_else(|_| DEFAULT_REF_IMAGE.into()),
            platform: std::env::var("RLX_TSAC_DOCKER_PLATFORM")
                .unwrap_or_else(|_| DEFAULT_REF_PLATFORM.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RefRoundtrip {
    pub engine: String,
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub output_bytes: u64,
    pub tsac_path: PathBuf,
    pub wav_path: PathBuf,
}

pub fn docker_ref_available(image: &str) -> bool {
    Command::new("docker")
        .args(["image", "inspect", image])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn run_docker_ref_roundtrip(
    opts: &DockerRefOptions,
    engine: &str,
    in_wav: &Path,
    work_dir: &Path,
    quality: u8,
    fast: bool,
) -> Result<RefRoundtrip> {
    if !docker_ref_available(&opts.image) {
        bail!(
            "Docker image `{}` not found — build with: bash crates/rlx-tsac/docker/run.sh build",
            opts.image
        );
    }
    if !in_wav.is_file() {
        bail!("input wav missing: {}", in_wav.display());
    }
    std::fs::create_dir_all(work_dir).with_context(|| format!("create {}", work_dir.display()))?;

    let container_wav = format!("/data/{}", in_wav.file_name().unwrap().to_string_lossy());
    let host_in = work_dir.join(in_wav.file_name().unwrap());
    if host_in != in_wav {
        std::fs::copy(in_wav, &host_in)
            .with_context(|| format!("copy {} -> {}", in_wav.display(), host_in.display()))?;
    }

    let fast_flag = if fast { "1" } else { "0" };
    let output = Command::new("docker")
        .args([
            "run",
            "--rm",
            "--platform",
            &opts.platform,
            "-v",
            &format!("{}:/data", work_dir.display()),
            &opts.image,
            engine,
            "/data",
            &container_wav,
            &quality.to_string(),
            fast_flag,
        ])
        .output()
        .context("docker run reference bench")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "docker reference bench failed (status {})\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let kv = parse_kv_lines(&stdout);
    let encode_ms = kv
        .get("ENCODE_MS")
        .and_then(|s| s.parse().ok())
        .context("missing ENCODE_MS from docker bench")?;
    let decode_ms = kv
        .get("DECODE_MS")
        .and_then(|s| s.parse().ok())
        .context("missing DECODE_MS from docker bench")?;
    let output_bytes = kv
        .get("BYTES")
        .and_then(|s| s.parse().ok())
        .context("missing BYTES from docker bench")?;
    let tsac_path = kv
        .get("TSAC_PATH")
        .map(PathBuf::from)
        .context("missing TSAC_PATH from docker bench")?;
    let wav_path = kv
        .get("WAV_PATH")
        .map(PathBuf::from)
        .context("missing WAV_PATH from docker bench")?;

    let tsac_path = remap_container_path(&tsac_path, work_dir);
    let wav_path = remap_container_path(&wav_path, work_dir);

    Ok(RefRoundtrip {
        engine: engine.to_string(),
        encode_ms,
        decode_ms,
        output_bytes,
        tsac_path,
        wav_path,
    })
}

fn remap_container_path(container: &Path, work_dir: &Path) -> PathBuf {
    container
        .strip_prefix("/data")
        .map(|rel| work_dir.join(rel))
        .unwrap_or_else(|_| work_dir.join(container.file_name().unwrap()))
}

fn parse_kv_lines(stdout: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in stdout.lines() {
        if let Some((k, v)) = line.split_once('=') {
            map.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    map
}
