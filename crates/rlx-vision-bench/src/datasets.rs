// RLX models — distributed vision training.
// SPDX-License-Identifier: GPL-3.0-only

//! Dataset loaders behind one shape-agnostic [`Split`] + [`DataSpec`], so the
//! same model/training/distributed code trains on any of them.
//!
//! | kind | shape (C×H×W) | classes | source |
//! |------|--------------|---------|--------|
//! | `Mnist` | 1×28×28 | 10 | idx-ubyte download (real) |
//! | `FashionMnist` | 1×28×28 | 10 | idx-ubyte download (real) |
//! | `Cifar10` | 3×32×32 | 10 | CIFAR binary download (real) |
//! | `Cifar100` | 3×32×32 | 100 | CIFAR binary download (real) |
//! | `ImageNet` | 3×224×224 | 1000 | local dir if `RLX_IMAGENET_DIR`, else **synthetic** |
//! | `Coco` | 3×640×640 | 80 | local dir if `RLX_COCO_DIR`, else **synthetic** |
//!
//! Images are stored per-sample as `pixels()` f32 in **`[C,H,W]` row-major**,
//! normalized to `[-1, 1]` — the layout the CNN reshapes to `[B,C,H,W]` and the
//! MLP flattens. Labels are `f32` (what `SoftmaxCrossEntropyWithLogits` wants).
//! ImageNet/COCO default to deterministic **synthetic** data — a real 150 GB
//! download isn't a unit-test dependency; they exist to exercise a *config*
//! (input size, class count, model FLOPs, device, transport), not accuracy.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Input geometry + label count of a dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataSpec {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub classes: usize,
}

impl DataSpec {
    /// Elements per image (`c·h·w`).
    pub const fn pixels(&self) -> usize {
        self.c * self.h * self.w
    }
}

/// One split (train or test). Carries its [`DataSpec`], so downstream code
/// reads geometry (pixels/classes/h/w/c) straight off the split.
pub struct Split {
    pub images: Vec<f32>,
    pub labels: Vec<f32>,
    pub spec: DataSpec,
}

impl Split {
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
    pub fn pixels(&self) -> usize {
        self.spec.pixels()
    }
    pub fn image(&self, i: usize) -> &[f32] {
        let px = self.spec.pixels();
        &self.images[i * px..(i + 1) * px]
    }
    pub fn label(&self, i: usize) -> usize {
        self.labels[i] as usize
    }
    /// Keep only the first `n` samples (used to bound synthetic / quick runs).
    fn truncate(&mut self, n: usize) {
        if self.len() > n {
            self.labels.truncate(n);
            self.images.truncate(n * self.spec.pixels());
        }
    }
}

/// A loaded dataset: two splits + its spec + a display name.
pub struct Data {
    pub train: Split,
    pub test: Split,
    pub spec: DataSpec,
    pub name: &'static str,
    /// `true` when the samples are synthetic (accuracy is not meaningful).
    pub synthetic: bool,
}

/// Which dataset to load.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DatasetKind {
    Mnist,
    FashionMnist,
    Cifar10,
    Cifar100,
    ImageNet,
    Coco,
}

impl DatasetKind {
    pub fn spec(self) -> DataSpec {
        match self {
            DatasetKind::Mnist | DatasetKind::FashionMnist => DataSpec {
                h: 28,
                w: 28,
                c: 1,
                classes: 10,
            },
            DatasetKind::Cifar10 => DataSpec {
                h: 32,
                w: 32,
                c: 3,
                classes: 10,
            },
            DatasetKind::Cifar100 => DataSpec {
                h: 32,
                w: 32,
                c: 3,
                classes: 100,
            },
            DatasetKind::ImageNet => DataSpec {
                h: 224,
                w: 224,
                c: 3,
                classes: 1000,
            },
            DatasetKind::Coco => DataSpec {
                h: 640,
                w: 640,
                c: 3,
                classes: 80,
            },
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DatasetKind::Mnist => "mnist",
            DatasetKind::FashionMnist => "fashion-mnist",
            DatasetKind::Cifar10 => "cifar10",
            DatasetKind::Cifar100 => "cifar100",
            DatasetKind::ImageNet => "imagenet",
            DatasetKind::Coco => "coco",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "mnist" => DatasetKind::Mnist,
            "fashion" | "fashion-mnist" | "fashionmnist" => DatasetKind::FashionMnist,
            "cifar10" | "cifar-10" => DatasetKind::Cifar10,
            "cifar100" | "cifar-100" => DatasetKind::Cifar100,
            "imagenet" => DatasetKind::ImageNet,
            "coco" => DatasetKind::Coco,
            _ => return None,
        })
    }

    pub fn all() -> &'static [DatasetKind] {
        use DatasetKind::*;
        &[Mnist, FashionMnist, Cifar10, Cifar100, ImageNet, Coco]
    }
}

/// Load a dataset. `max_train` / `max_test` cap the sample counts (applied after
/// load) — essential for the big synthetic sets and quick harness runs.
pub fn load(
    kind: DatasetKind,
    max_train: Option<usize>,
    max_test: Option<usize>,
) -> Result<Data, String> {
    let spec = kind.spec();
    let ((mut train, mut test), synthetic) = match kind {
        DatasetKind::Mnist => (load_idx(kind, spec)?, false),
        DatasetKind::FashionMnist => (load_idx(kind, spec)?, false),
        DatasetKind::Cifar10 => (load_cifar(kind, spec)?, false),
        DatasetKind::Cifar100 => (load_cifar(kind, spec)?, false),
        DatasetKind::ImageNet => {
            let (tr, te, syn) = load_local_or_synthetic(kind, spec, "RLX_IMAGENET_DIR", 4096, 512);
            ((tr, te), syn)
        }
        DatasetKind::Coco => {
            let (tr, te, syn) = load_local_or_synthetic(kind, spec, "RLX_COCO_DIR", 1024, 256);
            ((tr, te), syn)
        }
    };
    if let Some(n) = max_train {
        train.truncate(n);
    }
    if let Some(n) = max_test {
        test.truncate(n);
    }
    Ok(Data {
        train,
        test,
        spec,
        name: kind.name(),
        synthetic,
    })
}

fn cache_root() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(format!("{home}/.cache/rlx-datasets"))
}

// ---- idx-ubyte (MNIST / Fashion-MNIST) -----------------------------------

fn idx_source(kind: DatasetKind) -> (PathBuf, &'static str, [&'static str; 4]) {
    let files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ];
    match kind {
        DatasetKind::Mnist => {
            // Reuse torchvision's cache layout so a prior MNIST download is found.
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            (
                PathBuf::from(format!("{home}/.cache/torchvision-mnist/MNIST/raw")),
                "https://ossci-datasets.s3.amazonaws.com/mnist",
                files,
            )
        }
        DatasetKind::FashionMnist => (
            cache_root().join("fashion-mnist"),
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion",
            files,
        ),
        _ => unreachable!("idx_source called for non-idx dataset"),
    }
}

fn load_idx(kind: DatasetKind, spec: DataSpec) -> Result<(Split, Split), String> {
    let (dir, mirror, files) = idx_source(kind);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    for f in files {
        ensure_gz(&dir, mirror, f)?;
    }
    Ok((
        idx_split(&dir.join(files[0]), &dir.join(files[1]), spec)?,
        idx_split(&dir.join(files[2]), &dir.join(files[3]), spec)?,
    ))
}

/// Download `<mirror>/<file>.gz` and `gunzip` it to `<dir>/<file>` if missing.
fn ensure_gz(dir: &Path, mirror: &str, file: &str) -> Result<(), String> {
    let out = dir.join(file);
    if out.exists() {
        return Ok(());
    }
    let gz = dir.join(format!("{file}.gz"));
    let url = format!("{mirror}/{file}.gz");
    download(&[url.as_str()], &gz)?;
    let ok = Command::new("gunzip")
        .args(["-f", gz.to_str().unwrap()])
        .status()
        .map_err(|e| format!("gunzip: {e}"))?
        .success();
    if !ok || !out.exists() {
        return Err(format!("gunzip failed: {}", gz.display()));
    }
    Ok(())
}

fn idx_split(imgs: &Path, lbls: &Path, spec: DataSpec) -> Result<Split, String> {
    let px = spec.pixels();
    let images = read_idx_images(imgs, px)?;
    let labels = read_idx_labels(lbls)?;
    if images.len() / px != labels.len() {
        return Err("image/label count mismatch".into());
    }
    Ok(Split {
        images,
        labels,
        spec,
    })
}

fn read_idx_images(path: &Path, pixels: usize) -> Result<Vec<f32>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if raw.len() < 16 {
        return Err(format!("{}: short header", path.display()));
    }
    if u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]) != 0x0803 {
        return Err(format!("{}: bad image magic", path.display()));
    }
    let n = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    let want = n * pixels;
    let body = &raw[16..];
    if body.len() < want {
        return Err(format!("{}: truncated", path.display()));
    }
    Ok(body[..want]
        .iter()
        .map(|&b| (b as f32 / 255.0) * 2.0 - 1.0)
        .collect())
}

fn read_idx_labels(path: &Path) -> Result<Vec<f32>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if raw.len() < 8 {
        return Err(format!("{}: short header", path.display()));
    }
    if u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]) != 0x0801 {
        return Err(format!("{}: bad label magic", path.display()));
    }
    let n = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    if raw.len() < 8 + n {
        return Err(format!("{}: truncated", path.display()));
    }
    Ok(raw[8..8 + n].iter().map(|&b| b as f32).collect())
}

// ---- CIFAR-10 / CIFAR-100 (binary) ---------------------------------------

fn load_cifar(kind: DatasetKind, spec: DataSpec) -> Result<(Split, Split), String> {
    let px = spec.pixels(); // 3*32*32 = 3072 — the on-disk image byte count too
    let (urls, subdir, label_bytes, train_files, test_file): (
        Vec<&str>,
        &str,
        usize,
        Vec<&str>,
        &str,
    ) = match kind {
        DatasetKind::Cifar10 => (
            // Reliable mirror first (HF hosts only Parquet, not this raw tarball),
            // canonical cs.toronto.edu as fallback.
            vec![
                "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-binary.tar.gz",
                "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
            ],
            "cifar-10-batches-bin",
            1,
            vec![
                "data_batch_1.bin",
                "data_batch_2.bin",
                "data_batch_3.bin",
                "data_batch_4.bin",
                "data_batch_5.bin",
            ],
            "test_batch.bin",
        ),
        DatasetKind::Cifar100 => (
            vec![
                "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-binary.tar.gz",
                "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
            ],
            "cifar-100-binary",
            2, // [coarse, fine]; we use the fine label (2nd byte)
            vec!["train.bin"],
            "test.bin",
        ),
        _ => unreachable!(),
    };

    let dir = cache_root().join(kind.name());
    let extracted = dir.join(subdir);
    if !extracted.join(test_file).exists() {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let tgz = dir.join("archive.tar.gz");
        if !tgz.exists() {
            download(&urls, &tgz)?;
        }
        let ok = Command::new("tar")
            .args(["xzf", tgz.to_str().unwrap(), "-C", dir.to_str().unwrap()])
            .status()
            .map_err(|e| format!("tar: {e}"))?
            .success();
        if !ok || !extracted.join(test_file).exists() {
            return Err(format!("tar extract failed for {}", kind.name()));
        }
    }

    let rec = label_bytes + px; // bytes per record
    let train = cifar_split(
        &train_files
            .iter()
            .map(|f| extracted.join(f))
            .collect::<Vec<_>>(),
        rec,
        label_bytes,
        spec,
    )?;
    let test = cifar_split(&[extracted.join(test_file)], rec, label_bytes, spec)?;
    Ok((train, test))
}

/// Parse CIFAR binary records: `[label_bytes | 3072 image bytes]`, image bytes
/// are `[R(1024) G(1024) B(1024)]` = `[C,H,W]` row-major, exactly our layout.
fn cifar_split(
    files: &[PathBuf],
    rec: usize,
    label_bytes: usize,
    spec: DataSpec,
) -> Result<Split, String> {
    let mut images = Vec::new();
    let mut labels = Vec::new();
    for path in files {
        let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if raw.len() % rec != 0 {
            return Err(format!("{}: size not a multiple of record", path.display()));
        }
        for chunk in raw.chunks_exact(rec) {
            // CIFAR-100: byte 0 = coarse, byte 1 = fine → use the last label byte.
            labels.push(chunk[label_bytes - 1] as f32);
            images.extend(
                chunk[label_bytes..]
                    .iter()
                    .map(|&b| (b as f32 / 255.0) * 2.0 - 1.0),
            );
        }
    }
    Ok(Split {
        images,
        labels,
        spec,
    })
}

// ---- ImageNet / COCO: local dir or synthetic -----------------------------

/// Load raw preprocessed tensors from `<env dir>/{train,test}.{images,labels}`
/// (f32 `[N,C,H,W]` + `u8`/`f32` labels) if the env var points at them, else
/// deterministic synthetic data of the right shape.
fn load_local_or_synthetic(
    kind: DatasetKind,
    spec: DataSpec,
    env: &str,
    n_train: usize,
    n_test: usize,
) -> (Split, Split, bool) {
    if let Ok(dir) = std::env::var(env) {
        match load_raw_dir(Path::new(&dir), spec) {
            Ok((tr, te)) => {
                eprintln!("  {}: loaded raw tensors from {env}={dir}", kind.name());
                return (tr, te, false);
            }
            Err(e) => eprintln!(
                "  {}: {env}={dir} not usable ({e}); using synthetic",
                kind.name()
            ),
        }
    }
    eprintln!(
        "  {}: no {env} set — using SYNTHETIC {}×{}×{} data ({} classes) for config/throughput testing",
        kind.name(),
        spec.c,
        spec.h,
        spec.w,
        spec.classes
    );
    (
        synthetic_split(spec, n_train, 0xA5A5),
        synthetic_split(spec, n_test, 0x5A5A),
        true,
    )
}

/// Raw layout: `<dir>/train.images` (f32, `N·pixels`), `<dir>/train.labels`
/// (f32, `N`), and the same for `test`. A minimal escape hatch to feed real
/// (pre-decoded, pre-resized) images without an image-codec dependency.
fn load_raw_dir(dir: &Path, spec: DataSpec) -> Result<(Split, Split), String> {
    let read_f32 = |p: PathBuf| -> Result<Vec<f32>, String> {
        let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        if b.len() % 4 != 0 {
            return Err(format!("{}: not f32-aligned", p.display()));
        }
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    let mk = |stem: &str| -> Result<Split, String> {
        let images = read_f32(dir.join(format!("{stem}.images")))?;
        let labels = read_f32(dir.join(format!("{stem}.labels")))?;
        let px = spec.pixels();
        if images.len() != labels.len() * px {
            return Err(format!(
                "{stem}: images {} != labels {} × {px}",
                images.len(),
                labels.len()
            ));
        }
        Ok(Split {
            images,
            labels,
            spec,
        })
    };
    Ok((mk("train")?, mk("test")?))
}

/// Deterministic pseudo-random split of the given shape — labels in
/// `0..classes`, images in `[-1, 1]`. Not learnable (random); for throughput /
/// config testing only.
fn synthetic_split(spec: DataSpec, n: usize, salt: u64) -> Split {
    let px = spec.pixels();
    let mut state = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = || -> u64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut images = Vec::with_capacity(n * px);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        labels.push((next() % spec.classes as u64) as f32);
        for _ in 0..px {
            images.push(((next() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0);
        }
    }
    Split {
        images,
        labels,
        spec,
    }
}

// ---- shared download -------------------------------------------------------

/// Download the first working mirror in `urls` to `out`. Robust against
/// slow/stalling mirrors: abort a transfer that drops below 10 KB/s for 30 s
/// (so `--retry` recovers it instead of hanging forever — cs.toronto.edu froze
/// cifar-100 at 0 KB/s mid-download), and fall through to the next mirror on
/// failure (cleaning any partial first).
fn download(urls: &[&str], out: &Path) -> Result<(), String> {
    let mut last = String::from("no mirrors provided");
    for &url in urls {
        eprintln!("  downloading {url}");
        let ok = Command::new("curl")
            .args([
                "-fsSL",
                "-o",
                out.to_str().unwrap(),
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--retry-all-errors",
                "--connect-timeout",
                "20",
                "--speed-limit",
                "10240",
                "--speed-time",
                "30",
                url,
            ])
            .status()
            .map_err(|e| format!("curl ({url}): {e}"))?
            .success();
        if ok {
            return Ok(());
        }
        last = format!("download failed: {url}");
        let _ = std::fs::remove_file(out); // clean partial before the next mirror
    }
    Err(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_are_consistent() {
        assert_eq!(DatasetKind::Mnist.spec().pixels(), 784);
        assert_eq!(DatasetKind::Cifar10.spec().pixels(), 3072);
        assert_eq!(DatasetKind::Cifar100.spec().classes, 100);
        assert_eq!(DatasetKind::ImageNet.spec().pixels(), 3 * 224 * 224);
        assert_eq!(
            DatasetKind::Coco.spec(),
            DataSpec {
                h: 640,
                w: 640,
                c: 3,
                classes: 80
            }
        );
    }

    #[test]
    fn parse_names() {
        assert_eq!(
            DatasetKind::parse("fashion"),
            Some(DatasetKind::FashionMnist)
        );
        assert_eq!(DatasetKind::parse("CIFAR-100"), Some(DatasetKind::Cifar100));
        assert_eq!(DatasetKind::parse("nope"), None);
    }

    #[test]
    fn synthetic_has_right_shape_and_range() {
        let spec = DataSpec {
            h: 8,
            w: 8,
            c: 3,
            classes: 5,
        };
        let s = synthetic_split(spec, 10, 1);
        assert_eq!(s.len(), 10);
        assert_eq!(s.images.len(), 10 * spec.pixels());
        assert_eq!(s.image(3).len(), spec.pixels());
        assert!(s.images.iter().all(|&v| (-1.0..=1.0).contains(&v)));
        assert!(s.labels.iter().all(|&l| (l as usize) < spec.classes));
    }

    #[test]
    fn truncate_bounds_samples() {
        let spec = DataSpec {
            h: 4,
            w: 4,
            c: 1,
            classes: 2,
        };
        let mut s = synthetic_split(spec, 100, 7);
        s.truncate(10);
        assert_eq!(s.len(), 10);
        assert_eq!(s.images.len(), 10 * spec.pixels());
    }
}
