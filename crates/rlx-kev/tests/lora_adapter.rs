//! Reading a peft adapter off disk and folding it into base weights.
//!
//! The orientation is the thing under test. peft stores `lora_A` as `[r, in]`
//! and `lora_B` as `[out, r]`; RLX keeps matmul weights as `[out, in]`. Fold
//! the delta the other way round and the model still loads, still runs, and
//! still returns a probability distribution — just the wrong one. So the
//! merge is checked against a hand-computed `B · A`, and the loader path is
//! exercised through a real `.safetensors` file rather than a constructed
//! struct.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rlx_core::weight_loader::WeightLoader;
use rlx_kev::lora::{LoraAdapter, module_path_for_gguf};
use rlx_kev::weights::LoraMergingLoader;

/// A scratch directory that removes itself.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        p.push(format!("rlx-kev-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("create temp dir");
        Self(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Write a minimal peft adapter: `q_proj` gets rank 1 on a `[2, 3]` weight.
fn write_adapter(dir: &Path, config_extra: &str) {
    let cfg = format!(
        r#"{{
  "r": 1,
  "lora_alpha": 2,
  "target_modules": ["q_proj"],
  "use_rslora": false,
  "use_dora": false,
  "trainable_token_indices": null{config_extra}
}}"#
    );
    std::fs::write(dir.join("adapter_config.json"), cfg).expect("write config");

    // A = [[1, 0, -1]]  (r=1, in=3)      B = [[1], [2]]  (out=2, r=1)
    let a = f32_bytes(&[1.0, 0.0, -1.0]);
    let b = f32_bytes(&[1.0, 2.0]);
    let tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::from([
        (
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1, 3], &a)
                .expect("view A"),
        ),
        (
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 1], &b)
                .expect("view B"),
        ),
    ]);
    let bytes = safetensors::serialize(&tensors, None).expect("serialize");
    std::fs::write(dir.join("adapter_model.safetensors"), bytes).expect("write adapter");
}

#[test]
fn reads_a_peft_adapter_and_folds_b_times_a() {
    let dir = TempDir::new("adapter");
    write_adapter(dir.path(), "");

    let ad = LoraAdapter::open(dir.path()).expect("opens");
    assert_eq!(ad.r, 1);
    assert_eq!(ad.alpha, 2.0);
    assert_eq!(ad.scale(), 2.0, "alpha / r");
    assert_eq!(ad.len(), 1);

    let module = "layers.0.self_attn.q_proj";
    assert!(
        ad.get(module).is_some(),
        "module keyed by its stripped path"
    );

    // Base [out=2, in=3] of zeros; the fold must leave scale * B·A.
    let mut w = vec![0.0f32; 6];
    assert!(ad.merge_into(module, &mut w, 2, 3).expect("merge"));
    assert_eq!(w, vec![2.0, 0.0, -2.0, 4.0, 0.0, -4.0]);

    // An untargeted module is a no-op, not an error.
    assert!(
        !ad.merge_into("layers.0.mlp.up_proj", &mut w, 2, 3)
            .expect("no-op")
    );
}

#[test]
fn a_shape_disagreement_is_an_error_not_a_silent_skip() {
    let dir = TempDir::new("shape");
    write_adapter(dir.path(), "");
    let ad = LoraAdapter::open(dir.path()).expect("opens");
    let mut w = vec![0.0f32; 12];
    let err = ad
        .merge_into("layers.0.self_attn.q_proj", &mut w, 3, 4)
        .expect_err("must refuse");
    assert!(err.to_string().contains("adapter is [2, 3]"), "{err}");
}

/// A token-trained adapter carries embedding rows a weight merge cannot
/// represent. kev refuses to merge those; so must we.
#[test]
fn trainable_token_indices_blocks_the_merge() {
    let dir = TempDir::new("tokens");
    write_adapter(dir.path(), r#", "x": 1"#);
    let cfg = std::fs::read_to_string(dir.path().join("adapter_config.json")).expect("read");
    let cfg = cfg.replace(
        r#""trainable_token_indices": null"#,
        r#""trainable_token_indices": {"embed_tokens": [1, 2]}"#,
    );
    std::fs::write(dir.path().join("adapter_config.json"), cfg).expect("write");

    let err = LoraAdapter::open(dir.path()).expect_err("must refuse");
    assert!(err.to_string().contains("trainable_token_indices"), "{err}");
}

#[test]
fn an_unpaired_adapter_tensor_is_rejected() {
    let dir = TempDir::new("orphan");
    write_adapter(dir.path(), "");
    // Rewrite the file with only the A side.
    let a = f32_bytes(&[1.0, 0.0, -1.0]);
    let tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::from([(
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1, 3], &a)
            .expect("view"),
    )]);
    let bytes = safetensors::serialize(&tensors, None).expect("serialize");
    std::fs::write(dir.path().join("adapter_model.safetensors"), bytes).expect("write");

    let err = LoraAdapter::open(dir.path()).expect_err("must refuse");
    assert!(err.to_string().contains("no lora_B"), "{err}");
}

/// A loader that serves one known tensor, so the merging wrapper can be
/// driven end to end under its real trait.
struct OneTensorLoader {
    key: String,
    data: Vec<f32>,
    shape: Vec<usize>,
    taken: bool,
}

impl WeightLoader for OneTensorLoader {
    fn format_id(&self) -> &'static str {
        "test"
    }
    fn len(&self) -> usize {
        usize::from(!self.taken)
    }
    fn take(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        if key != self.key || self.taken {
            anyhow::bail!("no such tensor: {key}");
        }
        self.taken = true;
        Ok((self.data.clone(), self.shape.clone()))
    }
    fn take_transposed(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (d, s) = self.take(key)?;
        let (r, c) = (s[0], s[1]);
        let mut t = vec![0f32; d.len()];
        for i in 0..r {
            for j in 0..c {
                t[j * r + i] = d[i * c + j];
            }
        }
        Ok((t, vec![c, r]))
    }
    fn remaining_keys(&self) -> Vec<String> {
        if self.taken {
            vec![]
        } else {
            vec![self.key.clone()]
        }
    }
}

/// The merge must happen against the GGUF-named tensor the model actually
/// asks for, via the shared name map.
#[test]
fn the_merging_loader_folds_on_take_by_gguf_name() {
    let dir = TempDir::new("loader");
    write_adapter(dir.path(), "");
    let ad = LoraAdapter::open(dir.path()).expect("opens");

    assert_eq!(
        module_path_for_gguf("blk.0.attn_q.weight").as_deref(),
        Some("layers.0.self_attn.q_proj"),
    );

    let inner = OneTensorLoader {
        key: "blk.0.attn_q.weight".into(),
        data: vec![0.0; 6],
        shape: vec![2, 3],
        taken: false,
    };
    let mut loader = LoraMergingLoader::new(inner, ad);
    let (w, shape) = loader.take("blk.0.attn_q.weight").expect("take");
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(w, vec![2.0, 0.0, -2.0, 4.0, 0.0, -4.0]);
    assert_eq!(loader.merged_count(), 1);
    assert!(loader.unmerged().is_empty());
}

/// `take_transposed` must fold in `[out, in]` and transpose afterwards. Doing
/// it the other way around is the silent-wrong case this guards.
#[test]
fn the_merging_loader_transposes_after_folding() {
    let dir = TempDir::new("loader-t");
    write_adapter(dir.path(), "");
    let ad = LoraAdapter::open(dir.path()).expect("opens");

    let inner = OneTensorLoader {
        key: "blk.0.attn_q.weight".into(),
        data: vec![0.0; 6],
        shape: vec![2, 3],
        taken: false,
    };
    let mut loader = LoraMergingLoader::new(inner, ad);
    let (w, shape) = loader
        .take_transposed("blk.0.attn_q.weight")
        .expect("take_transposed");
    assert_eq!(shape, vec![3, 2]);
    // Transpose of [[2, 0, -2], [4, 0, -4]].
    assert_eq!(w, vec![2.0, 4.0, 0.0, 0.0, -2.0, -4.0]);
}

/// An adapter whose modules never find a base tensor would leave those layers
/// running bare. That has to be loud.
#[test]
fn an_adapter_module_that_never_merges_is_reported() {
    let dir = TempDir::new("unmerged");
    write_adapter(dir.path(), "");
    let ad = LoraAdapter::open(dir.path()).expect("opens");

    let inner = OneTensorLoader {
        key: "blk.0.ffn_up.weight".into(),
        data: vec![0.0; 6],
        shape: vec![2, 3],
        taken: false,
    };
    let mut loader = LoraMergingLoader::new(inner, ad);
    let _ = loader.take("blk.0.ffn_up.weight").expect("take");
    assert_eq!(loader.merged_count(), 0);
    assert_eq!(loader.unmerged(), vec!["layers.0.self_attn.q_proj"]);
}
