// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// Env-gated: validate qwen35 GGUF tensor shapes vs parsed config.
//
//   QWEN35_GGUF_PATH=/path/to/model.gguf cargo test -p rlx-models qwen35_weight_shapes --release -- --nocapture

mod compile_support;

use rlx_gguf::GgufFile;
use rlx_models::qwen35::Qwen35Config;
use std::path::PathBuf;

#[test]
fn qwen35_weight_shapes_match_config() {
    let path = match std::env::var("QWEN35_GGUF_PATH") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => PathBuf::from("/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf"),
    };
    if !path.is_file() {
        eprintln!("skip qwen35_weight_shapes: missing {}", path.display());
        return;
    }

    let raw = GgufFile::from_path(&path).expect("open gguf");
    let cfg = Qwen35Config::from_gguf(&raw).expect("parse cfg");

    let n_state = cfg.ssm_state_size;
    let n_k = cfg.ssm_group_count;
    let n_v = cfg.ssm_time_step_rank;
    let key_dim = n_state * n_k;
    let value_dim_rlx = n_state * n_v;
    let value_dim_llama = cfg.ssm_inner_size;
    let conv_rlx = key_dim * 2 + value_dim_rlx;
    let conv_llama = value_dim_llama + 2 * key_dim;

    eprintln!(
        "cfg: hidden={} layers={} heads={} kv_heads={} key_len={} rope_dim={} rope_sections={:?}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.key_length,
        cfg.rope_dim_count,
        cfg.rope_dim_sections,
    );
    eprintln!(
        "ssm: state={} inner={} group={} dt_rank={} conv_k={} full_attn_iv={}",
        cfg.ssm_state_size,
        cfg.ssm_inner_size,
        cfg.ssm_group_count,
        cfg.ssm_time_step_rank,
        cfg.ssm_conv_kernel,
        cfg.full_attention_interval,
    );
    eprintln!(
        "dims: key_dim={key_dim} value_rlx={value_dim_rlx} value_llama={value_dim_llama} \
         conv_rlx={conv_rlx} conv_llama={conv_llama}"
    );

    let qkv = raw
        .tensors
        .get("blk.0.attn_qkv.weight")
        .expect("blk.0.attn_qkv");
    let conv = raw
        .tensors
        .get("blk.0.ssm_conv1d.weight")
        .expect("blk.0.ssm_conv1d");
    eprintln!(
        "blk.0.attn_qkv raw shape={:?} (ggml innermost-first)",
        qkv.shape
    );
    eprintln!("blk.0.ssm_conv1d raw shape={:?}", conv.shape);

    // After `GgufLoader::take` reversal: [out, in] for 2D mats.
    let qkv_shape = {
        let mut s = qkv.shape.clone();
        s.reverse();
        s
    };
    let conv_shape = {
        let mut s = conv.shape.clone();
        s.reverse();
        s
    };
    eprintln!("blk.0.attn_qkv normalized shape={qkv_shape:?}");
    eprintln!("blk.0.ssm_conv1d normalized shape={conv_shape:?}");

    assert_eq!(
        qkv_shape,
        vec![conv_rlx, cfg.hidden_size],
        "attn_qkv out dim should match 2*key_dim + value_dim"
    );
    assert_eq!(
        conv_shape,
        vec![conv_llama, cfg.ssm_conv_kernel],
        "ssm_conv1d should match llama conv_channels x kernel"
    );
    assert_eq!(
        value_dim_rlx, value_dim_llama,
        "RLX value_dim must match ssm_inner_size"
    );
}
