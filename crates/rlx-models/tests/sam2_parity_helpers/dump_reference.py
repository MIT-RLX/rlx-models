#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#
"""
Dump pytorch-reference SAM 2 activations to disk for parity comparison.

Invoked as a subprocess by `tests/sam2_parity.rs`. Subprocess approach
avoids pulling pyo3 + a system Python into the Rust build — we just
need a `python` on PATH with `sam2` + `torch` installed.

Required env vars (set by the Rust harness):
  RLX_SAM2_WEIGHTS    path to sam2_hiera_*.safetensors (or .pt)
  RLX_SAM2_CONFIG     reference config name (e.g. "sam2_hiera_b+")
  RLX_SAM2_IMAGE_BIN  path to a raw f32 NCHW image of shape [3,1024,1024]
                      (host-side preprocessed, ImageNet-normalized).
  RLX_SAM2_OUT_DIR    directory to write `.f32` outputs into.

Writes (each as a contiguous f32 LE blob, NCHW unless noted):
  encoder_stage_0.f32   [1, 112, 256, 256]   (B+: stage 0 BHWC)
  encoder_stage_1.f32   [1, 224, 128, 128]
  encoder_stage_2.f32   [1, 448,  64,  64]
  encoder_stage_3.f32   [1, 896,  32,  32]
  fpn_level_0.f32       [1, 256, 256, 256]   (stride 4)
  fpn_level_1.f32       [1, 256, 128, 128]
  fpn_level_2.f32       [1, 256,  64,  64]
  fpn_level_3.f32       [1, 256,  32,  32]

For the decoder path (optional second pass), set RLX_SAM2_RUN_DECODER=1
and additionally provide RLX_SAM2_POINTS=path to f32 [N,2] coords and
RLX_SAM2_LABELS=path to f32 [N] labels; the script then writes
`mask_logits.f32` and `iou_pred.f32`.
"""

import os
import sys
import numpy as np

def env(name: str) -> str:
    v = os.environ.get(name)
    if v is None:
        print(f"missing env var: {name}", file=sys.stderr)
        sys.exit(2)
    return v


def main() -> int:
    try:
        import torch
        from sam2.build_sam import build_sam2
    except ImportError as e:
        print(f"sam2 + torch must be installed: {e}", file=sys.stderr)
        return 3

    weights = env("RLX_SAM2_WEIGHTS")
    cfg_name = env("RLX_SAM2_CONFIG")  # e.g. "sam2_hiera_b+"
    image_bin = env("RLX_SAM2_IMAGE_BIN")
    out_dir = env("RLX_SAM2_OUT_DIR")

    os.makedirs(out_dir, exist_ok=True)

    img = np.fromfile(image_bin, dtype=np.float32)
    assert img.size == 3 * 1024 * 1024, f"image must be 3·1024·1024 f32, got {img.size}"
    img = img.reshape(1, 3, 1024, 1024)

    # `sam2.build_sam2` calls `torch.load(ckpt, weights_only=True)` and
    # then `state_dict["model"]`. The original sam2 .pt fails under
    # torch 2.6+ weights_only=True ("Unsupported operand 240"). The
    # Rust side has already converted to safetensors via
    # `pt_to_safetensors.py`, so accept either format here: if the
    # caller passed a safetensors path, hot-write a clean .pt with
    # the canonical `{"model": state_dict}` envelope (torch can
    # safely round-trip a plain tensor dict under weights_only=True).
    if weights.endswith(".safetensors"):
        from safetensors.torch import load_file
        import tempfile
        state = load_file(weights)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save({"model": state}, tmp.name)
            weights = tmp.name
    device = os.environ.get("RLX_SAM2_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "RLX_SAM2_DEVICE=cuda but torch reports no CUDA device — "
            "falling back to CPU (parity numbers are device-independent "
            "in fp32, but speed will suffer)",
            file=sys.stderr,
        )
        device = "cpu"
    model = build_sam2(f"configs/sam2/{cfg_name}.yaml", weights, device=device)
    model.eval()

    # All comparisons against Rust are in f32; force fp32 to keep
    # numerical envelope tight (autocast/bf16 would inflate diffs).
    with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
        x = torch.from_numpy(img).to(device).float()

        # Dump patch-embed-only intermediate (no position embedding).
        # Hiera's `patch_embed` is `Conv2d → permute(NCHW→NHWC)`.
        trunk = model.image_encoder.trunk
        patch = trunk.patch_embed(x)  # [1, H, W, E] NHWC
        np.asarray(patch.detach().cpu().numpy(), dtype=np.float32).tofile(
            os.path.join(out_dir, "patch_embed.f32")
        )

        # Dump post-pos-embed (patch + interp(pos) + tile(pos_window)).
        # Mirror trunk.forward: pos = trunk._get_pos_embed(patch.shape[1:3])
        pos = trunk._get_pos_embed(patch.shape[1:3])
        post_pos = patch + pos
        np.asarray(post_pos.detach().cpu().numpy(), dtype=np.float32).tofile(
            os.path.join(out_dir, "patch_plus_pos.f32")
        )

        # ── Per-substep dumps of block 0 (parity bisect) ──
        # Reproduce MultiScaleBlock.forward(block0) step-by-step so we
        # can pinpoint exactly which substep diverges.
        blk0 = trunk.blocks[0]
        x_b0 = post_pos
        shortcut0 = x_b0
        n1_out = blk0.norm1(x_b0)
        np.asarray(n1_out.detach().cpu().numpy(), dtype=np.float32).tofile(
            os.path.join(out_dir, "block0_post_norm1.f32")
        )
        # window_partition (block 0 of tiny: ws=8, no q_pool)
        from sam2.modeling.backbones.hieradet import window_partition, window_unpartition
        ws0 = blk0.window_size
        if ws0 > 0:
            wp, pad_hw0 = window_partition(n1_out, ws0)
        else:
            wp = n1_out
            pad_hw0 = (n1_out.shape[1], n1_out.shape[2])
        np.asarray(wp.detach().cpu().numpy(), dtype=np.float32).tofile(
            os.path.join(out_dir, "block0_post_partition.f32")
        )
        # attention forward
        attn0 = blk0.attn(wp)
        np.asarray(attn0.detach().cpu().numpy(), dtype=np.float32).tofile(
            os.path.join(out_dir, "block0_post_attn_windowed.f32")
        )
        # window_unpartition
        if ws0 > 0:
            attn0_unp = window_unpartition(attn0, ws0, pad_hw0, (x_b0.shape[1], x_b0.shape[2]))
        else:
            attn0_unp = attn0
        np.asarray(attn0_unp.detach().cpu().numpy(), dtype=np.float32).tofile(
            os.path.join(out_dir, "block0_post_attn_unpartitioned.f32")
        )
        post_attn_res0 = shortcut0 + attn0_unp
        np.asarray(post_attn_res0.detach().cpu().numpy(), dtype=np.float32).tofile(
            os.path.join(out_dir, "block0_post_attn_residual.f32")
        )
        n2_out = blk0.norm2(post_attn_res0)
        np.asarray(n2_out.detach().cpu().numpy(), dtype=np.float32).tofile(
            os.path.join(out_dir, "block0_post_norm2.f32")
        )
        mlp0 = blk0.mlp(n2_out)
        np.asarray(mlp0.detach().cpu().numpy(), dtype=np.float32).tofile(
            os.path.join(out_dir, "block0_post_mlp.f32")
        )

        # ── Block-1 substep dumps (only when block 1 is a q_pool block) ──
        # Tiny/small have q_pool at block 1; base+/large have it at
        # block 2. The Rust harness reads these blobs only for tiny
        # bisect (`if bisect_path.exists() { ... }`), so skip when
        # block 1 isn't a q_pool block.
        blk1 = trunk.blocks[1]
        if hasattr(blk1, "proj"):
            x_b1 = blk0(x_b0)
            np.asarray(x_b1.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "block1_input.f32")
            )
            n1_out_b1 = blk1.norm1(x_b1)
            np.asarray(n1_out_b1.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "block1_post_norm1.f32")
            )
            proj_out = blk1.proj(n1_out_b1)
            np.asarray(proj_out.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "block1_shortcut_pre_pool.f32")
            )
            from sam2.modeling.backbones.hieradet import do_pool
            shortcut_b1 = do_pool(proj_out, blk1.pool)
            np.asarray(shortcut_b1.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "block1_shortcut_pooled.f32")
            )
            ws1_in = blk1.window_size
            wp1, pad_hw1 = window_partition(n1_out_b1, ws1_in)
            np.asarray(wp1.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "block1_post_partition.f32")
            )
            attn_w1 = blk1.attn(wp1)
            np.asarray(attn_w1.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "block1_post_attn_windowed.f32")
            )
            if blk1.q_stride:
                ws1_out = blk1.window_size // blk1.q_stride[0]
                H_new = x_b1.shape[1] // blk1.q_stride[0]
                W_new = x_b1.shape[2] // blk1.q_stride[1]
                pad_h = (ws1_out - H_new % ws1_out) % ws1_out
                pad_w = (ws1_out - W_new % ws1_out) % ws1_out
                pad_hw1 = (H_new + pad_h, W_new + pad_w)
                attn_unp_1 = window_unpartition(
                    attn_w1, ws1_out, pad_hw1,
                    (H_new, W_new),
                )
            else:
                attn_unp_1 = window_unpartition(attn_w1, ws1_in, pad_hw1,
                    (x_b1.shape[1], x_b1.shape[2]))
            np.asarray(attn_unp_1.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "block1_post_attn_unpartitioned.f32")
            )
            post_attn_b1 = shortcut_b1 + attn_unp_1
            np.asarray(post_attn_b1.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "block1_post_attn_residual.f32")
            )
            n2_out_b1 = blk1.norm2(post_attn_b1)
            mlp_out_b1 = blk1.mlp(n2_out_b1)
            block1_out = post_attn_b1 + mlp_out_b1
            np.asarray(block1_out.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "block1_output.f32")
            )

        # SAM2Base.forward_image pre-applies the mask decoder's
        # `conv_s0` and `conv_s1` to backbone_fpn[0]/[1] (cached for
        # multiple clicks per image). For *encoder parity* we want the
        # raw 256-channel FpnNeck output, so call image_encoder
        # directly. The high-res mask-decoder path will get its own
        # parity test that does include conv_s0/s1.
        backbone_out = model.image_encoder(x)
        fpn = backbone_out["backbone_fpn"]
        for i, feat in enumerate(fpn):
            feat_np = feat.detach().cpu().numpy().astype(np.float32, copy=False)
            feat_np.tofile(os.path.join(out_dir, f"fpn_level_{i}.f32"))

        # Hiera intermediates (per-stage pre-neck outputs). Reference's
        # `Hiera.forward` returns `outputs` directly when run as a
        # backbone-only module; SAM2Base wraps it but the post-stage
        # tensors are accessible via the trunk.
        trunk_out = model.image_encoder.trunk(x)
        for i, feat in enumerate(trunk_out):
            feat_np = feat.detach().cpu().numpy().astype(np.float32, copy=False)
            feat_np.tofile(os.path.join(out_dir, f"encoder_stage_{i}.f32"))

    if os.environ.get("RLX_SAM2_RUN_DECODER") == "1":
        pts_path = env("RLX_SAM2_POINTS")
        lbl_path = env("RLX_SAM2_LABELS")
        pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 2)
        lbls = np.fromfile(lbl_path, dtype=np.float32)
        with torch.no_grad():
            # Bypass SAM2ImagePredictor entirely — its set_image() does
            # an u8 round-trip on the input image that introduces ~1e-3
            # quantization noise vs running forward_image() directly on
            # the same normalized tensor. Mirror set_image()'s post-
            # forward feature plumbing manually.
            backbone_out2 = model.forward_image(x)
            _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out2)
            # vision_feats are reshaped flat (S, B, C) per the reference
            # plumbing. Reconstruct the (B, C, H, W) view the decoder
            # expects, then take the last (stride-16) as image_embed
            # and the first two as high_res_feats.
            feat_sizes = [(256, 256), (128, 128), (64, 64)]
            B = x.shape[0]
            feats_bchw = [
                f.permute(1, 2, 0).view(B, -1, h_, w_)
                for f, (h_, w_) in zip(vision_feats[-3:], feat_sizes)
            ]
            image_embed = feats_bchw[-1]
            high_res_feats = feats_bchw[:-1]

            # Prompt encoder: pass normalized point coords directly.
            in_pts = torch.from_numpy(pts).float().unsqueeze(0)  # [1, N, 2]
            in_lbls = torch.from_numpy(lbls).float().unsqueeze(0)  # [1, N]
            # Normalize: pixel coords / 1024.
            in_pts_n = in_pts / 1024.0
            sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                points=(in_pts_n * 1024.0, in_lbls),  # prompt_encoder expects raw pixel coords
                boxes=None,
                masks=None,
            )
            # Dump these so the Rust side can compare prompt encoder too.
            np.asarray(sparse_embeddings.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "sparse_embeddings.f32")
            )
            np.asarray(dense_embeddings.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "dense_embeddings.f32")
            )

            # Dump the dense PE returned by get_dense_pe() — used by
            # the mask decoder as image_pe (the random-Fourier kind).
            ref_pe = model.sam_prompt_encoder.get_dense_pe()
            np.asarray(ref_pe.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "decoder_image_pe.f32")
            )
            np.asarray(image_embed.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "decoder_image_embed.f32")
            )
            # Bisect-friendly dumps: run the TwoWayTransformer alone first
            # to capture `hs` (the post-attention token stack) and `src`
            # (the post-attention image features). Mirrors mask_decoder
            # internals so we can compare each stage.
            tokens_for_dump = torch.cat(
                [
                    model.sam_mask_decoder.obj_score_token.weight,
                    model.sam_mask_decoder.iou_token.weight,
                    model.sam_mask_decoder.mask_tokens.weight,
                ],
                dim=0,
            ).unsqueeze(0).expand(sparse_embeddings.size(0), -1, -1)
            tokens_for_dump = torch.cat([tokens_for_dump, sparse_embeddings], dim=1)
            src_for_dump = image_embed + dense_embeddings
            pe_for_dump = model.sam_prompt_encoder.get_dense_pe()
            hs_dump, src_post_dump = model.sam_mask_decoder.transformer(
                src_for_dump, pe_for_dump, tokens_for_dump
            )
            np.asarray(hs_dump.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_hs.f32")
            )
            np.asarray(src_post_dump.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_src.f32")
            )

            # Per-step bisect of TwoWayTransformer.layer[0]. Mirror
            # forward() exactly.
            twt = model.sam_mask_decoder.transformer
            blk = twt.layers[0]
            image_emb_flat = image_embed + dense_embeddings
            image_emb_flat = image_emb_flat.flatten(2).permute(0, 2, 1)  # [B, N_img, C]
            image_pe_flat = pe_for_dump.flatten(2).permute(0, 2, 1)  # [B, N_img, C]
            queries0 = tokens_for_dump
            keys0 = image_emb_flat
            np.asarray(queries0.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_l0_queries_in.f32")
            )
            np.asarray(keys0.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_l0_keys_in.f32")
            )
            # Self-attn (skip_first_layer_pe=True for layer 0)
            q_sa = blk.self_attn(q=queries0, k=queries0, v=queries0)
            np.asarray(q_sa.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_l0_post_self_attn.f32")
            )
            q_after_sa = q_sa  # no residual since skip_first_layer_pe=True
            q_after_n1 = blk.norm1(q_after_sa)
            np.asarray(q_after_n1.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_l0_post_norm1.f32")
            )
            # Cross-attn token→image
            q_pe_in = q_after_n1 + queries0  # query_pe = queries0 (point_embedding)
            k_pe_in = keys0 + image_pe_flat
            np.asarray(k_pe_in.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_l0_cross_t2i_kpe.f32")
            )
            # Dump k_proj(k_pe_in) to bisect linear vs attention math.
            kproj = blk.cross_attn_token_to_image.k_proj(k_pe_in)
            np.asarray(kproj.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_l0_cross_t2i_kproj.f32")
            )
            ca1_out = blk.cross_attn_token_to_image(q=q_pe_in, k=k_pe_in, v=keys0)
            np.asarray(ca1_out.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_l0_post_cross_t2i.f32")
            )
            q_after_ca1 = q_after_n1 + ca1_out
            q_after_n2 = blk.norm2(q_after_ca1)
            np.asarray(q_after_n2.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_l0_post_norm2.f32")
            )
            mlp_out = blk.mlp(q_after_n2)
            np.asarray(mlp_out.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "tw_l0_post_mlp.f32")
            )

            low_res_masks, iou_pred, sam_tokens, object_score_logits = (
                model.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=high_res_feats,
                )
            )
            np.asarray(low_res_masks.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "mask_logits.f32")
            )
            np.asarray(iou_pred.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "iou_pred.f32")
            )
            np.asarray(object_score_logits.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "object_score.f32")
            )
            np.asarray(sam_tokens.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "sam_tokens_out.f32")
            )

    if os.environ.get("RLX_SAM2_RUN_MEMORY_ENCODER") == "1":
        # Dump memory encoder I/O.
        # pix_feat = stride-16 features (image_embed), mask = full-res
        # mask logits (use the chosen best of low_res_masks upsampled
        # to 1024x1024).
        with torch.inference_mode():
            backbone_out2 = model.forward_image(x)
            _, vision_feats2, _, _ = model._prepare_backbone_features(backbone_out2)
            feat_sizes = [(256, 256), (128, 128), (64, 64)]
            B = x.shape[0]
            feats_bchw = [
                f.permute(1, 2, 0).view(B, -1, h_, w_)
                for f, (h_, w_) in zip(vision_feats2[-3:], feat_sizes)
            ]
            pix_feat = feats_bchw[-1]
            # Synthetic full-res mask: linear gradient, deterministic.
            yy = torch.arange(1024, dtype=torch.float32).view(1, 1, 1024, 1) - 512.0
            xx = torch.arange(1024, dtype=torch.float32).view(1, 1, 1, 1024) - 512.0
            mask_full = yy * 0.01 + xx * 0.005
            np.asarray(pix_feat.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "memenc_pix_feat.f32")
            )
            np.asarray(mask_full.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "memenc_mask.f32")
            )
            mem_out = model.memory_encoder(pix_feat, mask_full, skip_mask_sigmoid=False)
            np.asarray(mem_out["vision_features"].detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "memenc_features.f32")
            )
            # vision_pos_enc is a list of one tensor.
            np.asarray(mem_out["vision_pos_enc"][0].detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "memenc_pos.f32")
            )

    if os.environ.get("RLX_SAM2_RUN_MEMORY_ATTENTION") == "1":
        # Dump memory attention forward: current frame's stride-32 features
        # (queries) cross-attend to memory bank (keys+values).
        with torch.inference_mode():
            backbone_out2 = model.forward_image(x)
            _, vision_feats2, vision_pos_enc2, _ = model._prepare_backbone_features(
                backbone_out2
            )
            # vision_feats2 / vision_pos_enc2 are in (S, B, C) flat form
            # per the reference plumbing. The memory_attention path
            # consumes the LAST feature level (stride-16 for 3-level
            # config) reshaped to (S, B, C) for transformer input.
            curr_feat = vision_feats2[-1]   # (S, B, C)
            curr_pos = vision_pos_enc2[-1]  # (S, B, C)
            # Build a synthetic memory bank: one spatial frame
            # (kv_in_dim=64-channel features at 64x64).
            n_spatial = 64 * 64
            torch.manual_seed(0)
            mem_feat = torch.randn(n_spatial, 1, model.memory_attention.layers[0].cross_attn_image.kv_in_dim)
            mem_pos = torch.randn(n_spatial, 1, model.memory_attention.layers[0].cross_attn_image.kv_in_dim)
            np.asarray(curr_feat.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "memattn_curr.f32")
            )
            np.asarray(curr_pos.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "memattn_curr_pos.f32")
            )
            np.asarray(mem_feat.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "memattn_mem.f32")
            )
            np.asarray(mem_pos.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "memattn_mem_pos.f32")
            )
            out = model.memory_attention(
                curr=[curr_feat],
                curr_pos=[curr_pos],
                memory=mem_feat,
                memory_pos=mem_pos,
                num_obj_ptr_tokens=0,
            )
            np.asarray(out.detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "memattn_output.f32")
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
