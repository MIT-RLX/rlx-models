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
Dump PyTorch-reference SAM3 activations for Rust parity tests.

Required env:
  RLX_SAM3_IMAGE_BIN   raw f32 NCHW [3,1008,1008], already SAM3-normalized
  RLX_SAM3_OUT_DIR     output directory

Weights:
  RLX_SAM3_WEIGHTS     local .pt or .safetensors checkpoint
  or RLX_SAM3_DOWNLOAD=1 to fetch facebook/sam3/sam3.pt via hf auth
"""

import os
import subprocess
import sys
import tempfile

import numpy as np


def env(name: str) -> str:
    v = os.environ.get(name)
    if v is None:
        print(f"missing env var: {name}", file=sys.stderr)
        sys.exit(2)
    return v


def maybe_download_weights() -> str:
    weights = os.environ.get("RLX_SAM3_WEIGHTS")
    if weights:
        return weights
    if os.environ.get("RLX_SAM3_DOWNLOAD") == "1":
        target = os.environ.get("RLX_SAM3_HF_DIR", "/tmp/rlx_sam3_hf")
        os.makedirs(target, exist_ok=True)
        subprocess.check_call(
            [
                "hf",
                "download",
                "facebook/sam3",
                "sam3.pt",
                "config.json",
                "--local-dir",
                target,
            ]
        )
        return os.path.join(target, "sam3.pt")
    print("set RLX_SAM3_WEIGHTS or RLX_SAM3_DOWNLOAD=1", file=sys.stderr)
    sys.exit(2)


def load_state_dict(path: str):
    """Load a SAM3 state dict from .pt or .safetensors."""
    import torch

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj


def main() -> int:
    try:
        import torch
    except ImportError as e:
        print(f"torch must be installed: {e}", file=sys.stderr)
        return 3

    # SAM3 hardcodes device="cuda" in several module __init__ paths
    # (position_encoding, decoder, etc.). On CPU-only hosts redirect those
    # to CPU by wrapping the relevant tensor factory functions so we don't
    # need to fork the upstream package.
    _device = os.environ.get("RLX_SAM3_DEVICE", "cpu")
    if _device == "cpu" and not torch.cuda.is_available():
        def _cpu_redirect(fn):
            def wrapper(*args, **kwargs):
                d = kwargs.get("device")
                if d is None:
                    pass
                elif isinstance(d, str) and d.startswith("cuda"):
                    kwargs["device"] = "cpu"
                elif isinstance(d, torch.device) and d.type == "cuda":
                    kwargs["device"] = torch.device("cpu")
                return fn(*args, **kwargs)
            return wrapper

        for _name in (
            "zeros", "ones", "empty", "full",
            "arange", "linspace", "rand", "randn",
            "tensor", "as_tensor",
        ):
            setattr(torch, _name, _cpu_redirect(getattr(torch, _name)))

    # SAM3's _load_checkpoint passes weights_only=True; safetensors round-
    # tripped through torch.save lands on opcodes torch's safe unpickler
    # rejects. Force weights_only=False for this run (file is local, ours).
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load

    # SAM3's `addmm_act` fused kernel pins mat1/mat2/bias to bf16. That makes
    # parity unverifiable and the trunk explodes when downstream layers expect
    # f32. Replace it with an f32-pure path before the model is constructed.
    import torch.nn.functional as _F
    import sam3.perflib.fused as _fused

    def _addmm_act_f32(activation, linear, mat1):
        w = linear.weight.detach()
        b = linear.bias.detach()
        out = _F.linear(mat1, w, b)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(out)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(out)
        raise ValueError(f"Unexpected activation {activation}")

    _fused.addmm_act = _addmm_act_f32
    # Patch any modules that imported the symbol before us.
    import sam3.model.vitdet as _vitdet
    _vitdet.addmm_act = _addmm_act_f32

    try:
        from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
    except ImportError as e:
        print(f"sam3 must be installed: {e}", file=sys.stderr)
        return 3

    image_bin = env("RLX_SAM3_IMAGE_BIN")
    out_dir = env("RLX_SAM3_OUT_DIR")
    weights_path = maybe_download_weights()
    device = os.environ.get("RLX_SAM3_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("RLX_SAM3_DEVICE=cuda but CUDA is unavailable; falling back to CPU", file=sys.stderr)
        device = "cpu"

    os.makedirs(out_dir, exist_ok=True)
    img = np.fromfile(image_bin, dtype=np.float32)
    assert img.size == 3 * 1008 * 1008, f"image must be 3*1008*1008 f32, got {img.size}"
    img = img.reshape(1, 3, 1008, 1008)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
        x = torch.from_numpy(img).to(device).float()
        image_model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            checkpoint_path=None,
            load_from_HF=False,
            enable_inst_interactivity=False,
            compile=False,
        )
        # Load weights manually to bypass _load_checkpoint's weights_only=True
        # path. SAM3's loader strips the "detector." prefix from full ckpts.
        full_state = load_state_dict(weights_path)
        sam3_image_ckpt = {
            (k.replace("detector.", "") if "detector." in k else k): v
            for k, v in full_state.items()
        }
        if image_model.inst_interactive_predictor is not None:
            sam3_image_ckpt.update(
                {
                    k.replace("tracker.", "inst_interactive_predictor.model."): v
                    for k, v in full_state.items()
                    if "tracker" in k
                }
            )
        missing, unexpected = image_model.load_state_dict(sam3_image_ckpt, strict=False)
        print(
            f"loaded sam3 checkpoint: missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
        )
        trunk = image_model.backbone.vision_backbone.trunk
        patch_module = trunk.patch_embed
        patch = patch_module.proj(x) if hasattr(patch_module, "proj") else patch_module(x)
        if patch.ndim == 4 and patch.shape[1] == 1024:
            patch = patch.permute(0, 2, 3, 1).contiguous()
        elif patch.ndim == 3:
            patch = patch.reshape(1, 72, 72, 1024).contiguous()
        np.asarray(patch.detach().cpu().numpy(), dtype=np.float32).tofile(
            os.path.join(out_dir, "patch_embed.f32")
        )

        # Full trunk output (after all ViT blocks + final LN, before neck).
        # Shape is [1, grid, grid, embed_dim] = [1, 72, 72, 1024]. Used as
        # the vision-encoder parity gate for the native Rust ViT.
        # The trunk goes through a fused addmm_act that internally promotes
        # to bf16 on some paths — force the trunk to float32 and disable any
        # ambient autocast so parity is deterministic.
        try:
            trunk.float()
            with torch.amp.autocast(device_type="cpu", enabled=False):
                trunk_out = trunk(x.float())
            trunk_last = trunk_out[-1] if isinstance(trunk_out, (list, tuple)) else trunk_out
            tl_perm = trunk_last
            if tl_perm.ndim == 4 and tl_perm.shape[1] == 1024:
                tl_perm = tl_perm.permute(0, 2, 3, 1).contiguous()
            np.asarray(tl_perm.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "vision_encoder.f32")
            )
        except Exception as e:
            print(f"warning: vision encoder dump failed: {e}", file=sys.stderr)
            trunk_out = None

        # Text encoder: dump tokenized prompt and the resized memory the
        # detector consumes. We always dump these — they don't depend on
        # the image or RLX_SAM3_RUN_IMAGE.
        try:
            prompt = os.environ.get("RLX_SAM3_TEXT_PROMPT", "person")
            txt_enc = image_model.backbone.language_backbone
            txt_enc.float()
            # Tokenize via the model's own tokenizer for byte parity with
            # Python inference (we reuse those tokens on the Rust side).
            tok = txt_enc.tokenizer([prompt], context_length=txt_enc.context_length)
            np.asarray(tok.detach().cpu().numpy(), dtype=np.int32).tofile(
                os.path.join(out_dir, "text_tokens.i32")
            )
            with torch.amp.autocast(device_type="cpu", enabled=False):
                # Replicate VETextEncoder.forward but force CPU+f32 and
                # bypass the autocast assertion paths.
                _, text_memory = txt_enc.encoder(tok)  # [b, seq, 1024]
                text_memory = text_memory.transpose(0, 1)
                text_memory_resized = txt_enc.resizer(text_memory)
            np.asarray(text_memory_resized.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "text_memory_resized.f32")
            )
        except Exception as e:
            print(f"warning: text encoder dump failed: {e}", file=sys.stderr)

        # Neck output: 4 multi-scale feature maps + sinusoidal pos encodings.
        try:
            if trunk_out is None:
                raise RuntimeError("trunk failed, skipping neck")
            neck = image_model.backbone.vision_backbone
            neck.float()
            with torch.amp.autocast(device_type="cpu", enabled=False):
                sam3_out, sam3_pos, _, _ = neck(x.float())
            for i, (feat, pos) in enumerate(zip(sam3_out, sam3_pos)):
                fd = feat.detach().float().cpu().numpy().astype("float32")
                pd = pos.detach().float().cpu().numpy().astype("float32")
                fd.tofile(os.path.join(out_dir, f"neck_level_{i}.f32"))
                pd.tofile(os.path.join(out_dir, f"neck_pos_{i}.f32"))
                # also dump shape so the rust harness can parse without
                # hard-coding (one int32 per dim, len from filename suffix).
                shape_arr = np.asarray(fd.shape, dtype=np.int32)
                shape_arr.tofile(os.path.join(out_dir, f"neck_level_{i}.shape"))
        except Exception as e:
            print(f"warning: neck dump failed: {e}", file=sys.stderr)

        # Detector encoder fusion: run on the last (scale=1.0) FPN level with
        # text-only prompt (no geometry). Captures inputs + outputs for the
        # Rust parity test in isolation.
        try:
            det = image_model
            det.float()
            with torch.amp.autocast(device_type="cpu", enabled=False):
                # Encoder consumes the FPN level (scale=1.0) only, since
                # num_feature_levels=1 and the trunk neck is sliced by
                # scalp=1 in SAM3VLBackbone.
                src_level = sam3_out[-2]  # scale 1.0, [1, 256, 72, 72]
                pos_level = sam3_pos[-2]
                np.asarray(src_level.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                    os.path.join(out_dir, "encoder_src.f32")
                )
                np.asarray(pos_level.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                    os.path.join(out_dir, "encoder_pos.f32")
                )

                # Prompt is the text memory only (no geometry / no visual).
                prompt = text_memory_resized.float()  # [seq, bs, 256]
                prompt_mask = (tok == 0).bool()  # [bs, seq]
                np.asarray(prompt.detach().cpu().numpy(), dtype=np.float32).tofile(
                    os.path.join(out_dir, "encoder_prompt.f32")
                )
                np.asarray(prompt_mask.detach().cpu().numpy().astype("uint8"), dtype=np.uint8).tofile(
                    os.path.join(out_dir, "encoder_prompt_mask.u8")
                )

                enc = det.transformer.encoder
                # Upstream's no-feat_sizes branch asserts `x.dim == 4`
                # which is broken (dim is a method, never == 4). Provide
                # feat_sizes + seq-first src so the working path is taken.
                bs = 1
                seq_src = src_level.flatten(2).permute(2, 0, 1).contiguous()  # [hw, bs, c]
                seq_pos = pos_level.flatten(2).permute(2, 0, 1).contiguous()
                memory_out = enc(
                    src=[seq_src],
                    prompt=prompt,
                    src_pos=[seq_pos],
                    src_key_padding_mask=[None],
                    prompt_key_padding_mask=prompt_mask,
                    feat_sizes=[(72, 72)],
                )
                mem = memory_out["memory"]  # [seq, bs, 256]
                np.asarray(mem.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                    os.path.join(out_dir, "encoder_memory.f32")
                )
        except Exception as e:
            print(f"warning: encoder dump failed: {e}", file=sys.stderr)
            memory_out = None

        # Detector decoder: feed encoder memory + text prompt and capture
        # stacked intermediate outputs + ref boxes + presence logits.
        try:
            if memory_out is None:
                raise RuntimeError("encoder failed, skipping decoder")
            dec = det.transformer.decoder
            dec.float()
            with torch.amp.autocast(device_type="cpu", enabled=False):
                bs = 1
                query_embed = dec.query_embed.weight  # [200, 256]
                tgt = query_embed.unsqueeze(1).repeat(1, bs, 1).contiguous()  # [200, 1, 256]
                hs, ref_boxes, presence_out, presence_feats = dec(
                    tgt=tgt,
                    memory=memory_out["memory"],
                    memory_key_padding_mask=memory_out["padding_mask"],
                    pos=memory_out["pos_embed"],
                    reference_boxes=None,
                    level_start_index=memory_out["level_start_index"],
                    spatial_shapes=memory_out["spatial_shapes"],
                    valid_ratios=memory_out["valid_ratios"],
                    tgt_mask=None,
                    memory_text=prompt,
                    text_attention_mask=prompt_mask,
                    apply_dac=False,
                )
            # hs: [num_layers, nq, bs, d_model]
            np.asarray(hs.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "decoder_intermediate.f32")
            )
            np.asarray(ref_boxes.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "decoder_ref_boxes.f32")
            )
            if presence_out is not None:
                np.asarray(presence_out.float().detach().cpu().numpy().squeeze(-1), dtype=np.float32).tofile(
                    os.path.join(out_dir, "decoder_presence_logits.f32")
                )
            if presence_feats is not None:
                np.asarray(presence_feats.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                    os.path.join(out_dir, "decoder_presence_feats.f32")
                )
        except Exception as e:
            print(f"warning: decoder dump failed: {e}", file=sys.stderr)
            hs = None

        # Segmentation head + dot product scoring.
        try:
            if hs is None:
                raise RuntimeError("decoder failed; skipping segmentation/scoring")
            seg = image_model.segmentation_head
            seg.float()
            scoring = image_model.dot_prod_scoring
            scoring.float()
            with torch.amp.autocast(device_type="cpu", enabled=False):
                # backbone_feats = sam3_out (the 3 levels after scalp=1).
                # Upstream takes backbone_fpn from backbone_out; with scalp=1
                # it is sam3_out[:-1] = [scale_4.0, scale_2.0, scale_1.0].
                backbone_feats = list(sam3_out[:-1])  # 3 levels
                # obj_queries: hs is [num_layers, nq, bs, d]; the model
                # transposes hs to [num_layers, bs, nq, d] before forward.
                hs_bf = hs.transpose(1, 2).contiguous()  # [L, B, Q, D]
                # encoder_hidden_states: memory_out["memory"] is [hw, bs, d].
                seg_out = seg(
                    backbone_feats=backbone_feats,
                    obj_queries=hs_bf,
                    image_ids=torch.zeros(1, dtype=torch.long),
                    encoder_hidden_states=memory_out["memory"],
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                mask_pred = seg_out["pred_masks"]
                sem_seg = seg_out["semantic_seg"]
                # dot product scoring: hs_bf is [L, B, Q, D].
                scores = scoring(hs_bf, prompt, prompt_mask).squeeze(-1)
                # Final boxes: apply bbox_embed to *every* layer's hs and
                # add to inv_sigmoid of corresponding ref boxes. We dump the
                # *last layer's* refined boxes since that's the natural
                # final prediction the public Sam3Processor uses.
                from sam3.model.model_misc import inverse_sigmoid as _inv_sig
                ref_boxes_bf = ref_boxes.transpose(1, 2).contiguous()  # [L, B, Q, 4]
                anchor_box_offsets = image_model.transformer.decoder.bbox_embed(hs_bf)
                final_boxes_cxcywh = (_inv_sig(ref_boxes_bf) + anchor_box_offsets).sigmoid()
                final_boxes_last_cxcywh = final_boxes_cxcywh[-1]  # [B, Q, 4]
                from sam3.model.box_ops import box_cxcywh_to_xyxy
                final_boxes_xyxy = box_cxcywh_to_xyxy(final_boxes_last_cxcywh)
                np.asarray(final_boxes_last_cxcywh.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                    os.path.join(out_dir, "final_boxes_cxcywh.f32")
                )
                np.asarray(final_boxes_xyxy.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                    os.path.join(out_dir, "final_boxes_xyxy.f32")
                )
            np.asarray(mask_pred.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "seg_mask_pred.f32")
            )
            np.asarray(sem_seg.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "seg_semantic.f32")
            )
            np.asarray(scores.float().detach().cpu().numpy(), dtype=np.float32).tofile(
                os.path.join(out_dir, "decoder_scores.f32")
            )
        except Exception as e:
            print(f"warning: segmentation/scoring dump failed: {e}", file=sys.stderr)

        if os.environ.get("RLX_SAM3_RUN_IMAGE", "0") == "1":
            prompt = os.environ.get("RLX_SAM3_TEXT_PROMPT", "person")
            try:
                from sam3.model.sam3_image_processor import Sam3Processor
                from PIL import Image

                # Convert normalized NCHW back to uint8 for the processor
                # path so postprocessing dumps match public inference.
                rgb = ((img[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).clip(0, 255).astype("uint8")
                processor = Sam3Processor(image_model)
                state = processor.set_image(Image.fromarray(rgb))
                out = processor.set_text_prompt(state=state, prompt=prompt)
                for key in ("masks", "boxes", "scores"):
                    if key in out:
                        val = out[key]
                        if hasattr(val, "detach"):
                            arr = val.detach().cpu().numpy()
                        else:
                            arr = np.asarray(val)
                        np.asarray(arr, dtype=np.float32).tofile(os.path.join(out_dir, f"image_{key}.f32"))
            except Exception as e:
                print(f"warning: image processor dump failed: {e}", file=sys.stderr)

        if os.environ.get("RLX_SAM3_RUN_VIDEO", "0") == "1":
            try:
                video_model = build_sam3_video_model(
                    checkpoint_path=weights,
                    load_from_HF=False,
                    device=device,
                    compile=False,
                    strict_state_dict_loading=False,
                )
                # Dump a model-construction sentinel for the Rust harness.
                np.asarray([1.0], dtype=np.float32).tofile(os.path.join(out_dir, "video_model_ready.f32"))
                del video_model
            except Exception as e:
                print(f"warning: video model dump failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
