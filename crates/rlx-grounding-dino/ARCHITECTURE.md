# Grounding DINO (grounding-dino-base) — authoritative weight map

Reverse-engineered from the published `model.safetensors` header (1206 tensors).
Names are exactly as stored in the HF checkpoint. `[out, in]` for Linear weights.

## Top-level modules
| prefix | role |
|--------|------|
| `model.backbone.conv_encoder.model.*` | Swin-B vision backbone |
| `model.text_backbone.*` | BERT text backbone (bert-base-uncased) |
| `model.text_projection.{weight[256,768],bias[256]}` | text 768→256 |
| `model.input_proj_vision.{0..3}` | neck: 1×1 conv+GroupNorm ×3, 3×3 s2 conv+GroupNorm ×1 → 256 |
| `model.level_embed [4,256]` | per-level embedding |
| `model.encoder.layers.{0..5}` | feature enhancer |
| `model.enc_output{,_norm}` | two-stage memory proj (Linear 256→256 + LN) |
| `model.encoder_output_bbox_embed.layers.{0,1,2}` | query-selection box MLP 256→256→256→4 |
| `model.query_position_embeddings.weight [900,256]` | decoder query pos (two-stage: not used as content) |
| `model.decoder.layers.{0..5}` | cross-modality decoder |
| `model.decoder.reference_points_head.layers.{0,1}` | sine(ref)→query-pos MLP 512→256→256 |
| `model.decoder.bbox_embed.0.layers.{0,1,2}` | box MLP 256→256→256→4 (SHARED across all 6 layers) |
| `model.decoder.layer_norm` | final decoder LN |
| class head | **parameter-free** contrastive dot-product (query · text_features) |

## Swin backbone (`...conv_encoder.model.`)
- `embeddings.patch_embeddings.projection` Conv2d(3,128,k4,s4) ; `embeddings.norm` LN(128)
- stages `encoder.layers.{0..3}`, depths `[2,2,18,2]`, dims `128,256,512,1024`, heads `[4,8,16,32]`, window 12
- block `...layers.{s}.blocks.{b}.`:
  - `layernorm_before` LN
  - `attention.self.{query,key,value}` Linear(d,d) ; `attention.self.relative_position_bias_table [(2*12-1)^2=529, heads]` ;
    `attention.self.relative_position_index [144,144]` (144 = 12·12)
  - `attention.output.dense` Linear(d,d)
  - `layernorm_after` LN ; `intermediate.dense` Linear(d,4d) ; `output.dense` Linear(4d,d)
  - even blocks: window attn (shift 0); odd blocks: shifted-window attn (shift 6)
- `...layers.{0,1,2}.downsample` PatchMerging: `norm` LN(4d) then `reduction` Linear(4d,2d) (no bias)
- `hidden_states_norms.stage{2,3,4}` LN on output feature maps (256/512/1024) → 3 maps to the neck

## Text backbone (`model.text_backbone.`) — standard BERT
- `embeddings.{word_embeddings[30522,768],position_embeddings[512,768],token_type_embeddings[2,768],LayerNorm}`
- `encoder.layer.{0..11}.attention.self.{query,key,value}` + `attention.output.{dense,LayerNorm}` +
  `intermediate.dense[3072,768]` + `output.{dense[768,3072],LayerNorm}` ; gelu (erf) ; eps 1e-12
- run with a **2-D phrase block-diagonal self-attention mask** (`get_text_self_attention_masks`), not just padding.

## Feature enhancer layer (`model.encoder.layers.{i}.`)
- `text_enhancer_layer`: text self-attn `self_attn.{query,key,value,out_proj}`(256), `layer_norm_before/after`,
  `fc1[1024,256]/fc2[256,1024]` (FFN, relu). (Pre/Post-LN per HF.)
- `fusion_layer` (BiAttention, inner dim 1024): `attn.{vision_proj,text_proj,values_vision_proj,values_text_proj}`[1024,256],
  `attn.{out_vision_proj,out_text_proj}`[256,1024], `layer_norm_vision/text`, `vision_param/text_param`[256] (layerscale γ).
- `deformable_layer` (vision MSDeformAttn): `self_attn.sampling_offsets[256,256]`,
  `self_attn.attention_weights[128,256]` (128=heads8·levels4·points4), `self_attn.value_proj/output_proj`[256,256],
  `self_attn_layer_norm`, `fc1[2048,256]/fc2[256,2048]`, `final_layer_norm`.
- Order (HF): fusion → text_enhancer → deformable.

## Decoder layer (`model.decoder.layers.{i}.`)
- `self_attn.{query,key,value,out_proj}`(256) + `self_attn_layer_norm`
- `encoder_attn_text.{query,key,value,out_proj}`(256) + `encoder_attn_text_layer_norm` (text cross-attn)
- `encoder_attn` MSDeformAttn (`sampling_offsets`,`attention_weights`,`value_proj`,`output_proj`) + `encoder_attn_layer_norm`
- `fc1[2048,256]/fc2[256,2048]` + `final_layer_norm` (FFN, relu)

## Hyperparameters (config.json)
d_model 256, enc/dec layers 6, heads 8, ffn 2048, levels 4, points 4, queries 900, max_text_len 256,
pos sine (temperature 20), activation relu. Swin window 12, image_size 384 (training; inference is variable size).
