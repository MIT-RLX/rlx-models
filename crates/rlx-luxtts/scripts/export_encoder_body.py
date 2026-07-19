#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Split the LuxTTS/ZipVoice `text_encoder.onnx` into a native-friendly ENCODER
# BODY whose only dynamic length is a single bound symbol.
#
# WHY: the original text_encoder takes TWO separate token inputs
# (`prompt_tokens`, `tokens`), concatenates + pads them internally to a DERIVED
# length `T + Tp + 1`, embeds, runs a 4-layer Zipformer, then applies a
# scalar-`num_frames` length regulator (repeat_interleave + tail). The derived
# concat length and the scalar length regulator are exactly the two things the
# rlx-onnx-import shape evaluator can't fold, so the whole graph won't run
# natively at the right shape.
#
# THE SPLIT (both parts verified BIT-EXACT vs onnxruntime, cos 1.0):
#   1. Rust does the token concat+pad: `input_ids = prompt_tokens ++ tokens ++ [0]`
#      (pad token 0 at the END) → a single `[1, S]` (S = Tp + T + 1) i64 sequence.
#   2. `encoder_body.onnx` (this script): a SINGLE input `/Pad_output_0` [1, S]
#      (the padded ids) → the encoder output `/text_encoder/Transpose_1_output_0`
#      [1, S, 100]. One dynamic length `S`; imports+runs natively bit-exact.
#   3. Rust does the length regulator on the [1, S, 100] encoder output:
#        seq       = S                       # 18
#        main_len  = S - 1                    # 17 (Slice_1 = enc[:, 0:-1, :])
#        num_frames = ceil((prompt_features_len / Tp) * (Tp + T) / speed)
#        repeat    = num_frames // main_len   # 6
#        tail      = num_frames - repeat*main_len   # 3
#        main      = repeat_interleave(enc[:, :-1, :], repeat, axis=1)  # [1,102,100]
#        tail_rows = enc[:, -1:, :] repeated `tail` times               # [1,3,100]
#        text_condition = concat(main, tail_rows, axis=1)               # [1,num_frames,100]
#
# Usage:
#   python export_encoder_body.py \
#       weights/tts/luxtts/text_encoder.onnx weights/tts/luxtts/encoder_body.onnx
import sys
import onnx
from onnx import TensorProto, helper
from onnx.utils import Extractor

SPLIT = "/text_encoder/Transpose_1_output_0"  # encoder output (feeds Slice_1/Slice_2)
PAD_IN = "/Pad_output_0"                        # padded ids (embed indices)


def main(src: str, dst: str) -> None:
    m = onnx.load(src)
    del m.graph.value_info[:]  # drop stale inferred types that confuse Extractor
    # Type the new single input and the split output so Extractor can cut here.
    m.graph.value_info.append(helper.make_tensor_value_info(PAD_IN, TensorProto.INT64, [1, "S"]))
    m.graph.value_info.append(helper.make_tensor_value_info(SPLIT, TensorProto.FLOAT, [1, "S", 100]))
    sub = Extractor(m).extract_model([PAD_IN], [SPLIT])
    onnx.save(sub, dst)
    ins = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in sub.graph.input]
    outs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in sub.graph.output]
    print(f"wrote {dst}: {len(sub.graph.node)} nodes; in={ins} out={outs}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: export_encoder_body.py <text_encoder.onnx> <encoder_body.onnx>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
