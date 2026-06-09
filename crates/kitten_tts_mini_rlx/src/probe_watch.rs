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

//! Canonical intermediate probes for Kitten ORT parity (one batch compile, many ActCopy outputs).

/// HIR node name → ONNX tensor name for ORT reference dumps.
pub const WATCH: &[(&str, &str)] = &[
    (
        "/bert/embeddings/LayerNorm/LayerNormalization",
        "/bert/embeddings/LayerNorm/LayerNormalization_output_0",
    ),
    (
        "/bert/encoder/embedding_hidden_mapping_in/Add",
        "/bert/encoder/embedding_hidden_mapping_in/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/LayerNorm/LayerNormalization",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/LayerNorm/LayerNormalization_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm/LayerNormalization",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm/LayerNormalization_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query_1/MatMul_quant_f32",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query_1/MatMul_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query_1/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/key_1/MatMul_quant_f32",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/key_1/MatMul_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/key_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/key_1/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/value_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/value_1/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Softmax",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Softmax_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Transpose",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Transpose_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Transpose_2",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Transpose_2_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Mul",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Mul_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Mul_1",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Mul_1_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Add_1",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Add_1_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense_1/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/MatMul",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/MatMul_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/MatMul",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/MatMul_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Add_1",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Add_1_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Reshape_3",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention_1/Reshape_3_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense_1/MatMul_quant_f32",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense_1/MatMul_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0_1/Add",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0_1/Add_output_0",
    ),
    (
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm_11/LayerNormalization",
        "/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm_11/LayerNormalization_output_0",
    ),
    ("/bert_encoder/Add", "/bert_encoder/Add_output_0"),
    ("/text_encoder_1/Expand", "/text_encoder_1/Expand_output_0"),
    ("/text_encoder_1/Concat_4", "/text_encoder_1/Concat_4"),
    ("/lstm/Transpose", "/lstm/Transpose_output_0"),
    (
        "/duration_proj/linear_layer/Add",
        "/duration_proj/linear_layer/Add_output_0",
    ),
    ("/Sigmoid", "/Sigmoid_output_0"),
    ("/ReduceSum", "/ReduceSum_output_0"),
];

pub fn matches_filter(hir_name: &str, filter: Option<&str>) -> bool {
    filter.is_none_or(|f| !f.is_empty() && hir_name.contains(f))
}
