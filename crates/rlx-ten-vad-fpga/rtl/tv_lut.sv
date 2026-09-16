// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

`include "tv_defs.svh"

// Piecewise-linear sigmoid/tanh over Q15.
//
// Each ROM word is {value, delta-to-next}, so interpolation is one read and
// one multiply. The last entry carries delta 0, so a saturated index returns
// the endpoint exactly — matching `rlx_ten_vad_core::fixed::lut`.
//
// Latency is one cycle: drive `x`/`sel`, read `y` on the next edge.
module tv_lut (
    input  logic               clk,
    input  logic               sel,   // 0 = sigmoid, 1 = tanh
    input  logic signed [31:0] x,
    output logic signed [31:0] y
);
  logic [31:0] slut [0:`TV_LUT_N];
  logic [31:0] tlut [0:`TV_LUT_N];
  initial begin
    $readmemh("sigmoid.mem", slut);
    $readmemh("tanh.mem", tlut);
  end

  logic [31:0] mag;
  logic [31:0] shifted;
  logic [11:0] pos;
  logic [8:0]  rem;
  logic        neg;

  always_comb begin
    neg     = x < 0;
    mag     = neg ? -x : x;
    shifted = mag >> `TV_LUT_SHIFT;
    pos     = (shifted > `TV_LUT_N) ? `TV_LUT_N : shifted[11:0];
    rem     = mag[`TV_LUT_SHIFT-1:0];
  end

  logic [31:0] word;
  logic [8:0]  rem_r;
  logic        neg_r, sel_r;
  always_ff @(posedge clk) begin
    word  <= sel ? tlut[pos] : slut[pos];
    rem_r <= rem;
    neg_r <= neg;
    sel_r <= sel;
  end

  logic signed [15:0] val, dlt;
  logic signed [31:0] raw;
  always_comb begin
    val = word[31:16];
    dlt = word[15:0];
    // Arithmetic shift, so the truncation direction matches Rust's `>>`.
    raw = 32'(val) + (($signed({1'b0, rem_r}) * 32'(dlt)) >>> `TV_LUT_SHIFT);
    if (sel_r) y = neg_r ? -raw : raw;                       // tanh is odd
    else       y = neg_r ? (32'sd1 << `TV_ACT_FRAC) - raw : raw;  // sigmoid
  end
endmodule
