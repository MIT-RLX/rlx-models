// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

`timescale 1ns / 1ps
`include "tv_defs.svh"

// Self-checking testbench: replays the golden feature stream and requires the
// RTL to reproduce `rlx_ten_vad_core::fixed` bit-for-bit.
//
// Frames must run in order and without reset between them — the LSTM state is
// part of the expected result, so a mismatch on frame N can be a fault in
// frame N or in any frame before it.
//
//   iverilog -g2012 -I rtl -o sim rtl/tv_lut.sv rtl/tv_core.sv tb/tv_tb.sv
//   (cd rtl && vvp ../sim +frames=250)
module tv_tb;
  localparam int FEATN = `TV_FEAT_N;

  logic clk = 0, rst_n = 0, start = 0, feat_we = 0;
  logic [11:0] feat_addr = 0;
  logic signed [31:0] feat_data = 0;
  logic busy, done;
  logic signed [31:0] prob;

  tv_core dut (
      .clk(clk), .rst_n(rst_n),
      .feat_we(feat_we), .feat_addr(feat_addr), .feat_data(feat_data),
      .start(start), .busy(busy), .done(done), .prob(prob)
  );

  always #5 clk = ~clk;

  logic [31:0] feats  [0:250*FEATN-1];
  logic [31:0] golden [0:249];

  int frames, bad, f, i;
  longint cyc_start, cyc, cyc_total;
  int cycles = 0;
  always @(posedge clk) cycles <= cycles + 1;

  initial begin
    $readmemh("../tb/golden_features.mem", feats);
    $readmemh("../tb/golden_probs.mem", golden);
    if (!$value$plusargs("frames=%d", frames)) frames = 250;

    repeat (4) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    bad = 0;
    cyc_total = 0;
    for (f = 0; f < frames; f++) begin
      for (i = 0; i < FEATN; i++) begin
        feat_we   <= 1'b1;
        feat_addr <= i[11:0];
        feat_data <= feats[f * FEATN + i];
        @(posedge clk);
      end
      feat_we <= 1'b0;
      @(posedge clk);

      cyc_start = cycles;
      start <= 1'b1;
      @(posedge clk);
      start <= 1'b0;
      // Gate on `busy`, not `done`: `done` stays asserted until the next run
      // starts, so waiting on it would sample the previous frame's result.
      wait (busy);
      wait (!busy);
      cyc = cycles - cyc_start;
      cyc_total += cyc;

      if (prob !== $signed(golden[f])) begin
        bad++;
        if (bad <= 10)
          $display("FRAME %0d MISMATCH: rtl=%0d (0x%08x) golden=%0d (0x%08x)",
                   f, prob, prob, $signed(golden[f]), golden[f]);
      end
      @(posedge clk);
    end

    $display("");
    $display("frames        %0d", frames);
    $display("mismatches    %0d", bad);
    $display("cycles/frame  %0d", cyc_total / frames);
    $display("min clock     %0.2f MHz for 62.5 fps", (cyc_total / frames) * 62.5 / 1.0e6);
    if (bad == 0) $display("RESULT PASS — RTL is bit-exact against the Rust fixed-point net");
    else          $display("RESULT FAIL");
    $finish;
  end

  initial begin
    // 250 frames x ~173 k cycles x 10 ns is ~433 ms of simulated time.
    #2_000_000_000;
    $display("RESULT FAIL — timeout");
    $finish;
  end
endmodule
