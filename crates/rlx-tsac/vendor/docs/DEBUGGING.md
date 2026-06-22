# tsac-ng Debugging Guide — Chain of Thought Records

This document records the step-by-step debugging journeys for the most
challenging bugs. Each entry shows the thought process, wrong turns taken,
and how the root cause was finally identified.

## D-001: NaN in GPU Backend Output

### Symptom
CUDA decoder output: NaN=0 on one run, NaN=130 on the next. Non-deterministic.

### Chain of Thought

**Step 1 — Narrow the NaN source**
```
Output has NaN in first 32 samples of each channel.
→ Check if NaN is in the model.0 conv output
→ model.0 output verified finite (84.8, 118.5, ...)
→ NaN appears AFTER blocks 1-4
→ Check block 1 conv_transpose output
```

**Step 2 — Found NaN in conv_transpose output**
```
Block 1 convt: 0/15360 NaN — clean!
But Block 2 convt: 1e26 values (overflow!)
Block 3 convt: NaN

Assumption: weight dequantization must be wrong.
→ Check dequant_weights function
→ Verified weight_g values (range 0.19-0.49, valid)
→ Verified weight_v raw data (235, 87, 216, ... valid)
→ Single-element dequant test: w = 0.35 (valid)
```

**Step 3 — Wrong turn: blamed -ffast-math**
```
isnan() returns false? Must be -ffast-math!
→ Removed -ffast-math from CMakeLists.txt
→ Still NaN present
→ Actually -ffast-math was masking detection, not causing NaN
```

**Step 4 — Bit-level analysis reveals pattern**
```
NaN values are all 0xffc00000 (quiet NaN with sign bit)
Channels 42-95 all NaN, channels 0-41 clean
→ This is a partial corruption, not a computation error
→ Hypothesis: weight data for some channels is garbage
```

**Step 5 — Found root cause: float32 weight_v**
```
data_size / dims_product:
  model.0.weight_v:     1.07  → uint8 (BF8)
  model.4.block.1.wv:   4.0   → FLOAT32!
  model.6.weight_v:     4.0   → FLOAT32!

model_loader set elem_size=1 for ALL weight_v (heuristic).
Reading float32 bytes as uint8 → garbage for model.4 and model.6.
Channels 42-95 correspond to model.4 output → NaN.
```

### Resolution
Detect float32 by comparing data_size with dims_product:
- `data_size == dims×4 → elem_size=4` (float32, direct copy)
- `data_size == dims → elem_size=1` (uint8 BF8, dequant)

---

## D-002: HIP GPU Page Fault

### Symptom
```
Memory access fault by GPU node-1 (Agent handle: 0x...)
on address 0x7f2eb6201000. Reason: Page not present.
```

### Chain of Thought

**Step 1 — Address analysis**
```
0x7f2eb6201000 is in CPU mmap'd address space.
→ GPU kernel accessing CPU address
→ Something passed a CPU pointer to GPU kernel
```

**Step 2 — Find the kernel with bad pointer**
```
Backtrace shows crash in mul_kernel called from dev_f32().
Line 106: mul_k<<<...>>>(t->dev_f32, t->dev_f32, dg, nel);

Check dg: weight_g tensor, 1536 floats on GPU.
Check nel: t->data_size = 11,796,480 bytes (not elements!).

nel is in BYTES but the kernel iterates over nel elements!
11,796,480 elements × 4 bytes = 47MB read from dg.
But dg only has 1536 floats = 6KB.

GPU reads dg[1536..11796479] — OUT OF BOUNDS!
```

**Step 3 — Why nel = data_size in bytes?**
```
dev_f32 function uses t->data_size which is stored in BYTES.
But the code uses it as ELEMENT COUNT:
  int nel = t->data_size;  // should be t->data_size / elem_size
  i8tof32_k<<<...>>>(t->dev_f32, raw, nel);
  mul_k<<<...>>>(t->dev_f32, t->dev_f32, dg, nel);
```

### Resolution
Replace GPU-side dequant with CPU-side (same as CUDA backend).
Upload_f32() computes correct element count from dims, not data_size.

---

## D-003: HIP convt_k Wrong Output (1e26 values)

### Symptom
Block 1 convt output: values ~ -122,522 (large but finite)
Block 2 convt output: values ~ 1.3e26 (overflow!)
Block 3 convt output: NaN

### Chain of Thought

**Step 1 — Is the weight format correct?**
```
CPU decoder produces correct output.
CUDA decoder produces correct output.
HIP decoder produces overflow.
→ Difference must be in weight layout or kernel grid
```

**Step 2 — Found: 1D grid instead of 2D**
```cpp
// HIP convt_k launch:
convt_k<<<num_blocks, BLK>>>(
    d_next, d_cur, w, cur_T, next_T, convt_K, cur_C, target_C, 2);

// num_blocks = (target_C * next_T + BLK - 1) / BLK = 60
// For block 1: Ci=1536, Ti=10
// Expected grid: dim3(1536, (10+255)/256) = dim3(1536, 1)
// Actual grid: <<<60, BLK>>> → 1D grid with 60 blocks
// blockIdx.x goes 0..59 instead of 0..1535
// Only 60/1536 input channels processed!
```

**Step 3 — Fix and test**
```
Fixed to <<<dim3(cur_C, (cur_T+BLK-1)/BLK), dim3(1, BLK)>>>
Still NaN. Additional issue: weight layout [Co,K,Ci] not transposed to [Co,Ci,K].
Fixed upload_f32 to transpose during dequant.
Result: NaN=0, nz=160.
```

### Resolution
Two bugs: 1D grid (only 4% of channels processed) + weight format mismatch.
Both fixed together.

---

## D-004: .txc File Rejected (Error -3)

### Symptom
```
tsac-ng --cuda c test.wav test.txc → Success
tsac-ng --cuda d test.txc test.wav → Error: -3 (TSAC_ERR_FORMAT)
```

### Chain of Thought

**Step 1 — First hypothesis: version byte order**
```
txc_write stores version as native LE (01 00 for 1).
txc_read reads version as BE: (data[4]<<8)|data[5] = 0x0100 = 256.
Check: if (version < 1 || version > 255) → REJECTED!

Fix: accept both BE and LE in txc_read.
Still fails! Different error.
```

**Step 2 — Check raw hex**
```
Header bytes:
0-3:  46 42 41 5a = "FBAZ" ✓
4-5:  01 00 = version (now accepted as LE = 1) ✓
6-7:  06 00 = flags(6) + n_codebooks(0)

n_codebooks = 0! → rejected (< 1)
But .txc_write set n_codebooks = 6.
```

**Step 3 — Analyze struct layout vs serialization**
```
TSCHeader struct (LE):
  offset 6-7: n_codebooks (uint16_t) = 0x0006 = bytes [06, 00]
  
But .txc format reads n_codebooks from byte 7 only (u8):
  data[7] = 0x00  → n_codebooks = 0!

The struct stores n_codebooks as uint16_t at bytes 6-7.
The .txc format stores n_codebooks as uint8_t at byte 7 only.
Low byte (0x06) at offset 6, high byte (0x00) at offset 7.
Reading byte 7 gets the HIGH byte = 0x00!
```

**Step 4 — Root cause: memcpy struct vs format mismatch**
```c
// txc_write:
memcpy(out_hdr, hdr, sizeof(TSCHeader));  // native struct layout
// bytes 6-7 are LE uint16 = [0x06, 0x00]

// txc_read:
hdr->n_codebooks = data[7];  // reads byte 7 = 0x00!
```

### Resolution
Manual byte-level serialization in txc_write instead of memcpy:
```
buf[6] = flags & 0xFF;          // u8
buf[7] = n_codebooks & 0xFF;    // u8
buf[8..11] = block_len (BE u32)
buf[12..15] = sample_rate (BE u32)
```

---

## D-005: CUDA Encoder Produces 4 Frames Instead of 75

### Symptom
24000 samples / 320 block_len → expected 75 frames.
CUDA encoder output: n_frames = 4.

### Chain of Thought

**Step 1 — Check n_frames calculation**
```c
int nf = (n_samples + block_len - 1) / block_len;
// (24000 + 320 - 1) / 320 = 75  ✓
```

**Step 2 — Check encoder loop**
```c
for (int blk = 4; blk >= 1; blk--) {
    next_T = cur_T / 2;  // ← BUG!
    // ... conv1d ...
    cur_T = next_T;
}
```

Each block divides T by 2. After 4 blocks:
75 → 37 → 18 → 9 → 4 frames.

**Step 3 — Is stride=2 correct for encoder?**
```
Decoder uses ConvTranspose1d(stride=2) for upsampling.
Encoder uses Conv1d(stride=1) — no downsampling in temporal dimension.
The DAC encoder does NOT downsample temporally.
The "downsampling" in DAC is in encoder dimension (1024→96→192→384→768→1536),
NOT in time dimension.
```

### Resolution
Changed `next_T = cur_T / 2` to `next_T = cur_T` (stride=1).
Removed conv1d_strided call, used regular conv1d.
