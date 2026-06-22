/// Discrete codec output from [`crate::MimiCodec::encode_pcm`].
#[derive(Debug, Clone)]
pub struct MimiCodes {
    /// Per-frame codebook indices, shape `[num_frames][num_quantizers]`.
    pub frames: Vec<Vec<u32>>,
    pub num_quantizers: usize,
}

impl MimiCodes {
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// HF layout `[num_quantizers][num_frames]` (same as `MimiModel.encode` output).
    pub fn to_hf_layout(&self) -> Vec<Vec<u32>> {
        let t = self.num_frames();
        if t == 0 {
            return Vec::new();
        }
        let k = self.frames[0].len();
        let mut out = vec![vec![0u32; t]; k];
        for (ti, row) in self.frames.iter().enumerate() {
            for (ki, &code) in row.iter().enumerate() {
                out[ki][ti] = code;
            }
        }
        out
    }

    pub fn from_hf_layout(codes: Vec<Vec<u32>>) -> Self {
        let k = codes.len();
        let t = codes.first().map(|r| r.len()).unwrap_or(0);
        let mut frames = Vec::with_capacity(t);
        for ti in 0..t {
            let mut row = Vec::with_capacity(k);
            for row_k in &codes {
                row.push(row_k[ti]);
            }
            frames.push(row);
        }
        Self {
            frames,
            num_quantizers: k,
        }
    }
}
