use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DacCodes {
    /// Per-frame codebook indices `[frame][quantizer]`.
    pub frames: Vec<Vec<u32>>,
    pub num_quantizers: usize,
}

impl DacCodes {
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Layout `[quantizer][frame]`.
    pub fn to_quantizer_layout(&self) -> Vec<Vec<u32>> {
        if self.frames.is_empty() {
            return Vec::new();
        }
        let t = self.frames.len();
        let k = self.num_quantizers;
        let mut out = vec![vec![0u32; t]; k];
        for (ti, row) in self.frames.iter().enumerate() {
            for (qi, &code) in row.iter().take(k).enumerate() {
                out[qi][ti] = code;
            }
        }
        out
    }

    pub fn to_quantizer_rows(&self) -> Vec<Vec<u32>> {
        self.to_quantizer_layout()
    }

    pub fn from_quantizer_layout(codes: Vec<Vec<u32>>) -> Self {
        let num_quantizers = codes.len();
        let t = codes.first().map(|r| r.len()).unwrap_or(0);
        let mut frames = Vec::with_capacity(t);
        for ti in 0..t {
            let mut row = Vec::with_capacity(num_quantizers);
            for qi in 0..num_quantizers {
                row.push(codes[qi][ti]);
            }
            frames.push(row);
        }
        Self {
            frames,
            num_quantizers,
        }
    }
}
