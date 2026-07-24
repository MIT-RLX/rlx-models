// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Exact ternary `{−1,0,+1}` for host train path (bake TQ2 / fused kernels).
//!
//! Prefer applying via [`crate::WakeCnnWeights::ternarize`] after SGD.

/// Which WakeCnn weight tensors to ternarize (biases stay f32).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TernaryOpts {
    pub conv: bool,
    pub fc: bool,
    /// Keep the top `keep_frac` of |w| as ±1; rest → 0.
    pub keep_frac: f32,
}

impl Default for TernaryOpts {
    fn default() -> Self {
        Self {
            conv: false,
            fc: true,
            keep_frac: 1.0 / 3.0,
        }
    }
}

impl TernaryOpts {
    pub fn fc_only() -> Self {
        Self::default()
    }

    pub fn all_weights() -> Self {
        Self {
            conv: true,
            fc: true,
            keep_frac: 1.0 / 3.0,
        }
    }

    /// Parse `fc` | `all` | `conv`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fc" | "fc-only" | "default" => Some(Self::fc_only()),
            "all" | "conv+fc" => Some(Self::all_weights()),
            "conv" | "conv-only" => Some(Self {
                conv: true,
                fc: false,
                keep_frac: 1.0 / 3.0,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TernaryStats {
    pub tensors: usize,
    pub elems: usize,
    pub nonzero: usize,
}

/// Exact ternary values `{−1, 0, +1}` (eligible for rlx-bake TQ2_0).
pub fn is_ternary_f32(v: &[f32]) -> bool {
    v.iter().all(|&x| x == -1.0 || x == 0.0 || x == 1.0)
}

pub fn ternarize(w: &[f32], keep_frac: f32) -> Vec<f32> {
    if w.is_empty() {
        return Vec::new();
    }
    let keep = keep_frac.clamp(0.01, 1.0);
    let mut abs: Vec<f32> = w.iter().map(|v| v.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_keep = (((abs.len() as f32) * keep).round() as usize).clamp(1, abs.len());
    let thr = if n_keep >= abs.len() {
        0.0
    } else {
        abs[abs.len() - n_keep]
    };
    w.iter()
        .map(|&v| {
            if v.abs() < thr {
                0.0
            } else if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

pub fn ternarize_inplace(w: &mut [f32], keep_frac: f32) {
    let t = ternarize(w, keep_frac);
    w.copy_from_slice(&t);
}
