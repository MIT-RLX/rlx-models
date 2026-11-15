// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! YuNet face detection on RLX: faces with a box, both eyes, nose and mouth corners.
//!
//! The network ([`model`]) is an RLX graph (rlx-ir ops, compiled and run by rlx-runtime on the
//! CPU, or any other device), its weights a safetensors file ([`weights`]). Pollen Robotics'
//! export (`face_detection_yunet_2026may.onnx`, the model the Reachy Mini daemon tracks faces
//! with) converts to it once with `rlx-yunet-convert` (feature `onnx`), which also backs the
//! parity test against ONNX Runtime; ONNX is not used to run the model.
//!
//! Pre- and post-processing are Pollen's `face_detector.py`: the BGR frame padded up to a
//! multiple of 32, pixel values as they are, score = √(class · objectness), boxes and landmarks
//! decoded per stride, greedy NMS.

pub mod model;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod weights;

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, ensure};
use rlx_runtime::{CompiledGraph, Session};

pub use rlx_runtime::Device;

pub use model::{OUTPUTS, build_graph};
pub use weights::Weights;

/// Feature strides of the three heads, and the padding the neck's upsampling needs.
pub const STRIDES: [usize; 3] = [8, 16, 32];
pub const MAX_STRIDE: usize = 32;

/// A face, in pixels of the frame given to [`YuNet::detect_bgr`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    /// x, y (top left), width, height.
    pub bbox: [f32; 4],
    /// The face's right eye (on the image's left), its left eye, nose, mouth right, mouth left.
    pub landmarks: [[f32; 2]; 5],
    pub score: f32,
}

impl Face {
    pub fn right_eye(&self) -> [f32; 2] {
        self.landmarks[0]
    }
    pub fn left_eye(&self) -> [f32; 2] {
        self.landmarks[1]
    }
    pub fn nose(&self) -> [f32; 2] {
        self.landmarks[2]
    }
    pub fn area(&self) -> f32 {
        self.bbox[2] * self.bbox[3]
    }
}

pub struct YuNet {
    weights: Weights,
    device: Device,
    /// Compiled per padded input size (a camera stream has one), weights already set.
    compiled: HashMap<(usize, usize), CompiledGraph>,
    /// Reused input buffer (planar BGR, padded).
    input: Vec<f32>,
    /// Faces scoring below this are dropped (Pollen: 0.6).
    pub score_threshold: f32,
    /// Overlap above which the weaker of two boxes is dropped (Pollen: 0.3).
    pub nms_threshold: f32,
}

impl YuNet {
    /// Weights from a safetensors file, run on `device`.
    pub fn load(weights: &Path, device: Device) -> Result<Self> {
        Ok(Self::new(Weights::load(weights)?, device))
    }

    pub fn new(weights: Weights, device: Device) -> Self {
        Self {
            weights,
            device,
            compiled: HashMap::new(),
            input: Vec::new(),
            score_threshold: 0.6,
            nms_threshold: 0.3,
        }
    }

    /// The raw outputs ([`OUTPUTS`] order) for a planar BGR input of `height` × `width`, both
    /// multiples of 32. Compiles for that size on first use.
    pub fn run(&mut self, input: &[f32], height: usize, width: usize) -> Result<Vec<Vec<f32>>> {
        ensure!(
            input.len() == 3 * height * width,
            "input is {} floats, want 3x{height}x{width}",
            input.len()
        );
        let graph = match self.compiled.entry((height, width)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let mut c = Session::new(self.device).compile(build_graph(height, width));
                for (name, v) in &self.weights.tensors {
                    c.set_param(name, v);
                }
                e.insert(c)
            }
        };
        let out = graph.run(&[("input", input)]);
        ensure!(
            out.len() == OUTPUTS.len(),
            "graph returned {} outputs",
            out.len()
        );
        Ok(out)
    }

    /// Faces in a packed BGR frame (`stride` bytes per row, at least `3 * width`).
    pub fn detect_bgr(
        &mut self,
        bgr: &[u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Vec<Face>> {
        let pw = width.div_ceil(MAX_STRIDE) * MAX_STRIDE;
        let ph = height.div_ceil(MAX_STRIDE) * MAX_STRIDE;
        // Planar BGR, zero padding right and below (the origin stays put).
        let mut x = std::mem::take(&mut self.input);
        x.clear();
        x.resize(3 * ph * pw, 0.0);
        for y in 0..height {
            let row = &bgr[y * stride..y * stride + 3 * width];
            for (i, px) in row.as_chunks::<3>().0.iter().enumerate() {
                for c in 0..3 {
                    x[(c * ph + y) * pw + i] = f32::from(px[c]);
                }
            }
        }
        let out = self.run(&x, ph, pw);
        self.input = x;
        Ok(self.decode(&out?, pw))
    }

    /// Faces from the raw outputs of a `width`-wide (padded) input.
    pub fn decode(&self, out: &[Vec<f32>], width: usize) -> Vec<Face> {
        let mut faces = Vec::new();
        for (level, s) in STRIDES.into_iter().enumerate() {
            let (cls, obj, bbox, kps) = (
                &out[level],
                &out[3 + level],
                &out[6 + level],
                &out[9 + level],
            );
            let cols = width / s;
            let sf = s as f32;
            for i in 0..cls.len() {
                // YuNet's confidence: the geometric mean of the class and objectness heads.
                let score = (cls[i].clamp(0.0, 1.0) * obj[i].clamp(0.0, 1.0)).sqrt();
                if score < self.score_threshold {
                    continue;
                }
                let (col, row) = ((i % cols) as f32, (i / cols) as f32);
                let b = &bbox[i * 4..i * 4 + 4];
                let (cx, cy) = ((col + b[0]) * sf, (row + b[1]) * sf);
                let (w, h) = (b[2].exp() * sf, b[3].exp() * sf);
                let k = &kps[i * 10..i * 10 + 10];
                let mut landmarks = [[0.0; 2]; 5];
                for (j, l) in landmarks.iter_mut().enumerate() {
                    *l = [(col + k[2 * j]) * sf, (row + k[2 * j + 1]) * sf];
                }
                faces.push(Face {
                    bbox: [cx - w / 2.0, cy - h / 2.0, w, h],
                    landmarks,
                    score,
                });
            }
        }
        nms(faces, self.nms_threshold)
    }
}

/// Greedy non-maximum suppression: best first, dropping boxes overlapping a kept one by more
/// than `iou` (intersection over union).
pub fn nms(mut faces: Vec<Face>, iou: f32) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::new();
    for f in faces {
        let overlaps = kept.iter().any(|k| {
            let w = ((k.bbox[0] + k.bbox[2]).min(f.bbox[0] + f.bbox[2]) - k.bbox[0].max(f.bbox[0]))
                .max(0.0);
            let h = ((k.bbox[1] + k.bbox[3]).min(f.bbox[1] + f.bbox[3]) - k.bbox[1].max(f.bbox[1]))
                .max(0.0);
            let inter = w * h;
            inter / (k.area() + f.area() - inter) > iou
        });
        if !overlaps {
            kept.push(f);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x: f32, score: f32) -> Face {
        Face {
            bbox: [x, 0.0, 10.0, 10.0],
            landmarks: [[0.0; 2]; 5],
            score,
        }
    }

    #[test]
    fn nms_keeps_the_best_of_overlapping_boxes() {
        let kept = nms(vec![face(0.0, 0.7), face(1.0, 0.9), face(50.0, 0.8)], 0.3);
        assert_eq!(
            kept.iter().map(|f| f.score).collect::<Vec<_>>(),
            vec![0.9, 0.8]
        );
    }

    /// Weights with a known answer: every conv the identity on its first channel (or the
    /// mean), so a constant image gives a predictable score everywhere. Checks that the graph
    /// compiles and runs at two sizes, with outputs of the right lengths.
    #[test]
    fn graph_runs_at_two_sizes() {
        let mut tensors = HashMap::new();
        for c in &model::CONVS {
            let n: usize = c.weight_shape().iter().product();
            tensors.insert(format!("{}.weight", c.name), vec![0.01; n]);
            tensors.insert(format!("{}.bias", c.name), vec![0.0; c.out_c]);
        }
        let mut net = YuNet::new(Weights { tensors }, Device::Cpu);
        for (h, w) in [(64, 96), (192, 320)] {
            let out = net.run(&vec![100.0; 3 * h * w], h, w).unwrap();
            for (i, s) in STRIDES.iter().enumerate() {
                let anchors = (h / s) * (w / s);
                assert_eq!(out[i].len(), anchors, "cls_{s}");
                assert_eq!(out[6 + i].len(), anchors * 4, "bbox_{s}");
                assert_eq!(out[9 + i].len(), anchors * 10, "kps_{s}");
                assert!(
                    out[i].iter().all(|v| (0.0..=1.0).contains(v)),
                    "cls_{s} not a probability"
                );
            }
        }
    }
}
