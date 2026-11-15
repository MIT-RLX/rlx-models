// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! YuNet as an RLX graph.
//!
//! - Backbone: a 3×3 stride-2 conv, then depthwise-separable blocks (1×1 conv, depthwise 3×3,
//!   ReLU) in six stages, with 2×2 max pools after stages 0, 2, 3 and 4: features at strides
//!   8 (stage 3), 16 (stage 4) and 32 (stage 5).
//! - Neck: a top-down pyramid, each level a separable block, the coarser level upsampled 2×
//!   (nearest) and added to the next.
//! - Heads per stride: class, objectness (sigmoid), box (4) and landmarks (10), each a 1×1
//!   conv then a 3×3 conv (depthwise but for the 1-channel heads), as `[1, anchors, k]`.
//!
//! Convolutions are named in the order the ONNX export lists them, so the converter can map
//! the export's weights by position and check every shape.

use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};

/// One convolution: name, input and output channels, kernel, stride, groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvSpec {
    pub name: &'static str,
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
    pub stride: usize,
    pub groups: usize,
}

impl ConvSpec {
    /// `[out, in/groups, k, k]`.
    pub fn weight_shape(&self) -> [usize; 4] {
        [self.out_c, self.in_c / self.groups, self.k, self.k]
    }
}

const fn conv(
    name: &'static str,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    groups: usize,
) -> ConvSpec {
    ConvSpec {
        name,
        in_c,
        out_c,
        k,
        stride,
        groups,
    }
}
const fn pw(name: &'static str, in_c: usize, out_c: usize) -> ConvSpec {
    conv(name, in_c, out_c, 1, 1, 1)
}
const fn dw(name: &'static str, c: usize) -> ConvSpec {
    conv(name, c, c, 3, 1, c)
}

/// Every convolution, in the ONNX export's order.
pub const CONVS: [ConvSpec; 53] = [
    conv("backbone.0.conv1", 3, 16, 3, 2, 1),
    pw("backbone.0.conv2.pw", 16, 16),
    dw("backbone.0.conv2.dw", 16),
    pw("backbone.1.conv1.pw", 16, 16),
    dw("backbone.1.conv1.dw", 16),
    pw("backbone.1.conv2.pw", 16, 32),
    dw("backbone.1.conv2.dw", 32),
    pw("backbone.2.conv1.pw", 32, 32),
    dw("backbone.2.conv1.dw", 32),
    pw("backbone.2.conv2.pw", 32, 64),
    dw("backbone.2.conv2.dw", 64),
    pw("backbone.3.conv1.pw", 64, 64),
    dw("backbone.3.conv1.dw", 64),
    pw("backbone.3.conv2.pw", 64, 64),
    dw("backbone.3.conv2.dw", 64),
    pw("backbone.4.conv1.pw", 64, 64),
    dw("backbone.4.conv1.dw", 64),
    pw("backbone.4.conv2.pw", 64, 64),
    dw("backbone.4.conv2.dw", 64),
    pw("backbone.5.conv1.pw", 64, 64),
    dw("backbone.5.conv1.dw", 64),
    pw("backbone.5.conv2.pw", 64, 64),
    dw("backbone.5.conv2.dw", 64),
    pw("neck.2.pw", 64, 64),
    dw("neck.2.dw", 64),
    pw("neck.1.pw", 64, 64),
    dw("neck.1.dw", 64),
    pw("neck.0.pw", 64, 64),
    dw("neck.0.dw", 64),
    pw("head.cls.0.pw", 64, 1),
    conv("head.cls.0.conv", 1, 1, 3, 1, 1),
    pw("head.cls.1.pw", 64, 1),
    conv("head.cls.1.conv", 1, 1, 3, 1, 1),
    pw("head.cls.2.pw", 64, 1),
    conv("head.cls.2.conv", 1, 1, 3, 1, 1),
    pw("head.bbox.0.pw", 64, 4),
    dw("head.bbox.0.conv", 4),
    pw("head.bbox.1.pw", 64, 4),
    dw("head.bbox.1.conv", 4),
    pw("head.bbox.2.pw", 64, 4),
    dw("head.bbox.2.conv", 4),
    pw("head.obj.0.pw", 64, 1),
    conv("head.obj.0.conv", 1, 1, 3, 1, 1),
    pw("head.obj.1.pw", 64, 1),
    conv("head.obj.1.conv", 1, 1, 3, 1, 1),
    pw("head.obj.2.pw", 64, 1),
    conv("head.obj.2.conv", 1, 1, 3, 1, 1),
    pw("head.kps.0.pw", 64, 10),
    dw("head.kps.0.conv", 10),
    pw("head.kps.1.pw", 64, 10),
    dw("head.kps.1.conv", 10),
    pw("head.kps.2.pw", 64, 10),
    dw("head.kps.2.conv", 10),
];

/// The graph's outputs, in this order (the ONNX export's): `[1, anchors, k]` each, anchors
/// row-major over the stride's grid.
pub const OUTPUTS: [&str; 12] = [
    "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32",
    "kps_8", "kps_16", "kps_32",
];

pub fn spec(name: &str) -> &'static ConvSpec {
    CONVS
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no conv {name}"))
}

/// The forward pass for a `1 × 3 × height × width` input (BGR, 0-255; both multiples of 32):
/// a graph with input `"input"`, params `NAME.weight` / `NAME.bias` for every [`CONVS`] entry,
/// and the twelve [`OUTPUTS`].
pub fn build_graph(height: usize, width: usize) -> Graph {
    build(height, width, false)
}

/// Intermediate feature maps [`build_graph_with_taps`] adds after the [`OUTPUTS`].
pub const TAPS: [&str; 8] = ["s8", "s16", "s32", "n32", "up32", "sum16", "n16", "n8"];

/// [`build_graph`] with the backbone and neck feature maps ([`TAPS`]) as extra outputs, to
/// compare devices or exports layer by layer.
pub fn build_graph_with_taps(height: usize, width: usize) -> Graph {
    build(height, width, true)
}

fn build(height: usize, width: usize, taps: bool) -> Graph {
    assert!(
        height.is_multiple_of(32) && width.is_multiple_of(32),
        "YuNet input {width}x{height}: not a multiple of 32"
    );
    let mut g = Graph::new("yunet");
    let x = g.input("input", Shape::new(&[1, 3, height, width], DType::F32));
    let conv = |g: &mut Graph, x: NodeId, name: &str| -> NodeId {
        let s = spec(name);
        let w = g.param(
            format!("{name}.weight"),
            Shape::new(&s.weight_shape(), DType::F32),
        );
        let b = g.param(format!("{name}.bias"), Shape::new(&[s.out_c], DType::F32));
        let pad = s.k / 2;
        let y = g.conv2d(
            x,
            w,
            [s.k, s.k],
            [s.stride, s.stride],
            [pad, pad],
            [1, 1],
            s.groups,
        );
        let b = g.reshape_(b, vec![1, s.out_c as i64, 1, 1]);
        g.add(y, b)
    };
    // 1×1 conv, depthwise 3×3, ReLU.
    let block = |g: &mut Graph, x: NodeId, name: &str| -> NodeId {
        let y = conv(g, x, &format!("{name}.pw"));
        let y = conv(g, y, &format!("{name}.dw"));
        g.relu(y)
    };
    let pool =
        |g: &mut Graph, x: NodeId| g.pool2d(x, rlx_ir::op::ReduceOp::Max, [2, 2], [2, 2], [0, 0]);

    let y = conv(&mut g, x, "backbone.0.conv1");
    let y = g.relu(y);
    let y = block(&mut g, y, "backbone.0.conv2");
    let y = pool(&mut g, y);
    let y = block(&mut g, y, "backbone.1.conv1");
    let y = block(&mut g, y, "backbone.1.conv2");
    let y = block(&mut g, y, "backbone.2.conv1");
    let y = block(&mut g, y, "backbone.2.conv2");
    let y = pool(&mut g, y);
    let y = block(&mut g, y, "backbone.3.conv1");
    let s8 = block(&mut g, y, "backbone.3.conv2");
    let y = pool(&mut g, s8);
    let y = block(&mut g, y, "backbone.4.conv1");
    let s16 = block(&mut g, y, "backbone.4.conv2");
    let y = pool(&mut g, s16);
    let y = block(&mut g, y, "backbone.5.conv1");
    let s32 = block(&mut g, y, "backbone.5.conv2");

    let n32 = block(&mut g, s32, "neck.2");
    let up32 = upsample2x(&mut g, n32);
    let sum16 = g.add(s16, up32);
    let n16 = block(&mut g, sum16, "neck.1");
    let up = upsample2x(&mut g, n16);
    let y = g.add(s8, up);
    let n8 = block(&mut g, y, "neck.0");

    let levels = [n8, n16, n32];
    let mut outs = Vec::new();
    for (head, k, sigmoid) in [
        ("cls", 1, true),
        ("obj", 1, true),
        ("bbox", 4, false),
        ("kps", 10, false),
    ] {
        for (i, &feat) in levels.iter().enumerate() {
            let y = conv(&mut g, feat, &format!("head.{head}.{i}.pw"));
            let y = conv(&mut g, y, &format!("head.{head}.{i}.conv"));
            // NCHW → [1, H·W, k].
            let y = g.transpose_(y, vec![0, 2, 3, 1]);
            let y = g.reshape_(y, vec![1, -1, k]);
            outs.push(if sigmoid { g.sigmoid(y) } else { y });
        }
    }
    if taps {
        outs.extend([s8, s16, s32, n32, up32, sum16, n16, n8]);
    }
    g.set_outputs(outs);
    g
}

/// Nearest-neighbour 2× upsample (ONNX `Resize`, nearest, asymmetric, floor, scale 2).
fn upsample2x(g: &mut Graph, x: NodeId) -> NodeId {
    let s = g.shape(x).clone();
    let [n, c, h, w] = [0, 1, 2, 3].map(|i| s.dim(i).unwrap_static());
    g.add_node(
        rlx_ir::op::Op::ResizeNearest2x,
        vec![x],
        Shape::new(&[n, c, 2 * h, 2 * w], DType::F32),
    )
}
