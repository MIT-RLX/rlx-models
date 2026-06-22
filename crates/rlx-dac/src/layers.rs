use crate::ops::{
    conv_transpose1d, conv1d_same, conv1d_stride, residual_add, snake1d, snake1d_mode, tanh_inplace,
};
use ndarray::{Array1, Array2, Array3, ArrayView2};

#[derive(Debug, Clone)]
pub struct WnConv1d {
    pub weight: Array3<f32>,
    pub bias: Option<Array1<f32>>,
    pub stride: usize,
    pub pad: usize,
    pub dilation: usize,
    pub same_length: bool,
}

impl WnConv1d {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        if self.same_length {
            conv1d_same(
                x,
                self.weight.view(),
                self.bias.as_ref().map(|b| b.view()),
                self.pad,
                1,
                self.dilation,
            )
        } else {
            conv1d_stride(
                x,
                self.weight.view(),
                self.bias.as_ref().map(|b| b.view()),
                self.stride,
                self.pad,
            )
        }
    }
}

#[derive(Debug, Clone)]
pub struct WnConvTranspose1d {
    pub weight: Array3<f32>,
    pub bias: Option<Array1<f32>>,
    pub stride: usize,
    pub pad: usize,
    /// Extra right-shift of the (length-preserving) output crop window, in output
    /// samples. 0 = PyTorch `ConvTranspose1d` convention (DAC). TSAC's C decoder
    /// centers the kernel at `K/2` instead of `stride/2`, i.e. `out_offset =
    /// pad` (= `stride/2`). Must satisfy `out_offset <= pad`.
    pub out_offset: usize,
}

impl WnConvTranspose1d {
    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        conv_transpose1d(
            x,
            self.weight.view(),
            self.bias.as_ref().map(|b| b.view()),
            self.stride,
            self.pad,
            self.out_offset,
        )
    }
}

#[derive(Debug, Clone)]
pub struct ResidualUnit {
    pub(crate) snake1_alpha: Array1<f32>,
    pub(crate) conv1: WnConv1d,
    pub(crate) snake2_alpha: Array1<f32>,
    pub(crate) conv2: WnConv1d,
}

impl ResidualUnit {
    /// Construct from explicit weights (e.g. when adapting a non-safetensors
    /// checkpoint such as TSAC's q8 model). `conv1`/`conv2` carry final
    /// (weight-normalized) weights as `[C_out, C_in, K]`.
    pub fn from_parts(
        snake1_alpha: Array1<f32>,
        conv1: WnConv1d,
        snake2_alpha: Array1<f32>,
        conv2: WnConv1d,
    ) -> Self {
        Self {
            snake1_alpha,
            conv1,
            snake2_alpha,
            conv2,
        }
    }

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        self.forward_mode(x, false)
    }

    pub fn forward_mode(&self, x: ArrayView2<f32>, c_snake: bool) -> Array2<f32> {
        let mut h = snake1d_mode(x, self.snake1_alpha.view(), c_snake);
        h = self.conv1.forward(h.view());
        h = snake1d_mode(h.view(), self.snake2_alpha.view(), c_snake);
        h = self.conv2.forward(h.view());
        residual_add(x, h.view())
    }
}

#[derive(Debug, Clone)]
pub struct EncoderBlock {
    pub(crate) residual_units: [ResidualUnit; 3],
    pub(crate) snake_alpha: Array1<f32>,
    pub(crate) downsample: WnConv1d,
}

impl EncoderBlock {
    /// Construct from explicit parts: three residual units, snake α, downsample.
    pub fn from_parts(
        residual_units: [ResidualUnit; 3],
        snake_alpha: Array1<f32>,
        downsample: WnConv1d,
    ) -> Self {
        Self {
            residual_units,
            snake_alpha,
            downsample,
        }
    }

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let mut h = self.residual_units[0].forward(x);
        h = self.residual_units[1].forward(h.view());
        h = self.residual_units[2].forward(h.view());
        h = snake1d(h.view(), self.snake_alpha.view());
        self.downsample.forward(h.view())
    }
}

#[derive(Debug, Clone)]
pub struct DecoderBlock {
    pub(crate) snake_alpha: Array1<f32>,
    pub(crate) upsample: WnConvTranspose1d,
    pub(crate) residual_units: [ResidualUnit; 3],
}

impl DecoderBlock {
    /// Construct from explicit parts (snake α, upsample conv-transpose, and the
    /// three residual units).
    pub fn from_parts(
        snake_alpha: Array1<f32>,
        upsample: WnConvTranspose1d,
        residual_units: [ResidualUnit; 3],
    ) -> Self {
        Self {
            snake_alpha,
            upsample,
            residual_units,
        }
    }

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        self.forward_mode(x, false)
    }

    pub fn forward_mode(&self, x: ArrayView2<f32>, c_snake: bool) -> Array2<f32> {
        let mut h = snake1d_mode(x, self.snake_alpha.view(), c_snake);
        h = self.upsample.forward(h.view());
        h = self.residual_units[0].forward_mode(h.view(), c_snake);
        h = self.residual_units[1].forward_mode(h.view(), c_snake);
        self.residual_units[2].forward_mode(h.view(), c_snake)
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    pub(crate) stem: WnConv1d,
    pub(crate) blocks: Vec<EncoderBlock>,
    pub(crate) snake_alpha: Array1<f32>,
    pub(crate) head: WnConv1d,
}

impl Encoder {
    /// Construct from explicit parts: stem conv, downsample blocks, snake α, head.
    pub fn from_parts(
        stem: WnConv1d,
        blocks: Vec<EncoderBlock>,
        snake_alpha: Array1<f32>,
        head: WnConv1d,
    ) -> Self {
        Self {
            stem,
            blocks,
            snake_alpha,
            head,
        }
    }

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let mut h = self.stem.forward(x);
        for block in &self.blocks {
            h = block.forward(h.view());
        }
        h = snake1d(h.view(), self.snake_alpha.view());
        self.head.forward(h.view())
    }
}

#[derive(Debug, Clone)]
pub struct Decoder {
    pub(crate) stem: WnConv1d,
    pub(crate) blocks: Vec<DecoderBlock>,
    pub(crate) snake_alpha: Array1<f32>,
    pub(crate) head: WnConv1d,
    /// Use TSAC's vendored snake indexing (see [`snake1d_mode`]). DAC = false.
    pub(crate) c_snake: bool,
}

impl Decoder {
    /// Construct from explicit parts: stem conv, upsample blocks, final snake α,
    /// and head conv. Lets external crates (e.g. rlx-tsac) build a DAC decoder
    /// from a non-safetensors checkpoint and reuse the graph + ndarray paths.
    /// `c_snake` selects TSAC's snake convention (false for standard DAC).
    pub fn from_parts(
        stem: WnConv1d,
        blocks: Vec<DecoderBlock>,
        snake_alpha: Array1<f32>,
        head: WnConv1d,
        c_snake: bool,
    ) -> Self {
        Self {
            stem,
            blocks,
            snake_alpha,
            head,
            c_snake,
        }
    }

    pub fn c_snake(&self) -> bool {
        self.c_snake
    }

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let cs = self.c_snake;
        let mut h = self.stem.forward(x);
        for block in &self.blocks {
            h = block.forward_mode(h.view(), cs);
        }
        h = snake1d_mode(h.view(), self.snake_alpha.view(), cs);
        h = self.head.forward(h.view());
        tanh_inplace(&mut h);
        h
    }
}
