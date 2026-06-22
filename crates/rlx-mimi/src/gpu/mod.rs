#[cfg(feature = "gpu-codec")]
pub mod candle;

#[cfg(feature = "gpu-codec")]
pub use candle::GpuMimiCodec;
