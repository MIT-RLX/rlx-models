//! Streaming duplex inference — frame-at-a-time PCM in/out.

mod duplex;
mod protocol;
mod sync;

#[cfg(feature = "tokio")]
mod tokio_impl;

pub use duplex::{DuplexStreamEngine, StreamStepOutput};
pub use protocol::{
    WsMsgType, decode_ws_message, encode_ws_audio, encode_ws_handshake, encode_ws_text,
};
pub use sync::{StreamCommand, StreamEvent, StreamHandle, StreamStats, spawn_duplex_stream};

#[cfg(feature = "tokio")]
pub use tokio_impl::{TokioStreamCommand, TokioStreamEvent, TokioStreamHandle, spawn_duplex_tokio};

/// Samples per 12.5 Hz Mimi frame @ 24 kHz.
pub const FRAME_SAMPLES: usize = 1920;
