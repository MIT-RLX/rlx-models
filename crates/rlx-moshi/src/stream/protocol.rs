/// Kyutai-style binary WebSocket message types (subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WsMsgType {
    Handshake = 0,
    Audio = 1,
    Text = 2,
    Control = 3,
    Error = 5,
    Ping = 6,
}

impl WsMsgType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Handshake),
            1 => Some(Self::Audio),
            2 => Some(Self::Text),
            3 => Some(Self::Control),
            5 => Some(Self::Error),
            6 => Some(Self::Ping),
            _ => None,
        }
    }
}

pub fn encode_ws_handshake() -> Vec<u8> {
    let mut msg = vec![WsMsgType::Handshake as u8];
    msg.extend_from_slice(&0u32.to_le_bytes());
    msg.extend_from_slice(&1u32.to_le_bytes());
    msg
}

pub fn encode_ws_audio(pcm: &[f32]) -> Vec<u8> {
    let mut msg = Vec::with_capacity(1 + pcm.len() * 4);
    msg.push(WsMsgType::Audio as u8);
    for &s in pcm {
        msg.extend_from_slice(&s.to_le_bytes());
    }
    msg
}

pub fn encode_ws_text(text: &str) -> Vec<u8> {
    let mut msg = Vec::with_capacity(1 + text.len());
    msg.push(WsMsgType::Text as u8);
    msg.extend_from_slice(text.as_bytes());
    msg
}

/// Decode client audio message → PCM samples.
pub fn decode_ws_message(data: &[u8]) -> anyhow::Result<Option<(WsMsgType, Vec<f32>)>> {
    if data.is_empty() {
        return Ok(None);
    }
    let Some(kind) = WsMsgType::from_u8(data[0]) else {
        anyhow::bail!("unknown ws msg type {}", data[0]);
    };
    match kind {
        WsMsgType::Audio => {
            let payload = &data[1..];
            ensure_audio_payload(payload)?;
            let mut pcm = Vec::with_capacity(payload.len() / 4);
            for chunk in payload.chunks_exact(4) {
                pcm.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(Some((kind, pcm)))
        }
        WsMsgType::Ping => Ok(Some((kind, Vec::new()))),
        WsMsgType::Control if data.len() >= 2 && data[1] == 0 => Ok(Some((kind, Vec::new()))),
        _ => Ok(Some((kind, Vec::new()))),
    }
}

fn ensure_audio_payload(payload: &[u8]) -> anyhow::Result<()> {
    if !payload.len().is_multiple_of(4) {
        anyhow::bail!("audio payload length {} not multiple of 4", payload.len());
    }
    Ok(())
}
