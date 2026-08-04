//! Byte-level tokenizer: token id == raw byte, so the vocabulary is exactly
//! 256 and encoding/decoding are trivial and lossless. This keeps the showcase
//! fully self-contained (no BPE training, no `tokenizer.json`) and maps 1:1 to
//! the corpus bytes, which lets the data loader `mmap` the file and slice
//! windows directly.

/// Vocabulary size for the byte-level tokenizer.
pub const VOCAB: usize = 256;

/// Encode text to byte-level token ids.
pub fn encode(text: &str) -> Vec<u32> {
    text.bytes().map(|b| b as u32).collect()
}

/// Decode byte-level token ids back to a UTF-8 string (lossy for invalid
/// sequences, which can occur mid-generation before a multi-byte char closes).
pub fn decode(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().map(|&i| (i & 0xff) as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Decode a single id to its byte.
pub fn byte_of(id: u32) -> u8 {
    (id & 0xff) as u8
}
