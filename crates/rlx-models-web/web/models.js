// What each model needs fetched, relative to its checkpoint directory.
export const MODELS = {
  "moonshine-streaming-tiny": {
    kind: "asr", dir: "asr/moonshine-streaming-tiny", label: "Moonshine streaming tiny (176 MB)",
    files: ["config.json", "model.safetensors", "tokenizer.json"],
  },
  "moonshine-streaming-small": {
    kind: "asr", dir: "asr/moonshine-streaming-small", label: "Moonshine streaming small (561 MB)",
    files: ["config.json", "model.safetensors", "tokenizer.json"],
  },
  "moonshine-streaming-medium": {
    kind: "asr", dir: "asr/moonshine-streaming-medium", label: "Moonshine streaming medium (1.06 GB)",
    files: ["config.json", "model.safetensors", "tokenizer.json"],
  },
  "embeddinggemma": {
    kind: "embed", dir: "embed/embeddinggemma-300m", label: "EmbeddingGemma-300M (1.21 GB)",
    files: ["config.json", "model.safetensors", "tokenizer.json", "2_Dense/model.safetensors",
      "3_Dense/model.safetensors", "sentence_bert_config.json", "config_sentence_transformers.json"],
  },
  "jina-v5": {
    kind: "embed", dir: "embed/jina-embeddings-v5-text-nano", label: "jina-embeddings-v5-text-nano (424 MB)",
    files: ["config.json", "model.safetensors", "tokenizer.json", "config_sentence_transformers.json",
      "adapters/retrieval/adapter_config.json", "adapters/retrieval/adapter_model.safetensors"],
  },
  "lfm2.5-230m": {
    kind: "gen", dir: "lm/lfm2.5-230m", label: "LFM2.5-230M (459 MB)",
    files: ["config.json", "model.safetensors", "tokenizer.json"],
  },
  "qwen3.5-0.8b-q4km": {
    kind: "qwen35", dir: "lm/qwen3.5-0.8b-q4km", label: "Qwen3.5-0.8B Q4_K_M (520 MB)",
    // model.gguf: unsloth/Qwen3.5-0.8B-GGUF; tokenizer.json: Qwen/Qwen3.5-0.8B.
    files: ["model.gguf", "tokenizer.json"],
  },
  "qwen3.5-0.8b-bf16": {
    kind: "qwen35", dir: "lm/qwen3.5-0.8b-bf16", label: "Qwen3.5-0.8B BF16 (1.5 GB)",
    // model.gguf: unsloth/Qwen3.5-0.8B-GGUF (BF16); tokenizer.json: Qwen/Qwen3.5-0.8B.
    files: ["model.gguf", "tokenizer.json"],
  },
  "lfm2.5-350m": {
    kind: "gen", dir: "lm/lfm2.5-350m", label: "LFM2.5-350M (709 MB)",
    files: ["config.json", "model.safetensors", "tokenizer.json"],
  },
};
