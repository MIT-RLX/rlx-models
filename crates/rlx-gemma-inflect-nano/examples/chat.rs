//! Interactive voice chat: type a message, Gemma 3 270M replies, and
//! Inflect-Nano speaks the reply out loud — all local, Metal by default.
//!
//! ```sh
//! just fetch-gemma3-270m
//! # export the Inflect bundle once (see crates/rlx-inflect-nano/README.md)
//! cargo run --release -p rlx-gemma-inflect-nano --features metal \
//!   --example chat -- --device metal --tts-device metal
//! ```
//!
//! Playback streams: each sentence of the reply is synthesized on the GPU and
//! pushed into a live output buffer that starts playing once ~1s is queued, so
//! you hear the answer almost immediately and it keeps flowing as Gemma's later
//! sentences are vocoded. In-session commands: `/reset` clears history,
//! `/quit` (or Ctrl-D) exits.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_cli::{ChatMessage, parse_gemma_device, req};
use rlx_gemma::GemmaRunner;
use rlx_gemma_inflect_nano::{encode_chat_turns, resolve_tts_device};
use rlx_inflect_nano::{InferOpts, InflectNano};
use rlx_qwen3::SampleOpts;

fn usage() {
    eprintln!(
        "chat — talk to Gemma 3 270M and hear the reply (Inflect-Nano TTS)\n\
         \n\
         Type a message and press enter. Commands: /reset, /quit (or Ctrl-D).\n\
         \n\
         Flags:\n\
           --gemma-gguf PATH     Gemma GGUF (default: RLX_GEMMA3_GGUF or /tmp/rlx-weights/gemma-3-270m.gguf)\n\
           --tokenizer PATH      tokenizer.json (default: sibling of GGUF or RLX_GEMMA3_TOKENIZER)\n\
           --inflect-data PATH   Inflect RLX bundle (default: RLX_INFLECT_NANO_DATA or weights/inflect-nano-rlx)\n\
           --system TEXT         Optional system prompt\n\
           --device DEVICE       Gemma backend (cpu, metal, mlx, …; default metal)\n\
           --tts-device DEVICE   Inflect vocoder backend (auto, cpu, metal, mlx, …; default auto)\n\
           --max-tokens N        Max new tokens per reply (default: 128)\n\
           --max-seq N           Compile context length (default: 1024)\n\
           --temp F              Sampling temperature; 0 = greedy (default: 0)\n\
           --prime-secs F        Audio buffered before playback starts (default: 4.0)\n\
           --speed F             Speaking rate at synthesis; <1 slower, >1 faster (default: 0.667 = 1.5x slower)\n\
           --sentence-pause F    Silence after each spoken sentence, seconds (default: 0.45)\n\
           --packed              Packed GGUF decode (default on)\n\
           --no-packed           Disable packed GGUF\n\
           --first-sentence      Speak only the first sentence of each reply\n\
           --no-audio            Print replies but skip synthesis + playback\n"
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        usage();
        return Ok(());
    }

    let mut gemma_gguf = rlx_gemma_inflect_nano::default_gemma_gguf();
    let mut tokenizer: Option<PathBuf> = rlx_gemma_inflect_nano::default_gemma_tokenizer();
    let mut inflect_data = rlx_gemma_inflect_nano::default_inflect_data_dir();
    let mut system: Option<String> = None;
    let mut device = "metal".to_string();
    let mut tts_device = "auto".to_string();
    let mut max_tokens = 128usize;
    let mut max_seq = 1024usize;
    let mut temp = 0.0f32;
    let mut prime_secs = 4.0f32;
    let mut speed = 0.667f32; // ≈1.5× slower — applied at SYNTHESIS (acoustic durations), not playback
    let mut sentence_pause = 0.45f32; // silence after each spoken sentence
    let mut packed = true;
    let mut first_sentence = false;
    let mut no_audio = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gemma-gguf" => gemma_gguf = req(&args, &mut i)?.into(),
            "--tokenizer" => tokenizer = Some(req(&args, &mut i)?.into()),
            "--inflect-data" => inflect_data = req(&args, &mut i)?.into(),
            "--system" => system = Some(req(&args, &mut i)?),
            "--device" => device = req(&args, &mut i)?,
            "--tts-device" => tts_device = req(&args, &mut i)?,
            "--max-tokens" => {
                max_tokens = req(&args, &mut i)?.parse().context("--max-tokens: usize")?;
            }
            "--max-seq" => max_seq = req(&args, &mut i)?.parse().context("--max-seq: usize")?,
            "--temp" => temp = req(&args, &mut i)?.parse().context("--temp: f32")?,
            "--prime-secs" => {
                prime_secs = req(&args, &mut i)?.parse().context("--prime-secs: f32")?;
            }
            "--speed" => speed = req(&args, &mut i)?.parse().context("--speed: f32")?,
            "--sentence-pause" => {
                sentence_pause = req(&args, &mut i)?.parse().context("--sentence-pause: f32")?;
            }
            "--packed" => {
                packed = true;
                i += 1;
            }
            "--no-packed" => {
                packed = false;
                i += 1;
            }
            "--first-sentence" => {
                first_sentence = true;
                i += 1;
            }
            "--no-audio" => {
                no_audio = true;
                i += 1;
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let tok = tokenizer.as_deref();
    rlx_gemma_inflect_nano::ensure_paths_exist(&gemma_gguf, tok, &inflect_data)?;

    // Load the tokenizer ONCE for streaming decode. The `decode_*_auto` helpers
    // reload tokenizer.json on every call (~370ms/token), which — not the model —
    // was the chat's real bottleneck (Metal decode itself is ~32 tok/s).
    let tok_path = tokenizer
        .clone()
        .or_else(|| rlx_qwen35::resolve_tokenizer_path(&gemma_gguf, None))
        .context("no tokenizer.json found for streaming decode")?;
    let dec_tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer {tok_path:?}: {e}"))?;

    let lm_device = parse_gemma_device(&device)?;
    let tts_dev = resolve_tts_device(&tts_device, lm_device)?;

    eprintln!(
        "[chat] gemma={gemma_gguf:?} inflect={inflect_data:?} lm={lm_device:?} tts={tts_dev:?} packed={packed}"
    );
    eprintln!("[chat] loading model + TTS bundle…");

    let sample = if temp > 0.0 {
        SampleOpts::temperature(temp, 0)
    } else {
        SampleOpts::greedy()
    };
    let mut runner = GemmaRunner::builder()
        .weights(&gemma_gguf)
        .device(lm_device)
        .max_seq(max_seq)
        .stream(true)
        .sample(sample)
        .packed_weights(packed)
        .build()?;
    // End-of-generation ids (Gemma3: 1, 106, 212 + GGUF eos). `generate` runs a
    // fixed token count and never stops on its own, so we truncate the reply at
    // the first end-of-turn ourselves.
    let eog = runner.config().eog_token_ids.clone();

    let inflect = InflectNano::load_from_dir(&inflect_data)?;
    // `speed < 1.0` slows the acoustic durations (natural pitch), so 0.667 ≈ 1.5× slower.
    let opts = InferOpts::with_speed(speed);

    // Conversation history. A seeded system turn (if any) survives /reset.
    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(s) = &system {
        history.push(ChatMessage::system(s.clone()));
    }
    let base_len = history.len();

    eprintln!("[chat] ready — type a message (/reset clears history, /quit or Ctrl-D exits)\n");

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("you> ");
        std::io::stdout().flush().ok();
        let Some(line) = lines.next() else {
            println!();
            break; // EOF (Ctrl-D)
        };
        let line = line.context("reading stdin")?;
        let user = line.trim();
        if user.is_empty() {
            continue;
        }
        match user {
            "/quit" | "/exit" | ":q" => break,
            "/reset" => {
                history.truncate(base_len);
                eprintln!("[chat] history cleared");
                continue;
            }
            _ => {}
        }

        history.push(ChatMessage::user(user.to_string()));
        let prompt_ids = encode_chat_turns(&gemma_gguf, tok, &history)?;

        // Open playback up front so each finished sentence can be fed to TTS the
        // moment the LLM produces it (pipelined) — we don't wait for the full reply.
        let mut player: Option<audio_out::StreamPlayer> = None;
        let mut fallback: Option<Vec<f32>> = None;
        if !no_audio {
            match audio_out::StreamPlayer::start(24_000, prime_secs) {
                Ok(p) => player = Some(p),
                Err(e) => {
                    eprintln!("[chat] live audio unavailable ({e}); using afplay");
                    fallback = Some(Vec::new());
                }
            }
        }

        print!("gemma> ");
        std::io::stdout().flush().ok();
        // Decode incrementally with the cached tokenizer; stream text to the
        // console AND push completed sentences to TTS as they arrive.
        let mut reply_ids: Vec<u32> = Vec::new();
        let mut shown = String::new(); // everything decoded so far (console)
        let mut pending = String::new(); // decoded text not yet sent to TTS
        let mut stopped = false;
        let mut spoke = false; // for --first-sentence
        let t_lm = Instant::now();
        runner.generate(&prompt_ids, max_tokens, |tok_id| {
            if stopped {
                return;
            }
            if eog.contains(&tok_id) {
                stopped = true;
                return;
            }
            reply_ids.push(tok_id);
            let Ok(text) = dec_tok.decode(&reply_ids, true) else {
                return;
            };
            // char-safe delta so a split multibyte token never panics on a byte slice
            let new: String = text.chars().skip(shown.chars().count()).collect();
            if !new.is_empty() {
                print!("{new}");
                std::io::stdout().flush().ok();
                pending.push_str(&new);
            }
            shown = text;
            // Speak each finished sentence immediately (split on . ! ? / newline).
            if !no_audio && !(first_sentence && spoke) {
                for sent in drain_sentences(&mut pending) {
                    synth_emit(&inflect, &opts, tts_dev, &sent, sentence_pause, &player, &mut fallback);
                    spoke = true;
                    if first_sentence {
                        pending.clear();
                        break;
                    }
                }
            }
        })?;
        let lm_secs = t_lm.elapsed().as_secs_f32();
        println!();

        let reply = shown.trim().to_string();
        let n_tok = reply_ids.len();
        eprintln!(
            "[gemma] {n_tok} tok in {lm_secs:.2}s ({:.1} tok/s)",
            n_tok as f32 / lm_secs.max(1e-6)
        );
        history.push(ChatMessage::assistant(reply.clone()));

        // Speak the trailing partial sentence, then drain playback to the end.
        if !no_audio && !(first_sentence && spoke) {
            let tail = pending.trim();
            if !tail.is_empty() {
                synth_emit(&inflect, &opts, tts_dev, tail, sentence_pause, &player, &mut fallback);
            }
        }
        if let Some(p) = &player {
            if let Err(e) = p.finish() {
                eprintln!("[chat] drain failed: {e}");
            }
        } else if let Some(buf) = &fallback {
            if !buf.is_empty() {
                rlx_gemma_inflect_nano::play_samples(buf, 24_000).ok();
            }
        }
        println!();
    }

    eprintln!("[chat] bye");
    Ok(())
}

/// Synthesize one sentence on `tts_dev` and queue it for playback, followed by
/// `sentence_pause` seconds of silence so the speech breathes. The cached/bucketed
/// vocoder graph is reused across calls. Errors are logged (not fatal) so a bad
/// chunk never aborts an in-flight reply. Speed/pacing is baked into the audio at
/// synthesis (via `opts`), so playback itself does no rate change.
fn synth_emit(
    inflect: &InflectNano,
    opts: &InferOpts,
    tts_dev: rlx_runtime::Device,
    text: &str,
    sentence_pause: f32,
    player: &Option<audio_out::StreamPlayer>,
    fallback: &mut Option<Vec<f32>>,
) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    match inflect.synthesize_on_cached(text, opts, tts_dev) {
        Ok(wav) => {
            enqueue(player, fallback, &wav.samples);
            let silence = (sentence_pause * wav.sample_rate as f32) as usize;
            if silence > 0 {
                enqueue(player, fallback, &vec![0.0f32; silence]);
            }
        }
        Err(e) => eprintln!("[chat] tts failed: {e}"),
    }
}

/// Push samples to the live stream if open, else buffer them for the afplay fallback.
fn enqueue(
    player: &Option<audio_out::StreamPlayer>,
    fallback: &mut Option<Vec<f32>>,
    samples: &[f32],
) {
    if let Some(p) = player {
        if let Err(e) = p.push(samples) {
            eprintln!("[chat] audio push failed: {e}");
        }
    } else if let Some(buf) = fallback {
        buf.extend_from_slice(samples);
    }
}

/// Pop every complete sentence (ending in `.`/`!`/`?`/newline) off the front of
/// `buf`, leaving any trailing partial sentence for the next token. Punctuation
/// stays on the sentence so the acoustic model keeps its natural intonation.
fn drain_sentences(buf: &mut String) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(pos) = buf.find(|c: char| matches!(c, '.' | '!' | '?' | '\n')) {
        let end = pos + buf[pos..].chars().next().map_or(1, |c| c.len_utf8());
        let sentence = buf[..end].trim().to_string();
        *buf = buf[end..].to_string();
        if !sentence.is_empty() {
            out.push(sentence);
        }
    }
    out
}

/// Live streaming speaker built on cpal (adapted from the rlx-moshi voice-chat
/// player): a shared queue drained by the output callback, resampled from the
/// 24 kHz source to the device rate. Playback is gated on a prime buffer so the
/// start is not choppy.
mod audio_out {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use anyhow::{Context, Result, bail};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{FromSample, SampleFormat, SizedSample};

    fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub struct StreamPlayer {
        stream: cpal::Stream,
        queue: Arc<Mutex<VecDeque<f32>>>,
        started: AtomicBool,
        pushed_src: AtomicUsize,
        prime_src: usize,
        src_rate: u32,
        out_rate: u32,
    }

    impl StreamPlayer {
        /// Open an output stream for `src_rate` audio, priming `prime_secs` of
        /// source samples before playback begins.
        pub fn start(src_rate: u32, prime_secs: f32) -> Result<Self> {
            let host = cpal::default_host();
            let device = host
                .default_output_device()
                .context("no default output device")?;
            let supported = pick_output_config(&device)?;
            let out_rate = supported.sample_rate().0;
            let channels = supported.channels() as usize;
            let cfg: cpal::StreamConfig = supported.config();
            let queue: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
            let stream = match supported.sample_format() {
                SampleFormat::F32 => build_out::<f32>(&device, &cfg, channels, queue.clone())?,
                SampleFormat::I16 => build_out::<i16>(&device, &cfg, channels, queue.clone())?,
                SampleFormat::U16 => build_out::<u16>(&device, &cfg, channels, queue.clone())?,
                other => bail!("unsupported speaker sample format {other:?}"),
            };
            Ok(Self {
                stream,
                queue,
                started: AtomicBool::new(false),
                pushed_src: AtomicUsize::new(0),
                prime_src: (prime_secs.max(0.0) * src_rate as f32) as usize,
                src_rate,
                out_rate,
            })
        }

        /// Queue more 24 kHz source samples; auto-starts playback once the prime
        /// buffer has been reached.
        pub fn push(&self, pcm_src: &[f32]) -> Result<()> {
            if pcm_src.is_empty() {
                return Ok(());
            }
            let resampled = resample_linear(pcm_src, self.src_rate, self.out_rate);
            lock(&self.queue).extend(resampled);
            let total = self.pushed_src.fetch_add(pcm_src.len(), Ordering::Relaxed) + pcm_src.len();
            if total >= self.prime_src && !self.started.swap(true, Ordering::Relaxed) {
                self.stream.play().context("start speaker stream")?;
            }
            Ok(())
        }

        /// Start playback if it never primed (short reply), then block until the
        /// queue drains.
        pub fn finish(&self) -> Result<()> {
            if !self.started.swap(true, Ordering::Relaxed) {
                self.stream.play().context("start speaker stream")?;
            }
            loop {
                if lock(&self.queue).is_empty() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            // small tail so the device flushes its own buffer
            std::thread::sleep(Duration::from_millis(150));
            Ok(())
        }
    }

    fn build_out<T>(
        device: &cpal::Device,
        cfg: &cpal::StreamConfig,
        channels: usize,
        queue: Arc<Mutex<VecDeque<f32>>>,
    ) -> Result<cpal::Stream>
    where
        T: SizedSample + FromSample<f32>,
    {
        let stream = device.build_output_stream(
            cfg,
            move |out: &mut [T], _: &cpal::OutputCallbackInfo| {
                let mut q = lock(&queue);
                for frame in out.chunks_mut(channels.max(1)) {
                    let s = q.pop_front().unwrap_or(0.0);
                    let v = T::from_sample(s);
                    for c in frame.iter_mut() {
                        *c = v;
                    }
                }
            },
            |e| eprintln!("speaker stream error: {e}"),
            None,
        )?;
        Ok(stream)
    }

    fn pick_output_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
        for _ in 0..3 {
            if let Ok(c) = device.default_output_config() {
                return Ok(c);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        let ranges = device
            .supported_output_configs()
            .context("no output configs")?;
        pick_from_ranges(ranges).context("no usable output config")
    }

    /// `default_output_config()` intermittently fails on macOS CoreAudio — retry
    /// then fall back to enumerating supported ranges (prefer f32 @ 48 k / 24 k).
    fn pick_from_ranges<I: Iterator<Item = cpal::SupportedStreamConfigRange>>(
        ranges: I,
    ) -> Option<cpal::SupportedStreamConfig> {
        let mut best: Option<cpal::SupportedStreamConfig> = None;
        for r in ranges {
            let (min, max) = (r.min_sample_rate().0, r.max_sample_rate().0);
            let want = if (min..=max).contains(&48_000) {
                48_000
            } else if (min..=max).contains(&24_000) {
                24_000
            } else {
                max
            };
            let cfg = r.with_sample_rate(cpal::SampleRate(want));
            if cfg.sample_format() == SampleFormat::F32 {
                return Some(cfg);
            }
            best.get_or_insert(cfg);
        }
        best
    }

    fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
        if from_hz == to_hz || samples.is_empty() {
            return samples.to_vec();
        }
        let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src = i as f64 * from_hz as f64 / to_hz as f64;
            let idx = src.floor() as usize;
            let frac = (src - idx as f64) as f32;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            out.push(a + (b - a) * frac);
        }
        out
    }
}
