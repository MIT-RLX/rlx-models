//! RLX TTS's host logic as an RLX program, answering `synthesize`: what
//! `RlxTts::synthesize_text` does on its portable path — the crate's text
//! frontend (`src/frontend`, its rule files and G2P network beside the
//! package) to phone ids; `encode`; the predictors to durations (the
//! cumulative sum's floors, pauses at least their minimum) and pitch /
//! energy buckets; `decode` over each frame's phone; the mel past its
//! leading frames through `wavernn` a frame at a time, each step's noise the
//! BNNS AES-CTR stream's Gumbel draws (256 for the coarse sample, 256 for the
//! fine); the μ-law IIR, the tail trim and the volume. `run` answers the mel
//! (`options.stage: "mel"`), or the vocoder's raw samples on a given mel
//! (`options.mel`; `options.greedy`, the argmax).

/// What the frontend's `crate::gguf_bundle` gives it: where its files are.
#[allow(dead_code)]
mod gguf_bundle {
    pub fn default_extract_dir() -> Option<std::path::PathBuf> {
        Some(std::path::PathBuf::from(crate::BUNDLE))
    }
}

#[allow(dead_code)]
#[path = "../../src/ops.rs"]
mod ops;

#[allow(dead_code, unused_imports, clippy::type_complexity)]
#[path = "../../src/frontend/mod.rs"]
mod frontend;

use frontend::{HydraLite, TextFrontend};
use rlx_program::{Message, Program, Tensor, run};
use serde_json::{Value, json};

const BUNDLE: &str = "/package/bundle";
/// The `pau` phone: at least the pause minimum.
const PAUSE_MARKER: usize = 72;

struct Post {
    volume_global: f32,
    volume_peak_ratio: f32,
    volume_smoothing: usize,
    wr_leading_silence_frames: usize,
    wr_trailing_trim_samples: usize,
    wr_mulaw_iir_alpha: Option<f32>,
}

struct Tts {
    frontend: HydraLite,
    seed: u64,
    alpha: f32,
    beta: f32,
    steps: usize,
    carry: usize,
    hidden: usize,
    n_mels: usize,
    phone_vocab: usize,
    pause_min_ms: f32,
    post: Post,
    sample_rate: u32,
}

fn first(outs: Vec<Tensor>, what: &str) -> Result<Tensor, String> {
    outs.into_iter()
        .next()
        .ok_or_else(|| format!("{what} gave nothing"))
}

fn named(outs: &[Tensor], n: &str) -> Result<Vec<f32>, String> {
    outs.iter()
        .find(|t| t.name == n)
        .map(Tensor::to_f32)
        .ok_or_else(|| format!("no output {n}"))
}

/// WaveRNN μ-law expand + one-pole IIR (`metrics::apply_wavernn_mulaw_iir`).
fn mulaw_iir(pcm: &mut [f32], alpha: f32) {
    let expand = |class: f32| {
        let x = (class / 255.0) * 2.0 - 1.0;
        let sign = if x < 0.0 {
            -1.0
        } else if x > 0.0 {
            1.0
        } else {
            0.0
        };
        sign * ((8.0 * x.abs()).exp2() - 1.0) * (1.0 / 255.0)
    };
    let alpha = alpha.clamp(0.0, 1.0);
    let mut prev = 0.0f32;
    let mut i = 0;
    while i + 1 < pcm.len() {
        let c0 = (((pcm[i] + 1.0) * 255.0 / 2.0).round() as i32).clamp(0, 255) as f32;
        let c1 = (((pcm[i + 1] + 1.0) * 255.0 / 2.0).round() as i32).clamp(0, 255) as f32;
        let o0 = (prev * alpha + expand(c0)).clamp(-1.0, 1.0);
        let o1 = (o0 * alpha + expand(c1)).clamp(-1.0, 1.0);
        pcm[i] = o0;
        pcm[i + 1] = o1;
        prev = o1;
        i += 2;
    }
}

/// `metrics::apply_output_volume`.
fn output_volume(pcm: &mut [f32], global: f32, peak_ratio: f32, smoothing: usize) {
    if pcm.is_empty() {
        return;
    }
    for s in pcm.iter_mut() {
        *s *= global;
    }
    let peak_now = pcm.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-8);
    let target = peak_ratio.clamp(0.05, 1.0);
    if peak_now > target {
        let scale = target / peak_now;
        let win = smoothing.max(1).min(pcm.len());
        for (i, s) in pcm.iter_mut().enumerate() {
            let g = if i < win {
                let t = i as f32 / win as f32;
                1.0 + (scale - 1.0) * t
            } else {
                scale
            };
            *s *= g;
        }
    }
}

impl Tts {
    /// Text → phone ids.
    fn phones(&self, text: &str) -> Result<Vec<usize>, String> {
        let phones = self
            .frontend
            .text_to_phones(text)
            .map_err(|e| e.to_string())?;
        self.frontend
            .phones_to_ids(&phones)
            .map_err(|e| e.to_string())
    }

    /// Phone ids → mel `[frames · 80]` (`FastSpeech2::infer`).
    fn mel(&self, ids: &[usize]) -> Result<(Vec<f32>, usize), String> {
        let t = ids.len();
        if t == 0 {
            return Err("empty phone id sequence".into());
        }
        let phones: Vec<i32> = ids
            .iter()
            .map(|&i| i.min(self.phone_vocab - 1) as i32)
            .collect();
        let outs = run("encode", vec![Tensor::i32("phones", vec![t], &phones)])?;
        let x = outs
            .iter()
            .find(|o| o.name == "x")
            .cloned()
            .ok_or("encode gave no x")?;
        let dur = named(&outs, "duration")?;
        let pitch = named(&outs, "pitch")?;
        let energy = named(&outs, "energy")?;
        // Durations: the cumulative sum's floors' differences.
        let mut durs = Vec::with_capacity(t);
        let (mut cum, mut prev) = (0.0f32, 0usize);
        for &v in &dur {
            cum += v.max(0.001).max(0.0);
            let c = cum.floor().max(0.0) as usize;
            durs.push(c.saturating_sub(prev));
            prev = c;
        }
        if durs.iter().sum::<usize>() == 0 {
            durs[0] = 1;
        }
        let pause = ((self.pause_min_ms.max(0.0) / 10.0).ceil() as usize).max(1);
        for (i, &pid) in ids.iter().enumerate() {
            if pid == PAUSE_MARKER {
                durs[i] = durs[i].max(pause);
            }
        }
        const LN40: f32 = 3.688_879_5;
        const PITCH_SCALE: f32 = 99.216_47;
        const ENERGY_SCALE: f32 = 27.826_088;
        let pitch_ids: Vec<i32> = pitch
            .iter()
            .map(|&v| {
                ((v.max(1e-5).ln() - LN40) * PITCH_SCALE)
                    .floor()
                    .clamp(0.0, 255.0) as i32
            })
            .collect();
        let energy_ids: Vec<i32> = energy
            .iter()
            .map(|&v| ((v + 0.2) * ENERGY_SCALE).floor().clamp(0.0, 255.0) as i32)
            .collect();
        let index: Vec<i32> = durs
            .iter()
            .enumerate()
            .flat_map(|(i, &d)| std::iter::repeat_n(i as i32, d))
            .collect();
        let f = index.len();
        let mut x = x;
        x.name = "x".into();
        let mel = first(
            run(
                "decode",
                vec![
                    x,
                    Tensor::i32("pitch_ids", vec![t], &pitch_ids),
                    Tensor::i32("energy_ids", vec![t], &energy_ids),
                    Tensor::i32("index", vec![f], &index),
                ],
            )?,
            "decode",
        )?
        .to_f32();
        Ok((mel, f))
    }

    /// Mel `[frames · 80]` → raw samples (`WaveRnn::infer`), two a step.
    fn vocode(&self, mel: &[f32], greedy: bool) -> Result<Vec<f32>, String> {
        let frames = mel.len() / self.n_mels;
        let mut rng = rlx_ir::rng::BnnsAesCtr128::new(self.seed);
        let mut carry = vec![0.0f32; self.carry];
        let mut pcm = Vec::with_capacity(frames * self.steps * 2);
        let mut noise = vec![0.0f32; self.steps * 512];
        for fi in 0..frames {
            if !greedy {
                for v in noise.iter_mut() {
                    *v = rng.gumbel(self.alpha, self.beta);
                }
            }
            let steps = first(
                run(
                    "wavernn",
                    vec![
                        Tensor::f32(
                            "frame",
                            vec![1, self.n_mels],
                            &mel[fi * self.n_mels..(fi + 1) * self.n_mels],
                        ),
                        Tensor::f32("carry", vec![self.carry], &carry),
                        Tensor::f32("noise", vec![self.steps, 512], &noise),
                    ],
                )?,
                "wavernn",
            )?
            .to_f32();
            for s in 0..self.steps {
                let row = &steps[s * self.carry..(s + 1) * self.carry];
                pcm.push(row[self.hidden + 1]);
                pcm.push(row[self.hidden + 2]);
            }
            carry.copy_from_slice(&steps[(self.steps - 1) * self.carry..]);
        }
        Ok(pcm)
    }

    /// `RlxTts::synthesize_phones`' post on the vocoder's samples.
    fn speak(&self, text: &str) -> Result<Vec<f32>, String> {
        let ids = self.phones(text)?;
        let (mel, f) = self.mel(&ids)?;
        let skip = self.post.wr_leading_silence_frames.min(f.saturating_sub(1));
        let mut pcm = self.vocode(&mel[skip * self.n_mels..], false)?;
        if let Some(alpha) = self.post.wr_mulaw_iir_alpha {
            mulaw_iir(&mut pcm, alpha);
        }
        let trim = self.post.wr_trailing_trim_samples;
        if trim > 0 && pcm.len() > trim {
            pcm.truncate(pcm.len() - trim);
        }
        output_volume(
            &mut pcm,
            self.post.volume_global,
            self.post.volume_peak_ratio,
            self.post.volume_smoothing,
        );
        Ok(pcm)
    }
}

impl Program for Tts {
    fn init(config: Value) -> Result<Self, String> {
        let u = |k: &str| config[k].as_u64().ok_or(format!("config has no {k}"));
        let f = |k: &str| {
            config[k]
                .as_f64()
                .map(|v| v as f32)
                .ok_or(format!("config has no {k}"))
        };
        let p = &config["post"];
        let pu = |k: &str| p[k].as_u64().unwrap_or(0) as usize;
        let pf = |k: &str| p[k].as_f64().unwrap_or(0.0) as f32;
        let frontend = HydraLite::open(std::path::Path::new(BUNDLE)).map_err(|e| e.to_string())?;
        Ok(Tts {
            frontend,
            seed: u("seed")?,
            alpha: f("gumbel_alpha")?,
            beta: f("gumbel_beta")?,
            steps: u("steps")? as usize,
            carry: u("carry")? as usize,
            hidden: u("hidden")? as usize,
            n_mels: u("n_mels")? as usize,
            phone_vocab: u("phone_vocab")? as usize,
            pause_min_ms: f("pause_min_duration_ms")?,
            post: Post {
                volume_global: pf("volume_global"),
                volume_peak_ratio: pf("volume_peak_ratio"),
                volume_smoothing: pu("volume_smoothing"),
                wr_leading_silence_frames: pu("wr_leading_silence_frames"),
                wr_trailing_trim_samples: pu("wr_trailing_trim_samples"),
                wr_mulaw_iir_alpha: p["wr_mulaw_iir_alpha"].as_f64().map(|v| v as f32),
            },
            sample_rate: u("sample_rate")? as u32,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let o = &req.body["options"];
        if let Some(s) = o["seed"].as_u64() {
            self.seed = s;
        }
        let text = match task {
            "synthesize" => req.body["text"].as_str().ok_or("synthesize takes text")?,
            "run" => o["text"].as_str().unwrap_or(""),
            _ => return Err(format!("RLX TTS answers synthesize and run, not {task}")),
        };
        if task == "run" {
            // A given mel through the vocoder, raw.
            if let Some(rows) = o["mel"].as_array() {
                let mel: Vec<f32> = rows
                    .iter()
                    .filter_map(Value::as_array)
                    .flat_map(|r| r.iter().filter_map(|v| v.as_f64().map(|v| v as f32)))
                    .collect();
                let pcm = self.vocode(&mel, o["greedy"].as_bool().unwrap_or(false))?;
                let n = pcm.len();
                return Ok(Message::tensors(vec![Tensor::f32("pcm", vec![n], &pcm)]));
            }
            let ids = self.phones(text)?;
            let (mel, f) = self.mel(&ids)?;
            return Ok(Message::tensors(vec![Tensor::f32(
                "mel",
                vec![f, self.n_mels],
                &mel,
            )]));
        }
        let wav = self.speak(text)?;
        let n = wav.len();
        Ok(Message::new(
            json!({"sample_rate": self.sample_rate}),
            vec![Tensor::f32("audio", vec![n], &wav)],
        ))
    }
}

rlx_program::program!(Tts);
