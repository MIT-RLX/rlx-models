use ndarray::ArrayView1;
use rand::Rng;
use rand::SeedableRng;

/// Greedy / top-k / temperature sampling over a 1-D logit vector.
#[derive(Clone)]
pub struct LogitsProcessor {
    pub temperature: f64,
    pub top_k: usize,
    pub seed: u64,
}

impl LogitsProcessor {
    pub fn new(temperature: f64, top_k: usize, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            seed,
        }
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn sample(&mut self, logits: ArrayView1<f32>) -> anyhow::Result<u32> {
        let v = logits.len();
        let mut idx_logits: Vec<(usize, f32)> = (0..v).map(|i| (i, logits[i])).collect();
        if self.top_k > 0 && self.top_k < v {
            idx_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx_logits.truncate(self.top_k);
        }
        if self.temperature <= 0.0 || (self.temperature - 1.0).abs() < 1e-6 && self.top_k == 1 {
            let best = idx_logits
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| *i)
                .unwrap_or(0);
            return Ok(best as u32);
        }
        let inv_t = 1.0 / self.temperature as f32;
        let max = idx_logits
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = idx_logits
            .iter()
            .map(|(_, l)| ((l - max) * inv_t).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        self.seed = rng.r#gen();
        let r: f32 = rng.r#gen::<f32>() * sum;
        let mut acc = 0.0f32;
        for (i, p) in probs.iter_mut().enumerate() {
            acc += *p;
            if r <= acc {
                return Ok(idx_logits[i].0 as u32);
            }
        }
        Ok(idx_logits.last().map(|(i, _)| *i).unwrap_or(0) as u32)
    }
}
