// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Training via compiled RLX backward graphs (all backends).

use crate::butterfly::ParamSlot;
use crate::compile::compile_train_backward;
use crate::config::{EncDecTrainConfig, TrainConfig, TransformDir};
use crate::device::resolve_train_device;
use crate::reference::fft_real_batch;
use crate::train::{
    EncDecTrainResult, TrainResult, evaluate_encdec_weights, evaluate_weights_dir, random_batch,
};
use crate::train_graph::{
    EncDecTrainGraph, build_encdec_train_graph, build_supervised_train_graph,
};
use crate::twiddle::exact_twiddles;
use crate::twiddle_stability::{apply_twiddle_update, lr_for_n_fft};
use crate::weights::{EncDecWeights, WeightStore, export_safetensors};
use anyhow::{Context, Result, ensure};
use rand::prelude::*;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::time::Instant;

fn sgd_step(weights: &mut WeightStore, grads: &HashMap<String, Vec<f32>>, lr: f32) {
    for (name, data) in &mut weights.0 {
        let Some(g) = grads.get(name) else {
            continue;
        };
        debug_assert_eq!(data.len(), g.len(), "grad len mismatch for {name}");
        for (w, &gi) in data.iter_mut().zip(g.iter()) {
            *w -= lr * gi;
        }
    }
}

fn stable_twiddle_step(
    params: &[ParamSlot],
    weights: &mut WeightStore,
    grads: &HashMap<String, Vec<f32>>,
    lr: f32,
    grad_clip: f32,
    project: bool,
) -> Result<()> {
    let mut flat = twiddles_from_slots(params, weights)?;
    let mut flat_grad = Vec::with_capacity(params.len());
    for slot in params {
        flat_grad.push(
            *grads
                .get(&slot.name)
                .with_context(|| format!("missing grad {}", slot.name))?
                .first()
                .context("empty grad")?,
        );
    }
    apply_twiddle_update(&mut flat, &flat_grad, lr, grad_clip, project);
    for (slot, &v) in params.iter().zip(flat.iter()) {
        weights.0.get_mut(&slot.name).expect("param").fill(v);
    }
    Ok(())
}

fn run_backward(
    exec: &mut CompiledGraph,
    feeds: &[(&str, &[f32])],
    params: &[ParamSlot],
) -> Result<(f32, HashMap<String, Vec<f32>>)> {
    let outs = exec.run(feeds);
    ensure!(!outs.is_empty(), "backward produced no outputs");
    let loss = outs[0].first().copied().unwrap_or(f32::NAN);
    let mut grads = HashMap::new();
    for (slot, gout) in params.iter().zip(outs.iter().skip(1)) {
        grads.insert(slot.name.clone(), gout.clone());
    }
    Ok((loss, grads))
}

fn init_twiddle_weights(params: &[ParamSlot], twiddles: &[f32]) -> Result<WeightStore> {
    ensure!(
        params.len() == twiddles.len(),
        "param count != twiddle count"
    );
    let mut store = WeightStore::default();
    for (slot, &v) in params.iter().zip(twiddles.iter()) {
        store.0.insert(slot.name.clone(), vec![v]);
    }
    Ok(store)
}

fn twiddles_from_slots(params: &[ParamSlot], weights: &WeightStore) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(params.len());
    for slot in params {
        out.push(
            *weights
                .0
                .get(&slot.name)
                .with_context(|| format!("missing {}", slot.name))?
                .first()
                .context("empty param")?,
        );
    }
    Ok(out)
}

pub fn train_butterfly_rlx(cfg: &TrainConfig, dir: TransformDir) -> Result<TrainResult> {
    cfg.model.validate()?;
    let device: Device = resolve_train_device(Some(&cfg.device))?;
    let graph = build_supervised_train_graph(&cfg.model, dir)?;
    let started = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let n = cfg.model.n_fft;
    let batch = cfg.model.batch;

    let flat_tw = exact_twiddles(&cfg.model);
    let mut weights = init_twiddle_weights(&graph.params, &flat_tw)?;
    let (_, mut backward) = compile_train_backward(device, graph.backward.clone(), "rlx-fft")?;
    for (name, data) in &weights.0 {
        backward.set_param(name, data);
    }

    let d_loss = [1.0f32];
    let mut last_mse;
    for step in 0..cfg.steps {
        let signal = if dir.is_forward() {
            random_batch(&mut rng, batch, n)
        } else {
            crate::train::random_complex_batch(&mut rng, batch, n)
        };
        let target = if dir.is_forward() {
            fft_real_batch(&signal, batch, n)?
        } else {
            crate::reference::ifft_complex_batch(&signal, batch, n)?
        };

        let (loss, grads) = run_backward(
            &mut backward,
            &[
                (graph.data_input, &signal),
                (graph.target_input, &target),
                ("d_output", &d_loss),
            ],
            &graph.params,
        )?;
        last_mse = loss;
        sgd_step(&mut weights, &grads, cfg.lr as f32);
        for (name, data) in &weights.0 {
            backward.set_param(name, data);
        }

        if cfg.log_every > 0 && (step + 1) % cfg.log_every == 0 {
            eprintln!("[train rlx {dir:?}] step {} mse={last_mse:.6e}", step + 1);
        }
    }

    let store = WeightStore::from_twiddles(&twiddles_from_slots(&graph.params, &weights)?, n);
    let (final_mse, max_err) = evaluate_weights_dir(&store, &cfg.model, 8, dir)?;

    if let Some(dir_path) = &cfg.out_dir {
        std::fs::create_dir_all(dir_path)?;
        export_safetensors(&dir_path.join("twiddles.safetensors"), &store)?;
    }

    Ok(TrainResult {
        final_mse,
        max_error: max_err,
        weights: store,
        steps: cfg.steps,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        direction: dir,
    })
}

pub fn train_encdec_rlx(cfg: &EncDecTrainConfig) -> Result<EncDecTrainResult> {
    cfg.model.validate()?;
    let device: Device = resolve_train_device(Some(&cfg.device))?;
    let graph = build_encdec_train_graph(&cfg.model, cfg.spectrum_weight)?;
    let started = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let n = cfg.model.n_fft;
    let batch = cfg.model.batch;

    let enc_tw = exact_twiddles(&cfg.model);
    let dec_tw = exact_twiddles(&cfg.model);
    let mut enc_weights = init_twiddle_weights(&graph.encoder_params, &enc_tw)?;
    let mut dec_weights = init_twiddle_weights(&graph.decoder_params, &dec_tw)?;
    let (_, mut backward) =
        compile_train_backward(device, graph.backward.clone(), "rlx-fft-encdec")?;
    for (name, data) in enc_weights.0.iter().chain(dec_weights.0.iter()) {
        backward.set_param(name, data);
    }

    let d_loss = [1.0f32];
    let mut last_recon;
    for step in 0..cfg.steps {
        let signal = random_batch(&mut rng, batch, n);
        let mut feeds: Vec<(&str, &[f32])> = vec![("signal", &signal), ("d_output", &d_loss)];
        let target_spec = if cfg.spectrum_weight > 0.0 {
            Some(fft_real_batch(&signal, batch, n)?)
        } else {
            None
        };
        if let Some(ref spec) = target_spec {
            feeds.insert(1, ("target_spectrum", spec.as_slice()));
        }

        let (loss, grads) = run_backward(&mut backward, &feeds, &chain_params(&graph))?;
        last_recon = loss;
        let lr = lr_for_n_fft(cfg.lr, n);
        stable_twiddle_step(
            &graph.encoder_params,
            &mut enc_weights,
            &grads,
            lr,
            cfg.grad_clip,
            cfg.project_twiddles,
        )?;
        stable_twiddle_step(
            &graph.decoder_params,
            &mut dec_weights,
            &grads,
            lr,
            cfg.grad_clip,
            cfg.project_twiddles,
        )?;
        for (name, data) in enc_weights.0.iter().chain(dec_weights.0.iter()) {
            backward.set_param(name, data);
        }

        if cfg.log_every > 0 && (step + 1) % cfg.log_every == 0 {
            eprintln!("[train rlx encdec] step {} loss={last_recon:.6e}", step + 1);
        }
    }

    let enc_flat = twiddles_from_slots(&graph.encoder_params, &enc_weights)?;
    let dec_flat = twiddles_from_slots(&graph.decoder_params, &dec_weights)?;
    let weights = EncDecWeights::from_twiddles(&enc_flat, &dec_flat, n);
    let (recon_mse, spec_mse, max_err) = evaluate_encdec_weights(&weights, &cfg.model, 8)?;

    if let Some(dir_path) = &cfg.out_dir {
        std::fs::create_dir_all(dir_path)?;
        export_safetensors(&dir_path.join("encdec.safetensors"), &weights.merged())?;
    }

    Ok(EncDecTrainResult {
        reconstruction_mse: recon_mse,
        spectrum_mse: spec_mse,
        roundtrip_max_error: max_err,
        weights,
        steps: cfg.steps,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn chain_params(graph: &EncDecTrainGraph) -> Vec<ParamSlot> {
    graph
        .encoder_params
        .iter()
        .chain(graph.decoder_params.iter())
        .cloned()
        .collect()
}
