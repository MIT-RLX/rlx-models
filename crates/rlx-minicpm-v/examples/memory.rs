// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! One phase of a MiniCPM-V chat turn per run, for its peak memory under
//! `/usr/bin/time -l`: `runner` (the crate's runner;
//! `runner-bf16` with its vision tower bf16), `export` (writing the
//! package; `export-f32` with its weights f32) or `package` (opening and
//! running it).
//!
//! ```text
//! cargo run -p rlx-minicpm-v --release --features export --example memory -- \
//!     runner|export|package DIR PACKAGE
//! ```

use std::path::PathBuf;

use rlx_minicpm_v::RgbImage;
use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use serde_json::json;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
}

/// This task's `phys_footprint` (`TASK_VM_INFO`, the 19th u64).
#[cfg(target_os = "macos")]
fn footprint() -> u64 {
    let mut info = [0i32; 128];
    let mut count = info.len() as u32;
    // SAFETY: a buffer of `count` naturals for TASK_VM_INFO (22).
    unsafe { task_info(mach_task_self(), 22, info.as_mut_ptr(), &mut count) };
    unsafe { *(info.as_ptr() as *const u64).add(18) }
}

#[cfg(not(target_os = "macos"))]
fn footprint() -> u64 {
    0
}

/// With `MEMORY_TRACE` set, print the footprint to stderr each time it
/// climbs 256 MiB past the last line, so the jumps land between the
/// runner's own log lines.
fn trace_footprint() {
    if std::env::var_os("MEMORY_TRACE").is_none() {
        return;
    }
    std::thread::spawn(|| {
        let t = std::time::Instant::now();
        let mut last = 0u64;
        loop {
            let f = footprint();
            if f > last + (256 << 20) || f + (1 << 30) < last {
                eprintln!(
                    "[memory] {:7.1}s footprint {:.2} GB",
                    t.elapsed().as_secs_f64(),
                    f as f64 / 1e9
                );
                last = f;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    });
}

fn main() -> anyhow::Result<()> {
    trace_footprint();
    let args: Vec<String> = std::env::args().collect();
    let (phase, dir, pkg) = (
        args[1].as_str(),
        PathBuf::from(&args[2]),
        PathBuf::from(&args[3]),
    );
    let photo =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rlx-locateanything/fixtures/sample.jpg");
    let img = image::open(&photo)?.to_rgb8();
    let (w, h) = img.dimensions();
    let rgb = img.into_raw();
    let q = "Describe this image in one sentence.";
    match phase {
        "runner" | "runner-bf16" => {
            let mut r = rlx_minicpm_v::MiniCpmVRunner::builder()
                .model_dir(&dir)
                .device(Device::Cpu)
                .bf16_vision(phase == "runner-bf16")
                .build()?;
            let pic = RgbImage {
                width: w as usize,
                height: h as usize,
                rgb,
            };
            let t = r.chat(
                &[pic],
                q,
                8,
                rlx_qwen35::SampleOpts::greedy(),
                false,
                |_| {},
            )?;
            println!("{t}");
        }
        "export" | "export-f32" => {
            let opts = rlx_minicpm_v::export::ExportOptions {
                max_context: Some(2048),
                weight_dtype: (phase == "export-f32").then_some(rlx_ir::DType::F32),
                ..Default::default()
            };
            println!("{}", rlx_minicpm_v::export::export_rlxp(&dir, &pkg, &opts)?);
        }
        "package" => {
            rlx_text::install_model_tokenizer();
            let mut m = rlx_runtime::model::open_source(pkg.to_str().unwrap(), json!({}))?;
            let image = Tensor::new(
                "image0",
                rlx_ir::DType::U8,
                vec![h as usize, w as usize, 3],
                rgb,
            );
            let outs = m.run(&[image], &json!({"prompt": q, "max_tokens": 8}))?;
            println!("{}", String::from_utf8_lossy(&outs[0].data));
        }
        other => anyhow::bail!("phase {other}: runner, export or package"),
    }
    Ok(())
}
