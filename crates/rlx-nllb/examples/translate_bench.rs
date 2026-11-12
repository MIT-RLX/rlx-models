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

//! Load / first / warm translation timings, with the process footprint
//! (`phys_footprint` on macOS — RSS undercounts compressed pages) after each
//! step and its sampled peak.
//!
//! ```sh
//! cargo run -p rlx-nllb --release --example translate_bench -- \
//!     weights/mt/nllb-200-distilled-600M [cpu|metal|mlx] [BEAMS] [repeat]
//! ```

use rlx_nllb::{GenerateConfig, NllbRunner};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

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

static PEAK: AtomicU64 = AtomicU64::new(0);

fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

const TEXTS: [&str; 3] = [
    "Hello, how are you?",
    "The weather is nice today, so we are going to the park.",
    "No Language Left Behind is a project that open-sources models capable of delivering \
     high-quality translations directly between any pair of 200+ languages, including \
     low-resource languages like Asturian, Luganda, Urdu and more.",
];

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("weights/mt/nllb-200-distilled-600M");
    let device = rlx_cli::parse_device(args.get(2).map(String::as_str).unwrap_or("cpu"))?;
    let beams: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(1);

    std::thread::spawn(|| {
        loop {
            PEAK.fetch_max(footprint(), Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });
    let t = Instant::now();
    let mut runner = NllbRunner::builder().weights(dir).device(device).build()?;
    println!(
        "load {:.2}s  footprint {:.2} GB",
        t.elapsed().as_secs_f64(),
        gb(footprint())
    );
    let opts = GenerateConfig::new(128).with_beams(beams);
    // `repeat`: the long sentence three times (steady state, one bucket).
    let repeat = args.get(4).map(String::as_str) == Some("repeat");
    for pass in ["first", "warm"] {
        let texts: Vec<&str> = if repeat {
            vec![TEXTS[2]; 3]
        } else {
            TEXTS.to_vec()
        };
        for text in texts {
            let t = Instant::now();
            let out = runner.translate(text, "eng_Latn", "spa_Latn", &opts)?;
            println!(
                "{pass:5} {:6.2}s  footprint {:.2} GB  {}",
                t.elapsed().as_secs_f64(),
                gb(footprint()),
                out.chars().take(48).collect::<String>()
            );
        }
    }
    PEAK.fetch_max(footprint(), Ordering::Relaxed);
    println!("peak footprint {:.2} GB", gb(PEAK.load(Ordering::Relaxed)));
    Ok(())
}
