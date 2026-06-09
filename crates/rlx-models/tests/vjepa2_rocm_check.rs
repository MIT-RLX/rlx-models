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

//! V-JEPA2 full pipeline check: CPU vs ROCm parity (skips when no driver).

#[path = "vjepa2/support.rs"]
mod support;

#[cfg(feature = "rocm")]
#[test]
fn vjepa2_tiny_pipeline_matches_cpu_on_rocm() {
    mod compile_support;

    use rlx_runtime::Device;

    if !rlx_rocm::is_available() {
        eprintln!("skip: ROCm not available");
        return;
    }
    let cpu = support::run_compiled_pipeline(Device::Cpu);
    let rocm = support::run_compiled_pipeline(Device::Rocm);
    support::assert_pipeline_close(&cpu, &rocm, Device::Rocm, 2e-2);
}
