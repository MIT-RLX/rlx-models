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

//! Spectral applications on top of the learned FFT — log-mel frontend,
//! Welch PSD, and top-K peak extraction with its compiled / cost-model /
//! strategy-picker variants.

pub mod mel;
pub mod peak;
pub mod welch;
pub mod welch_peaks_compile;
pub mod welch_peaks_cost;
pub mod welch_peaks_picker;
