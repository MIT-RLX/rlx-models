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

use criterion::{Criterion, criterion_group, criterion_main};
use rlx_clinicalbert::ClinicalBertVariant;

fn bench_head_dims(c: &mut Criterion) {
    c.bench_function("preset_vocab_sizes", |b| {
        b.iter(|| {
            ClinicalBertVariant::Huang.preset().vocab_size
                + ClinicalBertVariant::BioClinical.preset().vocab_size
                + ClinicalBertVariant::BioDischarge.preset().vocab_size
        });
    });
}

criterion_group!(benches, bench_head_dims);
criterion_main!(benches);
