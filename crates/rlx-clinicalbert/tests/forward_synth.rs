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

use rlx_clinicalbert::ClinicalBertVariant;

#[test]
fn clinicalbert_presets_are_bert_base_shaped() {
    for variant in [
        ClinicalBertVariant::Huang,
        ClinicalBertVariant::BioClinical,
        ClinicalBertVariant::BioDischarge,
    ] {
        let cfg = variant.preset();
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_attention_heads, 12);
        assert!(cfg.vocab_size > 28_000);
    }
}
