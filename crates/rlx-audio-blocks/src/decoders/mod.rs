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

//! Model-agnostic sequence decoders.
//!
//! These are pure host-side decode loops: the acoustic encoder runs in the RLX
//! graph, and the token-serial prediction/joint search runs here in exact `f32`
//! — the standard pattern for transducer greedy search (see
//! `rlx-nemotron-asr`'s RNN-T decoder for the sibling implementation).

pub mod tdt;
pub mod transducer;

pub use tdt::{
    TdtAlgorithm, TdtDecodeResult, TdtDecoderCore, TdtJointStep, run_tdt_decoder,
    run_tdt_greedy_duration_loop,
};
pub use transducer::{
    GreedyTransducerResult, StatelessTransducerCore, TransducerStep,
    run_stateless_transducer_greedy,
};
