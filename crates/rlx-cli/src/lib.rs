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

//! Shared pieces for per-model `rlx-<family>` binaries and the optional
//! `rlx-run` multiplexer.

pub mod args;
pub mod auto_dispatch;
pub mod chat;
pub mod compat;
pub mod device;
pub mod format;
pub mod inspect;
pub mod lm_args;
pub mod lm_runner;
pub mod loader;
pub mod mtmd;
pub mod registry;
pub mod weights_resolve;

pub use args::req;
pub use auto_dispatch::{
    SniffedFrom, SniffedRunner, UnimplementedArch, arch_runner_name, auto_dispatch,
    auto_runner_name, auto_sniff, known_unimplemented_arch, known_unimplemented_keys,
    model_type_runner_name, run_auto,
};
pub use chat::{ChatMessage, ChatTemplate, ChatTemplateSource, auto_chat_template};
pub use compat::{
    CompatSource, CompatibilityReport, CompatibilityStatus, GgufRequiredFields, check_hf_repo,
    check_path, looks_like_hf_repo, run_check,
};
pub use device::{
    parse_device, parse_gemma_device, parse_llada2_device, parse_llama32_device, parse_lm_device,
    parse_qwen35_device, parse_sam_device, parse_standard_device,
};
pub use format::WeightFormat;
pub use inspect::{estimate_qwen35_footprint, fmt_bytes, list_mtp_keys, run_inspect};
pub use lm_args::LmCliArgs;
pub use lm_runner::LmRunner;
pub use loader::{
    GgufDirGuide, GgufTensorNameResolver, LoadOpts, RegisteredFormat, ResolveOpts,
    WeightFormatRegistration, debug_resolve_name, gguf_dir_guide, list_registered_formats,
    load_weight_map_resolved, load_weights_resolved, open_gguf_loader, open_loader,
    open_loader_resolved, open_loader_resolved_with_options, open_loader_with_format, open_map,
    open_map_with, open_weight_map_resolved, open_weights, open_weights_resolved, open_with,
    register_gguf_tensor_resolver, register_weight_format, weights,
};
pub use mtmd::{AssembledTurn, MediaSource, MtmdContext, MtmdTurn};
pub use registry::{
    ModelRunner, dispatch, dispatch_help, register_cli, register_runner, registered_runners,
    run_registered,
};
pub use rlx_core::weights_discover::{
    DiscoverOpts, DiscoveredFormat, DiscoveredWeight, WeightSourceKind, default_source_roots,
    looks_like_filesystem_path, resolve_weight_query, resolve_weight_query_in_roots,
    resolve_weights_path_or_query, scan_weights, scan_weights_in_roots,
};
pub use rlx_core::{STANDARD_DEVICE_NAMES, validate_sam_device, validate_standard_device};
pub use weights_resolve::{WeightsResolveCli, resolve_weights_cli};
