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

//! Lower training backward ops for Metal / MLX / CUDA before compile.

use rlx_autodiff::decompose_backward_ops_except;
use rlx_autodiff::legalize_reduce::legalize_multi_axis_reduce;
use rlx_ir::Graph;
use rlx_ir::op::OpKind;
use rlx_runtime::Device;

pub fn prepare_backward_for_device(graph: Graph, device: Device) -> Graph {
    if !needs_portable_backward_prep(device) {
        return graph;
    }
    let g = legalize_multi_axis_reduce(graph);
    match device {
        Device::Mlx => decompose_backward_ops_except(
            g,
            &[
                OpKind::Conv2dBackwardInput,
                OpKind::Conv2dBackwardWeight,
                OpKind::AttentionBackward,
            ],
        ),
        Device::Metal | Device::Cuda => decompose_backward_ops_except(
            g,
            &[
                OpKind::RmsNormBackwardInput,
                OpKind::RmsNormBackwardGamma,
                OpKind::AttentionBackward,
                OpKind::Conv2dBackwardInput,
                OpKind::Conv2dBackwardWeight,
            ],
        ),
        _ => g,
    }
}

pub fn needs_portable_backward_prep(device: Device) -> bool {
    matches!(
        device,
        Device::Metal | Device::Mlx | Device::Cuda | Device::Gpu
    )
}
