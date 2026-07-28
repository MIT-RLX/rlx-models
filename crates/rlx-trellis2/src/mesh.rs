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

//! Flexible dual-grid → triangle mesh extraction
//! (`o_voxel.convert.flexible_dual_grid_to_mesh`, inference path).
//!
//! The upstream `_C` calls are only a spatial hashmap; the geometry itself is
//! pure and ports directly. Each active voxel carries **one** dual vertex
//! `mesh_vertices = (coord + dual_offset)·voxel_size + aabb.min`. For every
//! voxel edge flagged as surface-intersected (`intersected[axis]`), the four
//! voxels around that edge (`EDGE_NEIGHBOR_OFFSET[axis]`) form a quad; if all
//! four are active it is emitted and triangulated along the diagonal favored by
//! the per-voxel `split_weight` (`sw₀·sw₂` vs `sw₁·sw₃`).

use crate::sparse::SparseTensor;
use std::collections::HashMap;

/// The four voxels around each axis-aligned edge (per axis), as offsets.
const EDGE_NEIGHBOR_OFFSET: [[[i32; 3]; 4]; 3] = [
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], // x-axis
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], // y-axis
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], // z-axis
];
/// Diagonal triangulations of a quad (`0-2` split and `1-3` split).
const QUAD_SPLIT_1: [usize; 6] = [0, 1, 2, 0, 2, 3];
/// Same triangles as upstream `quad_split_2 = [0,1,3, 3,1,2]` (winding-matched).
const QUAD_SPLIT_2: [usize; 6] = [0, 1, 3, 3, 1, 2];

/// A triangle mesh (world-space vertices + triangle indices).
#[derive(Clone, Default)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

/// Mesh plus sparse PBR voxel attributes (TRELLIS.2 layout).
///
/// Attribute channels per voxel: RGB base color `0..3`, metallic `3`,
/// roughness `4`, alpha `5`.
///
/// [`MeshWithPbr::to_obj`] is geometry-only; [`MeshWithPbr::to_ply`] /
/// [`MeshWithPbr::to_glb`] paint vertices from nearest-voxel base color.
/// Official TRELLIS UV-atlas + PBR GLB bake (`o_voxel.postprocess.to_glb`) is
/// not ported here.
#[derive(Clone, Default)]
pub struct MeshWithPbr {
    pub mesh: Mesh,
    pub coords: Vec<[i32; 3]>,
    /// `[N, 6]` row-major PBR attributes in `[0, 1]`.
    pub attrs: Vec<f32>,
    pub grid_size: usize,
}

impl MeshWithPbr {
    pub fn to_obj(&self) -> String {
        self.mesh.to_obj()
    }

    /// ASCII PLY with per-vertex RGB from nearest PBR voxel (`attrs[0..3]`).
    ///
    /// Dual-grid mesh vertices are one-per-shape-voxel in decode order when
    /// `mesh.vertices.len() == coords.len()`; otherwise each vertex is mapped
    /// by nearest integer grid cell in world AABB `[-0.5, 0.5]³`.
    pub fn to_ply(&self) -> String {
        let n_v = self.mesh.vertices.len();
        let n_f = self.mesh.faces.len();
        let colors = self.vertex_base_colors();
        let mut s = String::with_capacity(n_v * 48 + n_f * 24 + 256);
        s.push_str("ply\nformat ascii 1.0\n");
        s.push_str(&format!("element vertex {n_v}\n"));
        s.push_str("property float x\nproperty float y\nproperty float z\n");
        s.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
        s.push_str(&format!("element face {n_f}\n"));
        s.push_str("property list uchar int vertex_indices\n");
        s.push_str("end_header\n");
        for (i, v) in self.mesh.vertices.iter().enumerate() {
            let [r, g, b] = colors[i];
            s.push_str(&format!("{} {} {} {} {} {}\n", v[0], v[1], v[2], r, g, b));
        }
        for f in &self.mesh.faces {
            s.push_str(&format!("3 {} {} {}\n", f[0], f[1], f[2]));
        }
        s
    }

    /// Binary glTF 2.0 (`.glb`) with UV-mapped **textures**:
    /// - `baseColorTexture` (RGB from PBR attrs / nearest voxel)
    /// - `metallicRoughnessTexture` (G=roughness, B=metallic) when attrs exist
    ///
    /// Each mesh vertex samples one atlas texel (packed `⌈√N⌉²` grid). This is
    /// not the official `o_voxel` UV-unwrap + remesh bake, but it is a real
    /// textured PBR GLB (not vertex `COLOR_0` only).
    pub fn to_glb(&self) -> Vec<u8> {
        let n_v = self.mesh.vertices.len();
        let n_f = self.mesh.faces.len();
        let (base_rgb, orm, has_orm) = self.vertex_pbr_texels();

        // Pack each vertex into a CELL×CELL block so nearest sampling is stable
        // in viewers (1×1 atlas cells look like sparkle/noise).
        const CELL: u32 = 8;
        let side = ((n_v as f32).sqrt().ceil() as u32).max(1);
        let atlas_side = (side * CELL).next_power_of_two().max(CELL);
        let tex_w = atlas_side;
        let tex_h = atlas_side;

        let mut base_img =
            image::RgbaImage::from_pixel(tex_w, tex_h, image::Rgba([200, 200, 200, 255]));
        let mut orm_img =
            image::RgbaImage::from_pixel(tex_w, tex_h, image::Rgba([255, 255, 0, 255]));
        let mut uvs = vec![[0.0f32; 2]; n_v];
        for i in 0..n_v {
            let ix = (i as u32) % side;
            let iy = (i as u32) / side;
            let [r, g, b] = base_rgb[i];
            let px0 = ix * CELL;
            let py0 = iy * CELL;
            for dy in 0..CELL {
                for dx in 0..CELL {
                    base_img.put_pixel(px0 + dx, py0 + dy, image::Rgba([r, g, b, 255]));
                    if has_orm {
                        let [ao, rough, metal] = orm[i];
                        orm_img.put_pixel(px0 + dx, py0 + dy, image::Rgba([ao, rough, metal, 255]));
                    }
                }
            }
            uvs[i] = [
                (px0 as f32 + CELL as f32 * 0.5) / tex_w as f32,
                1.0 - (py0 as f32 + CELL as f32 * 0.5) / tex_h as f32,
            ];
        }

        let base_png = encode_png_rgba(&base_img);
        let orm_png = if has_orm {
            Some(encode_png_rgba(&orm_img))
        } else {
            None
        };

        let mut bin: Vec<u8> = Vec::with_capacity(n_v * 20 + n_f * 12 + base_png.len() + 64);
        let mut min_p = [f32::INFINITY; 3];
        let mut max_p = [f32::NEG_INFINITY; 3];
        for v in &self.mesh.vertices {
            for k in 0..3 {
                min_p[k] = min_p[k].min(v[k]);
                max_p[k] = max_p[k].max(v[k]);
                bin.extend_from_slice(&v[k].to_le_bytes());
            }
        }
        let pos_len = bin.len();
        align4(&mut bin);

        let uv_offset = bin.len();
        for uv in &uvs {
            bin.extend_from_slice(&uv[0].to_le_bytes());
            bin.extend_from_slice(&uv[1].to_le_bytes());
        }
        let uv_len = n_v * 8;
        align4(&mut bin);

        let index_offset = bin.len();
        for f in &self.mesh.faces {
            for &idx in f {
                bin.extend_from_slice(&idx.to_le_bytes());
            }
        }
        let index_len = n_f * 12;
        align4(&mut bin);

        let base_img_offset = bin.len();
        bin.extend_from_slice(&base_png);
        let base_img_len = base_png.len();
        align4(&mut bin);

        let (orm_img_offset, orm_img_len) = if let Some(ref png) = orm_png {
            let off = bin.len();
            bin.extend_from_slice(png);
            let len = png.len();
            align4(&mut bin);
            (Some(off), Some(len))
        } else {
            (None, None)
        };

        let bin_len = bin.len();

        // bufferViews: 0=pos, 1=uv, 2=indices, 3=base png, [4=orm png]
        let mut buffer_views = vec![
            serde_json::json!({"buffer":0,"byteOffset":0,"byteLength":pos_len,"target":34962}),
            serde_json::json!({"buffer":0,"byteOffset":uv_offset,"byteLength":uv_len,"target":34962}),
            serde_json::json!({"buffer":0,"byteOffset":index_offset,"byteLength":index_len,"target":34963}),
            serde_json::json!({"buffer":0,"byteOffset":base_img_offset,"byteLength":base_img_len}),
        ];
        let mut images = vec![serde_json::json!({"bufferView":3,"mimeType":"image/png"})];
        let mut textures = vec![serde_json::json!({"source":0,"sampler":0})];
        let mut pbr = serde_json::json!({
            "baseColorTexture": { "index": 0 },
            "metallicFactor": if has_orm { 1.0 } else { 0.0 },
            "roughnessFactor": 1.0
        });
        if let (Some(off), Some(len)) = (orm_img_offset, orm_img_len) {
            let bv = buffer_views.len();
            buffer_views.push(serde_json::json!({"buffer":0,"byteOffset":off,"byteLength":len}));
            let img_i = images.len();
            images.push(serde_json::json!({"bufferView":bv,"mimeType":"image/png"}));
            let tex_i = textures.len();
            textures.push(serde_json::json!({"source":img_i,"sampler":0}));
            pbr["metallicRoughnessTexture"] = serde_json::json!({ "index": tex_i });
        }

        let json = serde_json::json!({
            "asset": { "version": "2.0", "generator": "rlx-trellis2" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0, "TEXCOORD_0": 1 },
                    "indices": 2,
                    "material": 0,
                    "mode": 4
                }]
            }],
            "materials": [{
                "name": "trellis2_pbr",
                "pbrMetallicRoughness": pbr,
                "doubleSided": true
            }],
            "samplers": [{
                "magFilter": 9728,
                "minFilter": 9728,
                "wrapS": 33071,
                "wrapT": 33071
            }],
            "textures": textures,
            "images": images,
            "buffers": [{ "byteLength": bin_len }],
            "bufferViews": buffer_views,
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": n_v,
                    "type": "VEC3",
                    "max": max_p,
                    "min": min_p
                },
                {
                    "bufferView": 1,
                    "componentType": 5126,
                    "count": n_v,
                    "type": "VEC2"
                },
                {
                    "bufferView": 2,
                    "componentType": 5125,
                    "count": n_f * 3,
                    "type": "SCALAR"
                }
            ]
        });

        pack_glb(json, &bin)
    }

    /// Per-vertex base RGB and optional ORM (AO, roughness, metallic) u8 texels.
    fn vertex_pbr_texels(&self) -> (Vec<[u8; 3]>, Vec<[u8; 3]>, bool) {
        let n_v = self.mesh.vertices.len();
        let colors = self.vertex_base_colors();
        if self.coords.is_empty() || self.attrs.len() < self.coords.len() * 6 {
            return (colors, vec![[255, 255, 0]; n_v], false);
        }
        let mut index: HashMap<[i32; 3], usize> = HashMap::with_capacity(self.coords.len());
        for (i, c) in self.coords.iter().enumerate() {
            index.insert(*c, i);
        }
        let mut orm = Vec::with_capacity(n_v);
        let aligned = n_v == self.coords.len() && self.attrs.len() == self.coords.len() * 6;
        let gs = self.grid_size.max(1) as f32;
        for i in 0..n_v {
            let ai = if aligned {
                i
            } else {
                let v = self.mesh.vertices[i];
                let gx = ((v[0] + 0.5) * gs).floor() as i32;
                let gy = ((v[1] + 0.5) * gs).floor() as i32;
                let gz = ((v[2] + 0.5) * gs).floor() as i32;
                nearest_attr_index(&index, [gx, gy, gz]).unwrap_or(0)
            };
            let o = ai * 6;
            let metal = (self.attrs[o + 3].clamp(0.0, 1.0) * 255.0).round() as u8;
            let rough = (self.attrs[o + 4].clamp(0.0, 1.0) * 255.0).round() as u8;
            orm.push([255, rough, metal]); // R=AO unused
        }
        (colors, orm, true)
    }

    /// Per-mesh-vertex base-color bytes from sparse PBR attrs.
    fn vertex_base_colors(&self) -> Vec<[u8; 3]> {
        let n_v = self.mesh.vertices.len();
        if self.coords.is_empty() || self.attrs.len() < 6 {
            return vec![[200, 200, 200]; n_v];
        }
        let mut index: HashMap<[i32; 3], usize> = HashMap::with_capacity(self.coords.len());
        for (i, c) in self.coords.iter().enumerate() {
            index.insert(*c, i);
        }
        // Fast path: dual-grid verts align 1:1 with shape/tex voxel order.
        if n_v == self.coords.len() && self.attrs.len() == self.coords.len() * 6 {
            return (0..n_v)
                .map(|i| {
                    let a = &self.attrs[i * 6..i * 6 + 3];
                    [
                        (a[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (a[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (a[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    ]
                })
                .collect();
        }
        let gs = self.grid_size.max(1) as f32;
        let mut out = Vec::with_capacity(n_v);
        for v in &self.mesh.vertices {
            let gx = ((v[0] + 0.5) * gs).floor() as i32;
            let gy = ((v[1] + 0.5) * gs).floor() as i32;
            let gz = ((v[2] + 0.5) * gs).floor() as i32;
            let rgb = nearest_attr_rgb(&index, &self.attrs, [gx, gy, gz]);
            out.push(rgb);
        }
        out
    }
}

fn nearest_attr_rgb(index: &HashMap<[i32; 3], usize>, attrs: &[f32], query: [i32; 3]) -> [u8; 3] {
    match nearest_attr_index(index, query) {
        Some(i) => attr_rgb(attrs, i),
        None => [200, 200, 200],
    }
}

fn nearest_attr_index(index: &HashMap<[i32; 3], usize>, query: [i32; 3]) -> Option<usize> {
    if let Some(&i) = index.get(&query) {
        return Some(i);
    }
    let mut best = None;
    let mut best_d = i32::MAX;
    for dx in -2..=2 {
        for dy in -2..=2 {
            for dz in -2..=2 {
                let c = [query[0] + dx, query[1] + dy, query[2] + dz];
                if let Some(&i) = index.get(&c) {
                    let d = dx.abs() + dy.abs() + dz.abs();
                    if d < best_d {
                        best_d = d;
                        best = Some(i);
                    }
                }
            }
        }
    }
    best
}

fn attr_rgb(attrs: &[f32], i: usize) -> [u8; 3] {
    let o = i * 6;
    if o + 3 > attrs.len() {
        return [200, 200, 200];
    }
    [
        (attrs[o].clamp(0.0, 1.0) * 255.0).round() as u8,
        (attrs[o + 1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (attrs[o + 2].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

fn align4(buf: &mut Vec<u8>) {
    while !buf.len().is_multiple_of(4) {
        buf.push(0);
    }
}

fn encode_png_rgba(img: &image::RgbaImage) -> Vec<u8> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    img.write_to(&mut cursor, image::ImageFormat::Png)
        .expect("png encode");
    cursor.into_inner()
}

fn pack_glb(json: serde_json::Value, bin: &[u8]) -> Vec<u8> {
    let mut json_bytes = serde_json::to_vec(&json).expect("glb json");
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    let bin_len = bin.len();
    let total = 12 + 8 + json_bytes.len() + 8 + bin_len;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes()); // JSON
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin_len as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942u32.to_le_bytes()); // BIN
    out.extend_from_slice(bin);
    out
}

/// Decode a `[N, 7]` FlexiDualGrid decoder output into a mesh.
///
/// Channel layout (per `FlexiDualGridVaeDecoder.forward`): `0..3` vertex offset
/// (→ `2·σ(·) - 0.5`, `voxel_margin = 0.5`), `3..6` intersected logits (→ `> 0`),
/// `6..7` split logit (→ `softplus`). `grid_size` is the decode resolution and
/// `aabb` the world box (`[-0.5,-0.5,-0.5]..[0.5,0.5,0.5]`).
pub fn dual_grid_to_mesh(
    decoded: &SparseTensor,
    grid_size: usize,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> Mesh {
    let n = decoded.n();
    debug_assert_eq!(decoded.c, 7);
    let voxel_size = [
        (aabb_max[0] - aabb_min[0]) / grid_size as f32,
        (aabb_max[1] - aabb_min[1]) / grid_size as f32,
        (aabb_max[2] - aabb_min[2]) / grid_size as f32,
    ];

    // per-voxel dual vertex, intersected flags, split weight
    let mut vertices = Vec::with_capacity(n);
    let mut intersected = vec![[false; 3]; n];
    let mut split_w = vec![0.0f32; n];
    for i in 0..n {
        let f = &decoded.feats[i * 7..i * 7 + 7];
        let off = [
            2.0 * sigmoid(f[0]) - 0.5,
            2.0 * sigmoid(f[1]) - 0.5,
            2.0 * sigmoid(f[2]) - 0.5,
        ];
        let [cx, cy, cz] = decoded.coords[i];
        vertices.push([
            (cx as f32 + off[0]) * voxel_size[0] + aabb_min[0],
            (cy as f32 + off[1]) * voxel_size[1] + aabb_min[1],
            (cz as f32 + off[2]) * voxel_size[2] + aabb_min[2],
        ]);
        intersected[i] = [f[3] > 0.0, f[4] > 0.0, f[5] > 0.0];
        split_w[i] = softplus(f[6]);
    }

    // spatial hashmap coord -> index (replaces the CUDA `_C` hashmap)
    let mut index: HashMap<[i32; 3], u32> = HashMap::with_capacity(n);
    for (i, c) in decoded.coords.iter().enumerate() {
        index.insert(*c, i as u32);
    }

    // emit quads around intersected edges, triangulate by split weight
    let mut faces = Vec::new();
    for i in 0..n {
        let c = decoded.coords[i];
        for axis in 0..3 {
            if !intersected[i][axis] {
                continue;
            }
            let mut quad = [0u32; 4];
            let mut ok = true;
            for k in 0..4 {
                let o = EDGE_NEIGHBOR_OFFSET[axis][k];
                let nb = [c[0] + o[0], c[1] + o[1], c[2] + o[2]];
                match index.get(&nb) {
                    Some(&idx) => quad[k] = idx,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let sw02 = split_w[quad[0] as usize] * split_w[quad[2] as usize];
            let sw13 = split_w[quad[1] as usize] * split_w[quad[3] as usize];
            let split = if sw02 > sw13 {
                &QUAD_SPLIT_1
            } else {
                &QUAD_SPLIT_2
            };
            faces.push([quad[split[0]], quad[split[1]], quad[split[2]]]);
            faces.push([quad[split[3]], quad[split[4]], quad[split[5]]]);
        }
    }

    Mesh { vertices, faces }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    // numerically stable ln(1+e^x)
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

impl Mesh {
    /// Write a Wavefront OBJ (vertices + triangular faces, 1-indexed).
    pub fn to_obj(&self) -> String {
        let mut s = String::with_capacity(self.vertices.len() * 24 + self.faces.len() * 20);
        for v in &self.vertices {
            s.push_str(&format!("v {} {} {}\n", v[0], v[1], v[2]));
        }
        for f in &self.faces {
            s.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_quad_two_triangles() {
        // 4 voxels forming one x-axis edge quad, all intersected on axis 0.
        // coords chosen so voxel 0's x-edge neighbors (offsets for x-axis) are
        // exactly the other three.
        let coords = vec![[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]];
        let mut feats = vec![0.0f32; 4 * 7];
        for i in 0..4 {
            // vertex offset logits 0 -> sigmoid .5 -> offset 0.5 (voxel center)
            feats[i * 7 + 6] = 1.0; // split weight positive
        }
        feats[3] = 1.0; // voxel 0 intersected on x-axis
        let st = SparseTensor::new(feats, coords, 7);
        let m = dual_grid_to_mesh(&st, 2, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
        assert_eq!(m.vertices.len(), 4);
        assert_eq!(m.faces.len(), 2, "one quad -> two triangles");
    }

    #[test]
    fn incomplete_quad_dropped() {
        // only 3 of the 4 edge neighbors present -> no face
        let coords = vec![[0, 0, 0], [0, 0, 1], [0, 1, 1]];
        let mut feats = vec![0.0f32; 3 * 7];
        feats[3] = 1.0;
        let st = SparseTensor::new(feats, coords, 7);
        let m = dual_grid_to_mesh(&st, 2, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
        assert_eq!(m.faces.len(), 0);
    }

    #[test]
    fn ply_writes_vertex_colors() {
        let mesh = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
        };
        let mut attrs = vec![0.0f32; 6];
        attrs[0] = 1.0; // red
        attrs[1] = 0.0;
        attrs[2] = 0.0;
        let mp = MeshWithPbr {
            mesh,
            coords: vec![[0, 0, 0]],
            attrs,
            grid_size: 2,
        };
        let ply = mp.to_ply();
        assert!(ply.contains("property uchar red"));
        assert!(ply.contains("255 0 0"), "expected red vertex: {ply}");
        assert!(ply.contains("3 0 1 2"));

        let glb = mp.to_glb();
        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(glb[4..8].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize,
            glb.len()
        );
        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let json = std::str::from_utf8(&glb[20..20 + json_len]).unwrap();
        assert!(json.contains("baseColorTexture"), "{json}");
        assert!(json.contains("TEXCOORD_0"), "{json}");
        assert!(json.contains("image/png"), "{json}");
        assert!(json.contains("metallicRoughnessTexture"), "{json}");
    }
}
