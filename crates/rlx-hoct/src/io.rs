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

//! Label I/O, CTC export, and minimal GEFF solution dumps.
//!
//! Primary path for challenge-style output is [`write_ctc`] /
//! [`write_solution`] with [`OutputFormat::Ctc`]. Full zarr-GEFF is out of
//! scope; [`write_geff_minimal`] writes a JSON lineage graph.

use crate::features::NodeFeatures;
use crate::ilp::IlpSolution;
use anyhow::{Context, Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{Array3, Array4};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::Path;

/// Simple raw format: `[ndim: u32][dim0: u32]…[data: little-endian u32 labels]`.
pub fn load_labels_raw(path: impl AsRef<Path>) -> Result<Array3<u32>> {
    let mut f = BufReader::new(File::open(path.as_ref())?);
    let ndim = f.read_u32::<LittleEndian>()? as usize;
    if ndim == 0 || ndim > 4 {
        bail!("unsupported ndim {ndim}");
    }
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(f.read_u32::<LittleEndian>()? as usize);
    }
    let count: usize = shape.iter().product();
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        data.push(f.read_u32::<LittleEndian>()?);
    }
    match ndim {
        2 => Array3::from_shape_vec((1, shape[0], shape[1]), data).context("2D labels"),
        3 => Array3::from_shape_vec((shape[0], shape[1], shape[2]), data).context("3D labels"),
        4 => {
            let a4 = Array4::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), data)?;
            if shape[3] == 1 {
                Ok(a4.remove_axis(ndarray::Axis(3)))
            } else {
                bail!("4D labels with last dim != 1 not supported");
            }
        }
        _ => bail!("unsupported ndim"),
    }
}

/// Write labels in the same raw format [`load_labels_raw`] reads.
pub fn write_labels_raw(path: impl AsRef<Path>, labels: &Array3<u32>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = File::create(path)?;
    let (t, h, w) = (
        labels.len_of(ndarray::Axis(0)),
        labels.len_of(ndarray::Axis(1)),
        labels.len_of(ndarray::Axis(2)),
    );
    f.write_all(&3u32.to_le_bytes())?;
    f.write_all(&(t as u32).to_le_bytes())?;
    f.write_all(&(h as u32).to_le_bytes())?;
    f.write_all(&(w as u32).to_le_bytes())?;
    for v in labels.iter() {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(feature = "tiff-io")]
pub fn load_tiff_stack(dir: impl AsRef<Path>) -> Result<Array3<u32>> {
    use tiff::decoder::{Decoder, DecodingResult};
    let dir = dir.as_ref();
    let mut paths: Vec<std::path::PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "tif" || x == "tiff"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no TIFF files in {}", dir.display());
    }
    let mut frames = Vec::new();
    let (mut h, mut w) = (0usize, 0usize);
    for p in &paths {
        let file = File::open(p)?;
        let mut dec = Decoder::new(BufReader::new(file))?;
        let img = dec.read_image()?;
        let (y, x) = dec.dimensions();
        h = y as usize;
        w = x as usize;
        let slice = match img {
            DecodingResult::U8(v) => v.into_iter().map(|v| v as u32).collect::<Vec<_>>(),
            DecodingResult::U16(v) => v.into_iter().map(|v| v as u32).collect(),
            DecodingResult::U32(v) => v,
            other => bail!("unsupported TIFF dtype: {other:?}"),
        };
        frames.push(slice);
    }
    let t = frames.len();
    let mut out = Array3::<u32>::zeros((t, h, w));
    for (ti, frame) in frames.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                out[[ti, y, x]] = frame[y * w + x];
            }
        }
    }
    Ok(out)
}

#[cfg(not(feature = "tiff-io"))]
pub fn load_tiff_stack(_dir: impl AsRef<Path>) -> Result<Array3<u32>> {
    bail!("TIFF support requires `--features tiff-io`")
}

/// One CTC tracklet row: `L B E P` (label, begin, end, parent; parent `0` = none).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtcTrack {
    pub label: u32,
    pub begin: u32,
    pub end: u32,
    pub parent: u32,
}

/// Reconstruct CTC tracklets from an ILP solution + node metadata.
///
/// Tracklets are maximal non-branching chains of active nodes linked by
/// solution edges. Division creates a new tracklet with `parent` set to the
/// parent's tracklet id.
pub fn assign_tracklets(nodes: &[NodeFeatures], sol: &IlpSolution) -> Vec<CtcTrack> {
    let active: HashSet<usize> = sol.active_nodes.iter().copied().collect();
    if active.is_empty() {
        return Vec::new();
    }

    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut parent_of: HashMap<usize, usize> = HashMap::new();
    for link in &sol.links {
        if active.contains(&link.src) && active.contains(&link.dst) {
            children.entry(link.src).or_default().push(link.dst);
            parent_of.insert(link.dst, link.src);
        }
    }
    for kids in children.values_mut() {
        kids.sort_by_key(|&j| {
            (
                nodes.get(j).map(|n| n.t as i32).unwrap_or(0),
                nodes.get(j).map(|n| n.label).unwrap_or(0),
            )
        });
    }

    let mut roots: Vec<usize> = active
        .iter()
        .copied()
        .filter(|i| !parent_of.contains_key(i))
        .collect();
    roots.sort_by_key(|&i| {
        (
            nodes.get(i).map(|n| n.t as i32).unwrap_or(0),
            nodes.get(i).map(|n| n.label).unwrap_or(0),
        )
    });

    let mut tracks: Vec<CtcTrack> = Vec::new();
    let mut node_track: HashMap<usize, u32> = HashMap::new();
    let mut next_id = 1u32;

    fn flush_track(
        chain: &[usize],
        parent_track: u32,
        nodes: &[NodeFeatures],
        tracks: &mut Vec<CtcTrack>,
        node_track: &mut HashMap<usize, u32>,
        next_id: &mut u32,
    ) -> u32 {
        if chain.is_empty() {
            return 0;
        }
        let id = *next_id;
        *next_id += 1;
        let begin = nodes[chain[0]].t.round() as u32;
        let end = nodes[*chain.last().unwrap()].t.round() as u32;
        for &n in chain {
            node_track.insert(n, id);
        }
        tracks.push(CtcTrack {
            label: id,
            begin,
            end,
            parent: parent_track,
        });
        id
    }

    fn walk(
        start: usize,
        parent_track: u32,
        children: &HashMap<usize, Vec<usize>>,
        nodes: &[NodeFeatures],
        tracks: &mut Vec<CtcTrack>,
        node_track: &mut HashMap<usize, u32>,
        next_id: &mut u32,
    ) {
        let mut chain = vec![start];
        let mut cur = start;
        loop {
            let kids = children.get(&cur).map(|v| v.as_slice()).unwrap_or(&[]);
            if kids.len() == 1 {
                chain.push(kids[0]);
                cur = kids[0];
                continue;
            }
            let tid = flush_track(&chain, parent_track, nodes, tracks, node_track, next_id);
            for &k in kids {
                walk(k, tid, children, nodes, tracks, node_track, next_id);
            }
            break;
        }
    }

    for root in roots {
        walk(
            root,
            0,
            &children,
            nodes,
            &mut tracks,
            &mut node_track,
            &mut next_id,
        );
    }
    tracks
}

/// Write CTC-style `res_track.txt` plus per-frame raw masks (`maskNNNN.raw`).
pub fn write_ctc(
    out_dir: impl AsRef<Path>,
    labels: &Array3<u32>,
    tracks: &[CtcTrack],
) -> Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    let track_path = out_dir.join("res_track.txt");
    let mut f = File::create(&track_path)?;
    for t in tracks {
        writeln!(f, "{} {} {} {}", t.label, t.begin, t.end, t.parent)?;
    }
    let t_max = labels.len_of(ndarray::Axis(0));
    let n_digits = ((t_max as f32).log10().floor() as usize + 1).max(3);
    for t in 0..t_max {
        let name = format!("mask{t:0n_digits$}");
        let path = out_dir.join(name);
        write_raw_mask(
            &path.with_extension("raw"),
            labels.slice(ndarray::s![t, .., ..]),
        )?;
    }
    Ok(())
}

fn write_raw_mask(path: &Path, frame: ndarray::ArrayView2<u32>) -> Result<()> {
    let (h, w) = (
        frame.len_of(ndarray::Axis(0)),
        frame.len_of(ndarray::Axis(1)),
    );
    let mut f = File::create(path)?;
    f.write_all(&2u32.to_le_bytes())?;
    f.write_all(&(h as u32).to_le_bytes())?;
    f.write_all(&(w as u32).to_le_bytes())?;
    for v in frame.iter() {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Minimal GEFF-compatible solution dump (JSON nodes + edges).
///
/// Full zarr-GEFF is deferred; this subset is enough for solution-graph
/// comparison and for tooling that only needs lineage edges.
pub fn write_geff_minimal(
    out_dir: impl AsRef<Path>,
    nodes: &[NodeFeatures],
    sol: &IlpSolution,
    tracks: &[CtcTrack],
) -> Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let active: HashSet<usize> = sol.active_nodes.iter().copied().collect();
    let node_json: Vec<_> = nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| active.contains(i))
        .map(|(i, n)| {
            json!({
                "id": i,
                "label": n.label,
                "t": n.t,
                "z": n.z,
                "y": n.y,
                "x": n.x,
                "solution": true,
            })
        })
        .collect();

    let edge_json: Vec<_> = sol
        .links
        .iter()
        .map(|l| {
            json!({
                "source_id": l.src,
                "target_id": l.dst,
                "delta_t": l.delta_t,
                "edge_id": l.edge_id,
                "solution": true,
            })
        })
        .collect();

    let track_json: Vec<_> = tracks
        .iter()
        .map(|t| {
            json!({
                "label": t.label,
                "begin": t.begin,
                "end": t.end,
                "parent": t.parent,
            })
        })
        .collect();

    let doc = json!({
        "format": "rlx-hoct-geff-minimal",
        "version": 1,
        "nodes": node_json,
        "edges": edge_json,
        "tracks": track_json,
        "appearances": sol.appearances,
        "disappearances": sol.disappearances,
        "divisions": sol.divisions,
    });

    let path = out_dir.join("tracks.json");
    let mut f = File::create(&path)?;
    serde_json::to_writer_pretty(&mut f, &doc)?;
    writeln!(f)?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Ctc,
    Geff,
}

impl OutputFormat {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "ctc" => Ok(Self::Ctc),
            "geff" => Ok(Self::Geff),
            other => bail!("unknown output format `{other}` (use ctc|geff)"),
        }
    }
}

/// Write solution in CTC and/or GEFF-minimal form.
pub fn write_solution(
    out_dir: impl AsRef<Path>,
    labels: &Array3<u32>,
    nodes: &[NodeFeatures],
    sol: &IlpSolution,
    format: OutputFormat,
) -> Result<()> {
    let tracks = assign_tracklets(nodes, sol);
    match format {
        OutputFormat::Ctc => write_ctc(out_dir, labels, &tracks),
        OutputFormat::Geff => write_geff_minimal(out_dir, nodes, sol, &tracks),
    }
}
