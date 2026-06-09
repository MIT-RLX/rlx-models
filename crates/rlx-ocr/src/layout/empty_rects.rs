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

//! Max-empty-rectangle search for column/block layout (ported from upstream `ocrs`).

use rten_imageproc::Rect;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Partition {
    score: f32,
    boundary: Rect,
    obstacles: Vec<Rect>,
}

impl PartialEq for Partition {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Partition {}

impl Ord for Partition {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

impl PartialOrd for Partition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    queue: BinaryHeap<Partition>,
    score: S,
    min_width: u32,
    min_height: u32,
}

impl<S> MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    fn new(obstacles: &[Rect], boundary: Rect, score: S, min_width: u32, min_height: u32) -> Self {
        let mut queue = BinaryHeap::new();
        let mut obstacles = obstacles.to_vec();
        obstacles.sort_by_key(|o| {
            let c = o.center();
            (c.x, c.y)
        });

        if !boundary.is_empty() {
            queue.push(Partition {
                score: score(boundary),
                boundary,
                obstacles: obstacles.clone(),
            });
        }

        Self {
            queue,
            score,
            min_width,
            min_height,
        }
    }
}

impl<S> Iterator for MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    type Item = Rect;

    fn next(&mut self) -> Option<Rect> {
        let choose_pivot = |r: &[Rect]| r[r.len() / 2];

        while let Some(part) = self.queue.pop() {
            let Partition {
                score: _,
                boundary: b,
                obstacles,
            } = part;

            if obstacles.is_empty() {
                return Some(b);
            }

            let pivot = choose_pivot(&obstacles);
            let right_rect = Rect::from_tlbr(b.top(), pivot.right(), b.bottom(), b.right());
            let left_rect = Rect::from_tlbr(b.top(), b.left(), b.bottom(), pivot.left());
            let top_rect = Rect::from_tlbr(b.top(), b.left(), pivot.top(), b.right());
            let bottom_rect = Rect::from_tlbr(pivot.bottom(), b.left(), b.bottom(), b.right());

            for sr in [top_rect, left_rect, bottom_rect, right_rect] {
                if (sr.width().max(0) as u32) < self.min_width
                    || (sr.height().max(0) as u32) < self.min_height
                    || sr.is_empty()
                {
                    continue;
                }

                let sr_obstacles: Vec<_> = obstacles
                    .iter()
                    .filter(|o| o.intersects(sr))
                    .copied()
                    .collect();
                assert!(sr_obstacles.len() < obstacles.len());

                self.queue.push(Partition {
                    score: (self.score)(sr),
                    obstacles: sr_obstacles,
                    boundary: sr,
                });
            }
        }

        None
    }
}

pub fn max_empty_rects<S>(
    obstacles: &[Rect],
    boundary: Rect,
    score: S,
    min_width: u32,
    min_height: u32,
) -> MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    MaxEmptyRects::new(obstacles, boundary, score, min_width, min_height)
}

pub trait FilterOverlapping {
    type Output: Iterator<Item = Rect>;
    fn filter_overlapping(self, factor: f32) -> Self::Output;
}

pub struct FilterRectIter<I: Iterator<Item = Rect>> {
    source: I,
    found: Vec<Rect>,
    overlap_threshold: f32,
}

impl<I: Iterator<Item = Rect>> FilterRectIter<I> {
    fn new(source: I, overlap_threshold: f32) -> Self {
        Self {
            source,
            found: Vec::new(),
            overlap_threshold,
        }
    }
}

impl<I: Iterator<Item = Rect>> Iterator for FilterRectIter<I> {
    type Item = Rect;

    fn next(&mut self) -> Option<Rect> {
        while let Some(r) = self.source.next() {
            if self
                .found
                .iter()
                .any(|f| f.iou(r) >= self.overlap_threshold)
            {
                continue;
            }
            self.found.push(r);
            return Some(r);
        }
        None
    }
}

impl<I: Iterator<Item = Rect>> FilterOverlapping for I {
    type Output = FilterRectIter<I>;

    fn filter_overlapping(self, factor: f32) -> Self::Output {
        FilterRectIter::new(self, factor)
    }
}
