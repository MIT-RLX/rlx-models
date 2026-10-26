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

//! `rlx_flux3::bbox` against BFL `flux3/bbox.py`, string for string, on the
//! cases `scripts/flux3_bbox_cases.py` wrote to `tests/data/bbox_cases.json`.

use rlx_flux3::bbox::{self, Json};

fn cases() -> Option<Json> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/bbox_cases.json");
    let Ok(s) = std::fs::read_to_string(&p) else {
        eprintln!("skip: no {}", p.display());
        return None;
    };
    Some(Json::parse(&s).unwrap())
}

fn list<'a>(c: &'a Json, key: &str) -> &'a [Json] {
    match c.get(key) {
        Some(Json::List(l)) => l,
        _ => panic!("no `{key}` list"),
    }
}

fn s<'a>(c: &'a Json, key: &str) -> &'a str {
    c.get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("no `{key}`"))
}

fn boxes(c: &Json) -> Option<Vec<Json>> {
    match c.get("boxes") {
        Some(Json::List(l)) => Some(l.clone()),
        _ => None,
    }
}

#[test]
fn to_canvas_matches_python() {
    let Some(all) = cases() else { return };
    for c in list(&all, "to_canvas") {
        let rows = bbox::to_canvas(&boxes(c).unwrap()).unwrap();
        assert_eq!(Json::List(rows).dumps(), s(c, "rows"));
    }
}

#[test]
fn detect_and_finished_match_python() {
    let Some(all) = cases() else { return };
    for c in list(&all, "detect") {
        let p = s(c, "prompt");
        let got = bbox::detect(p).map(|(t, rows)| (t, Json::List(rows).dumps()));
        let want = match c.get("detected") {
            Some(Json::List(v)) => Some((
                v[0].as_str().unwrap().to_string(),
                v[1].as_str().unwrap().to_string(),
            )),
            _ => None,
        };
        assert_eq!(got, want, "detect({p:.80})");
        assert_eq!(
            bbox::is_finished(p),
            c.get("finished") == Some(&Json::Bool(true)),
            "is_finished({p:.80})"
        );
    }
}

#[test]
fn plan_matches_python() {
    let Some(all) = cases() else { return };
    for c in list(&all, "plan") {
        let n = match c.get("n_images") {
            Some(Json::Int(n)) => *n as usize,
            _ => panic!("n_images"),
        };
        let b = boxes(c);
        let got = bbox::plan(s(c, "prompt"), b.as_deref(), n).unwrap();
        assert_eq!(got.task, s(c, "task"));
        assert_eq!(got.text, s(c, "text"));
        let want_locked = c.get("locked").and_then(Json::as_str).map(str::to_string);
        assert_eq!(got.locked.map(|l| Json::List(l).dumps()), want_locked);
    }
}

#[test]
fn reconcile_matches_python() {
    let Some(all) = cases() else { return };
    for c in list(&all, "reconcile") {
        let Json::List(locked) = Json::parse(s(c, "locked")).unwrap() else {
            panic!("locked")
        };
        assert_eq!(bbox::reconcile(s(c, "text"), &locked), s(c, "out"));
    }
}
