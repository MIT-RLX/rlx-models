//! Deterministic synthetic English prompt corpus (≥1000 unique lines).
//!
//! Built from combinatorial templates so stress benches need no paid LLM API.
//! Optional JSONL / plain-text files can augment or replace the built-in set.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};

/// One stress-bench utterance.
#[derive(Debug, Clone)]
pub struct CorpusItem {
    pub id: String,
    pub text: String,
    pub category: &'static str,
}

/// Generate `n` unique phrases (default stress size is 1000+).
pub fn generate_corpus(n: usize, seed: u64) -> Vec<CorpusItem> {
    let mut out = Vec::with_capacity(n);
    let mut seen = HashSet::new();
    let mut i = 0u64;
    // Cap attempts so we still return if uniqueness stalls (should not for n≤5k).
    let max_attempts = (n as u64).saturating_mul(32).max(10_000);
    while out.len() < n && i < max_attempts {
        let item = phrase_at(seed.wrapping_add(i));
        i += 1;
        if seen.insert(item.text.clone()) {
            let id = format!("syn_{:04}", out.len());
            out.push(CorpusItem {
                id,
                text: item.text,
                category: item.category,
            });
        }
    }
    out
}

/// Load extra lines from a file (`#` comments / blank lines skipped).
/// Each non-empty line becomes `file_NNNN`.
pub fn load_corpus_file(path: &Path) -> Result<Vec<CorpusItem>> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("read corpus {}", path.display()))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        // JSONL: {"text":"..."} or plain text.
        let text = if t.starts_with('{') {
            let v: serde_json::Value = serde_json::from_str(t)
                .with_context(|| format!("jsonl line in {}", path.display()))?;
            v.get("text")
                .and_then(|x| x.as_str())
                .unwrap_or(t)
                .to_string()
        } else {
            t.to_string()
        };
        if text.is_empty() {
            continue;
        }
        out.push(CorpusItem {
            id: format!("file_{:04}", out.len()),
            text,
            category: "file",
        });
    }
    Ok(out)
}

struct Draft {
    text: String,
    category: &'static str,
}

fn phrase_at(seed: u64) -> Draft {
    let kind = (seed % 12) as usize;
    match kind {
        0 => stmt(seed, "everyday"),
        1 => stmt(seed, "travel"),
        2 => stmt(seed, "tech"),
        3 => question(seed),
        4 => command(seed),
        5 => numberish(seed),
        6 => time_place(seed),
        7 => name_intro(seed),
        8 => weather(seed),
        9 => shopping(seed),
        10 => longish(seed),
        _ => mixed(seed),
    }
}

fn pick<'a>(seed: u64, lane: u64, xs: &[&'a str]) -> &'a str {
    xs[((seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(lane * 0x85EB_CA6B))
        % xs.len() as u64) as usize]
}

fn stmt(seed: u64, cat: &'static str) -> Draft {
    let subj = pick(
        seed,
        1,
        &[
            "The traveler",
            "A quiet robot",
            "My neighbor",
            "The engineer",
            "Our team",
            "The librarian",
            "A small bird",
            "The pilot",
            "Her brother",
            "His sister",
            "The baker",
            "A young student",
            "The captain",
            "Our guide",
            "The musician",
        ],
    );
    let verb = pick(
        seed,
        2,
        &[
            "noticed",
            "described",
            "remembered",
            "explained",
            "mentioned",
            "recorded",
            "shared",
            "finished",
            "started",
            "checked",
            "opened",
            "closed",
            "carried",
            "followed",
            "measured",
        ],
    );
    let obj = match cat {
        "travel" => pick(
            seed,
            3,
            &[
                "the river crossing before sunset",
                "a narrow path through the pines",
                "the ferry schedule on the dock",
                "ticket prices for the morning train",
                "the map folded in the backpack",
                "cold air above the mountain pass",
                "footprints near the wooden bridge",
                "lanterns along the harbor wall",
            ],
        ),
        "tech" => pick(
            seed,
            3,
            &[
                "the latest build logs on the server",
                "a subtle bug in the audio pipeline",
                "latency numbers from the metal path",
                "the checksum for the weight bundle",
                "quantization error on the fine head",
                "a green light on the network rack",
                "the diff between native and reference",
                "memory pressure during long decode",
            ],
        ),
        _ => pick(
            seed,
            3,
            &[
                "a soft breeze across the porch",
                "fresh bread cooling on the counter",
                "the dog sleeping by the window",
                "pages turning in an old book",
                "coffee steaming beside the keyboard",
                "rain tapping on the glass roof",
                "children laughing in the yard",
                "a letter waiting on the table",
            ],
        ),
    };
    Draft {
        text: format!("{subj} {verb} {obj}."),
        category: cat,
    }
}

fn question(seed: u64) -> Draft {
    let q = pick(
        seed,
        1,
        &[
            "Can you repeat that a little slower",
            "Where should we meet after lunch",
            "How long will the transfer take",
            "Did the package arrive this morning",
            "Which platform leaves for Boston first",
            "Would you like tea or water",
            "Is the library open on Sunday",
            "What time does the store close tonight",
            "Could you spell your last name",
            "Are the samples ready for review",
            "Should we restart the service now",
            "Have you seen the missing notebook",
        ],
    );
    Draft {
        text: format!("{q}?"),
        category: "question",
    }
}

fn command(seed: u64) -> Draft {
    let c = pick(
        seed,
        1,
        &[
            "Please open the window a few inches",
            "Turn left at the next traffic light",
            "Save the draft before you leave",
            "Speak clearly into the microphone",
            "Set a timer for twelve minutes",
            "Send the report to the shared folder",
            "Play the sample at half volume",
            "Check the battery level again",
            "Write down the confirmation number",
            "Keep walking until you see the oak tree",
            "Pause the recording for a moment",
            "Read the next sentence out loud",
        ],
    );
    Draft {
        text: format!("{c}."),
        category: "command",
    }
}

fn numberish(seed: u64) -> Draft {
    let n1 = 10 + (seed % 90) as u32;
    let n2 = 100 + ((seed / 7) % 900) as u32;
    let n3 = 1 + ((seed / 13) % 31) as u32;
    let tmpl = pick(
        seed,
        1,
        &[
            "Order number {a} ships in {b} hours.",
            "Room {a} is on floor {c}.",
            "The invoice totals {b} dollars and {c} cents.",
            "Call extension {a} before {c} o'clock.",
            "There are {a} samples in batch {b}.",
            "Gate {c} boards in {a} minutes.",
            "We need {a} copies of page {c}.",
            "The code is {a} {b} {c}.",
        ],
    );
    Draft {
        text: tmpl
            .replace("{a}", &n1.to_string())
            .replace("{b}", &n2.to_string())
            .replace("{c}", &n3.to_string()),
        category: "numbers",
    }
}

fn time_place(seed: u64) -> Draft {
    let day = pick(
        seed,
        1,
        &[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
    );
    let place = pick(
        seed,
        2,
        &[
            "downtown",
            "the airport",
            "Central Station",
            "the clinic",
            "Riverside Park",
            "the museum lobby",
            "gate twelve",
            "the east entrance",
        ],
    );
    let when = pick(
        seed,
        3,
        &[
            "early morning",
            "around noon",
            "late afternoon",
            "just after dusk",
            "before midnight",
            "at half past three",
        ],
    );
    Draft {
        text: format!("On {day} we will meet {place} {when}."),
        category: "time_place",
    }
}

fn name_intro(seed: u64) -> Draft {
    let name = pick(
        seed,
        1,
        &[
            "Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn",
            "Cameron", "Harper", "Drew", "Nora", "Elena", "Marcus",
        ],
    );
    let role = pick(
        seed,
        2,
        &[
            "a systems engineer",
            "a speech researcher",
            "a product designer",
            "a field technician",
            "a language teacher",
            "a studio producer",
            "a travel writer",
            "a clinic coordinator",
        ],
    );
    let city = pick(
        seed,
        3,
        &[
            "Austin",
            "Seattle",
            "Chicago",
            "Boston",
            "Denver",
            "Portland",
            "Atlanta",
            "Minneapolis",
        ],
    );
    Draft {
        text: format!("Hello, my name is {name} and I am {role} from {city}."),
        category: "intro",
    }
}

fn weather(seed: u64) -> Draft {
    let w = pick(
        seed,
        1,
        &[
            "Clear skies are expected through the evening with a light breeze from the west.",
            "Scattered showers may pass before noon, then cooler air arrives.",
            "Fog will lift by mid morning along the coast and inland valleys.",
            "Temperatures climb into the upper seventies under bright sun.",
            "A winter advisory warns of icy patches on untreated roads overnight.",
            "Humidity stays high while thunderstorms build over the plains.",
            "Winds calm after sunset, making for a quiet night outdoors.",
            "Snow flurries are possible above three thousand feet this afternoon.",
        ],
    );
    Draft {
        text: w.to_string(),
        category: "weather",
    }
}

fn shopping(seed: u64) -> Draft {
    let item = pick(
        seed,
        1,
        &[
            "fresh apples",
            "whole grain bread",
            "oat milk",
            "cheddar cheese",
            "green tea",
            "rice noodles",
            "black pepper",
            "olive oil",
            "paper towels",
            "sparkling water",
        ],
    );
    let store = pick(
        seed,
        2,
        &[
            "the corner market",
            "aisle four",
            "the bakery counter",
            "the cold case",
            "online pickup",
            "the farmers stall",
        ],
    );
    Draft {
        text: format!("Please add {item} from {store} to the list."),
        category: "shopping",
    }
}

fn longish(seed: u64) -> Draft {
    let a = stmt(seed, "everyday")
        .text
        .trim_end_matches('.')
        .to_string();
    let b = pick(
        seed,
        9,
        &[
            "After that, take a short walk and stretch your shoulders.",
            "Meanwhile, keep notes so nothing important is forgotten.",
            "If anything seems unclear, ask one careful follow-up question.",
            "Then double-check the timestamps before you archive the file.",
            "Finally, thank everyone who helped along the way.",
            "Later, compare the new recording with yesterday's take.",
        ],
    );
    Draft {
        text: format!("{a}. {b}"),
        category: "long",
    }
}

fn mixed(seed: u64) -> Draft {
    let adj = pick(
        seed,
        1,
        &[
            "careful", "rapid", "gentle", "precise", "cheerful", "steady", "curious", "patient",
        ],
    );
    let noun = pick(
        seed,
        2,
        &[
            "summary",
            "demo",
            "rehearsal",
            "handoff",
            "checklist",
            "warmup",
            "review",
            "handover",
        ],
    );
    let adv = pick(
        seed,
        3,
        &[
            "today",
            "this week",
            "before launch",
            "after lunch",
            "once more",
            "without delay",
        ],
    );
    Draft {
        text: format!("Please give a {adj} {noun} {adv}."),
        category: "mixed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_at_least_1000_unique() {
        let c = generate_corpus(1000, 42);
        assert_eq!(c.len(), 1000);
        let set: HashSet<_> = c.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(set.len(), 1000);
        assert!(c.iter().all(|x| !x.text.is_empty()));
    }

    #[test]
    fn corpus_deterministic() {
        let a = generate_corpus(50, 7);
        let b = generate_corpus(50, 7);
        assert_eq!(
            a.iter().map(|x| &x.text).collect::<Vec<_>>(),
            b.iter().map(|x| &x.text).collect::<Vec<_>>()
        );
    }
}
