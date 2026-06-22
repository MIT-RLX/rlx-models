//! Port of the subset of `inflect.number_to_words` exercised by
//! `tiny_tts/text/english_utils/number_norm.py::normalize_numbers`.
//!
//! Covers: cardinals (`andword=""`), the 1000–3000 "year" special-casing
//! (group=2, zero="oh"), ordinals (`andword="and"`), and currency expansion.
//! Validated against Python `inflect` over a wide range in the frontend tests.

const UNIT: [&str; 10] = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
];
const TEEN: [&str; 10] = [
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TEN: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const MILL: [&str; 12] = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
];

/// `tenfn(tens, units)` for a 0..99 value split into tens/units.
fn ten_fn(t: usize, u: usize) -> String {
    if t == 1 {
        TEEN[u].to_string()
    } else if t > 0 {
        if u > 0 {
            format!("{}-{}", TEN[t], UNIT[u])
        } else {
            TEN[t].to_string()
        }
    } else {
        UNIT[u].to_string()
    }
}

/// Spell 1..=999 (no period word). `andword` is "" or "and".
fn below_thousand(v: usize, andword: &str) -> String {
    let h = v / 100;
    let rem = v % 100;
    let (t, u) = (rem / 10, rem % 10);
    if h > 0 {
        let mut s = format!("{} hundred", UNIT[h]);
        if rem > 0 {
            s.push(' ');
            if !andword.is_empty() {
                s.push_str(andword);
                s.push(' ');
            }
            s.push_str(&ten_fn(t, u));
        }
        s
    } else {
        ten_fn(t, u)
    }
}

/// `number_to_words(n, andword=…)` for non-negative integers (group=0).
fn cardinal_u64(n: u64, andword: &str) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    let mut groups = Vec::new();
    let mut x = n;
    while x > 0 {
        groups.push((x % 1000) as usize);
        x /= 1000;
    }
    let lowest_idx = groups.iter().position(|&v| v != 0).unwrap();
    let mut parts: Vec<String> = Vec::new();
    for idx in (0..groups.len()).rev() {
        let v = groups[idx];
        if v == 0 {
            continue;
        }
        let mut s = below_thousand(v, andword);
        if idx > 0 {
            s.push(' ');
            s.push_str(MILL[idx]);
        }
        parts.push(s);
    }
    // COMMA_WORD: drop the comma before the final group iff it's a single token
    // (the lowest period group, value < 100).
    let remove_comma = lowest_idx == 0 && groups[0] < 100 && parts.len() >= 2;
    if remove_comma {
        let last = parts.pop().unwrap();
        format!("{} {}", parts.join(", "), last)
    } else {
        parts.join(", ")
    }
}

/// Cardinal of a signed integer with `andword=""` ("minus" sign like inflect).
pub fn cardinal(n: i64) -> String {
    if n < 0 {
        format!("minus {}", cardinal_u64((-n) as u64, ""))
    } else {
        cardinal_u64(n as u64, "")
    }
}

/// `_sub_ord`: turn a cardinal word string into its ordinal form.
fn sub_ord(val: &str) -> String {
    // ordinal_suff = (ty|one|two|three|five|eight|nine|twelve)\Z
    const SUFF: [(&str, &str); 8] = [
        ("twelve", "twelfth"),
        ("three", "third"),
        ("eight", "eighth"),
        ("nine", "ninth"),
        ("five", "fifth"),
        ("one", "first"),
        ("two", "second"),
        ("ty", "tieth"),
    ];
    for (k, v) in SUFF {
        if let Some(stem) = val.strip_suffix(k) {
            return format!("{stem}{v}");
        }
    }
    format!("{val}th")
}

/// `_expand_ordinal`: ordinal words for a non-negative integer (andword="and").
pub fn ordinal(n: u64) -> String {
    sub_ord(&cardinal_u64(n, "and"))
}

/// group=2 year spelling (two-digit groups), `zero` token e.g. "oh".
pub fn year_group2(n: u64, zero: &str) -> String {
    let digits: Vec<u32> = n
        .to_string()
        .chars()
        .map(|c| c.to_digit(10).unwrap())
        .collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i + 1 < digits.len() {
        let t = digits[i] as usize;
        let u = digits[i + 1] as usize;
        let word = if t > 0 {
            ten_fn(t, u)
        } else if u > 0 {
            format!("{zero} {}", UNIT[u])
        } else {
            format!("{zero} {zero}")
        };
        out.push(word);
        i += 2;
    }
    if i < digits.len() {
        // leftover single digit (group1bsub): unit if nonzero else zero
        let u = digits[i] as usize;
        out.push(if u > 0 {
            UNIT[u].to_string()
        } else {
            zero.to_string()
        });
    }
    out.join(" ")
}

/// `_expand_number`: the generic int spelling with the 1000–3000 year rules.
pub fn expand_number(n: i64) -> String {
    if n > 1000 && n < 3000 {
        let num = n as u64;
        if num == 2000 {
            return "two thousand".to_string();
        } else if (2000..2010).contains(&num) {
            return format!("two thousand {}", cardinal_u64(num % 100, "and"));
        } else if num.is_multiple_of(100) {
            return format!("{} hundred", cardinal_u64(num / 100, "and"));
        } else {
            return year_group2(num, "oh");
        }
    }
    cardinal(n)
}

/// Currency table entry: (cent, cents, unit, units).
struct Currency {
    cent: &'static str,
    cents: &'static str,
    unit: &'static str,
    units: &'static str,
}

fn currency_for(sym: char) -> Currency {
    match sym {
        '$' => Currency {
            cent: "cent",
            cents: "cents",
            unit: "dollar",
            units: "dollars",
        },
        '€' => Currency {
            cent: "cent",
            cents: "cents",
            unit: "euro",
            units: "euros",
        },
        '£' => Currency {
            cent: "penny",
            cents: "pence",
            unit: "pound sterling",
            units: "pounds sterling",
        },
        '¥' => Currency {
            cent: "sen",
            cents: "sen",
            unit: "yen",
            units: "yen",
        },
        _ => Currency {
            cent: "cent",
            cents: "cents",
            unit: "dollar",
            units: "dollars",
        },
    }
}

/// `__expand_currency`: returns text with *digit* integer/fraction + unit words
/// (the digits are re-spelled by the later number pass), matching Python.
pub fn expand_currency(sym: char, value: &str) -> String {
    let c = currency_for(sym);
    let cleaned = value.replace(',', "");
    let parts: Vec<&str> = cleaned.split('.').collect();
    if parts.len() > 2 {
        return format!("{value} {}", c.units);
    }
    let mut text: Vec<String> = Vec::new();
    let integer: i64 = if parts[0].is_empty() {
        0
    } else {
        parts[0].parse().unwrap_or(0)
    };
    if integer > 0 {
        let unit = if integer == 1 { c.unit } else { c.units };
        text.push(format!("{integer} {unit}"));
    }
    let fraction: i64 = if parts.len() > 1 && !parts[1].is_empty() {
        parts[1].parse().unwrap_or(0)
    } else {
        0
    };
    if fraction > 0 {
        // inflection.get(fraction/100, inflection[0.02]): only 0.01 → singular cent.
        let unit = if fraction == 1 { c.cent } else { c.cents };
        text.push(format!("{fraction} {unit}"));
    }
    if text.is_empty() {
        return format!("zero {}", c.units);
    }
    text.join(" ")
}
