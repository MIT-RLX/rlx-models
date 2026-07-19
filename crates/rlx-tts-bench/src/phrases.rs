//! Fixed bench phrases (override via CLI).

pub const DEFAULT_SHORT: &str = "The quick brown fox jumps over the lazy dog near the river bank.";

pub const DEFAULT_LONG: &str = "\
Once upon a time in a quiet valley, a traveler paused by the river and listened \
to the wind in the trees. Birds called from the hills, and the water carried \
stories of distant mountains down toward the sea. The traveler smiled, took a \
deep breath, and continued walking toward the next town.";

pub const FOX_WORDS: &[&str] = &["quick", "brown", "fox", "jumps", "lazy", "dog"];

pub fn content_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}
