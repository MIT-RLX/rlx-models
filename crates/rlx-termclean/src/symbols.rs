//! The symbol inventory used to synthesize terminal "chrome".
//!
//! This is deliberately broad — the whole point of the dataset is to teach the
//! model that a rich variety of special UTF-8 glyphs and ASCII art are *chrome*
//! (to drop), distinct from the same-looking symbols that appear as real
//! *content* (to keep). If a glyph family is missing here, the model never
//! learns to strip it, so we cover the common Unicode blocks a TUI draws with:
//!   - Box Drawing            U+2500–U+257F  (borders, gutters, tables)
//!   - Block Elements         U+2580–U+259F  (bars, scrollbars, shading)
//!   - Braille Patterns       U+2800–U+28FF  (spinners)
//!   - Arrows / bullets / misc symbols        (list markers, indicators)
//!
//! Plus raw ANSI CSI escape sequences (SGR colors/attrs).

/// A complete box-drawing character set for one border style.
#[derive(Clone, Copy)]
pub struct BoxStyle {
    pub name: &'static str,
    pub tl: &'static str,
    pub tr: &'static str,
    pub bl: &'static str,
    pub br: &'static str,
    pub h: &'static str,
    pub v: &'static str,
    pub ltee: &'static str,
    pub rtee: &'static str,
    pub ttee: &'static str,
    pub btee: &'static str,
    pub cross: &'static str,
}

pub const LIGHT: BoxStyle = BoxStyle {
    name: "light",
    tl: "┌",
    tr: "┐",
    bl: "└",
    br: "┘",
    h: "─",
    v: "│",
    ltee: "├",
    rtee: "┤",
    ttee: "┬",
    btee: "┴",
    cross: "┼",
};
pub const HEAVY: BoxStyle = BoxStyle {
    name: "heavy",
    tl: "┏",
    tr: "┓",
    bl: "┗",
    br: "┛",
    h: "━",
    v: "┃",
    ltee: "┣",
    rtee: "┫",
    ttee: "┳",
    btee: "┻",
    cross: "╋",
};
pub const DOUBLE: BoxStyle = BoxStyle {
    name: "double",
    tl: "╔",
    tr: "╗",
    bl: "╚",
    br: "╝",
    h: "═",
    v: "║",
    ltee: "╠",
    rtee: "╣",
    ttee: "╦",
    btee: "╩",
    cross: "╬",
};
pub const ROUNDED: BoxStyle = BoxStyle {
    name: "rounded",
    tl: "╭",
    tr: "╮",
    bl: "╰",
    br: "╯",
    h: "─",
    v: "│",
    ltee: "├",
    rtee: "┤",
    ttee: "┬",
    btee: "┴",
    cross: "┼",
};
pub const ASCII: BoxStyle = BoxStyle {
    name: "ascii",
    tl: "+",
    tr: "+",
    bl: "+",
    br: "+",
    h: "-",
    v: "|",
    ltee: "+",
    rtee: "+",
    ttee: "+",
    btee: "+",
    cross: "+",
};
pub const DASHED: BoxStyle = BoxStyle {
    name: "dashed",
    tl: "┌",
    tr: "┐",
    bl: "└",
    br: "┘",
    h: "┄",
    v: "┆",
    ltee: "├",
    rtee: "┤",
    ttee: "┬",
    btee: "┴",
    cross: "┼",
};

pub static BOX_STYLES: &[BoxStyle] = &[LIGHT, HEAVY, DOUBLE, ROUNDED, ASCII, DASHED];

/// Full block, used to fill progress bars.
pub const BAR_FULL: &str = "█";
/// Horizontal ellipsis, used to mark truncated content.
pub const ELLIPSIS: &str = "…";

/// Shading / partial blocks (scrollbars, empty bar track, dithering).
pub static SHADES: &[&str] = &["░", "▒", "▓", "█"];
/// Left-to-right partial blocks (fine-grained bars).
pub static PARTIAL_BLOCKS: &[&str] = &["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];

/// Spinner frame sets (a single frame is drawn per sample).
pub static SPINNERS: &[&[&str]] = &[
    &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    &["|", "/", "-", "\\"],
    &["◐", "◓", "◑", "◒"],
    &["▖", "▘", "▝", "▗"],
    &["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
];

/// List bullet markers.
pub static BULLETS: &[&str] = &["•", "◦", "‣", "▪", "▸", "–", "*", "-", "·", "○", "●", "»"];

/// Arrows and indicators (also appear as content in `corpus::UNI` — the model
/// must disambiguate by role, which is the point).
pub static ARROWS: &[&str] = &[
    "→", "←", "↑", "↓", "▶", "◀", "▲", "▼", "»", "«", "›", "↦", "⇒",
];

/// SGR (Select Graphic Rendition) parameter strings: attributes, 16-color,
/// bright, 256-color, and truecolor. Rendered as `\x1b[<params>m`.
pub static SGR_PARAMS: &[&str] = &[
    "1",
    "2",
    "3",
    "4",
    "7",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "90",
    "91",
    "92",
    "93",
    "94",
    "95",
    "96",
    "1;31",
    "1;32",
    "1;34",
    "4;36",
    "7;33",
    "38;5;208",
    "38;5;45",
    "38;5;196",
    "48;5;236",
    "38;2;255;135;0",
    "38;2;80;250;123",
];
