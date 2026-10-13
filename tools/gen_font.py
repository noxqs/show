#!/usr/bin/env python3
"""Generate src/font.rs from ASCII art.

The OSD needs text, and a font crate plus an embedded TTF is a lot of machinery
for a readout that is mostly digits. A 5x7 bitmap font costs about 350 bytes and
no dependencies. Written as art here so a wrong pixel is visible rather than
hidden in a hex table.
"""
G = {
 ' ': "..... ..... ..... ..... ..... ..... .....",
 'A': ".###. #...# #...# ##### #...# #...# #...#",
 'B': "####. #...# ####. #...# #...# #...# ####.",
 'C': ".###. #...# #.... #.... #.... #...# .###.",
 'D': "####. #...# #...# #...# #...# #...# ####.",
 'E': "##### #.... ####. #.... #.... #.... #####",
 'F': "##### #.... ####. #.... #.... #.... #....",
 'G': ".###. #...# #.... #.### #...# #...# .###.",
 'H': "#...# #...# ##### #...# #...# #...# #...#",
 'I': "..#.. ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
 'J': "....# ....# ....# ....# #...# #...# .###.",
 'K': "#...# #..#. #.#.. ##... #.#.. #..#. #...#",
 'L': "#.... #.... #.... #.... #.... #.... #####",
 'M': "#...# ##.## #.#.# #...# #...# #...# #...#",
 'N': "#...# ##..# #.#.# #..## #...# #...# #...#",
 'O': ".###. #...# #...# #...# #...# #...# .###.",
 'P': "####. #...# #...# ####. #.... #.... #....",
 'Q': ".###. #...# #...# #...# #.#.# #..#. .##.#",
 'R': "####. #...# #...# ####. #.#.. #..#. #...#",
 'S': ".###. #...# #.... .###. ....# #...# .###.",
 'T': "##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
 'U': "#...# #...# #...# #...# #...# #...# .###.",
 'V': "#...# #...# #...# #...# #...# .#.#. ..#..",
 'W': "#...# #...# #...# #.#.# #.#.# ##.## #...#",
 'X': "#...# #...# .#.#. ..#.. .#.#. #...# #...#",
 'Y': "#...# #...# .#.#. ..#.. ..#.. ..#.. ..#..",
 'Z': "##### ....# ...#. ..#.. .#... #.... #####",
 '0': ".###. #...# #..## #.#.# ##..# #...# .###.",
 '1': "..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.",
 '2': ".###. #...# ....# ...#. ..#.. .#... #####",
 '3': "####. ....# ....# .###. ....# ....# ####.",
 '4': "...#. ..##. .#.#. #..#. ##### ...#. ...#.",
 '5': "##### #.... ####. ....# ....# #...# .###.",
 '6': ".###. #.... #.... ####. #...# #...# .###.",
 '7': "##### ....# ...#. ..#.. .#... .#... .#...",
 '8': ".###. #...# #...# .###. #...# #...# .###.",
 '9': ".###. #...# #...# .#### ....# ....# .###.",
 '.': "..... ..... ..... ..... ..... .##.. .##..",
 ',': "..... ..... ..... ..... .##.. .##.. .#...",
 ':': "..... .##.. .##.. ..... .##.. .##.. .....",
 '-': "..... ..... ..... ##### ..... ..... .....",
 '_': "..... ..... ..... ..... ..... ..... #####",
 '/': "....# ....# ...#. ..#.. .#... #.... #....",
 '(': "..##. .#... .#... .#... .#... .#... ..##.",
 ')': ".##.. ...#. ...#. ...#. ...#. ...#. .##..",
 '[': ".###. .#... .#... .#... .#... .#... .###.",
 ']': ".###. ...#. ...#. ...#. ...#. ...#. .###.",
 '%': "#...# #...# ...#. ..#.. .#... #...# #...#",
 '+': "..... ..#.. ..#.. ##### ..#.. ..#.. .....",
 '>': "#.... .#... ..#.. ...#. ..#.. .#... #....",
 '<': "....# ...#. ..#.. .#... ..#.. ...#. ....#",
 '*': "..... #.#.# .###. ##### .###. #.#.# .....",
 '=': "..... ..... ##### ..... ##### ..... .....",
 '!': "..#.. ..#.. ..#.. ..#.. ..#.. ..... ..#..",
 '?': ".###. #...# ....# ...#. ..#.. ..... ..#..",
 "'": "..#.. ..#.. ..... ..... ..... ..... .....",
 # Lowercase. Subtitles are dialogue, and dialogue set entirely in capitals
 # reads as shouting, so the font grew a second case rather than the captions
 # being folded. Seven rows is too few for a true descender below the baseline,
 # so the descenders of g j p q y take the bottom row and the bowl sits one row
 # higher — one baseline for both cases, which matters more at this size than a
 # textbook descender does.
 'a': "..... ..... .###. ....# .#### #...# .####",
 'b': "#.... #.... ####. #...# #...# #...# ####.",
 'c': "..... ..... .###. #.... #.... #.... .###.",
 'd': "....# ....# .#### #...# #...# #...# .####",
 'e': "..... ..... .###. #...# ##### #.... .###.",
 'f': "..##. .#..# .#... ###.. .#... .#... .#...",
 'g': "..... ..... .#### #...# .#### ....# .###.",
 'h': "#.... #.... ####. #...# #...# #...# #...#",
 'i': "..#.. ..... .##.. ..#.. ..#.. ..#.. .###.",
 'j': "...#. ..... ..##. ...#. ...#. #..#. .##..",
 'k': "#.... #.... #..#. #.#.. ##... #.#.. #..#.",
 'l': ".##.. ..#.. ..#.. ..#.. ..#.. ..#.. .###.",
 'm': "..... ..... ##.#. #.#.# #.#.# #.#.# #.#.#",
 'n': "..... ..... ####. #...# #...# #...# #...#",
 'o': "..... ..... .###. #...# #...# #...# .###.",
 'p': "..... ..... ####. #...# #...# ####. #....",
 'q': "..... ..... .#### #...# #...# .#### ....#",
 'r': "..... ..... #.##. ##..# #.... #.... #....",
 's': "..... ..... .#### #.... .###. ....# ####.",
 't': ".#... .#... ###.. .#... .#... .#..# ..##.",
 'u': "..... ..... #...# #...# #...# #..## .##.#",
 'v': "..... ..... #...# #...# #...# .#.#. ..#..",
 'w': "..... ..... #...# #...# #.#.# #.#.# .#.#.",
 'x': "..... ..... #...# .#.#. ..#.. .#.#. #...#",
 'y': "..... ..... #...# #...# .#### ....# .###.",
 'z': "..... ..... ##### ...#. ..#.. .#... #####",
 ';': "..... .##.. .##.. ..... .##.. .##.. .#...",
 '"': ".#.#. .#.#. ..... ..... ..... ..... .....",
 '#': ".#.#. ##### .#.#. ##### .#.#. ..... .....",
 '&': ".##.. #..#. .##.. .##.# #..#. #..#. .##.#",
 '@': ".###. #...# #.### #.#.# #.### #.... .###.",
}

rows = []
for ch in sorted(G):
    art = G[ch].split()
    assert len(art) == 7, f"{ch!r} has {len(art)} rows"
    bits = []
    for r in art:
        assert len(r) == 5, f"{ch!r} row {r!r} is not 5 wide"
        bits.append("0b" + "".join("1" if c == "#" else "0" for c in r) + "_000")
    rows.append((ch, bits))

out = ['''//! A 5x7 bitmap font for the on-screen display.
//!
//! Generated by `tools/gen_font.py` — edit the ASCII art there, not this file.
//!
//! Both cases. The readouts are written in capitals, but subtitles are
//! dialogue, and dialogue set entirely in capitals reads as shouting.
//! An unmapped character falls back to its uppercase form rather than a hole.

/// Glyph width in pixels, before scaling.
pub const WIDTH: usize = 5;
/// Glyph height in pixels, before scaling.
pub const HEIGHT: usize = 7;
/// Blank columns between glyphs.
pub const SPACING: usize = 1;

/// Rows are top to bottom; bit 7 is the leftmost pixel.
pub fn glyph(c: char) -> Option<&\'static [u8; HEIGHT]> {
    let find = |c: char| GLYPHS.iter().find(|(ch, _)| *ch == c).map(|(_, bits)| bits);
    find(c).or_else(|| find(fold(c)))
}

/// Strip an accent down to the letter underneath, keeping the case.
///
/// Subtitles are full of these, and a caption reading "CAF" because the E was
/// acute is worse than one reading "CAFE". Latin-1 and the common Latin
/// Extended-A letters only; anything else is left alone and simply has no
/// glyph.
fn fold(c: char) -> char {
    match c {
        \'\\u{00C0}\'..=\'\\u{00C5}\' | \'\\u{0100}\' | \'\\u{0102}\' | \'\\u{0104}\' => \'A\',
        \'\\u{00C7}\' | \'\\u{0106}\' | \'\\u{010C}\' => \'C\',
        \'\\u{00C8}\'..=\'\\u{00CB}\' | \'\\u{0112}\' | \'\\u{011A}\' | \'\\u{0118}\' => \'E\',
        \'\\u{00CC}\'..=\'\\u{00CF}\' | \'\\u{012A}\' => \'I\',
        \'\\u{00D1}\' | \'\\u{0143}\' | \'\\u{0147}\' => \'N\',
        \'\\u{00D2}\'..=\'\\u{00D6}\' | \'\\u{00D8}\' | \'\\u{014C}\' | \'\\u{0150}\' => \'O\',
        \'\\u{00D9}\'..=\'\\u{00DC}\' | \'\\u{016A}\' | \'\\u{016E}\' | \'\\u{0170}\' => \'U\',
        \'\\u{00DD}\' => \'Y\',
        \'\\u{015A}\' | \'\\u{0160}\' => \'S\',
        \'\\u{0179}\' | \'\\u{017B}\' | \'\\u{017D}\' => \'Z\',
        \'\\u{0141}\' => \'L\',
        \'\\u{00E0}\'..=\'\\u{00E5}\' | \'\\u{0101}\' | \'\\u{0103}\' | \'\\u{0105}\' => \'a\',
        \'\\u{00E7}\' | \'\\u{0107}\' | \'\\u{010D}\' => \'c\',
        \'\\u{00E8}\'..=\'\\u{00EB}\' | \'\\u{0113}\' | \'\\u{011B}\' | \'\\u{0119}\' => \'e\',
        \'\\u{00EC}\'..=\'\\u{00EF}\' | \'\\u{012B}\' => \'i\',
        \'\\u{00F1}\' | \'\\u{0144}\' | \'\\u{0148}\' => \'n\',
        \'\\u{00F2}\'..=\'\\u{00F6}\' | \'\\u{00F8}\' | \'\\u{014D}\' | \'\\u{0151}\' => \'o\',
        \'\\u{00F9}\'..=\'\\u{00FC}\' | \'\\u{016B}\' | \'\\u{016F}\' | \'\\u{0171}\' => \'u\',
        \'\\u{00FD}\' | \'\\u{00FF}\' => \'y\',
        \'\\u{00DF}\' | \'\\u{015B}\' | \'\\u{0161}\' => \'s\',
        \'\\u{017A}\' | \'\\u{017C}\' | \'\\u{017E}\' => \'z\',
        \'\\u{0142}\' => \'l\',
        // A typographic quote or dash is a character we do have, spelled
        // differently; subtitle files use them constantly.
        \'\\u{2018}\' | \'\\u{2019}\' => \'\\\'\',
        \'\\u{201C}\' | \'\\u{201D}\' => \'"\',
        \'\\u{2013}\' | \'\\u{2014}\' => \'-\',
        \'\\u{2026}\' => \'.\',
        other => other.to_uppercase().next().unwrap_or(other),
    }
}

/// Width in pixels of `text` at scale 1.
pub fn text_width(text: &str) -> usize {
    let n = text.chars().count();
    if n == 0 { 0 } else { n * (WIDTH + SPACING) - SPACING }
}

#[rustfmt::skip]
static GLYPHS: &[(char, [u8; HEIGHT])] = &[''']
for ch, bits in rows:
    lit = "'\\''" if ch == "'" else ("'\\\\'" if ch == "\\" else f"'{ch}'")
    out.append(f"    ({lit}, [{', '.join(bits)}]),")
out.append("];")
out.append('''
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_glyph_is_reachable_and_both_cases_exist() {
        assert!(glyph('A').is_some());
        assert!(glyph('a').is_some(), "dialogue needs lowercase");
        assert_ne!(glyph('a'), glyph('A'), "the two cases must differ");
        assert_eq!(glyph('\\u{00C9}'), glyph('E'), "accents fall back to the base letter");
        assert_eq!(glyph('\\u{00E9}'), glyph('e'), "and keep their case");
        assert_eq!(glyph('\\u{2019}'), glyph('\\''), "typographic quotes map to the plain one");
        assert!(glyph('0').is_some());
        assert!(glyph(' ').is_some());
        assert!(glyph('\\u{263A}').is_none(), "unmapped characters must not panic");
    }

    #[test]
    fn space_is_blank_and_a_digit_is_not() {
        assert!(glyph(' ').unwrap().iter().all(|&r| r == 0));
        assert!(glyph('8').unwrap().iter().any(|&r| r != 0));
    }

    #[test]
    fn text_width_accounts_for_spacing() {
        assert_eq!(text_width(""), 0);
        assert_eq!(text_width("A"), WIDTH);
        assert_eq!(text_width("AB"), WIDTH * 2 + SPACING);
    }
}''')
open("src/font.rs", "w").write("\n".join(out) + "\n")
print(f"wrote src/font.rs with {len(rows)} glyphs")
