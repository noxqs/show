//! Real type, for subtitles.
//!
//! The 5x7 bitmap font the rest of the interface uses is right for a readout
//! and wrong for dialogue: it has no curves, no proportional widths and no
//! anti-aliasing, and a paragraph set in it is tiring in a way a line of
//! timings is not. Subtitles are the one thing in this program that is read
//! rather than glanced at.
//!
//! The face is Noto Sans, embedded. That is deliberately the same typeface
//! libass ends up using on this desktop — it asks fontconfig for `sans-serif`
//! and gets this — but embedding it keeps the promise the rest of the program
//! makes: no freetype, no fontconfig, no harfbuzz, nothing to install. It is
//! rasterised by `ab_glyph`, which is pure Rust.
//!
//! What is given up against libass is real text shaping: no ligatures, no
//! contextual forms, no bidirectional reordering. For Latin subtitles that
//! costs nothing visible. For Arabic or Devanagari it would, and those want a
//! shaper rather than a better font.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};

/// OFL 1.1, and the licence travels with the binary in `assets/`.
const NOTO_SANS: &[u8] = include_bytes!("../assets/NotoSans-Regular.ttf");

fn face() -> &'static FontRef<'static> {
    static FACE: std::sync::OnceLock<FontRef<'static>> = std::sync::OnceLock::new();
    FACE.get_or_init(|| {
        FontRef::try_from_slice(NOTO_SANS).expect("the embedded font is built into the binary")
    })
}

/// Distance from one baseline to the next.
pub fn line_height(px: f32) -> f32 {
    let scaled = face().as_scaled(PxScale::from(px));
    scaled.height() + scaled.line_gap()
}

/// Width of `text` when set at `px`.
pub fn measure(text: &str, px: f32) -> f32 {
    let scaled = face().as_scaled(PxScale::from(px));
    let mut width = 0.0;
    let mut previous = None;
    for c in text.chars() {
        let id = scaled.glyph_id(c);
        if let Some(previous) = previous {
            width += scaled.kern(previous, id);
        }
        width += scaled.h_advance(id);
        previous = Some(id);
    }
    width
}

/// Whether the face has a glyph of its own for `c`.
///
/// A face without one draws `.notdef`, which in Noto is an empty box — the
/// squares that appear around song lyrics, because subtitle files mark those
/// with musical notes this face does not carry.
pub fn supported(c: char) -> bool {
    face().glyph_id(c).0 != 0
}

/// Remove what the face cannot draw.
///
/// Dropping a character loses something, and it is the lesser loss: a box
/// conveys nothing at all, while the words around it still read. Invisible
/// formatting characters go too — they have glyphs, so they survive the test
/// above, but a subtitle has no use for a byte-order mark.
pub fn strip_unsupported(text: &str) -> String {
    let kept: String = text
        .chars()
        .filter(|c| !matches!(c, '\u{feff}' | '\u{200b}' | '\u{200e}' | '\u{200f}' | '\u{00ad}'))
        .filter(|c| c.is_whitespace() || supported(*c))
        .collect();

    // Removing a character can leave the gap it sat in, and a line that was
    // "\u{266a} Hello \u{266a}" should not come out padded at both ends.
    kept.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A rasterised line: coverage from 0 to 255, one byte per pixel.
pub struct Mask {
    pub coverage: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Mask {
    pub fn at(&self, x: usize, y: usize) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.coverage[y * self.width + x]
    }
}

/// Rasterise one line into a coverage mask.
///
/// A mask rather than a drawing, because the caller needs the same shape
/// several times over: an outline is this stamped around itself in black
/// before being stamped once in white, and re-rasterising for each of those
/// would be nine times the work for an identical result.
pub fn mask(text: &str, px: f32) -> Mask {
    let font = face();
    let scaled = font.as_scaled(PxScale::from(px));

    let width = measure(text, px).ceil() as usize + 2;
    let height = line_height(px).ceil() as usize + 2;
    let baseline = scaled.ascent();
    let mut coverage = vec![0u8; width * height];

    let mut pen = 1.0f32;
    let mut previous = None;
    for c in text.chars() {
        let id = scaled.glyph_id(c);
        if let Some(previous) = previous {
            pen += scaled.kern(previous, id);
        }
        let glyph = id.with_scale_and_position(PxScale::from(px), point(pen, baseline));
        if let Some(outlined) = font.outline_glyph(glyph) {
            let bounds = outlined.px_bounds();
            outlined.draw(|gx, gy, c| {
                let x = bounds.min.x as i32 + gx as i32;
                let y = bounds.min.y as i32 + gy as i32;
                if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
                    return;
                }
                let i = y as usize * width + x as usize;
                // Glyphs can overlap where they are tightly kerned; the darker
                // of the two is what a single stroke would have produced.
                let v = (c * 255.0).round().clamp(0.0, 255.0) as u8;
                coverage[i] = coverage[i].max(v);
            });
        }
        pen += scaled.h_advance(id);
        previous = Some(id);
    }

    Mask { coverage, width, height }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_face_loads() {
        assert!(measure("A", 32.0) > 0.0);
    }

    #[test]
    fn width_grows_with_the_text_and_the_size() {
        assert!(measure("AA", 32.0) > measure("A", 32.0));
        assert!(measure("A", 64.0) > measure("A", 32.0));
    }

    #[test]
    fn the_face_is_proportional_rather_than_fixed_width() {
        // The whole reason for using it: an i is not as wide as an m.
        assert!(measure("i", 32.0) < measure("m", 32.0));
    }

    #[test]
    fn a_mask_has_ink_in_it() {
        let m = mask("Hello", 32.0);
        let ink = m.coverage.iter().filter(|c| **c > 0).count();
        assert!(ink > 20, "only {ink} pixels of ink");
        assert!(m.coverage.iter().any(|c| *c > 200), "nothing is fully covered");
    }

    #[test]
    fn a_mask_is_anti_aliased() {
        // Partial coverage is the difference from the bitmap font.
        let m = mask("So", 48.0);
        assert!(
            m.coverage.iter().any(|c| *c > 20 && *c < 235),
            "no partial coverage: the edges are hard"
        );
    }

    #[test]
    fn accents_and_quotes_are_real_glyphs_now() {
        // The bitmap font folds these to a base letter; a real face has them.
        assert!(measure("é", 32.0) > 0.0);
        assert!(measure("ñ", 32.0) > 0.0);
        assert!(mask("é", 32.0).coverage.iter().any(|c| *c > 0));
    }

    #[test]
    fn characters_the_face_lacks_are_reported_as_such() {
        assert!(supported('A') && supported('é') && supported('\u{2014}'));
        // Musical notes, which subtitle files put around song lyrics, and
        // which this face does not have.
        assert!(!supported('\u{266a}'), "the face turned out to have a note after all");
        assert!(!supported('\u{1f600}'), "no emoji either");
    }

    #[test]
    fn stripping_leaves_the_words_and_drops_the_boxes() {
        assert_eq!(strip_unsupported("\u{266a} Hello there \u{266a}"), "Hello there");
        assert_eq!(strip_unsupported("Caf\u{e9} \u{2014} naive"), "Caf\u{e9} \u{2014} naive");
        // A line that was nothing but notes has nothing left to say.
        assert_eq!(strip_unsupported("\u{266a}\u{266b}"), "");
    }

    #[test]
    fn invisible_formatting_characters_go_too() {
        // These have glyphs, so the coverage test alone would keep them.
        assert_eq!(strip_unsupported("\u{feff}Hello\u{200b}"), "Hello");
    }

    #[test]
    fn stripping_does_not_leave_the_gap_behind() {
        // The note and its space both go, rather than one of them.
        let out = strip_unsupported("\u{266a}  Singing  now  \u{266a}");
        assert_eq!(out, "Singing now");
    }

    #[test]
    fn a_blank_line_rasterises_to_nothing() {
        let m = mask("   ", 32.0);
        assert!(m.coverage.iter().all(|c| *c == 0));
    }
}
