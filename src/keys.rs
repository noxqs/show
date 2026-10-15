//! The keyboard reference page.
//!
//! Every binding this program has, on a page of its own. It used to live as a
//! section inside the setup panel, which put a list you consult while looking
//! at a picture behind a menu you open to change a setting, and made a page
//! that was already long enough to need shrinking longer still.
//!
//! This is the one place the list is written down. The setup panel now carries
//! a single row pointing here, so there is no second copy to drift.

use crate::osd::{Overlay, Row};
use crate::sheet::Sheet;

/// The bindings, grouped as someone hunting for one would scan them: what
/// opens a page, then moving about, then the picture, then a film, then the
/// mouse.
///
/// Kept beside the dispatch in `app.rs` by nothing but care, which is why the
/// tests check the ones that would be missed.
pub const SECTIONS: &[(&str, &[(&str, &str)])] = &[
    ("PAGES", &[
        ("F1 OR ?", "SETUP AND STATUS"),
        ("F2", "MOSAIC GRID"),
        ("R", "INCLUDE SUBFOLDERS ON / OFF (PICTURES, GRID)"),
        ("S IN THE GRID", "SORT: FOLDER / NAME / DATE / FIRST OF EACH"),
        ("F8", "MODELS, RATE, SHARPNESS"),
        ("H", "THIS PAGE"),
        ("ENTER OR B", "DETAILED VIEW"),
        ("ALT", "SUBTITLE MENU (VIDEO)"),
    ]),
    ("MOVING BETWEEN FILES", &[
        ("N / P", "NEXT / PREVIOUS FILE"),
        ("CTRL TAB", "NEXT SHOW WINDOW  SHIFT: PREVIOUS"),
        ("CTRL C", "COPY THE IMAGE"),
        ("LEFT / RIGHT", "PREVIOUS / NEXT IMAGE"),
        ("PGUP / PGDN", "PREVIOUS / NEXT IMAGE"),
        ("HOME / END", "FIRST / LAST IMAGE"),
        ("BACKSPACE", "PARENT FOLDER"),
        ("DEL", "DELETE IMAGE"),
    ]),
    ("FINDING THINGS", &[
        ("/", "TYPE WHAT TO FIND"),
        ("ENTER", "SEARCH FOR IT"),
        ("ESC", "CLEAR WHAT WAS FOUND"),
        ("F7", "FIND OBJECTS ON / OFF"),
        ("F9", "DEPTH: OVER / ONLY / OFF"),
        ("CTRL CLICK", "CUT OUT WHAT IS THERE"),
    ]),
    ("THE PICTURE", &[
        ("0 / 1", "FIT / 100%"),
        ("Z", "100% / FIT / COVER WINDOW"),
        ("+ / -", "ZOOM IN / OUT"),
        ("W A S D", "PAN"),
        ("F OR F11", "FULLSCREEN"),
        ("I", "SMOOTHING"),
        ("[ / ]", "TURN LEFT / RIGHT (NOT SAVED)"),
        ("F3", "HOLD ZOOM BETWEEN IMAGES"),
        ("F4", "DEBUG INFO ON / OFF"),
        ("T", "TIMING READOUT"),
        ("F6", "GROUP ALIKE IMAGES"),
    ]),
    ("A FILM", &[
        ("SPACE", "PAUSE"),
        (", / .", "STEP ONE FRAME"),
        ("LEFT / RIGHT", "KEYFRAME  CTRL 10S  SHIFT 30S"),
        ("M", "MUTE"),
        ("R", "REPEAT THIS FILM"),
        ("WHEEL", "VOLUME"),
        ("V", "NEXT SUBTITLE TRACK"),
        ("CTRL < / >", "SUBTITLES SMALLER / BIGGER"),
    ]),
    ("THE MOUSE", &[
        ("WHEEL", "ZOOM AT POINTER"),
        ("CTRL SHIFT WHEEL", "GROW / SHRINK WINDOW"),
        ("DRAG", "MOVE THE WINDOW (FULLSCREEN: PAN)"),
        ("SHIFT DRAG", "PAN THE PICTURE OR FILM"),
        ("DOUBLE CLICK", "FIT TO WINDOW"),
        ("RIGHT CLICK", "AUTOSCALE"),
        ("CLICK ON VIDEO", "PAUSE / RESUME"),
        ("DRAG THE BAR", "SCRUB"),
        ("CLICK A THING", "FOLLOW IT (F7 ON)"),
        ("CTRL CLICK", "CUT OUT WHAT IS THERE"),
    ]),
    ("LEAVING", &[("ESC OR Q", "QUIT")]),
];

/// A page of bindings, and the presentation state every page here shares.
pub struct Keys {
    sheet: Sheet,
}

impl Keys {
    pub fn open() -> Keys {
        Keys { sheet: Sheet::filling() }
    }

    /// Where the page sits relative to centred, after dragging.
    pub fn offset(&self) -> (i32, i32) {
        self.sheet.offset()
    }

    /// Wheel: step the text size.
    pub fn zoom(&mut self, steps: f32, win: (f32, f32)) {
        self.sheet.zoom(steps, win);
    }

    pub fn pan(&mut self, dx: i32, dy: i32) {
        self.sheet.pan(dx, dy);
    }

    /// Arrows and page keys move the page rather than a highlight: there is
    /// nothing on it to choose, and a list longer than a short window is one
    /// you need to be able to reach the bottom of.
    pub fn scroll(&mut self, lines: f32) {
        self.sheet.pan(0, -(lines * 40.0) as i32);
    }

    pub fn ensure_overlay(&mut self, win: (f32, f32)) {
        let rows = self.rows();
        self.sheet.ensure(&rows, win);
    }

    /// The rasterised page, valid after [`Keys::ensure_overlay`].
    pub fn cached(&self) -> Option<&Overlay> {
        self.sheet.cached()
    }

    fn rows(&self) -> Vec<Row> {
        let mut rows = vec![
            Row::good("SHOW - KEYS", crate::version_short().to_uppercase()),
        ];
        for (heading, bindings) in SECTIONS {
            rows.push(Row::plain(""));
            rows.push(Row::dim(*heading, ""));
            for (keys, what) in *bindings {
                rows.push(Row::new(*keys, *what, crate::osd::TEXT_COLOUR));
            }
        }
        rows.push(Row::plain(""));
        rows.push(Row::dim("WHEEL RESIZES", "DRAG MOVES"));
        rows.push(Row::dim("UP/DOWN SCROLLS", "H OR ESC CLOSE"));
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Vec<&'static str> {
        SECTIONS.iter().flat_map(|(_, b)| b.iter().map(|(k, _)| *k)).collect()
    }

    #[test]
    fn the_page_lists_the_bindings_that_matter() {
        let labels = labels();
        for essential in ["ESC OR Q", "DEL", "F1 OR ?", "H", "M", "W A S D", "SPACE"] {
            assert!(labels.contains(&essential), "{essential} is missing from the key page");
        }
    }

    #[test]
    fn every_binding_says_what_it_does() {
        for (heading, bindings) in SECTIONS {
            assert!(!bindings.is_empty(), "{heading} is an empty section");
            for (keys, what) in *bindings {
                assert!(!keys.is_empty(), "a binding under {heading} has no key");
                assert!(!what.is_empty(), "{keys} under {heading} says nothing");
            }
        }
    }

    #[test]
    fn the_page_fits_a_short_window_and_can_still_be_scrolled() {
        // The list is long enough that a laptop screen has to shrink it, and
        // long enough that reaching the bottom by dragging has to work.
        let mut page = Keys::open();
        page.ensure_overlay((1920.0, 600.0));
        let before = page.offset();
        page.scroll(3.0);
        assert_ne!(page.offset(), before, "the page did not move");
    }
}
