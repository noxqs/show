//! The presentation state a full-screen page needs, without its contents.
//!
//! Three places in this program show a list of rows the user moves through: the
//! setup panel, the subtitle picker, and the plugins page. What they share is
//! not what they say but how they behave — a highlighted row, a text size that
//! shrinks to fit the window, a drag offset, and a rasterised overlay that must
//! be thrown away whenever any of those change.
//!
//! That shrink-to-fit loop is the reason this is extracted rather than copied.
//! It has a test of its own because it was got wrong once, and two copies of it
//! would be two places to get it wrong again.
//!
//! Deliberately holds no actions and no configuration. What a page *does* when
//! a row is chosen is the page's business; this only knows how to show it.

use crate::osd::{self, Overlay, Row};

pub struct Sheet {
    /// Index of the highlighted row, within whatever the page counts as its
    /// selectable set — not necessarily the row index, since headings and
    /// blank lines are not selectable.
    pub selected: usize,
    /// Text size chosen by the user, overriding the automatic fit.
    scale: Option<f32>,
    /// Drag offset from centre, so a page taller than the screen can still be
    /// read all the way down.
    offset: (i32, i32),
    overlay: Option<Overlay>,
    /// Window height the current overlay was laid out for.
    laid_out_for: Option<u32>,
    /// The size the automatic fit last settled on, for as long as the page
    /// stays open at the same window height.
    fitted: Option<f32>,
    /// The widest this page has needed at that size, so it does not narrow
    /// again when a row says less than it did.
    widest: Option<(f32, u32)>,
    /// Set when something changed that the caller should redraw for.
    pub dirty: bool,
    /// Fit by growing as well as shrinking: the largest text the window
    /// holds, for a page that is only ever read — the keys.
    fill: bool,
}

impl Default for Sheet {
    fn default() -> Self {
        Sheet::new()
    }
}

impl Sheet {
    pub fn new() -> Sheet {
        Sheet {
            selected: 0,
            scale: None,
            offset: (0, 0),
            overlay: None,
            laid_out_for: None,
            fitted: None,
            widest: None,
            dirty: true,
            fill: false,
        }
    }

    /// As [`Sheet::new`], for a page that should fill the window: its text as
    /// large as fits, rather than panel-sized and only ever made smaller.
    pub fn filling() -> Sheet {
        Sheet { fill: true, ..Sheet::new() }
    }

    /// Throw away the rasterised overlay. Called by every mutator, so a page
    /// that changes its rows cannot forget to.
    ///
    /// What it deliberately does *not* throw away is the window size the page
    /// was laid out for, nor the size and width settled on there. Those exist
    /// to tell a resize — where starting the fit over is right — from a change
    /// of rows, where it is not: forgetting them made every keypress look like
    /// a resize, and the page grew under the reader on the first one.
    pub fn invalidate(&mut self) {
        self.overlay = None;
        self.dirty = true;
    }

    /// Move the highlight, wrapping. `len` is the number of selectable rows.
    pub fn move_selection(&mut self, delta: isize, len: usize) {
        if len == 0 {
            return;
        }
        let len = len as isize;
        self.selected = (((self.selected as isize + delta) % len + len) % len) as usize;
        self.invalidate();
    }

    /// Step the text size. Panning is reset so a zoom cannot leave the page
    /// parked off-screen.
    pub fn zoom(&mut self, steps: f32, win: (f32, f32)) {
        let current = self.scale.unwrap_or_else(|| {
            // Picking up from the auto-fitted size means the first notch
            // adjusts what is on screen rather than jumping to a default.
            self.cached()
                .map(|o| o.height as f32 / (win.1 * 0.96))
                .filter(|r| *r > 1.0)
                .map(|r| (osd::SCALE_PANEL / r).max(osd::SCALE_MIN))
                .unwrap_or(osd::SCALE_PANEL)
        });
        let next =
            (current + steps.signum() * osd::SCALE_STEP).clamp(osd::SCALE_MIN, osd::SCALE_MAX);
        if (next - current).abs() < f32::EPSILON && self.scale.is_some() {
            return;
        }
        self.scale = Some(next);
        self.offset = (0, 0);
        self.invalidate();
    }

    pub fn pan(&mut self, dx: i32, dy: i32) {
        self.offset = (self.offset.0 + dx, self.offset.1 + dy);
        self.dirty = true;
    }

    pub fn offset(&self) -> (i32, i32) {
        self.offset
    }

    pub fn cached(&self) -> Option<&Overlay> {
        self.overlay.as_ref()
    }

    /// Never narrower than this page has already been, at this text size.
    fn floor(&self, scale: f32) -> u32 {
        match self.widest {
            Some((had, width)) if (had - scale).abs() < f32::EPSILON => width,
            _ => 0,
        }
    }

    /// Nor wider. Once a page has been laid out, its width is settled for as
    /// long as it stays open at this text size, and a row that says more than
    /// any row has said so far is elided into it rather than allowed to push
    /// the edges out. Remembering only a floor stopped the page narrowing and
    /// left it free to grow: activating `INSTALL` adds a status line carrying
    /// a full path, which is longer than anything else on the page, and the
    /// whole panel jumped wider on the keypress.
    fn ceiling(&self, scale: f32, window: u32) -> u32 {
        match self.floor(scale) {
            0 => window,
            settled => settled,
        }
    }

    /// Rasterise `rows` if the current overlay is missing or was laid out for a
    /// different window.
    ///
    /// With no explicit size, the largest that fits the height is chosen by
    /// stepping down rather than falling to the next whole size, so a page that
    /// only just overflows loses only a little.
    pub fn ensure(&mut self, rows: &[Row], win: (f32, f32)) {
        if self.overlay.is_some() && self.laid_out_for == Some(win.1 as u32) {
            return;
        }

        // The page is what is being read, so it may use the whole window width
        // rather than shortening anything that would have fitted.
        let wide = (win.0 * 0.96) as u32;
        // The size is settled once per window size and then left alone. A
        // page whose rows change while it is open — a status line appearing, a
        // row that reads shorter than it did — must not resize itself: pressing
        // a key and watching the whole page grow, or shrink, reads as a fault.
        // A page that later overflows by a row can be dragged; a page that
        // moves under the reader cannot be un-moved.
        let same_window = self.laid_out_for == Some(win.1 as u32);
        let overlay = match (self.scale, same_window.then_some(self.fitted).flatten()) {
            // An explicit choice is honoured even if it overflows: that is what
            // dragging is for.
            (Some(scale), _) => osd::render_menu_at_least(
                rows,
                scale,
                self.ceiling(scale, wide),
                self.floor(scale),
            ),
            // Already settled at this window size.
            (None, Some(scale)) => osd::render_menu_at_least(
                rows,
                scale,
                self.ceiling(scale, wide),
                self.floor(scale),
            ),
            // First time at this size: step down until it fits.
            (None, None) if !self.fill => {
                let limit = win.1 * 0.96;
                let mut scale = osd::SCALE_PANEL;
                let mut best = osd::render_menu(rows, scale, wide);
                while (best.height as f32) > limit && scale > osd::SCALE_MIN {
                    scale = (scale - osd::SCALE_STEP).max(osd::SCALE_MIN);
                    best = osd::render_menu(rows, scale, wide);
                }
                self.fitted = Some(scale);
                self.widest = None;
                best
            }
            // Filling: as large as the window holds, with nothing shortened.
            // Text size is linear in the scale, so one layout at the smallest
            // size says roughly where the largest is; it is then checked, and
            // stepped down only if rounding made it a little too big.
            (None, None) => {
                let limit = win.1 * 0.96;
                let natural = |scale: f32| osd::render_menu(rows, scale, u32::MAX);
                let small = natural(osd::SCALE_MIN);
                let estimate = osd::SCALE_MIN
                    * (wide as f32 / small.width.max(1) as f32).min(limit / small.height.max(1) as f32);
                let mut scale = ((estimate / osd::SCALE_STEP).floor() * osd::SCALE_STEP)
                    .clamp(osd::SCALE_MIN, osd::SCALE_MAX);
                let mut best = natural(scale);
                while (best.width > wide || best.height as f32 > limit) && scale > osd::SCALE_MIN {
                    scale = (scale - osd::SCALE_STEP).max(osd::SCALE_MIN);
                    best = natural(scale);
                }
                self.fitted = Some(scale);
                self.widest = None;
                best
            }
        };

        let scale = self.scale.or(self.fitted).unwrap_or(osd::SCALE_PANEL);
        self.widest = match self.widest {
            Some((had, width)) if same_window && (had - scale).abs() < f32::EPSILON => {
                Some((had, width.max(overlay.width)))
            }
            _ => Some((scale, overlay.width)),
        };
        self.overlay = Some(overlay);
        self.laid_out_for = Some(win.1 as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize) -> Vec<Row> {
        (0..n).map(|i| Row::plain(format!("ROW {i}"))).collect()
    }

    #[test]
    fn a_page_does_not_narrow_when_a_row_says_less() {
        // Moving the highlight down the plugins page changes the footer, and a
        // menu that changes width as you move through it looks broken.
        let win = (1920.0, 1080.0);
        let mut sheet = Sheet::new();
        let long = vec![
            Row::dim("CUT OUT WHAT I CLICK", "NOT INSTALLED    42.6 MB"),
            Row::plain("ENTER INSTALL / DEL REMOVE / V VERIFY"),
        ];
        sheet.ensure(&long, win);
        let wide = sheet.cached().expect("laid out").width;

        sheet.invalidate();
        sheet.ensure(&[Row::dim("ONNX RUNTIME", "READY")], win);
        assert_eq!(sheet.cached().expect("laid out").width, wide, "the page narrowed");
    }

    #[test]
    fn a_page_does_not_widen_when_a_row_says_more() {
        // The other half of the same rule, and the one that was still broken:
        // pressing enter on INSTALL adds a status line carrying a full path,
        // which is longer than any row on the page. Remembering only a floor
        // stopped the page narrowing and left it free to grow, so the panel
        // jumped wider on the keypress.
        let win = (1920.0, 1080.0);
        let mut sheet = Sheet::new();
        sheet.ensure(&[Row::good("SHOW - SETUP", "1.1.0"), Row::dim("INSTALL", "NOT INSTALLED")], win);
        let wide = sheet.cached().expect("laid out").width;

        sheet.invalidate();
        sheet.ensure(
            &[
                Row::good("SHOW - SETUP", "1.1.0"),
                Row::dim("INSTALL", "NOT INSTALLED"),
                Row::dim("INSTALLED /HOME/SOMEBODY/.LOCAL/BIN/SHOW (NOT ON PATH)", ""),
            ],
            win,
        );
        assert_eq!(sheet.cached().expect("laid out").width, wide, "the page widened");
    }

    #[test]
    fn a_settled_width_survives_being_laid_out_again() {
        // The width is fed back in as the ceiling for every later layout, so
        // the arithmetic that turns it into a character budget has to round
        // trip. One character short and the widest row would pick up an
        // ellipsis it did not have, on a page that had not changed.
        let win = (1920.0, 1080.0);
        let page = [Row::dim("CUT OUT WHAT I CLICK", "NOT INSTALLED    42.6 MB")];
        let mut sheet = Sheet::new();
        sheet.ensure(&page, win);
        let first = sheet.cached().expect("laid out").rgba.clone();

        sheet.invalidate();
        sheet.ensure(&page, win);
        assert_eq!(sheet.cached().expect("laid out").rgba, first, "the same page rendered differently");
    }

    #[test]
    fn activating_a_row_does_not_rescale_the_page() {
        // The flow that was actually broken: lay out, then change the rows
        // through `invalidate` exactly as pressing enter does. The earlier
        // version of this test restored `laid_out_for` by hand, which is the
        // very thing the code was failing to keep — so it passed while the
        // menu still grew on the first keypress.
        let win = (1920.0, 600.0);
        let mut sheet = Sheet::new();
        sheet.ensure(&rows(40), win);
        let scale = sheet.fitted.expect("an automatic fit was chosen");
        let width = sheet.cached().expect("laid out").width;

        sheet.invalidate();
        sheet.ensure(&rows(41), win); // a status line appeared
        assert_eq!(sheet.fitted, Some(scale), "the page shrank when a row was added");

        sheet.invalidate();
        sheet.ensure(&rows(8), win); // and now the page says much less
        assert_eq!(sheet.fitted, Some(scale), "the page grew when it got shorter");
        assert_eq!(sheet.cached().expect("laid out").width, width, "and it changed width");
    }

    #[test]
    fn a_page_does_not_resize_itself_when_its_rows_change() {
        // Pressing enter on a row adds a status line, and a row can come back
        // shorter than it went in. Neither is a reason to rescale the page
        // under the reader.
        let win = (1920.0, 600.0);
        let mut sheet = Sheet::new();
        sheet.ensure(&rows(40), win);
        let small = sheet.cached().expect("laid out").height;
        let scale = sheet.fitted.expect("an automatic fit was chosen");

        // Now the page says less, which at the old rule would have let the
        // text grow.
        sheet.invalidate();
        sheet.ensure(&rows(6), win);
        assert_eq!(sheet.fitted, Some(scale), "the page rescaled itself");
        assert!(sheet.cached().expect("laid out").height < small, "it should be shorter");
    }

    #[test]
    fn a_new_window_size_starts_the_fit_over() {
        // Settling the size is per window size, not for ever: a smaller window
        // has to be fitted again, and that is the one time the page may resize
        // under the reader — because the window it is in just did.
        let mut sheet = Sheet::new();
        sheet.ensure(&rows(8), (1920.0, 1080.0));
        let roomy = sheet.fitted.expect("fitted");

        sheet.ensure(&rows(80), (1920.0, 400.0));
        assert!(sheet.fitted.expect("fitted") < roomy, "a short window must shrink the page");
    }

    #[test]
    fn the_selection_wraps_at_both_ends() {
        let mut sheet = Sheet::new();
        sheet.move_selection(-1, 3);
        assert_eq!(sheet.selected, 2, "moving up from the first wraps to the last");
        sheet.move_selection(1, 3);
        assert_eq!(sheet.selected, 0);
    }

    #[test]
    fn an_empty_page_does_not_panic() {
        let mut sheet = Sheet::new();
        sheet.move_selection(1, 0);
        assert_eq!(sheet.selected, 0);
    }

    #[test]
    fn a_page_shrinks_to_fit_the_screen() {
        // More rows than a short window holds at full size, but few enough
        // that shrinking can rescue them.
        let mut sheet = Sheet::new();
        let win = (1920.0, 400.0);
        sheet.ensure(&rows(16), win);
        let height = sheet.cached().expect("rasterised").height as f32;
        assert!(height <= win.1 * 0.96 + 1.0, "did not shrink to fit: {height}");

        // The same rows in a tall window are not shrunk at all.
        let mut roomy = Sheet::new();
        roomy.ensure(&rows(16), (1920.0, 1080.0));
        assert!(
            roomy.cached().unwrap().height > sheet.cached().unwrap().height,
            "shrank when there was room not to"
        );
    }

    #[test]
    fn shrinking_stops_at_the_smallest_size_rather_than_forever() {
        // Far too many rows for the window even at the minimum size. The page
        // overflows, which is what dragging is for; the loop must still end.
        let mut sheet = Sheet::new();
        sheet.ensure(&rows(80), (1920.0, 200.0));
        assert!(sheet.cached().expect("rasterised").height > 200);
    }

    #[test]
    fn an_explicit_size_is_honoured_even_when_it_overflows() {
        let mut sheet = Sheet::new();
        let win = (1920.0, 400.0);
        sheet.zoom(1.0, win);
        sheet.zoom(1.0, win);
        sheet.ensure(&rows(40), win);
        // Dragging is what a page too large to fit is for.
        assert!(sheet.cached().is_some());
    }

    #[test]
    fn changing_the_selection_throws_the_overlay_away() {
        let mut sheet = Sheet::new();
        sheet.ensure(&rows(4), (1920.0, 1080.0));
        assert!(sheet.cached().is_some());
        sheet.move_selection(1, 4);
        assert!(sheet.cached().is_none(), "a stale overlay would show the old highlight");
    }

    #[test]
    fn a_resize_relays_out_but_a_redraw_does_not() {
        let mut sheet = Sheet::new();
        sheet.ensure(&rows(4), (1920.0, 1080.0));
        let first = sheet.cached().expect("rasterised").id;
        sheet.ensure(&rows(4), (1920.0, 1080.0));
        assert_eq!(sheet.cached().unwrap().id, first, "laid out twice for one size");
        sheet.ensure(&rows(4), (1920.0, 720.0));
        assert_ne!(sheet.cached().unwrap().id, first, "a new height needs a new layout");
    }

    #[test]
    fn panning_moves_without_relaying_out() {
        let mut sheet = Sheet::new();
        sheet.ensure(&rows(4), (1920.0, 1080.0));
        let id = sheet.cached().expect("rasterised").id;
        sheet.pan(10, -20);
        assert_eq!(sheet.offset(), (10, -20));
        assert_eq!(sheet.cached().unwrap().id, id, "panning must not re-rasterise");
    }
}
