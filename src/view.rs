//! Zoom and pan state.
//!
//! The canonical state is "which image-space point sits at the centre of the
//! window, and how many screen pixels does one image pixel cover". Everything
//! the renderer needs falls out of that, and it stays stable across window
//! resizes and fullscreen toggles.

pub const MIN_ZOOM: f32 = 0.01;
pub const MAX_ZOOM: f32 = 64.0;

#[derive(Clone, Copy, Debug)]
pub struct View {
    /// Screen pixels per image pixel. 1.0 is 100%.
    pub zoom: f32,
    /// Image-space point pinned to the centre of the window.
    pub center: [f32; 2],
    /// While set, the view re-fits itself on every resize.
    pub fitting: bool,
    /// Grow images to fill the window rather than stopping at 1:1.
    ///
    /// Off, a small picture stays its own size instead of being blown up into
    /// a blurry wall. On, everything is scaled to the window. Either way the
    /// aspect ratio is preserved and the whole image stays visible.
    pub autoscale: bool,
    /// Fit so the image covers the whole window rather than showing all of
    /// it: the larger of the two scales, cropping the axis that overflows
    /// instead of leaving bars beside the other. Always enlarges if it has to
    /// — a cover that left a border would not be one.
    pub cover: bool,
    /// Quarter turns clockwise the picture is shown at. Only how it is
    /// shown: the file is never touched.
    ///
    /// Everything else here — `center`, the rects, the clamps — is in the
    /// turned picture's own coordinates, so fitting a portrait photograph
    /// turned on its side is just fitting a landscape one. The way back into
    /// the file's coordinates is [`View::image_point_in`].
    pub turn: u8,
}

impl Default for View {
    fn default() -> Self {
        View {
            zoom: 1.0,
            center: [0.0, 0.0],
            fitting: true,
            autoscale: false,
            cover: false,
            turn: 0,
        }
    }
}

/// Where a point of the picture lands once turned `turn` quarter turns
/// clockwise, for a picture `img` in size before turning.
pub fn turn_point(turn: u8, p: [f32; 2], img: (f32, f32)) -> [f32; 2] {
    match turn % 4 {
        0 => p,
        1 => [img.1 - p[1], p[0]],
        2 => [img.0 - p[0], img.1 - p[1]],
        _ => [p[1], img.0 - p[0]],
    }
}

/// The inverse of [`turn_point`]: which point of the picture is shown here.
pub fn unturn_point(turn: u8, d: [f32; 2], img: (f32, f32)) -> [f32; 2] {
    match turn % 4 {
        0 => d,
        1 => [d[1], img.1 - d[0]],
        2 => [img.0 - d[0], img.1 - d[1]],
        _ => [img.0 - d[1], d[0]],
    }
}

impl View {
    /// The picture's size as shown: on its side, a quarter turn swaps the axes.
    pub fn oriented(&self, img: (f32, f32)) -> (f32, f32) {
        if self.turn % 2 == 1 { (img.1, img.0) } else { img }
    }

    /// Turn by `quarters` (clockwise positive), keeping the same part of the
    /// picture in the middle of the window.
    pub fn rotate(&mut self, quarters: i32, img: (f32, f32), win: (f32, f32)) {
        let middle = unturn_point(self.turn, self.center, img);
        self.turn = (self.turn as i32 + quarters).rem_euclid(4) as u8;
        self.center = turn_point(self.turn, middle, img);
        let shown = self.oriented(img);
        self.clamp_center(shown, win);
    }

    /// Zoom at which the whole image is visible. Capped at 100% unless
    /// autoscaling: blowing a 200px icon up to fill a 4K screen looks far
    /// worse than showing it at its native size.
    pub fn fit_zoom(&self, img: (f32, f32), win: (f32, f32)) -> f32 {
        let img = self.oriented(img);
        if img.0 <= 0.0 || img.1 <= 0.0 {
            return 1.0;
        }
        // The smaller of the two scales, so the whole image stays visible and
        // neither axis is scaled differently from the other.
        let (x, y) = (win.0 / img.0, win.1 / img.1);
        if self.cover {
            return x.max(y);
        }
        let z = x.min(y);
        if self.autoscale { z } else { z.min(1.0) }
    }

    /// Fit, enlarging if the window is bigger than the picture.
    ///
    /// For video, where filling the window is the whole point. The autoscale
    /// setting exists so a small photograph is not blown up to fill a screen,
    /// which is a sensible thing to ask about a photograph and not a question
    /// anyone asks about a film — fullscreen that left a 1080p film sitting
    /// 1:1 in the middle of a larger screen would be fullscreen in name only.
    pub fn fit_filling(&mut self, img: (f32, f32), win: (f32, f32)) {
        let img = self.oriented(img);
        let zoom = if img.0 <= 0.0 || img.1 <= 0.0 {
            1.0
        } else if self.cover {
            (win.0 / img.0).max(win.1 / img.1)
        } else {
            (win.0 / img.0).min(win.1 / img.1)
        };
        self.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.center = [img.0 / 2.0, img.1 / 2.0];
        self.fitting = true;
    }

    pub fn fit(&mut self, img: (f32, f32), win: (f32, f32)) {
        self.zoom = self.fit_zoom(img, win).clamp(MIN_ZOOM, MAX_ZOOM);
        let img = self.oriented(img);
        self.center = [img.0 / 2.0, img.1 / 2.0];
        self.fitting = true;
    }

    /// Snap to a fixed zoom, keeping the current centre.
    pub fn set_zoom(&mut self, zoom: f32, img: (f32, f32), win: (f32, f32)) {
        self.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.fitting = false;
        self.clamp_center(self.oriented(img), win);
    }

    /// Zoom about a cursor position (physical pixels, top-left origin), so the
    /// image point under the pointer stays put. This is the whole trick to
    /// wheel zoom feeling right.
    pub fn zoom_about(&mut self, factor: f32, cursor: (f32, f32), img: (f32, f32), win: (f32, f32)) {
        // The point under the cursor, which is what has to stay put. Not
        // `image_point_in`, because zooming past the edge of the picture is
        // ordinary and must not be refused the way a click on nothing is.
        let offset = (cursor.0 - win.0 / 2.0, cursor.1 - win.1 / 2.0);
        let anchor = [
            offset.0 / self.zoom + self.center[0],
            offset.1 / self.zoom + self.center[1],
        ];

        let new_zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        if new_zoom == self.zoom {
            return;
        }
        self.zoom = new_zoom;
        self.center = [anchor[0] - offset.0 / new_zoom, anchor[1] - offset.1 / new_zoom];
        self.fitting = false;
        self.clamp_center(self.oriented(img), win);
    }

    /// Drag the image by a screen-space delta.
    ///
    /// Leaves `fitting` alone: only a zoom somebody chose ends fitting. A pan
    /// is looking around the picture at the size it was fitted to, and a stray
    /// drag across a picture that fits entirely — which moves nothing — used
    /// to leave the next picture unfitted.
    pub fn pan(&mut self, dx: f32, dy: f32, img: (f32, f32), win: (f32, f32)) {
        self.center[0] -= dx / self.zoom;
        self.center[1] -= dy / self.zoom;
        self.clamp_center(self.oriented(img), win);
    }

    /// Keep the image from being flung off-screen: axes larger than the window
    /// are bounded to their edges, axes smaller than it stay centred. `img`
    /// is the size as shown, already turned.
    fn clamp_center(&mut self, img: (f32, f32), win: (f32, f32)) {
        for (axis, (i, w)) in [(img.0, win.0), (img.1, win.1)].into_iter().enumerate() {
            let visible = w / self.zoom;
            self.center[axis] = if i <= visible {
                i / 2.0
            } else {
                self.center[axis].clamp(visible / 2.0, i - visible / 2.0)
            };
        }
    }

    /// The image quad in normalised device coordinates, filling the window.
    #[cfg(test)]
    pub fn ndc_rect(&self, img: (f32, f32), win: (f32, f32)) -> [f32; 4] {
        self.ndc_rect_in(img, (0.0, 0.0, win.0, win.1), win)
    }

    /// The image quad in NDC, confined to a viewport within the window: the
    /// picture as shown, turned, so on its side it is tall where the file is
    /// wide. The shaders turn the texture to match.
    ///
    /// The viewport is what the detailed view needs: the picture occupies the
    /// left of the screen while a file list holds the right, and every piece of
    /// view arithmetic — fit, zoom, pan — should behave as though the viewport
    /// were the whole window.
    pub fn ndc_rect_in(
        &self,
        img: (f32, f32),
        viewport: (f32, f32, f32, f32),
        win: (f32, f32),
    ) -> [f32; 4] {
        let (vx, vy, vw, vh) = viewport;
        let img = self.oriented(img);

        // Position within the viewport, then offset into the window.
        let sx0 = vx + (0.0 - self.center[0]) * self.zoom + vw / 2.0;
        let sy0 = vy + (0.0 - self.center[1]) * self.zoom + vh / 2.0;
        let sx1 = vx + (img.0 - self.center[0]) * self.zoom + vw / 2.0;
        let sy1 = vy + (img.1 - self.center[1]) * self.zoom + vh / 2.0;

        let ndc_x = |x: f32| 2.0 * x / win.0 - 1.0;
        let ndc_y = |y: f32| 1.0 - 2.0 * y / win.1;

        // Screen y grows downwards, NDC y grows upwards, so the image's bottom
        // edge (sy1) is the NDC minimum.
        [ndc_x(sx0), ndc_y(sy1), ndc_x(sx1), ndc_y(sy0)]
    }


    /// Where in the picture a point on screen falls.
    ///
    /// The exact inverse of `ndc_rect_in`, and it has to stay that way: this
    /// is how a click finds the thing it landed on, and being a few pixels out
    /// picks the wrong one. `None` when the point is outside the picture —
    /// clicking the letterbox is clicking nothing. In the file's own
    /// coordinates, however the picture is turned.
    pub fn image_point_in(
        &self,
        at: (f32, f32),
        img: (f32, f32),
        viewport: (f32, f32, f32, f32),
    ) -> Option<[f32; 2]> {
        let (vx, vy, vw, vh) = viewport;
        let point = [
            (at.0 - vx - vw / 2.0) / self.zoom + self.center[0],
            (at.1 - vy - vh / 2.0) / self.zoom + self.center[1],
        ];
        let shown = self.oriented(img);
        let inside = point[0] >= 0.0 && point[1] >= 0.0 && point[0] < shown.0 && point[1] < shown.1;
        inside.then(|| unturn_point(self.turn, point, img))
    }

    /// Carry the current zoom onto a differently sized image.
    ///
    /// The scale is deliberately kept — that is the point of holding zoom — but
    /// the centre has to be pulled back inside the new image, or a wide picture
    /// followed by a narrow one would open somewhere off its own edge.
    pub fn retarget(&mut self, img: (f32, f32), win: (f32, f32)) {
        let img = self.oriented(img);
        self.center = [
            self.center[0].min(img.0).max(0.0),
            self.center[1].min(img.1).max(0.0),
        ];
        self.fitting = false;
        self.clamp_center(img, win);
    }

    /// Turn autoscaling on or off, re-fitting to show the difference at once.
    pub fn set_autoscale(&mut self, on: bool, img: (f32, f32), win: (f32, f32)) {
        self.autoscale = on;
        self.fit(img, win);
    }

    /// The zoom as a percentage, for display.
    pub fn percent(&self) -> f32 {
        self.zoom * 100.0
    }

    /// True once one image pixel covers more than one screen pixel, which is
    /// when nearest-neighbour sampling starts being what you actually want.
    pub fn magnifying(&self) -> bool {
        self.zoom > 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: (f32, f32) = (1000.0, 500.0);
    const WIN: (f32, f32) = (800.0, 600.0);

    #[test]
    fn fit_shrinks_but_does_not_enlarge() {
        let mut v = View::default();
        v.fit(IMG, WIN);
        assert!((v.zoom - 0.8).abs() < 1e-6, "zoom {}", v.zoom);
        // Small image: capped at 100%.
        v.fit((100.0, 100.0), WIN);
        assert_eq!(v.zoom, 1.0);

        v.autoscale = true;
        v.fit((100.0, 100.0), WIN);
        assert_eq!(v.zoom, 6.0);
    }

    #[test]
    fn only_zooming_ends_fitting() {
        let (img, win) = ((400.0, 300.0), (1600.0, 900.0));
        let mut v = View::default();
        v.fit(img, win);
        // A drag across a picture that fits: nothing moves, and it stays fitted.
        v.pan(40.0, -25.0, img, win);
        assert!(v.fitting, "a pan is not a choice of zoom");
        v.zoom_about(1.5, (800.0, 450.0), img, win);
        assert!(!v.fitting, "a zoom is");
    }

    #[test]
    fn cover_fills_the_window_on_both_axes() {
        // Tall picture, wide window: cover scales to the width and crops the
        // height, where fit scales to the height and leaves bars.
        let (img, win) = ((1000.0, 2000.0), (1600.0, 900.0));
        let mut v = View { cover: true, ..View::default() };
        v.fit(img, win);
        assert!((v.zoom - 1.6).abs() < 1e-6, "cover: {}", v.zoom);
        assert!(img.0 * v.zoom >= win.0 && img.1 * v.zoom >= win.1);

        v.cover = false;
        v.fit(img, win);
        assert!((v.zoom - 0.45).abs() < 1e-6, "fit: {}", v.zoom);
    }

    #[test]
    fn autoscale_decides_whether_a_small_image_grows() {
        let img = (100.0, 100.0);
        let win = (800.0, 600.0);

        let mut v = View::default();
        v.fit(img, win);
        assert_eq!(v.zoom, 1.0, "without autoscale a small image keeps its size");

        v.set_autoscale(true, img, win);
        // Limited by the shorter axis, so the whole image stays visible.
        assert!((v.zoom - 6.0).abs() < 1e-6, "autoscale grows to the window: {}", v.zoom);

        v.set_autoscale(false, img, win);
        assert_eq!(v.zoom, 1.0, "and turning it off puts it back");
    }

    #[test]
    fn a_large_image_shrinks_either_way() {
        let img = (4000.0, 3000.0);
        let win = (800.0, 600.0);
        for autoscale in [false, true] {
            let mut v = View { autoscale, ..View::default() };
            v.fit(img, win);
            assert!((v.zoom - 0.2).abs() < 1e-6, "autoscale={autoscale}: {}", v.zoom);
        }
    }

    #[test]
    fn retargeting_keeps_the_zoom_and_rescues_the_centre() {
        let mut v = View { zoom: 3.0, center: [900.0, 400.0], ..View::default() };
        // Moving to a much smaller image: the old centre is outside it.
        v.retarget((100.0, 80.0), WIN);
        assert_eq!(v.zoom, 3.0, "holding zoom must not change the scale");
        assert!(v.center[0] <= 100.0 && v.center[1] <= 80.0, "{:?}", v.center);
        assert!(v.center[0] >= 0.0 && v.center[1] >= 0.0, "{:?}", v.center);
    }

    #[test]
    fn an_unscaled_image_reports_one_hundred_percent() {
        let mut v = View::default();
        // 1920x1080 inside a larger window: shrink-only fit leaves it at 1:1.
        v.fit((1920.0, 1080.0), (3072.0, 1728.0));
        assert!((v.percent() - 100.0).abs() < 0.01, "{}", v.percent());
    }

    #[test]
    fn zoom_about_keeps_the_point_under_the_cursor() {
        let mut v = View {
            zoom: 1.0,
            center: [500.0, 250.0],
            fitting: false,
            autoscale: true,
            cover: false,
            turn: 0,
        };
        let cursor = (600.0, 200.0);

        let whole = (0.0, 0.0, WIN.0, WIN.1);
        let before = v.image_point_in(cursor, IMG, whole).expect("cursor is over the image");
        v.zoom_about(2.5, cursor, IMG, WIN);
        let after = v.image_point_in(cursor, IMG, whole).expect("cursor is over the image");

        assert!((before[0] - after[0]).abs() < 0.01, "{before:?} vs {after:?}");
        assert!((before[1] - after[1]).abs() < 0.01, "{before:?} vs {after:?}");
    }

    #[test]
    fn a_click_lands_where_the_quad_says_it_should() {
        // The check that matters is the round trip: put a point on screen with
        // `ndc_rect_in`, read it back with `image_point_in`, land in the same
        // place. Anything else is two derivations of the same arithmetic
        // agreeing with each other rather than with what is drawn.
        let v = View { zoom: 1.7, center: [420.0, 260.0], fitting: false, autoscale: false, cover: false, turn: 0 };
        let img = (1600.0, 900.0);
        let win = (1920.0, 1080.0);
        let viewport = (0.0, 48.0, 1920.0, 1032.0);

        let r = v.ndc_rect_in(img, viewport, win);
        // The quad's top-left corner in window pixels, back out of NDC.
        let left = (r[0] + 1.0) / 2.0 * win.0;
        let top = (1.0 - r[3]) / 2.0 * win.1;

        let corner = v.image_point_in((left + 0.5, top + 0.5), img, viewport).expect("on it");
        assert!(corner[0].abs() < 0.6 && corner[1].abs() < 0.6, "{corner:?}");

        // And a point a known distance in is that many image pixels in.
        let inner = v
            .image_point_in((left + 170.0, top + 85.0), img, viewport)
            .expect("on it");
        assert!((inner[0] - 100.0).abs() < 0.01, "{inner:?}");
        assert!((inner[1] - 50.0).abs() < 0.01, "{inner:?}");
    }

    #[test]
    fn clicking_the_letterbox_is_clicking_nothing() {
        let v = View { zoom: 1.0, center: [400.0, 300.0], fitting: true, autoscale: false, cover: false, turn: 0 };
        let img = (800.0, 600.0);
        let win = (1600.0, 600.0);
        let whole = (0.0, 0.0, win.0, win.1);
        // The image is centred, so the far left of the window is beside it.
        assert!(v.image_point_in((10.0, 300.0), img, whole).is_none());
        assert!(v.image_point_in((800.0, 300.0), img, whole).is_some());
    }

    #[test]
    fn a_viewport_confines_the_image_to_part_of_the_window() {
        // An 800x600 image at 1:1 inside the left half of a 1600x600 window.
        let v = View { zoom: 1.0, center: [400.0, 300.0], fitting: true, autoscale: false, cover: false, turn: 0 };
        let win = (1600.0, 600.0);
        let r = v.ndc_rect_in((800.0, 600.0), (0.0, 0.0, 800.0, 600.0), win);

        // Left edge at the window edge, right edge at the middle.
        assert!((r[0] + 1.0).abs() < 1e-5, "{r:?}");
        assert!(r[2].abs() < 1e-5, "should stop at the centre of the window: {r:?}");
        // Full height either way.
        assert!((r[1] + 1.0).abs() < 1e-5 && (r[3] - 1.0).abs() < 1e-5, "{r:?}");
    }

    #[test]
    fn a_full_window_viewport_matches_the_plain_rect() {
        let v = View { zoom: 0.5, center: [300.0, 200.0], fitting: false, autoscale: false, cover: false, turn: 0 };
        let win = (1024.0, 768.0);
        let plain = v.ndc_rect(IMG, win);
        let viewport = v.ndc_rect_in(IMG, (0.0, 0.0, win.0, win.1), win);
        for (a, b) in plain.iter().zip(viewport.iter()) {
            assert!((a - b).abs() < 1e-6, "{plain:?} vs {viewport:?}");
        }
    }

    #[test]
    fn ndc_rect_fills_the_screen_when_fitted_exactly() {
        // A 800x600 image in a 800x600 window at 100% covers the full viewport.
        let v = View {
            zoom: 1.0,
            center: [400.0, 300.0],
            fitting: true,
            autoscale: false,
            cover: false,
            turn: 0,
        };
        let r = v.ndc_rect((800.0, 600.0), WIN);
        for (got, want) in r.iter().zip([-1.0, -1.0, 1.0, 1.0]) {
            assert!((got - want).abs() < 1e-5, "{r:?}");
        }
    }

    #[test]
    fn small_images_stay_centred_and_large_ones_stay_on_screen() {
        let mut v = View {
            zoom: 1.0,
            center: [0.0, 0.0],
            fitting: false,
            autoscale: false,
            cover: false,
            turn: 0,
        };
        // Image narrower than the window: centre is forced to the middle.
        v.pan(9999.0, 9999.0, (100.0, 100.0), WIN);
        assert_eq!(v.center, [50.0, 50.0]);

        // Image wider than the window: centre is bounded by the edges.
        v.center = [0.0, 0.0];
        v.pan(-9999.0, 0.0, (4000.0, 4000.0), WIN);
        assert!(v.center[0] <= 4000.0 - WIN.0 / 2.0 + 0.01, "{:?}", v.center);
    }

    #[test]
    fn turning_and_unturning_are_inverses() {
        let img = (1000.0, 500.0);
        for turn in 0..4 {
            for p in [[0.0, 0.0], [123.0, 45.0], [1000.0, 500.0], [999.0, 1.0]] {
                let back = unturn_point(turn, turn_point(turn, p, img), img);
                assert!((back[0] - p[0]).abs() < 1e-4 && (back[1] - p[1]).abs() < 1e-4, "{turn}: {p:?} -> {back:?}");
            }
        }
        // A quarter right puts the top-left corner at the top-right.
        assert_eq!(turn_point(1, [0.0, 0.0], img), [500.0, 0.0]);
    }

    #[test]
    fn a_picture_on_its_side_fits_as_a_tall_one() {
        let (img, win) = ((2000.0, 1000.0), (800.0, 800.0));
        let mut v = View::default();
        v.rotate(1, img, win);
        v.fit(img, win);
        // Now 1000 wide and 2000 tall: the height decides.
        assert!((v.zoom - 0.4).abs() < 1e-6, "{}", v.zoom);
        let r = v.ndc_rect(img, win);
        assert!((r[3] - r[1]) > (r[2] - r[0]), "should stand upright: {r:?}");
    }

    #[test]
    fn a_click_on_a_turned_picture_finds_the_point_in_the_file() {
        // 400x200 turned right, at 1:1 in a window it fits: on screen it is
        // 200 wide and 400 tall, centred.
        let (img, win) = ((400.0, 200.0), (600.0, 600.0));
        let mut v = View::default();
        v.rotate(1, img, win);
        v.fit(img, win);
        let whole = (0.0, 0.0, win.0, win.1);
        // The screen's top-right of the picture is the file's top-left.
        let p = v.image_point_in((399.5, 100.5), img, whole).expect("on it");
        assert!(p[0] < 1.0 && p[1] < 1.0, "{p:?}");
        // The screen's bottom-left is the file's bottom-right.
        let p = v.image_point_in((200.5, 499.5), img, whole).expect("on it");
        assert!(p[0] > 399.0 && p[1] > 199.0, "{p:?}");
    }

    #[test]
    fn turning_keeps_the_middle_of_a_zoomed_picture() {
        let (img, win) = ((4000.0, 3000.0), (800.0, 600.0));
        let mut v = View { zoom: 2.0, center: [1000.0, 700.0], fitting: false, ..View::default() };
        let whole = (0.0, 0.0, win.0, win.1);
        let before = v.image_point_in((400.0, 300.0), img, whole).unwrap();
        v.rotate(1, img, win);
        let after = v.image_point_in((400.0, 300.0), img, whole).unwrap();
        assert!((before[0] - after[0]).abs() < 0.01 && (before[1] - after[1]).abs() < 0.01, "{before:?} {after:?}");
        v.rotate(-1, img, win);
        assert_eq!(v.turn, 0);
        v.rotate(-1, img, win);
        assert_eq!(v.turn, 3);
    }
}
