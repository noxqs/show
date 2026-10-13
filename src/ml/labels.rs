//! Turning what was found into a grid of instance ids.
//!
//! The grid is what the GPU draws: one byte per cell, zero for nothing, and
//! otherwise a small number the shader looks up a colour for. Coarse on
//! purpose — a grid this wide is enough to place a box or a mask over a
//! picture, at a fraction of the memory of doing it at screen resolution.
//!
//! Ids are palette slots, and the slot is the *object*, not its kind: two
//! people are two colours, because the question being asked of a mask is
//! usually "which one is that" rather than "what is that". The slot comes from
//! the tracker's id rather than from position in the list, so it stays with a
//! thing as it moves and does not flicker when the model reorders its answers
//! between frames. The bar naming them is drawn from the same slots, so it
//! remains the legend.
//!
//! Fifteen slots, wrapping: a sixteenth object on screen shares a colour with
//! the first. The alternative is colours nobody can tell apart, which is not a
//! legend either.

use super::vision::Found;

/// How wide the grid is. Height follows the picture's shape.
///
/// 960 rather than 480: an outline is only as fine as the grid it is written
/// into, and half a megabyte a frame is nothing beside the picture it is drawn
/// over. It is still far below screen resolution, deliberately — the mask
/// being sampled is coarser than this, and a grid finer than the answer only
/// costs upload.
pub const WIDTH: u32 = 960;

/// How many colours a mask can be drawn in.
///
/// The shader's palette holds sixteen and slot zero means nothing-here, so
/// fifteen are left for things.
pub const SLOTS: u32 = 15;

/// The slot the cut-out is drawn in.
///
/// Reserved rather than derived: a cut-out has no tracker id — it is whatever
/// was last pointed at — and it appears *beside* the tracked things rather
/// than instead of them, so it needs a colour of its own that the bar can
/// name. Without this it took slot one and clashed with a tracked object.
pub const CUT_OUT: u8 = SLOTS as u8;

/// The palette slot an object is drawn in, from its tracker id.
///
/// Never zero, which the shader reads as background — an object given slot
/// zero would simply not be drawn.
pub fn slot(id: u32) -> u8 {
    if id == 0 {
        return CUT_OUT;
    }
    (id % (SLOTS - 1) + 1) as u8
}

/// The colour a slot is drawn in.
///
/// The one place that mapping is written down, so the bar and the mask cannot
/// disagree about which colour a thing is — which is the whole basis of the
/// bar being a legend rather than a list.
pub fn colour_of(slot: u8) -> [u8; 4] {
    crate::ml::colour(slot.max(1) as usize - 1)
}

/// The grid, and the size it was drawn at.
pub struct Grid {
    pub cells: Vec<u8>,
    pub size: (u32, u32),
}

impl Grid {
    /// A grid shaped for a picture of `source` size.
    pub fn for_picture(source: (u32, u32)) -> Grid {
        let (w, h) = (source.0.max(1), source.1.max(1));
        let width = WIDTH.min(w);
        // Rounded up so a very wide picture still gets at least one row.
        let height = ((width as u64 * h as u64) / w as u64).max(1) as u32;
        Grid { cells: vec![0; (width * height) as usize], size: (width, height) }
    }

    pub fn clear(&mut self) {
        self.cells.fill(0);
    }

    pub fn is_empty(&self) -> bool {
        self.cells.iter().all(|c| *c == 0)
    }

    /// Draw a filled rectangle in picture coordinates.
    fn fill(&mut self, source: (u32, u32), bounds: (f32, f32, f32, f32), id: u8) {
        let (sw, sh) = (source.0.max(1) as f32, source.1.max(1) as f32);
        let (gw, gh) = (self.size.0 as i64, self.size.1 as i64);
        let to_cell = |x: f32, y: f32| {
            (
                ((x / sw) * gw as f32).round() as i64,
                ((y / sh) * gh as f32).round() as i64,
            )
        };
        let (x0, y0) = to_cell(bounds.0, bounds.1);
        let (x1, y1) = to_cell(bounds.2, bounds.3);

        for y in y0.max(0)..y1.min(gh) {
            for x in x0.max(0)..x1.min(gw) {
                self.cells[(y * gw + x) as usize] = id;
            }
        }
    }

    /// Draw every found thing as a filled box.
    ///
    /// Filled rather than outlined, because the shader draws the outline: it
    /// finds edges by comparing neighbours, so what it needs is the region,
    /// not a picture of its border.
    pub fn draw(&mut self, found: &[crate::ml::segment::Segmented], source: (u32, u32)) {
        self.clear();
        self.add_boxes(found, source);
    }

    /// Fill boxes over whatever is already there.
    ///
    /// Takes references rather than a slice for the same reason the outline
    /// path does: the caller often holds a borrowed subset, and a `Segmented`
    /// carries a mask that is not worth copying to read a rectangle off it.
    pub fn add_boxes<'a, I>(&mut self, found: I, source: (u32, u32))
    where
        I: IntoIterator<Item = &'a crate::ml::segment::Segmented>,
        I::IntoIter: DoubleEndedIterator,
    {
        // Least confident first, so the most confident ends up on top where
        // two overlap.
        for item in found.into_iter().rev() {
            self.fill(source, item.found.bounds, slot(item.id));
        }
    }

    /// Draw outlines that are already on this grid.
    ///
    /// The difference from `add_outlines` is what is left to undo. Those masks
    /// come back in the model's own letterboxed square and have to be mapped
    /// out of it; SAM 3's arrive at the picture's own resolution and are
    /// reduced to exactly this grid on the way in, so mapping them again puts
    /// every outline somewhere the thing is not — which is how it looked the
    /// first time this was wired up: faint smears near the edges rather than
    /// people.
    pub fn add_cells<'a, I>(&mut self, found: I)
    where
        I: IntoIterator<Item = &'a crate::ml::segment::Segmented>,
        I::IntoIter: DoubleEndedIterator,
    {
        let (gw, gh) = (self.size.0 as usize, self.size.1 as usize);
        // Least confident first, so the most confident ends up on top.
        for item in found.into_iter().rev() {
            let (mw, mh) = item.size;
            if mw == 0 || mh == 0 || item.mask.len() < mw * mh {
                continue;
            }
            let id = slot(item.id);
            // Sampled rather than required to match. The worker decodes the
            // picture at whatever size suits it, which is not always the size
            // the viewer is showing — and a mask that was silently skipped
            // because it was a few pixels out is a model that appears to find
            // nothing. Both grids describe the whole picture, so the mapping
            // is a straight proportion.
            for y in 0..gh {
                let sy = (y * mh / gh).min(mh - 1);
                for x in 0..gw {
                    let sx = (x * mw / gw).min(mw - 1);
                    if item.mask[sy * mw + sx] > crate::ml::segment::INSIDE {
                        self.cells[y * gw + x] = id;
                    }
                }
            }
        }
    }

    /// Draw outlines rather than boxes, where a model provided them.
    ///
    /// The mask arrives at the prototype grid's resolution — 160 across the
    /// whole frame — while this grid is 480 across. So one mask cell covers
    /// several cells here and has to be drawn as its *footprint*, not as a
    /// point: writing one cell per mask cell leaves a dotted outline with gaps
    /// the shader would then find edges all through.
    ///
    /// Worth being plain about what this resolution means: a distant object's
    /// outline is a handful of cells however good the model is. This is
    /// instance segmentation, not matting.
    pub fn draw_outlines(
        &mut self,
        found: &[crate::ml::segment::Segmented],
        input: &crate::ml::prep::Input,
        source: (u32, u32),
    ) {
        self.draw_outlines_of(found.iter(), input, source);
    }

    /// The same, over whatever selection of things the caller has in hand.
    ///
    /// Following one object draws only that one, and it holds a borrowed
    /// subset rather than a copied vector — the masks are the largest thing
    /// in a result and copying them to draw one would be work for nothing.
    pub fn draw_outlines_of<'a, I>(
        &mut self,
        found: I,
        input: &crate::ml::prep::Input,
        source: (u32, u32),
    ) where
        I: IntoIterator<Item = &'a crate::ml::segment::Segmented>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.clear();
        self.add_outlines(found, input, source);
    }

    /// Draw outlines over whatever is already there.
    ///
    /// Two models can be looking at the same picture — one naming everything
    /// in it, one answering where a finger was pointed — and they place their
    /// masks with different geometry. Adding rather than replacing is what
    /// lets both appear at once without either owning the grid.
    pub fn add_outlines<'a, I>(
        &mut self,
        found: I,
        input: &crate::ml::prep::Input,
        source: (u32, u32),
    ) where
        I: IntoIterator<Item = &'a crate::ml::segment::Segmented>,
        I::IntoIter: DoubleEndedIterator,
    {
        let (gw, gh) = (self.size.0 as i64, self.size.1 as i64);
        let (sw, sh) = (source.0.max(1) as f32, source.1.max(1) as f32);

        // Least confident first, so the most confident ends up on top.
        for item in found.into_iter().rev() {
            let id = slot(item.id);
            if item.size.0 == 0 || item.size.1 == 0 {
                continue;
            }
            // One mask cell is this many of the model's own pixels.
            let per_cell = input.size as f32 / item.grid.0.max(1) as f32;

            // This grid is walked, not the mask, and each of its cells asks the
            // mask what is there. That is what raises the resolution: the
            // answer is interpolated between the model's cells, so the edge
            // lands where the numbers cross rather than on a mask-cell
            // boundary. Walking the mask instead can only stamp squares, and
            // the size of those squares is then the finest edge possible.
            let (mut lo_x, mut lo_y) = (f32::MAX, f32::MAX);
            let (mut hi_x, mut hi_y) = (f32::MIN, f32::MIN);
            for (px, py) in [
                (item.origin.0, item.origin.1),
                (item.origin.0 + item.size.0, item.origin.1 + item.size.1),
            ] {
                let (sx, sy) = input.to_source(px as f32 * per_cell, py as f32 * per_cell);
                let (cx, cy) = ((sx / sw) * gw as f32, (sy / sh) * gh as f32);
                lo_x = lo_x.min(cx);
                lo_y = lo_y.min(cy);
                hi_x = hi_x.max(cx);
                hi_y = hi_y.max(cy);
            }
            let x0 = (lo_x.floor() as i64).clamp(0, gw);
            let y0 = (lo_y.floor() as i64).clamp(0, gh);
            let x1 = (hi_x.ceil() as i64).clamp(x0, gw);
            let y1 = (hi_y.ceil() as i64).clamp(y0, gh);

            // The mask, sampled between its cells. Outside the window the
            // model gave, there is nothing.
            let at = |x: f32, y: f32| -> f32 {
                let (mx, my) = (x - item.origin.0 as f32 - 0.5, y - item.origin.1 as f32 - 0.5);
                let (w, h) = (item.size.0 as f32, item.size.1 as f32);
                if mx <= -1.0 || my <= -1.0 || mx >= w || my >= h {
                    return f32::MIN;
                }
                let (fx, fy) = (mx.floor(), my.floor());
                let (tx, ty) = (mx - fx, my - fy);
                let cell = |cx: f32, cy: f32| -> f32 {
                    let cx = (cx.clamp(0.0, w - 1.0)) as usize;
                    let cy = (cy.clamp(0.0, h - 1.0)) as usize;
                    item.mask[cy * item.size.0 + cx]
                };
                let top = cell(fx, fy) * (1.0 - tx) + cell(fx + 1.0, fy) * tx;
                let bottom = cell(fx, fy + 1.0) * (1.0 - tx) + cell(fx + 1.0, fy + 1.0) * tx;
                top * (1.0 - ty) + bottom * ty
            };

            for cy in y0..y1 {
                for cx in x0..x1 {
                    // The grid cell's middle, back in the model's own square,
                    // and from there in mask cells.
                    let (px, py) = ((cx as f32 + 0.5) / gw as f32 * sw,
                                    (cy as f32 + 0.5) / gh as f32 * sh);
                    let (mx, my) = (px / input.scale + input.pad.0, py / input.scale + input.pad.1);
                    let (mx, my) = (mx / per_cell, my / per_cell);
                    if at(mx, my) > crate::ml::segment::INSIDE {
                        self.cells[(cy * gw + cx) as usize] = id;
                    }
                }
            }
        }
    }
}

/// The colours a grid's ids index, as the shader wants them.
///
/// Slot zero is never read — a cell of zero returns before the lookup — so the
/// table starts at one and matches `draw`'s `class % 15 + 1`.
pub fn palette() -> Vec<[f32; 4]> {
    let mut out = vec![[0.0; 4]];
    for slot in 0..15usize {
        let c = super::colour(slot);
        out.push([
            c[0] as f32 / 255.0,
            c[1] as f32 / 255.0,
            c[2] as f32 / 255.0,
            1.0,
        ]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(class: usize, bounds: (f32, f32, f32, f32)) -> Found {
        Found { class, confidence: 0.9, bounds }
    }

    /// A box as the grid now takes them: with the tracker id that decides its
    /// colour. One is the first id the tracker hands out.
    fn thing(class: usize, bounds: (f32, f32, f32, f32)) -> crate::ml::segment::Segmented {
        with_id(class, bounds, 1)
    }

    fn with_id(
        class: usize,
        bounds: (f32, f32, f32, f32),
        id: u32,
    ) -> crate::ml::segment::Segmented {
        let mut item = crate::ml::segment::Segmented::from_box(found(class, bounds));
        item.id = id;
        item
    }

    #[test]
    fn the_grid_keeps_the_picture_s_shape() {
        let grid = Grid::for_picture((1920, 1080));
        assert_eq!(grid.size.0, WIDTH);
        assert_eq!(grid.size.1, WIDTH * 9 / 16, "16:9 should stay 16:9");

        let tall = Grid::for_picture((1080, 1920));
        assert!(tall.size.1 > tall.size.0, "a portrait picture should be taller than wide");
    }

    #[test]
    fn a_box_lands_where_it_was_asked_for() {
        let source = (1000, 1000);
        let mut grid = Grid::for_picture(source);
        // The middle quarter of the picture.
        grid.draw(&[thing(0, (250.0, 250.0, 750.0, 750.0))], source);

        let (w, h) = (grid.size.0 as usize, grid.size.1 as usize);
        let at = |x: usize, y: usize| grid.cells[y * w + x];
        assert_ne!(at(w / 2, h / 2), 0, "the middle should be covered");
        assert_eq!(at(2, 2), 0, "the corner should not be");
    }

    #[test]
    fn nothing_found_leaves_an_empty_grid() {
        let mut grid = Grid::for_picture((640, 480));
        grid.draw(&[thing(0, (0.0, 0.0, 100.0, 100.0))], (640, 480));
        assert!(!grid.is_empty());
        grid.draw(&[], (640, 480));
        assert!(grid.is_empty(), "the last result was left on screen");
    }

    #[test]
    fn a_box_running_off_the_picture_is_clipped_rather_than_wrapping() {
        let source = (640, 480);
        let mut grid = Grid::for_picture(source);
        // Well outside on every side.
        grid.draw(&[thing(0, (-500.0, -500.0, 5000.0, 5000.0))], source);
        // Every cell covered, and no panic.
        assert!(grid.cells.iter().all(|c| *c != 0));
    }

    #[test]
    fn an_id_is_never_zero_because_zero_means_nothing_there() {
        let source = (640, 480);
        let mut grid = Grid::for_picture(source);
        // The id that lands on the wrap — slot zero if the arithmetic is
        // wrong, and slot zero is not drawn at all.
        grid.draw(&[with_id(0, (0.0, 0.0, 640.0, 480.0), SLOTS - 1)], source);
        assert!(grid.cells.iter().all(|c| *c != 0), "a found thing became background");
        for id in 0..64u32 {
            assert_ne!(slot(id), 0, "id {id} would not be drawn");
        }
    }

    #[test]
    fn a_mask_already_on_the_grid_is_drawn_where_it_is() {
        let source = (200u32, 100u32);
        let mut grid = Grid::for_picture(source);
        let (gw, gh) = (grid.size.0 as usize, grid.size.1 as usize);
        // Inside on the left half only.
        let mut item = with_id(0, (0.0, 0.0, 100.0, 100.0), 3);
        item.mask = (0..gw * gh).map(|i| if i % gw < gw / 2 { 1.0 } else { -1.0 }).collect();
        item.size = (gw, gh);
        item.grid = (gw, gh);
        grid.clear();
        grid.add_cells(std::iter::once(&item));

        let at = |x: usize, y: usize| grid.cells[y * gw + x];
        assert_eq!(at(gw / 4, gh / 2), slot(3), "the left should carry its slot");
        assert_eq!(at(gw * 3 / 4, gh / 2), 0, "the right should be untouched");
    }

    #[test]
    fn a_mask_at_another_size_is_sampled_onto_this_grid() {
        // The worker decodes at its own size. A mask that only lands when the
        // two happen to agree is a model that silently finds nothing.
        let mut grid = Grid::for_picture((200, 100));
        let (gw, gh) = (grid.size.0 as usize, grid.size.1 as usize);
        assert_ne!((gw, gh), (40, 20), "the fixture must be a different size");

        let mut item = with_id(0, (0.0, 0.0, 100.0, 100.0), 2);
        // Left half inside, at a coarser resolution than the grid.
        item.size = (40, 20);
        item.mask = (0..40 * 20).map(|i| if i % 40 < 20 { 1.0 } else { -1.0 }).collect();
        grid.clear();
        grid.add_cells(std::iter::once(&item));

        let at = |x: usize, y: usize| grid.cells[y * gw + x];
        assert_eq!(at(gw / 4, gh / 2), slot(2), "the left should carry its slot");
        assert_eq!(at(gw * 3 / 4, gh / 2), 0, "the right should be untouched");
    }

    #[test]
    fn a_mask_with_no_size_is_ignored_rather_than_panicking() {
        let mut grid = Grid::for_picture((200, 100));
        let mut item = with_id(0, (0.0, 0.0, 10.0, 10.0), 1);
        item.mask = vec![1.0; 9];
        item.size = (0, 0);
        grid.clear();
        grid.add_cells(std::iter::once(&item));
        assert!(grid.is_empty());
    }

    #[test]
    fn two_of_a_kind_are_drawn_in_different_slots() {
        // The change this file exists for: colour follows the object, not its
        // kind, so two people are told apart on sight.
        let source = (200, 100);
        let mut grid = Grid::for_picture(source);
        grid.draw(
            &[with_id(0, (0.0, 0.0, 90.0, 100.0), 1), with_id(0, (110.0, 0.0, 200.0, 100.0), 2)],
            source,
        );
        let (w, h) = (grid.size.0 as usize, grid.size.1 as usize);
        let at = |x: usize, y: usize| grid.cells[y * w + x];
        let left = at(w / 8, h / 2);
        let right = at(w - w / 8, h / 2);
        assert_ne!(left, 0, "the left person was not drawn");
        assert_ne!(right, 0, "the right person was not drawn");
        assert_ne!(left, right, "two people came out the same colour");
    }

    #[test]
    fn a_cut_out_does_not_take_a_tracked_thing_s_colour() {
        // A cut-out has no tracker id. Its slot is reserved, so it cannot
        // collide with anything the tracker hands out.
        assert_eq!(slot(0), CUT_OUT);
        for id in 1..1000u32 {
            assert_ne!(slot(id), CUT_OUT, "tracked id {id} landed on the cut-out's slot");
        }
    }

    #[test]
    fn the_most_confident_is_drawn_on_top_where_two_overlap() {
        let source = (100, 100);
        let mut grid = Grid::for_picture(source);
        let whole = (0.0, 0.0, 100.0, 100.0);
        // Sorted most-confident-first, as the decoder returns them.
        let mut sure = with_id(0, whole, 1);
        sure.found.confidence = 0.9;
        let mut unsure = with_id(2, whole, 2);
        unsure.found.confidence = 0.3;
        grid.draw(&[sure, unsure], source);

        let middle = grid.cells[(grid.size.1 / 2 * grid.size.0 + grid.size.0 / 2) as usize];
        assert_eq!(middle, slot(1), "the less confident box covered the more confident one");
    }

    /// A drawn shape must be solid. The shader finds edges by comparing
    /// neighbours, so a region with holes in it comes out as edge everywhere —
    /// which is what stamping one grid cell per mask cell used to produce, and
    /// what asking each grid cell what is under it cannot.
    #[test]
    fn a_drawn_shape_has_no_holes_in_it() {
        use crate::ml::prep::Input;
        use crate::ml::segment::Segmented;
        use crate::ml::vision::Found;

        let source = (1920u32, 1080u32);
        let mut grid = Grid::for_picture(source);
        let mut input = Input::new(640);
        // 1920 wide into a 640 square: a third, with letterbox top and bottom.
        input.scale = 3.0;
        input.pad = (0.0, (640.0 - 1080.0 / 3.0) / 2.0);

        // A solid 8x8 block of mask, well inside the frame.
        let block = 8;
        let item = Segmented {
            id: 0,
            found: Found { class: 0, confidence: 0.9, bounds: (0.0, 0.0, 1.0, 1.0) },
            mask: vec![1.0; block * block],
            origin: (40, 60),
            size: (block, block),
            grid: (160, 160),
        };
        grid.draw_outlines(&[item], &input, source);

        // Every row of the drawn region must be continuous: no gaps inside it.
        let w = grid.size.0 as usize;
        let mut worst_gap = 0;
        for row in grid.cells.chunks(w) {
            let filled: Vec<usize> =
                row.iter().enumerate().filter(|(_, c)| **c != 0).map(|(i, _)| i).collect();
            for pair in filled.windows(2) {
                worst_gap = worst_gap.max(pair[1] - pair[0]);
            }
        }
        assert!(worst_gap <= 1, "the outline came out dotted, worst gap {worst_gap}");
        assert!(grid.cells.iter().any(|c| *c != 0), "nothing was drawn at all");
    }

    #[test]
    fn the_palette_leaves_slot_zero_alone() {
        let table = palette();
        assert_eq!(table[0], [0.0; 4], "slot zero is background and must not be drawn");
        assert_eq!(table.len(), 16);
        assert_ne!(table[1], table[2], "two classes must not share a colour");
    }
}
