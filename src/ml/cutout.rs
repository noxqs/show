//! Cutting out whatever was pointed at.
//!
//! A detector answers "what is in this picture" from a fixed list of eighty
//! things. This answers a different question — "what is *that*" — for anything
//! at all, named or not, because the prompt is a point rather than a class.
//!
//! The work splits in two, and the split is the whole reason this is usable:
//! an encoder turns the picture into an embedding, which is slow (a fifth of a
//! second), and a decoder answers a point against that embedding, which is
//! not (a hundredth). Clicking a second time on the same picture skips the
//! slow half entirely.
//!
//! Everything here except the session handling is ordinary arithmetic, and is
//! tested as such: the square the encoder wants, the point mapped into it, and
//! the mask mapped back out.

use crate::ml::prep::Input;
use crate::ml::segment::Segmented;
use crate::ml::vision::Found;
use crate::video::Frame;

/// The square the encoder was trained on.
pub const SIDE: u32 = 1024;

/// The side of the mask it answers with.
pub const GRID: usize = 256;

/// The class given to something the detector has no name for.
///
/// Deliberately past the end of the class list, so `Found::label` says
/// "thing" — which is the honest answer, and better than borrowing a name
/// from a model that was never asked.
pub const UNNAMED: usize = crate::ml::vision::CLASSES.len();

/// A picture in the form the encoder wants it: RGB, interleaved, 0..255,
/// the long side scaled to `SIDE` and the rest left black.
pub struct Square {
    pub pixels: Vec<f32>,
}

/// How the square maps back to the picture it came from.
///
/// Expressed as a [`Input`] so a cut-out draws through exactly the same path
/// as a segmentation model's outline — the drawing code asks the geometry
/// where a mask cell belongs and does not care which model produced it.
///
/// The padding is zero because this encoder wants the picture in the corner,
/// not centred: getting that wrong puts every mask a hundred pixels from what
/// was clicked, which is not obviously a coordinate fault when you see it.
pub fn geometry(source: (u32, u32)) -> Input {
    let (sw, sh) = (source.0.max(1) as f32, source.1.max(1) as f32);
    let fit = SIDE as f32 / sw.max(sh);
    let mut input = Input::new(0);
    input.size = SIDE;
    input.scale = 1.0 / fit;
    input.pad = (0.0, 0.0);
    input
}

/// Where a point in the picture falls in the encoder's square.
pub fn point_in_square(point: (f32, f32), source: (u32, u32)) -> (f32, f32) {
    let (sw, sh) = (source.0.max(1) as f32, source.1.max(1) as f32);
    let fit = SIDE as f32 / sw.max(sh);
    (point.0 * fit, point.1 * fit)
}

/// Convert a frame into the encoder's square.
///
/// Bilinear, because the reduction is large and a nearest sample throws away
/// most of what the model is being asked to look at.
pub fn square(frame: &Frame) -> Square {
    let side = SIDE as usize;
    let (sw, sh) = (frame.width as f32, frame.height as f32);
    let mut pixels = vec![0.0f32; side * side * 3];
    if sw <= 0.0 || sh <= 0.0 {
        return Square { pixels };
    }

    let fit = SIDE as f32 / sw.max(sh);
    let (dw, dh) = ((sw * fit).round() as usize, (sh * fit).round() as usize);
    let back = 1.0 / fit;
    let (rows, offsets) = frame.colour.matrix();

    for y in 0..dh.min(side) {
        let sy = ((y as f32 + 0.5) * back - 0.5).clamp(0.0, sh - 1.0);
        let (y0, fy) = (sy.floor(), sy.fract());
        let (y0, y1) = (y0 as usize, (y0 as usize + 1).min(frame.height as usize - 1));
        let cy = ((sy * 0.5) as usize).min(frame.chroma.1 as usize - 1);

        for x in 0..dw.min(side) {
            let sx = ((x as f32 + 0.5) * back - 0.5).clamp(0.0, sw - 1.0);
            let (x0, fx) = (sx.floor(), sx.fract());
            let (x0, x1) = (x0 as usize, (x0 as usize + 1).min(frame.width as usize - 1));

            let stride = frame.width as usize;
            let luma = |i: usize| frame.planes[0][i] as f32 / 255.0;
            let top = luma(y0 * stride + x0) * (1.0 - fx) + luma(y0 * stride + x1) * fx;
            let bottom = luma(y1 * stride + x0) * (1.0 - fx) + luma(y1 * stride + x1) * fx;
            let luma = top * (1.0 - fy) + bottom * fy;

            let cx = ((sx * 0.5) as usize).min(frame.chroma.0 as usize - 1);
            let ci = cy * frame.chroma.0 as usize + cx;
            let cb = frame.planes[1][ci] as f32 / 255.0;
            let cr = frame.planes[2][ci] as f32 / 255.0;

            let ycc = [luma - offsets[0], cb - offsets[1], cr - offsets[1]];
            let at = (y * side + x) * 3;
            for (c, m) in rows.iter().enumerate() {
                let v = m[0] * ycc[0] + m[1] * ycc[1] + m[2] * ycc[2];
                // 0..255 rather than 0..1: this encoder does its own
                // normalisation, and handing it values it has already scaled
                // gives a mask that is nearly right, which is the worst kind.
                pixels[at + c] = v.clamp(0.0, 1.0) * 255.0;
            }
        }
    }

    Square { pixels }
}

/// Turn a mask the decoder produced into something drawable.
///
/// The mask covers the whole padded square, most of which is usually not
/// picture at all, so it is cropped to what was actually marked and clipped to
/// the picture's own edge. `None` when nothing was marked — a click on
/// something the model cannot separate is a real answer, not an error.
pub fn shape(
    mask: &[f32],
    grid: usize,
    source: (u32, u32),
    score: f32,
) -> Option<Segmented> {
    if grid == 0 || mask.len() < grid * grid {
        return None;
    }
    let geom = geometry(source);
    let per_cell = SIDE as f32 / grid as f32;

    // The picture occupies only the top-left of the square; cells past it are
    // padding and anything marked there is noise.
    let (sw, sh) = (source.0.max(1) as f32, source.1.max(1) as f32);
    let inside = |cx: usize, cy: usize| -> bool {
        let (x, y) = geom.to_source(cx as f32 * per_cell, cy as f32 * per_cell);
        x < sw && y < sh
    };

    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for cy in 0..grid {
        for cx in 0..grid {
            // Logits, not probabilities: this decoder emits the value before
            // the sigmoid, so zero is the boundary.
            if mask[cy * grid + cx] <= 0.0 || !inside(cx, cy) {
                continue;
            }
            x0 = x0.min(cx);
            y0 = y0.min(cy);
            x1 = x1.max(cx + 1);
            y1 = y1.max(cy + 1);
        }
    }
    if x0 == usize::MAX {
        return None;
    }

    let (w, h) = (x1 - x0, y1 - y0);
    let mut cropped = vec![f32::MIN; w * h];
    for y in 0..h {
        for x in 0..w {
            let cell = (y0 + y) * grid + (x0 + x);
            // Kept as the model's own numbers: the edge lies between cells,
            // and the numbers on either side are what say where.
            if inside(x0 + x, y0 + y) {
                cropped[y * w + x] = mask[cell];
            }
        }
    }

    let (bx0, by0) = geom.to_source(x0 as f32 * per_cell, y0 as f32 * per_cell);
    let (bx1, by1) = geom.to_source(x1 as f32 * per_cell, y1 as f32 * per_cell);
    Some(Segmented {
        found: Found {
            class: UNNAMED,
            confidence: score,
            bounds: (bx0, by0, bx1.min(sw), by1.min(sh)),
        },
        id: 0,
        mask: cropped,
        origin: (x0, y0),
        size: (w, h),
        grid: (grid, grid),
    })
}

/// The middle of a cut-out, as a fraction of the picture.
///
/// Where to point next, when the thing being pointed at is moving. The
/// average of the marked cells rather than the middle of the box, because a
/// person's box is mostly not the person and the point has to land on them.
pub fn middle_of(cut: &Segmented, source: (u32, u32)) -> Option<(f32, f32)> {
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0u64);
    for y in 0..cut.size.1 {
        for x in 0..cut.size.0 {
            if cut.mask[y * cut.size.0 + x] > crate::ml::segment::INSIDE {
                sx += (cut.origin.0 + x) as f64;
                sy += (cut.origin.1 + y) as f64;
                n += 1;
            }
        }
    }
    if n == 0 {
        return None;
    }
    let grid = cut.grid.0.max(1) as f64;
    let per_cell = SIDE as f64 / grid;
    let geom = geometry(source);
    let (px, py) = geom.to_source(
        ((sx / n as f64 + 0.5) * per_cell) as f32,
        ((sy / n as f64 + 0.5) * per_cell) as f32,
    );
    let (sw, sh) = (source.0.max(1) as f32, source.1.max(1) as f32);
    Some((px / sw, py / sh))
}

#[cfg(feature = "ml")]
pub use with_ort::Cutter;

#[cfg(feature = "ml")]
mod with_ort {
    use super::*;
    use anyhow::{Context, Result};
    use crate::ml::catalogue::{Capability, spec};

    /// The encoder and the decoder, held open together.
    ///
    /// Both sessions are leaked for the reason given in `vision.rs`: with a
    /// `dlopen`ed runtime, destroying one segfaults the process at exit.
    pub struct Cutter {
        encoder: std::mem::ManuallyDrop<ort::session::Session>,
        decoder: std::mem::ManuallyDrop<ort::session::Session>,
    }

    impl Cutter {
        pub fn open() -> Result<Cutter> {
            crate::ml::runtime::prepare()?;
            let spec = spec(Capability::PointSegment);
            let dir = crate::ml::store::directory(spec).context("no data directory")?;

            // An eighth of the cores rather than the quarter the detector
            // takes. This one runs back to back for as long as something is
            // being followed, rather than for twenty milliseconds five times a
            // second, so it leaves more of the machine alone. Measured at this
            // setting: following a cut-out through a film costs the pacing
            // nothing that can be told from not doing it.
            let threads = std::thread::available_parallelism()
                .map(|n| (n.get() / 8).max(2))
                .unwrap_or(2);
            let open = |name: &str| -> Result<ort::session::Session> {
                let path = dir.join(name);
                anyhow::ensure!(
                    path.is_file(),
                    "cut out is not installed — press F8, or run `show --plugins install {}`",
                    Capability::PointSegment.id()
                );
                let mut builder = crate::ml::runtime::session(threads)?;
                crate::vram::charging(&path, || builder.commit_from_file(&path))
                    .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))
            };

            let encoder = open(spec.files[0].name)?;
            let decoder = open(spec.files[1].name)?;
            Ok(Cutter {
                encoder: std::mem::ManuallyDrop::new(encoder),
                decoder: std::mem::ManuallyDrop::new(decoder),
            })
        }

        /// The slow half: turn a picture into an embedding.
        pub fn embed(&mut self, square: &Square) -> Result<Vec<f32>> {
            let side = SIDE as usize;
            let value = ort::value::Value::from_array((
                [side, side, 3],
                square.pixels.clone().into_boxed_slice(),
            ))
            .map_err(|e| anyhow::anyhow!("building the input: {e}"))?;
            let name = self
                .encoder
                .inputs()
                .first()
                .map(|i| i.name().to_string())
                .context("the encoder declares no input")?;
            let outputs = self
                .encoder
                .run(ort::inputs![name.as_str() => value])
                .map_err(|e| anyhow::anyhow!("encoding: {e}"))?;
            let (_, values) = outputs
                .get("image_embeddings")
                .context("the encoder has no `image_embeddings` output")?
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("reading the embedding: {e}"))?;
            Ok(values.to_vec())
        }

        /// Sharpen a detector's boxes into outlines, against one embedding.
        ///
        /// The detector says *what* and roughly *where*; this says exactly
        /// where. One encode for the picture, then a few milliseconds per box,
        /// which is what makes doing it for everything in the frame possible
        /// at all.
        ///
        /// The class comes back untouched, so what was a person stays a person
        /// and keeps its colour: this changes the shape of the answer, not the
        /// answer.
        pub fn refine(
            &mut self,
            embedding: &[f32],
            boxes: &[Found],
            source: (u32, u32),
        ) -> Result<Vec<Segmented>> {
            let mut out = Vec::with_capacity(boxes.len());
            for found in boxes {
                let (x0, y0) = point_in_square((found.bounds.0, found.bounds.1), source);
                let (x1, y1) = point_in_square((found.bounds.2, found.bounds.3), source);
                // A box is given to this model as two points labelled 2 and 3
                // — its own convention for a corner pair, as against 1 for
                // "this is the thing" and 0 for "this is not".
                let coords = vec![x0, y0, x1, y1];
                let labels = vec![2.0f32, 3.0];
                let Some(mut cut) = self.decode(embedding, coords, labels, source)? else {
                    continue;
                };
                cut.found.class = found.class;
                cut.found.confidence = found.confidence;
                out.push(cut);
            }
            Ok(out)
        }

        /// The fast half: what is at this point, given that embedding.
        ///
        /// The model offers several answers to the same click — a thing, its
        /// parts, the thing it is part of — and the first is its own view of
        /// the whole object, which is what someone pointing at something
        /// means. The rest are not guessed between here.
        pub fn cut(
            &mut self,
            embedding: &[f32],
            point: (f32, f32),
            source: (u32, u32),
        ) -> Result<Option<Segmented>> {
            let at = point_in_square(point, source);
            self.decode(embedding, vec![at.0, at.1], vec![1.0], source)
        }

        /// Run the decoder for one prompt, whatever shape it is.
        fn decode(
            &mut self,
            embedding: &[f32],
            coords: Vec<f32>,
            labels: Vec<f32>,
            source: (u32, u32),
        ) -> Result<Option<Segmented>> {
            let points = coords.len() / 2;
            let inputs = ort::inputs![
                "image_embeddings" => ort::value::Value::from_array(([1usize, 256, 64, 64], embedding.to_vec().into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("passing the embedding: {e}"))?,
                "point_coords" => ort::value::Value::from_array(([1usize, points, 2], coords.into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("passing the point: {e}"))?,
                "point_labels" => ort::value::Value::from_array(([1usize, points], labels.into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("passing the label: {e}"))?,
                "mask_input" => ort::value::Value::from_array(([1usize, 1, GRID, GRID], vec![0.0f32; GRID * GRID].into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("passing the mask: {e}"))?,
                "has_mask_input" => ort::value::Value::from_array(([1usize], vec![0.0f32].into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("passing the mask flag: {e}"))?,
                // Asked for in the picture's own pixels, and ignored: the
                // low-resolution mask is what gets drawn, so letting the model
                // upsample to a 4K frame would be megabytes of work thrown
                // away.
                "orig_im_size" => ort::value::Value::from_array(([2usize], vec![source.1 as f32, source.0 as f32].into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("passing the size: {e}"))?,
            ];

            let outputs = self
                .decoder
                .run(inputs)
                .map_err(|e| anyhow::anyhow!("cutting out: {e}"))?;
            let (dims, low) = outputs
                .get("low_res_masks")
                .context("the decoder has no `low_res_masks` output")?
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("reading the mask: {e}"))?;
            let score = outputs
                .get("iou_predictions")
                .and_then(|v| v.try_extract_tensor::<f32>().ok())
                .and_then(|(_, v)| v.first().copied())
                .unwrap_or(0.0);

            let grid = *dims.last().unwrap_or(&(GRID as i64)) as usize;
            anyhow::ensure!(low.len() >= grid * grid, "the mask is smaller than it claims");
            Ok(shape(&low[..grid * grid], grid, source, score.clamp(0.0, 1.0)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_maps_into_the_square_and_the_mask_maps_back() {
        // A 1920x1080 frame: the width is the long side, so the picture fills
        // the top 576 rows of the square and the rest is padding.
        let source = (1920u32, 1080u32);
        let fit = SIDE as f32 / 1920.0;
        let at = point_in_square((960.0, 540.0), source);
        assert!((at.0 - 960.0 * fit).abs() < 0.01, "{at:?}");
        assert!((at.1 - 540.0 * fit).abs() < 0.01, "{at:?}");

        let geom = geometry(source);
        let back = geom.to_source(at.0, at.1);
        assert!((back.0 - 960.0).abs() < 0.01, "{back:?}");
        assert!((back.1 - 540.0).abs() < 0.01, "{back:?}");
    }

    #[test]
    fn a_mask_becomes_a_shape_in_the_pictures_own_pixels() {
        let source = (1000u32, 1000u32); // square, so the fit is exactly 1024/1000
        let mut mask = vec![-1.0f32; GRID * GRID];
        // Mark the middle quarter.
        for y in GRID / 4..GRID * 3 / 4 {
            for x in GRID / 4..GRID * 3 / 4 {
                mask[y * GRID + x] = 1.0;
            }
        }
        let cut = shape(&mask, GRID, source, 0.9).expect("something was marked");
        assert_eq!(cut.size, (GRID / 2, GRID / 2));
        assert_eq!(cut.origin, (GRID / 4, GRID / 4));
        assert_eq!(cut.found.label(), "thing", "a cut-out has no name to borrow");
        // A quarter in, in a 1000-pixel picture scaled to 1024.
        let expected = 1000.0 / 4.0;
        assert!((cut.found.bounds.0 - expected).abs() < 2.0, "{:?}", cut.found.bounds);
        assert!((cut.found.bounds.2 - expected * 3.0).abs() < 2.0, "{:?}", cut.found.bounds);
    }

    #[test]
    fn the_middle_of_a_shape_is_where_its_marks_are() {
        let source = (1000u32, 1000u32);
        let mut mask = vec![-1.0f32; GRID * GRID];
        // A blob in the top-left quarter.
        for y in 20..60 {
            for x in 30..70 {
                mask[y * GRID + x] = 1.0;
            }
        }
        let cut = shape(&mask, GRID, source, 0.9).expect("something was marked");
        let (x, y) = middle_of(&cut, source).expect("it has a middle");
        // Cell 50 of 256 across the square, and the square is the picture here.
        assert!((x - 50.0 / 256.0).abs() < 0.02, "{x}");
        assert!((y - 40.0 / 256.0).abs() < 0.02, "{y}");
    }

    #[test]
    fn nothing_marked_is_no_shape_at_all() {
        assert!(shape(&vec![-1.0; GRID * GRID], GRID, (640, 480), 0.5).is_none());
    }

    #[test]
    fn marks_in_the_padding_are_ignored() {
        // A wide picture leaves the bottom of the square empty. Anything the
        // model marks down there is not part of the picture, and taking it
        // would stretch the shape to somewhere the user cannot see.
        let source = (1024u32, 256u32);
        let mut mask = vec![-1.0f32; GRID * GRID];
        for y in GRID * 3 / 4..GRID {
            for x in 0..GRID {
                mask[y * GRID + x] = 1.0;
            }
        }
        assert!(shape(&mask, GRID, source, 0.5).is_none(), "padding was taken for picture");
    }
}
