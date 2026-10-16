//! Turning a decoded frame into what a vision model expects.
//!
//! Every ONNX detection model in this design wants the same thing: a square
//! RGB image, planar, `f32` in 0..1, at a fixed size. A decoded frame is none
//! of those — it is planar YUV 4:2:0 at the film's own aspect ratio. This is
//! the work in between, and it is worth measuring on its own, because it
//! happens for every frame whether or not the model turns out to be fast.
//!
//! The picture is letterboxed rather than stretched. Stretching changes the
//! shape of everything in the frame, and a detector trained on undistorted
//! boxes will place undistorted boxes on a distorted picture — which is wrong
//! in a way that is hard to see and impossible to correct afterwards.

use crate::video::Frame;

/// What models in this design are trained at.
pub const DEFAULT_SIZE: u32 = 640;

/// The grey a letterbox is padded with.
///
/// 114/255, which is what the YOLO family trains against. The padding is part
/// of the input the model sees, so matching the training value costs nothing
/// and removes one difference between our input and the model's expectation.
const PAD: f32 = 114.0 / 255.0;

/// A frame prepared for a model, and how to read its output back.
pub struct Input {
    /// Planar RGB, `3 * size * size` floats in 0..1.
    pub data: Vec<f32>,
    pub size: u32,
    /// Source pixels per model pixel. Multiply a model coordinate by this,
    /// after removing `pad`, to get back to the picture.
    pub scale: f32,
    /// Letterbox offset within the square, in model pixels.
    pub pad: (f32, f32),
}

impl Input {
    pub fn new(size: u32) -> Input {
        Input {
            data: vec![PAD; 3 * (size * size) as usize],
            size,
            scale: 1.0,
            pad: (0.0, 0.0),
        }
    }

    /// The letterbox a picture of this size would get, without doing the work.
    ///
    /// Drawing a mask needs the same mapping the model saw, and recomputing it
    /// from the picture's size is exact — the alternative is carrying a copy
    /// of it alongside every result, where it could drift.
    pub fn for_source(source: (u32, u32)) -> Input {
        let mut input = Input::new(DEFAULT_SIZE);
        let (sw, sh) = (source.0.max(1) as f32, source.1.max(1) as f32);
        let fit = (DEFAULT_SIZE as f32 / sw).min(DEFAULT_SIZE as f32 / sh);
        let (dw, dh) = ((sw * fit).round(), (sh * fit).round());
        input.scale = 1.0 / fit;
        input.pad = (
            ((DEFAULT_SIZE as f32 - dw) / 2.0).floor(),
            ((DEFAULT_SIZE as f32 - dh) / 2.0).floor(),
        );
        input
    }

    /// Map a point in model space back to the source picture.
    ///
    /// The other half of letterboxing: a model reports boxes in its own square,
    /// and they mean nothing until they are back in the picture's coordinates.
    #[allow(dead_code)]
    pub fn to_source(&self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.pad.0) * self.scale, (y - self.pad.1) * self.scale)
    }
}

/// Fill `out` from `frame`, letterboxed and converted.
///
/// Reuses the buffer: at five inferences a second a fresh 4.9 MB allocation
/// each time is avoidable waste, and the padding only has to be written where
/// the picture does not reach.
pub fn letterbox(frame: &Frame, out: &mut Input) {
    let size = out.size as usize;
    let (sw, sh) = (frame.width as f32, frame.height as f32);
    if sw <= 0.0 || sh <= 0.0 {
        return;
    }

    // The longer side decides; the shorter one is centred in what is left.
    let fit = (size as f32 / sw).min(size as f32 / sh);
    let (dw, dh) = ((sw * fit).round(), (sh * fit).round());
    let (ox, oy) = (((size as f32 - dw) / 2.0).floor(), ((size as f32 - dh) / 2.0).floor());

    out.scale = 1.0 / fit;
    out.pad = (ox, oy);
    out.data.fill(PAD);

    let (rows, offsets) = frame.colour.matrix();
    let plane = size * size;

    for y in 0..dh as usize {
        // Bilinear vertically and horizontally on luma: at this reduction a
        // nearest sample drops most of the detail the model is looking for.
        let sy = ((y as f32 + 0.5) * out.scale - 0.5).clamp(0.0, sh - 1.0);
        let (y0, fy) = (sy.floor(), sy.fract());
        let (y0, y1) = (y0 as usize, (y0 as usize + 1).min(frame.height as usize - 1));

        // Chroma is half resolution already; sampling it nearest costs
        // nothing visible and halves the work.
        let cy = ((sy * 0.5) as usize).min(frame.chroma.1 as usize - 1);

        let row = (y + oy as usize) * size + ox as usize;
        for x in 0..dw as usize {
            let sx = ((x as f32 + 0.5) * out.scale - 0.5).clamp(0.0, sw - 1.0);
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
            let at = row + x;
            for (c, m) in rows.iter().enumerate() {
                let v = m[0] * ycc[0] + m[1] * ycc[1] + m[2] * ycc[2];
                out.data[c * plane + at] = v.clamp(0.0, 1.0);
            }
        }
    }
}

/// The same picture, squashed to a square of bytes rather than letterboxed
/// into a square of floats.
///
/// SAM 3 wants both of those differences. Its encoder declares
/// `uint8[3,1008,1008]` and normalises inside the graph, and its exporter
/// resizes to the square without preserving the aspect — so a letterbox here
/// would place every mask slightly wrong, in a way that looks like the model
/// being imprecise rather than like a bug. Matching what the model was called
/// with matters more than what is theoretically nicer to look at.
pub fn squash_u8(frame: &Frame, size: usize) -> Vec<u8> {
    let mut out = vec![0u8; 3 * size * size];
    let (sw, sh) = (frame.width as f32, frame.height as f32);
    if sw <= 0.0 || sh <= 0.0 || size == 0 {
        return out;
    }
    let (rows, offsets) = frame.colour.matrix();
    let plane = size * size;
    // Each axis scales on its own: that is what makes this a squash.
    let (xs, ys) = (sw / size as f32, sh / size as f32);

    for y in 0..size {
        let sy = ((y as f32 + 0.5) * ys - 0.5).clamp(0.0, sh - 1.0);
        let (y0, fy) = (sy.floor(), sy.fract());
        let (y0, y1) = (y0 as usize, (y0 as usize + 1).min(frame.height as usize - 1));
        let cy = ((sy * 0.5) as usize).min(frame.chroma.1 as usize - 1);

        for x in 0..size {
            let sx = ((x as f32 + 0.5) * xs - 0.5).clamp(0.0, sw - 1.0);
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
            let at = y * size + x;
            for (c, m) in rows.iter().enumerate() {
                let v = m[0] * ycc[0] + m[1] * ycc[1] + m[2] * ycc[2];
                out[c * plane + at] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

/// A decoded still, in the planar form this module expects.
///
/// The preparation is written against video frames because that is where the
/// volume is; a still is converted to the same shape rather than every step
/// after it growing a second path through it. `None` when the picture has no
/// size, which is the only way this can fail.
///
/// It costs a pass over the picture, so the caller is expected to keep the
/// result rather than call it per frame — there is only ever one still on
/// screen at a time.
pub fn from_pixels(rgba: &[u8], width: u32, height: u32) -> Option<Frame> {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || rgba.len() < w * h * 4 {
        return None;
    }
    let chroma = ((w as u32).div_ceil(2), (h as u32).div_ceil(2));

    let mut y = vec![0u8; w * h];
    let mut u = vec![128u8; (chroma.0 * chroma.1) as usize];
    let mut v = vec![128u8; (chroma.0 * chroma.1) as usize];

    // BT.709 full range, which is what the `Colour` below tells the converter
    // to undo.
    for row in 0..h {
        for col in 0..w {
            let p = (row * w + col) * 4;
            let (r, g, b) = (rgba[p] as f32, rgba[p + 1] as f32, rgba[p + 2] as f32);
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            y[row * w + col] = luma.clamp(0.0, 255.0) as u8;
            if row % 2 == 0 && col % 2 == 0 {
                let ci = (row / 2) * chroma.0 as usize + col / 2;
                u[ci] = ((b - luma) / 1.8556 + 128.0).clamp(0.0, 255.0) as u8;
                v[ci] = ((r - luma) / 1.5748 + 128.0).clamp(0.0, 255.0) as u8;
            }
        }
    }

    Some(Frame {
        key: true,
        generation: 0,
        planes: crate::video::Planes::cpu([std::sync::Arc::new(y), std::sync::Arc::new(u), std::sync::Arc::new(v)]),
        width,
        height,
        chroma,
        at: std::time::Duration::ZERO,
        colour: crate::video::Colour { hd: true, full_range: true },
    })
}

/// A frame's worth of work, for reporting.
pub fn bytes_per_input(size: u32) -> u64 {
    3 * (size as u64) * (size as u64) * 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::Colour;
    use std::time::Duration;

    /// A flat frame of one colour, for checking the conversion end to end.
    fn frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> Frame {
        let chroma = (width.div_ceil(2), height.div_ceil(2));
        Frame {
            key: true,
            generation: 0,
            planes: crate::video::Planes::cpu([
                std::sync::Arc::new(vec![y; (width * height) as usize]),
                std::sync::Arc::new(vec![u; (chroma.0 * chroma.1) as usize]),
                std::sync::Arc::new(vec![v; (chroma.0 * chroma.1) as usize]),
            ]),
            width,
            height,
            chroma,
            at: Duration::ZERO,
            colour: Colour { hd: true, full_range: true },
        }
    }

    #[test]
    fn a_square_picture_fills_the_square() {
        let mut input = Input::new(64);
        letterbox(&frame(100, 100, 128, 128, 128), &mut input);
        assert_eq!(input.pad, (0.0, 0.0), "a square needs no letterbox");
        assert!((input.scale - 100.0 / 64.0).abs() < 0.01);
    }

    #[test]
    fn a_wide_picture_is_padded_top_and_bottom() {
        let mut input = Input::new(64);
        letterbox(&frame(128, 64, 128, 128, 128), &mut input);
        assert_eq!(input.pad.0, 0.0, "a wide picture spans the full width");
        assert_eq!(input.pad.1, 16.0, "and is centred vertically");

        // The corner is padding, the middle is picture.
        assert!((input.data[0] - PAD).abs() < 1e-6, "the corner should be padding");
        let middle = 32 * 64 + 32;
        assert!((input.data[middle] - PAD).abs() > 1e-3, "the middle should be picture");
    }

    #[test]
    fn the_aspect_ratio_is_preserved_rather_than_stretched() {
        let mut input = Input::new(64);
        letterbox(&frame(200, 100, 128, 128, 128), &mut input);
        // 200x100 fits as 64x32, so 16 rows of padding above and below.
        assert_eq!(input.pad, (0.0, 16.0));
    }

    #[test]
    fn white_converts_to_white_and_black_to_black() {
        // Full-range BT.709: Y=255 with neutral chroma is white.
        let mut input = Input::new(32);
        letterbox(&frame(32, 32, 255, 128, 128), &mut input);
        let plane = 32 * 32;
        for c in 0..3 {
            let v = input.data[c * plane + plane / 2];
            assert!(v > 0.95, "channel {c} should be white, got {v}");
        }

        letterbox(&frame(32, 32, 0, 128, 128), &mut input);
        for c in 0..3 {
            let v = input.data[c * plane + plane / 2];
            assert!(v < 0.05, "channel {c} should be black, got {v}");
        }
    }

    #[test]
    fn a_model_coordinate_maps_back_to_the_picture() {
        let mut input = Input::new(64);
        letterbox(&frame(128, 64, 128, 128, 128), &mut input);
        // The centre of the model square is the centre of the picture.
        let (x, y) = input.to_source(32.0, 32.0);
        assert!((x - 64.0).abs() < 1.0, "x came back as {x}");
        assert!((y - 32.0).abs() < 1.0, "y came back as {y}");
    }

    #[test]
    fn the_buffer_is_reused_without_leaving_the_last_frame_behind() {
        let mut input = Input::new(64);
        letterbox(&frame(64, 64, 255, 128, 128), &mut input);
        // A smaller picture must not leave the previous one showing round it.
        letterbox(&frame(64, 32, 0, 128, 128), &mut input);
        assert!(
            (input.data[0] - PAD).abs() < 1e-6,
            "the old frame is still in the padding: {}",
            input.data[0]
        );
    }
}
