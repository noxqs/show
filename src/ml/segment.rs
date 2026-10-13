//! Reading a segmentation model's output.
//!
//! A different shape from the detector's, and the difference is instructive:
//! this one emits **8400 raw anchors** with no suppression done, so the
//! overlapping guesses have to be thinned here. It also emits 32 *prototype*
//! masks, and each detection carries 32 coefficients that mix them into its
//! own outline. That is how a segmentation model avoids emitting a full mask
//! per instance — it emits a basis and the weights.
//!
//! Two consequences worth knowing. The masks are 160x160 for the whole frame,
//! so a small object's outline is a handful of cells however good the model
//! is: this is instance segmentation, not matting, and "pixel perfect" is not
//! on offer at this resolution. And mixing the prototypes costs 32 multiplies
//! per cell, so it is done only inside each detection's own box rather than
//! over the whole frame.

use super::prep::Input;
use super::vision::{Found, Thresholds};

/// How much two boxes may overlap before the weaker is taken as a duplicate.
pub const DUPLICATE: f32 = 0.45;

/// Where a mask cell is more likely to be the thing than not.
///
/// Zero because these are logits — the value before the sigmoid — and a
/// probability of one half is a logit of nothing at all.
pub const INSIDE: f32 = 0.0;

/// One thing found, with its outline.
pub struct Segmented {
    pub found: Found,
    /// Which track this belongs to, or zero before anything has followed it.
    pub id: u32,
    /// One value per cell, positive inside the thing. Covers `origin`..`size`
    /// of the prototype grid rather than the whole of it.
    ///
    /// Kept as the model's own numbers rather than thresholded to yes and no:
    /// the boundary lies *between* cells, and only the numbers on either side
    /// say where. Thresholding here would throw that away and leave a mask no
    /// finer than the model's grid, however fine the screen is.
    pub mask: Vec<f32>,
    pub origin: (usize, usize),
    pub size: (usize, usize),
    /// The prototype grid this is a window into.
    pub grid: (usize, usize),
}

impl Segmented {
    /// A plain box, as an outline that fills its own rectangle.
    ///
    /// So a detector without masks and one with them produce the same shape,
    /// and nothing downstream has to ask which it is dealing with.
    pub fn from_box(found: Found) -> Segmented {
        Segmented { found, id: 0, mask: Vec::new(), origin: (0, 0), size: (0, 0), grid: (0, 0) }
    }

    /// Whether this carries a real outline rather than just a box.
    pub fn has_outline(&self) -> bool {
        !self.mask.is_empty() && self.size.0 > 0 && self.size.1 > 0
    }
}

/// Which thing a point falls on, if any.
///
/// Smallest first, because things sit inside other things: clicking a dog in
/// someone's arms should choose the dog, which is the more specific answer to
/// where the pointer was. Only things being followed can be chosen — one that
/// has been seen once may not be there at all.
pub fn pick(found: &[Segmented], point: (f32, f32)) -> Option<&Segmented> {
    let area = |b: (f32, f32, f32, f32)| (b.2 - b.0) * (b.3 - b.1);
    found
        .iter()
        .filter(|s| s.id != 0)
        .filter(|s| {
            let b = s.found.bounds;
            point.0 >= b.0 && point.0 <= b.2 && point.1 >= b.1 && point.1 <= b.3
        })
        .min_by(|a, b| area(a.found.bounds).total_cmp(&area(b.found.bounds)))
}

/// Decode `output0` and `output1` into things with outlines.
///
/// `boxes` is `[1, 4 + classes + coefficients, anchors]`, transposed — every
/// anchor's first field, then every anchor's second, and so on. `protos` is
/// `[1, coefficients, height, width]`.
pub fn decode(
    boxes_shape: &[usize],
    boxes: &[f32],
    protos_shape: &[usize],
    protos: &[f32],
    input: &Input,
    limits: Thresholds,
) -> Vec<Segmented> {
    let [_, fields, anchors] = boxes_shape else { return Vec::new() };
    let [_, coefficients, ph, pw] = protos_shape else { return Vec::new() };
    let (fields, anchors) = (*fields, *anchors);
    let (coefficients, ph, pw) = (*coefficients, *ph, *pw);

    // 4 box values and the mask coefficients bracket the class scores.
    let Some(classes) = fields.checked_sub(4 + coefficients).filter(|c| *c > 0) else {
        return Vec::new();
    };
    if boxes.len() < fields * anchors || protos.len() < coefficients * ph * pw {
        return Vec::new();
    }

    // Transposed: field f of anchor a is at f * anchors + a.
    let at = |field: usize, anchor: usize| boxes[field * anchors + anchor];

    // Some exports give the box in model pixels, some in fractions of the
    // input. Nothing in the file says which, and guessing wrong puts every
    // box in the top-left corner, so it is measured: no real detection in a
    // 640-pixel square has a centre or a side under two pixels, and every
    // fraction is at most one.
    let widest = (0..4)
        .flat_map(|f| (0..anchors).map(move |a| at(f, a).abs()))
        .fold(0.0f32, f32::max);
    let units = if widest <= 2.0 { input.size as f32 } else { 1.0 };

    let mut kept: Vec<(Found, Vec<f32>)> = Vec::new();
    for anchor in 0..anchors {
        let (class, score) = (0..classes)
            .map(|c| (c, at(4 + c, anchor)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, 0.0));
        if score < limits.confidence {
            continue;
        }

        // Centre, width and height in the model's own square.
        let (cx, cy, w, h) = (
            at(0, anchor) * units,
            at(1, anchor) * units,
            at(2, anchor) * units,
            at(3, anchor) * units,
        );
        let (x1, y1) = input.to_source(cx - w / 2.0, cy - h / 2.0);
        let (x2, y2) = input.to_source(cx + w / 2.0, cy + h / 2.0);

        let coeffs: Vec<f32> =
            (0..coefficients).map(|c| at(4 + classes + c, anchor)).collect();
        kept.push((
            Found {
                class,
                confidence: score,
                bounds: (x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2)),
            },
            coeffs,
        ));
    }

    // This model does not suppress its own duplicates, so the same thing
    // arrives several times with slightly different boxes. Strongest first,
    // then drop anything that mostly covers something already kept.
    kept.sort_by(|a, b| b.0.confidence.total_cmp(&a.0.confidence));
    let mut chosen: Vec<(Found, Vec<f32>)> = Vec::new();
    for (found, coeffs) in kept {
        let duplicate = chosen.iter().any(|(other, _)| {
            other.class == found.class
                && super::track::overlap(other.bounds, found.bounds) > limits.duplicate
        });
        if !duplicate {
            chosen.push((found, coeffs));
        }
    }

    // The prototype grid covers the model's square, so the box has to be put
    // back into that square to know which cells to mix.
    let scale = pw as f32 / input.size as f32;
    chosen
        .into_iter()
        .map(|(found, coeffs)| {
            let to_proto = |x: f32, y: f32| {
                let (mx, my) = (x / input.scale + input.pad.0, y / input.scale + input.pad.1);
                ((mx * scale) as i64, (my * scale) as i64)
            };
            let (bx0, by0) = to_proto(found.bounds.0, found.bounds.1);
            let (bx1, by1) = to_proto(found.bounds.2, found.bounds.3);

            let x0 = bx0.clamp(0, pw as i64) as usize;
            let y0 = by0.clamp(0, ph as i64) as usize;
            let x1 = bx1.clamp(x0 as i64, pw as i64) as usize;
            let y1 = by1.clamp(y0 as i64, ph as i64) as usize;
            let (w, h) = (x1 - x0, y1 - y0);

            // Mixed only inside the box: 32 multiplies a cell over the whole
            // grid, for every instance, would be most of the frame budget.
            let mut mask = vec![f32::MIN; w * h];
            for y in 0..h {
                for x in 0..w {
                    let cell = (y0 + y) * pw + (x0 + x);
                    let mut sum = 0.0;
                    for (c, weight) in coeffs.iter().enumerate() {
                        sum += weight * protos[c * ph * pw + cell];
                    }
                    mask[y * w + x] = sum;
                }
            }

            Segmented { found, id: 0, mask, origin: (x0, y0), size: (w, h), grid: (pw, ph) }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one anchor's worth of output in the transposed layout.
    fn output(
        anchors: usize,
        classes: usize,
        coefficients: usize,
        entries: &[(usize, [f32; 4], usize, f32, f32)],
    ) -> (Vec<usize>, Vec<f32>) {
        let fields = 4 + classes + coefficients;
        let mut data = vec![0.0f32; fields * anchors];
        for (anchor, boxed, class, score, coeff) in entries {
            for (i, v) in boxed.iter().enumerate() {
                data[i * anchors + anchor] = *v;
            }
            data[(4 + class) * anchors + anchor] = *score;
            for c in 0..coefficients {
                data[(4 + classes + c) * anchors + anchor] = *coeff;
            }
        }
        (vec![1, fields, anchors], data)
    }

    /// Prototypes that are all ones, so any positive coefficient means inside.
    fn protos(coefficients: usize, size: usize) -> (Vec<usize>, Vec<f32>) {
        (vec![1, coefficients, size, size], vec![1.0; coefficients * size * size])
    }

    fn input() -> Input {
        // A 640x640 picture, so model space and picture space are the same and
        // the arithmetic under test is only the mask's.
        let mut input = Input::new(640);
        input.scale = 1.0;
        input.pad = (0.0, 0.0);
        input
    }

    #[test]
    fn a_confident_anchor_becomes_a_thing_with_an_outline() {
        let (bs, boxes) = output(4, 80, 32, &[(0, [320.0, 320.0, 160.0, 160.0], 0, 0.9, 1.0)]);
        let (ps, protos) = protos(32, 160);
        let found = decode(&bs, &boxes, &ps, &protos, &input(), Thresholds::default());

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].found.label(), "person");
        // The box is 160 wide in a 640 square, so 40 cells of a 160 grid.
        assert!((found[0].size.0 as i32 - 40).abs() <= 1, "{:?}", found[0].size);
        assert!(
            found[0].mask.iter().all(|c| *c > INSIDE),
            "all-ones prototypes should be inside"
        );
    }

    fn thing(id: u32, class: usize, bounds: (f32, f32, f32, f32)) -> Segmented {
        let mut s = Segmented::from_box(Found { class, confidence: 0.9, bounds });
        s.id = id;
        s
    }

    #[test]
    fn clicking_picks_the_smallest_thing_under_the_pointer() {
        let found = [
            thing(1, 0, (0.0, 0.0, 400.0, 400.0)),
            thing(2, 16, (100.0, 100.0, 200.0, 200.0)),
            thing(3, 2, (600.0, 600.0, 700.0, 700.0)),
        ];
        assert_eq!(pick(&found, (150.0, 150.0)).map(|s| s.id), Some(2), "the inner one");
        assert_eq!(pick(&found, (350.0, 350.0)).map(|s| s.id), Some(1), "only the outer one");
        assert_eq!(pick(&found, (900.0, 900.0)).map(|s| s.id), None, "nothing there");
    }

    #[test]
    fn something_seen_only_once_cannot_be_followed() {
        // Id zero is a detection the tracker has not settled on. Following one
        // would mean following something that may not exist.
        let found = [thing(0, 0, (0.0, 0.0, 400.0, 400.0))];
        assert!(pick(&found, (200.0, 200.0)).is_none());
    }

    #[test]
    fn a_box_given_as_a_fraction_lands_in_the_same_place() {
        // The model this was written for turned out to emit fractions, and
        // reading them as pixels put every person in the top-left corner.
        let (bs, pixels) = output(4, 80, 32, &[(0, [320.0, 320.0, 160.0, 160.0], 0, 0.9, 1.0)]);
        let (bs2, fractions) =
            output(4, 80, 32, &[(0, [0.5, 0.5, 0.25, 0.25], 0, 0.9, 1.0)]);
        let (ps, protos) = protos(32, 160);

        let one = decode(&bs, &pixels, &ps, &protos, &input(), Thresholds::default());
        let two = decode(&bs2, &fractions, &ps, &protos, &input(), Thresholds::default());
        assert_eq!(one.len(), 1);
        assert_eq!(two.len(), 1);
        assert_eq!(one[0].found.bounds, two[0].found.bounds);
    }

    #[test]
    fn duplicate_guesses_at_the_same_thing_are_thinned() {
        // What the model really emits: the same object several times over.
        let (bs, boxes) = output(
            3,
            80,
            32,
            &[
                (0, [320.0, 320.0, 160.0, 160.0], 0, 0.90, 1.0),
                (1, [322.0, 318.0, 158.0, 162.0], 0, 0.85, 1.0),
                (2, [325.0, 322.0, 161.0, 159.0], 0, 0.80, 1.0),
            ],
        );
        let (ps, protos) = protos(32, 160);
        let found = decode(&bs, &boxes, &ps, &protos, &input(), Thresholds::default());
        assert_eq!(found.len(), 1, "the duplicates were kept");
        assert!((found[0].found.confidence - 0.90).abs() < 0.01, "the weaker one won");
    }

    #[test]
    fn two_things_apart_are_both_kept() {
        let (bs, boxes) = output(
            2,
            80,
            32,
            &[
                (0, [100.0, 100.0, 80.0, 80.0], 0, 0.9, 1.0),
                (1, [500.0, 500.0, 80.0, 80.0], 0, 0.9, 1.0),
            ],
        );
        let (ps, protos) = protos(32, 160);
        assert_eq!(decode(&bs, &boxes, &ps, &protos, &input(), Thresholds::default()).len(), 2);
    }

    #[test]
    fn a_negative_mix_is_outside_the_thing() {
        // Prototypes of one and a negative coefficient: the sigmoid lands
        // below the threshold, so nothing is inside.
        let (bs, boxes) = output(1, 80, 32, &[(0, [320.0, 320.0, 160.0, 160.0], 0, 0.9, -1.0)]);
        let (ps, protos) = protos(32, 160);
        let found = decode(&bs, &boxes, &ps, &protos, &input(), Thresholds::default());
        assert_eq!(found.len(), 1);
        assert!(
            found[0].mask.iter().all(|c| *c <= INSIDE),
            "everything was called inside"
        );
    }

    #[test]
    fn a_shape_that_does_not_add_up_is_ignored() {
        let (ps, protos) = protos(32, 160);
        let input = input();
        // Fields that leave no room for classes.
        assert!(decode(&[1, 36, 10], &[0.0; 360], &ps, &protos, &input, Thresholds::default()).is_empty());
        // Buffers shorter than the shape claims.
        assert!(decode(&[1, 116, 8400], &[0.0; 16], &ps, &protos, &input, Thresholds::default()).is_empty());
    }
}
