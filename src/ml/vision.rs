//! Running a detector, and reading what it says.
//!
//! One session, held open, fed one prepared image at a time. Every call into
//! `ort` is in this file and `runtime.rs`, so the crate being a release
//! candidate with an unstable API is a contained problem rather than a diffuse
//! one.

use anyhow::{Context, Result};

use super::prep::Input;

/// The eighty things the COCO-trained models know about.
pub const CLASSES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];

/// One thing found, in source-picture pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub class: usize,
    pub confidence: f32,
    /// Left, top, right, bottom.
    pub bounds: (f32, f32, f32, f32),
}

impl Found {
    pub fn label(&self) -> &'static str {
        CLASSES.get(self.class).copied().unwrap_or("thing")
    }
}

/// Below this a detection is noise. The models emit a long tail of very low
/// scores; drawing them would be drawing static.
pub const MIN_CONFIDENCE: f32 = 0.25;

/// What the models are asked to accept, as opposed to what they emit.
///
/// Both numbers were constants, which made them arguments nobody could reach:
/// the far-off and the half-hidden were dropped and the only way to see them
/// was to edit and rebuild. They travel together because they answer the same
/// question — what counts as a thing — and are set together on the F8 page.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Thresholds {
    /// How sure the model must be before a thing is drawn at all.
    pub confidence: f32,
    /// How much two of a kind may overlap before the weaker is taken for the
    /// model's own repeat rather than a second thing.
    pub duplicate: f32,
}

impl Default for Thresholds {
    fn default() -> Thresholds {
        Thresholds { confidence: MIN_CONFIDENCE, duplicate: crate::ml::segment::DUPLICATE }
    }
}

impl Thresholds {
    /// Clamped to what the models can be asked for. Zero confidence is every
    /// anchor the model has, which is tens of thousands of boxes of static.
    pub fn new(confidence: f32, duplicate: f32) -> Thresholds {
        Thresholds {
            confidence: confidence.clamp(0.05, 0.95),
            duplicate: duplicate.clamp(0.1, 0.95),
        }
    }
}

#[cfg(feature = "ml")]
pub use with_ort::Detector;

#[cfg(feature = "ml")]
mod with_ort {
    use super::*;
    use crate::ml::catalogue::{Capability, spec};

    /// A loaded model, held open.
    ///
    /// The session is deliberately never dropped. Destroying one with a
    /// `dlopen`ed ONNX Runtime leaves the process to segfault inside the
    /// library at exit — measured, not assumed: the drop itself completes and
    /// execution carries on normally, and the crash arrives later during
    /// teardown. Not destroying it avoids the crash entirely.
    ///
    /// The cost is one leaked session per detector, which is the model's
    /// memory until the process ends. That is affordable precisely because a
    /// detector is meant to be long-lived — one per process, opened when
    /// inference is switched on. Opening and dropping them in a loop would
    /// leak in earnest, so do not.
    pub struct Detector {
        session: std::mem::ManuallyDrop<ort::session::Session>,
        /// What the model wants its input called.
        input: String,
    }

    impl Detector {
        /// Open a model, and warm it up.
        pub fn open_for(which: Capability) -> Result<Detector> {
            crate::ml::runtime::prepare()?;
            let spec = spec(which);
            let dir = crate::ml::store::directory(spec).context("no data directory")?;
            let path = dir.join(spec.files[0].name);
            anyhow::ensure!(
                path.is_file(),
                "{} is not installed — press F8, or run `show --plugins install {}`",
                spec.name.to_lowercase(),
                which.id()
            );

            // A quarter of the cores, matching what similarity.rs reserves and
            // for the same reason: taking every core would stall the decoding
            // this is supposed to run alongside.
            let threads = std::thread::available_parallelism()
                .map(|n| (n.get() / 4).max(1))
                .unwrap_or(1);

            // `ort`'s errors carry a non-Send payload, so they are turned
            // into messages here rather than propagated. Nothing downstream
            // wants the type; it wants to know what went wrong.
            // `ort`'s errors carry a non-Send payload, so they are turned into
            // messages here rather than propagated. Nothing downstream wants
            // the type; it wants to know what went wrong.
            let mut builder = crate::ml::runtime::session(threads)?;
            let session = crate::vram::charging(&path, || builder.commit_from_file(&path))
                .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))?;

            let input = session
                .inputs()
                .first()
                .map(|i| i.name().to_string())
                .context("the model declares no input")?;
            Ok(Detector { session: std::mem::ManuallyDrop::new(session), input })
        }

        /// What the model calls its tensors, for reporting.
        pub fn describe(&self) -> String {
            let inputs: Vec<&str> = self.session.inputs().iter().map(|o| o.name()).collect();
            let outputs: Vec<&str> = self.session.outputs().iter().map(|o| o.name()).collect();
            format!("in [{}] out [{}]", inputs.join(", "), outputs.join(", "))
        }

        /// Run one prepared image, returning outlines when the model has them.
        ///
        /// Two families of model reach here and they agree about nothing —
        /// different input names, different output names, different layouts.
        /// Which one this is, is decided by what it declares rather than by
        /// what it is called, because the names have already proved
        /// misleading once.
        pub fn look(
            &mut self,
            input: &Input,
            limits: Thresholds,
        ) -> Result<Vec<crate::ml::segment::Segmented>> {
            let outputs = self.infer(input)?;
            let take = |name: &str| -> Result<(Vec<usize>, &[f32])> {
                let value = outputs
                    .get(name)
                    .with_context(|| format!("the model has no `{name}` output"))?;
                let (shape, values) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("reading `{name}`: {e}"))?;
                Ok((shape.iter().map(|d| *d as usize).collect(), values))
            };
            let (bs, boxes) = take("output0")?;
            let (ps, protos) = take("output1")?;
            Ok(crate::ml::segment::decode(&bs, boxes, &ps, protos, input, limits))
        }

        /// Run one prepared image.
        pub fn run(&mut self, input: &Input, limits: Thresholds) -> Result<Vec<Found>> {
            let outputs = self.infer(input)?;
            // Looked up by name rather than by position: the order two
            // tensors come back in is the exporter's business, and reading
            // boxes as scores produces confident nonsense rather than an
            // error.
            let take = |name: &str| -> Result<(Vec<usize>, &[f32])> {
                let value = outputs
                    .get(name)
                    .with_context(|| format!("the model has no `{name}` output"))?;
                let (shape, values) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("reading `{name}`: {e}"))?;
                Ok((shape.iter().map(|d| *d as usize).collect(), values))
            };
            let (shape, logits) = take("logits")?;
            let (_, boxes) = take("pred_boxes")?;
            Ok(decode(&shape, logits, boxes, input, limits))
        }

        /// Feed one prepared image through the session.
        fn infer(&mut self, input: &Input) -> Result<ort::session::SessionOutputs<'_>> {
            let size = input.size as usize;
            let tensor = ort::value::Tensor::from_array((
                [1usize, 3, size, size],
                input.data.clone().into_boxed_slice(),
            ))
            .map_err(|e| anyhow::anyhow!("building the input tensor: {e}"))?;

            let outputs = self
                .session
                .run(ort::inputs![self.input.as_str() => tensor])
                .map_err(|e| anyhow::anyhow!("running the model: {e}"))?;

            Ok(outputs)
        }
    }
}

/// Turn the model's two output tensors into boxes in the source picture.
///
/// The layout is the DETR family's, which this model uses despite its name:
/// a fixed number of *queries*, each with a score per class and a box. There
/// is no background class and no non-maximum suppression to do — a query
/// either scores high enough to be something or it does not.
///
/// Scores are logits, so they pass through a sigmoid. Boxes are centre, width
/// and height as fractions of the input square, which is the letterboxed
/// square rather than the picture — so they are turned into corners first and
/// mapped back through `Input::to_source`, which undoes the letterbox.
pub fn decode(
    shape: &[usize],
    logits: &[f32],
    boxes: &[f32],
    input: &Input,
    limits: Thresholds,
) -> Vec<Found> {
    let [_, queries, classes] = shape else { return Vec::new() };
    let (queries, classes) = (*queries, *classes);
    if logits.len() < queries * classes || boxes.len() < queries * 4 {
        return Vec::new();
    }
    let square = input.size as f32;

    let mut found = Vec::new();
    for query in 0..queries {
        let row = &logits[query * classes..(query + 1) * classes];
        // The best class for this query, and how sure it is.
        let (class, logit) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, v)| (i, *v))
            .unwrap_or((0, f32::MIN));
        let confidence = 1.0 / (1.0 + (-logit).exp());
        if confidence < limits.confidence {
            continue;
        }

        let b = &boxes[query * 4..query * 4 + 4];
        let (cx, cy, w, h) = (b[0] * square, b[1] * square, b[2] * square, b[3] * square);
        let (x1, y1) = input.to_source(cx - w / 2.0, cy - h / 2.0);
        let (x2, y2) = input.to_source(cx + w / 2.0, cy + h / 2.0);
        found.push(Found {
            class,
            confidence,
            bounds: (x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2)),
        });
    }
    // Most confident first, so a caller taking the top few takes the best few.
    found.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> Input {
        // A 1280x640 picture letterboxed into 640: half scale, 160 of padding
        // above and below.
        let mut input = Input::new(640);
        input.scale = 2.0;
        input.pad = (0.0, 160.0);
        input
    }

    /// A logit that sigmoids to roughly `p`.
    fn logit(p: f32) -> f32 {
        (p / (1.0 - p)).ln()
    }

    #[test]
    fn a_confident_query_becomes_a_box_in_picture_coordinates() {
        let input = input();
        // One query, class 0, centred in the square and a quarter of it wide.
        let mut logits = vec![logit(0.01); 80];
        logits[0] = logit(0.9);
        let boxes = vec![0.5, 0.5, 0.25, 0.25];

        let found = decode(&[1, 1, 80], &logits, &boxes, &input, Thresholds::default());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].label(), "person");
        assert!((found[0].confidence - 0.9).abs() < 0.01, "{}", found[0].confidence);

        // Centre of the square is 320,320; a quarter is 160 wide, so the box
        // spans 240..400 in model space. Undoing the letterbox: x doubles,
        // y loses 160 first.
        let (x1, y1, x2, y2) = found[0].bounds;
        assert!((x1 - 480.0).abs() < 1.0, "x1 {x1}");
        assert!((x2 - 800.0).abs() < 1.0, "x2 {x2}");
        assert!((y1 - 160.0).abs() < 1.0, "y1 {y1}");
        assert!((y2 - 480.0).abs() < 1.0, "y2 {y2}");
    }

    #[test]
    fn unconfident_queries_are_dropped_rather_than_drawn() {
        let input = input();
        // What the model really emits: 300 queries, nearly all of them noise.
        let mut logits = vec![logit(0.02); 300 * 80];
        let mut boxes = vec![0.5f32; 300 * 4];
        // One real detection, at query 7.
        logits[7 * 80 + 2] = logit(0.8);
        boxes[7 * 4..7 * 4 + 4].copy_from_slice(&[0.5, 0.5, 0.1, 0.1]);

        let found = decode(&[1, 300, 80], &logits, &boxes, &input, Thresholds::default());
        assert_eq!(found.len(), 1, "the noise was drawn as boxes");
        assert_eq!(found[0].label(), "car");
    }

    #[test]
    fn the_best_class_wins_within_a_query() {
        let input = input();
        let mut logits = vec![logit(0.05); 80];
        logits[2] = logit(0.6);
        logits[16] = logit(0.85);
        let found =
            decode(&[1, 1, 80], &logits, &vec![0.5, 0.5, 0.2, 0.2], &input, Thresholds::default());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].label(), "dog", "the weaker class won");
    }

    #[test]
    fn results_are_most_confident_first() {
        let input = input();
        let mut logits = vec![logit(0.01); 3 * 80];
        logits[80 * 0] = logit(0.30);
        logits[80 * 1] = logit(0.90);
        logits[80 * 2] = logit(0.60);
        let boxes = vec![0.5f32; 12];
        let found = decode(&[1, 3, 80], &logits, &boxes, &input, Thresholds::default());
        let scores: Vec<i32> = found.iter().map(|f| (f.confidence * 100.0) as i32).collect();
        assert_eq!(scores, vec![90, 60, 30]);
    }

    #[test]
    fn a_truncated_output_is_ignored_rather_than_read_past_its_end() {
        let input = input();
        // Shapes that promise more than the buffers hold. Better to find
        // nothing than to index into whatever follows.
        assert!(decode(&[1, 300, 80], &[0.0; 16], &[0.0; 16], &input, Thresholds::default())
            .is_empty());
        assert!(decode(&[1, 25200, 85], &[0.0; 8], &[0.0; 8], &input, Thresholds::default())
            .is_empty());
    }

    #[test]
    fn every_class_index_has_a_name() {
        assert_eq!(CLASSES.len(), 80);
        assert_eq!(CLASSES[0], "person");
        assert_eq!(CLASSES[2], "car");
        let found = Found { class: 999, confidence: 1.0, bounds: (0.0, 0.0, 1.0, 1.0) };
        assert_eq!(found.label(), "thing");
    }
}
