//! Segmentation from a phrase rather than from a list of classes.
//!
//! The other models here know eighty things and say which of them they can
//! see. This one is told what to look for — "a red car", "the dog" — and finds
//! every instance of it, whether or not anything was ever trained to name it.
//! That is also why it answers the question the duplicate-overlap slider was
//! invented to work around: finding all four people in a crowded shot is what
//! it is for, rather than something to be recovered by loosening a heuristic.
//!
//! Three graphs, and they are not run at the same rate, which is the whole
//! shape of this file:
//!
//! * the **image encoder** is the expensive one, run once per picture;
//! * the **language encoder** is run once per *phrase*, and its answer is kept
//!   — the prompt changes when somebody types, not when the film moves;
//! * the **decoder** is cheap and runs per picture-and-phrase.
//!
//! The calling convention was read out of the exporter's own inference code
//! rather than guessed at, after guessing produced nothing at all six times
//! running. Two pieces of it are counter-intuitive enough to be worth naming
//! here: the prompt is padded with **zero** and not with CLIP's end token, and
//! the empty geometric slot is marked with `box_masks` **true** — false means
//! "this box is real", which anchors every query to a box at the origin.

use anyhow::{Context, Result};

use super::segment::Segmented;
use super::vision::Found;

/// The square the image encoder declares, and what its exporter resizes to.
pub const INPUT: usize = 1008;

/// The class given to everything this model finds.
///
/// It has no class list — it found what it was asked for — so its results
/// share the cut-out's unnamed class rather than pretending to a name from a
/// vocabulary that had no say in it.
pub use super::cutout::UNNAMED;

/// How sure the model must be before a thing is kept.
///
/// The graph hands back every query it scored; this is the exporter's own
/// default for what counts, and it is high because the scores are decisive —
/// on a clear subject they come back above 0.9 rather than hovering.
pub const KEEP: f32 = 0.5;

#[cfg(feature = "ml")]
pub use with_ort::Sam3;

#[cfg(feature = "ml")]
mod with_ort {
    use super::*;
    use crate::ml::catalogue::{Capability, spec};

    /// The three graphs, and the tokenizer that feeds the middle one.
    ///
    /// Sessions are never dropped, for the reason `vision.rs` sets out at
    /// length: destroying one with a `dlopen`ed runtime leaves the process to
    /// segfault at exit. One of these per process, opened when it is switched
    /// on.
    pub struct Sam3 {
        image: std::mem::ManuallyDrop<ort::session::Session>,
        language: std::mem::ManuallyDrop<ort::session::Session>,
        decoder: std::mem::ManuallyDrop<ort::session::Session>,
        tokens: crate::ml::clip::Tokenizer,
        /// The phrase last encoded, and what it encoded to. Keeping it is what
        /// makes a second frame cost the image encoder and nothing else.
        cached: Option<(String, Vec<bool>, Vec<f32>)>,
    }

    impl Sam3 {
        pub fn open() -> Result<Sam3> {
            crate::ml::runtime::prepare()?;
            let spec = spec(Capability::TextSegment);
            let dir = crate::ml::store::directory(spec).context("no data directory")?;
            let graph = |name: &str| dir.join(name);
            anyhow::ensure!(
                graph("sam3_image_encoder.onnx").is_file(),
                "{} is not installed — press F8, or run `show --plugins install sam3`",
                spec.name.to_lowercase()
            );

            // A quarter of the cores, as the other models take: this runs
            // beside a decoder that needs the rest.
            let threads = std::thread::available_parallelism()
                .map(|n| (n.get() / 4).max(1))
                .unwrap_or(1);
            let tokens = crate::ml::clip::Tokenizer::load(&dir.join("merges.txt"))
                .context("reading the tokenizer's merges")?;

            // The decoder first, and it is the smallest: whether the card can
            // run *these* graphs is decided once, on the one that costs least
            // to be refused. The other two then go straight where they are
            // going. Asking three times would print the runtime's page-long
            // complaint three times on every launch.
            let (decoder, on_card) =
                crate::ml::runtime::commit(&graph("sam3_decoder.onnx"), threads)?;
            let open = |path: std::path::PathBuf| -> Result<ort::session::Session> {
                if on_card {
                    crate::ml::runtime::commit(&path, threads).map(|(session, _)| session)
                } else {
                    crate::ml::runtime::commit_on_cpu(&path, threads)
                }
            };

            Ok(Sam3 {
                image: std::mem::ManuallyDrop::new(open(graph("sam3_image_encoder.onnx"))?),
                language: std::mem::ManuallyDrop::new(open(graph("sam3_language_encoder.onnx"))?),
                decoder: std::mem::ManuallyDrop::new(decoder),
                tokens,
                cached: None,
            })
        }

        /// Encode a picture. The expensive half, and the only part that has to
        /// happen again when the picture changes.
        pub fn look(&mut self, frame: &crate::video::Frame) -> Result<Embedding> {
            let pixels = crate::ml::prep::squash_u8(frame, INPUT);
            let tensor =
                ort::value::Tensor::from_array(([3usize, INPUT, INPUT], pixels.into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("building the image tensor: {e}"))?;
            let outputs = self
                .image
                .run(ort::inputs!["image" => tensor])
                .map_err(|e| anyhow::anyhow!("running the image encoder: {e}"))?;

            let mut taken = Vec::new();
            for name in
                ["vision_pos_enc_2", "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"]
            {
                let value = outputs
                    .get(name)
                    .with_context(|| format!("the image encoder has no `{name}` output"))?;
                let (shape, values) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("reading `{name}`: {e}"))?;
                taken.push((
                    shape.iter().map(|d| *d as usize).collect::<Vec<usize>>(),
                    values.to_vec(),
                ));
            }
            Ok(Embedding { parts: taken, source: (frame.width, frame.height) })
        }

        /// Encode a phrase, reusing the last answer when it has not changed.
        ///
        /// Worth the branch: the language encoder carries 1.6 GB of weights,
        /// and a film asked the same question of every frame would otherwise
        /// pay for it on every one of them.
        fn phrase(&mut self, text: &str) -> Result<(&[bool], &[f32])> {
            let fresh = self.cached.as_ref().is_none_or(|(had, _, _)| had != text);
            if fresh {
                let ids = self.tokens.encode(text, crate::ml::clip::CONTEXT);
                let tensor = ort::value::Tensor::from_array((
                    [1usize, crate::ml::clip::CONTEXT],
                    ids.into_boxed_slice(),
                ))
                .map_err(|e| anyhow::anyhow!("building the token tensor: {e}"))?;
                let outputs = self
                    .language
                    .run(ort::inputs!["tokens" => tensor])
                    .map_err(|e| anyhow::anyhow!("running the language encoder: {e}"))?;

                let mask = outputs
                    .get("text_attention_mask")
                    .context("the language encoder has no `text_attention_mask` output")?
                    .try_extract_tensor::<bool>()
                    .map_err(|e| anyhow::anyhow!("reading the attention mask: {e}"))?
                    .1
                    .to_vec();
                // `text_memory`, not `text_embeds`: the decoder's
                // `language_features` is [32, 1, 256] and the embeds are 1024
                // wide. The shapes settle which is meant.
                let memory = outputs
                    .get("text_memory")
                    .context("the language encoder has no `text_memory` output")?
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("reading the text memory: {e}"))?
                    .1
                    .to_vec();
                self.cached = Some((text.to_string(), mask, memory));
            }
            let (_, mask, memory) = self.cached.as_ref().expect("just set");
            Ok((mask, memory))
        }

        /// Ask an encoded picture what it holds that matches `text`.
        pub fn find(&mut self, picture: &Embedding, text: &str, keep: f32) -> Result<Vec<Segmented>> {
            let (source_w, source_h) = picture.source;
            let (mask, memory) = self.phrase(text)?;
            let mask = mask.to_vec();
            let memory = memory.to_vec();

            let tensor = |shape: &[usize], values: Vec<f32>| {
                ort::value::Tensor::from_array((shape.to_vec(), values.into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("building a decoder input: {e}"))
            };
            let named = |i: usize| -> Result<ort::value::Tensor<f32>> {
                let (shape, values) = &picture.parts[i];
                tensor(shape, values.clone())
            };

            let language_mask =
                ort::value::Tensor::from_array(([1usize, mask.len()], mask.into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("building the language mask: {e}"))?;
            let language_features = tensor(&[crate::ml::clip::CONTEXT, 1, 256], memory)?;

            // The empty geometric slot. `true` is what marks it unused —
            // `false` declares a real box, and a real box at the origin is
            // what silently returned nothing at all for an afternoon.
            let box_coords = tensor(&[1, 1, 4], vec![0.0; 4])?;
            let box_labels =
                ort::value::Tensor::from_array(([1usize, 1], vec![1i64].into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("building the box labels: {e}"))?;
            let box_masks =
                ort::value::Tensor::from_array(([1usize, 1], vec![true].into_boxed_slice()))
                    .map_err(|e| anyhow::anyhow!("building the box masks: {e}"))?;

            let height = ort::value::Tensor::from_array((
                Vec::<usize>::new(),
                vec![source_h as i64].into_boxed_slice(),
            ))
            .map_err(|e| anyhow::anyhow!("building the height: {e}"))?;
            let width = ort::value::Tensor::from_array((
                Vec::<usize>::new(),
                vec![source_w as i64].into_boxed_slice(),
            ))
            .map_err(|e| anyhow::anyhow!("building the width: {e}"))?;

            let outputs = self
                .decoder
                .run(ort::inputs![
                    "original_height" => height,
                    "original_width" => width,
                    "vision_pos_enc_2" => named(0)?,
                    "backbone_fpn_0" => named(1)?,
                    "backbone_fpn_1" => named(2)?,
                    "backbone_fpn_2" => named(3)?,
                    "language_mask" => language_mask,
                    "language_features" => language_features,
                    "box_coords" => box_coords,
                    "box_labels" => box_labels,
                    "box_masks" => box_masks,
                ])
                .map_err(|e| anyhow::anyhow!("running the decoder: {e}"))?;

            let (_, boxes) = outputs
                .get("boxes")
                .context("the decoder has no `boxes` output")?
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("reading the boxes: {e}"))?;
            let (_, scores) = outputs
                .get("scores")
                .context("the decoder has no `scores` output")?
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("reading the scores: {e}"))?;
            let (mask_shape, masks) = outputs
                .get("masks")
                .context("the decoder has no `masks` output")?
                .try_extract_tensor::<bool>()
                .map_err(|e| anyhow::anyhow!("reading the masks: {e}"))?;
            let mask_shape: Vec<usize> = mask_shape.iter().map(|d| *d as usize).collect();

            Ok(assemble(boxes, scores, &mask_shape, masks, (source_w, source_h), keep))
        }
    }
}

/// One picture, encoded. Held so a second phrase costs only the decoder.
pub struct Embedding {
    /// `vision_pos_enc_2` then `backbone_fpn_0..2`, each with its shape.
    parts: Vec<(Vec<usize>, Vec<f32>)>,
    source: (u32, u32),
}

impl Embedding {
    pub fn source(&self) -> (u32, u32) {
        self.source
    }
}

/// Turn the decoder's three tensors into the shape the rest of the program
/// already draws.
///
/// The masks arrive as booleans at the picture's own resolution, which is a
/// different thing from what every other model here produces — a small grid of
/// logits. They are reduced to that form rather than a second path being added
/// downstream: the grid is what `labels.rs` samples, what tracking compares
/// and what the shader is handed, and one shape through all of it is worth a
/// pass over the mask.
fn assemble(
    boxes: &[f32],
    scores: &[f32],
    mask_shape: &[usize],
    masks: &[bool],
    source: (u32, u32),
    keep: f32,
) -> Vec<Segmented> {
    // [N, 1, H, W], and the middle dimension is always one.
    let (count, height, width) = match mask_shape {
        [n, _, h, w] => (*n, *h, *w),
        _ => return Vec::new(),
    };
    if count == 0 || height == 0 || width == 0 {
        return Vec::new();
    }

    // The grid every other model's outlines live on, so these land on it too.
    let grid = crate::ml::labels::Grid::for_picture(source).size;
    let (gw, gh) = (grid.0 as usize, grid.1 as usize);
    let plane = height * width;

    let mut out = Vec::new();
    for i in 0..count.min(scores.len()) {
        if scores[i] < keep {
            continue;
        }
        let b = boxes.get(i * 4..i * 4 + 4).unwrap_or(&[0.0; 4]);
        let mask = &masks[i * plane..(i + 1) * plane];

        // Sampled rather than averaged: the model has already decided, and
        // what is wanted here is the same decision at a coarser spacing.
        // Positive means inside, to match every other mask in the program.
        let mut cells = vec![-1.0f32; gw * gh];
        for y in 0..gh {
            let sy = ((y as f32 + 0.5) / gh as f32 * height as f32) as usize;
            let sy = sy.min(height - 1);
            for x in 0..gw {
                let sx = ((x as f32 + 0.5) / gw as f32 * width as f32) as usize;
                let sx = sx.min(width - 1);
                if mask[sy * width + sx] {
                    cells[y * gw + x] = 1.0;
                }
            }
        }

        out.push(Segmented {
            found: Found {
                class: UNNAMED,
                confidence: scores[i],
                bounds: (b[0], b[1], b[2], b[3]),
            },
            id: 0,
            mask: cells,
            origin: (0, 0),
            size: (gw, gh),
            grid: (gw, gh),
        });
    }
    // Most confident first, as every other model returns them.
    out.sort_by(|a, b| b.found.confidence.total_cmp(&a.found.confidence));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_of(w: usize, h: usize, f: impl Fn(usize, usize) -> bool) -> Vec<bool> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                out.push(f(x, y));
            }
        }
        out
    }

    #[test]
    fn a_confident_thing_survives_and_a_doubtful_one_does_not() {
        let masks = [mask_of(8, 8, |_, _| true), mask_of(8, 8, |_, _| true)].concat();
        let found = assemble(
            &[0.0, 0.0, 8.0, 8.0, 0.0, 0.0, 8.0, 8.0],
            &[0.9, 0.1],
            &[2, 1, 8, 8],
            &masks,
            (8, 8),
            KEEP,
        );
        assert_eq!(found.len(), 1, "the doubtful one should have been dropped");
        assert!((found[0].found.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn the_outline_lands_where_the_mask_was() {
        // A mask covering the left half only. After reduction the left of the
        // grid must be inside and the right outside, or every drawn outline
        // is in the wrong place.
        let (w, h) = (64, 64);
        let masks = mask_of(w, h, |x, _| x < w / 2);
        let found = assemble(&[0.0, 0.0, 32.0, 64.0], &[0.9], &[1, 1, h, w], &masks, (w as u32, h as u32), KEEP);
        assert_eq!(found.len(), 1);
        let item = &found[0];
        assert!(item.has_outline(), "a mask this size should count as an outline");
        let (gw, gh) = item.size;
        let at = |x: usize, y: usize| item.mask[y * gw + x];
        assert!(at(gw / 4, gh / 2) > 0.0, "the left should be inside");
        assert!(at(gw * 3 / 4, gh / 2) < 0.0, "the right should be outside");
    }

    #[test]
    fn nothing_found_is_an_empty_list_rather_than_a_panic() {
        assert!(assemble(&[], &[], &[0, 1, 8, 8], &[], (8, 8), KEEP).is_empty());
        assert!(assemble(&[], &[], &[], &[], (8, 8), KEEP).is_empty());
    }

    #[test]
    fn results_come_back_most_confident_first() {
        let masks = mask_of(4, 4 * 3, |_, _| true);
        let found = assemble(
            &[0.0; 12],
            &[0.6, 0.95, 0.8],
            &[3, 1, 4, 4],
            &masks,
            (4, 4),
            KEEP,
        );
        let scores: Vec<f32> = found.iter().map(|f| f.found.confidence).collect();
        assert_eq!(scores, vec![0.95, 0.8, 0.6]);
    }
}
