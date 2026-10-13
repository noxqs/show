//! How far away everything is, for every frame, beside playback.
//!
//! The same shape as the detection watcher — one thread, one session, a slot
//! of one — but without the thinning to a rate. Depth is a picture rather than
//! a readout, and a picture that updates once a second is a slideshow laid
//! over a film. So every frame is offered, and the worker takes whichever is
//! newest when it is free. On the card that is every frame; on the processor
//! it is a few a second, and the map simply lags the picture by that much.
//!
//! What comes back is relative, not metric: the model says what is nearer
//! than what, not by how many metres. That is what a colour map can show
//! anyway, and it is what the small model is good at.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::subject::Subject;
use super::timeline::Stamped;
use crate::video::Frame;

/// The longer side of what the model is shown, in pixels.
///
/// The size it was trained at. Smaller is faster and blurrier at the edges of
/// things; the patch size is fourteen, so both sides are rounded to that.
const LONG_SIDE: u32 = 518;
const PATCH: u32 = 14;

/// What the model was trained on, per channel: ImageNet's statistics.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// How quickly the brightness scale follows the picture.
///
/// Each map is stretched to fill the colour range, and stretched afresh every
/// frame the colours pulse as things enter and leave the shot. Following the
/// range rather than jumping to it keeps near things the same colour from one
/// frame to the next.
const FOLLOW: f32 = 0.2;

/// One map: a byte per cell, 255 nearest, 0 furthest.
pub struct Map {
    pub cells: Vec<u8>,
    pub size: (u32, u32),
}

struct Shared {
    pending: Mutex<Option<Stamped<Subject>>>,
    latest: Mutex<Option<Stamped<Map>>>,
    /// The last inference, in microseconds, for the readout.
    last_us: AtomicU64,
    examined: AtomicU64,
    skipped: AtomicU64,
}

/// Depth estimation running beside the picture.
pub struct Depth {
    shared: Arc<Shared>,
    _stop: Arc<()>,
    failure: Arc<Mutex<Option<String>>>,
}

impl Depth {
    /// Start the worker. The model opens on it, not here: the caller is the
    /// display, and loading a hundred megabytes onto the card is not quick.
    pub fn start() -> Depth {
        let shared = Arc::new(Shared {
            pending: Mutex::new(None),
            latest: Mutex::new(None),
            last_us: AtomicU64::new(0),
            examined: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
        });
        let stop = Arc::new(());
        let failure = Arc::new(Mutex::new(None::<String>));

        let worker = shared.clone();
        let watch = Arc::downgrade(&stop);
        let said = failure.clone();
        let spawned = std::thread::Builder::new()
            .name("ml-depth".into())
            .spawn(move || run(worker, watch, said));
        if let Err(e) = spawned {
            *failure.lock().expect("failure lock") = Some(format!("{e}"));
        }
        Depth { shared, _stop: stop, failure }
    }

    /// Offer a frame. Never blocks and never copies the picture; a frame
    /// offered while the last one is still waiting replaces it.
    pub fn offer(&self, frame: &Frame, generation: u64) {
        let Ok(mut slot) = self.shared.pending.try_lock() else {
            self.shared.skipped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if slot.is_some() {
            self.shared.skipped.fetch_add(1, Ordering::Relaxed);
        }
        *slot = Some(Stamped { at: frame.at, generation, what: Subject::Frame(frame.clone_ref()) });
    }

    /// Offer a still, read again from disk by the worker.
    pub fn offer_picture(&self, path: &std::path::Path, generation: u64) {
        let Ok(mut slot) = self.shared.pending.try_lock() else { return };
        *slot = Some(Stamped {
            at: Duration::ZERO,
            generation,
            what: Subject::Picture(path.to_path_buf()),
        });
    }

    /// The newest map since the last call, if there is one.
    pub fn collect(&self) -> Option<Stamped<Map>> {
        self.shared.latest.lock().expect("latest lock").take()
    }

    /// How long one look takes, and how many frames went by unlooked at.
    pub fn stats(&self) -> (Duration, u64, u64) {
        (
            Duration::from_micros(self.shared.last_us.load(Ordering::Relaxed)),
            self.shared.examined.load(Ordering::Relaxed),
            self.shared.skipped.load(Ordering::Relaxed),
        )
    }

    pub fn failed(&self) -> Option<String> {
        self.failure.lock().expect("failure lock").clone()
    }
}

/// The model's input size for a picture: the long side at [`LONG_SIDE`], the
/// short one in proportion, both a whole number of patches.
fn input_size(source: (u32, u32)) -> (u32, u32) {
    let (w, h) = (source.0.max(1) as f32, source.1.max(1) as f32);
    let scale = LONG_SIDE as f32 / w.max(h);
    let round = |v: f32| (((v / PATCH as f32).round() as u32).max(1)) * PATCH;
    (round(w * scale), round(h * scale))
}

/// Fill `out` with the frame at `size`, planar RGB, normalised as the model
/// was trained.
///
/// Squashed rather than letterboxed: the map is drawn over the whole picture,
/// so it should cover the whole picture, and rounding to patches changes the
/// aspect by a percent or two at most.
fn prepare(frame: &Frame, size: (u32, u32), out: &mut Vec<f32>) {
    let (w, h) = (size.0 as usize, size.1 as usize);
    let plane = w * h;
    out.clear();
    out.resize(3 * plane, 0.0);
    let (sw, sh) = (frame.width as f32, frame.height as f32);
    if sw <= 0.0 || sh <= 0.0 {
        return;
    }
    let (rows, offsets) = frame.colour.matrix();
    let (xs, ys) = (sw / w as f32, sh / h as f32);
    let stride = frame.width as usize;

    for y in 0..h {
        let sy = ((y as f32 + 0.5) * ys - 0.5).clamp(0.0, sh - 1.0);
        let (y0, fy) = (sy.floor() as usize, sy.fract());
        let y1 = (y0 + 1).min(frame.height as usize - 1);
        let cy = ((sy * 0.5) as usize).min(frame.chroma.1 as usize - 1);

        for x in 0..w {
            let sx = ((x as f32 + 0.5) * xs - 0.5).clamp(0.0, sw - 1.0);
            let (x0, fx) = (sx.floor() as usize, sx.fract());
            let x1 = (x0 + 1).min(frame.width as usize - 1);

            let luma = |i: usize| frame.planes[0][i] as f32 / 255.0;
            let top = luma(y0 * stride + x0) * (1.0 - fx) + luma(y0 * stride + x1) * fx;
            let bottom = luma(y1 * stride + x0) * (1.0 - fx) + luma(y1 * stride + x1) * fx;
            let luma = top * (1.0 - fy) + bottom * fy;

            let cx = ((sx * 0.5) as usize).min(frame.chroma.0 as usize - 1);
            let ci = cy * frame.chroma.0 as usize + cx;
            let cb = frame.planes[1][ci] as f32 / 255.0;
            let cr = frame.planes[2][ci] as f32 / 255.0;

            let ycc = [luma - offsets[0], cb - offsets[1], cr - offsets[1]];
            let at = y * w + x;
            for (c, m) in rows.iter().enumerate() {
                let v = (m[0] * ycc[0] + m[1] * ycc[1] + m[2] * ycc[2]).clamp(0.0, 1.0);
                out[c * plane + at] = (v - MEAN[c]) / STD[c];
            }
        }
    }
}

/// The range a map is stretched over, followed from frame to frame.
///
/// The ends are percentiles rather than the extremes: one hot pixel at the
/// edge of the frame would otherwise squash everything else into a band of
/// two or three colours.
struct Range {
    low: f32,
    high: f32,
    generation: Option<u64>,
}

impl Range {
    fn new() -> Range {
        Range { low: 0.0, high: 1.0, generation: None }
    }

    /// Fold this map's range in, and turn the map into bytes against it.
    fn quantise(&mut self, values: &[f32], generation: u64, out: &mut Vec<u8>) {
        let (low, high) = percentiles(values);
        if self.generation != Some(generation) {
            // A new film or a seek: nothing to follow, so take it as it is.
            (self.low, self.high) = (low, high);
            self.generation = Some(generation);
        } else {
            self.low += (low - self.low) * FOLLOW;
            self.high += (high - self.high) * FOLLOW;
        }
        let span = (self.high - self.low).max(1e-6);
        out.clear();
        out.extend(values.iter().map(|v| (((v - self.low) / span).clamp(0.0, 1.0) * 255.0).round() as u8));
    }
}

/// The second and ninety-eighth percentiles, from a sample of the values.
fn percentiles(values: &[f32]) -> (f32, f32) {
    // A few thousand values say as much about the range as all of them.
    let step = (values.len() / 4096).max(1);
    let mut sample: Vec<f32> = values.iter().step_by(step).copied().filter(|v| v.is_finite()).collect();
    if sample.is_empty() {
        return (0.0, 1.0);
    }
    let last = sample.len() - 1;
    let at = |p: f32| ((last as f32 * p).round() as usize).min(last);
    let lo = at(0.02);
    let hi = at(0.98);
    let low = *sample.select_nth_unstable_by(lo, f32::total_cmp).1;
    let high = *sample.select_nth_unstable_by(hi, f32::total_cmp).1;
    (low, high)
}

#[cfg(feature = "ml")]
fn run(shared: Arc<Shared>, stop: std::sync::Weak<()>, failure: Arc<Mutex<Option<String>>>) {
    let mut model = match with_ort::Model::open() {
        Ok(model) => model,
        Err(e) => {
            *failure.lock().expect("failure lock") = Some(format!("{e:#}"));
            return;
        }
    };
    let mut input = Vec::new();
    let mut range = Range::new();

    loop {
        if stop.upgrade().is_none() {
            return;
        }
        let taken = shared.pending.lock().expect("pending lock").take();
        let Some(job) = taken else {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        };
        let Some(picture) = job.what.frame() else { continue };

        let started = std::time::Instant::now();
        let size = input_size((picture.width, picture.height));
        prepare(&picture, size, &mut input);
        let (values, out) = match model.infer(&input, size) {
            Ok(result) => result,
            Err(e) => {
                *failure.lock().expect("failure lock") = Some(format!("{e:#}"));
                return;
            }
        };
        let mut cells = Vec::with_capacity(values.len());
        range.quantise(&values, job.generation, &mut cells);
        shared.last_us.store(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        shared.examined.fetch_add(1, Ordering::Relaxed);

        *shared.latest.lock().expect("latest lock") = Some(Stamped {
            at: job.at,
            generation: job.generation,
            what: Map { cells, size: out },
        });
    }
}

#[cfg(not(feature = "ml"))]
fn run(_shared: Arc<Shared>, _stop: std::sync::Weak<()>, failure: Arc<Mutex<Option<String>>>) {
    *failure.lock().expect("failure lock") =
        Some("this build has no inference; rebuild with `--features ml`".to_string());
}

#[cfg(feature = "ml")]
mod with_ort {
    use anyhow::{Context, Result};

    use crate::ml::catalogue::{Capability, spec};

    /// The session, held open and never dropped, for the reason
    /// `vision::Detector` gives: dropping a session from a `dlopen`ed runtime
    /// crashes the process at exit.
    pub struct Model {
        session: std::mem::ManuallyDrop<ort::session::Session>,
        input: String,
        output: String,
    }

    impl Model {
        pub fn open() -> Result<Model> {
            let spec = spec(Capability::Depth);
            let dir = crate::ml::store::directory(spec).context("no data directory")?;
            let path = dir.join(spec.files[0].name);
            anyhow::ensure!(
                path.is_file(),
                "depth is not installed — press F8, or run `show --plugins install {}`",
                Capability::Depth.id()
            );
            // A quarter of the cores, as detection takes: on the processor
            // this is the slow path, and it must not starve the decoder.
            let threads = std::thread::available_parallelism()
                .map(|n| (n.get() / 4).max(1))
                .unwrap_or(1);
            let (session, _) = crate::ml::runtime::commit(&path, threads)?;
            let input = session
                .inputs()
                .first()
                .map(|i| i.name().to_string())
                .context("the model declares no input")?;
            let output = session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .context("the model declares no output")?;
            Ok(Model { session: std::mem::ManuallyDrop::new(session), input, output })
        }

        /// Run one prepared picture. Returns the raw map and its size, which
        /// is the input's size for this model but is read from the output
        /// rather than assumed.
        pub fn infer(&mut self, data: &[f32], size: (u32, u32)) -> Result<(Vec<f32>, (u32, u32))> {
            let (w, h) = (size.0 as usize, size.1 as usize);
            let tensor = ort::value::Tensor::from_array((
                [1usize, 3, h, w],
                data.to_vec().into_boxed_slice(),
            ))
            .map_err(|e| anyhow::anyhow!("building the input tensor: {e}"))?;
            let outputs = self
                .session
                .run(ort::inputs![self.input.as_str() => tensor])
                .map_err(|e| anyhow::anyhow!("running the model: {e}"))?;
            let value = outputs
                .get(self.output.as_str())
                .with_context(|| format!("the model has no `{}` output", self.output))?;
            let (shape, values) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("reading the depth: {e}"))?;
            // [1, h, w] or [1, 1, h, w]: the last two are the map.
            let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            let [.., oh, ow] = dims[..] else { anyhow::bail!("a depth map of shape {dims:?}") };
            anyhow::ensure!(values.len() >= oh * ow, "a depth map shorter than its shape");
            Ok((values[..oh * ow].to_vec(), (ow as u32, oh as u32)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resident memory, in megabytes.
    fn resident_mb() -> u64 {
        let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
        let pages: u64 = statm.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
        pages * 4096 / 1_000_000
    }

    /// Memory stays flat across many looks. Ignored: needs the model.
    #[test]
    #[ignore]
    fn depth_memory() {
        let path = std::path::Path::new("testdata/corpus/base.jpg");
        let picture = Subject::Picture(path.to_path_buf()).frame().expect("decodes");
        let depth = Depth::start();
        let mut seen = 0;
        let mut marks = Vec::new();
        while seen < 1000 {
            if let Some(why) = depth.failed() {
                panic!("{why}");
            }
            depth.offer(&picture, 1);
            if depth.collect().is_some() {
                seen += 1;
                if seen % 200 == 0 {
                    marks.push(resident_mb());
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        println!("resident MB every 200 looks: {marks:?}");
    }

    /// End to end on the real model, timed. Ignored: it needs the runtime and
    /// the model installed. `cargo test --release -- --ignored depth_speed --nocapture`
    #[test]
    #[ignore]
    fn depth_speed() {
        let path = std::path::Path::new("testdata/corpus/base.jpg");
        let depth = Depth::start();
        let wait = |depth: &Depth| {
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            loop {
                if let Some(why) = depth.failed() {
                    panic!("depth failed: {why}");
                }
                if let Some(map) = depth.collect() {
                    return map;
                }
                assert!(std::time::Instant::now() < deadline, "no map within a minute");
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        // The first includes loading and warming the session.
        depth.offer_picture(path, 1);
        let first = wait(&depth);
        let picture = Subject::Picture(path.to_path_buf()).frame().expect("decodes");
        let mut times = Vec::new();
        for _ in 0..30 {
            depth.offer(&picture, 1);
            wait(&depth);
            times.push(depth.stats().0);
        }
        times.sort();
        let spread: Vec<u8> = {
            let mut c = first.what.cells.clone();
            c.sort_unstable();
            vec![c[0], c[c.len() / 2], c[c.len() - 1]]
        };
        let deciles: Vec<u8> = {
            let mut c = first.what.cells.clone();
            c.sort_unstable();
            (0..=10).map(|i| c[(c.len() - 1) * i / 10]).collect()
        };
        println!("deciles {deciles:?}");
        println!(
            "{}x{} picture, map {:?}, min/median/max {:?}; per look: best {:?}, median {:?}, worst {:?}",
            picture.width, picture.height, first.what.size, spread, times[0], times[15], times[29]
        );
        assert!(spread[2] > spread[0], "a flat map");
    }

    #[test]
    fn the_input_is_whole_patches_in_the_picture_s_proportion() {
        let (w, h) = input_size((1920, 1080));
        assert_eq!(w % PATCH, 0);
        assert_eq!(h % PATCH, 0);
        assert_eq!(w, 518);
        assert!((w as f32 / h as f32 - 16.0 / 9.0).abs() < 0.05, "{w}x{h}");
        let (w, h) = input_size((1080, 1920));
        assert_eq!(h, 518);
        assert!(w < h);
    }

    #[test]
    fn one_bright_pixel_does_not_flatten_the_rest() {
        let mut values: Vec<f32> = (0..10_000).map(|i| i as f32 / 10_000.0).collect();
        values[0] = 1_000.0;
        let mut range = Range::new();
        let mut out = Vec::new();
        range.quantise(&values, 1, &mut out);
        // The middle of the ramp lands in the middle of the colours.
        assert!((out[5_000] as i32 - 128).abs() < 10, "{}", out[5_000]);
    }

    #[test]
    fn the_range_follows_rather_than_jumps_within_a_film() {
        let mut range = Range::new();
        let mut out = Vec::new();
        let ramp: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        range.quantise(&ramp, 1, &mut out);
        let wider: Vec<f32> = ramp.iter().map(|v| v * 2.0).collect();
        range.quantise(&wider, 1, &mut out);
        assert!(range.high < 1900.0, "jumped straight to the new range");
        // A new generation starts afresh.
        range.quantise(&wider, 2, &mut out);
        assert!(range.high > 1900.0);
    }
}
