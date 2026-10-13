//! Detection running beside playback, without disturbing it.
//!
//! One thread, one session, a queue of one. Frames are *offered* rather than
//! sent: if the thread is busy the frame is dropped and the next one taken
//! instead. That is the whole scheduling policy, and it is the right one —
//! this is a readout, and a readout that is one frame behind is invisible,
//! while a decoder made to wait for a readout is a stutter anybody can see.
//!
//! Cancellation is the `Arc<()>` / `Weak::upgrade` idiom already used by
//! `Player` and `BackFill`: drop the watcher and the thread notices between
//! frames and stops.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::video::Frame;

use super::catalogue::Capability;
use super::segment::Segmented;
use super::subject::Subject;
use super::timeline::Stamped;

/// How often to look, when nothing says otherwise.
///
/// Once a second. Detection is for knowing what is on screen, which changes
/// with the shot rather than with the frame — but following one thing through
/// a moving picture wants more, which is why the rate is a setting rather than
/// this constant.
const DEFAULT_HZ: u32 = 1;

/// One look at one picture.
///
/// The size is part of the result rather than assumed by the reader: a film's
/// frames are its own size, but a photograph may have been decoded smaller
/// than it is, and everything said about it is then in those pixels.
pub struct Seen {
    pub found: Vec<Segmented>,
    pub size: (u32, u32),
}

/// What the watcher has to say about itself, for the readout.
#[derive(Clone, Copy, Default)]
pub struct Stats {
    /// Frames actually run through the model.
    pub examined: u64,
    /// Frames offered while it was busy, and therefore skipped.
    pub skipped: u64,
    /// How long the last inference took.
    pub last: Duration,
}

struct Shared {
    /// Milliseconds between looks. Changed while running, so the setting takes
    /// effect on the next frame rather than the next film.
    every_ms: AtomicU64,
    /// The frame waiting to be looked at, if any. One slot, deliberately:
    /// holding a queue would mean looking at frames that have already gone.
    pending: Mutex<Option<Stamped<Subject>>>,
    results: Mutex<Vec<Stamped<Seen>>>,
    stats: Mutex<Stats>,
    skipped: AtomicU64,
    /// What counts as a thing, as f32 bits. Atomics rather than a lock: they
    /// are read once a frame on the worker and written by a keypress on the
    /// display, and neither has anything to wait for.
    confidence: AtomicU32,
    duplicate: AtomicU32,
}

impl Shared {
    fn thresholds(&self) -> super::vision::Thresholds {
        super::vision::Thresholds::new(
            f32::from_bits(self.confidence.load(Ordering::Relaxed)),
            f32::from_bits(self.duplicate.load(Ordering::Relaxed)),
        )
    }
}

pub struct Watcher {
    shared: Arc<Shared>,
    _stop: Arc<()>,
    /// When a frame was last accepted, so offers are thinned to `EVERY`.
    last_offer: Instant,
    /// Set when the model could not be opened, so the caller can say why.
    pub failure: Option<String>,
}

impl Watcher {
    /// Start looking. Opening the model happens on the worker thread, because
    /// it takes tens of milliseconds and the caller is the display.
    /// How often it looks, as times a second.
    pub fn set_rate(&self, hz: u32) {
        let ms = 1000 / hz.clamp(1, 20) as u64;
        self.shared.every_ms.store(ms, Ordering::Relaxed);
    }

    /// What counts as a thing. Takes effect on the next frame looked at.
    pub fn set_thresholds(&self, limits: super::vision::Thresholds) {
        let limits = super::vision::Thresholds::new(limits.confidence, limits.duplicate);
        self.shared.confidence.store(limits.confidence.to_bits(), Ordering::Relaxed);
        self.shared.duplicate.store(limits.duplicate.to_bits(), Ordering::Relaxed);
    }

    pub fn start() -> Watcher {
        let shared = Arc::new(Shared {
            pending: Mutex::new(None),
            results: Mutex::new(Vec::new()),
            stats: Mutex::new(Stats::default()),
            skipped: AtomicU64::new(0),
            confidence: AtomicU32::new(super::vision::MIN_CONFIDENCE.to_bits()),
            duplicate: AtomicU32::new(super::segment::DUPLICATE.to_bits()),
            every_ms: AtomicU64::new((1000 / DEFAULT_HZ.clamp(1, 20)) as u64),
        });
        let stop = Arc::new(());
        let failure = Arc::new(Mutex::new(None::<String>));

        let watch = Arc::downgrade(&stop);
        let worker = shared.clone();
        let said = failure.clone();
        let spawned = std::thread::Builder::new()
            .name("ml-detect".into())
            .spawn(move || run(worker, watch, said));

        if let Err(e) = spawned {
            *failure.lock().expect("failure lock") = Some(format!("{e}"));
        }
        let failure = failure.lock().expect("failure lock").clone();
        Watcher {
            shared,
            _stop: stop,
            last_offer: Instant::now() - Duration::from_secs(1),
            failure,
        }
    }

    /// Which model is doing the looking, for anything that reports on it.
    ///
    /// The choice is made from what is installed, so it can change between
    /// runs; a measurement that does not say which one it measured is not
    /// worth much.
    pub fn kind(&self) -> &'static str {
        match chosen() {
            Capability::SegmentBest => "segmentation (best)",
            Capability::SegmentBetter => "segmentation (better)",
            Capability::Segment => "segmentation",
            _ => "boxes",
        }
    }

    /// Offer a frame. Never blocks, never copies the picture.
    pub fn offer(&mut self, frame: &Frame, generation: u64) {
        let every = Duration::from_millis(self.shared.every_ms.load(Ordering::Relaxed));
        if self.last_offer.elapsed() < every {
            return;
        }
        self.last_offer = Instant::now();

        let mut slot = match self.shared.pending.try_lock() {
            Ok(slot) => slot,
            // The worker has it; it is busy, and this frame is skipped.
            Err(_) => {
                self.shared.skipped.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        if slot.is_some() {
            self.shared.skipped.fetch_add(1, Ordering::Relaxed);
        }
        // Cheap since the planes are shared; this is three atomic increments.
        *slot = Some(Stamped { at: frame.at, generation, what: Subject::Frame(frame.clone_ref()) });
    }

    /// Offer a still, which is read from disk rather than held in memory.
    ///
    /// Unlike a film there is no next frame to try instead, so this is offered
    /// until something comes back rather than thinned to a rate.
    pub fn offer_picture(&mut self, path: &std::path::Path, generation: u64) {
        let Ok(mut slot) = self.shared.pending.try_lock() else { return };
        if slot.is_some() {
            return;
        }
        *slot = Some(Stamped {
            at: std::time::Duration::ZERO,
            generation,
            what: Subject::Picture(path.to_path_buf()),
        });
    }

    /// Take whatever has been worked out since the last call.
    pub fn collect(&mut self) -> Vec<Stamped<Seen>> {
        std::mem::take(&mut *self.shared.results.lock().expect("results lock"))
    }

    pub fn stats(&self) -> Stats {
        let mut stats = *self.shared.stats.lock().expect("stats lock");
        stats.skipped = self.shared.skipped.load(Ordering::Relaxed);
        stats
    }

    /// Whether the model failed to open, and why.
    pub fn failed(&self) -> Option<&str> {
        self.failure.as_deref()
    }
}

#[cfg(feature = "ml")]
fn run(shared: Arc<Shared>, stop: std::sync::Weak<()>, failure: Arc<Mutex<Option<String>>>) {
    let which = chosen();
    let outlines = which != Capability::Detect;

    let mut tracker = super::track::Tracker::new();
    let mut generation = u64::MAX;

    let mut detector = match super::vision::Detector::open_for(which) {
        Ok(detector) => detector,
        Err(e) => {
            *failure.lock().expect("failure lock") = Some(format!("{e:#}"));
            return;
        }
    };
    let mut input = super::prep::Input::new(super::prep::DEFAULT_SIZE);

    loop {
        if stop.upgrade().is_none() {
            return;
        }
        let taken = shared.pending.lock().expect("pending lock").take();
        let Some(frame) = taken else {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        };

        let Some(picture) = frame.what.frame() else { continue };
        super::prep::letterbox(&picture, &mut input);
        // Read per frame, not per run: the sliders on the F8 page move while
        // a film is playing, and a threshold that only took effect on the next
        // file would read as a control that does nothing.
        let limits = shared.thresholds();
        let started = Instant::now();
        let outcome = if outlines {
            detector.look(&input, limits)
        } else {
            // Boxes come back as outlines covering their whole rectangle, so
            // everything downstream has one shape to deal with rather than two.
            detector.run(&input, limits).map(|found| {
                found.into_iter().map(super::segment::Segmented::from_box).collect()
            })
        };
        let mut found = match outcome {
            Ok(found) => found,
            Err(e) => {
                *failure.lock().expect("failure lock") = Some(format!("{e:#}"));
                return;
            }
        };
        let took = started.elapsed();

        // A new file, or a seek, means nothing carries over: what was being
        // followed is not on screen any more, and matching against it would
        // hand an old id to a new thing.
        if frame.generation != generation {
            tracker.clear();
            generation = frame.generation;
        }
        let boxes: Vec<_> = found.iter().map(|s| s.found.clone()).collect();
        let ids = tracker.update(&boxes);
        for (item, id) in found.iter_mut().zip(ids) {
            item.id = id;
        }
        // One sighting is as likely to be a shadow as a thing. Drawing from
        // the second one costs a fifth of a second and stops the flicker.
        //
        // A photograph is the exception: it is looked at once and never
        // changes, so there is no flicker to prevent and no second sighting
        // coming. Waiting for one would mean never drawing anything at all.
        if matches!(frame.what, Subject::Frame(_)) {
            found.retain(|item| tracker.is_settled(item.id));
        }

        shared.results.lock().expect("results lock").push(Stamped {
            at: frame.at,
            generation: frame.generation,
            what: Seen { found, size: (picture.width, picture.height) },
        });
        let mut stats = shared.stats.lock().expect("stats lock");
        stats.examined += 1;
        stats.last = took;
    }
}

/// Which model does the looking.
///
/// The one chosen on the plugins page, if it is still installed. Otherwise the
/// best that is — installing a model is a reasonable way of asking for it —
/// falling back to boxes. So removing the model in use does not break looking,
/// it just makes it coarser.
pub fn chosen() -> Capability {
    if let Some(picked) = super::store::preferred()
        && picked.is_looker()
        && super::store::installed(picked)
    {
        return picked;
    }
    for which in [Capability::SegmentBest, Capability::SegmentBetter, Capability::Segment] {
        if super::store::installed(which) {
            return which;
        }
    }
    Capability::Detect
}

/// Without inference compiled in there is nothing to run.
#[cfg(not(feature = "ml"))]
fn run(_shared: Arc<Shared>, _stop: std::sync::Weak<()>, failure: Arc<Mutex<Option<String>>>) {
    *failure.lock().expect("failure lock") =
        Some("this build has no inference; rebuild with `--features ml`".to_string());
}

/// What is in view, as a tag per *thing* — not per kind.
///
/// Masks are coloured per object, so the legend has to be too: a single PERSON
/// x3 tag in one colour would name three things that are drawn in three.
///
/// Ordering is chosen for a bar that must not reshuffle itself between frames:
/// the biggest group first and ties broken by name, exactly as when this
/// counted kinds, then within a kind by tracker id — which is assigned in the
/// order things were first seen and does not change while they stay on screen.
/// So the numbering is stable, and a thing keeps its number for as long as it
/// keeps its colour.
pub fn tally(found: &[Segmented]) -> Vec<super::bar::Tag> {
    let mut kinds: Vec<(usize, Vec<u32>)> = Vec::new();
    for item in found {
        match kinds.iter_mut().find(|(class, _)| *class == item.found.class) {
            Some((_, ids)) => ids.push(item.id),
            None => kinds.push((item.found.class, vec![item.id])),
        }
    }
    kinds.sort_by(|a, b| {
        b.1.len().cmp(&a.1.len()).then_with(|| {
            super::vision::CLASSES
                .get(a.0)
                .unwrap_or(&"")
                .cmp(super::vision::CLASSES.get(b.0).unwrap_or(&""))
        })
    });

    let mut tags = Vec::new();
    for (class, mut ids) in kinds {
        ids.sort_unstable();
        for (i, id) in ids.into_iter().enumerate() {
            tags.push(super::bar::Tag {
                class,
                ordinal: i + 1,
                slot: super::labels::slot(id),
            });
        }
    }
    tags
}

#[cfg(test)]
mod tests {
    /// The still path, end to end, against the real models.
    ///
    /// Ignored by default because it needs the runtime and a model installed,
    /// which most machines running the tests will not have. Run it with
    /// `cargo test --release -- --ignored still_picture` when the app says it
    /// is looking and never finds anything.
    #[test]
    #[ignore]
    fn a_still_picture_is_looked_at() {
        // Any picture will do to prove the path works; point it at one with
        // people in it (SHOW_TEST_PICTURE) to see it find something.
        let named = std::env::var("SHOW_TEST_PICTURE").ok();
        let path = std::path::Path::new(
            named.as_deref().unwrap_or("testdata/corpus/base.jpg"),
        );
        assert!(path.is_file(), "no picture to look at");

        let mut watcher = super::Watcher::start();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        let mut results = Vec::new();
        while std::time::Instant::now() < deadline && results.is_empty() {
            if let Some(why) = watcher.failed() {
                panic!("the model did not open: {why}");
            }
            watcher.offer_picture(path, 1);
            std::thread::sleep(std::time::Duration::from_millis(50));
            results = watcher.collect();
        }
        let seen = results.first().expect("nothing came back within twenty seconds");
        println!(
            "looked at {}x{}, found {}",
            seen.what.size.0,
            seen.what.size.1,
            seen.what.found.len()
        );
        assert_eq!(seen.generation, 1, "the answer is about the picture that was asked about");
        assert!(seen.what.size.0 > 0 && seen.what.size.1 > 0, "no size came back");
    }

    use super::*;

    fn found(class: usize, confidence: f32) -> Segmented {
        Segmented::from_box(super::super::vision::Found {
            class,
            confidence,
            bounds: (0.0, 0.0, 1.0, 1.0),
        })
    }

    /// The same, with the tracker id the colour is taken from.
    fn tracked(class: usize, confidence: f32, id: u32) -> Segmented {
        let mut item = found(class, confidence);
        item.id = id;
        item
    }

    #[test]
    fn a_tally_names_each_thing_rather_than_each_kind() {
        // Two people and a car are three tags, not two: they are drawn in
        // three colours, so a legend of two would name one of them wrongly.
        let tags = tally(&[tracked(0, 0.9, 1), tracked(0, 0.8, 2), tracked(2, 0.7, 3)]);
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0].class, 0);
        assert_eq!(tags[1].class, 0);
        assert_eq!(tags[2].class, 2, "the pair sorts before the single");
        assert_eq!((tags[0].ordinal, tags[1].ordinal), (1, 2), "numbered within their kind");
        assert_ne!(tags[0].slot, tags[1].slot, "two people got the same colour");
    }

    #[test]
    fn a_thing_keeps_its_number_when_the_model_reorders_its_answers() {
        // The bar must not renumber people because the detector returned them
        // the other way round this frame.
        let one = tally(&[tracked(0, 0.9, 4), tracked(0, 0.8, 9)]);
        let other = tally(&[tracked(0, 0.8, 9), tracked(0, 0.9, 4)]);
        assert_eq!(one, other, "the same two people read differently");
    }

    #[test]
    fn the_order_does_not_wobble_when_counts_tie() {
        // Two kinds, one each. Sorted by name so the bar reads the same way
        // whichever order the model happened to report them in.
        let one = tally(&[tracked(2, 0.9, 1), tracked(0, 0.8, 2)]);
        let other = tally(&[tracked(0, 0.8, 2), tracked(2, 0.9, 1)]);
        assert_eq!(one, other);
        assert_eq!(one[0].class, 2, "car sorts before person");
    }

    #[test]
    fn nothing_found_is_an_empty_tally_rather_than_a_row_of_zeroes() {
        assert!(tally(&[]).is_empty());
    }
}
