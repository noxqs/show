//! Answering a click without stopping the film.
//!
//! Encoding a picture takes about a fifth of a second, which is five dropped
//! frames if it happens on the display thread. So it happens here instead, on
//! a thread of its own, and the answer arrives when it arrives.
//!
//! One request at a time, and a later one replaces an earlier one that has
//! not started: if someone clicks twice, they want the second answer. The
//! embedding of the last picture encoded is kept, so a second click on the
//! same frame skips the slow half and comes back in milliseconds.
//!
//! Cancellation is the `Arc<()>` / `Weak::upgrade` idiom used everywhere else
//! in this codebase: drop the pointer and the thread notices and stops.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::segment::Segmented;
pub use super::subject::Subject;

/// What is being asked about the picture.
pub enum Prompt {
    /// One point, as a fraction of the picture. A fraction rather than a pixel
    /// because the worker may decode the picture at a different size than the
    /// screen is showing, and a fraction means the same thing at every size.
    Point((f32, f32)),
    /// A detector's boxes, in the pixels it was looking at, to be sharpened
    /// into outlines. They carry their classes so what comes back is still a
    /// person or a car rather than an anonymous shape.
    Boxes { found: Vec<crate::ml::vision::Found>, size: (u32, u32) },
}

/// A question waiting to be answered.
struct Question {
    subject: Subject,
    /// Which picture this was, so a stale answer can be recognised.
    generation: u64,
    at: Duration,
    prompt: Prompt,
}

/// What came back.
pub struct Answer {
    pub generation: u64,
    pub at: Duration,
    /// One outline for a click, one per box for a refinement, and none at all
    /// when the model could not separate anything.
    pub cuts: Vec<Segmented>,
    /// Whether this sharpened a detector's boxes rather than answering a
    /// click. The two are drawn differently and must not be confused.
    pub refined: bool,
    /// The size the cut-out was worked out at, which is what it has to be
    /// drawn against.
    pub size: (u32, u32),
    pub took: Duration,
    /// Whether the picture had to be encoded, or the last one still served.
    pub encoded: bool,
}

struct Shared {
    asking: Mutex<Option<Question>>,
    answers: Mutex<Vec<Answer>>,
}

pub struct Pointer {
    shared: Arc<Shared>,
    _stop: Arc<()>,
    pub failure: Option<String>,
}

impl Pointer {
    pub fn start() -> Pointer {
        let shared = Arc::new(Shared {
            asking: Mutex::new(None),
            answers: Mutex::new(Vec::new()),
        });
        let stop = Arc::new(());
        let failure = Arc::new(Mutex::new(None::<String>));

        let watch = Arc::downgrade(&stop);
        let worker = shared.clone();
        let said = failure.clone();
        let spawned = std::thread::Builder::new()
            .name("ml-cutout".into())
            .spawn(move || run(worker, watch, said));
        if let Err(e) = spawned {
            *failure.lock().expect("failure lock") = Some(format!("{e}"));
        }
        let failure = failure.lock().expect("failure lock").clone();
        Pointer { shared, _stop: stop, failure }
    }

    /// Ask what is at a point of a picture.
    ///
    /// Replaces any question not yet started: two clicks in a row mean the
    /// first answer is no longer wanted.
    pub fn ask(&self, subject: Subject, generation: u64, at: Duration, prompt: Prompt) {
        let question = Question { subject, generation, at, prompt };
        *self.shared.asking.lock().expect("asking lock") = Some(question);
    }

    /// Take whatever has been answered since the last call.
    pub fn collect(&mut self) -> Vec<Answer> {
        std::mem::take(&mut *self.shared.answers.lock().expect("answers lock"))
    }

    pub fn failed(&self) -> Option<&str> {
        self.failure.as_deref()
    }
}

#[cfg(feature = "ml")]
fn run(shared: Arc<Shared>, stop: std::sync::Weak<()>, failure: Arc<Mutex<Option<String>>>) {
    let mut cutter = match super::cutout::Cutter::open() {
        Ok(cutter) => cutter,
        Err(e) => {
            *failure.lock().expect("failure lock") = Some(format!("{e:#}"));
            return;
        }
    };

    // The last picture encoded, and which one it was. Clicking twice on the
    // same frame is the common case — a first click that chose a shoulder,
    // then a second that means the whole person — and this is what makes the
    // second one instant.
    let mut held: Option<(u64, Duration, Vec<f32>, (u32, u32))> = None;

    loop {
        if stop.upgrade().is_none() {
            return;
        }
        let taken = shared.asking.lock().expect("asking lock").take();
        let Some(question) = taken else {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        };

        let started = Instant::now();

        // The embedding of the last picture looked at, if it is this one.
        // Keeping the size with it means a second click needs neither the
        // encoder nor the decode that fed it.
        let ready = held
            .as_ref()
            .filter(|(g, at, _, _)| *g == question.generation && *at == question.at)
            .map(|(_, _, _, size)| *size);

        let size = match ready {
            Some(size) => size,
            None => {
                let Some(frame) = question.subject.frame() else {
                    continue;
                };
                let size = (frame.width, frame.height);
                let square = super::cutout::square(&frame);
                match cutter.embed(&square) {
                    Ok(embedding) => {
                        held = Some((question.generation, question.at, embedding, size))
                    }
                    Err(e) => {
                        *failure.lock().expect("failure lock") = Some(format!("{e:#}"));
                        return;
                    }
                }
                size
            }
        };
        let Some((_, _, embedding, _)) = held.as_ref() else { continue };

        let refined = matches!(question.prompt, Prompt::Boxes { .. });
        let outcome = match &question.prompt {
            Prompt::Point((x, y)) => {
                let point = (x * size.0 as f32, y * size.1 as f32);
                cutter.cut(embedding, point, size).map(|cut| cut.into_iter().collect())
            }
            // The boxes were worked out against a picture of some size; this
            // one may have been decoded at another, so they are carried across
            // rather than assumed to match.
            Prompt::Boxes { found, size: was } => {
                let scaled: Vec<_> = found
                    .iter()
                    .map(|f| {
                        let (sx, sy) = (
                            size.0 as f32 / was.0.max(1) as f32,
                            size.1 as f32 / was.1.max(1) as f32,
                        );
                        let b = f.bounds;
                        crate::ml::vision::Found {
                            class: f.class,
                            confidence: f.confidence,
                            bounds: (b.0 * sx, b.1 * sy, b.2 * sx, b.3 * sy),
                        }
                    })
                    .collect();
                cutter.refine(embedding, &scaled, size)
            }
        };
        match outcome {
            Ok(cuts) => shared.answers.lock().expect("answers lock").push(Answer {
                generation: question.generation,
                at: question.at,
                cuts,
                refined,
                size,
                took: started.elapsed(),
                encoded: ready.is_none(),
            }),
            Err(e) => {
                *failure.lock().expect("failure lock") = Some(format!("{e:#}"));
                return;
            }
        }
    }
}

#[cfg(not(feature = "ml"))]
fn run(_shared: Arc<Shared>, _stop: std::sync::Weak<()>, failure: Arc<Mutex<Option<String>>>) {
    *failure.lock().expect("failure lock") =
        Some("this build has no inference; rebuild with `--features ml`".to_string());
}
