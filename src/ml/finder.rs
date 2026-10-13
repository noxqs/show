//! The thread that answers "where is the thing I named".
//!
//! Built on the same shape as `pointer.rs` and deliberately not inside it.
//! That worker holds the cut-out model open for as long as it runs; this one
//! holds three graphs and three and a half gigabytes, and only from the moment
//! somebody first asks a question. Sharing a thread would mean loading both
//! whichever was wanted, which is the wrong trade in both directions.
//!
//! One question at a time, and a newer one replaces an older one that has not
//! started — typing a second phrase means the first answer is not wanted.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::segment::Segmented;
use super::subject::Subject;

/// A question waiting to be answered.
struct Question {
    subject: Subject,
    /// Which picture this was, so a stale answer can be recognised.
    generation: u64,
    at: Duration,
    phrase: String,
}

/// What came back.
pub struct Answer {
    pub generation: u64,
    pub at: Duration,
    /// The phrase this answers, so a readout can say what was asked.
    pub phrase: String,
    /// One outline per thing found, already on the label grid.
    pub found: Vec<Segmented>,
    /// The size it was worked out against.
    pub size: (u32, u32),
    pub took: Duration,
}

struct Shared {
    asking: Mutex<Option<Question>>,
    answers: Mutex<Vec<Answer>>,
    /// Set while a question is being worked on, so the caller can say so
    /// rather than leaving the screen unchanged for six seconds.
    busy: Mutex<bool>,
}

pub struct Finder {
    shared: Arc<Shared>,
    _stop: Arc<()>,
    pub failure: Option<String>,
}

impl Finder {
    pub fn start() -> Finder {
        let shared = Arc::new(Shared {
            asking: Mutex::new(None),
            answers: Mutex::new(Vec::new()),
            busy: Mutex::new(false),
        });
        let stop = Arc::new(());
        let failure = Arc::new(Mutex::new(None::<String>));

        let watch = Arc::downgrade(&stop);
        let worker = shared.clone();
        let said = failure.clone();
        let spawned = std::thread::Builder::new()
            .name("ml-finder".into())
            .spawn(move || run(worker, watch, said));
        if let Err(e) = spawned {
            *failure.lock().expect("failure lock") = Some(format!("{e}"));
        }
        let failure = failure.lock().expect("failure lock").clone();
        Finder { shared, _stop: stop, failure }
    }

    /// Ask what in this picture matches `phrase`.
    pub fn ask(&self, subject: Subject, generation: u64, at: Duration, phrase: &str) {
        let question =
            Question { subject, generation, at, phrase: phrase.to_string() };
        *self.shared.asking.lock().expect("asking lock") = Some(question);
    }

    /// Take whatever has been answered since the last call.
    pub fn collect(&mut self) -> Vec<Answer> {
        std::mem::take(&mut *self.shared.answers.lock().expect("answers lock"))
    }

    /// Whether a question is in flight, question included or not.
    pub fn busy(&self) -> bool {
        *self.shared.busy.lock().expect("busy lock")
            || self.shared.asking.lock().expect("asking lock").is_some()
    }

    pub fn failed(&self) -> Option<&str> {
        self.failure.as_deref()
    }
}

#[cfg(feature = "ml")]
fn run(shared: Arc<Shared>, stop: std::sync::Weak<()>, failure: Arc<Mutex<Option<String>>>) {
    // Opened on the first question rather than at start-up: this is several
    // gigabytes of weights, and a viewer that paid for them before being asked
    // anything would be a viewer nobody leaves running.
    let mut model: Option<super::sam3::Sam3> = None;
    // The picture last encoded, and which one it was. Asking a second phrase
    // of the same frame is the common case, and this is what makes it cost the
    // decoder alone rather than the encoder again.
    let mut held: Option<(u64, Duration, super::sam3::Embedding)> = None;

    loop {
        if stop.upgrade().is_none() {
            return;
        }
        let taken = shared.asking.lock().expect("asking lock").take();
        let Some(question) = taken else {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        };
        *shared.busy.lock().expect("busy lock") = true;

        if model.is_none() {
            match super::sam3::Sam3::open() {
                Ok(opened) => model = Some(opened),
                Err(e) => {
                    *failure.lock().expect("failure lock") = Some(format!("{e:#}"));
                    *shared.busy.lock().expect("busy lock") = false;
                    return;
                }
            }
        }
        let Some(model) = model.as_mut() else { continue };

        let fresh = held
            .as_ref()
            .is_none_or(|(g, at, _)| *g != question.generation || *at != question.at);
        if fresh {
            let Some(frame) = question.subject.frame() else {
                *shared.busy.lock().expect("busy lock") = false;
                continue;
            };
            match model.look(&frame) {
                Ok(embedding) => held = Some((question.generation, question.at, embedding)),
                Err(e) => {
                    *failure.lock().expect("failure lock") = Some(format!("{e:#}"));
                    *shared.busy.lock().expect("busy lock") = false;
                    return;
                }
            }
        }
        let Some((_, _, embedding)) = held.as_ref() else {
            *shared.busy.lock().expect("busy lock") = false;
            continue;
        };

        let started = Instant::now();
        match model.find(embedding, &question.phrase, super::sam3::KEEP) {
            Ok(found) => shared.answers.lock().expect("answers lock").push(Answer {
                generation: question.generation,
                at: question.at,
                phrase: question.phrase,
                found,
                size: embedding.source(),
                took: started.elapsed(),
            }),
            Err(e) => {
                *failure.lock().expect("failure lock") = Some(format!("{e:#}"));
                *shared.busy.lock().expect("busy lock") = false;
                return;
            }
        }
        *shared.busy.lock().expect("busy lock") = false;
    }
}

#[cfg(not(feature = "ml"))]
fn run(_shared: Arc<Shared>, _stop: std::sync::Weak<()>, failure: Arc<Mutex<Option<String>>>) {
    *failure.lock().expect("failure lock") =
        Some("this build has no inference".to_string());
}
