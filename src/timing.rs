//! Thread-aware profiler for the load pipeline.
//!
//! The whole point of this viewer is latency, so every stage between "process
//! started" and "pixels on screen" gets an event. Events carry a *lane* so the
//! report can show which work overlapped: decoding runs on a worker thread
//! while the GPU is still being initialised on the main thread, and the only
//! way to see whether that overlap is actually paying off is to draw it.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static PROF: OnceLock<Profiler> = OnceLock::new();

#[derive(Clone)]
pub struct Event {
    pub lane: &'static str,
    pub name: &'static str,
    /// Offset from the profiler origin (first instruction of `main`).
    pub start: Duration,
    pub dur: Duration,
    /// Free-form suffix, e.g. "3.2 MB" or "6000x4000".
    pub note: Option<String>,
}

pub struct Profiler {
    origin: Instant,
    events: Mutex<Vec<Event>>,
    enabled: bool,
}

impl Profiler {
    fn record(&self, ev: Event) {
        self.events.lock().unwrap().push(ev);
    }


}

/// Must be the first thing `main` does; the origin timestamp anchors everything.
///
/// Recording is always on: it costs one mutex push per stage, a few dozen per
/// image, and the on-screen display needs the numbers even when no report was
/// asked for. `enabled` gates *printing*, not collection.
pub fn init(enabled: bool) {
    let _ = PROF.set(Profiler {
        origin: Instant::now(),
        events: Mutex::new(Vec::with_capacity(64)),
        enabled,
    });
}

/// A copy of everything recorded so far, oldest first.
pub fn events() -> Vec<Event> {
    let mut events = prof().events.lock().unwrap().clone();
    events.sort_by_key(|e| e.start);
    events
}

fn prof() -> &'static Profiler {
    PROF.get_or_init(|| Profiler {
        origin: Instant::now(),
        events: Mutex::new(Vec::new()),
        enabled: false,
    })
}


/// Drop everything recorded so far, so the next report covers one run only.
pub fn reset() {
    if let Some(p) = PROF.get() {
        p.events.lock().unwrap().clear();
    }
}


/// Scoped timer. The event is recorded when the guard drops.
#[must_use = "the span ends when the guard is dropped"]
pub struct Span {
    lane: &'static str,
    name: &'static str,
    start: Instant,
    note: Option<String>,
}

impl Span {
    /// Attach a note (size, dimensions, backend name, ...) to the event.
    pub fn note(&mut self, note: impl Into<String>) {
        self.note = Some(note.into());
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let p = prof();
        p.record(Event {
            lane: self.lane,
            name: self.name,
            start: self.start.saturating_duration_since(p.origin),
            dur: self.start.elapsed(),
            note: self.note.take(),
        });
    }
}

pub fn span(lane: &'static str, name: &'static str) -> Span {
    Span { lane, name, start: Instant::now(), note: None }
}

/// A zero-duration event, for "this happened at time T" moments.
pub fn mark(lane: &'static str, name: &'static str) {
    let p = prof();
    p.record(Event { lane, name, start: p.origin.elapsed(), dur: Duration::ZERO, note: None });
}

/// A zero-duration event carrying an explanation.
pub fn mark_note(lane: &'static str, name: &'static str, note: impl Into<String>) {
    let p = prof();
    p.record(Event {
        lane,
        name,
        start: p.origin.elapsed(),
        dur: Duration::ZERO,
        note: Some(note.into()),
    });
}

/// Record a stage measured elsewhere (e.g. on another thread, or derived).
pub fn record(lane: &'static str, name: &'static str, start: Duration, dur: Duration, note: Option<String>) {
    prof().record(Event { lane, name, start, dur, note });
}

/// Time spent between `execve` and the first instruction of `main`: dynamic
/// linking, relocations, runtime init. Real cost for a default image opener,
/// and invisible to any timer we could start ourselves.
#[cfg(target_os = "linux")]
pub fn pre_main() -> Option<Duration> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // `comm` can contain spaces and parens, so fields are only unambiguous
    // after the final ')'. `state` is the first field past it, which puts
    // `starttime` (field 22) at index 19.
    let rest = &stat[stat.rfind(')')? + 1..];
    let starttime: u64 = rest.split_whitespace().nth(19)?.parse().ok()?;
    let uptime: f64 = std::fs::read_to_string("/proc/uptime").ok()?.split_whitespace().next()?.parse().ok()?;
    // USER_HZ is 100 on every Linux target we care about.
    Some(Duration::from_secs_f64((uptime - starttime as f64 / 100.0).max(0.0)))
}

#[cfg(not(target_os = "linux"))]
pub fn pre_main() -> Option<Duration> {
    None
}

const BAR_WIDTH: usize = 44;

/// Print a waterfall of everything recorded so far.
pub fn report(title: &str) {
    let p = prof();
    if !p.enabled {
        return;
    }
    let mut events = p.events.lock().unwrap().clone();
    if events.is_empty() {
        return;
    }
    events.sort_by_key(|e| e.start);

    // Normalise to the first event so a report that covers the second file of a
    // run still reads as a waterfall starting at zero.
    let base = events[0].start;
    for e in &mut events {
        e.start -= base;
    }

    let span_end = events.iter().map(|e| e.start + e.dur).max().unwrap_or_default();
    let total = span_end.as_secs_f64().max(1e-9);

    let lane_w = events.iter().map(|e| e.lane.len()).max().unwrap_or(4).max(4);
    let name_w = events.iter().map(|e| e.name.len()).max().unwrap_or(5).max(5);

    let offset = if base.is_zero() {
        String::new()
    } else {
        format!("  (from t+{:.1} ms)", base.as_secs_f64() * 1e3)
    };
    eprintln!();
    eprintln!("  {title}  —  {:.2} ms end to end{offset}", span_end.as_secs_f64() * 1e3);
    eprintln!(
        "  {:lane_w$}  {:name_w$}  {:>9}  {:>9}",
        "lane", "stage", "start ms", "dur ms"
    );

    for e in &events {
        let s = e.start.as_secs_f64();
        let d = e.dur.as_secs_f64();
        // An event starting at the very end of the span still needs a cell to
        // live in, so pin the start inside the bar before sizing it.
        let lo = (((s / total) * BAR_WIDTH as f64) as usize).min(BAR_WIDTH - 1);
        let hi = ((((s + d) / total) * BAR_WIDTH as f64).ceil() as usize).clamp(lo + 1, BAR_WIDTH);

        let mut bar = String::with_capacity(BAR_WIDTH);
        for i in 0..BAR_WIDTH {
            bar.push(if i >= lo && i < hi {
                if d == 0.0 { '|' } else { '#' }
            } else {
                '.'
            });
        }

        eprintln!(
            "  {:lane_w$}  {:name_w$}  {:>9.3}  {:>9.3}  {bar} {}",
            e.lane,
            e.name,
            s * 1e3,
            d * 1e3,
            e.note.as_deref().unwrap_or("")
        );
    }
    eprintln!();
}
