//! Decoded frames kept in memory, so stepping back a frame costs nothing.
//!
//! Stepping *forward* is easy: the next frame is already decoded and waiting.
//! Backwards is the hard direction, because inter frames reference earlier
//! ones and a codec cannot run in reverse. The usual answer is to seek to the
//! keyframe before the target and decode forward, discarding — which on a file
//! with ten-second keyframes means decoding two hundred frames to move back
//! one, and that is a visible pause on every press.
//!
//! So the frames the player has already shown are kept instead. That is the
//! whole trick, and its virtue is what it does *not* cost: no second decoder,
//! no extra CPU, nothing asked of the GPU that is busy drawing the film. The
//! frames were decoded anyway. Only the memory is new.
//!
//! Bounded two ways, because one is not enough. Bytes stop a large film
//! exhausting the machine; a time span stops a small one hoarding half an
//! hour of memory it will never be asked for. Whichever binds first wins.

use std::collections::VecDeque;
use std::time::Duration;

use crate::video::Frame;

/// How far back the cache will hold, when memory allows.
///
/// Three seconds, matched to how far ahead the decoder runs, so stepping is
/// instant for the same distance in either direction. Long enough to look at a
/// moment from either side; short enough that two of them fit comfortably in
/// memory even at 4K.
pub const DEFAULT_SPAN: Duration = Duration::from_secs(3);

pub struct FrameCache {
    frames: VecDeque<Frame>,
    bytes: u64,
    budget: u64,
    span: Duration,
    /// The film's frame interval, once known: what "the next frame" means.
    interval: Option<Duration>,
}

impl FrameCache {
    pub fn new(budget: u64) -> FrameCache {
        FrameCache { frames: VecDeque::new(), bytes: 0, budget, span: DEFAULT_SPAN, interval: None }
    }

    /// Tell the cache the film's frame interval. Without it, only a gap of a
    /// whole second counts as a break in the run — and a scrub of less than
    /// that left a hole the step keys then jumped straight across.
    pub fn set_interval(&mut self, interval: Option<Duration>) {
        self.interval = interval.filter(|i| !i.is_zero());
    }

    /// The widest gap between two frames that still counts as consecutive:
    /// a frame and a half, so a dropped or doubled timestamp is not a break
    /// but a missing frame is. A second when the rate is not known.
    pub fn continuous(&self) -> Duration {
        self.interval.map_or(Duration::from_secs(1), |i| i * 3 / 2)
    }

    /// Bytes one frame occupies.
    fn size_of(frame: &Frame) -> u64 {
        frame.planes.bytes()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Media time between the oldest and newest frame held.
    pub fn span(&self) -> Duration {
        match (self.frames.front(), self.frames.back()) {
            (Some(first), Some(last)) => last.at.saturating_sub(first.at),
            _ => Duration::ZERO,
        }
    }

    /// Keep a frame that has just been shown.
    ///
    /// Frames arrive in playback order, so the oldest is always at the front
    /// and eviction is a pop rather than a search.
    pub fn push(&mut self, frame: Frame) {
        // A seek breaks the run, and it breaks it in both directions. Going
        // backwards is obvious. Going forwards is the one that bites: the
        // frames from before the seek are still there, still older, and
        // stepping back would find one of them and jump the whole distance
        // that was just seeked over — a step key that moves ten minutes.
        let continuous = self.continuous();
        let broken = self.frames.back().is_some_and(|last| {
            frame.at < last.at || frame.at.saturating_sub(last.at) > continuous
        });
        if broken {
            self.clear();
        }
        self.bytes += Self::size_of(&frame);
        self.frames.push_back(frame);
        self.trim();
    }

    fn trim(&mut self) {
        while self.frames.len() > 1 && (self.bytes > self.budget || self.span() > self.span) {
            if let Some(dropped) = self.frames.pop_front() {
                self.bytes -= Self::size_of(&dropped);
            }
        }
    }

    /// Add frames that belong *before* what is held.
    ///
    /// A backward fill runs forward — a decoder cannot do anything else — so
    /// it produces a run of frames ending where the cache begins. They are
    /// prepended rather than pushed, since pushing would read as a jump
    /// backwards and throw away everything already held.
    ///
    /// Only frames that actually meet the front are kept: a run with a hole in
    /// it would let a step land the wrong side of the gap.
    pub fn prefill(&mut self, frames: Vec<Frame>) {
        let Some(front) = self.frames.front().map(|f| f.at) else {
            for frame in frames {
                self.push(frame);
            }
            return;
        };
        // The run has to reach the front, or stepping back from the front
        // would cross the hole between them.
        let reaches = frames
            .iter()
            .rev()
            .find(|f| f.at < front)
            .is_some_and(|last| front.saturating_sub(last.at) <= self.continuous());
        if !reaches {
            return;
        }
        for frame in frames.into_iter().rev() {
            if frame.at >= self.frames.front().map(|f| f.at).unwrap_or(front) {
                continue;
            }
            self.bytes += Self::size_of(&frame);
            self.frames.push_front(frame);
        }
        self.trim_around(front);
    }

    /// Evict from whichever end is further from `position`.
    ///
    /// A rear-view cache trims its oldest, which is right while playing and
    /// wrong once it holds frames on both sides: the frames just fetched for
    /// stepping back are the oldest, and trimming by age would drop them the
    /// moment they arrived.
    pub fn trim_around(&mut self, position: Duration) {
        while self.frames.len() > 1 && (self.bytes > self.budget || self.span() > self.span) {
            let front = self.frames.front().map(|f| f.at).unwrap_or_default();
            let back = self.frames.back().map(|f| f.at).unwrap_or_default();
            let drop_front = position.saturating_sub(front) >= back.saturating_sub(position);
            let dropped =
                if drop_front { self.frames.pop_front() } else { self.frames.pop_back() };
            if let Some(dropped) = dropped {
                self.bytes -= Self::size_of(&dropped);
            }
        }
    }

    /// Media time held before and after `position`.
    pub fn around(&self, position: Duration) -> (Duration, Duration) {
        let before = self
            .frames
            .front()
            .map(|f| position.saturating_sub(f.at))
            .unwrap_or_default();
        let after = self
            .frames
            .back()
            .map(|f| f.at.saturating_sub(position))
            .unwrap_or_default();
        (before, after)
    }

    pub fn clear(&mut self) {
        self.frames.clear();
        self.bytes = 0;
    }

    /// The frame immediately before `at`, if it is held.
    ///
    /// Strictly before, and by more than a hair: timestamps are floating point
    /// and a frame must not be found to precede itself.
    pub fn before(&self, at: Duration) -> Option<&Frame> {
        const EPSILON: Duration = Duration::from_micros(500);
        let cutoff = at.saturating_sub(EPSILON);
        self.frames.iter().rev().find(|f| f.at < cutoff)
    }

    /// The frame being shown at `at` — the last one at or before it.
    ///
    /// `before` deliberately excludes the frame exactly at the time, because
    /// stepping back means the one before this. Pointing at the picture means
    /// this one, so it needs its own answer rather than a nudged argument.
    pub fn showing(&self, at: Duration) -> Option<&Frame> {
        const EPSILON: Duration = Duration::from_micros(500);
        let cutoff = at + EPSILON;
        self.frames.iter().rev().find(|f| f.at <= cutoff)
    }

    /// The frame immediately after `at`, if it is held.
    pub fn after(&self, at: Duration) -> Option<&Frame> {
        const EPSILON: Duration = Duration::from_micros(500);
        let cutoff = at + EPSILON;
        self.frames.iter().find(|f| f.at > cutoff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(at_ms: u64, bytes: usize) -> Frame {
        Frame {
            key: false,
            generation: 0,
            planes: crate::video::Planes::cpu([
                std::sync::Arc::new(vec![0u8; bytes]),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            ]),
            width: 2,
            height: 2,
            chroma: (1, 1),
            at: Duration::from_millis(at_ms),
            colour: crate::video::Colour { hd: true, full_range: false },
        }
    }


    #[test]
    fn a_gap_of_a_few_frames_breaks_the_run_once_the_rate_is_known() {
        let mut cache = FrameCache::new(u64::MAX);
        cache.set_interval(Some(Duration::from_secs_f64(1.0 / 24.0)));
        // Played up to 3.2 s, then scrubbed 0.8 s on: under the old
        // one-second rule that kept both sides of the hole.
        for i in 0..4 {
            cache.push(frame(3200 - (3 - i) * 42, 10));
        }
        cache.push(frame(4000, 10));
        assert_eq!(cache.len(), 1, "the frames before the scrub went");
        // Consecutive frames still join up.
        cache.push(frame(4042, 10));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn a_backfill_that_does_not_reach_the_front_is_not_taken() {
        let mut cache = FrameCache::new(u64::MAX);
        cache.set_interval(Some(Duration::from_secs_f64(1.0 / 24.0)));
        cache.push(frame(4000, 10));
        // Ends half a second short of what is held.
        cache.prefill(vec![frame(3000, 10), frame(3042, 10), frame(3500, 10)]);
        assert_eq!(cache.len(), 1);
        // Ends one frame short: taken.
        cache.prefill(vec![frame(3917, 10), frame(3958, 10)]);
        assert_eq!(cache.len(), 3);
    }
    #[test]
    fn frames_are_found_on_either_side() {
        let mut cache = FrameCache::new(1 << 20);
        for i in 0..5 {
            cache.push(frame(i * 40, 16));
        }
        let at = Duration::from_millis(80);
        assert_eq!(cache.before(at).map(|f| f.at), Some(Duration::from_millis(40)));
        assert_eq!(cache.after(at).map(|f| f.at), Some(Duration::from_millis(120)));
    }

    #[test]
    fn a_frame_does_not_precede_itself() {
        let mut cache = FrameCache::new(1 << 20);
        cache.push(frame(0, 16));
        cache.push(frame(40, 16));
        // Asking from the oldest frame's own timestamp: there is nothing
        // before it, and returning itself would freeze the step key.
        assert!(cache.before(Duration::ZERO).is_none());
    }

    #[test]
    fn the_budget_evicts_the_oldest() {
        // Room for three frames of 100 bytes.
        let mut cache = FrameCache::new(300);
        for i in 0..6 {
            cache.push(frame(i * 40, 100));
        }
        assert!(cache.bytes() <= 300, "over budget: {}", cache.bytes());
        // The newest survive; the oldest are gone.
        assert!(cache.before(Duration::from_millis(200)).is_some());
        assert!(cache.before(Duration::from_millis(40)).is_none());
    }

    #[test]
    fn the_span_bounds_a_small_film_that_memory_would_not() {
        // Tiny frames: memory would hold hours, the span must not.
        let mut cache = FrameCache::new(1 << 30);
        for i in 0..200 {
            cache.push(frame(i * 1000, 8));
        }
        assert!(
            cache.span() <= DEFAULT_SPAN,
            "held {:?}, more than the span allows",
            cache.span()
        );
    }

    #[test]
    fn seeking_backwards_starts_the_run_again() {
        let mut cache = FrameCache::new(1 << 20);
        for i in 0..5 {
            cache.push(frame(1000 + i * 40, 16));
        }
        // A seek back to the start of the film: what is held is no longer the
        // run of frames behind the play head, so it starts again.
        cache.push(frame(0, 16));
        assert_eq!(cache.len(), 1, "the old run must be dropped");
        assert!(cache.before(Duration::ZERO).is_none(), "nothing precedes the new start");
        // And the stale frames are really gone rather than merely unreachable.
        assert_eq!(cache.bytes(), 16);
    }

    #[test]
    fn seeking_forwards_also_starts_the_run_again() {
        let mut cache = FrameCache::new(1 << 20);
        for i in 0..5 {
            cache.push(frame(i * 40, 16));
        }
        // A seek forward by a minute: the old frames are still older, and
        // stepping back must not jump the whole minute to reach one.
        cache.push(frame(60_000, 16));
        assert_eq!(cache.len(), 1, "the run before the seek must be dropped");
        assert!(cache.before(Duration::from_millis(60_000)).is_none());
    }

    #[test]
    fn an_ordinary_gap_between_frames_is_not_a_seek() {
        let mut cache = FrameCache::new(1 << 20);
        // Half a second between frames — slow, but continuous playback, not a
        // seek. A decoder that stalls briefly must not lose the cache.
        cache.push(frame(0, 16));
        cache.push(frame(500, 16));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn a_backward_fill_is_prepended_rather_than_pushed() {
        let mut cache = FrameCache::new(1 << 20);
        for i in 4..8 {
            cache.push(frame(i * 40, 16));
        }
        // What a backward fill produces: an earlier run, decoded forwards,
        // ending where the cache begins.
        cache.prefill((0..4).map(|i| frame(i * 40, 16)).collect());
        assert_eq!(cache.len(), 8, "the earlier run must join the later one");
        assert_eq!(
            cache.before(Duration::from_millis(160)).map(|f| f.at),
            Some(Duration::from_millis(120))
        );
    }

    #[test]
    fn trimming_keeps_what_is_nearest_the_play_head() {
        // Room for three frames only.
        let mut cache = FrameCache::new(300);
        for i in 0..3 {
            cache.push(frame(i * 40, 100));
        }
        // Sitting on the newest frame: the oldest is the one to lose.
        cache.trim_around(Duration::from_millis(80));
        assert!(cache.before(Duration::from_millis(80)).is_some());

        // Sitting on the oldest: the newest is now the far end. Adding a
        // fourth frame must cost the far one, not the one under the play head.
        let mut cache = FrameCache::new(300);
        for i in 0..4 {
            cache.push(frame(i * 40, 100));
        }
        cache.trim_around(Duration::ZERO);
        assert!(
            cache.after(Duration::ZERO).is_some(),
            "the frames next to the play head were dropped"
        );
        assert!(cache.bytes() <= 300);
    }

    #[test]
    fn around_reports_both_directions() {
        let mut cache = FrameCache::new(1 << 20);
        for i in 0..11 {
            cache.push(frame(i * 100, 16));
        }
        let (before, after) = cache.around(Duration::from_millis(500));
        assert_eq!(before, Duration::from_millis(500));
        assert_eq!(after, Duration::from_millis(500));
    }

    #[test]
    fn the_span_is_reported_from_both_sides_of_the_play_head() {
        // Small frames, so the span bounds the cache long before memory does.
        let mut cache = FrameCache::new(1 << 30);
        for i in 0..16 {
            cache.push(frame(i * 1000, 8));
        }
        let newest = cache.frames.back().map(|f| f.at).unwrap_or_default();
        let (before, after) = cache.around(newest);
        assert!(before > Duration::from_secs(2), "held only {before:?} behind");
        assert_eq!(after, Duration::ZERO, "nothing has been decoded ahead here");
    }
}

/// A run of frames being decoded to fill the gap behind the play head.
///
/// The pity the user named: a decoder cannot run backwards, so filling the
/// three seconds *before* a moment means seeking to the keyframe before them
/// and decoding forwards to reach it. That is the same direction of work as
/// the read-ahead, on a different range — which is why it pays to do both at
/// once. Two decoders each covering three seconds finish in the time one would
/// take to cover six.
pub struct BackFill {
    frames: std::sync::mpsc::Receiver<Frame>,
    /// Frames gathered so far. Held rather than inserted as they arrive: they
    /// come out earliest-first, and the cache can only take a run that meets
    /// its own front, so the run is completed before it is handed over.
    pending: Vec<Frame>,
    /// Where the run must reach.
    to: Duration,
    _stop: std::sync::Arc<()>,
}

impl BackFill {
    /// Start decoding the `span` of film before `to`.
    ///
    /// Software decoding by preference: the GPU is drawing the film, and this
    /// is work nobody is waiting on. Some codecs leave no choice — this build
    /// has no software AV1 decoder, because FFmpeg's own is hardware-only and
    /// the software ones are separate libraries we do not carry — so a
    /// decoder that produces nothing is retried on the GPU rather than
    /// silently filling nothing.
    pub fn start(path: &std::path::Path, to: Duration, span: Duration) -> Option<BackFill> {
        let from = to.checked_sub(span)?;
        let (sender, frames) = std::sync::mpsc::channel();
        let stop = std::sync::Arc::new(());
        let watch = std::sync::Arc::downgrade(&stop);
        let owned = path.to_path_buf();

        std::thread::Builder::new()
            .name("backfill".into())
            .spawn(move || {
                for hardware in [false, true] {
                    if watch.upgrade().is_none() {
                        return;
                    }
                    if fill(&owned, from, to, hardware, &sender, &watch) {
                        return;
                    }
                }
            })
            .ok()?;

        Some(BackFill { frames, pending: Vec::new(), to, _stop: stop })
    }
}

/// Decode `from`..`to` and send what falls inside. Returns whether it worked.
fn fill(
    path: &std::path::Path,
    from: Duration,
    to: Duration,
    hardware: bool,
    sender: &std::sync::mpsc::Sender<Frame>,
    watch: &std::sync::Weak<()>,
) -> bool {
    use std::time::Instant;
    /// Long enough for a seek and a first frame on any codec this plays.
    const FIRST_FRAME: Duration = Duration::from_secs(2);

    let Ok(mut player) = crate::video::Player::open_video_only(path, hardware) else {
        return false;
    };
    player.seek(from, false);

    let started = Instant::now();
    let mut produced = false;
    loop {
        if watch.upgrade().is_none() {
            return true;
        }
        match player.try_next() {
            // Past the point the cache already covers.
            Some(frame) if frame.at >= to => return true,
            Some(frame) if frame.at < from => produced = true,
            Some(frame) => {
                produced = true;
                if sender.send(frame).is_err() {
                    return true;
                }
            }
            None if player.finished() => return produced,
            None => {
                // Nothing at all after a generous wait means this decoder
                // cannot handle the codec; say so rather than filling nothing.
                if !produced && started.elapsed() > FIRST_FRAME {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

impl BackFill {
    /// What the run covers so far.
    pub fn gathered(&self) -> Duration {
        match (self.pending.first(), self.pending.last()) {
            (Some(first), Some(last)) => last.at.saturating_sub(first.at),
            _ => Duration::ZERO,
        }
    }

    pub fn to(&self) -> Duration {
        self.to
    }

    /// Collect what has been decoded. Returns the run once it is complete.
    pub fn collect(&mut self) -> Option<Vec<Frame>> {
        loop {
            match self.frames.try_recv() {
                Ok(frame) => self.pending.push(frame),
                Err(std::sync::mpsc::TryRecvError::Empty) => return None,
                // The thread has finished; the run is as complete as it will
                // get, and a short one is still worth having.
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Some(std::mem::take(&mut self.pending));
                }
            }
        }
    }
}
