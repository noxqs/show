//! Sound output, and the clock that video follows.
//!
//! Two pieces. [`Sink`] owns a writer thread that keeps the device fed from a
//! ring buffer; [`Clock`] answers "what is audible right now", which is the
//! timeline video is presented against once there is sound to follow.
//!
//! The clock is anchored rather than counted. Recording that a particular
//! sample index carries a particular media timestamp, and then measuring how
//! many samples have gone past it, survives a decoder that starts at a nonzero
//! timestamp and a device whose buffer is deep — both of which a plain
//! "samples played so far" counter gets wrong.

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::alsa::Pcm;

/// Everything is resampled to this before it reaches the device. Picking one
/// rate keeps the resampler configuration fixed for the life of a file.
pub const RATE: u32 = 48_000;
pub const CHANNELS: u32 = 2;

/// How much audio to keep queued ahead of the device.
///
/// Deep on purpose: this is what stops a scheduling hiccup in the decode thread
/// from becoming an audible gap. Half a second of stereo floats is 192 KB,
/// which is nothing beside a single video frame.
const BUFFER: Duration = Duration::from_millis(500);

/// How much the device itself holds. Smaller means pause and volume take
/// effect sooner; too small and every scheduling delay is an underrun.
const DEVICE_LATENCY: Duration = Duration::from_millis(80);

/// Samples handed to the device per write.
const CHUNK_FRAMES: usize = 1024;

struct Shared {
    ring: Mutex<VecDeque<f32>>,
    /// Signalled when the writer frees space, and when the producer adds some.
    changed: Condvar,
    capacity: usize,
    shutdown: AtomicBool,
    /// Set by a seek: the writer drops what the device is holding, so the new
    /// position is heard now rather than after the old buffer drains.
    flush: AtomicBool,
    /// Media time before which pushed sound is dropped, as `f64` bits; zero
    /// for none. Set by a skip further than the ring reaches.
    floor: AtomicU64,
    /// Bumped by every skip, so a push waiting for room knows the rest of
    /// what it holds may now be behind the sound.
    skips: AtomicU64,
}

/// What is audible now, in media time.
pub struct Clock {
    /// When the counters below were last written, as nanoseconds since
    /// `base`. The device is only heard from once per write, so between
    /// writes the position is carried forward by the wall clock; without that
    /// the timeline is a staircase with a step per block, and presenting
    /// frames against a staircase is visible as judder.
    updated_ns: AtomicU64,
    base: std::time::Instant,
    /// Frames handed to the device.
    consumed: AtomicU64,
    /// Media timestamp of frame `anchor_frame`, as `f64` bits.
    anchor_pts: AtomicU64,
    anchor_frame: AtomicU64,
    /// What the device is still holding, in frames.
    latency_frames: AtomicI64,
    valid: AtomicBool,
    playing: AtomicBool,
    /// Set when the device was fed silence because the decoder had not kept
    /// up. The next real data re-anchors the timeline against it.
    underran: AtomicBool,
    /// How many times that has happened, and how many blocks have gone out.
    /// Every underrun is an audible gap and a re-anchored timeline, so the
    /// count is the difference between "playing" and "limping".
    pub underruns: AtomicU64,
    pub blocks: AtomicU64,
}

impl Clock {
    fn new() -> Clock {
        Clock {
            updated_ns: AtomicU64::new(0),
            base: std::time::Instant::now(),
            consumed: AtomicU64::new(0),
            anchor_pts: AtomicU64::new(0),
            anchor_frame: AtomicU64::new(0),
            latency_frames: AtomicI64::new(0),
            valid: AtomicBool::new(false),
            playing: AtomicBool::new(false),
            underran: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            blocks: AtomicU64::new(0),
        }
    }

    /// Record that the frame at `frame` carries media timestamp `pts`.
    fn anchor(&self, pts: f64, frame: u64) {
        self.anchor_pts.store(pts.to_bits(), Ordering::Relaxed);
        self.anchor_frame.store(frame, Ordering::Relaxed);
        self.valid.store(true, Ordering::Release);
    }

    /// Gate the device.
    ///
    /// Playback starts stopped, which matters more than it sounds: a device
    /// that began consuming the moment it opened would be ahead of the first
    /// decoded video frame, and with audio as the master the picture would
    /// spend the start of every file catching up.
    pub fn set_playing(&self, playing: bool) {
        self.playing.store(playing, Ordering::Release);
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Acquire)
    }

    /// Stop reporting a position until the next anchor. For a seek, whose new
    /// timeline only exists once the first sound from there is queued.
    pub fn forget(&self) {
        self.valid.store(false, Ordering::Release);
    }

    /// Media position currently audible, or `None` before the first anchor.
    pub fn position(&self) -> Option<Duration> {
        if !self.valid.load(Ordering::Acquire) {
            return None;
        }
        let pts = f64::from_bits(self.anchor_pts.load(Ordering::Relaxed));
        let anchor = self.anchor_frame.load(Ordering::Relaxed);
        let consumed = self.consumed.load(Ordering::Relaxed);
        let latency = self.latency_frames.load(Ordering::Relaxed);

        // Frames handed over since the anchor, less what has not been heard.
        let ahead = consumed.saturating_sub(anchor) as f64 - latency as f64;
        let mut seconds = pts + ahead / RATE as f64;

        // Carry forward from the last device update. Bounded, because a writer
        // that has stopped must not have its clock run away from the audio:
        // one block's worth is the most that can legitimately have elapsed.
        let elapsed_ns = self.base.elapsed().as_nanos() as u64;
        let since = elapsed_ns.saturating_sub(self.updated_ns.load(Ordering::Relaxed));
        let block_ns = (CHUNK_FRAMES as u64 * 1_000_000_000) / RATE as u64;
        seconds += since.min(block_ns) as f64 / 1e9;
        // Anchored but not yet moved is a real position of zero, not an
        // absent one: `None` here means "no timeline", and returning it for
        // the first instant of a file would hand the first frames back to the
        // wall clock and then switch masters underneath them.
        Some(Duration::from_secs_f64(seconds.max(0.0)))
    }
}

/// Playback gain, as a plain multiplier.
pub struct Volume(AtomicU32);

impl Volume {
    /// Above 1.0 is allowed: quiet sources are common enough that a ceiling of
    /// exactly unity would be the wrong call.
    pub const MAX: f32 = 1.5;

    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    pub fn set(&self, value: f32) {
        self.0.store(value.clamp(0.0, Self::MAX).to_bits(), Ordering::Relaxed);
    }
}

/// An open audio device being fed by the decoder.
pub struct Sink {
    shared: Arc<Shared>,
    clock: Arc<Clock>,
    volume: Arc<Volume>,
    /// Frames pushed into the ring since the file started.
    produced: u64,
    /// Whether the next push re-establishes the timeline.
    needs_anchor: bool,
}

impl Sink {
    /// Open the default device and start the writer thread.
    pub fn open() -> Result<Sink> {
        let mut pcm = Pcm::open(RATE, CHANNELS, DEVICE_LATENCY.as_micros() as u32)?;

        let capacity = (RATE as f64 * BUFFER.as_secs_f64()) as usize * CHANNELS as usize;
        let shared = Arc::new(Shared {
            ring: Mutex::new(VecDeque::with_capacity(capacity)),
            changed: Condvar::new(),
            capacity,
            shutdown: AtomicBool::new(false),
            flush: AtomicBool::new(false),
            floor: AtomicU64::new(0),
            skips: AtomicU64::new(0),
        });
        let clock = Arc::new(Clock::new());
        let volume = Arc::new(Volume(AtomicU32::new(1.0f32.to_bits())));

        let thread_shared = shared.clone();
        let thread_clock = clock.clone();
        let thread_volume = volume.clone();

        std::thread::Builder::new()
            .name("audio-out".into())
            .spawn(move || {
                let mut block = vec![0i16; CHUNK_FRAMES * CHANNELS as usize];
                let mut was_playing = false;
                while !thread_shared.shutdown.load(Ordering::Acquire) {
                    let gain = thread_volume.get();
                    let playing = thread_clock.is_playing();
                    // Starting again: the device is holding the silence it was
                    // fed while stopped, and the sound would come out only
                    // after it — with the picture, shown against this clock,
                    // stalled for as long. Drop it so the sound starts now.
                    let starting = playing && !was_playing;
                    was_playing = playing;
                    if thread_shared.flush.swap(false, Ordering::AcqRel) || starting {
                        pcm.reset();
                    }

                    let taken = if playing {
                        let mut ring = thread_shared.ring.lock().expect("audio ring poisoned");
                        // Wait for something to play rather than spinning, but
                        // wake often enough to notice a pause or a shutdown.
                        if ring.is_empty() {
                            let (guard, _) = thread_shared
                                .changed
                                .wait_timeout(ring, Duration::from_millis(10))
                                .expect("audio ring poisoned");
                            ring = guard;
                        }
                        let take = block.len().min(ring.len());
                        for (dst, src) in block[..take].iter_mut().zip(ring.drain(..take)) {
                            *dst = to_pcm(src * gain);
                        }
                        drop(ring);
                        thread_shared.changed.notify_all();
                        take
                    } else {
                        0
                    };

                    // A pause writes silence rather than stopping the device.
                    // Stopping means an underrun and a restart glitch on every
                    // resume; silence costs nothing and keeps the clock honest.
                    // The same applies to an empty ring, which is a decoder
                    // that has fallen behind, not the end of the file.
                    block[taken..].fill(0);
                    if pcm.write(&block).is_err() {
                        break;
                    }

                    thread_clock
                        .latency_frames
                        .store(pcm.delay_frames(), Ordering::Relaxed);

                    if playing {
                        // The whole block reached the device, silence included,
                        // so the whole block is time that has passed. Counting
                        // only the real samples would freeze the clock exactly
                        // when the decoder was behind — and since video is
                        // shown against this clock, a frozen clock stops the
                        // picture, which stops the queue draining, which stops
                        // the decoder for good.
                        thread_clock
                            .consumed
                            .fetch_add((block.len() / CHANNELS as usize) as u64, Ordering::Relaxed);
                        // Published last, and after the counters it refers
                        // to, so a reader interpolating from it can never be
                        // extrapolating from a stale count.
                        thread_clock.updated_ns.store(
                            thread_clock.base.elapsed().as_nanos() as u64,
                            Ordering::Release,
                        );
                        thread_clock.blocks.fetch_add(1, Ordering::Relaxed);
                        if taken < block.len() {
                            thread_clock.underruns.fetch_add(1, Ordering::Relaxed);
                            // Time advanced without content, so the mapping
                            // from samples to timestamps is now wrong by
                            // however much silence went out. Re-anchor on what
                            // arrives next rather than carrying the error.
                            thread_clock.underran.store(true, Ordering::Release);
                        }
                    }
                }
            })?;

        Ok(Sink { shared, clock, volume, produced: 0, needs_anchor: true })
    }

    pub fn clock(&self) -> Arc<Clock> {
        self.clock.clone()
    }

    /// Throw away everything queued, here and in the device, and forget the
    /// timeline. Called on a seek: the audio for the old position is no longer
    /// wanted, and playing it out first is exactly the lag a seek is meant to
    /// avoid.
    pub fn flush(&mut self) {
        self.shared.ring.lock().expect("audio ring poisoned").clear();
        self.shared.flush.store(true, Ordering::Release);
        self.shared.floor.store(0, Ordering::Release);
        self.shared.changed.notify_all();
        // The counters only mean anything together: the clock maps consumed
        // frames onto an anchor, so dropping queued audio without resetting
        // both would leave the two describing different moments.
        self.clock.valid.store(false, Ordering::Release);
        self.clock.consumed.store(0, Ordering::Relaxed);
        self.clock.anchor_frame.store(0, Ordering::Relaxed);
        self.clock.underran.store(false, Ordering::Release);
        self.produced = 0;
        self.needs_anchor = true;
    }

    pub fn volume(&self) -> Arc<Volume> {
        self.volume.clone()
    }

    /// A handle for moving the queued sound forward from another thread.
    pub fn skipper(&self) -> Skipper {
        Skipper { shared: self.shared.clone(), clock: self.clock.clone() }
    }

    /// Queue interleaved stereo samples carrying media timestamp `pts`.
    ///
    /// Blocks while the ring is full, which is the backpressure that stops the
    /// decoder racing ahead of the speaker. Returns false once the sink is
    /// shutting down, which is the producer's cue to stop.
    ///
    /// `stale` is asked while waiting for room, and a true answer abandons
    /// the rest. A seek holds the device stopped until the new position is on
    /// screen, so a push of the old position's sound into a full ring would
    /// otherwise wait forever — and so would the seek, behind it.
    pub fn push(&mut self, mut samples: &[f32], mut pts: Duration, stale: impl Fn() -> bool) -> bool {
        let floor = f64::from_bits(self.shared.floor.load(Ordering::Acquire));
        if floor > 0.0 {
            // Skipped past: drop it, or the part of it that falls short.
            let frames = samples.len() / CHANNELS as usize;
            let at = pts.as_secs_f64();
            if at + frames as f64 / RATE as f64 <= floor {
                return true;
            }
            let skip = (((floor - at) * RATE as f64).round().max(0.0) as usize).min(frames);
            samples = &samples[skip * CHANNELS as usize..];
            pts = pts.max(Duration::from_secs_f64(floor));
            self.shared.floor.store(0, Ordering::Release);
        }
        if samples.is_empty() {
            return true;
        }
        let skips = self.shared.skips.load(Ordering::Acquire);
        // Anchor before queueing, so the timestamp names this chunk's first
        // frame rather than one somewhere inside the last one.
        if self.clock.underran.swap(false, Ordering::AcqRel) {
            self.needs_anchor = true;
        }
        if self.needs_anchor {
            // After silence the device has consumed frames the producer never
            // supplied, so the anchor is placed at the device's count rather
            // than the producer's.
            let at = self
                .clock
                .consumed
                .load(Ordering::Relaxed)
                .max(self.produced);
            self.clock.anchor(pts.as_secs_f64(), at);
            self.produced = at;
            self.needs_anchor = false;
        }

        let mut written = 0;
        while written < samples.len() {
            if self.shared.shutdown.load(Ordering::Acquire) {
                return false;
            }
            if stale() {
                return true;
            }
            // Skipped while waiting: the rest is behind the sound now. The
            // skip has asked for the next push to re-anchor.
            if self.shared.skips.load(Ordering::Acquire) != skips {
                return true;
            }
            let mut ring = self.shared.ring.lock().expect("audio ring poisoned");
            let room = self.shared.capacity.saturating_sub(ring.len());
            if room == 0 {
                let (guard, _) = self
                    .shared
                    .changed
                    .wait_timeout(ring, Duration::from_millis(100))
                    .expect("audio ring poisoned");
                drop(guard);
                continue;
            }
            let take = room.min(samples.len() - written);
            ring.extend(&samples[written..written + take]);
            drop(ring);
            self.shared.changed.notify_all();
            written += take;
        }
        self.produced += (samples.len() / CHANNELS as usize) as u64;
        true
    }
}

impl Drop for Sink {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.changed.notify_all();
    }
}

/// Moves the sound forward to meet the picture, by dropping what is queued.
///
/// Stepping frames while paused moves the picture and not the sound, and on
/// resuming the picture would wait, frozen, for the sound to catch up. The
/// sound for those frames is usually still sitting in the ring, so throwing it
/// away realigns the two at no cost — where a seek would decode the whole
/// group of pictures again.
pub struct Skipper {
    shared: Arc<Shared>,
    clock: Arc<Clock>,
}

impl Skipper {
    /// Make `to` the next thing heard. Only while stopped, when the writer is
    /// not draining the ring underneath. Sound not yet decoded that far is
    /// dropped as it arrives. False when it cannot: stopped is not the case,
    /// there is no timeline yet, or the sound is already past `to`.
    pub fn skip_to(&self, to: Duration) -> bool {
        // Close enough not to be worth a seek: well under a frame.
        const SLACK: f64 = 0.01;
        if self.clock.is_playing() || !self.clock.valid.load(Ordering::Acquire) {
            return false;
        }
        let mut ring = self.shared.ring.lock().expect("audio ring poisoned");
        // The ring's first sample, in media time. Not `position`, which
        // allows for what the device holds — and a resume throws that away.
        let pts = f64::from_bits(self.clock.anchor_pts.load(Ordering::Relaxed));
        let anchor = self.clock.anchor_frame.load(Ordering::Relaxed);
        let consumed = self.clock.consumed.load(Ordering::Relaxed);
        let head = pts + consumed.saturating_sub(anchor) as f64 / RATE as f64;
        let gap = to.as_secs_f64() - head;
        if gap < -SLACK {
            return false;
        }
        let wanted = (gap.max(0.0) * RATE as f64).round() as usize;
        let frames = wanted.min(ring.len() / CHANNELS as usize);
        ring.drain(..frames * CHANNELS as usize);
        // As if the device had played them, which keeps the anchor true.
        self.clock.consumed.fetch_add(frames as u64, Ordering::Relaxed);
        if frames < wanted {
            // Further than is queued: what arrives next is dropped up to
            // `to`, and anchors the timeline afresh where it starts.
            self.shared.floor.store(to.as_secs_f64().to_bits(), Ordering::Release);
            self.shared.skips.fetch_add(1, Ordering::AcqRel);
            self.clock.underran.store(true, Ordering::Release);
        }
        drop(ring);
        self.shared.changed.notify_all();
        true
    }
}

/// Convert one sample to the device's format, clipping rather than wrapping.
///
/// Wrapping is the difference between a loud passage sounding loud and it
/// sounding like tearing paper, and gain above unity makes it reachable.
fn to_pcm(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The most a position may legitimately have run on past the arithmetic
    /// its inputs describe.
    ///
    /// `position` deliberately carries wall time forward from the last device
    /// update, bounded at one block, so that the picture keeps moving between
    /// them. A test that demands the exact arithmetic is therefore a test that
    /// fails whenever the machine takes more than a microsecond to reach the
    /// next line — which these did, on any loaded machine, for as long as they
    /// asserted to 1e-6. **The tolerance a clock needs is the slack the clock
    /// is documented to have.**
    const CARRY: f64 = CHUNK_FRAMES as f64 / RATE as f64;

    /// Assert a position, allowing that carry — and only in the direction it
    /// can actually go. It is added, never subtracted, so a clock running
    /// *behind* its anchor is still a failure here.
    #[track_caller]
    fn assert_at(position: Duration, expected: f64) {
        let at = position.as_secs_f64();
        assert!(
            at >= expected - 1e-9 && at <= expected + CARRY,
            "expected {expected}s, plus up to {CARRY}s of carry; got {at}s"
        );
    }

    #[test]
    fn loud_samples_clip_rather_than_wrap() {
        assert_eq!(to_pcm(2.0), i16::MAX);
        assert_eq!(to_pcm(-2.0), -i16::MAX);
        assert_eq!(to_pcm(0.0), 0);
    }

    #[test]
    fn volume_is_bounded_at_both_ends() {
        let v = Volume(AtomicU32::new(1.0f32.to_bits()));
        v.set(-1.0);
        assert_eq!(v.get(), 0.0);
        v.set(99.0);
        assert_eq!(v.get(), Volume::MAX);
    }

    #[test]
    fn the_clock_is_silent_until_anchored() {
        let clock = Clock::new();
        assert!(clock.position().is_none(), "no anchor means no timeline");
        clock.anchor(2.0, 0);
        clock.consumed.store(RATE as u64, Ordering::Relaxed);
        // One second of frames past an anchor at two seconds.
        assert_at(clock.position().expect("anchored"), 3.0);
    }

    #[test]
    fn device_latency_is_subtracted_from_the_position() {
        let clock = Clock::new();
        clock.anchor(0.0, 0);
        clock.consumed.store(RATE as u64, Ordering::Relaxed);
        // Half a second still sitting in the device buffer has not been heard.
        clock.latency_frames.store(RATE as i64 / 2, Ordering::Relaxed);
        assert_at(clock.position().expect("anchored"), 0.5);
    }

    /// The bug this pins: video is shown against this clock, so a clock that
    /// stops while the decoder is behind stops the picture, which stops the
    /// frame queue draining, which stops the decoder — permanently.
    #[test]
    fn the_clock_keeps_moving_when_the_decoder_falls_behind() {
        let clock = Clock::new();
        clock.anchor(0.0, 0);
        let before = clock.position().expect("anchored");
        // A block of silence went to the device because nothing was queued.
        clock.consumed.fetch_add(1024, Ordering::Relaxed);
        clock.underran.store(true, Ordering::Release);
        let after = clock.position().expect("anchored");
        assert!(after > before, "the clock must not freeze on an underrun");
    }

    #[test]
    fn re_anchoring_after_silence_uses_what_the_device_consumed() {
        let clock = Clock::new();
        clock.anchor(0.0, 0);
        // Half a second of silence went out while the decoder caught up.
        clock.consumed.store(RATE as u64 / 2, Ordering::Relaxed);

        // The data that arrives next belongs at 5 s, and is anchored against
        // the device's count rather than the producer's.
        let at = clock.consumed.load(Ordering::Relaxed);
        clock.anchor(5.0, at);
        // The new data must play at its own timestamp, not from the start.
        assert_at(clock.position().expect("anchored"), 5.0);
    }

    #[test]
    fn an_anchor_partway_in_does_not_replay_the_start() {
        let clock = Clock::new();
        // The timeline starts at ten seconds, at frame 1000.
        clock.anchor(10.0, 1000);
        clock.consumed.store(1000 + RATE as u64, Ordering::Relaxed);
        assert_at(clock.position().expect("anchored"), 11.0);
    }
}
