//! Video playback.
//!
//! Decoding runs on its own thread and pushes frames into a small bounded
//! queue. Bounded is the important word: a decoder left to run flat out will
//! produce frames faster than a screen can show them and turn a two-hour film
//! into an out-of-memory error, so the queue filling is what tells it to wait.
//!
//! Presentation is the caller's job. This module answers "what should be on
//! screen at time t" and nothing else — it owns no window, no clock the user
//! can see, and no notion of being paused.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::time::Duration;

use anyhow::{Context, Result};

/// How far ahead of the picture to keep decoded frames.
///
/// Matched to how far back the frame cache reaches, so stepping a frame at a
/// time is instant for the same distance either way. It is not a background
/// task and does not run ahead through the film: the decoder simply waits a
/// few seconds further on than it used to, and stops there.
const READ_AHEAD: Duration = Duration::from_secs(3);

/// Ceiling on what that read-ahead may occupy.
///
/// Three seconds is 225 MB at 1080p and 930 MB at 4K. The budget binds first
/// on the largest formats, giving them fewer seconds rather than more memory.
const READ_AHEAD_BYTES: u64 = 1 << 30;

/// The smallest useful queue: enough to ride out a slow decode of a complex
/// frame even when a frame is large enough that the budget allows few.
const MIN_QUEUE_DEPTH: usize = 8;

/// Frames to hold ahead, for a film of this shape.
fn queue_depth(width: u32, height: u32, frame_rate: f64) -> usize {
    // 4:2:0 planar: a byte of luma per pixel, half as much again for chroma.
    let frame_bytes = (width as u64 * height as u64 * 3 / 2).max(1);
    let by_time = (READ_AHEAD.as_secs_f64() * frame_rate.max(1.0)) as usize;
    let by_memory = (READ_AHEAD_BYTES / frame_bytes) as usize;
    by_time.min(by_memory).max(MIN_QUEUE_DEPTH)
}

/// Extensions we will try to open as video.
const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "m4v", "mkv", "webm", "mov", "avi", "mpg", "mpeg", "ts", "m2ts", "wmv", "flv", "ogv",
];

/// Render nodes to try for hardware decoding, in order.
///
/// Which node is which card varies by machine, and the first one is not
/// necessarily the one with a working VA driver — on this hardware the NVIDIA
/// node has none while the AMD one does. So each is tried until one takes.
const RENDER_NODES: &[&str] = &[
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/renderD130",
];

/// A VA-API device, held for as long as the decoder that uses it.
struct HwDevice(*mut ffmpeg_next::sys::AVBufferRef);

// Safety: an AVBufferRef is reference-counted and FFmpeg's own hardware
// contexts are documented as usable from the thread that owns them; this one
// is created on and confined to the decode thread.
unsafe impl Send for HwDevice {}

impl Drop for HwDevice {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ffmpeg_next::sys::av_buffer_unref(&mut self.0) };
        }
    }
}

/// Ask FFmpeg for a VA-API device on the first render node that provides one.
fn open_hw_device() -> Option<(HwDevice, String)> {
    for node in RENDER_NODES {
        if !Path::new(node).exists() {
            continue;
        }
        let Ok(c_node) = std::ffi::CString::new(*node) else { continue };
        let mut device: *mut ffmpeg_next::sys::AVBufferRef = std::ptr::null_mut();

        // Safety: FFmpeg fills `device` on success and leaves it null on
        // failure; the string is valid for the duration of the call.
        let rc = unsafe {
            ffmpeg_next::sys::av_hwdevice_ctx_create(
                &mut device,
                ffmpeg_next::sys::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                c_node.as_ptr(),
                std::ptr::null_mut(),
                0,
            )
        };
        if rc >= 0 && !device.is_null() {
            return Some((HwDevice(device), node.to_string()));
        }
    }
    None
}

/// Choose the hardware pixel format when the decoder offers one.
///
/// Without this FFmpeg falls back to software even with a device attached:
/// the callback is how a caller says it is prepared to handle GPU surfaces.
unsafe extern "C" fn pick_hw_format(
    _ctx: *mut ffmpeg_next::sys::AVCodecContext,
    formats: *const ffmpeg_next::sys::AVPixelFormat,
) -> ffmpeg_next::sys::AVPixelFormat {
    let mut candidate = formats;
    unsafe {
        while *candidate != ffmpeg_next::sys::AVPixelFormat::AV_PIX_FMT_NONE {
            if *candidate == ffmpeg_next::sys::AVPixelFormat::AV_PIX_FMT_VAAPI {
                return *candidate;
            }
            candidate = candidate.add(1);
        }
        // Nothing we can use; let FFmpeg pick, which means software.
        *formats
    }
}

/// Codecs worth naming in the capability report: the ones people actually
/// have files of, rather than everything FFmpeg knows.
const REPORTED: &[(&str, ffmpeg_next::codec::Id)] = &[
    ("H.264", ffmpeg_next::codec::Id::H264),
    ("HEVC", ffmpeg_next::codec::Id::HEVC),
    ("AV1", ffmpeg_next::codec::Id::AV1),
    ("VP9", ffmpeg_next::codec::Id::VP9),
    ("VP8", ffmpeg_next::codec::Id::VP8),
    ("MPEG-2", ffmpeg_next::codec::Id::MPEG2VIDEO),
    ("MPEG-4", ffmpeg_next::codec::Id::MPEG4),
    ("VC-1", ffmpeg_next::codec::Id::VC1),
    ("WMV3", ffmpeg_next::codec::Id::WMV3),
    ("Theora", ffmpeg_next::codec::Id::THEORA),
    ("ProRes", ffmpeg_next::codec::Id::PRORES),
    ("DNxHD", ffmpeg_next::codec::Id::DNXHD),
    ("FFV1", ffmpeg_next::codec::Id::FFV1),
    ("DV", ffmpeg_next::codec::Id::DVVIDEO),
    ("AAC", ffmpeg_next::codec::Id::AAC),
    ("MP3", ffmpeg_next::codec::Id::MP3),
    ("AC-3", ffmpeg_next::codec::Id::AC3),
    ("E-AC-3", ffmpeg_next::codec::Id::EAC3),
    ("DTS", ffmpeg_next::codec::Id::DTS),
    ("TrueHD", ffmpeg_next::codec::Id::TRUEHD),
    ("Opus", ffmpeg_next::codec::Id::OPUS),
    ("Vorbis", ffmpeg_next::codec::Id::VORBIS),
    ("FLAC", ffmpeg_next::codec::Id::FLAC),
    ("ALAC", ffmpeg_next::codec::Id::ALAC),
    ("WMA", ffmpeg_next::codec::Id::WMAV2),
];

/// Which of the reported codecs this build can decode.
pub fn capabilities() -> Vec<(&'static str, bool)> {
    let _ = ffmpeg_next::init();
    REPORTED
        .iter()
        .map(|(label, id)| (*label, ffmpeg_next::codec::decoder::find(*id).is_some()))
        .collect()
}

/// Container extensions we will try, for the report.
pub fn containers() -> &'static [&'static str] {
    VIDEO_EXTENSIONS
}

/// The picture size of a video, read from its header alone.
///
/// Cheap enough to do before the window exists, which is the point: the window
/// is sized to the film rather than the film letterboxed into a window that
/// was already the wrong shape.
pub fn probe_size(path: &Path) -> Option<(u32, u32)> {
    ffmpeg_next::init().ok()?;
    let input = ffmpeg_next::format::input(path).ok()?;
    let stream = input.streams().best(ffmpeg_next::media::Type::Video)?;
    // Safety: reading two ints from parameters owned by the open input.
    let (w, h) = unsafe {
        let raw = stream.parameters().as_ptr();
        ((*raw).width.max(0) as u32, (*raw).height.max(0) as u32)
    };
    (w > 0 && h > 0).then_some((w, h))
}

pub fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// One decoded frame, as three planes ready to upload.
///
/// Kept planar deliberately. A 4:2:0 frame is 1.5 bytes per pixel against the
/// 4 that RGBA would need, so this is both less conversion work and less than
/// half the upload — and the GPU does the colour transform for nothing.
pub struct Frame {
    /// Whether this frame is a keyframe — one decoding could have started at.
    /// Stepping backwards needs to know where the current one began, since
    /// the play position on its own says nothing about that.
    pub key: bool,
    /// Which seek generation produced this. Frames from before a seek are
    /// still in flight when it happens, and showing them would make the
    /// picture jump back before it jumps forward.
    pub generation: u64,
    /// Luma, then the two chroma planes, each tightly packed.
    ///
    /// Reference counted so a frame can be handed to more than one consumer
    /// without copying it. Three megabytes at 1080p is a memcpy nobody
    /// notices once, and a memcpy that matters at twenty-five a second — the
    /// frame cache holds them, the display uploads them, and analysis wants
    /// them too. `Arc` makes all three free; nothing here ever mutates a
    /// plane after `pack` builds it.
    pub planes: [std::sync::Arc<Vec<u8>>; 3],
    pub width: u32,
    pub height: u32,
    /// Chroma plane dimensions, which are smaller when subsampled.
    pub chroma: (u32, u32),
    /// How to get from YUV to RGB for this frame.
    pub colour: Colour,
    /// When this frame should appear, measured from the start of the file.
    pub at: Duration,
}

/// The colour transform a frame needs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Colour {
    /// BT.709 for HD and above, BT.601 below it.
    pub hd: bool,
    /// Full-range (0-255) rather than the usual studio range (16-235).
    pub full_range: bool,
}

impl Frame {
    /// A copy that shares the planes rather than duplicating them.
    ///
    /// Three atomic increments. This used to be a three-megabyte memcpy,
    /// acceptable only because stepping happens at the speed of a key press;
    /// sharing makes it cheap enough to do per frame, which is what lets a
    /// frame reach analysis without slowing the picture down.
    pub fn clone_ref(&self) -> Frame {
        Frame {
            key: self.key,
            generation: self.generation,
            planes: self.planes.clone(),
            width: self.width,
            height: self.height,
            chroma: self.chroma,
            at: self.at,
            colour: self.colour,
        }
    }
}

impl Colour {
    /// Matrix rows and the offsets subtracted before applying them.
    ///
    /// Studio-range video does not use the whole byte, so luma is rescaled from
    /// 16..235 and chroma from 16..240. Folding that into the matrix keeps the
    /// shader to one subtract and three dot products.
    pub fn matrix(&self) -> ([[f32; 3]; 3], [f32; 2]) {
        // Luma coefficients: BT.709 vs BT.601.
        let (kr, kb) = if self.hd { (0.2126, 0.0722) } else { (0.299, 0.114) };
        let kg = 1.0 - kr - kb;

        let (y_scale, c_scale, y_offset) = if self.full_range {
            (1.0, 1.0, 0.0)
        } else {
            (255.0 / 219.0, 255.0 / 224.0, 16.0 / 255.0)
        };

        let rows = [
            [y_scale, 0.0, c_scale * 2.0 * (1.0 - kr)],
            [
                y_scale,
                -c_scale * 2.0 * kb * (1.0 - kb) / kg,
                -c_scale * 2.0 * kr * (1.0 - kr) / kg,
            ],
            [y_scale, c_scale * 2.0 * (1.0 - kb), 0.0],
        ];
        (rows, [y_offset, 0.5])
    }
}

/// What the file is, known before the first frame arrives.
#[derive(Clone, Debug)]
pub struct Info {
    pub width: u32,
    pub height: u32,
    pub duration: Option<Duration>,
    pub frame_rate: f64,
    pub codec: String,
    /// The render node decoding this, when hardware decoding is in use.
    pub hardware: Option<String>,
}

pub struct Player {
    frames: Receiver<Frame>,
    /// Bumped on every seek; frames stamped with an older value are dropped.
    generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Where an exact seek asked to land, in seconds as `f64` bits, or zero.
    /// Both decoders skip what falls short of it: the sound so it starts
    /// where the picture does, the picture so frames nobody will see are not
    /// downloaded and copied. Written before the generation is bumped, so a
    /// decoder reads the value belonging to the seek it is acting on.
    exact_from: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// The latest seek not yet acted on. A slot rather than a queue: only
    /// the last of a burst matters, and a slot cannot be full.
    seeks: std::sync::Arc<std::sync::Mutex<Option<Seek>>>,
    cues: Receiver<crate::subtitles::Cue>,
    /// Subtitle streams found in the container, for the track menu.
    pub subtitle_tracks: Vec<crate::subtitles::Track>,
    /// Set once the decoder has handed over the last frame in the file, and
    /// cleared by a seek. The threads outlive the end of the film, so the
    /// channel closing can no longer be what says it is over.
    ended: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Frames decoded and waiting. Counted rather than asked of the channel,
    /// which does not report its own depth.
    queued: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    depth: usize,
    /// Dropped to tell the decode thread to stop.
    _stop: std::sync::Arc<()>,
    pub info: Info,
    pub path: PathBuf,
    /// Present only when the file has sound and a device took it.
    audio: Option<AudioTrack>,
}

/// The handles the player keeps once audio is running: the timeline video is
/// shown against, and the gain the user can turn.
pub struct AudioTrack {
    pub clock: std::sync::Arc<crate::audio::Clock>,
    pub volume: std::sync::Arc<crate::audio::Volume>,
    pub skipper: crate::audio::Skipper,
    pub codec: String,
    pub channels: u16,
    pub rate: u32,
}

impl Player {
    /// Open a file and start decoding into the queue.
    pub fn open(path: &Path, hardware: bool) -> Result<Player> {
        Player::open_with(path, hardware, true)
    }

    /// Open without sound.
    ///
    /// A decoder filling a cache ahead of the picture wants no part of the
    /// audio device — there is only one, the film already owns it, and the
    /// sink's backpressure would hold this decoder to real time when the whole
    /// point is to run faster than that.
    pub fn open_video_only(path: &Path, hardware: bool) -> Result<Player> {
        Player::open_with(path, hardware, false)
    }

    /// Whether hardware decoding is actually in use, as opposed to asked for.
    /// Frames decoded and waiting ahead of the picture, and the most there
    /// could be.
    pub fn read_ahead(&self) -> (usize, usize) {
        (self.queued.load(std::sync::atomic::Ordering::Relaxed).min(self.depth), self.depth)
    }

    /// Bytes one decoded frame occupies, for reporting.
    pub fn frame_bytes(&self) -> u64 {
        self.info.width as u64 * self.info.height as u64 * 3 / 2
    }

    pub fn is_hardware(&self) -> bool {
        self.info.hardware.is_some()
    }

    fn open_with(path: &Path, hardware: bool, want_audio: bool) -> Result<Player> {
        // Idempotent, and cheap after the first call.
        ffmpeg_next::init().map_err(|e| anyhow::anyhow!("ffmpeg init: {e}"))?;

        let input = ffmpeg_next::format::input(path)
            .map_err(|e| anyhow::anyhow!("opening {}: {e}", path.display()))?;
        let stream = input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .context("no video stream")?;

        let index = stream.index();
        let time_base = f64::from(stream.time_base());
        let context = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| anyhow::anyhow!("codec parameters: {e}"))?;
        let codec = context.id().name().to_string();

        // The device has to be attached before the codec is opened, which is
        // why the raw context is reached for here rather than after.
        let device = hardware.then(open_hw_device).flatten();
        let mut context = context;
        if let Some((device, _)) = device.as_ref() {
            // Safety: the context is not yet open, and the device outlives the
            // decoder because both are moved into the decode thread together.
            unsafe {
                let raw = context.as_mut_ptr();
                (*raw).hw_device_ctx = ffmpeg_next::sys::av_buffer_ref(device.0);
                (*raw).get_format = Some(pick_hw_format);
            }
        }
        let node = device.as_ref().map(|(_, node)| node.clone());

        let decoder = context.decoder().video().map_err(|e| anyhow::anyhow!("decoder: {e}"))?;

        let info = Info {
            width: decoder.width(),
            height: decoder.height(),
            duration: (input.duration() > 0).then(|| {
                Duration::from_secs_f64(
                    input.duration() as f64 / f64::from(ffmpeg_next::ffi::AV_TIME_BASE),
                )
            }),
            frame_rate: {
                let r = f64::from(stream.avg_frame_rate());
                if r > 0.0 { r } else { 25.0 }
            },
            codec,
            hardware: node,
        };

        // Sound is opened before the thread starts so a failure is reported
        // against the file being opened rather than from a worker later.
        let (audio_setup, audio_track) = match want_audio.then(|| open_audio(&input)).flatten() {
            Some((setup, track)) => (Some(setup), Some(track)),
            None => (None, None),
        };

        // Subtitles are read from the same packet stream as everything else.
        let (subtitle_tracks, subtitle_decoders) =
            crate::subtitles::open_embedded(&input, (info.width, info.height));
        // Unbounded: captions are tiny and sparse, and blocking the decoder on
        // one would be a strange way to lose a film.
        let (cue_sender, cues) = std::sync::mpsc::channel();

        let depth = queue_depth(info.width, info.height, info.frame_rate);
        let (sender, frames) = std::sync::mpsc::sync_channel(depth);
        let seeks = std::sync::Arc::new(std::sync::Mutex::new(None));
        let seek_requests = seeks.clone();
        let generation = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let exact_from = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let ended = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let queued = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stop = std::sync::Arc::new(());
        let watch = std::sync::Arc::downgrade(&stop);

        // Three threads, because one cannot do this job. Reading, decoding
        // pictures and decoding sound all block on different things, and a
        // single thread doing all three stops reading whenever any one of them
        // is full — which starves the others. See PACKET_QUEUE.
        let audio_index = audio_setup.as_ref().map(|s| s.index);
        let (video_packets, video_rx) = std::sync::mpsc::sync_channel::<Msg>(PACKET_QUEUE);
        let audio_channel = audio_setup
            .is_some()
            .then(|| std::sync::mpsc::sync_channel::<Msg>(PACKET_QUEUE));
        let (audio_packets, audio_rx) = match audio_channel {
            Some((tx, rx)) => (Some(tx), Some(rx)),
            None => (None, None),
        };

        let owned_path = path.to_path_buf();
        let demux_generation = generation.clone();
        let demux_watch = watch.clone();
        std::thread::Builder::new()
            .name("demux".into())
            .spawn(move || {
                if let Err(e) = demux_loop(
                    &owned_path,
                    index,
                    audio_index,
                    video_packets,
                    audio_packets,
                    subtitle_decoders,
                    cue_sender,
                    seek_requests,
                    demux_generation,
                    demux_watch,
                ) {
                    eprintln!("show: reading stopped: {e:#}");
                }
            })
            .context("spawning the demux thread")?;

        let video_generation = generation.clone();
        let video_from = exact_from.clone();
        let video_ended = ended.clone();
        let video_queued = queued.clone();
        let video_watch = watch.clone();
        std::thread::Builder::new()
            .name("video-decode".into())
            .spawn(move || {
                // The device is moved in so it outlives every frame it backs.
                let _device = device;
                if let Err(e) =
                    video_loop(
                        video_rx,
                        time_base,
                        decoder,
                        sender,
                        video_generation,
                        video_ended,
                        video_queued,
                        video_from,
                        video_watch,
                    )
                {
                    eprintln!("show: video decoding stopped: {e:#}");
                }
            })
            .context("spawning the video thread")?;

        if let (Some(setup), Some(rx)) = (audio_setup, audio_rx) {
            let audio_generation = generation.clone();
            let audio_start = exact_from.clone();
            std::thread::Builder::new()
                .name("audio-decode".into())
                .spawn(move || audio_loop(rx, setup, audio_generation, audio_start, watch))
                .context("spawning the audio thread")?;
        }

        Ok(Player {
            frames,
            _stop: stop,
            info,
            path: path.to_path_buf(),
            audio: audio_track,
            generation,
            exact_from,
            seeks,
            cues,
            subtitle_tracks,
            ended,
            queued,
            depth,
        })
    }

    /// The next frame, if one is ready. Never blocks.
    pub fn try_next(&mut self) -> Option<Frame> {
        loop {
            match self.frames.try_recv() {
                // Frames decoded before the last seek are stale by definition.
                Ok(frame) if frame.generation != self.generation() => {
                    self.queued.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
                Ok(frame) => {
                    self.queued.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    return Some(frame);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return None,
                // Only reached if a decode thread stopped outright.
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.ended.store(true, std::sync::atomic::Ordering::Release);
                    return None;
                }
            }
        }
    }

    fn generation(&self) -> u64 {
        self.generation.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Jump to `to`, clamped to the file.
    ///
    /// Returns where it will land. The decode thread does the work; this only
    /// invalidates what is already in flight so the old position cannot appear
    /// after the new one. `forward` rounds up to the next keyframe rather than
    /// down to the last one.
    pub fn seek(&mut self, to: Duration, forward: bool) -> Duration {
        self.seek_from(to, forward, false)
    }

    /// Jump to exactly `to`: the caller decodes past the frames between the
    /// keyframe and there, so the sound skips them too.
    ///
    /// Without this the sound starts at the keyframe — seconds early, and on
    /// some files at the very start — while the picture waits on the target,
    /// frozen until the audio has caught up to it.
    pub fn seek_exact(&mut self, to: Duration, forward: bool) -> Duration {
        self.seek_from(to, forward, true)
    }

    fn seek_from(&mut self, to: Duration, forward: bool, exact: bool) -> Duration {
        let to = match self.info.duration {
            // Never quite the end: seeking exactly to it lands past the last
            // frame and the file simply reports itself finished.
            Some(d) => to.min(d.saturating_sub(Duration::from_millis(500))),
            None => to,
        };
        let from = if exact { to.as_secs_f64() } else { 0.0 };
        self.exact_from.store(from.to_bits(), std::sync::atomic::Ordering::Release);
        // The clock still describes the old position until the audio thread
        // gets round to flushing. Read in the meantime, it would show the
        // first frames of the new place against the old place's timeline.
        if let Some(audio) = self.audio.as_ref() {
            audio.clock.forget();
        }
        {
            // Bumped under the lock, so a reader that notices the new
            // generation always finds the request that goes with it.
            let mut slot = self.seeks.lock().expect("seek slot poisoned");
            let generation = self.generation.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
            *slot = Some(Seek { to, forward, generation });
        }
        // Seeking back into a film that has ended un-ends it.
        self.ended.store(false, std::sync::atomic::Ordering::Release);
        to
    }

    /// True once the decoder has run out of file and the queue is drained.
    pub fn finished(&self) -> bool {
        self.ended.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn audio(&self) -> Option<&AudioTrack> {
        self.audio.as_ref()
    }

    /// Captions decoded since the last call.
    pub fn try_cues(&mut self) -> Vec<crate::subtitles::Cue> {
        self.cues.try_iter().collect()
    }

    /// What is audible now, or `None` when nothing is.
    ///
    /// `None` covers a silent file and, just as importantly, one whose device
    /// has not started yet — the caller must fall back to its own clock in
    /// both cases. A stopped device's position never moves, and video shown
    /// against a clock that never moves is video that never starts: the first
    /// frame is not due, so the device is never started, so the clock never
    /// moves. Opus reaches that state routinely, because its pre-skip puts the
    /// first audio timestamp a little after the first picture's.
    pub fn audio_position(&self) -> Option<Duration> {
        let audio = self.audio.as_ref()?;
        if !audio.clock.is_playing() {
            return None;
        }
        audio.clock.position()
    }

    /// Start or stop the device. Video-only files ignore this.
    pub fn set_playing(&self, playing: bool) {
        if let Some(audio) = self.audio.as_ref() {
            audio.clock.set_playing(playing);
        }
    }
}

/// Everything the decode thread needs to turn packets into sound.
struct AudioSetup {
    index: usize,
    time_base: f64,
    decoder: ffmpeg_next::decoder::Audio,
    resampler: ffmpeg_next::software::resampling::Context,
    sink: crate::audio::Sink,
    /// Media time before which decoded sound is thrown away, set by an exact
    /// seek and cleared once reached.
    skip_until: Option<f64>,
}

/// Open the best audio stream and a device for it.
///
/// Every failure here is soft: a file whose sound we cannot play is still a
/// file worth watching, so this reports and returns `None` rather than
/// refusing to open the video.
fn open_audio(input: &ffmpeg_next::format::context::Input) -> Option<(AudioSetup, AudioTrack)> {
    let stream = input.streams().best(ffmpeg_next::media::Type::Audio)?;
    let index = stream.index();
    let time_base = f64::from(stream.time_base());

    let context = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|e| eprintln!("show: audio parameters: {e}"))
        .ok()?;
    let codec = context.id().name().to_string();
    let decoder = context
        .decoder()
        .audio()
        .map_err(|e| eprintln!("show: audio decoder: {e}"))
        .ok()?;

    let resampler = resampler_for(&decoder)
        .map_err(|e| eprintln!("show: resampler: {e}"))
        .ok()?;

    let sink = crate::audio::Sink::open()
        .map_err(|e| eprintln!("show: {e:#}"))
        .ok()?;

    let track = AudioTrack {
        clock: sink.clock(),
        volume: sink.volume(),
        skipper: sink.skipper(),
        codec,
        channels: decoder.channels(),
        rate: decoder.rate(),
    };
    Some((AudioSetup { index, time_base, decoder, resampler, sink, skip_until: None }, track))
}

/// A converter from what the decoder produces to what the sink takes.
fn resampler_for(
    decoder: &ffmpeg_next::decoder::Audio,
) -> Result<ffmpeg_next::software::resampling::Context, ffmpeg_next::Error> {
    // A stream that never declared its layout still has a channel count.
    let layout = if decoder.channel_layout().is_empty() {
        ffmpeg_next::util::channel_layout::ChannelLayout::default(decoder.channels() as i32)
    } else {
        decoder.channel_layout()
    };
    ffmpeg_next::software::resampling::Context::get(
        decoder.format(),
        layout,
        decoder.rate(),
        ffmpeg_next::format::Sample::F32(ffmpeg_next::format::sample::Type::Packed),
        ffmpeg_next::util::channel_layout::ChannelLayout::STEREO,
        crate::audio::RATE,
    )
}

/// Decode one audio packet's worth of frames and queue them.
///
/// Returns false when the sink has gone away, which ends playback.
fn feed_audio(
    setup: &mut AudioSetup,
    packet: &ffmpeg_next::codec::packet::Packet,
    decoded: &mut ffmpeg_next::util::frame::Audio,
    resampled: &mut ffmpeg_next::util::frame::Audio,
    stale: &dyn Fn() -> bool,
) -> bool {
    if setup.decoder.send_packet(packet).is_err() {
        return true;
    }
    while setup.decoder.receive_frame(decoded).is_ok() {
        let pts = decoded.pts().unwrap_or(0).max(0);
        let mut at = pts as f64 * setup.time_base;
        // Room for everything this frame converts to, plus whatever the
        // resampler is still holding. Left to itself, `run` sizes the output
        // by the *input's* sample count, which upsampling overflows: 22.05 kHz
        // to 48 kHz emitted under half of each frame, the rest piled up inside
        // the resampler, and the sound fell further behind the picture with
        // every frame.
        // Safety: a live context, and a count the decoder just produced.
        let room = unsafe {
            ffmpeg_next::ffi::swr_get_out_samples(setup.resampler.as_mut_ptr(), decoded.samples() as i32)
        };
        *resampled = ffmpeg_next::util::frame::Audio::new(
            ffmpeg_next::format::Sample::F32(ffmpeg_next::format::sample::Type::Packed),
            room.max(decoded.samples() as i32) as usize,
            ffmpeg_next::util::channel_layout::ChannelLayout::STEREO,
        );
        if setup.resampler.run(decoded, resampled).is_err() {
            continue;
        }
        let mut samples = packed_f32(resampled);
        if let Some(from) = setup.skip_until {
            // Short of an exact seek's target: drop it, or the part of it
            // that falls short, so the timeline begins where the picture does.
            let rate = crate::audio::RATE as f64;
            let frames = samples.len() / crate::audio::CHANNELS as usize;
            if at + frames as f64 / rate <= from {
                continue;
            }
            let skip = (((from - at) * rate).round().max(0.0) as usize).min(frames);
            samples = &samples[skip * crate::audio::CHANNELS as usize..];
            at = at.max(from);
            setup.skip_until = None;
        }
        if !setup.sink.push(samples, Duration::from_secs_f64(at), stale) {
            return false;
        }
    }
    true
}

/// Interpret a packed float frame's buffer as samples.
///
/// Packed means one plane holding every channel interleaved, which is exactly
/// the layout the sink wants, so this is a view rather than a copy.
fn packed_f32(frame: &ffmpeg_next::util::frame::Audio) -> &[f32] {
    let bytes = frame.data(0);
    let count = frame.samples() * crate::audio::CHANNELS as usize;
    // Safety: the frame was produced by the resampler as packed f32, and
    // FFmpeg aligns its buffers well past four bytes. `count` cannot exceed
    // the plane because the plane holds exactly samples x channels floats.
    let available = bytes.len() / std::mem::size_of::<f32>();
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, count.min(available)) }
}

/// Where to jump to, and which way to round.
///
/// Decoding can only start at a keyframe, so every seek lands on one and the
/// direction decides which. Rounding down is right for scrubbing — the frame
/// you asked for is then decoded on the way past — and rounding up is what
/// makes stepping forward through keyframes possible at all.
#[derive(Clone, Copy)]
pub struct Seek {
    pub to: Duration,
    pub forward: bool,
    /// The generation packets read after this seek are stamped with.
    pub generation: u64,
}

/// A packet on its way to a decoder, tagged with the seek it belongs to.
///
/// `End` says the file has run out, which the decoders need to hear so they
/// can flush whatever they are still holding. It is a message rather than the
/// channel closing, because the threads must stay alive: a film that has
/// reached its end can still be seeked back into, and a pipeline that shut
/// itself down at the last frame leaves nothing for the seek to reach.
enum Msg {
    Packet(u64, ffmpeg_next::codec::packet::Packet),
    End(u64),
}

/// How many compressed packets may wait ahead of each decoder.
///
/// This is the number that fixes badly interleaved files. A container is free
/// to place a stretch of video packets well ahead of the audio that goes with
/// them, and reading forward is the only way to reach that audio. If the
/// demuxer blocks on decoded *frames* — eight of them, raw — it cannot read
/// forward, the sound runs dry, and the timeline video is shown against jerks
/// every time it does. Buffering *packets* instead costs almost nothing:
/// these are compressed, a few kilobytes each, so a queue this deep holds many
/// seconds of a file for a handful of megabytes.
const PACKET_QUEUE: usize = 512;

/// Read the file and hand packets to the decoders.
///
/// Deliberately does no decoding of its own beyond subtitles, which are sparse
/// enough to cost nothing. Everything else is another thread's job, so that
/// nothing downstream can stop this one reading.
#[allow(clippy::too_many_arguments)]
fn demux_loop(
    path: &Path,
    video_index: usize,
    audio_index: Option<usize>,
    video_packets: SyncSender<Msg>,
    audio_packets: Option<SyncSender<Msg>>,
    mut subtitles: crate::subtitles::Decoders,
    cues: std::sync::mpsc::Sender<crate::subtitles::Cue>,
    seeks: std::sync::Arc<std::sync::Mutex<Option<Seek>>>,
    generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
    stop: std::sync::Weak<()>,
) -> Result<()> {
    use std::sync::atomic::Ordering;

    let mut input = ffmpeg_next::format::input(path)
        .map_err(|e| anyhow::anyhow!("reopening {}: {e}", path.display()))?;

    let mut packet = ffmpeg_next::codec::packet::Packet::empty();
    // The generation of the position actually being read. Not the shared
    // counter: that moves the moment a seek is asked for, and packets read
    // from the old position before the seek is acted on would carry the new
    // number — so the decoders took them as the new position, and then the
    // stream jumped back to the keyframe underneath them.
    let mut current = generation.load(Ordering::Acquire);
    loop {
        if stop.upgrade().is_none() {
            return Ok(());
        }

        // Coalesce: only the last request in a burst is worth acting on, which
        // is what makes holding an arrow key down cheap.
        let request = seeks.lock().expect("seek slot poisoned").take();
        if let Some(request) = request {
            current = request.generation;
            let ts = (request.to.as_secs_f64() * f64::from(ffmpeg_next::ffi::AV_TIME_BASE)) as i64;
            // The range is what picks the keyframe: everything up to `ts`
            // rounds down to the last one at or before it, everything from
            // `ts` rounds up to the first one at or after.
            // The lower bound matters. With none at all, a file whose index
            // FFmpeg does not trust is seeked by guessing at byte positions,
            // and the guess lands at the start of the file — measured on
            // several files here, where every backward seek returned 0.00
            // however far in it was asked to go. Giving it a floor a few
            // seconds below the target leaves it a keyframe to find and a
            // reason not to rewind past it.
            const NEARBY: i64 = 8 * ffmpeg_next::ffi::AV_TIME_BASE as i64;
            let result = if request.forward {
                input.seek(ts, ts..)
            } else {
                input.seek(ts, ts.saturating_sub(NEARBY).max(0)..ts + 1)
            };
            if let Err(e) = result {
                // Rounding up has nowhere to go at the end of the file, which
                // is not a failure worth reporting — the film simply ends.
                if !request.forward {
                    eprintln!("show: seek failed: {e}");
                }
            }
        }

        match packet.read(&mut input) {
            Ok(()) => {}
            Err(ffmpeg_next::Error::Eof) => {
                // Tell the decoders to flush, then wait. Ending the thread
                // here would be the end of the film in a stronger sense than
                // intended: seeking back into it needs this loop alive.
                let tag = current;
                let _ = video_packets.try_send(Msg::End(tag));
                if let Some(audio) = audio_packets.as_ref() {
                    let _ = audio.try_send(Msg::End(tag));
                }
                loop {
                    if stop.upgrade().is_none() {
                        return Ok(());
                    }
                    if generation.load(Ordering::Acquire) != tag {
                        // A seek arrived; go round and act on it.
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                continue;
            }
            // A damaged packet is not the end of the file.
            Err(_) => continue,
        }
        let stream = packet.stream();
        let generation_now = current;

        if subtitles.handles(stream) {
            for cue in subtitles.decode(stream, &packet) {
                if cues.send(cue).is_err() {
                    return Ok(());
                }
            }
            continue;
        }

        let queue = if Some(stream) == audio_index {
            audio_packets.as_ref()
        } else if stream == video_index {
            Some(&video_packets)
        } else {
            None
        };
        let Some(queue) = queue else { continue };

        let mut pending = Msg::Packet(generation_now, packet.clone());
        loop {
            match queue.try_send(pending) {
                Ok(()) => break,
                Err(TrySendError::Full(returned)) => {
                    if stop.upgrade().is_none() {
                        return Ok(());
                    }
                    // A seek makes everything in hand stale; stop trying to
                    // place it and go round to act on the seek instead.
                    if generation.load(Ordering::Acquire) != generation_now {
                        break;
                    }
                    pending = returned;
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(TrySendError::Disconnected(_)) => return Ok(()),
            }
        }
    }
    // Not reachable: the loop above only leaves by returning, because the end
    // of the file is a state to wait in rather than a reason to stop.
}

/// Decode video packets into frames.
fn video_loop(
    packets: Receiver<Msg>,
    time_base: f64,
    mut decoder: ffmpeg_next::decoder::Video,
    sender: SyncSender<Frame>,
    generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ended: std::sync::Arc<std::sync::atomic::AtomicBool>,
    queued: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    exact_from: std::sync::Arc<std::sync::atomic::AtomicU64>,
    stop: std::sync::Weak<()>,
) -> Result<()> {
    use std::sync::atomic::Ordering;

    // Most video is already 4:2:0 planar, in which case nothing is converted
    // at all. Anything else — 4:2:2, 4:4:4, hardware NV12 — is brought to
    // YUV420P, which is still far cheaper than going all the way to RGBA.
    // Built on the first frame: with hardware decoding the decoder reports a
    // GPU format, and what actually arrives is whatever the download produces.
    let mut scaler: Option<ffmpeg_next::software::scaling::Context> = None;
    let mut scaler_for: Option<ffmpeg_next::format::Pixel> = None;

    let colour = Colour {
        hd: decoder.height() >= 720,
        full_range: matches!(
            decoder.format(),
            ffmpeg_next::format::Pixel::YUVJ420P
                | ffmpeg_next::format::Pixel::YUVJ422P
                | ffmpeg_next::format::Pixel::YUVJ444P
        ),
    };

    let mut decoded = ffmpeg_next::util::frame::Video::empty();
    let mut converted = ffmpeg_next::util::frame::Video::empty();
    let mut downloaded = ffmpeg_next::util::frame::Video::empty();
    let mut seen = generation.load(Ordering::Acquire);
    let mut skip_until: Option<f64> = None;

    while let Ok(message) = packets.recv() {
        if stop.upgrade().is_none() {
            return Ok(());
        }
        let (tag, packet) = match message {
            Msg::Packet(tag, packet) => (tag, packet),
            Msg::End(tag) => {
                // Out of file: hand over whatever the codec is still holding,
                // then say so. The thread stays alive for a seek back in.
                if tag == generation.load(Ordering::Acquire) {
                    flush_decoder(
                        &mut decoder,
                        &mut decoded,
                        &mut downloaded,
                        &mut converted,
                        scaler.as_mut(),
                        time_base,
                        colour,
                        tag,
                        &sender,
                    );
                    ended.store(true, Ordering::Release);
                }
                continue;
            }
        };
        // Packets from before the last seek decode to pictures nobody wants.
        if tag != generation.load(Ordering::Acquire) {
            continue;
        }
        if tag != seen {
            // First packet of a new position: the codec still holds state for
            // the old one, and decoding across the join produces garbage.
            decoder.flush();
            seen = tag;
            ended.store(false, Ordering::Release);
            // Short of an exact target by more than the player's own 20 ms
            // allowance, a frame would only be decoded past. Most of what it
            // costs is downloading it from the GPU and copying it, so it is
            // dropped before either: that is most of the wait after a seek.
            let from = f64::from_bits(exact_from.load(Ordering::Acquire));
            skip_until = (from > 0.0).then_some(from - 0.02);
        }

        if decoder.send_packet(&packet).is_err() {
            continue;
        }
        while decoder.receive_frame(&mut decoded).is_ok() {
            let pts = decoded.pts().unwrap_or(0);
            if let Some(until) = skip_until {
                if (pts as f64 * time_base) < until {
                    continue;
                }
                skip_until = None;
            }
            // Read before any conversion: a scaled copy is a new frame and
            // does not carry this.
            let key = decoded.is_key();
            // A hardware frame lives in GPU memory and has to be fetched
            // before anything can read it.
            let decoded = match download(&mut decoded, &mut downloaded) {
                Ok(frame) => frame,
                Err(e) => {
                    eprintln!("show: {e:#}");
                    continue;
                }
            };
            let format = decoded.format();
            if format != ffmpeg_next::format::Pixel::YUV420P && scaler_for != Some(format) {
                scaler = Some(
                    ffmpeg_next::software::scaling::Context::get(
                        format,
                        decoded.width(),
                        decoded.height(),
                        ffmpeg_next::format::Pixel::YUV420P,
                        decoded.width(),
                        decoded.height(),
                        ffmpeg_next::software::scaling::Flags::BILINEAR,
                    )
                    .map_err(|e| anyhow::anyhow!("scaler: {e}"))?,
                );
                scaler_for = Some(format);
            }

            let source = match (format, scaler.as_mut()) {
                (ffmpeg_next::format::Pixel::YUV420P, _) => decoded,
                (_, Some(scaler)) => {
                    scaler
                        .run(decoded, &mut converted)
                        .map_err(|e| anyhow::anyhow!("scaling: {e}"))?;
                    &converted
                }
                (_, None) => decoded,
            };

            let at = Duration::from_secs_f64(pts.max(0) as f64 * time_base);
            let mut pending = pack(source, at, colour, tag, key);

            // Blocks once the queue is full, which is the backpressure that
            // keeps a decoder from racing ahead of the screen. Only this
            // thread waits here now, so the sound keeps flowing regardless.
            loop {
                match sender.try_send(pending) {
                    Ok(()) => {
                        queued.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                    Err(TrySendError::Full(returned)) => {
                        if stop.upgrade().is_none() {
                            return Ok(());
                        }
                        if returned.generation != generation.load(Ordering::Acquire) {
                            break;
                        }
                        pending = returned;
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(TrySendError::Disconnected(_)) => return Ok(()),
                }
            }
        }
    }

    Ok(())
}


/// Hand over the frames a decoder is still holding at the end of the file.
#[allow(clippy::too_many_arguments)]
fn flush_decoder(
    decoder: &mut ffmpeg_next::decoder::Video,
    decoded: &mut ffmpeg_next::util::frame::Video,
    downloaded: &mut ffmpeg_next::util::frame::Video,
    converted: &mut ffmpeg_next::util::frame::Video,
    mut scaler: Option<&mut ffmpeg_next::software::scaling::Context>,
    time_base: f64,
    colour: Colour,
    tag: u64,
    sender: &SyncSender<Frame>,
) {
    let _ = decoder.send_eof();
    while decoder.receive_frame(decoded).is_ok() {
        let pts = decoded.pts().unwrap_or(0);
        let key = decoded.is_key();
        let Ok(frame) = download(decoded, downloaded) else { break };
        let source = match (frame.format(), scaler.as_deref_mut()) {
            (ffmpeg_next::format::Pixel::YUV420P, _) => frame,
            (_, Some(scaler)) => {
                if scaler.run(frame, converted).is_err() {
                    break;
                }
                &*converted
            }
            (_, None) => frame,
        };
        let at = Duration::from_secs_f64(pts.max(0) as f64 * time_base);
        if sender.send(pack(source, at, colour, tag, key)).is_err() {
            break;
        }
    }
    // Ready for a seek back into the file.
    decoder.flush();
}

/// Decode audio packets and feed the device.
fn audio_loop(
    packets: Receiver<Msg>,
    mut setup: AudioSetup,
    generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
    exact_from: std::sync::Arc<std::sync::atomic::AtomicU64>,
    stop: std::sync::Weak<()>,
) {
    use std::sync::atomic::Ordering;

    let mut decoded = ffmpeg_next::util::frame::Audio::empty();
    let mut resampled = ffmpeg_next::util::frame::Audio::empty();
    let mut seen = generation.load(Ordering::Acquire);

    while let Ok(message) = packets.recv() {
        if stop.upgrade().is_none() {
            return;
        }
        let (tag, packet) = match message {
            Msg::Packet(tag, packet) => (tag, packet),
            // Nothing to do: what is already queued plays out, and a seek
            // will flush it if one comes.
            Msg::End(_) => continue,
        };
        if tag != generation.load(Ordering::Acquire) {
            continue;
        }
        if tag != seen {
            // A seek: drop the codec's state and everything queued for the
            // old position, here rather than in the demuxer, so the discard
            // and the refill cannot cross.
            setup.decoder.flush();
            setup.sink.flush();
            // The resampler holds a little of the old position too; a fresh
            // one starts clean.
            if let Ok(fresh) = resampler_for(&setup.decoder) {
                setup.resampler = fresh;
            }
            let from = f64::from_bits(exact_from.load(Ordering::Acquire));
            setup.skip_until = (from > 0.0).then_some(from);
            seen = tag;
        }
        let stale = || generation.load(Ordering::Acquire) != tag;
        if !feed_audio(&mut setup, &packet, &mut decoded, &mut resampled, &stale) {
            return;
        }
    }
}

/// Copy the three planes out of FFmpeg's buffers into tightly packed blocks.
///
/// FFmpeg pads rows to its own alignment, so this cannot be one memcpy per
/// plane. It is still only 1.5 bytes per pixel, against 4 for the RGBA this
/// replaced.
fn pack(
    frame: &ffmpeg_next::util::frame::Video,
    at: Duration,
    colour: Colour,
    generation: u64,
    key: bool,
) -> Frame {
    let (width, height) = (frame.width(), frame.height());
    // 4:2:0, so chroma is half in each direction, rounded up for odd sizes.
    let chroma = (width.div_ceil(2), height.div_ceil(2));

    let plane = |index: usize, w: u32, h: u32| -> Vec<u8> {
        let stride = frame.stride(index);
        let data = frame.data(index);
        let row = w as usize;
        let mut out = vec![0u8; row * h as usize];
        for y in 0..h as usize {
            let src = y * stride;
            if src + row <= data.len() {
                out[y * row..(y + 1) * row].copy_from_slice(&data[src..src + row]);
            }
        }
        out
    };

    Frame {
        key,
        generation,
        planes: [
            std::sync::Arc::new(plane(0, width, height)),
            std::sync::Arc::new(plane(1, chroma.0, chroma.1)),
            std::sync::Arc::new(plane(2, chroma.0, chroma.1)),
        ],
        width,
        height,
        chroma,
        colour,
        at,
    }
}

/// Bring a hardware frame into system memory, or pass a software one through.
///
/// Decoding on the GPU leaves the frame there. Reading it back is the cost of
/// not having a zero-copy path to the display yet — still far cheaper than
/// decoding on the CPU, and the obvious thing to remove later.
fn download<'a>(
    decoded: &'a mut ffmpeg_next::util::frame::Video,
    scratch: &'a mut ffmpeg_next::util::frame::Video,
) -> Result<&'a ffmpeg_next::util::frame::Video> {
    if decoded.format() != ffmpeg_next::format::Pixel::VAAPI {
        return Ok(decoded);
    }
    // Safety: both frames are valid; FFmpeg allocates into `scratch`.
    let rc = unsafe {
        ffmpeg_next::sys::av_hwframe_transfer_data(scratch.as_mut_ptr(), decoded.as_ptr(), 0)
    };
    if rc < 0 {
        anyhow::bail!("could not read a frame back from the GPU (error {rc})");
    }
    Ok(scratch)
}

/// Convert a frame to RGBA on the CPU.
///
/// Only for the software presentation path, which has no shader to do it in.
/// The GPU path never calls this — avoiding it is the point of keeping frames
/// planar.
pub fn to_rgba(frame: &Frame) -> Vec<u8> {
    let (rows, offsets) = frame.colour.matrix();
    let (w, h) = (frame.width as usize, frame.height as usize);
    let cw = frame.chroma.0 as usize;
    let mut out = vec![0u8; w * h * 4];

    for y in 0..h {
        for x in 0..w {
            let luma = frame.planes[0].get(y * w + x).copied().unwrap_or(0);
            // Nearest chroma sample: the fallback path is not where quality
            // is won.
            let ci = (y / 2) * cw + (x / 2);
            let cb = frame.planes[1].get(ci).copied().unwrap_or(128);
            let cr = frame.planes[2].get(ci).copied().unwrap_or(128);

            let yv = luma as f32 / 255.0 - offsets[0];
            let uv = cb as f32 / 255.0 - offsets[1];
            let vv = cr as f32 / 255.0 - offsets[1];

            let d = (y * w + x) * 4;
            for (c, row) in rows.iter().enumerate() {
                let value = row[0] * yv + row[1] * uv + row[2] * vv;
                out[d + c] = (value.clamp(0.0, 1.0) * 255.0) as u8;
            }
            out[d + 3] = 255;
        }
    }
    out
}

#[cfg(test)]
mod queue_tests {
    use super::{MIN_QUEUE_DEPTH, queue_depth};

    #[test]
    fn the_read_ahead_window_is_held_for_a_normal_film() {
        // 1080p25: three seconds is 75 frames, well inside the budget.
        assert_eq!(queue_depth(1920, 1080, 25.0), 75);
    }

    #[test]
    fn a_large_format_is_bound_by_memory_rather_than_time() {
        // 4K at 60 fps would be over a gigabyte for three seconds, so it
        // gets fewer seconds instead of more memory.
        let depth = queue_depth(3840, 2160, 60.0);
        assert!(depth < 180, "4K60 should not hold the full window: {depth}");
        let bytes = depth as u64 * (3840 * 2160 * 3 / 2);
        assert!(bytes <= 1 << 30, "over the budget: {bytes}");
    }

    #[test]
    fn even_an_enormous_frame_gets_a_usable_queue() {
        // A frame so large the budget allows none: the decoder still needs
        // room to work ahead or every complex frame is a stall.
        assert_eq!(queue_depth(20000, 20000, 25.0), MIN_QUEUE_DEPTH);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of reference-counted planes: handing a frame to a second
    /// consumer must not copy three megabytes.
    #[test]
    fn cloning_a_frame_shares_its_planes_rather_than_copying_them() {
        let frame = Frame {
            key: true,
            generation: 0,
            planes: [
                std::sync::Arc::new(vec![7u8; 1024]),
                std::sync::Arc::new(vec![8u8; 256]),
                std::sync::Arc::new(vec![9u8; 256]),
            ],
            width: 32,
            height: 32,
            chroma: (16, 16),
            at: Duration::ZERO,
            colour: Colour { hd: true, full_range: false },
        };
        let copy = frame.clone_ref();
        for i in 0..3 {
            assert!(
                std::sync::Arc::ptr_eq(&frame.planes[i], &copy.planes[i]),
                "plane {i} was copied instead of shared"
            );
        }
        // And the copy still reads as the same picture.
        assert_eq!(copy.planes[0][0], 7);
        assert_eq!(copy.at, frame.at);
    }

    #[test]
    fn video_extensions_are_recognised_case_insensitively() {
        assert!(is_video(Path::new("clip.mp4")));
        assert!(is_video(Path::new("CLIP.MKV")));
        assert!(is_video(Path::new("/a/b/movie.WebM")));
        assert!(!is_video(Path::new("photo.jpg")));
        assert!(!is_video(Path::new("noextension")));
    }

    #[test]
    fn the_colour_matrix_round_trips_a_known_colour() {
        // Studio-range BT.709 white: Y=235, chroma centred.
        let colour = Colour { hd: true, full_range: false };
        let (rows, offsets) = colour.matrix();
        let (y, u, v) = (235.0 / 255.0 - offsets[0], 0.0, 0.0);
        let rgb: Vec<f32> =
            rows.iter().map(|r| r[0] * y + r[1] * u + r[2] * v).collect();
        for c in &rgb {
            assert!((c - 1.0).abs() < 0.01, "white should map to 1.0, got {c}");
        }

        // And studio black.
        let (y, u, v) = (16.0 / 255.0 - offsets[0], 0.0, 0.0);
        let rgb: Vec<f32> = rows.iter().map(|r| r[0] * y + r[1] * u + r[2] * v).collect();
        for c in &rgb {
            assert!(c.abs() < 0.01, "black should map to 0.0, got {c}");
        }
    }

    #[test]
    fn full_range_needs_no_rescaling() {
        let (rows, offsets) = Colour { hd: true, full_range: true }.matrix();
        assert_eq!(offsets[0], 0.0, "full range starts at zero");
        assert!((rows[0][0] - 1.0).abs() < 1e-6, "and is not stretched");
    }

    #[test]
    fn opening_something_that_is_not_video_fails_rather_than_panics() {
        // Both paths must fail cleanly, not just the software one.
        for hardware in [false, true] {
            assert!(Player::open(Path::new("/nonexistent-video-for-tests.mp4"), hardware).is_err());
        }
    }
}
