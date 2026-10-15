//! Window, event loop, and the interaction between decoding and presenting.
//!
//! Three things run in parallel here, because measurement said the naive order
//! spends most of its time waiting:
//!
//! * The **load** starts before the event loop exists, so decoding overlaps
//!   window creation.
//! * The **first decode** asks for roughly a screen's worth of pixels rather
//!   than every pixel. For JPEG that is a scaled inverse DCT; full resolution
//!   follows on a second thread and swaps in underneath the view.
//! * The **first frame goes out through the CPU**, while Vulkan is still
//!   starting up on a third thread. Bringing up a GPU costs ~45 ms and
//!   rasterising one frame costs a few, so waiting for the GPU to show anything
//!   would be the single most expensive thing the program does.
//!
//! The view always works in *source* image coordinates, never in whatever
//! resolution is currently loaded or on the GPU. That is what makes both
//! handovers invisible: the image occupies the same rectangle throughout, and
//! only the sharpness behind it changes.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::platform::startup_notify::WindowAttributesExtStartupNotify;
#[cfg(all(unix, not(target_os = "macos")))]
use winit::platform::wayland::WindowAttributesExtWayland;
#[cfg(all(unix, not(target_os = "macos")))]
use winit::platform::x11::WindowAttributesExtX11;
use winit::window::ActivationToken;
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorIcon, Fullscreen, Window, WindowId};

use crate::browser::{Activate, Browser};
use crate::cli::Args;
use crate::config::{self, Config};
use crate::decode::Target;
use crate::gpu::{Core, Gpu};
use crate::loader::{self, Io, Loaded, human_bytes};
use crate::osd::{self, Overlay};
use crate::mosaic::{Mosaic, Thumb};
use crate::keys::Keys;
use crate::panel::Panel;
use crate::playlist::Playlist;
use crate::software::Software;
use crate::timing;
use crate::view::View;

/// The subtitle picker: what is loaded, and what could be.
struct SubtitleMenu {
    entries: Vec<MenuEntry>,
    selected: usize,
    overlay: Option<Overlay>,
}

enum MenuEntry {
    Off,
    /// A track already loaded; the index is into `Subtitles::tracks`.
    Loaded(usize),
    /// A file in the folder that has not been read yet.
    File(std::path::PathBuf),
}

/// Drawing a tiled image from its tiles, past the zoom its overview covers.
///
/// The overview stays on screen underneath, so a tile not read yet shows as
/// that — blurry, never a hole — and a coarser tile already on the card is
/// drawn over it where it can do better.
struct Tiling {
    pyramid: Arc<crate::tiled::Pyramid>,
    loader: crate::tiles::TileLoader,
    /// The zoom up to which the overview is sharp. Tiles are only wanted past
    /// it; below it they would be finer than the screen can show.
    overview_zoom: f32,
    /// What was last asked of the loader, so an unchanged view asks nothing.
    requested: Vec<crate::tiles::TileKey>,
    /// What the last frame drew from and what the screen still needs, spared
    /// by eviction: a tile that has just arrived must not be pushed out by the
    /// next one before it has been drawn once.
    in_use: std::collections::HashSet<crate::tiles::TileKey>,
    /// The level being drawn, and how many of its tiles on screen are in, for
    /// the debug readout. `None` while the overview is enough.
    drawing: Option<(usize, usize, usize)>,
}

/// A backstop, not a working limit: a large file without a pyramid of its own
/// is given stored levels the first time it is opened (`src/pyramid.rs`), so a
/// screen needs a few dozen tiles at any zoom. Only a view needing more than
/// this — a file whose levels could not be stored — falls back to the
/// overview.
const MAX_TILES_ON_SCREEN: usize = 6000;

/// Where a press lands on the window switcher.
enum PeerHit {
    /// Not on it; nothing to do with it.
    Nothing,
    /// The button: drop the list, or put it away.
    Button,
    /// A row of the list: that window.
    Window(String),
    /// Elsewhere while the list is down: put it away.
    Away,
}

/// A grid set aside while looking at one of its images.
struct GridBehind {
    mosaic: Mosaic,
    /// The list the grid was laid out from. Choosing an image in the detailed
    /// view replaces the live one with that folder's, so the grid keeps its own.
    playlist: Playlist,
    /// False once a file has been deleted from the list, which shifts every
    /// cell after it and leaves the cached thumbnails on the wrong ones.
    thumbs_valid: bool,
}

/// How far the pointer has to move to count as reaching for the mouse, and
/// how long the title bar stays after it last did.
const TITLE_BAR_NUDGE: f32 = 15.0;
const TITLE_BAR_LINGER: std::time::Duration = std::time::Duration::from_secs(2);

/// What pressing on the window frame does.
enum ChromeHit {
    Close,
    Move,
    Resize(winit::window::ResizeDirection),
}

/// The window's own frame: a title bar and a border, in place of the
/// desktop's decorations.
///
/// Only when windowed. Fullscreen has no frame to draw and no room to spare.
struct Chrome {
    bar_height: f32,
    border: f32,
}

impl Chrome {
    /// Thin, and scaled to the screen so it is the same apparent size on a
    /// laptop panel and on a large monitor.
    fn for_window(win: (f32, f32)) -> Chrome {
        let scale = (win.1 / 640.0).clamp(1.0, 3.0);
        Chrome { bar_height: (22.0 * scale).round(), border: (2.0 * scale).round() }
    }

    fn text_scale(&self) -> f32 {
        (self.bar_height / 11.0).clamp(1.5, 4.0)
    }
}

/// Type size for a caption over a picture `height` pixels tall.
fn caption_px(height: f32) -> f32 {
    (height * 0.045).clamp(14.0, 96.0)
}

/// Where a caption's top edge should sit.
///
/// In the black band below the picture when there is one deep enough to hold
/// it — a letterboxed film has that space going spare, and text there covers
/// nothing. Otherwise inside the picture, low but clear of the very edge.
///
/// The band has to hold the caption *and* keep it off both edges, or the
/// choice trades covering the film for touching the frame.
fn caption_y(picture_bottom: f32, win_h: f32, caption_h: f32) -> i32 {
    let band = win_h - picture_bottom;
    if band >= caption_h + Corner::MARGIN * 2.0 {
        // Centred in the band, which is where the eye expects it.
        (picture_bottom + (band - caption_h) / 2.0) as i32
    } else {
        (win_h - caption_h - win_h * 0.08 - Corner::MARGIN).max(0.0) as i32
    }
}

/// [`caption_y`] for a caption drawn at `scale` times its usual size: placed
/// as the usual one would be, then grown about its middle, so changing the
/// size leaves the words where the eye already is.
fn caption_y_scaled(picture_bottom: f32, win_h: f32, caption_h: f32, scale: f32) -> i32 {
    let usual = caption_h / scale.max(0.01);
    let middle = caption_y(picture_bottom, win_h, usual) as f32 + usual / 2.0;
    (middle - caption_h / 2.0).clamp(0.0, (win_h - caption_h).max(0.0)) as i32
}

/// The first line of a message, for somewhere with room for one.
fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or(text)
}

/// Whether a press and release were a click rather than the ends of a drag.
///
/// The distance is what separates them: a pan across a film and a click to
/// pause it are the same two events, and only the movement between tells them
/// apart. A release with no press belongs to a gesture that started elsewhere
/// — dragging in from outside the window — and is not a click.
fn was_click(press: Option<(f32, f32)>, release: (f32, f32)) -> bool {
    /// Generous enough for a hand that is not quite still, tight enough that
    /// an intentional drag is never mistaken for a click.
    const SLOP: f32 = 6.0;
    press.is_some_and(|at| {
        (at.0 - release.0).abs() < SLOP && (at.1 - release.1).abs() < SLOP
    })
}

/// The window a video of `video` should open in on a `screen`.
///
/// The film's own resolution, so one video pixel is one screen pixel and
/// nothing is resampled to show it — which is both the sharpest it can look
/// and the cheapest it can be drawn. Only shrunk, never enlarged, and only
/// when it would otherwise not fit: a 4K film on a 1080p screen has to come
/// down, and it comes down keeping its shape.
fn window_size_for(video: (u32, u32), screen: (u32, u32)) -> (u32, u32) {
    let (vw, vh) = (video.0.max(1) as f32, video.1.max(1) as f32);
    // Room for the desktop's own furniture — a panel, a title bar — so a film
    // exactly as tall as the screen is not born half off it.
    let (room_w, room_h) = (screen.0 as f32 * 0.9, screen.1 as f32 * 0.9);
    let fit = (room_w / vw).min(room_h / vh).min(1.0);
    (
        (vw * fit).round().clamp(320.0, screen.0 as f32) as u32,
        (vh * fit).round().clamp(180.0, screen.1 as f32) as u32,
    )
}

/// Where an overlay sits in the window.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomCentre,
    /// Bottom right, clear of the seek bar. The frame readout.
    BottomRight,
    /// The window's own frame, positioned exactly rather than by corner.
    TitleBar,
    BorderLeft,
    BorderRight,
    BorderBottom,
    /// Centred horizontally, sitting a little above the bottom edge. Where
    /// subtitles have always gone, and clear of the seek bar.
    Caption,
    /// An explicit position in window pixels.
    ///
    /// For things placed against the picture rather than against the window —
    /// a label beside what it labels. Nothing stacks against it, because two
    /// things at chosen positions were each chosen for a reason.
    At(i32, i32),
    Centre,
    /// Covers the window from the origin. The grid uses this; nothing stacks
    /// against it, and everything else draws on top.
    Fill,
    /// Flush to the right edge, full height. The browser panel.
    RightEdge,
}

impl Corner {
    /// Inset from the window edge, in pixels.
    const MARGIN: f32 = 18.0;
    /// Vertical space between overlays sharing a corner.
    const STACK_GAP: i32 = 10;

    fn origin(self, ov: &Overlay, win: (f32, f32)) -> (i32, i32) {
        let (w, h) = (ov.width as f32, ov.height as f32);
        match self {
            Corner::TopLeft => (Self::MARGIN as i32, Self::MARGIN as i32),
            Corner::TopRight => ((win.0 - w - Self::MARGIN) as i32, Self::MARGIN as i32),
            Corner::BottomLeft => (Self::MARGIN as i32, (win.1 - h - Self::MARGIN) as i32),
            Corner::BottomCentre => {
                (((win.0 - w) / 2.0) as i32, (win.1 - h - Self::MARGIN) as i32)
            }
            // Lifted clear of the seek bar, which shares the bottom edge and
            // appears whenever the pointer moves.
            // Placed against the window edges; the sizes already match.
            Corner::TitleBar => (0, 0),
            Corner::BorderLeft => (0, (win.1 - h as f32) as i32),
            Corner::BorderRight => ((win.0 - w as f32) as i32, (win.1 - h as f32) as i32),
            Corner::BorderBottom => (0, (win.1 - h as f32) as i32),
            Corner::BottomRight => (
                (win.0 - w - Self::MARGIN) as i32,
                (win.1 - h - Self::MARGIN - win.1 * 0.055) as i32,
            ),
            // Horizontally centred; the vertical placement depends on where
            // the picture ends and is settled by `caption_y` at draw time.
            Corner::At(x, y) => (x, y),
            Corner::Caption => (
                ((win.0 - w) / 2.0) as i32,
                (win.1 - h - win.1 * 0.08 - Self::MARGIN) as i32,
            ),
            Corner::Centre => (((win.0 - w) / 2.0) as i32, ((win.1 - h) / 2.0) as i32),
            Corner::Fill => (0, 0),
            Corner::RightEdge => ((win.0 - w) as i32, 0),
        }
    }
}

/// Convert a pixel rectangle to the clip-space rectangle the shader wants.
fn ndc_rect_from_pixels(x0: i32, y0: i32, w: u32, h: u32, win: (f32, f32)) -> [f32; 4] {
    let (x0, y0) = (x0 as f32, y0 as f32);
    let (x1, y1) = (x0 + w as f32, y0 + h as f32);
    [
        2.0 * x0 / win.0 - 1.0,
        1.0 - 2.0 * y1 / win.1,
        2.0 * x1 / win.0 - 1.0,
        1.0 - 2.0 * y0 / win.1,
    ]
}

/// Wheel notches scale zoom by this each step: small enough to feel continuous,
/// large enough that a flick crosses a useful range.
/// How the found things are drawn: a solid edge this many label cells thick,
/// and the inside left mostly alone. Filling every found thing hides more of
/// the film than it explains, so the fill is a hint rather than a coat.
const OUTLINE: f32 = 1.0;
const FILL: f32 = 0.16;
/// How much of the picture shows through the depth colours, at first.
const DEPTH_BLEND: f32 = 0.6;

/// Arrow-key seek distances for the modified arrows. Unmodified ones step
/// keyframe to keyframe, whose distance the file decides.
const SEEK_MEDIUM: std::time::Duration = std::time::Duration::from_secs(10);
const SEEK_LONG: std::time::Duration = std::time::Duration::from_secs(30);

const ZOOM_STEP: f32 = 1.15;
const KEY_ZOOM_STEP: f32 = 1.25;
const KEY_PAN_PIXELS: f32 = 80.0;

/// Above this much decoded RGBA, full resolution is not fetched speculatively.
/// A 400 megapixel panorama is 1.5 GB and several seconds of work; spending
/// that on someone who only glanced at the image is rude, so those wait until a
/// zoom actually asks for the detail.
const SPECULATIVE_FULL_RES_BUDGET: u64 = 512 << 20;

/// Where the browser's text size starts. Ctrl+wheel moves it by
/// `osd::SCALE_STEP` between `osd::SCALE_MIN` and `osd::SCALE_MAX`.
const DEFAULT_BROWSER_SCALE: f32 = 2.0;

/// How much history the frame-rate chart shows, and how finely.
const FPS_WINDOW: std::time::Duration = std::time::Duration::from_secs(5);
const FPS_BUCKET: std::time::Duration = std::time::Duration::from_millis(250);
/// How often the chart is redrawn. Often enough to feel live, rarely enough
/// that rasterising it is not part of the frame budget.
const FPS_REFRESH: std::time::Duration = std::time::Duration::from_millis(250);

/// Pixels scrolled per wheel notch in the grid.
const WHEEL_SCROLL_PIXELS: f32 = 140.0;

/// Cells fetched per pass beyond the visible ones. Small enough that the
/// decode pool stays available for whatever the user scrolls to next.
const PREFETCH_BATCH: usize = 8;

/// How long a new window keeps asking the window manager for the keyboard.
///
/// Measured in time rather than frames, because what is being waited for is a
/// round trip to the X server and not a number of pictures: a release build
/// draws its first ten frames in a few milliseconds, spends every attempt
/// before the server has answered, and gives up before it could have worked —
/// which is exactly how the frame-counted version passed under a debug build
/// and failed under a real one.
const FOCUS_GRACE: std::time::Duration = std::time::Duration::from_millis(1500);

/// Identifies which image a load belongs to. Holding an arrow key down starts
/// loads faster than they finish, so results have to be matched against what
/// the user is actually looking at now, not whatever was requested last.
type Generation = u64;

enum UserEvent {
    /// The fast, possibly reduced-resolution first paint.
    Preview(Generation, Box<Result<Loaded>>),
    /// Full resolution, loaded behind the preview.
    Full(Generation, Box<Result<Loaded>>),
    /// The image after this one, read ahead while nothing else is happening.
    Neighbour(PathBuf, Box<Result<Loaded>>),
    /// Instance, adapter and device, built off the main thread.
    GpuReady(Box<Result<Core>>),
    /// One grid cell, decoded at cell resolution.
    Thumb(Generation, usize, Box<Result<Loaded>>),
    /// Similarity analysis is under way.
    AnalysisProgress(Generation, usize, usize),
    /// Similarity analysis finished, with one hash per playlist entry.
    AnalysisDone(Generation, Box<Vec<Option<u64>>>),
    /// A plugin is downloading or unpacking.
    PluginProgress(crate::ml::catalogue::Capability, crate::plugins::Status),
    /// A plugin install finished, well or badly.
    PluginDone(crate::ml::catalogue::Capability, Box<Result<()>>),
    /// The folder of the file the viewer was opened on, listed and sorted.
    Folder(Box<Playlist>),
    /// Images arrived in, or left, the folder being shown.
    FolderChanged,
    /// Picked from the tray menu, or from another window's strip: come to
    /// the front, and go fullscreen too when asked.
    Raise(bool),
    /// The open windows, or what one of them shows, changed.
    Peers,
    /// The full-resolution picture, on the card and ready to draw.
    Uploaded(Generation, Box<crate::gpu::Uploaded>),
    /// One tile of a tiled image, read.
    Tile(Generation, crate::tiles::TileKey, Box<Result<crate::tiled::Tile>>),
    /// Where on the map an image belongs, if anywhere: its path, and every
    /// placement found for it.
    Geo(PathBuf, Vec<crate::geo::Source>),
    /// Sizes and tile levels for rows of the detailed view, a batch at a
    /// time; `true` on the last batch of a folder.
    FileInfo(Vec<(PathBuf, crate::browser::FileInfo)>, bool),
    /// A background thumbnail was written. The generation is carried for
    /// symmetry with the other events, though nothing needs it yet.
    ThumbnailWritten(#[allow(dead_code)] Generation),
}

#[derive(Clone, Copy, PartialEq)]
enum FullRes {
    /// What we have is already the whole image.
    NotNeeded,
    /// Worth loading, but not started: too big to fetch speculatively.
    Deferred,
    Loading,
    Done,
}

/// What is currently being shown, independent of who is drawing it.
struct Displayed {
    /// True image dimensions. All view maths is in these coordinates.
    source: (f32, f32),
    /// Dimensions of the pixels we actually have.
    tex: (u32, u32),
    /// Held only while the CPU is presenting; freed once the GPU has a copy.
    pixels: Option<Vec<u8>>,
    /// Highest zoom these pixels can serve at one texel per screen pixel.
    adequate_zoom: f32,
    /// Which pyramid level these pixels came from, for a tiled image.
    tiles: Option<crate::decode::TileLevels>,
}

/// The desktop's launch token, captured before any thread exists.
///
/// A launcher with `StartupNotify=true` puts an id here and spins the busy
/// cursor until something ends the sequence. winit's X11 backend sends exactly
/// that message, but only for a window created with the token, so it has to be
/// carried from the environment to `create_window`.
///
/// Read and cleared in `main`: the variable must not survive into `xdg-mime`
/// and friends, which would otherwise claim our launch as theirs — and removing
/// an environment variable is only sound while single-threaded.
static ACTIVATION_TOKEN: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// Must be called from `main` before starting any thread.
pub fn capture_activation_token() {
    let token = std::env::var("DESKTOP_STARTUP_ID")
        .ok()
        .or_else(|| std::env::var("XDG_ACTIVATION_TOKEN").ok());
    let _ = ACTIVATION_TOKEN.set(token);

    // Safety: called from `main` before any thread is spawned, which is the
    // condition `remove_var` requires.
    unsafe {
        std::env::remove_var("DESKTOP_STARTUP_ID");
        std::env::remove_var("XDG_ACTIVATION_TOKEN");
    }
}

pub fn run(args: Args) -> Result<()> {
    // The decode is the long pole of a cold start, so a file named on the
    // command line starts decoding before anything else: before the display
    // connection (tens of milliseconds cold) and before its folder is listed.
    // The thread only needs the event loop to hand its result over, and waits
    // for it at the end, by which time it has long existed. Video is the
    // exception: it needs a window to pace itself against.
    // The first parallel job pays for starting the pool's threads — thirty-two
    // of them here — and that is usually the first frame's rasterise. Started
    // now, on the side, it is ready by then.
    rayon::spawn(|| {});

    let proxy_slot = ProxySlot::default();
    let early = args.files.first().filter(|p| args.files.len() == 1 && !crate::video::is_video(p));
    if let Some(path) = early {
        spawn_early_load(proxy_slot.clone(), path.clone(), args.io, args.preview);
    }

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    proxy_slot.fill(proxy.clone());

    // Several files named on the command line are the sequence; one file means
    // the folder it lives in is; nothing at all means the Pictures folder, so
    // launching from the applications menu lands somewhere useful.
    //
    // One file is shown on its own at first. Listing its folder is deferred
    // until the picture is up — see `start_folder_scan` — since showing it is
    // the whole point of starting, and a large folder on a cold disk is not
    // free to list.
    let playlist = match args.files.len() {
        0 => Playlist::from_dir(&Playlist::default_start_dir()),
        _ => Playlist::from_list(&args.files),
    };

    // Otherwise start decoding now, so that by the time anything can draw the
    // pixels are ready.
    if early.is_none()
        && let Some(path) = playlist.current()
        && !crate::video::is_video(path)
    {
        spawn_load(&proxy, path.to_path_buf(), args.io, args.preview, false, 0);
    }

    let mut app = App::new(args, proxy, playlist);
    event_loop.run_app(&mut app)?;

    // Saving the thumbnail index waits for the cache, but not for long: a
    // worker already mid-write is let finish, and if the cache is still busy
    // after that the save is skipped. The index is bookkeeping — what it
    // misses, the next run notices and redoes — and not worth a window that
    // will not close.
    let quitting = std::time::Instant::now();
    let mut waited = std::time::Duration::ZERO;
    if let Some(cache) = &app.thumbs {
        const PATIENCE: std::time::Duration = std::time::Duration::from_millis(500);
        let held = loop {
            match cache.try_lock() {
                Ok(guard) => break Some(guard),
                Err(std::sync::TryLockError::Poisoned(_)) => break None,
                Err(std::sync::TryLockError::WouldBlock) if quitting.elapsed() < PATIENCE => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    eprintln!("show: thumbnail cache still busy; its index is saved next time");
                    break None;
                }
            }
        };
        waited = quitting.elapsed();
        if let Some(mut cache) = held {
            // Bounded here rather than as it grows: trimming is a directory
            // scan, and once per session is often enough for a cache that
            // gains a few megabytes an hour at most.
            if let Err(e) = cache.trim(crate::thumbs::DISK_BUDGET) {
                eprintln!("show: could not trim the thumbnail cache: {e:#}");
            }
            if let Err(e) = cache.save_index() {
                eprintln!("show: could not save the thumbnail index: {e:#}");
            }
        }
    }
    // Said only when it was slow, so a slow quit can be told apart.
    if quitting.elapsed() > std::time::Duration::from_millis(200) {
        eprintln!(
            "show: quitting took {:.0} ms ({:.0} ms waiting for the thumbnail cache)",
            quitting.elapsed().as_secs_f64() * 1e3,
            waited.as_secs_f64() * 1e3
        );
    }

    // Not dropped: tearing down the GPU device, the textures and a session's
    // worth of thumbnails cost 20-50 ms after the window had already gone, and
    // everything worth keeping is saved by now. The process ends straight
    // after this — see `leave_now` — and the system reclaims the rest.
    let result = match app.exit_error.take() {
        Some(e) => Err(e),
        None => Ok(()),
    };
    std::mem::forget(app);
    result
}

fn spawn_load(
    proxy: &EventLoopProxy<UserEvent>,
    path: PathBuf,
    io: Io,
    target: Option<Target>,
    full: bool,
    generation: Generation,
) {
    let proxy = proxy.clone();
    std::thread::Builder::new()
        .name(if full { "decode-full".into() } else { "decode".to_string() })
        .spawn(move || {
            // A HEIC's own thumbnail goes up first, in a few milliseconds,
            // while the real decode takes a few hundred.
            if !full {
                let (proxy, path) = (proxy.clone(), path.clone());
                std::thread::spawn(move || {
                    if let Some(thumb) = loader::load_thumbnail(&path, target) {
                        let _ = proxy.send_event(UserEvent::Preview(generation, Box::new(Ok(thumb))));
                    }
                });
            }
            let result = Box::new(loader::load(&path, io, target));
            let _ = proxy.send_event(if full {
                UserEvent::Full(generation, result)
            } else {
                UserEvent::Preview(generation, result)
            });
        })
        .expect("spawning the decode thread");
}

/// Let everything else go first: the calling thread, and every thread it goes
/// on to start — which on Linux inherit it — run at a lower priority.
///
/// For the full decode, which on a HEIC is thirty-two libheif threads flat
/// out: whatever the main thread is doing while they run — drawing, handling
/// input — should not have to queue behind them. Lowered, they still get
/// every core nothing else wants.
fn background_priority() {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn gettid() -> i32;
            fn setpriority(which: i32, who: u32, prio: i32) -> i32;
        }
        const PRIO_PROCESS: i32 = 0;
        // Safety: both only read or adjust this thread's own scheduling.
        unsafe { setpriority(PRIO_PROCESS, gettid() as u32, 10) };
    }
}

/// Set the moment the window is closing. Thumbnail work still queued checks it
/// and does nothing: a recursive grid queues thousands of decodes, and the
/// last ones to start should not be holding the cache while quitting waits.
static QUITTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Opened when the first frame is on screen.
///
/// A HEIC's full decode waits here behind its thumbnail: run alongside, its
/// thirty-two threads made the main thread's first draw anything from 2 ms to
/// 57 ms, and the thumbnail is only worth having if it is on screen at once.
static FIRST_FRAME: (std::sync::Mutex<bool>, std::sync::Condvar) =
    (std::sync::Mutex::new(false), std::sync::Condvar::new());

fn first_frame_shown() {
    *FIRST_FRAME.0.lock().unwrap() = true;
    FIRST_FRAME.1.notify_all();
}

/// Wait for the first frame, but never long: if it is slow for some other
/// reason, the real picture should not be slower still.
fn wait_for_first_frame() {
    const PATIENCE: std::time::Duration = std::time::Duration::from_millis(300);
    let guard = FIRST_FRAME.0.lock().unwrap();
    let _ = FIRST_FRAME.1.wait_timeout_while(guard, PATIENCE, |shown| !*shown);
}

/// Where the event loop's proxy is left for threads started before it exists.
///
/// A slot rather than a channel because more than one thread waits on it, and
/// each only when it has something to send — so a slow display connection
/// never holds up a decode, only its delivery.
#[derive(Clone, Default)]
struct ProxySlot(Arc<(std::sync::Mutex<Option<EventLoopProxy<UserEvent>>>, std::sync::Condvar)>);

impl ProxySlot {
    fn fill(&self, proxy: EventLoopProxy<UserEvent>) {
        let (slot, ready) = &*self.0;
        *slot.lock().unwrap() = Some(proxy);
        ready.notify_all();
    }

    /// Blocks until the event loop exists. If it never does, the process is
    /// already on its way out and this thread goes with it.
    fn wait(&self) -> EventLoopProxy<UserEvent> {
        let (slot, ready) = &*self.0;
        let mut guard = slot.lock().unwrap();
        loop {
            if let Some(proxy) = guard.as_ref() {
                return proxy.clone();
            }
            guard = ready.wait(guard).unwrap();
        }
    }
}

/// As [`spawn_load`] for the first image, started before the event loop
/// exists.
///
/// A HEIC's own thumbnail comes first, while the machine is otherwise quiet:
/// started alongside, the full decode's thirty-two threads slowed the
/// thumbnail, the window and the first draw alike. The full decode then waits
/// for the thumbnail to be on screen — see [`FIRST_FRAME`].
fn spawn_early_load(proxy: ProxySlot, path: PathBuf, io: Io, target: Option<Target>) {
    std::thread::Builder::new()
        .name("decode".into())
        .spawn(move || {
            if let Some(thumb) = loader::load_thumbnail(&path, target) {
                let proxy = proxy.clone();
                std::thread::spawn(move || {
                    let _ = proxy.wait().send_event(UserEvent::Preview(0, Box::new(Ok(thumb))));
                });
                wait_for_first_frame();
            }
            background_priority();
            let result = Box::new(loader::load(&path, io, target));
            let _ = proxy.wait().send_event(UserEvent::Preview(0, result));
        })
        .expect("spawning the decode thread");
}

/// Build the expensive, window-independent half of the GPU stack off-thread.
///
/// The surface and swapchain are left for the main thread: they are cheap, and
/// they must not exist while the software path still owns the window.
fn spawn_gpu_init(proxy: &EventLoopProxy<UserEvent>, adapter: Option<String>, backends: wgpu::Backends) {
    let proxy = proxy.clone();
    std::thread::Builder::new()
        .name("gpu-init".into())
        .spawn(move || {
            let core = Core::new_default(adapter.as_deref(), backends);
            let _ = proxy.send_event(UserEvent::GpuReady(Box::new(core)));
        })
        .expect("spawning the gpu init thread");
}

struct App {
    args: Args,
    proxy: EventLoopProxy<UserEvent>,
    playlist: Playlist,
    /// Bumped on every navigation; loads tagged with anything older are stale.
    generation: Generation,
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    software: Option<Software>,
    /// Set when a decode finishes before anything can draw it.
    pending: Option<(Loaded, bool)>,
    shown: Option<Displayed>,
    full_res: FullRes,
    /// The image the next step will land on, read ahead: its path, and the
    /// result once it is in. See [`App::prefetch_neighbour`].
    neighbour: Option<(PathBuf, Option<Loaded>)>,
    /// Which way the last step went, so the read-ahead follows it.
    step_dir: isize,
    view: View,
    cursor: (f32, f32),
    dragging: bool,
    /// When and where the last left press landed, for double-click detection.
    last_click: Option<(std::time::Instant, (f32, f32))>,
    shift_held: bool,
    ctrl_held: bool,
    /// `None` follows the zoom level; `Some` is an explicit user override.
    smooth_override: Option<bool>,
    fullscreen: bool,
    /// The pointer shape last asked for, so it is only set when it changes.
    cursor_icon: CursorIcon,
    /// Until when the window is still owed the keyboard; see `claim_focus`.
    focus_until: Option<std::time::Instant>,
    /// Whether the current image has had a frame on screen yet.
    presented: bool,
    /// The load breakdown, rasterised once per image.
    osd: Option<Overlay>,
    /// Zoom and memory, top left. Rebuilt when either changes.
    status_badge: Option<Overlay>,
    /// The current file's name, top right. Shown whatever the debug switch says
    /// — knowing which picture is on screen is not a diagnostic.
    filename_badge: Option<Overlay>,
    /// Whether the pointer is over the window. The filename and the seek bar
    /// are for whoever is looking at the window with the mouse; moved away,
    /// the picture is left on its own.
    pointer_inside: bool,
    /// When the pointer last moved a real distance, and from where — so a
    /// hand resting on the mouse does not keep the title bar up.
    last_motion: std::time::Instant,
    motion_from: (f32, f32),
    /// Whether the title bar was showing at the last look, so its going is
    /// noticed and drawn.
    title_bar_was: bool,
    /// Shown instead of an image when there is nothing to display.
    notice: Option<Overlay>,
    /// The phrase being typed, drawn. Present exactly while `typing` is.
    prompt_osd: Option<Overlay>,
    /// The volume readout, and when it stops being shown. Transient: a level
    /// you are not currently setting is clutter over a film.
    volume_osd: Option<(Overlay, std::time::Instant)>,
    /// The seek bar, and when it fades. Shown on mouse movement, the way a
    /// player's chrome appears when you reach for it.
    scrubber: Option<(Overlay, std::time::Instant)>,
    /// The position the visible bar was drawn for, so it is redrawn as
    /// playback moves rather than showing the moment it appeared.
    scrubber_at: Option<std::time::Duration>,
    /// Set while the handle is being dragged. The bar stays up throughout, and
    /// the position it shows is the one being dragged to rather than the one
    /// playing.
    scrubbing: bool,
    /// Where the left button went down, so a release can tell a click from the
    /// end of a drag. A pan across a film must not also pause it.
    press_at: Option<(f32, f32)>,
    /// Last seek issued during a drag, so a drag across the bar is a handful
    /// of seeks rather than one per pixel of mouse movement.
    scrub_sent: Option<std::time::Instant>,
    /// Where the last exact seek from the bar went. A click is a press and a
    /// release at the same spot, and seeking again on the release threw away
    /// the decoding the press had already done.
    scrub_exact: Option<std::time::Duration>,
    /// A volume change not yet written to the config.
    volume_unsaved: bool,
    /// Silenced, without having forgotten the level to come back to.
    ///
    /// Not a setting: mute is something you do for the length of a phone call,
    /// and a viewer that started up silent because of one would look broken.
    /// It does outlive the file, though — muting a film and stepping to the
    /// next one is not a request to be shouted at again.
    muted: bool,
    /// Media position of the last frame shown, for the seek bar and for
    /// working out where a seek should start from.
    position: std::time::Duration,
    /// Where the picture currently on screen began decoding. Stepping back
    /// works from here rather than from the play position: mid-way through a
    /// group of pictures those are different places, and stepping back from
    /// the play position lands on the keyframe already showing — so during
    /// playback the key appears to do nothing but twitch.
    last_keyframe: std::time::Duration,
    /// Frames already shown, so stepping back is instant.
    frame_cache: crate::framecache::FrameCache,
    /// Filling the gap behind the play head after a jump, when there is one.
    backfill: Option<crate::framecache::BackFill>,
    /// The cache readout, and what it last reported, so it is only rebuilt
    /// when it would actually differ.
    cache_badge: Option<Overlay>,
    cache_shown: Option<(usize, u32)>,
    /// Which frame is being looked at, shown while stepping through them.
    frame_badge: Option<Overlay>,
    /// The window frame, and the size it was built for.
    chrome: Option<(Overlay, Overlay, Overlay, Overlay)>,
    chrome_for: Option<(u32, u32, String)>,
    subtitles: crate::subtitles::Subtitles,
    /// The subtitle menu, when Alt has opened it.
    menu: Option<SubtitleMenu>,
    /// The plugins page, when F8 has opened it.
    plugins: Option<crate::plugins::Plugins>,
    /// Detection, when it is switched on.
    watcher: Option<crate::ml::watcher::Watcher>,
    /// Depth estimation, when it is switched on.
    depth: Option<crate::ml::depth::Depth>,
    /// The newest depth map, and the file it belongs to.
    depth_map: Option<crate::ml::timeline::Stamped<crate::ml::depth::Map>>,
    /// How strongly the map covers the picture: part way, or entirely.
    depth_opacity: f32,
    /// What has been found, against the film's own clock.
    seen: crate::ml::timeline::Timeline<crate::ml::watcher::Seen>,
    /// The bar along the top naming what is in view, and what it was built
    /// for, so it is only redrawn when it would read differently.
    tag_bar: Option<Overlay>,
    tags_shown: Option<crate::ml::bar::Legend>,
    /// The thing a click chose to follow, and when it was last seen.
    following: Option<(u32, usize, std::time::Instant)>,
    /// Answers clicks that mean "what is *that*", off the display thread.
    pointer: Option<crate::ml::pointer::Pointer>,
    /// What the last such click cut out, and which picture it came from.
    cut: Option<crate::ml::segment::Segmented>,
    /// The phrase being typed, while it is being typed. `Some("")` is an
    /// empty prompt that is open, which is not the same as no prompt at all.
    typing: Option<String>,
    /// The worker that answers a phrase, started at the first question.
    finder: Option<crate::ml::finder::Finder>,
    /// What the last phrase found, and which picture it was about.
    found: Vec<crate::ml::segment::Segmented>,
    found_of: Option<(Generation, std::time::Duration)>,
    /// The phrase those belong to, for the readout.
    found_phrase: String,
    cut_of: Option<(u64, std::time::Duration)>,
    /// The size the cut-out was worked out at, which it has to be drawn
    /// against — a photograph is looked at reduced.
    cut_size: Option<(u32, u32)>,
    /// Where to point next, as a fraction of the picture, while a cut-out is
    /// being followed through a moving film. `None` once it is dismissed.
    cut_track: Option<(f32, f32)>,
    /// A question already asked and not yet answered, so only one is in
    /// flight at a time.
    cut_asked: bool,
    /// When the last question was asked, so following does not run flat out.
    cut_last: Option<std::time::Instant>,
    /// Detections sharpened into outlines by the cut-out model, and which
    /// picture they belong to.
    sharp: Vec<crate::ml::segment::Segmented>,
    sharp_of: Option<(u64, std::time::Duration, (u32, u32))>,
    /// A sharpening already asked for and not yet answered.
    sharp_asked: bool,
    /// Where a scrub asked to land. Decoding can only start at a keyframe, so
    /// the frames between it and here are decoded but not shown.
    seek_target: Option<std::time::Duration>,
    /// A seek made while paused, whose frame has not been shown yet: paused
    /// means nothing plays, not that the picture ignores the playhead.
    paused_seek: bool,
    /// Frames were stepped through while paused, so the picture is no longer
    /// where the sound and the decoder stopped. Settled on resuming.
    stepped: bool,
    /// What to draw over the picture, as instance ids.
    labels: Option<crate::ml::labels::Grid>,
    /// The caption on screen, and the cue it was built from, so an unchanged
    /// caption is not re-rasterised every frame.
    caption: Option<Overlay>,
    /// Which cue the caption on screen was built from, and at what size. The
    /// size belongs in the key: a caption rasterised for one scale is the
    /// wrong caption at another, and while paused nothing else would ever
    /// prompt a rebuild.
    caption_for: Option<(usize, std::time::Duration, u32)>,
    /// Ids of the overlays currently on the GPU, so the draw loop can tell
    /// whether the stack changed without being told.
    uploaded: Vec<u64>,
    /// The setup panel, when open. Takes over the keyboard while it is.
    panel: Option<Panel>,
    /// The keyboard reference, when open. Takes over the keyboard likewise —
    /// which is the joke, but it also has to, or the keys being described
    /// would fire underneath it.
    keys: Option<Keys>,
    /// The file list beside the image, when in detailed view.
    browser: Option<Browser>,
    browser_overlay: Option<Overlay>,
    /// The video being played, if the current file is one.
    video: Option<crate::video::Player>,
    /// The next frame, held until its moment arrives.
    next_frame: Option<crate::video::Frame>,
    /// Wall-clock instant that corresponds to the start of the file.
    playback_origin: Option<std::time::Instant>,
    paused: bool,
    /// The decoder has run out of file and the queue is drained.
    playback_ended: bool,
    /// Start the film again when it reaches the end.
    ///
    /// Not a setting: repeat is something you switch on for the thing you are
    /// watching now — a short clip you want to look at twice — and a viewer
    /// that silently looped every film because of a choice made a week ago
    /// would be a viewer that would not stop.
    repeat: bool,
    /// Colour transform for the frames currently being drawn.
    video_colour: Option<crate::video::Colour>,
    /// When each recent frame was shown, for the rate chart.
    frame_times: std::collections::VecDeque<std::time::Instant>,
    /// The chart itself, rebuilt a few times a second.
    fps_chart: Option<Overlay>,
    fps_built: Option<std::time::Instant>,
    /// Measured presentation rate, which is what actually reaches the screen.
    shown_fps: f32,
    /// An image the browser selection has landed on, waiting for the selection
    /// to settle before it is decoded.
    pending_preview: Option<PathBuf>,
    /// When the selection last moved. The preview waits on this rather than
    /// on input in general, so resting a hand on the mouse cannot hold it off.
    selection_moved: std::time::Instant,
    /// Browser zoom step. The bitmap font only scales by whole numbers, so
    /// each pair of steps shares a text size and differs in panel width —
    /// which makes the sequence feel finer than the font alone allows.
    browser_scale: f32,
    /// The grid view, when open. Replaces the image entirely while it is.
    mosaic: Option<Mosaic>,
    /// The composited grid, kept for the software path to blit.
    mosaic_composite: Option<Overlay>,
    /// The folder the grid is walking with its subfolders, while `r` has it
    /// doing so. `None` is the one folder, as usual.
    recursive_root: Option<std::path::PathBuf>,
    /// The grid's corner badges, side by side: the sort order, and whether
    /// subfolders are included. Rebuilt when either changes.
    grid_badge: Overlay,
    /// The folder and name of the grid cell under the pointer, and which
    /// file that label was drawn for.
    hover_label: Option<(std::path::PathBuf, Overlay)>,
    /// The grid's scroll position along the bottom, and the width and mark
    /// position it was drawn for.
    scroll_marker: Option<((u32, u32), Overlay)>,
    /// The playlist was swapped under the grid and no longer holds the image
    /// behind it, so leaving the grid has to decode one even if the selection
    /// never moved.
    current_is_stale: bool,
    /// The grid is on its way out, waiting only for the chosen image to decode.
    leaving_mosaic: bool,
    /// The grid an image was opened from, kept so escape goes back to it
    /// exactly as it was. One step, not a history: opening from the grid again
    /// replaces it.
    grid_behind: Option<GridBehind>,
    /// The file the viewer was opened on, whose folder has yet to be listed.
    folder_to_scan: Option<PathBuf>,
    /// What the detailed view knows about each image, read in the background
    /// and kept for the session, so a folder visited again is instant.
    file_info: std::collections::HashMap<PathBuf, crate::browser::FileInfo>,
    /// The folder whose images are being read right now, if any.
    info_scan: Option<PathBuf>,
    /// The image whose map placement has been asked about, so each is asked
    /// once, and the badge saying it has one.
    geo_for: Option<PathBuf>,
    geo_badge: Option<Overlay>,
    /// Drawing the current image from its tiles, when it has them.
    tiling: Option<Tiling>,
    /// Watching the folder the list came from — and its subfolders, when they
    /// are in the list — for images arriving or leaving; and which folder,
    /// and whether recursively, so a change of folder restarts it.
    folder_watch: Option<crate::watch::Watcher>,
    watching: Option<(PathBuf, bool)>,
    /// The list is files named one by one on the command line, not a folder:
    /// nothing to watch, and nothing to rebuild it from.
    list_named: bool,
    /// This window on the session bus, for the tray icon's menu. Joined after
    /// the first frame, so the bus is never in the way of the first picture.
    presence: Option<crate::tray::Presence>,
    /// Keeps the screensaver off while a video plays.
    awake: crate::awake::Awake,
    /// The open windows, as the list the filename drops down: each row's
    /// span within it and the window it raises, and whether it is down.
    peer_list: Option<(Overlay, Vec<(i32, i32, String)>)>,
    peer_list_open: bool,
    /// Where the filename badge was last drawn — the window switcher's
    /// button — as (x, y, width, height) in window pixels.
    filename_at: Option<(i32, i32, i32, i32)>,
    /// What the badge last said about the tiles, so it is only redrawn when
    /// that changes.
    tile_badge_for: (Option<(usize, usize, usize)>, Option<usize>),
    /// What is on screen is a HEIC's embedded thumbnail, with the real decode
    /// still to come.
    showing_thumbnail: bool,
    /// The load generation whose real decode is already on screen, so its
    /// thumbnail, if it comes late, is dropped.
    sharp_for: Option<Generation>,
    /// Progress readout while the folder is being analysed.
    analysis: Option<Overlay>,
    analysing: bool,
    /// Shared with the background workers that read and fill it.
    thumbs: Option<std::sync::Arc<std::sync::Mutex<crate::thumbs::Cache>>>,
    /// When the user last did anything, for deciding we are idle.
    last_activity: std::time::Instant,
    /// Files still to be given a thumbnail, newest request first.
    thumb_queue: Vec<PathBuf>,
    thumb_workers: usize,
    config: Config,
    /// A pending delete waiting on y/n.
    confirming_delete: Option<Overlay>,
    osd_visible: bool,
    exit_error: Option<anyhow::Error>,
}

impl App {
    fn new(args: Args, proxy: EventLoopProxy<UserEvent>, mut playlist: Playlist) -> Self {
        let config = Config::load();
        // A folder is walked in the chosen order. Files named one by one on
        // the command line keep the order they were named in, and a single
        // file's folder is sorted as it arrives.
        // One image per folder is a way of looking at the grid; opening a
        // single picture should not leave only that picture to walk through.
        let folder_to_scan = (args.files.len() == 1).then(|| args.files[0].clone());
        let list_named = args.files.len() > 1;
        if args.files.is_empty() {
            playlist.sort(match config.sort {
                crate::playlist::Sort::NthPerFolder(_) => crate::playlist::Sort::Folder,
                by => by,
            });
        }
        // The flag forces it on for this run; otherwise the setting decides.
        let view = View {
            autoscale: args.upscale || config.autoscale,
            cover: config.cover,
            ..View::default()
        };
        let fullscreen = !args.windowed;
        App {
            args,
            proxy,
            playlist,
            generation: 0,
            window: None,
            gpu: None,
            software: None,
            pending: None,
            shown: None,
            full_res: FullRes::NotNeeded,
            neighbour: None,
            step_dir: 1,
            view,
            cursor: (0.0, 0.0),
            dragging: false,
            last_click: None,
            shift_held: false,
            ctrl_held: false,
            smooth_override: None,
            fullscreen,
            cursor_icon: CursorIcon::Default,
            focus_until: None,
            presented: false,
            osd: None,
            status_badge: None,
            filename_badge: None,
            pointer_inside: true,
            last_motion: std::time::Instant::now(),
            motion_from: (0.0, 0.0),
            title_bar_was: true,
            notice: None,
            prompt_osd: None,
            volume_osd: None,
            scrubber: None,
            scrubber_at: None,
            scrubbing: false,
            press_at: None,
            scrub_sent: None,
            scrub_exact: None,
            volume_unsaved: false,
            muted: false,
            position: std::time::Duration::ZERO,
            last_keyframe: std::time::Duration::ZERO,
            frame_cache: crate::framecache::FrameCache::new(config.frame_cache_bytes()),
            backfill: None,
            cache_badge: None,
            cache_shown: None,
            frame_badge: None,
            chrome: None,
            chrome_for: None,
            subtitles: crate::subtitles::Subtitles::new(),
            menu: None,
            plugins: None,
            watcher: None,
            depth: None,
            depth_map: None,
            depth_opacity: DEPTH_BLEND,
            seen: crate::ml::timeline::Timeline::new(),
            tag_bar: None,
            tags_shown: None,
            following: None,
            pointer: None,
            cut: None,
            typing: None,
            finder: None,
            found: Vec::new(),
            found_of: None,
            found_phrase: String::new(),
            cut_of: None,
            cut_size: None,
            cut_track: None,
            cut_asked: false,
            cut_last: None,
            sharp: Vec::new(),
            sharp_of: None,
            sharp_asked: false,
            seek_target: None,
            paused_seek: false,
            stepped: false,
            labels: None,
            caption: None,
            caption_for: None,
            uploaded: Vec::new(),
            panel: None,
            keys: None,
            browser: None,
            browser_overlay: None,
            video: None,
            awake: crate::awake::Awake::new(),
            next_frame: None,
            playback_origin: None,
            paused: false,
            playback_ended: false,
            repeat: false,
            video_colour: None,
            frame_times: std::collections::VecDeque::new(),
            fps_chart: None,
            fps_built: None,
            shown_fps: 0.0,
            pending_preview: None,
            selection_moved: std::time::Instant::now(),
            browser_scale: DEFAULT_BROWSER_SCALE,
            mosaic: None,
            recursive_root: None,
            grid_badge: osd::sort_icon(config.sort, osd::SCALE_PANEL),
            hover_label: None,
            scroll_marker: None,
            current_is_stale: false,
            mosaic_composite: None,
            leaving_mosaic: false,
            grid_behind: None,
            folder_to_scan,
            file_info: std::collections::HashMap::new(),
            info_scan: None,
            geo_for: None,
            geo_badge: None,
            tiling: None,
            presence: None,
            folder_watch: None,
            watching: None,
            list_named,
            peer_list: None,
            peer_list_open: false,
            filename_at: None,
            tile_badge_for: (None, None),
            showing_thumbnail: false,
            sharp_for: None,
            analysis: None,
            analysing: false,
            thumbs: crate::thumbs::Cache::open()
                .map(|c| std::sync::Arc::new(std::sync::Mutex::new(c))),
            last_activity: std::time::Instant::now(),
            thumb_queue: Vec::new(),
            thumb_workers: 0,
            config,
            confirming_delete: None,
            osd_visible: true,
            exit_error: None,
        }
    }

    /// The whole window, for placing overlays.
    fn window_size(&self) -> (f32, f32) {
        if let Some(gpu) = &self.gpu {
            return gpu.size();
        }
        match &self.window {
            Some(w) => {
                let s = w.inner_size();
                (s.width.max(1) as f32, s.height.max(1) as f32)
            }
            None => (1.0, 1.0),
        }
    }

    /// The region the picture occupies, which is the whole window unless the
    /// browser is taking the right of it. All view arithmetic uses this, so
    /// fitting and zooming behave as though it were the window.
    fn win_size(&self) -> (f32, f32) {
        let win = self.window_size();
        match self.browser_width() {
            Some(panel) => ((win.0 - panel).max(1.0), win.1),
            None => win,
        }
    }

    /// Text size for the current browser zoom.
    fn browser_scale(&self) -> f32 {
        self.browser_scale
    }

    /// Width of the browser panel, when it is open.
    ///
    /// Grows with the zoom: larger text in a panel of fixed width would just
    /// mean shorter filenames.
    fn browser_width(&self) -> Option<f32> {
        self.browser.as_ref()?;
        let win = self.window_size();
        let fraction = 0.24 + (self.browser_scale - osd::SCALE_MIN) * 0.044;
        Some((win.0 * fraction).clamp(300.0, 900.0).min(win.0 * 0.6))
    }

    /// Whether the pointer is over the browser panel rather than the picture.
    fn cursor_over_browser(&self) -> bool {
        match self.browser_width() {
            Some(panel) => self.cursor.0 >= self.window_size().0 - panel,
            None => false,
        }
    }

    /// How tall the tag bar is, or zero when it is not showing.
    ///
    /// It steals space rather than floating: the codebase's own rule is that a
    /// readout half-hidden by our own furniture looks like a fault, and a bar
    /// over the top of a film hides the film.
    fn tag_bar_height(&self) -> f32 {
        match &self.tag_bar {
            Some(bar) => bar.height as f32,
            None => 0.0,
        }
    }

    fn image_viewport(&self) -> (f32, f32, f32, f32) {
        let size = self.win_size();
        let tags = self.tag_bar_height();
        match self.chrome_metrics() {
            // Inside the border, and under the title bar rather than below
            // it: the bar shows only while the mouse is moving over the
            // window, so the picture has the whole frame the rest of the time
            // and is not refitted each time the bar comes and goes.
            Some(c) => (
                c.border,
                c.border + tags,
                (size.0 - c.border * 2.0).max(1.0),
                (size.1 - c.border * 2.0 - tags).max(1.0),
            ),
            None => (0.0, tags, size.0, (size.1 - tags).max(1.0)),
        }
    }

    /// The title bar shows while the pointer is over the window and has moved
    /// more than [`TITLE_BAR_NUDGE`] within [`TITLE_BAR_LINGER`]: reaching for
    /// the mouse brings it up, and it gets out of the picture's way after.
    fn title_bar_shown(&self) -> bool {
        self.pointer_inside && self.last_motion.elapsed() < TITLE_BAR_LINGER
    }

    /// Redraw when the title bar comes or goes, and say when it next might.
    fn tick_title_bar(&mut self) -> Option<std::time::Instant> {
        let shown = self.title_bar_shown();
        if shown != self.title_bar_was {
            self.title_bar_was = shown;
            self.redraw();
        }
        shown.then(|| self.last_motion + TITLE_BAR_LINGER)
    }

    /// The frame's dimensions, or `None` when there is no frame to draw.
    fn chrome_metrics(&self) -> Option<Chrome> {
        // Fullscreen has no window edge to mark, and the grid and the browser
        // lay themselves out against the whole window.
        if self.fullscreen || self.mosaic.is_some() {
            return None;
        }
        Some(Chrome::for_window(self.window_size()))
    }

    /// What a press on the window frame means, if it landed on one.
    fn chrome_hit(&self, at: (f32, f32)) -> Option<ChromeHit> {
        use winit::window::ResizeDirection::*;
        let c = self.chrome_metrics()?;
        let win = self.window_size();

        // A little wider than the border is drawn: a two-pixel target is a
        // poor thing to have to hit, and the desktop's own edges are generous
        // for the same reason.
        let grab = (c.border * 3.0).max(6.0);
        let (left, right) = (at.0 <= grab, at.0 >= win.0 - grab);
        let (top, bottom) = (at.1 <= grab, at.1 >= win.1 - grab);

        if let Some(direction) = match (top, bottom, left, right) {
            (true, _, true, _) => Some(NorthWest),
            (true, _, _, true) => Some(NorthEast),
            (_, true, true, _) => Some(SouthWest),
            (_, true, _, true) => Some(SouthEast),
            (true, ..) => Some(North),
            (_, true, ..) => Some(South),
            (_, _, true, _) => Some(West),
            (_, _, _, true) => Some(East),
            _ => None,
        } {
            return Some(ChromeHit::Resize(direction));
        }

        // The bar is only there to be pressed while it is showing.
        if at.1 < c.bar_height && self.title_bar_shown() {
            // The close mark sits at the right-hand end of the bar.
            let close_from = win.0 - c.bar_height * 1.6;
            return Some(if at.0 >= close_from { ChromeHit::Close } else { ChromeHit::Move });
        }
        None
    }

    /// Show the desktop's resize arrows over the frame's edges and corners,
    /// as its own decorations would have.
    fn update_cursor_icon(&mut self) {
        use winit::window::ResizeDirection::*;
        let icon = match self.chrome_hit(self.cursor) {
            Some(ChromeHit::Resize(d)) => match d {
                North => CursorIcon::NResize,
                South => CursorIcon::SResize,
                East => CursorIcon::EResize,
                West => CursorIcon::WResize,
                NorthEast => CursorIcon::NeResize,
                NorthWest => CursorIcon::NwResize,
                SouthEast => CursorIcon::SeResize,
                SouthWest => CursorIcon::SwResize,
            },
            _ => CursorIcon::Default,
        };
        if icon != self.cursor_icon {
            self.cursor_icon = icon;
            if let Some(w) = &self.window {
                w.set_cursor(icon);
            }
        }
    }

    /// Build the frame when the window size or the title changes.
    fn tick_chrome(&mut self) {
        let Some(c) = self.chrome_metrics() else {
            if self.chrome.is_some() {
                self.chrome = None;
                self.chrome_for = None;
            }
            return;
        };
        let win = self.window_size();
        let title = self
            .playlist
            .current()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "show".into());
        let key = (win.0 as u32, win.1 as u32, title.clone());
        if self.chrome_for.as_ref() == Some(&key) {
            return;
        }
        self.chrome_for = Some(key);

        let (w, h) = (win.0 as u32, win.1 as u32);
        let (bar, border) = (c.bar_height as u32, c.border as u32);
        self.chrome = Some((
            osd::title_bar(&title.to_uppercase(), w, bar, c.text_scale()),
            osd::chrome_strip(border, h.saturating_sub(bar)),
            osd::chrome_strip(border, h.saturating_sub(bar)),
            osd::chrome_strip(w, border),
        ));
    }

    /// Magnified pixels should stay crisp squares; anything else looks better
    /// filtered. A reduced stand-in is always filtered — its texels are not the
    /// image's pixels, so drawing them as hard squares would be a lie.
    fn smooth(&self) -> bool {
        let magnifying_real_pixels =
            self.view.magnifying() && self.full_res != FullRes::Deferred && self.full_res != FullRes::Loading;
        self.smooth_override.unwrap_or(!magnifying_real_pixels)
    }

    /// Take ownership of a decoded image and show it however we currently can.
    fn install(&mut self, loaded: Loaded, is_full: bool) {
        if self.window.is_none() {
            timing::mark("main", "install_pending");
            self.pending = Some((loaded, is_full));
            return;
        }
        let _s = timing::span("main", "install");

        // The thumbnail and the real decode race; a thumbnail arriving after
        // the real thing would replace a sharp picture with a blurry one.
        let is_thumbnail = loaded.decoded.path == "libheif thumbnail";
        if is_thumbnail && self.sharp_for == Some(self.generation) {
            return;
        }
        if !is_thumbnail && !is_full {
            self.sharp_for = Some(self.generation);
        }

        let Loaded { path, file_size, decoded } = loaded;
        let source = (decoded.source_width as f32, decoded.source_height as f32);
        // Read everything off the decode before its pixel buffer is moved out.
        let (src_w, src_h) = (decoded.source_width, decoded.source_height);
        let (tex_w, tex_h) = (decoded.width, decoded.height);
        let (format, decoder, reduced) = (decoded.format, decoded.path, decoded.is_reduced());
        let adequate_zoom = decoded.adequate_zoom();
        let tiles = decoded.tiles.clone();
        // A tiled image is drawn from its tiles past what the overview covers,
        // so a full-resolution load — capped, and a whole-image decode — is
        // not wanted.
        let tiled = tiles.is_some() && !is_full && self.start_tiling(&path, adequate_zoom);

        // The real decode replacing a thumbnail of itself: the view stays as
        // the user has it, since it is the same picture, only sharper.
        let upgrading = std::mem::replace(&mut self.showing_thumbnail, decoder == "libheif thumbnail")
            && self.shown.as_ref().is_some_and(|s| s.source == source);

        // A turn is for the picture it was asked for, not the next one.
        if !upgrading && !is_full {
            self.view.turn = 0;
        }
        if tiled {
            if !upgrading {
                self.view.fit(source, self.win_size());
            }
            self.full_res = FullRes::NotNeeded;
        } else if !is_full && upgrading {
            self.full_res = if reduced { FullRes::Deferred } else { FullRes::NotNeeded };
        } else if !is_full {
            // Holding only applies once there is a zoom worth carrying: the
            // first image of a session still gets fitted, and so does every
            // image after a fitted one. Holding a fit's percentage would carry
            // one picture's fit onto a picture of another size, and it would
            // no longer fit — what is held is a zoom somebody chose.
            if self.config.hold_zoom && self.shown.is_some() && !self.view.fitting {
                self.view.retarget(source, self.win_size());
            } else {
                self.view.fit(source, self.win_size());
            }
            self.full_res = if reduced { FullRes::Deferred } else { FullRes::NotNeeded };
        } else {
            self.full_res = FullRes::Done;
        }

        self.shown = Some(Displayed {
            source,
            tex: (tex_w, tex_h),
            adequate_zoom,
            pixels: Some(decoded.rgba),
            tiles,
        });
        // The chosen image has arrived, so the grid can go now.
        if !is_full && self.leaving_mosaic {
            self.leaving_mosaic = false;
            self.mosaic = None;
            self.mosaic_composite = None;
        }

        if is_full {
            self.upload_to_gpu_in_background();
        } else {
            self.upload_to_gpu();
        }
        self.build_status_badge();
        self.build_filename_badge();
        if self.browser.is_some() {
            self.build_browser_overlay();
        }
        if self.thumb_queue.is_empty() && self.thumb_workers == 0 {
            self.refill_thumb_queue();
        }

        if let Some(w) = &self.window {
            let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            let place = if self.playlist.len() > 1 {
                format!("  [{}/{}]", self.playlist.position() + 1, self.playlist.len())
            } else {
                String::new()
            };
            w.set_title(&format!("{name} — {src_w}x{src_h} {}{place}", format.name()));
            w.request_redraw();
        }

        eprintln!(
            "{} {}: {}x{} {} · {} · {} · {}",
            if is_full { "full " } else { "shown" },
            path.display(),
            src_w,
            src_h,
            format.name(),
            human_bytes(file_size),
            decoder,
            if reduced {
                format!("decoded to {tex_w}x{tex_h}, full resolution to follow")
            } else {
                "full resolution".to_string()
            }
        );
    }

    /// Hand the current pixels to the GPU, if there is one yet, and release the
    /// CPU copy: only the software path needs it, and that path has ended.
    fn upload_to_gpu(&mut self) {
        let (Some(gpu), Some(shown)) = (self.gpu.as_mut(), self.shown.as_mut()) else { return };
        let Some(pixels) = shown.pixels.take() else { return };

        gpu.upload(&pixels, shown.tex.0, shown.tex.1);
        // The GPU may have shrunk the texture further to fit its own limits, so
        // ask what actually landed rather than assuming.
        shown.adequate_zoom /= gpu.downscaled_by() as f32;
    }

    /// As [`App::upload_to_gpu`], without holding up the window: for the
    /// full-resolution picture arriving behind a reduced one, which is
    /// already on screen and can stay there until the upload lands. Someone
    /// panning or zooming meanwhile should not feel the copy.
    fn upload_to_gpu_in_background(&mut self) {
        let (Some(gpu), Some(shown)) = (self.gpu.as_ref(), self.shown.as_mut()) else { return };
        let Some(pixels) = shown.pixels.take() else { return };

        let proxy = self.proxy.clone();
        let generation = self.generation;
        gpu.upload_in_background(pixels, shown.tex.0, shown.tex.1, move |uploaded| {
            let _ = proxy.send_event(UserEvent::Uploaded(generation, Box::new(uploaded)));
        });
    }

    /// Swap the software surface for a real one. Softbuffer is dropped first:
    /// the two must never hold the window at the same time.
    fn adopt_gpu(&mut self, core: Core) {
        let Some(window) = self.window.clone() else { return };
        self.software = None;

        let gpu = match Gpu::from_core(core, window.clone(), self.args.vsync) {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                // The off-thread adapter was chosen without knowing our window.
                // If it cannot actually present to it, redo the selection the
                // slow, surface-aware way.
                eprintln!("show: off-thread adapter unusable ({e:#}); re-selecting");
                timing::mark("gpu", "adapter_reselect");
                Gpu::new(window.clone(), self.args.adapter.as_deref(), self.args.vsync, self.args.backends).ok()
            }
        };

        match gpu {
            Some(gpu) => {
                self.gpu = Some(gpu);
                timing::mark("gpu", "took_over");
                self.upload_to_gpu();
                    }
            None => {
                // Keep drawing on the CPU rather than showing nothing.
                eprintln!("show: no usable GPU; continuing on the CPU");
                self.software = Software::new(window).ok();
            }
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Move through the folder. The image on screen stays put until the new
    /// one has decoded, which is far calmer than blanking to black on every
    /// key press.
    fn navigate(&mut self, delta: isize) {
        if !self.playlist.step(delta) {
            return;
        }
        if delta != 0 {
            self.step_dir = delta.signum();
        }
        self.start_current();
    }

    fn jump_to_end(&mut self, last: bool) {
        if self.playlist.go_to_end(last) {
            self.start_current();
        }
    }

    fn start_current(&mut self) {
        let Some(path) = self.playlist.current().map(|p| p.to_path_buf()) else { return };

        // Anything still decoding belongs to the previous image now.
        self.generation = self.generation.wrapping_add(1);
        // Its map placement and its tiles belonged to the previous image too.
        self.geo_badge = None;
        self.geo_for = None;
        self.tiling = None;
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.clear_tiles();
        }
        self.stop_video();

        if crate::video::is_video(&path) {
            match crate::video::Player::open(&path, self.config.hardware_decode) {
                Ok(player) => {
                    self.begin_video(player);
                    return;
                }
                // A file we cannot open as video is worth reporting, not
                // worth silently trying to decode as a picture.
                Err(e) => eprintln!("show: {}: {e:#}", path.display()),
            }
        }
        // Each image reports its own load, not a running total. After the first
        // one this correctly excludes GPU startup, which is paid once.
        timing::reset();
        self.presented = false;
        self.osd = None;
        self.full_res = FullRes::NotNeeded;
        // Without hold zoom, every image is fitted. With it, the view is left
        // as it is: fitted stays fitted, and only a zoom somebody chose is
        // carried to the next picture. Clearing it here made every picture
        // keep the last one's percentage, fitted or not.
        if !self.config.hold_zoom {
            self.view.fitting = true;
        }
        // Read ahead already: handed over as if it had just been decoded, so
        // everything after is the same path a fresh load takes.
        if let Some((ahead, Some(loaded))) = self.neighbour.take()
            && ahead == path
        {
            let _ = self.proxy.send_event(UserEvent::Preview(self.generation, Box::new(Ok(loaded))));
            return;
        }
        spawn_load(&self.proxy, path, self.args.io, self.args.preview, false, self.generation);
    }

    /// Read the image the next step will land on, once the current one is up.
    ///
    /// One image, in the direction of travel, at the same resolution the first
    /// paint uses, on a low-priority thread: what pressing next would decode
    /// anyway, done while the person is still looking. A film is never read
    /// ahead, nor anything bigger than the speculative full-size budget.
    fn prefetch_neighbour(&mut self) {
        if self.video.is_some() || self.playlist.len() < 2 {
            return;
        }
        let n = self.playlist.len() as isize;
        let index = ((self.playlist.position() as isize + self.step_dir) % n + n) % n;
        let Some(path) = self.playlist.at(index as usize).map(|p| p.to_path_buf()) else { return };
        if crate::video::is_video(&path) || self.neighbour.as_ref().is_some_and(|(p, _)| *p == path) {
            return;
        }
        self.neighbour = Some((path.clone(), None));
        let (proxy, io, target) = (self.proxy.clone(), self.args.io, self.args.preview);
        let _ = std::thread::Builder::new().name("read-ahead".into()).spawn(move || {
            background_priority();
            let result = Box::new(loader::load(&path, io, target));
            let _ = proxy.send_event(UserEvent::Neighbour(path, result));
        });
    }

    fn stop_video(&mut self) {
        self.video = None;
        self.video_colour = None;
        self.playback_ended = false;
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.clear_yuv();
        }
        self.next_frame = None;
        self.playback_origin = None;
        self.paused = false;
        self.volume_osd = None;
        self.scrubber = None;
        self.scrubber_at = None;
        self.scrubbing = false;
        self.scrub_sent = None;
        self.scrub_exact = None;
        self.position = std::time::Duration::ZERO;
        self.last_keyframe = std::time::Duration::ZERO;
        self.frame_cache.clear();
        self.backfill = None;
        self.cache_badge = None;
        self.cache_shown = None;
        self.frame_badge = None;
        self.subtitles.clear();
        self.seen.clear();
        self.tags_shown = None;
        self.following = None;
        self.forget_cut();
        self.menu = None;
        self.caption = None;
        self.caption_for = None;
    }

    /// Show a video: size the view to it and let the clock start on the first
    /// frame rather than now, so a slow open does not eat the beginning.
    fn begin_video(&mut self, player: crate::video::Player) {
        let info = player.info.clone();
        self.frame_cache.set_interval(
            (info.frame_rate > 0.0).then(|| std::time::Duration::from_secs_f64(1.0 / info.frame_rate)),
        );
        let source = (info.width as f32, info.height as f32);
        eprintln!(
            "video {}: {}x{} {} {:.3} fps{} · {}",
            player.path.display(),
            info.width,
            info.height,
            info.codec,
            info.frame_rate,
            info.duration
                .map(|d| format!(" {:.1}s", d.as_secs_f64()))
                .unwrap_or_default(),
            match &info.hardware {
                Some(node) => format!("hardware decode on {node}"),
                None => "software decode".to_string(),
            }
        );

        // Tracks in the container, then any file sitting beside it. External
        // files are numbered above the container's streams so the two sets of
        // identifiers cannot collide.
        self.subtitles.clear();
        for track in self.video_subtitle_tracks(&player) {
            self.subtitles.add_track(track);
        }
        self.load_external_subtitles(&player.path.clone());

        // Whatever the volume was last set to, including across sessions —
        // unless the sound is muted, which follows the session rather than the
        // file.
        if let Some(track) = player.audio() {
            track.volume.set(if self.muted { 0.0 } else { self.config.volume });
        }

        self.video = Some(player);
        self.playback_origin = None;
        self.paused = false;
        self.shown = Some(Displayed {
            source,
            tex: (info.width, info.height),
            adequate_zoom: 1.0,
            pixels: None,
            tiles: None,
        });
        self.view.turn = 0;
        self.view.fit(source, self.win_size());
        self.build_status_badge();
        self.build_filename_badge();
        self.presented = false;
        self.redraw();
    }

    /// Present whatever frame is due, and say when the next one is.
    fn tick_video(&mut self) -> Option<std::time::Instant> {
        self.video.as_ref()?;
        if self.paused {
            return self.show_paused_seek();
        }

        let mut caught_up = 0usize;
        loop {
            if self.next_frame.is_none() {
                self.next_frame = self.video.as_mut().and_then(|p| p.try_next());
            }
            let Some(frame) = self.next_frame.as_ref() else {
                // Out of file: the last frame stays up, and the readout says so
                // rather than leaving a stalled rate to be puzzled over.
                if self.video.as_ref().is_some_and(|p| p.finished()) {
                    if self.repeat {
                        // Back to the top rather than stopping. Exact, so a
                        // film with a cold open starts where it starts and not
                        // at the first keyframe after it.
                        self.seek_to(std::time::Duration::ZERO, false, true);
                        return Some(std::time::Instant::now());
                    }
                    if !self.playback_ended {
                        self.playback_ended = true;
                        self.build_fps_chart(std::time::Instant::now());
                        self.redraw();
                    }
                    return None;
                }
                // Nothing decoded yet. Come back shortly rather than sleeping
                // until the next input event.
                return Some(std::time::Instant::now() + std::time::Duration::from_millis(4));
            };

            // With sound, the speaker sets the timeline and the picture is
            // shown against it: audio that stretches to match the video is
            // audible, whereas a frame held a moment longer is not.
            let catching_up = self
                .seek_target
                .is_some_and(|t| frame.at + std::time::Duration::from_millis(20) < t);
            if let Some(now) = self.video.as_ref().and_then(|p| p.audio_position())
                && !catching_up
            {
                if frame.at > now {
                    // Wake near the frame's due time rather than spinning.
                    let wait = (frame.at - now).min(std::time::Duration::from_millis(32));
                    return Some(std::time::Instant::now() + wait);
                }
            } else if !catching_up {
                // No audio: the first frame's own timestamp is the zero point,
                // however long opening the file took.
                let origin = *self
                    .playback_origin
                    .get_or_insert_with(|| std::time::Instant::now() - frame.at);
                let due = origin + frame.at;
                if due > std::time::Instant::now() {
                    return Some(due);
                }
            }

            // Still catching up to where the scrub asked for: decode past it
            // rather than showing it. Bounded, so a long gap between keyframes
            // cannot lock the window up — the rest is caught next time round.
            if let Some(target) = self.seek_target {
                let early = frame.at + std::time::Duration::from_millis(20) < target;
                if early {
                    self.next_frame = None;
                    caught_up += 1;
                    if caught_up < 400 {
                        continue;
                    }
                    return Some(std::time::Instant::now());
                }
                self.seek_target = None;
            }

            let frame = self.next_frame.take().expect("checked above");
            // Sound starts with the first picture, so the two begin together
            // rather than the audio running ahead while the decoder warms up.
            // From the picture's moment, too: after a seek the sound can begin
            // well short of the first frame — at the keyframe, or wherever a
            // badly indexed file's guess put it — and the picture would sit
            // frozen while it caught up. Until the sound has a timeline there
            // is nothing to line up, so the picture runs on the wall clock and
            // the sound joins it once it has one.
            if let Some(audio) = self.video.as_ref().and_then(|p| p.audio())
                && !audio.clock.is_playing()
                && audio.clock.position().is_some()
            {
                audio.skipper.skip_to(frame.at);
                audio.clock.set_playing(true);
            }
            self.show_frame(frame);
        }
    }

    /// While paused: if the playhead was moved, decode forward to where it
    /// now is and show that one frame — no sound, no playing on.
    ///
    /// A seek lands on a keyframe, often well before the target, so frames
    /// short of it are decoded past rather than shown. Bounded per call, as
    /// playback's catching up is, so a long gap between keyframes cannot lock
    /// the window; the rest is done on the next turn of the loop.
    fn show_paused_seek(&mut self) -> Option<std::time::Instant> {
        if !self.paused_seek {
            return None;
        }
        let target = self.seek_target.unwrap_or(self.position);
        let started = std::time::Instant::now();
        let mut last_early = None;
        loop {
            let next = self.next_frame.take().or_else(|| self.video.as_mut().and_then(|p| p.try_next()));
            let Some(frame) = next else {
                if self.video.as_ref().is_some_and(|p| p.finished()) {
                    // The end came first: the last frame there is, is it.
                    if let Some(frame) = last_early {
                        self.land_paused(frame);
                    }
                    self.paused_seek = false;
                    return None;
                }
                return Some(std::time::Instant::now() + std::time::Duration::from_millis(4));
            };
            if frame.at + std::time::Duration::from_millis(20) < target {
                last_early = Some(frame);
                if started.elapsed() < std::time::Duration::from_millis(30) {
                    continue;
                }
                return Some(std::time::Instant::now());
            }
            self.land_paused(frame);
            self.paused_seek = false;
            return None;
        }
    }

    /// Show the frame a paused seek arrived at, and keep it for stepping.
    fn land_paused(&mut self, frame: crate::video::Frame) {
        self.seek_target = None;
        if !self.scrubbing {
            self.position = frame.at;
        }
        self.present_frame(&frame);
        self.frame_cache.push(frame);
        self.build_frame_badge();
        self.caption_for = None;
        self.tick_subtitles();
        self.redraw();
    }

    fn show_frame(&mut self, frame: crate::video::Frame) {
        // A drag owns the readout until it ends; otherwise frames arriving
        // from the seek would drag the handle back out from under the pointer.
        if !self.scrubbing {
            self.position = frame.at;
        }
        if frame.key {
            self.last_keyframe = frame.at;
        }
        // Every frame's moment is kept, so the chart can show the shape of the
        // rate rather than an average that hides every stutter.
        let now = std::time::Instant::now();
        self.frame_times.push_back(now);
        while self.frame_times.front().is_some_and(|t| now.duration_since(*t) > FPS_WINDOW) {
            self.frame_times.pop_front();
        }

        let rebuild = self.fps_built.is_none_or(|t| now.duration_since(t) >= FPS_REFRESH);
        if rebuild {
            self.fps_built = Some(now);
            self.build_fps_chart(now);
            if self.args.timing {
                eprintln!("show: presenting {:.1} fps", self.shown_fps);
            }
        }

        self.present_frame(&frame);
        // Offered to detection, which takes it or does not. Never blocks, and
        // never copies the picture — the planes are shared.
        if let Some(watcher) = self.watcher.as_mut() {
            watcher.offer(&frame, self.generation);
        }
        // Kept so stepping back a frame costs nothing. It was decoded anyway.
        self.frame_cache.push(frame);
    }

    /// Put a frame on screen. Takes a reference, so a frame held in the cache
    /// can be shown again without being given up.
    fn present_frame(&mut self, frame: &crate::video::Frame) {
        // Every frame put on screen, stepped to or played: the map is meant
        // to be of the picture that is showing.
        if let Some(depth) = self.depth.as_ref() {
            depth.offer(frame, self.generation);
        }
        if let Some(gpu) = self.gpu.as_mut() {
            let planes = [
                frame.planes[0].as_slice(),
                frame.planes[1].as_slice(),
                frame.planes[2].as_slice(),
            ];
            gpu.upload_yuv(planes, (frame.width, frame.height), frame.chroma);
        } else if let Some(shown) = self.shown.as_mut() {
            // No GPU to convert on, so it happens here. Only the fallback
            // path pays this.
            shown.pixels = Some(crate::video::to_rgba(frame));
        }
        if let Some(shown) = self.shown.as_mut() {
            shown.tex = (frame.width, frame.height);
        }
        self.video_colour = Some(frame.colour);
        self.redraw();
    }

    /// Bucket recent frames into a rate over time, and draw it.
    fn build_fps_chart(&mut self, now: std::time::Instant) {
        let Some(player) = self.video.as_ref() else {
            self.fps_chart = None;
            return;
        };
        let target = player.info.frame_rate as f32;
        let codec = player.info.codec.clone();

        let buckets = (FPS_WINDOW.as_secs_f32() / FPS_BUCKET.as_secs_f32()).round() as usize;
        let mut counts = vec![0f32; buckets];
        for shown in &self.frame_times {
            let age = now.duration_since(*shown).as_secs_f32();
            // Oldest on the left, newest on the right.
            let index = buckets as f32 - 1.0 - (age / FPS_BUCKET.as_secs_f32());
            if index >= 0.0 && (index as usize) < buckets {
                counts[index as usize] += 1.0;
            }
        }
        // Frames per bucket to frames per second.
        let per_second = 1.0 / FPS_BUCKET.as_secs_f32();
        for c in counts.iter_mut() {
            *c *= per_second;
        }

        // Buckets from before the first frame have no rate to report. Plotting
        // them as zero draws a floor across the left of the chart and makes
        // every film look as though it began by stalling; the bucket holding
        // the first frame goes too, since only part of it was playing and it
        // would read low for the same reason.
        let mut samples: Vec<Option<f32>> = counts.iter().map(|c| Some(*c)).collect();

        // The newest bucket is only part-way through: it covers the quarter
        // second up to now, but only the elapsed fraction of that has had a
        // chance to put frames in it, so it always reads low and the line
        // dives at the right-hand end. It is not a measurement yet.
        if let Some(last) = samples.last_mut() {
            *last = None;
        }
        let known_from = match self.frame_times.front() {
            Some(oldest) => {
                let age = now.duration_since(*oldest).as_secs_f32();
                let index = buckets as f32 - 1.0 - age / FPS_BUCKET.as_secs_f32();
                (index.max(0.0) as usize + 1).min(buckets)
            }
            None => buckets,
        };
        for sample in samples.iter_mut().take(known_from) {
            *sample = None;
        }

        // Measured from the actual interval the frames span rather than by
        // counting them into a one-second bin, which can only ever produce
        // whole numbers and hides the difference between 29.97 and 30.
        let recent: Vec<&std::time::Instant> = self
            .frame_times
            .iter()
            .filter(|t| now.duration_since(**t) <= std::time::Duration::from_secs(1))
            .collect();
        self.shown_fps = match (recent.first(), recent.last()) {
            (Some(first), Some(last)) if recent.len() > 1 => {
                let span = last.duration_since(**first).as_secs_f32();
                if span > 0.0 { (recent.len() - 1) as f32 / span } else { 0.0 }
            }
            _ => 0.0,
        };

        // Says which path is running: the difference is large enough to want
        // to know without checking the terminal.
        let hardware = self.video.as_ref().is_some_and(|p| p.info.hardware.is_some());
        let label = if self.playback_ended {
            "ENDED".to_string()
        } else if self.paused {
            "PAUSED".to_string()
        } else if hardware {
            format!("{codec} HW")
        } else {
            codec.clone()
        };
        self.fps_chart = Some(osd::chart(
            &label,
            &format!("{:.1} / {:.2} FPS", self.shown_fps, target),
            &samples,
            target,
            // Half the size of the panel text: it is a glance, not a readout,
            // and at full size it took a quarter of the picture.
            osd::SCALE_PANEL / 2.0,
        ));
    }

    /// Fetch full resolution, if it is wanted and not already on its way.
    fn request_full_res(&mut self) {
        if self.full_res != FullRes::Deferred {
            return;
        }
        let Some(path) = self.playlist.current().map(|p| p.to_path_buf()) else { return };
        self.full_res = FullRes::Loading;
        timing::mark("decode", "full_res_requested");
        spawn_load(&self.proxy, path, self.args.io, None, true, self.generation);
    }

    /// Speculatively load full resolution once the first frame is up, as long
    /// as it is cheap enough to be worth doing unasked.
    fn maybe_prefetch_full_res(&mut self) {
        if self.full_res != FullRes::Deferred || self.video.is_some() {
            return;
        }
        let Some(shown) = &self.shown else { return };
        if (shown.source.0 as u64 * shown.source.1 as u64 * 4) <= SPECULATIVE_FULL_RES_BUDGET {
            self.request_full_res();
        }
    }

    /// Grow or shrink the window, keeping its shape, and the picture with it.
    ///
    /// Fitted, the picture follows the window on its own when the resize
    /// lands. Zoomed, it is scaled by the same amount, or a bigger window
    /// would only show more of the margin round a picture that stayed put.
    fn resize_window_by(&mut self, factor: f32) {
        // Smallest a window may be made this way, on its shorter side.
        const SMALLEST: f32 = 200.0;
        if self.fullscreen {
            return;
        }
        let Some(window) = self.window.as_ref() else { return };
        let now = window.inner_size();
        let (w, h) = (now.width.max(1) as f32, now.height.max(1) as f32);
        let mut scale = factor;
        // No bigger than the screen, no smaller than usable.
        if let Some(monitor) = window.current_monitor() {
            let screen = monitor.size();
            scale = scale.min((screen.width as f32 / w).min(screen.height as f32 / h));
        }
        scale = scale.max(SMALLEST / w.min(h));
        if (scale - 1.0).abs() < 1e-3 {
            return;
        }
        // A maximized window refuses to be resized.
        window.set_maximized(false);
        let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(
            (w * scale).round() as u32,
            (h * scale).round() as u32,
        ));
        if !self.view.fitting {
            let middle = (w / 2.0, h / 2.0);
            self.zoom_by(scale, middle);
        }
    }

    fn zoom_by(&mut self, factor: f32, at: (f32, f32)) {
        let Some(shown) = &self.shown else { return };
        let (source, adequate) = (shown.source, shown.adequate_zoom);
        self.view.zoom_about(factor, at, source, self.win_size());
        self.build_status_badge();

        // Past this point the stand-in is being magnified, so the detail is
        // worth waiting for even if it was too expensive to prefetch. A film
        // has no fuller resolution to fetch, and asking would set an image
        // load going against the video's own path.
        if self.video.is_none() && self.view.zoom > adequate {
            self.request_full_res();
        }
        self.redraw();
    }

    /// Rasterise the breakdown for the image just shown, and put it on the GPU
    /// if the GPU is the one drawing.
    fn build_osd(&mut self) {
        self.osd = Some(osd::render_scaled(&osd::rows(), osd::SCALE_DEBUG));
    }

    /// The filename, larger than the diagnostics beneath it.
    fn build_filename_badge(&mut self) {
        let name = self.playlist.current().and_then(|p| p.file_name()).map(|n| {
            let mut label = n.to_string_lossy().into_owned();
            if self.playlist.len() > 1 {
                label.push_str(&format!(
                    "   {}/{}",
                    self.playlist.position() + 1,
                    self.playlist.len()
                ));
            }
            label
        });
        // Bounded by the window, not by a character count: there is a whole
        // screen of horizontal space and no reason to shorten a filename.
        let width = (self.win_size().0 * 0.9) as u32;
        // With other windows open the name is also the switcher's button, and
        // says so.
        let more = self.peer_list.is_some();
        self.filename_badge = name.map(|label| {
            let label = if more { format!("{label}   v") } else { label };
            osd::render_within(&[osd::Row::plain(label)], osd::SCALE_PANEL, width)
        });
    }

    /// The zoom and memory readout. Rebuilt only when something it shows has
    /// changed, so dragging a wheel does not re-rasterise text every frame.
    fn build_status_badge(&mut self) {
        if self.shown.is_none() && self.mosaic.is_none() {
            self.status_badge = None;
            return;
        }

        let zoom = format!("{:.0}%", self.view.percent());

        // The frame rate lives in its own chart; this stays about the view.
        // The mode as well as the number: the number alone cannot say whether
        // the next picture will be fitted or keep this percentage.
        let mode = match (self.view.fitting, self.view.cover) {
            (true, true) => "COVER",
            (true, false) => "FIT",
            (false, _) if self.config.hold_zoom => "HELD",
            (false, _) => "MANUAL",
        };
        let mut rows = vec![osd::Row::good(format!("ZOOM ({mode})"), zoom)];
        rows.extend([
            // Worth stating plainly: it explains why a small image is filling
            // the screen, or why it is not.
            osd::Row::dim("AUTOSCALE", if self.view.autoscale { "ON" } else { "OFF" }),
            osd::Row::dim("FIT", if self.view.cover { "COVER" } else { "WHOLE IMAGE" }),
        ]);
        // Worth saying: a held zoom is why the next image did not fit itself.
        if self.config.hold_zoom {
            rows.push(osd::Row::dim("HOLD ZOOM", "ON"));
        }
        // Drawing from tiles: the level being drawn, and how much of the
        // screen's worth of it has arrived.
        if let Some((level, have, need)) = self.tiling.as_ref().and_then(|t| t.drawing) {
            let (w, h) = self.tiling.as_ref().map(|t| (t.pyramid.levels[level].width, t.pyramid.levels[level].height)).unwrap_or((0, 0));
            let held = self.gpu.as_ref().map(|g| g.tile_usage()).unwrap_or((0, 0));
            rows.push(osd::Row::good("DRAWING", format!("LEVEL {} · {w}x{h}", level + 1)));
            rows.push(osd::Row::dim("ON SCREEN", format!("{have} / {need} TILES")));
            rows.push(osd::Row::dim(
                "ON CARD",
                format!("{} TILES · {}", held.0, crate::loader::human_bytes(held.1)),
            ));
        }
        // A tiled image: every level the file holds, the one being shown in
        // brackets, and how much it was shrunk on top of that.
        if let Some(t) = self.shown.as_ref().and_then(|s| s.tiles.as_ref()) {
            let (w, h) = t.levels.get(t.chosen).copied().unwrap_or((0, 0));
            let shrink = if t.shrink > 1 { format!(" /{}", t.shrink) } else { String::new() };
            rows.push(osd::Row::good(
                "OVERVIEW",
                format!("LEVEL {} OF {} · {w}x{h}{shrink}", t.chosen + 1, t.levels.len()),
            ));
            rows.push(osd::Row::dim("TILE SIZE", format!("{}x{}", t.tile.0, t.tile.1)));
            let widths: Vec<String> = t
                .levels
                .iter()
                .enumerate()
                .map(|(i, (w, _))| if i == t.chosen { format!("[{w}]") } else { w.to_string() })
                .collect();
            // Wrapped a few to a row: a nine-level pyramid on one line would
            // stretch the badge across the screen.
            for (i, chunk) in widths.chunks(5).enumerate() {
                rows.push(osd::Row::dim(if i == 0 { "LEVELS" } else { "" }, chunk.join(" ")));
            }
        }
        if let Some(bytes) = config::process_memory_bytes() {
            let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let budget = self.config.memory_budget_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
            rows.push(osd::Row::dim("MEM", format!("{gb:.2} / {budget:.1} GB")));
        }
        // Which model is looking, and how often. Choosing one on F8 and seeing
        // no difference should be answerable by looking, rather than by
        // wondering whether the choice took.
        if let Some(watcher) = self.watcher.as_ref() {
            let spec = crate::ml::catalogue::spec(crate::ml::watcher::chosen());
            let where_ = if crate::ml::runtime::on_card().is_none() { "CARD" } else { "CPU" };
            rows.push(osd::Row::dim(
                "LOOKING",
                format!("{} · {} HZ · {where_}", spec.version, self.config.look_hz),
            ));
            let stats = watcher.stats();
            rows.push(osd::Row::dim(
                "LAST LOOK",
                format!("{:.0} MS · {} SEEN", stats.last.as_secs_f64() * 1000.0, stats.examined),
            ));
        }
        self.status_badge = Some(osd::render_scaled(&rows, osd::SCALE_DEBUG));
    }

    /// Rebuild the zoom badge when what the tiles say about themselves has
    /// changed — the level, or how many have arrived — and not otherwise.
    fn update_tile_badge(&mut self) {
        let now = self.tiling.as_ref().and_then(|t| t.drawing);
        let usage = self.gpu.as_ref().map(|g| g.tile_usage().0);
        if (now, usage) != self.tile_badge_for {
            self.tile_badge_for = (now, usage);
            self.build_status_badge();
        }
    }

    /// Delete the current image, asking first unless told not to.
    fn request_delete(&mut self) {
        let Some(path) = self.playlist.current().map(|p| p.to_path_buf()) else { return };
        if !self.config.confirm_delete {
            self.perform_delete();
            return;
        }

        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let fate = if self.config.delete_to_trash { "MOVE TO TRASH" } else { "DELETE PERMANENTLY" };
        self.confirming_delete = Some(osd::message(&[
            fate,
            &name,
            "",
            "Y  CONFIRM        N OR ESC  CANCEL",
        ]));
        self.redraw();
    }

    fn cancel_delete(&mut self) {
        self.confirming_delete = None;
        self.redraw();
    }

    fn perform_delete(&mut self) {
        self.confirming_delete = None;
        let Some(path) = self.playlist.current().map(|p| p.to_path_buf()) else { return };

        match config::delete_file(&path, self.config.delete_to_trash) {
            Ok(what) => {
                let verb = match what {
                    config::Deleted::Trashed => "trashed",
                    config::Deleted::Removed => "deleted",
                };
                eprintln!("show: {verb} {}", path.display());
            }
            Err(e) => {
                // Nothing was removed, so the view must not move on.
                eprintln!("show: {e:#}");
                    self.redraw();
                return;
            }
        }

        // The grid set aside keeps its own list, which may hold the file too.
        if let Some(behind) = self.grid_behind.as_mut()
            && behind.playlist.remove(&path)
        {
            behind.thumbs_valid = false;
        }
        if self.playlist.remove_current() {
            self.start_current();
        } else {
            // That was the last one.
            self.shown = None;
            self.osd = None;
            self.notice = None;
        }
        self.redraw();
    }



    /// Hash every image in the folder, then reorder the grid so alike pictures
    /// sit together.
    fn start_analysis(&mut self) {
        if self.analysing || self.playlist.is_empty() {
            return;
        }
        let paths: Vec<PathBuf> =
            (0..self.playlist.len()).filter_map(|i| self.playlist.at(i).map(|p| p.to_path_buf())).collect();

        self.analysing = true;
        self.build_analysis_overlay(0, paths.len());

        let proxy = self.proxy.clone();
        let (io, generation) = (self.args.io, self.generation);
        std::thread::Builder::new()
            .name("similarity".into())
            .spawn(move || {
                let progress = |done, total| {
                    let _ = proxy.send_event(UserEvent::AnalysisProgress(generation, done, total));
                };
                let (hashes, cache) = crate::similarity::analyse(&paths, io, progress);
                if let Err(e) = cache.save() {
                    eprintln!("show: could not save the similarity cache: {e:#}");
                }
                let _ = proxy.send_event(UserEvent::AnalysisDone(generation, Box::new(hashes)));
            })
            .expect("spawning the similarity thread");
    }

    /// Queue every file in the folder that has no thumbnail yet.
    ///
    /// Only ever done while idle, and one batch at a time: generating
    /// thumbnails is a courtesy to the next visit, not something worth making
    /// the current one worse for.
    fn refill_thumb_queue(&mut self) {
        let Some(cache) = self.thumbs.clone() else { return };
        let Ok(cache) = cache.lock() else { return };

        self.thumb_queue = (0..self.playlist.len())
            .filter_map(|i| self.playlist.at(i))
            .filter(|p| !cache.has(p))
            .map(|p| p.to_path_buf())
            .collect();
        // Nearest to the current image first: `tick_idle_thumbnails` pops from
        // the end, so the closest files are placed last.
        let here = self.playlist.position() as i64;
        let distance: std::collections::HashMap<PathBuf, i64> = (0..self.playlist.len())
            .filter_map(|i| self.playlist.at(i).map(|p| (p.to_path_buf(), (i as i64 - here).abs())))
            .collect();
        self.thumb_queue
            .sort_by_key(|p| std::cmp::Reverse(distance.get(p).copied().unwrap_or(i64::MAX)));
    }

    /// Start a few background thumbnail generations, if we are idle enough.
    fn tick_idle_thumbnails(&mut self) {
        const IDLE_AFTER: std::time::Duration = std::time::Duration::from_millis(400);
        const IN_FLIGHT: usize = 3;

        if self.analysing
            || self.thumb_workers >= IN_FLIGHT
            || self.last_activity.elapsed() < IDLE_AFTER
        {
            return;
        }
        let Some(cache) = self.thumbs.clone() else { return };

        while self.thumb_workers < IN_FLIGHT {
            let Some(path) = self.thumb_queue.pop() else { return };
            if cache.lock().is_ok_and(|c| c.has(&path)) {
                continue;
            }

            self.thumb_workers += 1;
            let (proxy, io) = (self.proxy.clone(), self.args.io);
            let generation = self.generation;
            let cache = cache.clone();
            rayon::spawn(move || {
                if QUITTING.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                // Decoded at thumbnail size, not full size: the point is a
                // small picture, and this is running behind the user's back.
                let target = crate::decode::Target {
                    width: crate::thumbs::THUMB_SIZE,
                    height: crate::thumbs::THUMB_SIZE,
                };
                // Shrunk and encoded first, with nothing held; the cache only
                // for writing the result down.
                if let Ok(loaded) = loader::load(&path, io, Some(target)) {
                    let d = &loaded.decoded;
                    if let Ok(encoded) = crate::thumbs::encode(&d.rgba, d.width, d.height)
                        && let Ok(mut cache) = cache.lock()
                    {
                        let _ = cache.record(&path, &encoded);
                    }
                }
                let _ = proxy.send_event(UserEvent::ThumbnailWritten(generation));
            });
        }
    }

    fn build_analysis_overlay(&mut self, done: usize, total: usize) {
        const BAR: usize = 24;
        let filled = if total == 0 { 0 } else { done * BAR / total };
        let bar: String = (0..BAR).map(|i| if i < filled { '=' } else { '-' }).collect();
        self.analysis = Some(osd::render_scaled(
            &[
                osd::Row::good("COMPARING IMAGES", format!("{done} / {total}")),
                osd::Row::dim(format!("[{bar}]"), ""),
            ],
            osd::SCALE_PANEL,
        ));
    }

    fn toggle_mosaic(&mut self) {
        if self.mosaic.is_some() {
            self.close_mosaic(false);
        } else if self.grid_behind.is_some() {
            self.return_to_grid();
        } else {
            // The grid always opens in folder order: this folder's images,
            // then its subfolders' when `r` has them, as the tree has them. A
            // sort or filter picked with `s` lasts until the grid is left; one
            // carried into the next opening — a one-per-folder filter above
            // all — could leave out the picture it was opened from and show
            // other folders' instead. Coming back with escape is different,
            // and restores the grid as it was.
            if self.config.sort != crate::playlist::Sort::Folder {
                self.config.sort = crate::playlist::Sort::Folder;
                self.save_settings();
                self.build_grid_badge();
            }
            let before = self.playlist.current().map(|p| p.to_path_buf());
            self.playlist.sort(crate::playlist::Sort::Folder);
            self.current_is_stale |= self.playlist.current() != before.as_deref();
            self.mosaic = Some(self.new_mosaic(self.playlist.position()));
            self.request_thumbs();
            self.redraw();
        }
    }

    /// Leave the grid, showing whichever image the grid had selected.
    ///
    /// `keep` is for opening an image from it: the grid is set aside rather
    /// than dropped, so escape can bring it back where it was.
    fn close_mosaic(&mut self, keep: bool) {
        let Some(mosaic) = self.mosaic.take() else { return };
        let stale = std::mem::take(&mut self.current_is_stale);
        if stale || mosaic.selected() != self.playlist.position() {
            self.playlist.go_to(mosaic.selected());
            self.start_current();
        } else {
            // Same image: put the single view back without re-decoding it.
            }
        self.hover_label = None;
        self.grid_behind = keep.then(|| GridBehind {
            mosaic,
            playlist: self.playlist.clone(),
            thumbs_valid: true,
        });
        self.redraw();
    }

    /// Bring back the grid the current image was opened from, scrolled where
    /// it was, with whatever is showing now selected if the grid has it.
    fn return_to_grid(&mut self) {
        let Some(GridBehind { mut mosaic, playlist, thumbs_valid }) = self.grid_behind.take()
        else {
            return;
        };
        let showing = self.playlist.current().map(|p| p.to_path_buf());
        self.playlist = playlist;
        if let Some(at) = showing.as_deref().and_then(|p| self.playlist.position_near(p)) {
            self.playlist.go_to(at);
        }
        // Wandered off to a folder the grid does not hold: the grid's own
        // selection stands, and leaving it again has to load that image.
        self.current_is_stale |= self.playlist.current() != showing.as_deref();

        let (win, count) = (self.win_size(), self.playlist.len());
        // Deleting a file shifts every index after it, so the row breaks are
        // worked out again from the list as it is now.
        let fresh = self.new_mosaic(0);
        mosaic.set_breaks(fresh.breaks().to_vec());
        mosaic.resume(self.playlist.position(), win, count, thumbs_valid);
        self.mosaic = Some(mosaic);
        self.request_thumbs();
        self.redraw();
    }

    /// Get ready to draw `path` from its tiles. False when it cannot be: no
    /// pyramid to be read after all, or no GPU to draw tiles with.
    fn start_tiling(&mut self, path: &std::path::Path, overview_zoom: f32) -> bool {
        if self.gpu.is_none() && self.software.is_none() {
            return false;
        }
        let Ok(Some(pyramid)) = crate::tiled::probe(path) else {
            self.tiling = None;
            return false;
        };
        // The levels the file has, and any stored for it: loading it made
        // them, if it is large and came without.
        let pyramid = Arc::new(pyramid);
        let (proxy, generation) = (self.proxy.clone(), self.generation);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(2, 8);
        let loader = crate::tiles::TileLoader::new(pyramid.clone(), threads, move |key, tile| {
            let _ = proxy.send_event(UserEvent::Tile(generation, key, Box::new(tile)));
        });
        self.tiling = Some(Tiling {
            pyramid,
            loader,
            overview_zoom,
            requested: Vec::new(),
            in_use: std::collections::HashSet::new(),
            drawing: None,
        });
        true
    }

    /// The tiles to draw this frame over the overview, coarsest first, and the
    /// ones still missing asked for. `image_rect` is where the whole image
    /// lands, in NDC; each tile is placed as its share of that.
    fn plan_tiles(&mut self, image_rect: [f32; 4]) -> Vec<(crate::tiles::TileKey, [f32; 4], [f32; 4])> {
        let viewport = self.image_viewport();
        let (Some(t), Some(gpu)) = (self.tiling.as_mut(), self.gpu.as_mut()) else { return Vec::new() };
        let view = self.view;
        // Tiles are cut from the file as it is, so they are planned in its
        // coordinates: the middle of the window back out of the turn, and the
        // window itself on its side when the picture is.
        let full = t.pyramid.full_size();
        let full = (full.0 as f32, full.1 as f32);
        let middle = crate::view::unturn_point(view.turn, view.center, full);
        let window = if view.turn % 2 == 1 { (viewport.3, viewport.2) } else { (viewport.2, viewport.3) };

        // The overview is all the screen can use: nothing to read.
        let plan = (view.zoom > t.overview_zoom * 1.01)
            .then(|| crate::tiles::plan(&t.pyramid, view.zoom, middle, window))
            .filter(|p| p.visible.len() <= MAX_TILES_ON_SCREEN);
        let Some(plan) = plan else {
            if !t.requested.is_empty() {
                t.loader.request(std::iter::empty(), |_| true);
                t.requested.clear();
            }
            t.in_use.clear();
            t.drawing = None;
            return Vec::new();
        };

        // Asked again when the view moves, and when the loader has gone quiet
        // with tiles still missing — one lost to eviction before the view
        // settled would otherwise never come back.
        let missing = plan.visible.iter().any(|k| !gpu.has_tile(k));
        if plan.visible != t.requested || (missing && t.loader.outstanding() == 0) {
            t.loader.request(plan.wanted(), |k| gpu.has_tile(k));
            t.requested = plan.visible.clone();
        }

        let place = |key: &crate::tiles::TileKey| -> [f32; 4] {
            let level = t.pyramid.levels[key.level];
            let (lw, lh) = (level.width as f32, level.height as f32);
            let fx0 = (key.tx * level.tile.0) as f32 / lw;
            let fy0 = (key.ty * level.tile.1) as f32 / lh;
            let fx1 = ((key.tx + 1) * level.tile.0).min(level.width) as f32 / lw;
            let fy1 = ((key.ty + 1) * level.tile.1).min(level.height) as f32 / lh;
            // Where that share of the file lands once turned.
            let a = crate::view::turn_point(view.turn, [fx0, fy0], (1.0, 1.0));
            let b = crate::view::turn_point(view.turn, [fx1, fy1], (1.0, 1.0));
            let (fx0, fx1) = (a[0].min(b[0]), a[0].max(b[0]));
            let (fy0, fy1) = (a[1].min(b[1]), a[1].max(b[1]));
            let (w, h) = (image_rect[2] - image_rect[0], image_rect[3] - image_rect[1]);
            [
                image_rect[0] + fx0 * w,
                image_rect[3] - fy1 * h,
                image_rect[0] + fx1 * w,
                image_rect[3] - fy0 * h,
            ]
        };

        // Tiles that are in; for the rest, stand-ins already on the card. The
        // finer tiles under a missing one first — just zoomed out, they are
        // all still there and sharper than anything — and where those are not
        // all in, the nearest coarser tile beneath them, drawn whole, once,
        // however many tiles it stands in for.
        let mut fine = Vec::new();
        let mut finer = std::collections::BTreeSet::new();
        let mut coarse = std::collections::BTreeSet::new();
        for key in &plan.visible {
            if gpu.has_tile(key) {
                fine.push(*key);
                continue;
            }
            let under: Vec<_> = crate::tiles::finer(&t.pyramid, *key).into_iter().filter(|k| gpu.has_tile(k)).collect();
            let covered = !under.is_empty() && under.len() == crate::tiles::finer(&t.pyramid, *key).len();
            finer.extend(under);
            if !covered
                && let Some((stand_in, _)) =
                    crate::tiles::fallback(&t.pyramid, *key).into_iter().find(|(k, _)| gpu.has_tile(k))
            {
                coarse.insert(stand_in);
            }
        }
        t.drawing = Some((plan.level, fine.len(), plan.visible.len()));
        // Spared by eviction: the screen, its stand-ins, and what was read
        // ahead — or a pan's worth of tiles could be read only to be dropped.
        t.in_use = plan.visible.iter().chain(&plan.ahead).chain(&coarse).chain(&finer).copied().collect();
        for key in &t.in_use {
            gpu.touch_tile(key);
        }

        // Coarsest first — the highest level index is the smallest image — then
        // the finer stand-ins, then the tiles that belong.
        coarse
            .iter()
            .rev()
            .chain(&finer)
            .chain(&fine)
            .map(|k| (*k, place(k), crate::gpu::FULL_UV))
            .collect()
    }

    /// Into fullscreen or out of it — the `f` key, and a window raised from
    /// another's strip while that one was fullscreen.
    fn toggle_fullscreen(&mut self) {
        self.fullscreen = !self.fullscreen;
        if let Some(w) = &self.window {
            w.set_fullscreen(self.fullscreen.then(|| Fullscreen::Borderless(None)));
            // A window born fullscreen has no windowed size to
            // go back to, so the desktop restores it to the
            // whole screen — where it counts as maximized and
            // refuses to be resized. Give it a real size.
            if !self.fullscreen {
                w.set_maximized(false);
                if let Some(m) = w.current_monitor() {
                    let (sw, sh) = (m.size().width, m.size().height);
                    let source = self
                        .shown
                        .as_ref()
                        .map(|s| self.view.oriented(s.source))
                        .map(|s| (s.0 as u32, s.1 as u32))
                        .unwrap_or((sw, sh));
                    // Under the 80% at which the desktop
                    // maximizes a window on its own.
                    let cap = ((sw as f32 * 0.75) as u32, (sh as f32 * 0.75) as u32);
                    let size = window_size_for(source, cap);
                    let _ = w.request_inner_size(winit::dpi::PhysicalSize::new(
                        size.0, size.1,
                    ));
                }
            }
        }
        // Back to the arrow, and told to the window: setting only the record
        // of it left a resize arrow from the old window edge on screen in
        // fullscreen, and since the record already said "arrow", nothing ever
        // put it right.
        self.cursor_icon = CursorIcon::Default;
        if let Some(w) = &self.window {
            w.set_cursor(CursorIcon::Default);
        }
        // Going fullscreen is a request to see it bigger, so
        // it refits even if the view had been zoomed. The
        // resize that follows does the arithmetic, once the
        // new size is actually known.
        self.view.fitting = true;
        self.peer_list_open = false;
        self.build_peer_strip();
        self.redraw();
    }

    /// The open windows, as the list the filename badge drops down: one row
    /// each, folder and file, this one highlighted, in the badge's own type.
    /// Nothing when this is the only window.
    fn build_peer_strip(&mut self) {
        let peers = self.presence.as_ref().map(|p| (p.peers(), p.own_name().to_string()));
        let had = self.peer_list.is_some();
        match peers.filter(|(p, _)| p.len() > 1) {
            None => {
                self.peer_list = None;
                self.peer_list_open = false;
            }
            Some((peers, own)) => {
                let width = (self.win_size().0 * 0.9) as u32;
                let rows: Vec<(Overlay, String)> = peers
                    .iter()
                    .map(|p| {
                        let label = match (p.folder.is_empty(), p.file.is_empty()) {
                            (false, false) => format!("{}  /  {}", p.folder, p.file),
                            (true, _) => p.file.clone(),
                            (_, true) => p.folder.clone(),
                        };
                        let row = if p.name == own { osd::Row::good(label, "") } else { osd::Row::plain(label) };
                        (osd::render_within(&[row.selected(p.name == own)], osd::SCALE_PANEL, width), p.name.clone())
                    })
                    .collect();
                let mut spans = Vec::new();
                let mut y = 0;
                for (overlay, name) in &rows {
                    spans.push((y, y + overlay.height as i32, name.clone()));
                    y += overlay.height as i32;
                }
                let parts: Vec<&Overlay> = rows.iter().map(|(o, _)| o).collect();
                self.peer_list = Some((osd::vstack(&parts), spans));
            }
        }
        // The badge gains or loses its marker when that changes.
        if had != self.peer_list.is_some() {
            self.build_filename_badge();
        }
    }

    /// Where the list drops to: right-aligned under the filename badge.
    fn peer_list_at(&self) -> Option<(i32, i32)> {
        let (bx, by, bw, bh) = self.filename_at?;
        let (list, _) = self.peer_list.as_ref()?;
        Some(((bx + bw - list.width as i32).max(0), by + bh + 4))
    }

    /// What a press at `at` lands on, as far as the window switcher goes.
    fn peer_hit(&self, at: (f32, f32)) -> PeerHit {
        if self.peer_list.is_none() {
            return PeerHit::Nothing;
        }
        let (x, y) = (at.0 as i32, at.1 as i32);
        if let Some((bx, by, bw, bh)) = self.filename_at
            && x >= bx && x < bx + bw && y >= by && y < by + bh
        {
            return PeerHit::Button;
        }
        if self.peer_list_open
            && let (Some((list, rows)), Some((lx, ly))) = (&self.peer_list, self.peer_list_at())
        {
            if x >= lx && x < lx + list.width as i32 {
                if let Some((_, _, name)) = rows.iter().find(|(a, b, _)| y - ly >= *a && y - ly < *b) {
                    return PeerHit::Window(name.clone());
                }
            }
            // Anywhere else while the list is down closes it.
            return PeerHit::Away;
        }
        PeerHit::Nothing
    }

    /// Ctrl+Tab: the next open window, or the one before with Shift,
    /// fullscreen if this one is.
    fn step_window(&mut self, back: bool) {
        let Some(presence) = self.presence.as_ref() else { return };
        let peers = presence.peers();
        let Some(here) = peers.iter().position(|p| p.name == presence.own_name()) else { return };
        if peers.len() < 2 {
            return;
        }
        let next = if back { (here + peers.len() - 1) % peers.len() } else { (here + 1) % peers.len() };
        presence.activate(&peers[next].name, self.fullscreen);
    }

    /// Whether a drag on the picture should move the window rather than pan.
    fn drags_window(&self) -> bool {
        // Windowed, a plain drag anywhere moves the window: picture, film or
        // empty space alike. Panning is shift + drag. Fullscreen has no window
        // to move, so there a plain drag pans.
        !(self.fullscreen
            || self.shift_held
            || self.mosaic.is_some()
            || self.browser.is_some()
            || self.press_at.is_none())
            && (self.video.is_some() || self.shown.is_some())
    }

    /// Ctrl+C: the picture on screen — or the one selected in the grid — on
    /// the clipboard, at full resolution.
    ///
    /// Served by a `show --copy` process of its own: an X11 clipboard holds no
    /// data, only the name of a program to ask, so the picture has to outlive
    /// this window to still paste after it is closed.
    fn copy_to_clipboard(&mut self) {
        let path = match &self.mosaic {
            Some(mosaic) => self.playlist.at(mosaic.selected()),
            None => self.playlist.current(),
        }
        .map(|p| p.to_path_buf());
        let Some(path) = path else { return };
        // Nothing to copy from a film: quietly nothing.
        if crate::video::is_video(&path) {
            return;
        }
        let Ok(exe) = std::env::current_exe() else { return };
        match std::process::Command::new(exe).arg("--copy").arg(&path).spawn() {
            Ok(mut child) => {
                // Reaped when it ends — when something else is copied.
                std::thread::spawn(move || child.wait());
                self.notify("IMAGE COPIED");
            }
            Err(e) => self.notify(&format!("NOT COPIED: {e}")),
        }
        self.redraw();
    }

    /// The folder the list came from, and whether its subfolders are in it:
    /// what to watch, and what to rebuild the list from when it changes.
    fn list_source(&self) -> Option<(PathBuf, bool)> {
        if let Some(root) = &self.recursive_root {
            return Some((root.clone(), true));
        }
        if self.list_named {
            return None;
        }
        let current = self.playlist.current()?;
        let dir = match current.parent() {
            Some(d) if d.as_os_str().is_empty() => PathBuf::from("."),
            Some(d) => d.to_path_buf(),
            None => return None,
        };
        Some((dir, false))
    }

    /// Start, move or stop watching, to follow where the list comes from.
    fn update_watch(&mut self) {
        let want = self.list_source();
        if want == self.watching {
            return;
        }
        self.folder_watch = want.as_ref().and_then(|(root, recursive)| {
            let proxy = std::sync::Mutex::new(self.proxy.clone());
            crate::watch::Watcher::start(root, *recursive, move || {
                if let Ok(p) = proxy.lock() {
                    let _ = p.send_event(UserEvent::FolderChanged);
                }
            })
        });
        self.watching = want;
    }

    /// Rebuild the list from its folder after images arrived or left, keeping
    /// what is on screen where it can: the same picture, the grid scrolled
    /// where it was, and every thumbnail already loaded.
    fn rescan_folder(&mut self) {
        let Some((root, recursive)) = self.watching.clone() else { return };
        let mut list = if recursive {
            crate::playlist::Playlist::from_dir_recursive(&root)
        } else {
            crate::playlist::Playlist::from_dir(&root)
        };
        let in_grid = self.mosaic.is_some();
        list.sort(match self.config.sort {
            crate::playlist::Sort::NthPerFolder(_) if !in_grid => crate::playlist::Sort::Folder,
            by => by,
        });
        let old: Vec<PathBuf> = (0..self.playlist.len()).filter_map(|i| self.playlist.at(i).map(|p| p.to_path_buf())).collect();
        let new: Vec<PathBuf> = (0..list.len()).filter_map(|i| list.at(i).map(|p| p.to_path_buf())).collect();
        if old == new {
            return;
        }
        // The folder emptied: leave what is on screen; there is nothing to go to.
        if new.is_empty() {
            return;
        }

        let current = self.playlist.current().map(|p| p.to_path_buf());
        let selected = self.mosaic.as_ref().and_then(|m| self.playlist.at(m.selected())).map(|p| p.to_path_buf());
        if let Some(at) = current.as_deref().and_then(|p| list.position_near(p)) {
            list.go_to(at);
        }
        let current_gone = list.current() != current.as_deref();
        self.playlist = list;

        if self.mosaic.is_some() {
            let index: std::collections::HashMap<&std::path::Path, usize> =
                new.iter().enumerate().map(|(i, p)| (p.as_path(), i)).collect();
            let selected = selected
                .as_deref()
                .and_then(|p| index.get(p).copied().or_else(|| self.playlist.position_near(p)))
                .unwrap_or(0);
            let breaks = self.new_mosaic(0).breaks().to_vec();
            // Cells still decoding were numbered for the old list.
            self.generation = self.generation.wrapping_add(1);
            if let Some(mosaic) = self.mosaic.as_mut() {
                mosaic.remap(|i| old.get(i).and_then(|p| index.get(p.as_path()).copied()), selected);
                mosaic.set_breaks(breaks);
            }
            self.current_is_stale |= current_gone;
            self.request_thumbs();
        } else if current_gone {
            // The picture on screen was deleted: on to its neighbour.
            self.start_current();
        }
        self.build_filename_badge();
        self.redraw();
    }

    /// Tell the tray menu what this window is showing: its folder, and file.
    fn describe_to_tray(&self) {
        let (Some(presence), Some(path)) = (&self.presence, self.playlist.current()) else { return };
        let name = |p: Option<&std::path::Path>| {
            p.and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
        };
        presence.describe(&name(path.parent()), &name(Some(path)));
    }

    /// Look for a map placement for the image on screen, once per image, on a
    /// thread: a GeoTIFF's tags are only a directory read, but that directory
    /// may be on a slow drive.
    fn start_geo_check(&mut self) {
        let Some(path) = self.playlist.current().map(|p| p.to_path_buf()) else { return };
        if self.geo_for.as_ref() == Some(&path) {
            return;
        }
        self.geo_for = Some(path.clone());
        self.geo_badge = None;
        let proxy = self.proxy.clone();
        std::thread::Builder::new()
            .name("geo".into())
            .spawn(move || {
                background_priority();
                let sources = crate::geo::detect(&path);
                let _ = proxy.send_event(UserEvent::Geo(path, sources));
            })
            .expect("spawning the geo check");
    }

    /// List the opened file's folder on a thread, once, after it is showing.
    fn start_folder_scan(&mut self) {
        let Some(start) = self.folder_to_scan.take() else { return };
        let (proxy, sort) = (self.proxy.clone(), self.config.sort);
        std::thread::Builder::new()
            .name("folder".into())
            .spawn(move || {
                let list = Self::scan_folder(&start, sort);
                let _ = proxy.send_event(UserEvent::Folder(Box::new(list)));
            })
            .expect("spawning the folder scan");
    }

    /// The folder around `start`, positioned on it and sorted for the viewer.
    fn scan_folder(start: &std::path::Path, sort: crate::playlist::Sort) -> Playlist {
        let _s = timing::span("io", "folder_scan");
        let mut list = Playlist::from_file(start);
        // One per folder is a grid view; opening a single picture should not
        // leave only that picture to walk through.
        list.sort(match sort {
            crate::playlist::Sort::NthPerFolder(_) => crate::playlist::Sort::Folder,
            by => by,
        });
        list
    }

    /// Take the listed folder in place of the lone opening file — unless
    /// something has replaced the list in the meantime, which wins.
    fn adopt_folder(&mut self, list: Playlist) {
        let untouched = self.playlist.len() == 1
            && self.playlist.current().and_then(|p| p.file_name())
                == list.current().and_then(|p| p.file_name());
        if !untouched || list.len() <= 1 {
            return;
        }
        self.playlist = list;
        if self.mosaic.is_some() {
            self.generation = self.generation.wrapping_add(1);
            self.mosaic = Some(self.new_mosaic(self.playlist.position()));
            self.request_thumbs();
        }
        // What was built for a folder of one now has a place to say.
        self.build_filename_badge();
        if let Some(w) = &self.window {
            let title = w.title();
            if !title.is_empty() {
                w.set_title(&format!(
                    "{title}  [{}/{}]",
                    self.playlist.position() + 1,
                    self.playlist.len()
                ));
            }
        }
        if self.thumb_workers == 0 {
            self.refill_thumb_queue();
        }
        self.redraw();
    }

    /// Have the grid include every folder beneath this one, or stop.
    ///
    /// The folder is the selected image's when turning on, and the same one
    /// again when turning off — not the subfolder the selection has wandered
    /// into, which would make the toggle a way of changing folder.
    ///
    /// Works in the grid and over a single picture alike; in the viewer it
    /// changes what next and previous walk through.
    fn toggle_recursive(&mut self) {
        let in_grid = self.mosaic.is_some();
        let selected = match self.mosaic.as_ref() {
            Some(mosaic) => self.playlist.at(mosaic.selected()),
            None => self.playlist.current(),
        }
        .map(|p| p.to_path_buf());
        // A root that does not hold what is selected is left over from a list
        // since replaced — another folder opened from the file panel, say. It
        // is not what `r` is being pressed about: taking it for "on" made the
        // press load that old folder instead.
        if let (Some(root), Some(at)) = (&self.recursive_root, &selected)
            && !at.starts_with(root)
        {
            self.recursive_root = None;
        }
        let turning_on = self.recursive_root.is_none();
        let root = self
            .recursive_root
            .clone()
            .or_else(|| selected.as_ref().and_then(|p| p.parent().map(|d| d.to_path_buf())))
            .unwrap_or_else(crate::playlist::Playlist::default_start_dir);

        let mut list = if turning_on {
            crate::playlist::Playlist::from_dir_recursive(&root)
        } else {
            crate::playlist::Playlist::from_dir(&root)
        };
        if list.is_empty() {
            self.notify("NO IMAGES HERE");
            self.redraw();
            return;
        }
        // Subfolders come in folder order: this folder's images first, then
        // each subfolder's, as the tree has them. Any other order — above all
        // a one-per-folder filter left on from before — could drop the picture
        // `r` was pressed on and fill the grid with other folders' images.
        if turning_on && self.config.sort != crate::playlist::Sort::Folder {
            self.config.sort = crate::playlist::Sort::Folder;
            self.save_settings();
        }
        // One per folder is a grid view; in the viewer it would leave nothing
        // to step through, so the viewer walks in folder order instead.
        list.sort(match self.config.sort {
            crate::playlist::Sort::NthPerFolder(_) if !in_grid => crate::playlist::Sort::Folder,
            by => by,
        });
        let kept = selected.as_deref().is_some_and(|p| list.select(p));

        self.playlist = list;
        self.list_named = false;
        self.recursive_root = turning_on.then_some(root);
        // The grid set aside was a picture of the old list; the list has been
        // changed on purpose, so it no longer stands to be gone back to.
        self.grid_behind = None;
        if in_grid {
            // The image behind the grid is the one the selection was on; if
            // the new list does not have it, whatever is chosen next has to
            // load.
            self.current_is_stale |= !kept;
            // Cells still decoding were numbered for the old list.
            self.generation = self.generation.wrapping_add(1);
            self.mosaic = Some(self.new_mosaic(self.playlist.position()));
        } else if !kept {
            // Turned off while on a picture from a subfolder: it is not in
            // the folder's own list, so show that list's first instead.
            self.start_current();
        }
        self.build_grid_badge();
        let count = self.playlist.len();
        // Say so when a filter is on: with only the nth image of each folder
        // shown, the picture that was selected can drop out of the list, and
        // the grid fills with other folders' images for no visible reason.
        let filter = match self.config.sort {
            crate::playlist::Sort::NthPerFolder(_) if in_grid => format!("  {}", self.config.sort.label()),
            _ => String::new(),
        };
        self.notify(&if turning_on {
            format!("SUBFOLDERS ON  {count} IMAGES{filter}")
        } else {
            format!("SUBFOLDERS OFF  {count} IMAGES{filter}")
        });
        self.request_thumbs();
        self.redraw();
    }

    /// A grid over the playlist, selecting `position`. Sorted by folder, each
    /// folder starts a row of its own.
    fn new_mosaic(&self, position: usize) -> Mosaic {
        let mut mosaic = Mosaic::new(position);
        // Each folder on a row of its own, in folder order.
        if self.config.sort == crate::playlist::Sort::Folder {
            let parent = |i: usize| self.playlist.at(i).and_then(|p| p.parent());
            let breaks = (1..self.playlist.len()).filter(|&i| parent(i) != parent(i - 1)).collect();
            mosaic.set_breaks(breaks);
        }
        mosaic
    }

    /// Label the grid cell under the pointer with its folder and name.
    /// Returns whether the label changed.
    fn update_hover(&mut self) -> bool {
        let win = self.win_size();
        let path = self.mosaic.as_ref().and_then(|m| {
            let count = m.count(self.playlist.len());
            let index = m.index_under(self.cursor, win, count)?;
            self.playlist.at(index).map(|p| p.to_path_buf())
        });
        if path.as_ref() == self.hover_label.as_ref().map(|(p, _)| p) {
            return false;
        }
        self.hover_label = path.map(|path| {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let text = match path.parent().and_then(|d| d.file_name()) {
                Some(folder) => format!("{}/{name}", folder.to_string_lossy()),
                None => name.into_owned(),
            };
            let overlay = osd::render_scaled(&[osd::Row::plain(text)], osd::SCALE_PANEL);
            (path, overlay)
        });
        true
    }

    /// Rebuild the grid's scroll marker when the window or the position moves.
    fn update_scroll_marker(&mut self) {
        let win = self.win_size();
        let fraction = self
            .mosaic
            .as_ref()
            .and_then(|m| m.scroll_fraction(win, m.count(self.playlist.len())));
        let Some(fraction) = fraction else {
            self.scroll_marker = None;
            return;
        };
        let width = (win.0 - Corner::MARGIN * 2.0).max(1.0) as u32;
        // Keyed by the pixel the mark lands on, so a scroll too small to move
        // it does not rebuild the strip.
        let key = (width, (fraction * width as f32) as u32);
        if self.scroll_marker.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        self.scroll_marker = Some((key, osd::scroll_marker(width, fraction, osd::SCALE_PANEL)));
    }

    /// Redraw the grid's corner badges after the sort or subfolders change.
    fn build_grid_badge(&mut self) {
        let sort = osd::sort_icon(self.config.sort, osd::SCALE_PANEL);
        self.grid_badge = if self.recursive_root.is_some() {
            let folders = osd::recursive_icon(osd::SCALE_PANEL);
            osd::hstack(&[&sort, &folders], 8)
        } else {
            sort
        };
    }

    /// Step the grid to the next sort order, keeping the same cell selected.
    fn cycle_sort(&mut self) {
        let Some(mosaic) = self.mosaic.as_ref() else { return };
        let selected = self.playlist.at(mosaic.selected()).map(|p| p.to_path_buf());
        let by = self.config.sort.next();
        self.config.sort = by;
        self.save_settings();

        // The playlist stays on the image behind the grid; the grid's own
        // selection is found again by path, so leaving still opens that one.
        let before = self.playlist.current().map(|p| p.to_path_buf());
        self.playlist.sort(by);
        // Filtering can set aside the image behind the grid; then its folder's
        // first stands in, and leaving the grid has to load that one.
        self.current_is_stale |= self.playlist.current() != before.as_deref();
        let position = selected.and_then(|p| self.playlist.position_near(&p)).unwrap_or(0);
        // Cells still decoding were numbered for the old order.
        self.generation = self.generation.wrapping_add(1);
        self.mosaic = Some(self.new_mosaic(position));
        self.build_grid_badge();
        self.notify(by.label());
        self.request_thumbs();
        self.redraw();
    }

    /// Kick off decodes for cells that are on screen and not yet loaded.
    fn request_thumbs(&mut self) {
        let (win, count) = (self.win_size(), self.playlist.len());
        let budget_bytes = self.config.memory_budget_bytes();
        let Some(mosaic) = self.mosaic.as_mut() else { return };
        let cell = mosaic.cell_size(win);
        let budget = budget_bytes;
        // Visible cells first and unconditionally: what is on screen is never
        // traded against the memory budget.
        let mut pending = mosaic.take_pending(win, count);
        mosaic.evict(win, count, budget);
        // Then the rest of the folder, a few at a time, while there is room.
        pending.extend(mosaic.take_prefetch(win, count, budget, PREFETCH_BATCH));

        for index in pending {
            let Some(path) = self.playlist.at(index).map(|p| p.to_path_buf()) else { continue };
            let proxy = self.proxy.clone();
            let (io, generation) = (self.args.io, self.generation);
            // Cell-resolution decoding is what makes a grid of 40 megapixel
            // photos affordable; rayon's pool bounds the concurrency for us.
            let target = crate::decode::Target { width: cell, height: cell };
            let cache = self.thumbs.clone();
            rayon::spawn(move || {
                if QUITTING.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                // A stored thumbnail, ours or the desktop's, is twenty times
                // faster than decoding even a small JPEG. The cache is held
                // only to look the file up; reading and decoding it, and the
                // desktop's, happen with it let go — a grid queues these by
                // the thousand, and holding it throughout queued them all
                // behind one another, and quitting behind them.
                let ours = (cell <= crate::thumbs::THUMB_SIZE)
                    .then(|| cache.as_ref().and_then(|c| c.lock().ok().and_then(|c| c.stored(&path))))
                    .flatten();
                let cached = ours
                    .and_then(|file| crate::thumbs::read(&file))
                    .or_else(|| {
                        (cell <= crate::thumbs::THUMB_SIZE)
                            .then(|| crate::thumbs::xdg::load(&path, cell))
                            .flatten()
                    });
                let result = match cached {
                    Some((rgba, w, h)) => Ok(loader::Loaded {
                        path: path.clone(),
                        file_size: 0,
                        decoded: crate::decode::Decoded::from_rgba(
                            w,
                            h,
                            crate::decode::Format::Jpeg,
                            rgba,
                            "thumbnail",
                        ),
                    }),
                    None => loader::load(&path, io, Some(target)),
                };
                let _ = proxy.send_event(UserEvent::Thumb(generation, index, Box::new(result)));
            });
        }
    }

    /// Open the detailed view: picture on the left, folder on the right.
    fn open_browser(&mut self) {
        let current = self.playlist.current().map(|p| p.to_path_buf());
        let dir = current
            .as_ref()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(crate::playlist::Playlist::default_start_dir);
        self.browser = Some(Browser::open(&dir, current.as_deref()));
        self.start_info_scan();
        self.refit_for_viewport();
        self.build_browser_overlay();
        self.redraw();
    }

    /// Read the size and tile levels of every image in the detailed view's
    /// folder, on a low-priority thread, reporting a batch at a time so the
    /// list fills in as it goes. Only headers are read, never pixels.
    fn start_info_scan(&mut self) {
        let Some(browser) = self.browser.as_ref() else { return };
        let dir = browser.dir().to_path_buf();
        if self.info_scan.as_ref() == Some(&dir) {
            return;
        }
        let todo: Vec<PathBuf> =
            browser.images().into_iter().filter(|p| !self.file_info.contains_key(p)).collect();
        if todo.is_empty() {
            return;
        }
        self.info_scan = Some(dir);
        let proxy = self.proxy.clone();
        std::thread::Builder::new()
            .name("file-info".into())
            .spawn(move || {
                background_priority();
                const EVERY: std::time::Duration = std::time::Duration::from_millis(100);
                let mut batch = Vec::new();
                let mut sent = std::time::Instant::now();
                for path in todo {
                    let info = crate::browser::FileInfo::probe(&path);
                    batch.push((path, info));
                    if sent.elapsed() >= EVERY {
                        let _ = proxy.send_event(UserEvent::FileInfo(std::mem::take(&mut batch), false));
                        sent = std::time::Instant::now();
                    }
                }
                let _ = proxy.send_event(UserEvent::FileInfo(batch, true));
            })
            .expect("spawning the file info scan");
    }

    fn close_browser(&mut self) {
        self.browser = None;
        self.browser_overlay = None;
        self.refit_for_viewport();
        self.redraw();
    }

    /// The picture's region changed, so a fitted image should re-fit to it.
    fn refit_for_viewport(&mut self) {
        if let Some(source) = self.shown.as_ref().map(|s| s.source) {
            if self.view.fitting {
                self.view.fit(source, self.win_size());
            }
            self.build_status_badge();
        }
    }

    /// Note that the selection has moved onto an image worth showing.
    ///
    /// Deliberately does not start the decode: holding an arrow key down a
    /// folder of forty megapixel photos would queue one decode per keystroke,
    /// all but the last of them thrown away. The load waits for the selection
    /// to stop moving.
    fn browser_selection_changed(&mut self) {
        self.selection_moved = std::time::Instant::now();
        self.build_browser_overlay();
        self.pending_preview = match self.browser.as_ref().and_then(|b| b.selected_entry()) {
            Some(entry) if entry.is_image() => {
                let path = entry.path().expect("an image has a path").to_path_buf();
                (self.playlist.current() != Some(&path)).then_some(path)
            }
            _ => None,
        };
        self.redraw();
    }

    /// Decode the settled selection, if there is one.
    fn tick_pending_preview(&mut self) {
        const SETTLE: std::time::Duration = std::time::Duration::from_millis(120);
        if self.pending_preview.is_none() || self.selection_moved.elapsed() < SETTLE {
            return;
        }
        let Some(path) = self.pending_preview.take() else { return };
        let Some(browser) = self.browser.as_ref() else { return };

        // Selecting an image also makes its folder the sequence the arrow keys
        // will walk once the browser closes.
        let images = browser.images();
        if let Some(at) = images.iter().position(|p| *p == path) {
            self.playlist = crate::playlist::Playlist::from_list(&images);
            self.list_named = false;
            // One folder's images: whatever subfolders `r` had added are gone.
            if self.recursive_root.take().is_some() {
                self.build_grid_badge();
            }
            self.playlist.go_to(at);
            self.start_current();
        }
    }

    /// Move into a folder, or show an image, per the selected row.
    fn browser_activate(&mut self) {
        let Some(browser) = self.browser.as_ref() else { return };
        match browser.activate() {
            Activate::Enter(dir) => {
                self.browser = Some(Browser::open(&dir, None));
                self.start_info_scan();
                self.build_browser_overlay();
                self.redraw();
            }
            Activate::Show(path) => {
                // Selection already put it on the left, so enter is the
                // commitment: go full screen with it.
                if self.playlist.current() == Some(&path) {
                    self.close_browser();
                } else {
                    self.pending_preview = Some(path);
                    self.selection_moved =
                        std::time::Instant::now() - std::time::Duration::from_secs(1);
                    self.tick_pending_preview();
                }
            }
            Activate::None => {}
        }
    }

    fn browser_up(&mut self) {
        let Some(browser) = self.browser.as_ref() else { return };
        let here = browser.dir().to_path_buf();
        if let Some(parent) = browser.parent() {
            // Land on the folder just left, so going up and down is reversible.
            self.browser = Some(Browser::open(&parent, Some(&here)));
            self.start_info_scan();
            self.build_browser_overlay();
            self.redraw();
        }
    }

    /// The panel: what the current image is, then where we are in the tree.
    fn build_browser_overlay(&mut self) {
        let Some(browser) = self.browser.as_ref() else {
            self.browser_overlay = None;
            return;
        };
        let Some(width) = self.browser_width() else { return };
        let height = self.window_size().1;

        // The info block describes the *selected* row, not the displayed
        // image: moving through a folder should tell you about what you are
        // pointing at, before you commit to opening it.
        let mut rows = Vec::new();
        match browser.selected_entry() {
            Some(entry) if entry.is_image() => {
                let path = entry.path().expect("an image always has a path");
                rows.push(osd::Row::good(entry.label(), ""));
                if let Some((w, h)) = crate::decode::probe_dimensions(path) {
                    rows.push(osd::Row::new("SIZE", format!("{w} x {h}"), osd::TEXT_COLOUR));
                }
                if let Ok(meta) = std::fs::metadata(path) {
                    rows.push(osd::Row::new("FILE", human_bytes(meta.len()), osd::TEXT_COLOUR));
                }
                if self.playlist.current() == Some(path) {
                    rows.push(osd::Row::new(
                        "ZOOM",
                        format!("{:.0}%", self.view.percent()),
                        osd::TEXT_COLOUR,
                    ));
                } else {
                    rows.push(osd::Row::dim("ENTER", "SHOW THIS IMAGE"));
                }
            }
            Some(entry) => {
                rows.push(osd::Row::good(entry.label(), ""));
                rows.push(osd::Row::dim("ENTER", "OPEN THIS FOLDER"));
            }
            None => rows.push(osd::Row::dim("EMPTY FOLDER", "")),
        }

        rows.push(osd::Row::plain(""));
        let here = browser.dir().file_name().map(|n| n.to_string_lossy().into_owned());
        rows.push(osd::Row::dim("FOLDER", here.unwrap_or_else(|| "/".into())));

        // Room for the list is whatever the info block left behind.
        let scale = self.browser_scale();
        let line = osd::line_height_at(scale) + (3.0 * scale).round() as usize;
        let used = rows.len() * line + (8.0 * scale).round() as usize;
        let visible = ((height as usize).saturating_sub(used) / line.max(1)).max(1);

        let window = browser.window(visible);
        // Size and tile levels on the left, filled in as the background scan
        // reports; folders get the same blank columns so every name lines up.
        for i in window {
            let entry = &browser.entries()[i];
            let info = entry.path().and_then(|p| self.file_info.get(p));
            let label = format!("{}{}", crate::browser::FileInfo::columns(info), entry.label());
            let row = if entry.is_image() { osd::Row::plain(label) } else { osd::Row::good(label, "") };
            rows.push(row.selected(i == browser.selected()));
        }

        self.browser_overlay =
            Some(osd::render_fixed(&rows, scale, (width as u32, height as u32)));
    }

    fn open_panel(&mut self) {
        self.panel = Some(Panel::open(&self.config));
        self.redraw();
    }

    /// What the settings say counts as a thing.
    fn thresholds(&self) -> crate::ml::vision::Thresholds {
        crate::ml::vision::Thresholds::new(
            self.config.min_confidence,
            self.config.duplicate_overlap,
        )
    }

    fn open_keys(&mut self) {
        self.keys = Some(Keys::open());
        self.redraw();
    }

    fn close_keys(&mut self) {
        self.keys = None;
        self.redraw();
    }

    /// Closing restores whichever overlay was underneath.
    fn close_panel(&mut self) {
        self.panel = None;
        self.redraw();
    }

    fn video_subtitle_tracks(&self, player: &crate::video::Player) -> Vec<crate::subtitles::Track> {
        player.subtitle_tracks.clone()
    }

    /// Read subtitle files sitting next to the video.
    ///
    /// Done here rather than in the decode thread because they are whole small
    /// files rather than a stream: reading one costs milliseconds and makes
    /// the track complete the instant it is selected.
    fn load_external_subtitles(&mut self, video: &std::path::Path) {
        const EXTERNAL_BASE: usize = 1 << 20;
        for (i, path) in crate::subtitles::find_beside(video).into_iter().enumerate() {
            match crate::subtitles::load_external(&path, EXTERNAL_BASE + i) {
                Ok((track, cues)) => {
                    eprintln!("subtitles {}: {} cues", path.display(), cues.len());
                    self.subtitles.add_track(track);
                    for cue in cues {
                        self.subtitles.push(cue);
                    }
                }
                // A subtitle file we cannot read is worth a line, not a
                // refusal to play the film.
                Err(e) => eprintln!("show: {}: {e:#}", path.display()),
            }
        }
    }

    /// Collect newly decoded captions and rebuild the one on screen if it
    /// changed. Called once per frame; does nothing on the vast majority.
    fn tick_subtitles(&mut self) {
        if let Some(player) = self.video.as_mut() {
            for cue in player.try_cues() {
                self.subtitles.push(cue);
            }
        }
        if self.subtitles.is_empty() {
            return;
        }

        let now = self.position;
        let win = self.win_size();
        let px = self.caption_size();
        let key = self
            .subtitles
            .at(now)
            .map(|c| (c.track, c.start, px.round() as u32));
        if key == self.caption_for {
            return;
        }
        self.caption_for = key;

        // A caption spanning the full width is hard to read back from; film
        // subtitling settles around two-thirds.
        let max_width = (win.0 * 0.86) as u32;

        self.caption = match self.subtitles.at(now).map(|c| c.payload.clone()) {
            Some(crate::subtitles::Payload::Lines(lines)) if !lines.is_empty() => {
                // Anything the face cannot draw would come out as an empty
                // box; the words around it are worth more than the box.
                let lines: Vec<String> = lines
                    .iter()
                    .map(|l| crate::typeface::strip_unsupported(l))
                    .filter(|l| !l.is_empty())
                    .collect();
                if lines.is_empty() {
                    None
                } else {
                    let wrapped = osd::wrap(&lines, px, max_width);
                    Some(osd::caption(&wrapped, px))
                }
            }
            Some(crate::subtitles::Payload::Images(images, canvas)) => {
                self.render_bitmap_caption(&images, canvas)
            }
            _ => None,
        };
        self.redraw();
    }

    /// Compose picture subtitles into one overlay.
    ///
    /// They are positioned in the canvas they were authored for, so the whole
    /// canvas is realised at the size it will be shown and the rectangles are
    /// placed inside it. Scaling each rectangle independently would lose the
    /// relationship between them, which is what puts two speakers' lines in
    /// the right places.
    fn render_bitmap_caption(
        &self,
        images: &[crate::subtitles::Image],
        canvas: (u32, u32),
    ) -> Option<Overlay> {
        if images.is_empty() || canvas.0 == 0 || canvas.1 == 0 {
            return None;
        }
        let win = self.win_size();
        let scale = (win.0 / canvas.0 as f32).min(win.1 / canvas.1 as f32).min(1.0);

        // Only the part of the canvas the captions actually occupy is kept, so
        // the overlay is the caption rather than a screen-sized transparency.
        let top = images.iter().map(|i| i.y).min()?;
        let bottom = images.iter().map(|i| i.y + i.height).max()?;
        let left = images.iter().map(|i| i.x).min()?;
        let right = images.iter().map(|i| i.x + i.width).max()?;

        let out_w = (((right - left) as f32) * scale).round().max(1.0) as usize;
        let out_h = (((bottom - top) as f32) * scale).round().max(1.0) as usize;
        let mut rgba = vec![0u8; out_w * out_h * 4];

        for image in images {
            for y in 0..out_h {
                for x in 0..out_w {
                    // Nearest neighbour: these are hard-edged palettised
                    // pictures, and smoothing them only blurs the edges.
                    let cx = left as f32 + x as f32 / scale;
                    let cy = top as f32 + y as f32 / scale;
                    if cx < image.x as f32 || cy < image.y as f32 {
                        continue;
                    }
                    let sx = (cx - image.x as f32) as usize;
                    let sy = (cy - image.y as f32) as usize;
                    if sx >= image.width as usize || sy >= image.height as usize {
                        continue;
                    }
                    let src = (sy * image.width as usize + sx) * 4;
                    if image.rgba[src + 3] == 0 {
                        continue;
                    }
                    let dst = (y * out_w + x) * 4;
                    rgba[dst..dst + 4].copy_from_slice(&image.rgba[src..src + 4]);
                }
            }
        }
        Some(Overlay::new(rgba, out_w as u32, out_h as u32))
    }

    /// Keep the window behind the play head filled.
    ///
    /// Playing forwards fills it for nothing — the frames were just shown. It
    /// is a jump that empties it, and after one the first press of `,` would
    /// otherwise pay to decode a whole group of pictures. So a second decoder
    /// covers the three seconds behind while the first covers the three ahead:
    /// the same work either way, split in two and done at once.
    fn tick_backfill(&mut self) {
        const WANT: std::time::Duration = crate::framecache::DEFAULT_SPAN;

        // Finish or abandon whatever is running first.
        if let Some(job) = self.backfill.as_mut() {
            // A jump while filling makes the run irrelevant before it lands.
            let stale = self.position.abs_diff(job.to()) > WANT * 2;
            if let Some(frames) = job.collect() {
                if !stale && !frames.is_empty() {
                    self.frame_cache.prefill(frames);
                }
                self.backfill = None;
            } else if stale {
                self.backfill = None;
            }
            return;
        }

        let Some(player) = self.video.as_ref() else { return };
        // Only worth doing where there is a gap to close, and only once the
        // picture has settled: starting one per frame during playback would be
        // a decoder per frame.
        const SLACK: std::time::Duration = std::time::Duration::from_millis(500);
        let (behind, ahead) = self.frame_cache.around(self.position);
        let path = player.path.clone();

        // Behind first: it is the direction that cannot be decoded on demand.
        if behind + SLACK < WANT && self.position >= SLACK {
            self.backfill = crate::framecache::BackFill::start(&path, self.position, WANT);
            return;
        }

        // Then ahead, but only while stopped. A paused film stops draining its
        // sound, which blocks the demuxer, which is why stepping forward runs
        // out at all — so the window in front is filled by a second decoder
        // that has no such problem. While playing there is nothing to do here:
        // the frames arrive by themselves.
        if self.paused
            && ahead + SLACK < WANT
            && let Some(end) = player.info.duration
            && self.position + SLACK < end
        {
            let to = (self.position + WANT).min(end);
            self.backfill = crate::framecache::BackFill::start(&path, to, WANT);
        }
    }

    /// Take in whatever the models have said, and draw it.
    ///
    /// One place, because the two of them share a picture, a grid and a bar:
    /// detection says what is in view, pointing says what one thing is, and a
    /// reader should not have to work out which of two functions owns the
    /// screen at any moment.
    fn tick_ml(&mut self) {
        // Nothing is looking and nothing has been pointed at: this runs on
        // every turn of the event loop, so it has to cost nothing when it has
        // nothing to do.
        if self.watcher.is_none() && self.pointer.is_none() && self.finder.is_none() {
            return;
        }
        const RECENT: std::time::Duration = std::time::Duration::from_millis(1500);
        const GIVE_UP: std::time::Duration = std::time::Duration::from_secs(3);

        let mut said: Vec<String> = Vec::new();
        let mut off = false;
        // Whether anything at all happened. This runs on every turn of the
        // event loop, and asking for a frame each time would keep a machine
        // showing a still picture permanently awake.
        let mut changed = false;

        // --- what came back from naming something ---
        if let Some(finder) = self.finder.as_mut() {
            if let Some(why) = finder.failed().map(str::to_string) {
                said.push(format!("FIND OFF: {}", first_line(&why).to_uppercase()));
                eprintln!("show: find stopped: {why}");
                self.finder = None;
                self.found.clear();
                self.found_of = None;
            } else {
                let answers = finder.collect();
                changed |= !answers.is_empty();
                for answer in answers {
                    // A phrase answered about a picture that is no longer on
                    // screen is not an answer about this one.
                    if answer.generation != self.generation {
                        continue;
                    }
                    let count = answer.found.len();
                    // Numbered here rather than by the tracker: nothing is
                    // being followed between frames, so the id exists only to
                    // give each thing a colour of its own.
                    self.found = answer.found;
                    for (i, item) in self.found.iter_mut().enumerate() {
                        item.id = i as u32 + 1;
                    }
                    self.found_of = Some((answer.generation, answer.at));
                    self.found_phrase = answer.phrase.clone();
                    self.tags_shown = None;
                    said.push(if count == 0 {
                        format!("NO {} HERE", answer.phrase.to_uppercase())
                    } else {
                        format!(
                            "{count} {} · {:.1}S",
                            answer.phrase.to_uppercase(),
                            answer.took.as_secs_f32()
                        )
                    });
                }
            }
        }

        // --- what came back from pointing at something ---
        if let Some(pointer) = self.pointer.as_mut() {
            if let Some(why) = pointer.failed().map(str::to_string) {
                said.push(format!("CUT OUT OFF: {}", first_line(&why).to_uppercase()));
                eprintln!("show: cut out stopped: {why}");
                self.pointer = None;
                self.forget_cut();
            } else {
                let answers = pointer.collect();
                changed |= !answers.is_empty();
                if !answers.is_empty() {
                    self.cut_asked = false;
                }
                for answer in answers {
                    // An answer to a question about a picture that has since
                    // been replaced is not wrong, it is simply about something
                    // else. It is dropped rather than drawn over the new one.
                    if answer.generation != self.generation {
                        continue;
                    }
                    let ms = answer.took.as_secs_f64() * 1000.0;

                    // Sharpened detections are a different thing entirely:
                    // they replace what the detector drew, and there is one
                    // per box rather than one at all.
                    if answer.refined {
                        self.sharp = answer.cuts;
                        self.sharp_of = Some((answer.generation, answer.at, answer.size));
                        self.sharp_asked = false;
                        continue;
                    }

                    match answer.cuts.into_iter().next() {
                        Some(cut) => {
                            // Said once, when it is first cut out. Following it
                            // through a film produces an answer every quarter
                            // second and announcing each would be a flicker of
                            // text over the picture.
                            if self.cut.is_none() {
                                said.push(format!(
                                    "CUT OUT · {:.0} MS{}",
                                    ms,
                                    if answer.encoded { "" } else { " (SAME PICTURE)" }
                                ));
                            }
                            // Where it had got to, which is where to point
                            // next. Without this the question would keep being
                            // asked at the spot the thing has walked away from.
                            if self.cut_track.is_some() {
                                if let Some(middle) =
                                    crate::ml::cutout::middle_of(&cut, answer.size)
                                {
                                    self.cut_track = Some(middle);
                                }
                            }
                            self.cut = Some(cut);
                            self.cut_of = Some((answer.generation, answer.at));
                            self.cut_size = Some(answer.size);
                        }
                        None => {
                            // Lost it: the thing left the frame, or turned into
                            // something the model will not separate. Saying so
                            // once is better than an outline that stays behind.
                            said.push("LOST WHAT WAS CUT OUT".to_string());
                            self.forget_cut();
                        }
                    }
                }
            }
        }

        // A cut-out belongs to the picture it was taken from, and a different
        // file is a different picture entirely.
        if let Some((generation, _)) = self.cut_of
            && generation != self.generation
        {
            self.forget_cut();
            changed = true;
        }

        // Following it: ask again about whatever is on screen now, one question
        // at a time. Each answer takes about a quarter of a second, so the
        // outline lags a moving subject by about that much — which is the price
        // of a model that looks at the whole picture, and is why the question
        // is asked at where the thing had got to rather than where it was
        // clicked.
        // Not flat out: three or four answers a second is as much as the eye
        // wants from an outline, and the gap is what keeps the decoder's cores
        // free between them.
        const BREATHE: std::time::Duration = std::time::Duration::from_millis(200);
        if let Some(where_) = self.cut_track
            && !self.cut_asked
            && self.cut_last.is_none_or(|t: std::time::Instant| t.elapsed() >= BREATHE)
            && self.video.is_some()
            && !self.paused
            && self.pointer.is_some()
            && self.cut_of.map(|(_, at)| at) != Some(self.position)
            && let Some(frame) = self.frame_cache.showing(self.position).map(|f| f.clone_ref())
        {
            if let Some(pointer) = self.pointer.as_ref() {
                pointer.ask(
                    crate::ml::pointer::Subject::Frame(frame),
                    self.generation,
                    self.position,
                    crate::ml::pointer::Prompt::Point(where_),
                );
                self.cut_asked = true;
            }
        }

        // --- what came back from looking at everything ---
        let mut lost = None;
        if let Some(watcher) = self.watcher.as_mut() {
            if let Some(why) = watcher.failed().map(str::to_string) {
                said.push(format!("DETECTION OFF: {}", first_line(&why).to_uppercase()));
                eprintln!("show: detection stopped: {why}");
                off = true;
            } else {
                let results = watcher.collect();
                changed |= !results.is_empty();
                for result in results {
                    self.seen.push(result);
                }
                self.seen.retire(self.position, self.generation);
            }
        }
        if off {
            self.stop_detection();
            for line in said {
                self.notify(&line);
            }
            return;
        }

        // A still is offered until something comes back about it: unlike a
        // film there is no next frame to try instead, so one dropped offer
        // would mean the picture was never looked at.
        if self.watcher.is_some()
            && self.video.is_none()
            && self.seen.at(self.position, RECENT).is_none()
            && let Some(path) = self.playlist.current().map(|p| p.to_path_buf())
        {
            let generation = self.generation;
            if let Some(watcher) = self.watcher.as_mut() {
                watcher.offer_picture(&path, generation);
            }
        }

        // While something is being followed it is the only thing drawn, and
        // the only thing named. Everything else is still being found — it is
        // simply not what was asked for.
        let latest = self.seen.at(self.position, RECENT);
        let held: Vec<&crate::ml::segment::Segmented> = match (self.following, latest) {
            (Some((id, ..)), Some(seen)) => seen.found.iter().filter(|s| s.id == id).collect(),
            (None, Some(seen)) => seen.found.iter().collect(),
            (_, None) => Vec::new(),
        };

        // Sharpening: the detector's boxes, sent to the model that draws a
        // proper outline. Only when nothing is being followed by hand — a
        // click is a direct request and takes the worker first — and only for
        // a picture still on screen.
        if self.config.sharpen
            && self.cut_track.is_none()
            && !self.cut_asked
            && !self.sharp_asked
            && self.pointer_ready()
            && let Some(seen) = latest
            && !seen.found.is_empty()
            && self.sharp_of.map(|(g, at, _)| (g, at)) != Some((self.generation, self.position))
            && let Some(subject) = self.subject_on_screen()
        {
            let boxes: Vec<crate::ml::vision::Found> =
                seen.found.iter().map(|s| s.found.clone()).collect();
            let size = seen.size;
            if let Some(pointer) = self.pointer.as_ref() {
                pointer.ask(
                    subject,
                    self.generation,
                    self.position,
                    crate::ml::pointer::Prompt::Boxes { found: boxes, size },
                );
                self.sharp_asked = true;
            }
        }

        // A thing goes behind something else and comes back; the tracker
        // survives that, and so should following. Only a real absence ends it.
        if let Some((id, class, last)) = self.following {
            if !held.is_empty() {
                self.following = Some((id, class, std::time::Instant::now()));
            } else if last.elapsed() > GIVE_UP {
                self.following = None;
                lost = Some(class);
                changed = true;
            }
        }

        let legend = match (self.following, self.cut.is_some()) {
            (Some((id, class, _)), _) => crate::ml::bar::Legend::Following { class, id },
            (None, true) => crate::ml::bar::Legend::CutOut {
                following: self.cut_track.is_some() && self.video.is_some(),
            },
            // Three states, not two: nothing yet, nothing there, and things.
            // The middle one used to read the same as the first, which made a
            // picture of a landscape look like a broken feature.
            (None, false) => match latest.map(|seen| crate::ml::watcher::tally(&seen.found)) {
                None => crate::ml::bar::Legend::Tags(Vec::new()),
                Some(tags) if tags.is_empty() => crate::ml::bar::Legend::Nothing,
                Some(tags) => crate::ml::bar::Legend::Tags(tags),
            },
        };

        // Everything is placed against the size of the picture the model was
        // actually shown, which is not always the size of the picture on
        // screen: a large photograph is looked at reduced.
        let looked_at = latest.map(|seen| seen.size);
        let source = looked_at
            .or(self.cut_size)
            .unwrap_or_else(|| {
                let source = self.shown.as_ref().map(|s| s.source).unwrap_or((1.0, 1.0));
                (source.0 as u32, source.1 as u32)
            });
        // What a phrase found belongs on the grid too, and it is already on
        // the grid's own geometry — reduced there when it came back — so it
        // needs no mapping, only stamping.
        let named: bool = !self.found.is_empty()
            && self.found_of.is_some_and(|(g, _)| g == self.generation);
        if !named {
            // A picture the phrase was not about must not keep its outlines.
            if !self.found.is_empty() && self.found_of.is_some_and(|(g, _)| g != self.generation) {
                self.found.clear();
                self.found_of = None;
            }
        }

        if !held.is_empty() || self.cut.is_some() || named {
            let wanted = crate::ml::labels::Grid::for_picture(source).size;
            let grid = self
                .labels
                .get_or_insert_with(|| crate::ml::labels::Grid::for_picture(source));
            if grid.size != wanted {
                *grid = crate::ml::labels::Grid::for_picture(source);
            }
            grid.clear();

            // Sharpened outlines stand in for the detector's own, when they
            // are about this very picture: they are the same things, said more
            // precisely, and the colours come from the same classes.
            let sharp = self
                .sharp_of
                .filter(|(g, at, _)| *g == self.generation && *at == self.position)
                .map(|(_, _, size)| size);
            if let Some(size) = sharp.filter(|_| !self.sharp.is_empty()) {
                let geometry = crate::ml::cutout::geometry(size);
                let following = self.following.map(|(_, class, _)| class);
                let chosen: Vec<&crate::ml::segment::Segmented> = self
                    .sharp
                    .iter()
                    .filter(|s| following.is_none_or(|class| s.found.class == class))
                    .collect();
                grid.add_outlines(chosen.into_iter(), &geometry, size);
            } else if held.iter().any(|s| s.has_outline()) {
                let input = crate::ml::prep::Input::for_source(source);
                grid.add_outlines(held.iter().copied(), &input, source);
            } else if !held.is_empty() {
                grid.add_boxes(held.iter().copied(), source);
            }

            // The cut-out came from a different model, which saw the picture
            // in a square of its own and possibly at a different size, so it
            // is placed by its own geometry against its own size.
            if let Some((cut, size)) = self.cut.as_ref().zip(self.cut_size) {
                let geometry = crate::ml::cutout::geometry(size);
                grid.add_outlines(std::iter::once(cut), &geometry, size);
            }

            if named {
                grid.add_cells(self.found.iter());

            }
        } else if let Some(grid) = self.labels.as_mut() {
            grid.clear();
        }

        // Said after the results have been let go of, because saying it
        // touches the whole of `self`.
        if let Some(class) = lost {
            let name = crate::ml::vision::CLASSES.get(class).copied().unwrap_or("thing");
            said.push(format!("LOST THE {}", name.to_uppercase()));
        }
        let quiet = said.is_empty();
        for line in said {
            self.notify(&line);
        }
        // The readout carries the last inference time, which changes with
        // every look; rebuilding it only when a result lands keeps that honest
        // without putting work on every frame.
        if changed && self.config.show_debug && self.config.show_zoom {
            self.build_status_badge();
        }

        // The bar only exists while something is looking; a cut-out on its own
        // does not steal a strip of the picture for one word.
        let settled = self.tags_shown.as_ref() == Some(&legend) && quiet;
        if self.watcher.is_none() || settled {
            // Redrawn only when something actually changed: this runs on every
            // turn of the loop, and asking for a frame each time would keep the
            // machine awake for a picture that is not moving.
            if changed || !quiet {
                self.redraw();
            }
            return;
        }
        self.tags_shown = Some(legend.clone());
        if std::env::var_os("SHOW_TRACE_ML").is_some() {
            let what = match &legend {
                crate::ml::bar::Legend::Tags(tags) => format!("tags {tags:?}"),
                crate::ml::bar::Legend::Following { class, id } => format!("following {class} #{id}"),
                crate::ml::bar::Legend::CutOut { following } => {
                    format!("cut out, following {following}")
                }
                crate::ml::bar::Legend::Nothing => "nothing it knows".to_string(),
            };
            eprintln!("T bar now says: {what}");
        }

        let win = self.window_size();
        self.tag_bar = Some(crate::ml::bar::draw(&legend, win.0 as u32));
        // The picture now has a different amount of room.
        self.refit_for_viewport();
        self.redraw();
    }

    /// Whether the cut-out worker is up and able to answer.
    fn pointer_ready(&self) -> bool {
        self.pointer.as_ref().is_some_and(|p| p.failed().is_none())
    }

    /// What the models should look at, matching what is on screen.
    fn subject_on_screen(&self) -> Option<crate::ml::pointer::Subject> {
        if self.video.is_some() {
            let frame = self.frame_cache.showing(self.position)?;
            return Some(crate::ml::pointer::Subject::Frame(frame.clone_ref()));
        }
        let path = self.playlist.current()?;
        Some(crate::ml::pointer::Subject::Picture(path.to_path_buf()))
    }

    /// Ask what is at a point, and say that the asking has started.
    fn cut_out_at(&mut self, at: (f32, f32)) {
        let Some(shown) = self.shown.as_ref() else { return };
        let source = shown.source;
        let Some(point) = self.view.image_point_in(at, source, self.image_viewport()) else {
            return;
        };

        // Clicking the same cut-out again puts it away, the way clicking a
        // followed thing again lets it go.
        if self.cut.as_ref().is_some_and(|cut| {
            let b = cut.found.bounds;
            point[0] >= b.0 && point[0] <= b.2 && point[1] >= b.1 && point[1] <= b.3
        }) {
            self.forget_cut();
            self.notify("CUT OUT CLEARED");
            self.tags_shown = None;
            self.tick_ml();
            return;
        }

        if !crate::ml::store::installed(crate::ml::catalogue::Capability::PointSegment) {
            self.notify("CUT OUT IS NOT INSTALLED — PRESS F8");
            return;
        }

        // A fraction rather than a pixel: the worker may decode a photograph
        // at a different size than the screen is showing it at, and a fraction
        // means the same thing at either.
        let where_ = (point[0] / source.0.max(1.0), point[1] / source.1.max(1.0));
        // Where to point next time. A film does not stop to be looked at: each
        // answer says where the thing had got to, and the next question is
        // asked there.
        self.cut_track = Some(where_);
        let subject = if self.video.is_some() {
            match self.frame_cache.showing(self.position) {
                Some(frame) => crate::ml::pointer::Subject::Frame(frame.clone_ref()),
                None => return,
            }
        } else {
            match self.playlist.current() {
                Some(path) => crate::ml::pointer::Subject::Picture(path.to_path_buf()),
                None => return,
            }
        };

        if self.pointer.is_none() {
            self.pointer = Some(crate::ml::pointer::Pointer::start());
        }
        let Some(pointer) = self.pointer.as_ref() else { return };
        pointer.ask(
            subject,
            self.generation,
            self.position,
            crate::ml::pointer::Prompt::Point(where_),
        );
        self.cut_asked = true;
        self.notify("CUTTING OUT...");
        self.redraw();
    }

    /// Send the typed phrase to the model.
    ///
    /// The picture is handed over as a subject rather than as pixels: a film
    /// gives the frame on screen, a photograph gives its path for the worker
    /// to decode at whatever size it likes. Both are what the cut-out already
    /// does, and for the same reasons.
    fn ask_for(&mut self, phrase: &str) {
        let phrase = phrase.trim();
        if phrase.is_empty() {
            self.forget_found();
            return;
        }
        if !crate::ml::store::installed(crate::ml::catalogue::Capability::TextSegment) {
            self.notify("SEGMENT WHAT I TYPE IS NOT INSTALLED — PRESS F8");
            return;
        }
        let subject = if self.video.is_some() {
            match self.frame_cache.showing(self.position) {
                Some(frame) => crate::ml::subject::Subject::Frame(frame.clone_ref()),
                None => return,
            }
        } else {
            match self.playlist.current() {
                Some(path) => crate::ml::subject::Subject::Picture(path.to_path_buf()),
                None => return,
            }
        };

        if self.finder.is_none() {
            self.finder = Some(crate::ml::finder::Finder::start());
        }
        let Some(finder) = self.finder.as_ref() else { return };
        finder.ask(subject, self.generation, self.position, phrase);
        self.found_phrase = phrase.to_string();
        // It takes seconds, not milliseconds, and a screen that does not
        // change for six seconds is a screen that looks broken.
        self.notify(&format!("LOOKING FOR {}...", phrase.to_uppercase()));
        self.redraw();
    }

    /// Put away whatever was found.
    ///
    /// The grid is emptied here rather than left to the next tick. Clearing it
    /// there does happen, but nothing about it asks for a frame, so the last
    /// mask uploaded stays on the card and the outlines sit on screen after
    /// being told to go — which looks exactly like a key that did nothing.
    fn forget_found(&mut self) {
        self.found.clear();
        self.found_of = None;
        self.found_phrase.clear();
        self.tags_shown = None;
        if let Some(grid) = self.labels.as_mut() {
            grid.clear();
        }
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.clear_mask();
        }
        self.redraw();
    }

    /// Put away whatever was cut out.
    fn forget_cut(&mut self) {
        self.cut = None;
        self.cut_of = None;
        self.cut_size = None;
        self.cut_track = None;
    }

    /// Follow whatever was clicked, if anything was.
    ///
    /// Returns whether the click was spent on a thing — a click on the film
    /// itself has to fall through and still pause it.
    fn follow_at(&mut self, at: (f32, f32)) -> bool {
        let Some(shown) = self.shown.as_ref() else { return false };
        let source = shown.source;
        let Some(point) = self.view.image_point_in(at, source, self.image_viewport()) else {
            return false;
        };

        const RECENT: std::time::Duration = std::time::Duration::from_millis(1500);
        let Some(seen) = self.seen.at(self.position, RECENT) else { return false };

        // Into the pixels the model was shown, which is not always the size on
        // screen: a photograph is looked at reduced.
        let where_ = (
            point[0] * seen.size.0 as f32 / source.0.max(1.0),
            point[1] * seen.size.1 as f32 / source.1.max(1.0),
        );
        let hit = crate::ml::segment::pick(&seen.found, where_).map(|s| (s.id, s.found.class));
        let Some((id, class)) = hit else { return false };

        // Clicking the thing already being followed lets it go, so the way out
        // is the way in rather than a key to remember.
        if self.following.map(|(had, ..)| had) == Some(id) {
            self.following = None;
            self.notify("FOLLOWING NOTHING");
        } else {
            self.following = Some((id, class, std::time::Instant::now()));
            let name = crate::ml::vision::CLASSES.get(class).copied().unwrap_or("thing");
            self.notify(&format!("FOLLOWING {} #{id}", name.to_uppercase()));
        }
        // The bar and the outlines both change, and neither waits for the next
        // inference to say so.
        self.tags_shown = None;
        self.tick_ml();
        true
    }

    fn start_detection(&mut self) {
        if self.watcher.is_some() {
            return;
        }
        let watcher = crate::ml::watcher::Watcher::start();
        watcher.set_rate(self.config.look_hz);
        watcher.set_thresholds(self.thresholds());
        self.watcher = Some(watcher);
        self.build_status_badge();
        self.tags_shown = None;
        self.tag_bar = Some(crate::ml::bar::render(&[], self.window_size().0 as u32));
        self.following = None;
        self.refit_for_viewport();
        self.notify("LOOKING FOR OBJECTS");
        self.redraw();
    }

    fn stop_detection(&mut self) {
        self.watcher = None;
        self.build_status_badge();
        self.following = None;
        self.seen.clear();
        // A cut-out is not detection and does not stop with it: it was asked
        // for separately and stays until it is dismissed.
        if self.cut.is_none() {
            self.labels = None;
        }
        self.tag_bar = None;
        self.tags_shown = None;
        self.refit_for_viewport();
        self.redraw();
    }

    /// F9: depth over the picture, then depth instead of it, then off.
    fn cycle_depth(&mut self) {
        if self.depth.is_none() {
            self.depth = Some(crate::ml::depth::Depth::start());
            self.depth_opacity = DEPTH_BLEND;
            self.depth_map = None;
            // Nothing new may be decoded while paused, so the frame already
            // up is offered now rather than waiting for one.
            if let Some(frame) = self.frame_cache.showing(self.position).map(|f| f.clone_ref()) {
                if let Some(depth) = self.depth.as_ref() {
                    depth.offer(&frame, self.generation);
                }
            }
            self.notify("DEPTH");
        } else if self.depth_opacity < 1.0 {
            self.depth_opacity = 1.0;
            self.notify("DEPTH ONLY");
        } else {
            self.stop_depth();
            self.notify("DEPTH OFF");
        }
        self.redraw();
    }

    fn stop_depth(&mut self) {
        self.depth = None;
        self.depth_map = None;
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.clear_depth();
        }
    }

    /// Take the newest map, and keep a still being looked at until it has one.
    fn tick_depth(&mut self) {
        let Some(depth) = self.depth.as_ref() else { return };
        if let Some(why) = depth.failed() {
            self.stop_depth();
            self.notify(&format!("DEPTH OFF: {}", first_line(&why).to_uppercase()));
            eprintln!("show: depth stopped: {why}");
            self.redraw();
            return;
        }
        let have = self.depth_map.as_ref().is_some_and(|m| m.generation == self.generation);
        if self.video.is_none()
            && !have
            && let Some(path) = self.playlist.current().map(|p| p.to_path_buf())
        {
            depth.offer_picture(&path, self.generation);
        }
        let Some(map) = depth.collect() else { return };
        if map.generation != self.generation {
            return;
        }
        self.depth_map = Some(map);
        // A playing film draws it with the next frame; anything standing
        // still has to be asked to.
        if self.video.is_none() || self.paused {
            self.redraw();
        }
    }

    fn toggle_detection(&mut self) {
        if self.watcher.is_some() {
            self.stop_detection();
            self.notify("DETECTION OFF");
        } else {
            self.start_detection();
        }
    }

    /// Rebuild the cache readout when what it says would change.
    ///
    /// Rebuilt on a whole second of span or a percent of fill, not on every
    /// frame: at twenty-five frames a second the text would be identical
    /// twenty-four times out of twenty-five, and rasterising it each time
    /// would put real work on the display path to no effect.
    fn tick_cache_badge(&mut self) {
        if self.video.is_none() || !self.config.show_debug || !self.config.show_zoom {
            if self.cache_badge.is_some() {
                self.cache_badge = None;
                self.cache_shown = None;
            }
            return;
        }
        let (behind, _) = self.frame_cache.around(self.position);
        let gathering = self.backfill.as_ref().map(|j| j.gathered()).unwrap_or_default();
        let back_span = behind + gathering;
        let back = back_span.as_secs_f32() / crate::framecache::DEFAULT_SPAN.as_secs_f32();

        let (queued, depth) = self.video.as_ref().map(|p| p.read_ahead()).unwrap_or((0, 1));
        let rate = self.video.as_ref().map(|p| p.info.frame_rate).unwrap_or(25.0).max(1.0);
        let frame_bytes = self.video.as_ref().map(|p| p.frame_bytes()).unwrap_or(0);
        let forward = queued as f32 / depth.max(1) as f32;

        // Rounded before comparing, so the readout is rebuilt when it would
        // read differently and not on every frame.
        let key = ((back * 100.0) as u32 as usize, (forward * 100.0) as u32);
        if self.cache_shown == Some(key) {
            return;
        }
        self.cache_shown = Some(key);

        let bar = |fraction: f32| -> String {
            const WIDTH: usize = 10;
            let filled = (fraction.clamp(0.0, 1.0) * WIDTH as f32).round() as usize;
            (0..WIDTH).map(|i| if i < filled { '=' } else { '-' }).collect()
        };

        // Two windows, because from any moment the next press could go either
        // way: what has been seen and what has been decoded ahead.
        self.cache_badge = Some(osd::render_scaled(&[
            osd::Row::dim(
                "BACK",
                format!(
                    "[{}] {:.1}S  {}",
                    bar(back),
                    back_span.as_secs_f32(),
                    crate::loader::human_bytes(self.frame_cache.bytes())
                ),
            ),
            osd::Row::dim(
                "FORWARD",
                format!(
                    "[{}] {:.1}S  {}",
                    bar(forward),
                    queued as f32 / rate as f32,
                    crate::loader::human_bytes(queued as u64 * frame_bytes)
                ),
            ),
        ], osd::SCALE_DEBUG));
    }

    /// Move one frame, and stop playback so it can be looked at.
    ///
    /// Backwards comes from the cache when the frame is held, which is the
    /// common case and costs nothing. Beyond it the only way back is to seek
    /// to the keyframe before and decode forward, which the seek path already
    /// does — it lands earlier than asked, but it moves, which is better than
    /// a key that stops working at the edge of the cache.
    fn step_frame(&mut self, forward: bool) {
        if self.video.is_none() {
            return;
        }
        // Stepping is inspection, and inspection means stopped.
        if !self.paused {
            self.paused = true;
            if let Some(player) = self.video.as_ref() {
                player.set_playing(false);
            }
            self.build_status_badge();
        }

        let at = self.position;
        // A cached frame only answers a step if it is the neighbour: the run
        // can have a hole in it, and a step across a hole is a jump.
        let near = self.frame_cache.continuous();
        if forward {
            // The next frame is already decoded and waiting; if the queue has
            // run dry, take the next one to arrive; and if nothing arrives,
            // go and get it.
            if let Some(frame) =
                self.frame_cache.after(at).filter(|f| f.at.saturating_sub(at) <= near).map(|f| f.clone_ref())
            {
                self.position = frame.at;
                self.present_frame(&frame);
            } else if let Some(frame) = self.decoded_neighbour(at, near) {
                self.position = frame.at;
                self.present_frame(&frame);
                self.frame_cache.push(frame);
            } else if !self.step_forward_uncached(at) {
                return;
            }
        } else if let Some(frame) =
            self.frame_cache.before(at).filter(|f| at.saturating_sub(f.at) <= near).map(|f| f.clone_ref())
        {
            self.position = frame.at;
            self.present_frame(&frame);
        } else if !self.step_back_uncached(at) {
            return;
        }

        self.stepped = true;
        self.build_frame_badge();
        self.show_scrubber();
        self.caption_for = None;
        self.tick_subtitles();
        self.redraw();
    }

    /// Bring sound and decoder to the stepped-to frame before playing on.
    ///
    /// Otherwise resuming waits: the sound picks up where the pause left it,
    /// frames behind the picture, and the picture holds until it catches up —
    /// seconds, if a step had to seek, since that restarts the sound at the
    /// keyframe. Cheapest first: the decoder's next frame is usually the
    /// right one already, and the sound for the frames stepped over is
    /// usually still queued and can simply be dropped. Only when either is
    /// wrong does it seek, exactly, to the frame on screen.
    ///
    /// Returns whether it had to seek.
    fn resync_after_steps(&mut self) -> bool {
        let at = self.position;
        let near = self.frame_cache.continuous();
        // Puts the decoder's next frame in `next_frame`, passing over any
        // already shown; a frame further on than the next means a gap.
        let next = match self.decoded_neighbour(at, near) {
            Some(frame) => {
                self.next_frame = Some(frame);
                true
            }
            None => self.next_frame.is_none(),
        };
        let Some(player) = self.video.as_ref() else { return false };
        let sound = player.audio().is_none_or(|a| a.skipper.skip_to(at));
        if next && sound {
            return false;
        }
        self.seek_to(at, false, true);
        true
    }

    /// Step back one frame with nothing cached to step back into.
    ///
    /// The only way backwards through an inter-coded stream: seek to the
    /// keyframe at or before the target and decode forward, keeping the last
    /// frame that falls short of it. The whole group of pictures may have to
    /// be decoded to move a single frame, which is why the cache exists — but
    /// when the cache is empty the key must still mean one frame rather than
    /// quietly becoming a seek to the previous keyframe.
    ///
    /// Bounded by a deadline, because a long group at four megabytes a frame
    /// is a real fraction of a second and the window must not stop answering.
    /// Falling short lands early rather than not moving, which is the same
    /// direction and a smaller lie than not responding at all.
    ///
    /// Returns whether anything was shown.
    fn step_back_uncached(&mut self, target: std::time::Duration) -> bool {
        use std::time::{Duration, Instant};
        const BUDGET: Duration = Duration::from_millis(400);

        let Some(player) = self.video.as_mut() else { return false };
        // Just before the target, so the seek cannot land on the target frame
        // itself and leave the picture where it started.
        let landing = target.saturating_sub(Duration::from_millis(1));
        player.seek(landing, false);
        self.next_frame = None;

        let deadline = Instant::now() + BUDGET;
        let mut best: Option<crate::video::Frame> = None;
        while Instant::now() < deadline {
            let Some(player) = self.video.as_mut() else { break };
            match player.try_next() {
                // Past the target: the one before it is the answer.
                Some(frame) if frame.at >= target => break,
                Some(frame) => best = Some(frame),
                None if player.finished() => break,
                None => std::thread::sleep(Duration::from_millis(2)),
            }
        }

        let Some(frame) = best else { return false };
        self.position = frame.at;
        self.present_frame(&frame);
        self.frame_cache.push(frame);
        true
    }

    /// The decoder's next frame, if it is the one after `at`.
    ///
    /// The decoder runs ahead of the picture — after a seek it keeps going and
    /// queues what it decodes — so its next frame can be well past the one on
    /// screen, and stepping to it was a jump. Frames at or behind `at` are
    /// passed over; one further ahead than a frame and a half is left for
    /// playback to show, and the step goes and decodes the right one instead.
    fn decoded_neighbour(&mut self, at: std::time::Duration, near: std::time::Duration) -> Option<crate::video::Frame> {
        const BEHIND: std::time::Duration = std::time::Duration::from_micros(500);
        loop {
            let frame = self.next_frame.take().or_else(|| self.video.as_mut().and_then(|p| p.try_next()))?;
            if frame.at <= at + BEHIND {
                continue;
            }
            if frame.at.saturating_sub(at) <= near {
                return Some(frame);
            }
            self.next_frame = Some(frame);
            return None;
        }
    }

    /// Step forward one frame when the decoder has stopped supplying them.
    ///
    /// It stops because the film is paused. Nothing is draining the sound, so
    /// the audio queue fills, so the demuxer blocks putting a packet into it,
    /// so no more video packets are read either — and stepping forward works
    /// only until whatever was already decoded runs out. Which it does, a few
    /// hundred frames in, and then the key stops doing anything.
    ///
    /// Seeking restarts all of that: it clears the queues, which unblocks the
    /// demuxer, and the frames start coming again. The seek lands at or before
    /// where we are, so the frames between there and here are decoded past —
    /// the same work the backward step does, in the other direction.
    ///
    /// Returns whether anything was shown.
    fn step_forward_uncached(&mut self, at: std::time::Duration) -> bool {
        use std::time::{Duration, Instant};
        // Longer than the backward step's budget: this one may have a whole
        // group of pictures to decode before it reaches the frame after the
        // one on screen, and a keypress that does nothing at all is worse than
        // one that takes a moment.
        const BUDGET: Duration = Duration::from_millis(900);

        let Some(player) = self.video.as_mut() else { return false };
        if player.finished() {
            return false;
        }
        // Round down to the keyframe at or before where we are, then decode
        // forward to the first frame past it.
        player.seek(at, false);
        self.next_frame = None;

        let deadline = Instant::now() + BUDGET;
        while Instant::now() < deadline {
            let Some(player) = self.video.as_mut() else { break };
            match player.try_next() {
                Some(frame) if frame.at > at => {
                    self.position = frame.at;
                    self.present_frame(&frame);
                    self.frame_cache.push(frame);
                    return true;
                }
                // Still short of where we are: keep it, it is the window the
                // next step back will want.
                Some(frame) => self.frame_cache.push(frame),
                None if player.finished() => break,
                None => std::thread::sleep(Duration::from_millis(2)),
            }
        }
        false
    }

    /// The frame number and its exact moment, for stepping through them.
    ///
    /// The number is padded to the width of the last frame in the film, so it
    /// does not change width as it counts and the readout stays still while
    /// stepping — the whole point being to compare one frame with the next.
    fn build_frame_badge(&mut self) {
        let Some(player) = self.video.as_ref() else { return };
        let rate = player.info.frame_rate.max(1.0);
        let frame = (self.position.as_secs_f64() * rate).round() as u64;
        let width = match player.info.duration {
            Some(d) => ((d.as_secs_f64() * rate).round() as u64).max(1).to_string().len(),
            // No duration to measure against; six digits is over three hours
            // at twenty-five a second.
            None => 6,
        };
        self.frame_badge = Some(osd::render_scaled(
            &[osd::Row::good(
                format!("{frame:0width$}"),
                osd::timecode_ms(self.position),
            )],
            osd::SCALE_PANEL * 1.4,
        ));
    }
    ///
    /// A keyframe is where decoding can begin, so landing on one is both exact
    /// and immediate — no frames have to be decoded and thrown away to reach
    /// it. How far it moves is therefore the film's business rather than ours:
    /// a few seconds in most encodes, less in one cut quickly.
    fn step_keyframe(&mut self, forward: bool) {
        // Just off the current position, so the seek cannot round back onto
        // the keyframe we are already sitting on.
        const NUDGE: std::time::Duration = std::time::Duration::from_millis(10);
        let target = if forward {
            // Forwards, the play position is the right place to measure from:
            // the next keyframe is the next one after now.
            self.position.saturating_add(NUDGE)
        } else {
            // Backwards it is not. Half way through a group of pictures the
            // keyframe before *now* is the one already on screen, so stepping
            // measures from where this picture began instead.
            self.last_keyframe.min(self.position).saturating_sub(NUDGE)
        };
        self.seek_to(target, forward, false);
    }

    /// Move by `delta`, clamped to the file.
    fn seek_by(&mut self, delta: std::time::Duration, forward: bool) {
        let target = if forward {
            self.position.saturating_add(delta)
        } else {
            self.position.saturating_sub(delta)
        };
        self.seek_to(target, forward, true);
    }

    /// Jump to `target`, clamped to the file.
    ///
    /// Both ends are walls rather than doors: running off the start does not
    /// step to the previous file, and running off the end leaves the last
    /// frame up. A seek key is a request about *this* film.
    ///
    /// `exact` says whether the number matters. Ten seconds back means ten
    /// seconds back, so the frames between the keyframe and there are decoded
    /// past rather than shown; stepping to a keyframe means the keyframe, and
    /// skipping past it would be skipping the thing that was asked for.
    fn seek_to(&mut self, target: std::time::Duration, forward: bool, exact: bool) {
        let Some(player) = self.video.as_mut() else { return };
        let duration = player.info.duration;

        let target = match duration {
            // Already at the end: there is nothing further to show, so the
            // last frame stays and the key does nothing.
            Some(d) if forward && self.position + std::time::Duration::from_millis(500) >= d => {
                self.show_scrubber();
                return;
            }
            Some(d) => target.min(d),
            None => target,
        };

        let landed = if exact { player.seek_exact(target, forward) } else { player.seek(target, forward) };
        // Hold the sound until the first picture from the new place is up, as
        // at the start of a file: the decoder may need a good part of a second
        // to get from the keyframe to the target, and the audio does not.
        player.set_playing(false);
        self.seek_target = exact.then_some(landed);
        self.stepped = false;
        self.position = landed;
        self.paused_seek = self.paused;
        // The wall clock is the fallback master and is anchored to the old
        // position; dropping it lets the first frame after the seek re-anchor.
        self.playback_origin = None;
        self.next_frame = None;
        self.playback_ended = false;
        self.subtitles.retire(landed);
        self.show_scrubber();
        self.redraw();
    }

    /// Say something briefly, in the corner the volume uses.
    fn notify(&mut self, text: &str) {
        let overlay = osd::render_scaled(&[osd::Row::plain(text)], osd::SCALE_PANEL);
        self.volume_osd =
            Some((overlay, std::time::Instant::now() + std::time::Duration::from_millis(1600)));
    }

    /// Write the settings out, saying so only when it fails.
    ///
    /// Every page changes settings in memory and leaves the writing to here,
    /// so that a page can be tested without touching the real file.
    fn save_settings(&mut self) {
        if let Err(e) = self.config.save() {
            self.notify(&format!("NOT SAVED: {e}"));
            eprintln!("show: could not save settings: {e:#}");
        }
    }

    fn open_plugins(&mut self) {
        let mut page = crate::plugins::Plugins::new();
        page.set_settings(self.config.look_hz, self.config.sharpen, self.thresholds());
        self.plugins = Some(page);
        self.redraw();
    }

    /// Fetch the highlighted plugin, on a thread, reporting as it goes.
    ///
    /// Never starts a second while one is running: two downloads writing
    /// progress into the same page would fight over it, and the page is the
    /// only place either of them is visible.
    fn install_plugin(&mut self) {
        let Some(page) = self.plugins.as_mut() else { return };
        if page.busy() {
            return;
        }
        if page.on_rate() {
            page.note("use left and right");
            self.redraw();
            return;
        }
        let id = page.selected();
        if let Some(needed) = page.missing_requirement(id) {
            page.note(format!("install {} first", needed.id()));
            self.redraw();
            return;
        }
        // Already there: enter means "use this one" rather than nothing at
        // all, which is how a model is chosen. Installing several and having
        // the app pick is not a choice, it is a guess.
        if crate::ml::store::installed(id) {
            match page.use_selected() {
                Ok(chosen) => {
                    page.note(format!("looking with {}", chosen.id()));
                    // The worker opened its model when it started, so it has to
                    // start again to pick up a different one.
                    if self.watcher.is_some() {
                        self.stop_detection();
                        self.start_detection();
                    }
                }
                Err(why) => page.note(why),
            }
            self.redraw();
            return;
        }

        page.set_status(id, crate::plugins::Status::Working("STARTING".into()));
        page.note(format!("fetching {}", id.id()));
        let proxy = self.proxy.clone();
        let spawned = std::thread::Builder::new().name("plugin-install".into()).spawn(move || {
            let report = |status| {
                let _ = proxy.send_event(UserEvent::PluginProgress(id, status));
            };
            let result = crate::ml::store::install(id, &mut |step| {
                report(match step {
                    crate::ml::store::Step::Fetching(done, total) => {
                        crate::plugins::Status::Working(format!(
                            "[{}] {}",
                            crate::plugins::bar(done as f32 / total.max(1) as f32),
                            crate::loader::human_bytes(done)
                        ))
                    }
                    crate::ml::store::Step::Verifying => {
                        crate::plugins::Status::Working("CHECKING THE DIGEST".into())
                    }
                    crate::ml::store::Step::Unpacking => {
                        crate::plugins::Status::Working("UNPACKING".into())
                    }
                });
            });
            let _ = proxy.send_event(UserEvent::PluginDone(id, Box::new(result)));
        });
        if let Err(e) = spawned {
            page.note(format!("could not start: {e}"));
        }
        self.redraw();
    }

    fn remove_plugin(&mut self) {
        let Some(page) = self.plugins.as_mut() else { return };
        if page.busy() {
            return;
        }
        let id = page.selected();
        let freed = crate::ml::store::disk_used(id);
        match crate::ml::store::remove(id) {
            Ok(()) => page.note(format!(
                "removed {}, freeing {}",
                id.id(),
                crate::loader::human_bytes(freed)
            )),
            Err(e) => page.note(format!("{e:#}")),
        }
        page.refresh();
        self.redraw();
    }

    /// Re-read every installed byte and check it against what was recorded.
    ///
    /// Deliberate rather than automatic: it is minutes of reading on a large
    /// model, and opening a page should never cost that.
    fn verify_plugin(&mut self) {
        let Some(page) = self.plugins.as_mut() else { return };
        let id = page.selected();
        match crate::ml::store::verify(id) {
            Ok(()) => page.note(format!("{} matches what was installed", id.id())),
            Err(e) => {
                page.set_status(id, crate::plugins::Status::Failed("DAMAGED".into()));
                page.note(format!("{e:#}"));
            }
        }
        self.redraw();
    }

    /// Open the subtitle picker, listing what is loaded and what could be.
    fn open_menu(&mut self) {
        let Some(player) = self.video.as_ref() else { return };
        let path = player.path.clone();

        let mut entries = vec![MenuEntry::Off];
        for i in 0..self.subtitles.tracks.len() {
            entries.push(MenuEntry::Loaded(i));
        }
        // Files not already loaded: everything in the folder, since subtitles
        // downloaded separately are rarely named to match the film.
        let loaded: Vec<std::path::PathBuf> = self
            .subtitles
            .tracks
            .iter()
            .filter_map(|t| t.external.clone())
            .collect();
        for file in crate::subtitles::find_in_folder(&path) {
            if !loaded.contains(&file) {
                entries.push(MenuEntry::File(file));
            }
        }

        // Open on whatever is showing, so the first thing the arrow keys move
        // away from is the current choice.
        let selected = match self.subtitles.selected_index() {
            Some(i) => i + 1,
            None => 0,
        };
        self.menu = Some(SubtitleMenu { entries, selected, overlay: None });
        self.build_menu();
        self.redraw();
    }

    fn build_menu(&mut self) {
        let Some(menu) = self.menu.as_ref() else { return };
        let mut rows = vec![osd::Row::dim("SUBTITLES", "")];

        for (i, entry) in menu.entries.iter().enumerate() {
            let (label, value) = match entry {
                MenuEntry::Off => ("OFF".to_string(), String::new()),
                MenuEntry::Loaded(t) => {
                    let track = &self.subtitles.tracks[*t];
                    let current = self.subtitles.selected_index() == Some(*t);
                    (track.label(), if current { "SHOWING".into() } else { "LOADED".into() })
                }
                MenuEntry::File(path) => (
                    path.file_name().unwrap_or_default().to_string_lossy().to_uppercase(),
                    "FILE".to_string(),
                ),
            };
            let row = if matches!(entry, MenuEntry::Off) && self.subtitles.selected().is_none() {
                osd::Row::good(label, "SHOWING")
            } else {
                osd::Row::new(label, value, osd::TEXT)
            };
            rows.push(row.selected(i == menu.selected));
        }

        rows.push(osd::Row::plain(""));
        rows.push(osd::Row::dim("UP/DOWN CHOOSE", "ENTER SELECT"));
        rows.push(osd::Row::dim("ALT OR ESC", "CLOSE"));

        let win = self.win_size();
        let overlay = osd::render_menu(&rows, osd::SCALE_PANEL, (win.0 * 0.8) as u32);
        if let Some(menu) = self.menu.as_mut() {
            menu.overlay = Some(overlay);
        }
    }

    /// Act on the highlighted entry.
    fn menu_activate(&mut self) {
        let Some(menu) = self.menu.as_ref() else { return };
        let Some(entry) = menu.entries.get(menu.selected) else { return };

        match entry {
            MenuEntry::Off => self.subtitles.select(None),
            MenuEntry::Loaded(t) => self.subtitles.select(Some(*t)),
            MenuEntry::File(path) => {
                // Numbered above the container's own streams so a file and an
                // embedded stream can never collide.
                const EXTERNAL_BASE: usize = 1 << 20;
                let index = EXTERNAL_BASE + self.subtitles.tracks.len();
                match crate::subtitles::load_external(&path.clone(), index) {
                    Ok((track, cues)) => {
                        self.subtitles.add_track(track);
                        for cue in cues {
                            self.subtitles.push(cue);
                        }
                        let last = self.subtitles.tracks.len() - 1;
                        self.subtitles.select(Some(last));
                    }
                    Err(e) => {
                        let name = path.file_name().unwrap_or_default().to_string_lossy();
                        eprintln!("show: {}: {e:#}", path.display());
                        self.notify(&format!("COULD NOT READ {}", name.to_uppercase()));
                    }
                }
            }
        }

        // The caption on screen belongs to whatever was showing before.
        self.caption = None;
        self.caption_for = None;
        self.tick_subtitles();
        self.menu = None;
        self.redraw();
    }

    /// Type size for captions, as a share of the picture's height on screen.
    ///
    /// The picture rather than the window, so zooming carries the subtitles
    /// with it — magnifying the film and leaving the words the size they were
    /// makes them shrink against everything around them. Clamped at both ends:
    /// a deep zoom would otherwise fill the screen with two words, and a tiny
    /// window would produce text nobody could read.
    fn caption_size(&self) -> f32 {
        caption_px(self.picture_height().unwrap_or_else(|| self.win_size().1)) * self.config.caption_scale
    }

    /// Ctrl < and >: subtitles a tenth smaller or larger, kept for next time.
    fn scale_captions(&mut self, steps: i32) {
        let scale = (self.config.caption_scale * 1.1f32.powi(steps)).clamp(0.5, 3.0);
        self.config.caption_scale = (scale * 20.0).round() / 20.0;
        if let Err(e) = self.config.save() {
            eprintln!("show: could not save the subtitle size: {e:#}");
        }
        self.notify(&format!("SUBTITLES {}%", (self.config.caption_scale * 100.0).round() as i32));
        self.redraw();
    }

    /// The picture's height on screen, in window pixels.
    fn picture_height(&self) -> Option<f32> {
        let source = self.shown.as_ref()?.source;
        let full = self.window_size();
        let rect = self.view.ndc_rect_in(source, self.image_viewport(), full);
        Some((rect[3] - rect[1]) * full.1 / 2.0)
    }

    /// The bottom edge of the picture, in window pixels.
    fn picture_bottom(&self) -> Option<f32> {
        let source = self.shown.as_ref()?.source;
        let full = self.window_size();
        let rect = self.view.ndc_rect_in(source, self.image_viewport(), full);
        // Clip space runs bottom-up from -1; the window runs top-down from 0.
        Some((1.0 - rect[1]) * full.1 / 2.0)
    }

    /// Fit whatever is on screen to the window.
    ///
    /// A film fills it; a picture obeys the autoscale setting.
    fn fit_current(&mut self) {
        let Some(source) = self.shown.as_ref().map(|s| s.source) else { return };
        let win = self.win_size();
        if self.video.is_some() {
            self.view.fit_filling(source, win);
        } else {
            self.view.fit(source, win);
        }
    }

    /// Turn what is on screen by quarter turns, clockwise positive. Only the
    /// view: nothing is written, and the next file opens upright.
    fn rotate_by(&mut self, quarters: i32) {
        let Some(source) = self.shown.as_ref().map(|s| s.source) else { return };
        let win = self.win_size();
        self.view.rotate(quarters, source, win);
        // A fitted picture refits, since on its side it fits differently; one
        // zoomed by hand keeps its zoom and the part of it in the middle.
        if self.view.fitting {
            self.fit_current();
        }
        self.notify(match self.view.turn {
            0 => "UPRIGHT",
            1 => "TURNED RIGHT",
            2 => "UPSIDE DOWN",
            _ => "TURNED LEFT",
        });
        self.redraw();
    }

    /// Fit to cover the window, or to show all of it, and refit to that now.
    fn set_cover(&mut self, on: bool) {
        self.view.cover = on;
        if self.config.cover != on {
            self.config.cover = on;
            self.save_settings();
        }
        self.fit_current();
    }

    /// Stop or start playback, and say so on screen.
    fn toggle_pause(&mut self) {
        if self.video.is_none() {
            return;
        }
        // Over: there is nothing to pause or resume, so play it again, from
        // the start — what a player's play button does at the end of a film.
        if self.playback_ended {
            self.seek_to(std::time::Duration::ZERO, false, true);
            self.paused = false;
            self.frame_badge = None;
            if let Some(player) = self.video.as_ref() {
                player.set_playing(true);
            }
            self.show_scrubber();
            self.redraw();
            return;
        }
        self.paused = !self.paused;
        if !self.paused {
            // Playing again: the frame readout was about one frame.
            self.frame_badge = None;
            if std::mem::take(&mut self.stepped) && self.resync_after_steps() {
                // The seek holds the sound and starts it with the first frame.
                self.build_status_badge();
                return;
            }
        }
        let playing = !self.paused;
        if let Some(player) = self.video.as_ref() {
            player.set_playing(playing);
        }
        // Both ways: the bar stays up while paused and fades normally once
        // running again.
        self.show_scrubber();
        if !self.paused {
            // Resume where it left off rather than jumping forward by however
            // long the pause lasted. Only the wall clock needs this; with
            // sound the audio clock did not advance while stopped.
            // Not yet decoded, the next frame is a frame after this one.
            let next = self.next_frame.as_ref().map_or(self.position, |f| f.at);
            if let Some(origin) = self.playback_origin.as_mut() {
                *origin = std::time::Instant::now() - next;
            }
        }
        self.build_status_badge();
        self.redraw();
    }

    /// Put the seek bar up, and give it a fresh moment on screen.
    fn show_scrubber(&mut self) {
        self.build_scrubber(true);
    }

    /// Draw the bar for the current position.
    ///
    /// `extend` is what separates the user asking for it from it merely being
    /// kept current: redrawing every second to move the timecode must not keep
    /// pushing the fade back, or it would never fade at all.
    fn build_scrubber(&mut self, extend: bool) {
        if self.video.is_none() {
            return;
        }
        let duration = self.video.as_ref().and_then(|p| p.info.duration);
        let win = self.win_size();
        let width = (win.0 * 0.6).clamp(320.0, 1100.0) as u32;

        let fraction = match duration {
            Some(d) if d > std::time::Duration::ZERO => {
                self.position.as_secs_f32() / d.as_secs_f32()
            }
            _ => 0.0,
        };
        let total = duration.map(osd::timecode).unwrap_or_else(|| "--:--".into());
        // Paused is a state with no other visible sign — the picture simply
        // stops, which on a still scene looks like nothing happened.
        let elapsed = if self.paused {
            format!("{}  PAUSED", osd::timecode(self.position))
        } else {
            osd::timecode(self.position)
        };
        let overlay = osd::scrubber(&elapsed, &total, fraction, width, osd::SCALE_COMPACT);
        self.scrubber_at = Some(self.position);
        let until = match (extend, self.scrubber.as_ref()) {
            (false, Some((_, until))) => *until,
            _ => std::time::Instant::now() + std::time::Duration::from_millis(2500),
        };
        self.scrubber = Some((overlay, until));
    }

    /// Keep the seek bar current, and drop it once it has had its moment.
    fn tick_scrubber(&mut self) -> Option<std::time::Instant> {
        let (_, until) = self.scrubber.as_ref()?;
        let until = *until;

        // While paused the bar stays up: it is the only thing on screen saying
        // so, and a bar that faded would leave a frozen picture unexplained.
        // The same while it is being dragged, for the obvious reason.
        // Paused but with the pointer gone, nobody is looking at the bar.
        let held = (self.paused && self.pointer_inside) || self.scrubbing;
        if !held && std::time::Instant::now() >= until {
            self.scrubber = None;
            self.scrubber_at = None;
            self.redraw();
            return None;
        }

        // A bar showing the moment it appeared rather than the moment now is
        // worse than no bar. Rebuilt a second at a time, which is all the
        // timecode resolves to anyway.
        let stale = self
            .scrubber_at
            .is_none_or(|at| at.as_secs() != self.position.as_secs());
        if stale {
            self.build_scrubber(false);
            self.redraw();
        }
        (!self.paused && !self.scrubbing).then_some(until)
    }

    /// Where the seek bar is on screen, if it is showing.
    fn scrubber_rect(&self) -> Option<(i32, i32, u32, u32)> {
        let (overlay, _) = self.scrubber.as_ref()?;
        let (x, y) = Corner::BottomCentre.origin(overlay, self.win_size());
        Some((x, y, overlay.width, overlay.height))
    }

    /// How far along the track a pointer at `cursor_x` is, as a fraction.
    ///
    /// Pure, and shared by the press and the drag, so the two cannot disagree
    /// about where a given pixel points.
    fn track_fraction(cursor_x: f32, overlay_x: i32, track_x: u32, track_w: u32) -> f32 {
        let along = cursor_x - (overlay_x + track_x as i32) as f32;
        (along / track_w.max(1) as f32).clamp(0.0, 1.0)
    }

    /// The position a click at `x` refers to, if it lands on the bar.
    ///
    /// The whole overlay is the target rather than the handle alone: a handle
    /// a few pixels wide is a poor thing to have to hit, and every player
    /// treats a click on the track as "go there".
    fn scrubber_hit(&self, cursor: (f32, f32)) -> Option<std::time::Duration> {
        let duration = self.video.as_ref()?.info.duration?;
        let (x, y, w, h) = self.scrubber_rect()?;

        // Generous vertically: the bar is thin, and the pointer arriving from
        // above should still catch it.
        const SLOP: f32 = 12.0;
        if cursor.1 < y as f32 - SLOP || cursor.1 > (y + h as i32) as f32 + SLOP {
            return None;
        }
        let (track_x, track_w) = osd::scrubber_track(w, osd::SCALE_COMPACT);
        let along = cursor.0 - (x + track_x as i32) as f32;
        if along < -SLOP || along > track_w as f32 + SLOP {
            return None;
        }
        Some(duration.mul_f32(Self::track_fraction(cursor.0, x, track_x, track_w)))
    }

    /// Move the handle to `to`, seeking at most every so often.
    ///
    /// The picture follows the drag rather than waiting for it to end, which
    /// is what makes finding a moment possible. Seeking on every mouse event
    /// would be a seek per pixel, so they are spaced out; the release always
    /// sends a final one so the drag ends exactly where it was left.
    fn scrub_to(&mut self, to: std::time::Duration, force: bool) {
        const EVERY: std::time::Duration = std::time::Duration::from_millis(120);
        self.position = to;
        self.build_scrubber(true);

        let due = force || self.scrub_sent.is_none_or(|t| t.elapsed() >= EVERY);
        // Already on its way there: the release of a click that did not move.
        let repeat = force && self.scrub_exact == Some(to);
        self.scrub_exact = force.then_some(to);
        if due && !repeat {
            // A seek can only start at a keyframe, so what comes back begins
            // as much as several seconds before what was asked for. Those
            // frames are decoded — there is no other way to reach the one
            // wanted — but showing them puts the picture, and the handle,
            // somewhere the user did not click. Only the settled seek does
            // this: during a drag the keyframes are the preview, and waiting
            // for exactness on every mouse move would make dragging crawl.
            self.seek_target = force.then_some(to);
            self.scrub_sent = Some(std::time::Instant::now());
            if let Some(player) = self.video.as_mut() {
                if force { player.seek_exact(to, false) } else { player.seek(to, false) };
                // Silent until the new place is on screen; see `seek_to`.
                player.set_playing(false);
                self.playback_origin = None;
                self.next_frame = None;
                self.playback_ended = false;
                // Paused, the picture still follows the playhead.
                self.paused_seek = self.paused;
            }
        }
        self.redraw();
    }

    /// Turn the volume up or down and show where it landed.
    fn adjust_volume(&mut self, steps: f32) {
        let Some(volume) = self.video.as_ref().and_then(|p| p.audio()).map(|a| a.volume.clone())
        else {
            return;
        };
        // A fixed fraction of full scale, so the whole range is a few flicks
        // of the wheel rather than a long grind at the quiet end.
        //
        // Reaching for the volume while muted is a request to hear something,
        // so it lifts the mute and starts from the level that was silenced
        // rather than climbing back up from nothing.
        let from = if self.muted { self.config.volume } else { volume.get() };
        self.muted = false;
        let level = (from + steps.signum() * 0.05).clamp(0.0, crate::audio::Volume::MAX);
        volume.set(level);
        // Remembered for the next file and the next session. Saving on every
        // notch would be a write per wheel click, so it waits for the readout
        // to go away — which is the point at which the user has settled.
        self.config.volume = level;
        self.volume_unsaved = true;
        self.build_volume_osd(level);
        self.redraw();
    }

    /// Silence the sound, remembering what it was.
    ///
    /// The saved level is deliberately left alone: mute is a moment, and
    /// coming back to a volume nobody chose would be worse than having no
    /// mute at all. So only the gain moves, and `config.volume` is what it
    /// comes back to.
    fn toggle_mute(&mut self) {
        let Some(volume) = self.video.as_ref().and_then(|p| p.audio()).map(|a| a.volume.clone())
        else {
            // A film with no sound has nothing to silence, and a key that
            // quietly does nothing is a key that looks broken.
            self.notify("NO SOUND");
            self.redraw();
            return;
        };
        self.muted = !self.muted;
        volume.set(if self.muted { 0.0 } else { self.config.volume });
        self.build_volume_osd(self.config.volume);
        self.redraw();
    }

    fn build_volume_osd(&mut self, level: f32) {
        const WIDTH: usize = 20;
        // Muted shows an empty bar rather than the level it will come back to:
        // a bar with something in it beside the word MUTED reads as a level.
        let filled = if self.muted {
            0
        } else {
            ((level / crate::audio::Volume::MAX) * WIDTH as f32).round() as usize
        };
        let bar: String = (0..WIDTH)
            .map(|i| if i < filled { '=' } else { '-' })
            .collect();
        let value = if self.muted {
            "MUTED".to_string()
        } else {
            format!("{:>4}%", (level * 100.0).round() as i32)
        };
        let rows = [osd::Row::new(
            "VOLUME",
            format!("[{bar}] {value:>5}"),
            osd::ACCENT,
        )];
        let overlay = osd::render_scaled(&rows, osd::SCALE_PANEL);
        self.volume_osd =
            Some((overlay, std::time::Instant::now() + std::time::Duration::from_millis(1400)));
    }

    /// Drop the volume readout once its moment has passed.
    fn tick_volume_osd(&mut self) -> Option<std::time::Instant> {
        let (_, until) = self.volume_osd.as_ref()?;
        let until = *until;
        if std::time::Instant::now() >= until {
            self.volume_osd = None;
            self.redraw();
            return None;
        }
        Some(until)
    }

    /// The line shown while a phrase is being typed.
    ///
    /// Rebuilt on every keystroke rather than cached: it is one short line of
    /// the bitmap font, it only exists while somebody is typing, and a cache
    /// keyed on the text would be the same work plus a comparison.
    fn build_prompt(&mut self) {
        let Some(text) = self.typing.as_ref() else {
            self.prompt_osd = None;
            return;
        };
        // A block for the caret, because there is no cursor to blink and a
        // line that ends in nothing does not read as waiting for input.
        let rows = [osd::Row::new("FIND", format!("{}\u{2588}", text.to_uppercase()), osd::ACCENT)];
        self.prompt_osd = Some(osd::render_scaled(&rows, osd::SCALE_PANEL));
    }

    /// Build the "nothing to show" notice, once.
    fn build_notice(&mut self) {
        if self.notice.is_some() {
            return;
        }
        let dir = crate::playlist::Playlist::default_start_dir();
        let where_ = dir.display().to_string();
        let notice = osd::message(&[
            "NO IMAGES FOUND",
            &format!("LOOKED IN {where_}"),
            "",
            "OPEN AN IMAGE FROM YOUR FILE MANAGER,",
            "OR RUN  SHOW /PATH/TO/PICTURE.JPG",
            "",
            "ESC OR Q TO QUIT",
        ]);
        self.notice = Some(notice);
    }



    /// Everything to draw over the image this frame, in back-to-front order.
    ///
    /// A modal takes the screen alone: a question about deleting a file should
    /// not share space with a zoom readout.
    /// Everything to draw over the picture, with the window frame last.
    ///
    /// The frame is appended here rather than inside `contents`, which returns
    /// early in several places — the grid, the panel, the empty-folder notice
    /// — and would drop the window's own edges on any of those paths.
    fn overlay_stack(&self) -> Vec<(&Overlay, Corner)> {
        let mut stack = self.overlay_contents();
        if let Some((bar, left, right, bottom)) = &self.chrome {
            if self.title_bar_shown() {
                stack.push((bar, Corner::TitleBar));
            }
            stack.push((left, Corner::BorderLeft));
            stack.push((right, Corner::BorderRight));
            stack.push((bottom, Corner::BorderBottom));
        }
        stack
    }

    /// The window switcher's list, when dropped down from the filename.
    fn push_peer_switcher<'a>(&'a self, stack: &mut Vec<(&'a Overlay, Corner)>) {
        if self.peer_list_open
            && let (Some((list, _)), Some((x, y))) = (&self.peer_list, self.peer_list_at())
        {
            stack.push((list, Corner::At(x, y)));
        }
    }

    fn overlay_contents(&self) -> Vec<(&Overlay, Corner)> {
        let in_grid = self.mosaic.is_some();

        // The grid, when open, is the backdrop the rest is drawn onto — so a
        // menu over it does not leave the screen black behind.
        let mut stack: Vec<(&Overlay, Corner)> = Vec::new();
        if in_grid {
            stack.extend(self.mosaic_composite.iter().map(|c| (c, Corner::Fill)));
        }

        if let Some(confirm) = &self.confirming_delete {
            stack.push((confirm, Corner::Centre));
            return stack;
        }
        if let Some(page) = self.keys.as_ref().and_then(|k| k.cached()) {
            stack.push((page, Corner::Centre));
            return stack;
        }
        if let Some(page) = self.plugins.as_ref().and_then(|p| p.cached()) {
            stack.push((page, Corner::Centre));
            return stack;
        }
        if let Some(panel) = self.panel.as_ref().and_then(|p| p.cached()) {
            stack.push((panel, Corner::Centre));
            return stack;
        }
        if let Some(progress) = &self.analysis {
            stack.push((progress, if in_grid { Corner::TopLeft } else { Corner::Centre }));
            return stack;
        }
        if let Some(browser) = &self.browser_overlay {
            // The panel and the picture share the window rather than overlap,
            // so the image badges still belong on screen beside it.
            stack.push((browser, Corner::RightEdge));
            if let Some(name) = &self.filename_badge {
                stack.push((name, Corner::TopLeft));
            }
            return stack;
        }
        if in_grid {
            stack.push((&self.grid_badge, Corner::TopRight));
            if let Some((_, label)) = &self.hover_label {
                stack.push((label, Corner::TopLeft));
            }
            if let Some((_, marker)) = &self.scroll_marker {
                stack.push((marker, Corner::BottomCentre));
            }
            // Top left, under the name: the bottom edge is the scroll strip's.
            if let Some((osd, _)) = &self.volume_osd {
                stack.push((osd, Corner::TopLeft));
            }
            return stack;
        }
        if self.shown.is_none() && !self.playlist.is_empty() {
            return Vec::new();
        }
        if self.shown.is_none() {
            return self.notice.iter().map(|n| (n, Corner::Centre)).collect();
        }

        let mut stack = Vec::with_capacity(4);
        // The filename is not a diagnostic, so F4 does not hide it — the
        // pointer leaving the window does.
        if let Some(name) = self.filename_badge.as_ref().filter(|_| self.pointer_inside) {
            stack.push((name, Corner::TopRight));
        }
        // Under the name: this picture knows where on the map it belongs.
        if let Some(badge) = &self.geo_badge {
            stack.push((badge, Corner::TopRight));
        }
        // The other open windows, dropped down from the filename.
        self.push_peer_switcher(&mut stack);
        // Nor is the volume: it is feedback for something the user is doing
        // right now, so it shows regardless of the debug switches.
        if let Some((osd, _)) = &self.volume_osd {
            stack.push((osd, Corner::BottomLeft));
        }
        // Above the volume's corner rather than in it: both can be on screen
        // at once, and what is being typed is the more important of the two.
        if let Some(prompt) = &self.prompt_osd {
            stack.push((prompt, Corner::BottomLeft));
        }
        if let Some((bar, _)) = &self.scrubber {
            stack.push((bar, Corner::BottomCentre));
        }
        if let Some(badge) = &self.frame_badge {
            stack.push((badge, Corner::BottomRight));
        }
        // Above the seek bar in the stack but below it on screen: a caption is
        // the reason the film is being watched, so nothing hides it.
        if let Some(caption) = &self.caption {
            stack.push((caption, Corner::Caption));
        }
        if let Some(bar) = &self.tag_bar {
            let top = self.chrome_metrics().map(|c| (c.border as i32, c.bar_height as i32));
            let (x, y) = top.unwrap_or((0, 0));
            stack.push((bar, Corner::At(x, y)));
        }
        // The picker sits over everything: it is a question waiting for an
        // answer, and nothing behind it can be acted on until it is closed.
        if let Some(overlay) = self.menu.as_ref().and_then(|m| m.overlay.as_ref()) {
            stack.push((overlay, Corner::Centre));
        }


        // F4 is the master switch; the individual settings apply beneath it.
        if !self.config.show_debug {
            return stack;
        }

        if self.config.show_zoom {
            if let Some(badge) = &self.status_badge {
                stack.push((badge, Corner::TopLeft));
            }
        }
        // How much can be stepped back through. Under the zoom badge, and
        // only while a film is playing — it means nothing for a picture.
        if let Some(badge) = &self.cache_badge {
            stack.push((badge, Corner::TopLeft));
        }
        // `t` overrides the setting for this session only; the config decides
        // what happens on the next launch.
        if self.config.show_stats && self.osd_visible {
            // A video has no load breakdown worth showing — it never finished
            // loading — so it gets the frame rate over time instead.
            match (&self.fps_chart, &self.osd) {
                (Some(chart), _) if self.video.is_some() => {
                    stack.push((chart, Corner::TopRight))
                }
                (_, Some(osd)) if self.video.is_none() => stack.push((osd, Corner::TopRight)),
                _ => {}
            }
        }
        stack
    }

    /// Upload the stack whenever it differs from what is already there.
    ///
    /// Derived rather than flagged: any change that produces a different set of
    /// overlays is picked up here, so no call site has to remember to announce
    /// itself. Comparing a handful of ids per frame costs nothing.
    fn sync_overlays(&mut self) {
        let win = self.win_size();
        if let Some(panel) = self.panel.as_mut() {
            panel.ensure_overlay(win);
        }
        if let Some(page) = self.plugins.as_mut() {
            page.ensure_overlay(win);
        }
        if let Some(page) = self.keys.as_mut() {
            page.ensure_overlay(win);
        }

        let wanted: Vec<u64> = self.overlay_stack().iter().map(|(ov, _)| ov.id).collect();
        if wanted == self.uploaded {
            return;
        }
        self.uploaded = wanted;

        // The stack borrows all of `self`, so the target is moved out for the
        // duration rather than copying every overlay's pixels to break the tie.
        let mut gpu = self.gpu.take();
        if let Some(gpu) = gpu.as_mut() {
            let uploads: Vec<(&[u8], u32, u32)> = self
                .overlay_stack()
                .iter()
                .map(|(ov, _)| (ov.rgba.as_slice(), ov.width, ov.height))
                .collect();
            gpu.set_overlays(&uploads);
        }
        self.gpu = gpu;
    }

    fn redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Pay off the focus request owed since the window was created.
    ///
    /// X11 refuses to activate a window it has not yet seen mapped, and winit
    /// reports that refusal by doing nothing at all: it drops `focus_window`
    /// unless its own visibility has already been flipped by the server's
    /// `VisibilityNotify`, which cannot arrive before the event loop starts
    /// pumping. So the request cannot be made where it belongs, at creation,
    /// and one retry is not enough either — it is repeated until the keyboard
    /// arrives or the grace period is up.
    ///
    /// It stops the moment the window has focus, so a window the user has
    /// deliberately clicked away from is never dragged back.
    ///
    /// Retried on a timer, returning when to try next. It used to ask for
    /// another frame instead, and that spun the loop — fifty redraws in the
    /// first 60 ms, each an X request — exactly when the first picture was
    /// trying to get through.
    fn claim_focus(&mut self) -> Option<std::time::Instant> {
        const RETRY: std::time::Duration = std::time::Duration::from_millis(25);
        let until = self.focus_until?;
        let w = self.window.as_ref()?;
        let now = std::time::Instant::now();
        if w.has_focus() || now > until {
            self.focus_until = None;
            return None;
        }
        w.focus_window();
        Some(now + RETRY)
    }

    fn draw(&mut self, el: &ActiveEventLoop) {
        let smooth = self.smooth();
        let win = self.win_size();

        if self.shown.is_none() && self.playlist.is_empty() {
            self.build_notice();
        }

        // The grid composites on the CPU and rides the overlay path, so the
        // image texture is left alone and returning to it costs nothing.
        if self.mosaic.as_mut().is_some_and(|m| m.take_dirty()) {
            let grid = self.mosaic.as_ref().unwrap();
            self.mosaic_composite = Some(grid.compose(win, grid.count(self.playlist.len())));
        }
        if self.mosaic.is_none() {
            self.mosaic_composite = None;
        }
        // Scrolling and re-sorting move cells under a still pointer.
        self.update_hover();
        self.update_scroll_marker();
        self.sync_overlays();

        // Two overlays in the same corner stack down it rather than colliding.
        let mut next_y: std::collections::HashMap<Corner, i32> = std::collections::HashMap::new();
        let full = self.window_size();
        // Where the picture ends, so subtitles can go under it rather than on
        // it. Resolved before the stack is borrowed.
        let picture_bottom = self.picture_bottom();
        let chrome = self.chrome_metrics();
        let tags = self.tag_bar_height() as i32;
        let filename_id = self.filename_badge.as_ref().map(|b| b.id);
        let mut filename_at = None;
        let placements: Vec<([f32; 4], (i32, i32))> = self
            .overlay_stack()
            .iter()
            .map(|(ov, corner)| {
                let (mut x, mut y) = corner.origin(ov, full);
                // Everything that hangs off an edge hangs off the inside of
                // the frame, not the outside: a badge drawn under our own
                // title bar is hidden by it, which looks like a mistake
                // because it is one.
                if let Some(c) = chrome.as_ref() {
                    match corner {
                        // Below the tag bar as well as the title bar: both
                        // take space off the top, and a badge drawn over
                        // either is a badge nobody can read.
                        Corner::TopLeft => {
                            y += c.bar_height as i32 + tags;
                            x += c.border as i32;
                        }
                        Corner::TopRight => {
                            y += c.bar_height as i32 + tags;
                            x -= c.border as i32;
                        }
                        Corner::BottomLeft | Corner::BottomCentre | Corner::BottomRight => {
                            y -= c.border as i32;
                        }
                        _ => {}
                    }
                }
                // Only the corners stack: a centred modal is always centred,
                // and a backdrop is always at the origin.
                if matches!(corner, Corner::Caption)
                    && let Some(bottom) = picture_bottom
                {
                    y = caption_y_scaled(bottom, full.1, ov.height as f32, self.config.caption_scale);
                }
                if matches!(corner, Corner::TopLeft | Corner::TopRight) {
                    let cursor = next_y.entry(*corner).or_insert(y);
                    y = *cursor;
                    *cursor += ov.height as i32 + Corner::STACK_GAP;
                }
                // The bottom corners stack the other way, and the cursor has
                // to be the bottom edge rather than the top: each overlay is a
                // different height, so stepping by the last one's height would
                // leave a gap or an overlap depending on which was taller.
                if matches!(corner, Corner::BottomRight) {
                    let bottom = next_y.entry(*corner).or_insert(y + ov.height as i32);
                    y = *bottom - ov.height as i32;
                    *bottom -= ov.height as i32 + Corner::STACK_GAP;
                }

                // A page is the only overlay the user can drag around.
                if let Some(panel) = &self.panel {
                    let (dx, dy) = panel.offset();
                    x += dx;
                    y += dy;
                }
                if let Some(page) = &self.plugins {
                    let (dx, dy) = page.offset();
                    x += dx;
                    y += dy;
                }
                if let Some(page) = &self.keys {
                    let (dx, dy) = page.offset();
                    x += dx;
                    y += dy;
                }
                if Some(ov.id) == filename_id {
                    filename_at = Some((x, y, ov.width as i32, ov.height as i32));
                }
                (ndc_rect_from_pixels(x, y, ov.width, ov.height, full), (x, y))
            })
            .collect();
        // The switcher's button is wherever the filename landed this frame.
        self.filename_at = filename_at;

        let in_mosaic = self.mosaic.is_some();
        // Resolved before the renderer is borrowed mutably.
        let viewport = if in_mosaic { (0.0, 0.0, full.0, full.1) } else { self.image_viewport() };
        let image_rect = match (in_mosaic, &self.shown) {
            (false, Some(s)) => Some(self.view.ndc_rect_in(s.source, viewport, full)),
            _ => None,
        };

        // Read by everything below that is drawn with the picture's rect.
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_turn(self.view.turn);
        }

        // Video's rectangle goes to the YUV pipeline instead of the image one.
        if let (Some(colour), Some(rect)) = (self.video_colour, image_rect) {
            let (rows, offsets) = colour.matrix();
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.set_yuv_colour(rect, rows, offsets);
            }
        }

        // What was found is drawn with the *same* rectangle as the picture, so
        // it letterboxes, pans and zooms with it and needs to know about none
        // of that.
        match (image_rect, self.labels.as_ref().filter(|g| !g.is_empty())) {
            (Some(rect), Some(grid)) => {
                let palette = crate::ml::labels::palette();
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.set_mask(&grid.cells, grid.size, rect, &palette, OUTLINE, FILL);
                }
            }
            _ => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.clear_mask();
                }
            }
        }

        match (image_rect, self.depth_map.as_ref().filter(|m| m.generation == self.generation)) {
            (Some(rect), Some(map)) if self.depth.is_some() => {
                let opacity = self.depth_opacity;
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.set_depth(&map.what.cells, map.what.size, rect, opacity);
                }
            }
            _ => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.clear_depth();
                }
            }
        }

        // A tiled image's tiles, over its overview; none in the grid, where
        // the picture is not drawn.
        let tile_draws = match (in_mosaic, image_rect) {
            (false, Some(rect)) if self.tiling.is_some() => self.plan_tiles(rect),
            _ => Vec::new(),
        };
        if self.tiling.is_some() {
            self.update_tile_badge();
        }

        if let Some(gpu) = self.gpu.as_mut() {
            // The grid draws as an overlay, so the image is skipped there.
            let rect = image_rect.unwrap_or([0.0, 0.0, 0.0, 0.0]);
            let rects: Vec<[f32; 4]> = placements.iter().map(|(r, _)| *r).collect();
            match gpu.render(rect, smooth && !in_mosaic, &tile_draws, &rects) {
                Ok(()) => {}
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    self.exit_error = Some(anyhow::anyhow!("GPU out of memory"));
                    el.exit();
                    return;
                }
                Err(e) => eprintln!("show: frame dropped: {e}"),
            }
        } else if let Some(mut sw) = self.software.take() {
            let view = self.view;
            let result = {
                // In the grid there is no image beneath: the composite is an
                // overlay and covers the window on its own.
                let (pixels, source) = match &self.shown {
                    Some(s) if !in_mosaic => {
                        (s.pixels.as_ref().map(|p| (p.as_slice(), s.tex)), s.source)
                    }
                    _ => (None, (1.0, 1.0)),
                };
                let overlays: Vec<(&Overlay, (i32, i32))> = self
                    .overlay_stack()
                    .into_iter()
                    .zip(placements.iter())
                    .map(|((ov, _), (_, at))| (ov, *at))
                    .collect();
                sw.present(pixels, &view, source, smooth, &overlays, viewport)
            };
            self.software = Some(sw);
            if let Err(e) = result {
                eprintln!("show: software frame dropped: {e:#}");
            }
        } else {
            return;
        }

        // Report once, on the first frame that actually had an image in it:
        // that is the moment the user sees something.
        if !self.presented && self.shown.is_some() {
            self.presented = true;
            timing::mark("win", "first_present");
            first_frame_shown();
            if self.presence.is_none() && !self.args.once {
                let raise = std::sync::Mutex::new(self.proxy.clone());
                let peers = std::sync::Mutex::new(self.proxy.clone());
                self.presence = crate::tray::join(
                    move |fullscreen| {
                        if let Ok(p) = raise.lock() {
                            let _ = p.send_event(UserEvent::Raise(fullscreen));
                        }
                    },
                    move || {
                        if let Ok(p) = peers.lock() {
                            let _ = p.send_event(UserEvent::Peers);
                        }
                    },
                );
                self.describe_to_tray();
            }
            timing::report("cold start");
            if self.args.once {
                el.exit();
                return;
            }
            // The breakdown can only be complete once the frame that shows the
            // image has actually gone out, so it lands on the next one.
            self.build_osd();
            self.build_status_badge();
            self.redraw();
            self.maybe_prefetch_full_res();
            self.prefetch_neighbour();
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    /// Called whenever the event queue empties: our definition of idle.
    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // Browsing away from a playing video has to be able to interrupt it,
        // so the preview is checked before the frame deadline claims the loop.
        self.tick_pending_preview();
        // A film is watched, not touched: keep the screen on while one plays.
        self.awake.set(self.video.is_some() && !self.paused && !self.playback_ended);
        self.update_watch();
        self.tick_subtitles();
        self.tick_backfill();
        self.tick_ml();
        self.tick_depth();
        self.tick_cache_badge();
        self.tick_chrome();
        let volume_until = self.tick_volume_osd();
        let scrubber_until = self.tick_scrubber();
        let transient = match (volume_until, scrubber_until) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let transient = match (transient, self.claim_focus()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let transient = match (transient, self.tick_title_bar()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };

        // Video next: a frame deadline outranks any background chore.
        if let Some(next) = self.tick_video() {
            // Unless the volume readout expires sooner, which is what makes it
            // disappear on its own rather than at the next frame.
            let next = transient.map_or(next, |until| next.min(until));
            el.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(next));
            return;
        }
        if let Some(until) = transient {
            el.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(until));
            return;
        }
        self.tick_idle_thumbnails();

        // A model answering on another thread has no way to wake this one, and
        // a still picture produces no frames to do it either. So while
        // anything is thinking, the loop keeps checking back.
        if self.watcher.is_some() || self.pointer.is_some() || self.finder.is_some() || self.depth.is_some()
        {
            el.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                std::time::Instant::now() + std::time::Duration::from_millis(60),
            ));
            return;
        }

        // Sleep until there is either input or something to do. Without the
        // timeout an idle program would never wake to start the next batch.
        if self.thumb_queue.is_empty() && self.thumb_workers == 0 && self.pending_preview.is_none()
        {
            el.set_control_flow(winit::event_loop::ControlFlow::Wait);
        } else {
            // A settling preview needs checking sooner than idle work does.
            let delay = if self.pending_preview.is_some() { 60 } else { 250 };
            el.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                std::time::Instant::now() + std::time::Duration::from_millis(delay),
            ));
        }
    }

    /// Last chance to write anything still only in memory.
    ///
    /// The volume is saved when its readout fades, which is the moment the
    /// user has settled on a level — but quitting inside that second and a
    /// half would otherwise throw the change away.
    fn exiting(&mut self, _el: &ActiveEventLoop) {
        QUITTING.store(true, std::sync::atomic::Ordering::Relaxed);
        // Gone from the screen first, before anything is saved or torn down:
        // quitting should look instant however long the rest takes.
        if let Some(w) = &self.window {
            w.set_visible(false);
        }
        if self.volume_unsaved {
            self.volume_unsaved = false;
            if let Err(e) = self.config.save() {
                eprintln!("show: could not save the volume: {e:#}");
            }
        }
    }

    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let mut attrs = Window::default_attributes().with_title("show");

        // WM_CLASS, and its Wayland equivalent. The desktop matches this
        // against StartupWMClass in the .desktop file; without it the taskbar
        // treats a running window as unrelated to the launcher, so pinning
        // gives you an icon that opens a second copy instead of raising this
        // one.
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            attrs = WindowAttributesExtX11::with_name(attrs, crate::install::WM_CLASS, "show");
            attrs = WindowAttributesExtWayland::with_name(attrs, crate::install::WM_CLASS, "show");
        }
        // A film is watched rather than inspected, and watching it does not
        // have to mean surrendering the screen. Videos therefore open in a
        // window shaped to the picture, at 90% of whichever screen dimension
        // binds first, unless fullscreen was asked for explicitly.
        if !self.args.fullscreen {
            if let Some((vw, vh)) = self
                .playlist
                .current()
                .filter(|p| crate::video::is_video(p))
                .and_then(crate::video::probe_size)
            {
                // The monitor comes from the loop we are already inside.
                // `profile::detect_screen` cannot be used here: it builds and
                // runs an EventLoop of its own, which fails outright when one
                // is already running on this thread — so it returned None
                // every time and left the window fullscreen.
                let screen = self.args.screen.or_else(|| {
                    el.primary_monitor()
                        .or_else(|| el.available_monitors().next())
                        .map(|m| (m.size().width, m.size().height))
                });
                if let Some(screen) = screen {
                    let size = window_size_for((vw, vh), screen);
                    attrs = attrs
                        .with_inner_size(winit::dpi::PhysicalSize::new(size.0, size.1));
                    self.fullscreen = false;
                }
            }
        }
        if self.fullscreen {
            attrs = attrs.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }
        // The desktop's title bar is replaced by one of our own, so it is not
        // asked for. Everything it did — moving, resizing, closing — has to be
        // provided here instead; see the mouse handling.
        attrs = attrs.with_decorations(false);
        // Handing the token over is what ends the launch sequence, and with it
        // the spinning cursor the desktop shows while waiting.
        if let Some(Some(token)) = ACTIVATION_TOKEN.get() {
            attrs = attrs.with_activation_token(ActivationToken::from_raw(token.clone()));
        }

        let window = {
            let _s = timing::span("win", "create");
            match el.create_window(attrs) {
                Ok(w) => Arc::new(w),
                Err(e) => {
                    self.exit_error = Some(e.into());
                    el.exit();
                    return;
                }
            }
        };
        self.window = Some(window.clone());

        // Ask for the keyboard. A fullscreen window is given focus by every
        // window manager; an undecorated windowed one is not always, and a
        // film that ignores every key until it is clicked looks broken.
        //
        // The request cannot be made here: X11 drops it on the floor until the
        // window is mapped, which cannot happen before this function returns
        // and the event loop starts pumping. So it is owed, not made, and paid
        // off from `draw` once frames are running.
        self.focus_until = Some(std::time::Instant::now() + FOCUS_GRACE);

        if self.args.software_first_frame {
            // Order matters: get the CPU able to draw before asking the GPU to
            // start, so a fast decode is never blocked behind driver startup.
            match Software::new(window.clone()) {
                Ok(sw) => self.software = Some(sw),
                Err(e) => eprintln!("show: no software surface ({e:#}); waiting for the GPU"),
            }
            spawn_gpu_init(&self.proxy, self.args.adapter.clone(), self.args.backends);
        } else {
            match Gpu::new(window, self.args.adapter.as_deref(), self.args.vsync, self.args.backends) {
                Ok(gpu) => self.gpu = Some(gpu),
                Err(e) => {
                    self.exit_error = Some(e);
                    el.exit();
                    return;
                }
            }
        }

        // The decode may already have finished while we were doing all that.
        if let Some((loaded, is_full)) = self.pending.take() {
            self.install(loaded, is_full);
        }
        // A video was deliberately not started before the window existed.
        if self.shown.is_none()
            && self.playlist.current().is_some_and(crate::video::is_video)
        {
            self.start_current();
        }

        // A build newer than the installed copy opens the menu on the redeploy
        // row, because that is the one moment the menu is what you came for.
        // Every other state is left to the one-line hint in the readout.
        if crate::install::should_open_setup(crate::install::state()) {
            self.open_panel();
        }
        self.redraw();
    }

    fn user_event(&mut self, el: &ActiveEventLoop, event: UserEvent) {
        let (generation, result, is_full) = match event {
            UserEvent::GpuReady(core) => {
                match *core {
                    Ok(core) => self.adopt_gpu(core),
                    // The software path is already drawing; losing the GPU
                    // costs us zoom smoothness, not the picture.
                    Err(e) => eprintln!("show: GPU unavailable, staying on the CPU: {e:#}"),
                }
                return;
            }
            UserEvent::Folder(list) => {
                self.adopt_folder(*list);
                return;
            }
            UserEvent::Raise(fullscreen) => {
                // Arrived at from another window's strip, which was fullscreen:
                // so is this, or stepping between windows would not stay so.
                if fullscreen && !self.fullscreen {
                    self.toggle_fullscreen();
                }
                if let Some(w) = &self.window {
                    w.set_minimized(false);
                    w.focus_window();
                }
                self.redraw();
                return;
            }
            UserEvent::FolderChanged => {
                self.rescan_folder();
                return;
            }
            UserEvent::Neighbour(path, result) => {
                // Kept only if it is still the one wanted, and not so big that
                // holding it on spec would crowd out what is on screen.
                if let (Some((wanted, slot)), Ok(loaded)) = (self.neighbour.as_mut(), *result)
                    && *wanted == path
                    && loaded.decoded.rgba.len() as u64 <= SPECULATIVE_FULL_RES_BUDGET
                {
                    *slot = Some(loaded);
                }
                return;
            }
            UserEvent::Peers => {
                self.build_peer_strip();
                self.redraw();
                return;
            }
            UserEvent::Uploaded(generation, uploaded) => {
                // A newer picture has replaced the one this was for.
                if generation != self.generation {
                    return;
                }
                if let (Some(gpu), Some(shown)) = (self.gpu.as_mut(), self.shown.as_mut()) {
                    let factor = uploaded.downscaled_by;
                    gpu.adopt(*uploaded);
                    shown.adequate_zoom /= factor as f32;
                }
                self.redraw();
                return;
            }
            UserEvent::Tile(generation, key, tile) => {
                if generation != self.generation {
                    return;
                }
                match *tile {
                    Ok(tile) => {
                        if let (Some(gpu), Some(t)) = (self.gpu.as_mut(), self.tiling.as_ref()) {
                            gpu.upload_tile(key, &tile);
                            gpu.evict_tiles(&t.in_use);
                        }
                        self.redraw();
                    }
                    Err(e) => eprintln!("show: tile {key:?}: {e:#}"),
                }
                return;
            }
            UserEvent::Geo(path, sources) => {
                // Only if it is still the image on screen.
                if self.geo_for.as_ref() == Some(&path) && !sources.is_empty() {
                    let labels: Vec<&str> = sources.iter().map(|s| s.label()).collect();
                    let icon = osd::geo_icon(osd::SCALE_PANEL);
                    let text = osd::render_scaled(&[osd::Row::good(labels.join(" + "), "")], osd::SCALE_PANEL);
                    self.geo_badge = Some(osd::hstack(&[&icon, &text], 6));
                    self.redraw();
                }
                return;
            }
            UserEvent::FileInfo(batch, last) => {
                self.file_info.extend(batch);
                if last {
                    self.info_scan = None;
                }
                if self.browser.is_some() {
                    self.build_browser_overlay();
                    self.redraw();
                }
                return;
            }
            UserEvent::ThumbnailWritten(_) => {
                self.thumb_workers = self.thumb_workers.saturating_sub(1);
                // Keep going while nothing else needs the machine.
                self.tick_idle_thumbnails();
                return;
            }
            UserEvent::PluginProgress(id, status) => {
                if let Some(page) = self.plugins.as_mut() {
                    page.set_status(id, status);
                    self.redraw();
                }
                return;
            }
            UserEvent::PluginDone(id, result) => {
                if let Some(page) = self.plugins.as_mut() {
                    match *result {
                        Ok(()) => {
                            page.refresh();
                            page.note(format!("{} installed", id.id()));
                        }
                        Err(e) => {
                            page.set_status(id, crate::plugins::Status::Failed("FAILED".into()));
                            // The whole reason rather than a summary: a digest
                            // mismatch and a dead connection need different
                            // things done about them.
                            page.note(format!("{e:#}"));
                        }
                    }
                    self.redraw();
                }
                return;
            }
            UserEvent::AnalysisProgress(generation, done, total) => {
                if generation == self.generation {
                    self.build_analysis_overlay(done, total);
                    self.redraw();
                }
                return;
            }
            UserEvent::AnalysisDone(generation, hashes) => {
                self.analysing = false;
                self.analysis = None;
                if generation == self.generation {
                    let order = crate::similarity::order_by_similarity(&hashes);
                    let found = hashes.iter().filter(|h| h.is_some()).count();
                    eprintln!("show: compared {found} of {} images", hashes.len());

                    // The result is an ordering of the grid, so the grid is
                    // where it is shown.
                    if self.mosaic.is_none() {
                        self.toggle_mosaic();
                    }
                    if let Some(mosaic) = self.mosaic.as_mut() {
                        mosaic.set_order(order);
                        // Kept, so selecting an image can highlight what it
                        // matches and score everything against it.
                        mosaic.set_hashes(*hashes);
                    }
                    self.request_thumbs();
                }
                self.redraw();
                return;
            }
            UserEvent::Thumb(generation, index, result) => {
                // A grid cell that arrives after the grid closed, or after the
                // folder changed, has nowhere to go.
                if generation == self.generation {
                    if let Some(mosaic) = self.mosaic.as_mut() {
                        match *result {
                            Ok(loaded) => {
                                let d = loaded.decoded;
                                mosaic.insert(
                                    index,
                                    Thumb { rgba: d.rgba, width: d.width, height: d.height },
                                );
                            }
                            // The cell stays a placeholder; one unreadable file
                            // should not stop the rest of the grid filling in.
                            Err(e) => {
                                eprintln!("show: thumbnail failed: {e:#}");
                                mosaic.mark_failed(index);
                            }
                        }
                        self.build_status_badge();
                        self.redraw();
                        self.request_thumbs();
                    }
                }
                return;
            }
            UserEvent::Preview(g, r) => (g, r, false),
            UserEvent::Full(g, r) => (g, r, true),
        };

        // The user has moved on; this decode is for an image no longer shown.
        if generation != self.generation {
            return;
        }

        match *result {
            Ok(loaded) => {
                self.install(loaded, is_full);
                // The picture is in: now the rest of its folder, and whether
                // it has a place on the map.
                self.start_folder_scan();
                self.start_geo_check();
                self.describe_to_tray();
            }
            Err(e) if is_full => {
                // The preview is still on screen and still correct; a failed
                // upgrade is not worth tearing the window down for.
                eprintln!("show: full-resolution load failed, keeping preview: {e:#}");
                self.full_res = FullRes::NotNeeded;
            }
            Err(e) => {
                if self.leaving_mosaic {
                    self.leaving_mosaic = false;
                    self.mosaic = None;
                    self.mosaic_composite = None;
                }
                // With a folder to move through, one unreadable file is a
                // message, not a reason to quit.
                let where_ = self.playlist.current().map(|p| p.display().to_string());
                eprintln!("show: {}: {e:#}", where_.as_deref().unwrap_or("(no file)"));
                // Nothing to show, so nothing to wait for: list the folder
                // now, since whether there is one decides what happens next.
                if let Some(start) = self.folder_to_scan.take() {
                    self.adopt_folder(Self::scan_folder(&start, self.config.sort));
                }
                if self.shown.is_none() && self.playlist.len() == 1 {
                    self.exit_error = Some(e);
                    el.exit();
                }
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Anything the user does postpones the background work.
        if !matches!(event, WindowEvent::RedrawRequested) {
            self.last_activity = std::time::Instant::now();
        }

        match event {
            WindowEvent::CloseRequested => el.exit(),

            WindowEvent::ModifiersChanged(mods) => {
                self.shift_held = mods.state().shift_key();
                self.ctrl_held = mods.state().control_key();
            }

            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }
                if let Some(sw) = self.software.as_mut() {
                    if let Err(e) = sw.resize(size.width, size.height) {
                        eprintln!("show: software resize failed: {e:#}");
                    }
                }
                if self.view.fitting {
                    self.fit_current();
                }
                self.redraw();
            }

            // Off the window, nothing is under the pointer: no cell to name.
            WindowEvent::CursorLeft { .. } => {
                if !self.dragging && !self.scrubbing {
                    self.cursor = (-1.0, -1.0);
                    // The pointer has gone, and so do the things it brought
                    // up: the filename, the seek bar and the window list.
                    self.pointer_inside = false;
                    self.scrubber = None;
                    self.scrubber_at = None;
                    self.peer_list_open = false;
                    self.update_hover();
                    self.redraw();
                }
            }

            WindowEvent::CursorEntered { .. } => {
                self.pointer_inside = true;
                self.redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                let now = (position.x as f32, position.y as f32);
                if !self.pointer_inside {
                    self.pointer_inside = true;
                    self.redraw();
                }
                // A real movement, not a hand settling on the mouse.
                if (now.0 - self.motion_from.0).hypot(now.1 - self.motion_from.1) > TITLE_BAR_NUDGE {
                    self.motion_from = now;
                    self.last_motion = std::time::Instant::now();
                    if !self.title_bar_was {
                        self.title_bar_was = true;
                        self.redraw();
                    }
                }
                if self.scrubbing {
                    self.cursor = now;
                    // Clamped along the bar rather than requiring the pointer
                    // to stay on it: dragging past either end should pin to
                    // that end, not drop the drag.
                    if let Some((x, _, w, _)) = self.scrubber_rect()
                        && let Some(duration) = self.video.as_ref().and_then(|p| p.info.duration)
                    {
                        let (track_x, track_w) = osd::scrubber_track(w, osd::SCALE_COMPACT);
                        let fraction = Self::track_fraction(now.0, x, track_x, track_w);
                        self.scrub_to(duration.mul_f32(fraction), false);
                    }
                    return;
                }
                if self.dragging {
                    let (dx, dy) = (now.0 - self.cursor.0, now.1 - self.cursor.1);
                    if let Some(page) = self.keys.as_mut() {
                        page.pan(dx as i32, dy as i32);
                        self.redraw();
                    } else if let Some(page) = self.plugins.as_mut() {
                        page.pan(dx as i32, dy as i32);
                        self.redraw();
                    } else if let Some(panel) = self.panel.as_mut() {
                        panel.pan(dx as i32, dy as i32);
                        self.redraw();
                    } else if self.drags_window() {
                        // Windowed and without shift: a drag moves the window.
                        // Only once it is a drag — past the click slop — so a
                        // click still pauses.
                        if !was_click(self.press_at, now) {
                            self.dragging = false;
                            self.press_at = None;
                            if let Some(w) = &self.window {
                                let _ = w.drag_window();
                            }
                        }
                    } else if let Some(source) = self.shown.as_ref().map(|s| s.source) {
                        self.view.pan(dx, dy, source, self.win_size());
                        self.redraw();
                    }
                }
                // Reaching for the mouse is how you ask a player where you
                // are, so the seek bar appears on any real movement and fades
                // once the pointer settles.
                if self.video.is_some() && (now.0 - self.cursor.0).abs() + (now.1 - self.cursor.1).abs() > 1.0
                {
                    self.show_scrubber();
                    self.redraw();
                }
                self.cursor = now;
                if !self.dragging {
                    self.update_cursor_icon();
                }
                if self.update_hover() {
                    self.redraw();
                }
            }

            // Right toggles autoscaling, and it sticks: the next image is
            // fitted the same way.
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                let on = !self.view.autoscale;
                self.config.autoscale = on;
                if let Err(e) = self.config.save() {
                    eprintln!("show: could not save settings: {e:#}");
                }
                if let Some(source) = self.shown.as_ref().map(|s| s.source) {
                    self.view.set_autoscale(on, source, self.win_size());
                } else {
                    self.view.autoscale = on;
                }
                self.build_status_badge();
                self.redraw();
            }

            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                // The window switcher, taken before anything under it sees the
                // press: its button drops the list, a row in the list is that
                // window, still fullscreen, and a press anywhere else while
                // the list is down only closes it.
                if state == ElementState::Pressed {
                    match self.peer_hit(self.cursor) {
                        PeerHit::Nothing => {}
                        PeerHit::Button => {
                            self.peer_list_open = !self.peer_list_open;
                            self.redraw();
                            return;
                        }
                        PeerHit::Window(name) => {
                            self.peer_list_open = false;
                            if let Some(presence) = &self.presence
                                && name != presence.own_name()
                            {
                                presence.activate(&name, self.fullscreen);
                            }
                            self.redraw();
                            return;
                        }
                        PeerHit::Away => {
                            self.peer_list_open = false;
                            self.redraw();
                            return;
                        }
                    }
                }
                // Releasing anywhere ends a drag of the handle, including off
                // the bar — which is where a fast drag usually ends up.
                if state == ElementState::Released && self.scrubbing {
                    self.scrubbing = false;
                    self.press_at = None;
                    let to = self.position;
                    self.scrub_to(to, true);
                    // Done: a later click on this same spot is a new seek.
                    self.scrub_exact = None;
                    self.dragging = false;
                    return;
                }

                // A click on the picture stops and starts a film, which is
                // what a click on a video means everywhere. Decided on release
                // and only if the pointer stayed put: a pan across the frame
                // is a drag, not a click, and must not pause it. Waiting out
                // the double-click interval to be certain would put a visible
                // delay on the commonest action in the player, so a double
                // click simply toggles twice and lands back where it started.
                if state == ElementState::Released {
                    let clicked = was_click(self.press_at.take(), self.cursor);
                    let on_picture = self.panel.is_none()
                        && self.mosaic.is_none()
                        && !(self.browser.is_some() && self.cursor_over_browser());

                    // Ctrl means "what is *that*" — a different question from
                    // "what is in this picture", answered by a different model,
                    // and it works whether or not anything is being detected.
                    if clicked && on_picture && self.ctrl_held {
                        self.dragging = false;
                        self.cut_out_at(self.cursor);
                        return;
                    }

                    // With things outlined on screen, clicking one means that
                    // one — it is visibly a target. Clicking anywhere else
                    // still means what it always did.
                    if clicked && on_picture && self.watcher.is_some() && self.follow_at(self.cursor)
                    {
                        self.dragging = false;
                        return;
                    }
                    if clicked && on_picture && self.video.is_some() {
                        self.dragging = false;
                        self.toggle_pause();
                        return;
                    }
                }
                if state == ElementState::Pressed {
                    self.press_at = Some(self.cursor);
                }

                self.dragging = state == ElementState::Pressed;
                // With a page up a left drag moves it; the double-click and
                // grid handling below belong to the image underneath.
                if self.panel.is_some() || self.plugins.is_some() || self.keys.is_some() {
                    return;
                }

                // The window's own frame comes first: it stands where the
                // desktop's decorations would have been, and has to behave the
                // way they did.
                if state == ElementState::Pressed
                    && let Some(action) = self.chrome_hit(self.cursor)
                {
                    self.dragging = false;
                    self.press_at = None;
                    match action {
                        ChromeHit::Close => el.exit(),
                        ChromeHit::Move => {
                            if let Some(w) = &self.window {
                                let _ = w.drag_window();
                            }
                        }
                        ChromeHit::Resize(direction) => {
                            if let Some(w) = &self.window {
                                let _ = w.drag_resize_window(direction);
                            }
                        }
                    }
                    return;
                }

                // The seek bar takes precedence over the picture beneath it,
                // so a press on the bar scrubs rather than starting a pan.
                if state == ElementState::Pressed
                    && let Some(to) = self.scrubber_hit(self.cursor)
                {
                    self.scrubbing = true;
                    self.dragging = false;
                    self.last_click = None;
                    self.scrub_to(to, true);
                    return;
                }

                if state == ElementState::Pressed {
                    // A double click is two presses close in both time and
                    // place; without the distance test a slow drag would count.
                    const INTERVAL: std::time::Duration = std::time::Duration::from_millis(400);
                    const SLOP: f32 = 8.0;
                    let now = std::time::Instant::now();
                    let doubled = self.last_click.is_some_and(|(t, at)| {
                        now.duration_since(t) < INTERVAL
                            && (at.0 - self.cursor.0).abs() < SLOP
                            && (at.1 - self.cursor.1).abs() < SLOP
                    });

                    // A click in the panel opens a folder. On an image it does
                    // nothing: opening on a single click makes every stray
                    // click a decode.
                    if self.browser.is_some() && self.cursor_over_browser() {
                        let is_folder = self
                            .browser
                            .as_ref()
                            .and_then(|b| b.selected_entry())
                            .is_some_and(|e| !e.is_image());
                        if is_folder {
                            self.browser_activate();
                        }
                        self.last_click = None;
                        return;
                    }

                    if self.mosaic.is_some() {
                        let win = self.win_size();
                        let count = self.mosaic.as_ref().unwrap().count(self.playlist.len());
                        let hit =
                            self.mosaic.as_ref().unwrap().hit_test(self.cursor, win, count);
                        if let Some(index) = hit {
                            self.mosaic.as_mut().unwrap().set_selected(index, win, count);
                            if doubled {
                                self.close_mosaic(true);
                                return;
                            }
                            self.request_thumbs();
                            self.redraw();
                        }
                        // A double click fits once, without changing the setting.
                    if doubled {
                        self.last_click = None;
                        if let Some(source) = self.shown.as_ref().map(|s| s.source) {
                            self.view.fit(source, self.win_size());
                            self.build_status_badge();
                            self.redraw();
                        }
                    } else {
                        self.last_click = Some((now, self.cursor));
                    }
                        return;
                    }

                    // A double click fits once, without changing the setting.
                    if doubled {
                        self.last_click = None;
                        if let Some(source) = self.shown.as_ref().map(|s| s.source) {
                            self.view.fit(source, self.win_size());
                            self.build_status_badge();
                            self.redraw();
                        }
                    } else {
                        self.last_click = Some((now, self.cursor));
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // Line deltas come from wheels, pixel deltas from touchpads;
                // normalise both to notches.
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                if notches == 0.0 {
                    return;
                }

                // Ctrl zooms the picture; with shift the window grows with it.
                if self.ctrl_held && self.shift_held {
                    self.resize_window_by(ZOOM_STEP.powf(notches));
                    return;
                }

                if self.keys.is_some() {
                    let win = self.win_size();
                    self.keys.as_mut().unwrap().zoom(notches, win);
                    self.redraw();
                    return;
                }
                if self.plugins.is_some() {
                    let win = self.win_size();
                    self.plugins.as_mut().unwrap().zoom(notches, win);
                    self.redraw();
                    return;
                }
                if self.panel.is_some() {
                    let win = self.win_size();
                    self.panel.as_mut().unwrap().zoom(notches, win);
                    self.redraw();
                    return;
                }

                // Over the browser panel the wheel belongs to the list.
                if self.browser.is_some() && self.cursor_over_browser() {
                    if self.ctrl_held {
                        self.browser_scale = (self.browser_scale
                            + notches.signum() * osd::SCALE_STEP)
                            .clamp(osd::SCALE_MIN, osd::SCALE_MAX);
                        // The picture's region changed with the panel width.
                        self.refit_for_viewport();
                    } else {
                        let steps = -notches.round() as isize;
                        self.browser.as_mut().unwrap().move_selection(steps.clamp(-4, 4));
                        self.browser_selection_changed();
                        return;
                    }
                    self.build_browser_overlay();
                    self.redraw();
                    return;
                }

                // Over a film the wheel is the volume knob, because that is
                // what is wanted mid-playback. Ctrl is the usual modifier for
                // zooming and keeps it available for looking closely at a
                // frame, which is exactly when it is wanted.
                if self.video.is_some() && self.browser.is_none() {
                    if self.ctrl_held {
                        self.zoom_by(ZOOM_STEP.powf(notches), self.cursor);
                    } else if self.video.as_ref().is_some_and(|p| p.audio().is_some()) {
                        self.adjust_volume(notches);
                    } else {
                        // A silent film has no volume to set, so the plain
                        // wheel may as well zoom rather than do nothing.
                        self.zoom_by(ZOOM_STEP.powf(notches), self.cursor);
                    }
                    return;
                }

                if self.mosaic.is_some() {
                    let win = self.win_size();
                    let count = self.mosaic.as_ref().unwrap().count(self.playlist.len());
                    let ctrl = self.ctrl_held;
                    let mosaic = self.mosaic.as_mut().unwrap();
                    // Plain wheel scrolls, which is what a wheel does over a
                    // list. Ctrl resizes the cells, the usual "zoom" modifier.
                    if ctrl {
                        mosaic.resize_cells(notches, win, count);
                    } else {
                        mosaic.scroll_by(-notches * WHEEL_SCROLL_PIXELS, win, count);
                    }
                    self.request_thumbs();
                    self.redraw();
                    return;
                }

                self.zoom_by(ZOOM_STEP.powf(notches), self.cursor);
            }

            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && self.ctrl_held
                    && self.typing.is_none()
                    && matches!(event.logical_key.as_ref(), Key::Character("c") | Key::Character("C")) =>
            {
                self.copy_to_clipboard();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && self.ctrl_held
                    && self.typing.is_none()
                    && event.logical_key == Key::Named(NamedKey::Tab) =>
            {
                // Ctrl+Tab steps through the open windows, whatever this one
                // is showing: a page, the grid or a picture.
                self.step_window(self.shift_held);
            }
            // Escape with the window list down closes the list, and nothing
            // more: it must not also leave the grid or quit.
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && self.peer_list_open
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                self.peer_list_open = false;
                self.redraw();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                // Typing owns the keyboard outright, and is checked before
                // everything else: every letter of a phrase is also a shortcut
                // — `f` is fullscreen, `q` quits — and a prompt that fired
                // them while being filled in would be unusable.
                if self.typing.is_some() {
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Escape) => {
                            self.typing = None;
                            self.build_prompt();
                            self.redraw();
                        }
                        Key::Named(NamedKey::Enter) => {
                            let phrase = self.typing.take().unwrap_or_default();
                            self.build_prompt();
                            self.ask_for(&phrase);
                        }
                        Key::Named(NamedKey::Backspace) => {
                            if let Some(text) = self.typing.as_mut() {
                                text.pop();
                            }
                            self.build_prompt();
                            self.redraw();
                        }
                        Key::Named(NamedKey::Space) => {
                            if let Some(text) = self.typing.as_mut() {
                                // Never at the front and never two together: a
                                // phrase is words, and the tokenizer would drop
                                // the difference anyway.
                                if !text.is_empty() && !text.ends_with(' ') {
                                    text.push(' ');
                                }
                            }
                            self.build_prompt();
                            self.redraw();
                        }
                        Key::Character(c) => {
                            if let Some(text) = self.typing.as_mut() {
                                // A prompt is a short phrase. The model takes
                                // 32 tokens and a token is often a whole word,
                                // so this is generous rather than tight.
                                if text.chars().count() < 64 {
                                    text.push_str(c);
                                }
                            }
                            self.build_prompt();
                            self.redraw();
                        }
                        _ => {}
                    }
                    return;
                }

                // The browser owns the arrows while it is open.
                if self.browser.is_some()
                    && self.confirming_delete.is_none()
                    && self.panel.is_none()
                    && self.keys.is_none()
                {
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Escape)
                        | Key::Character("q")
                        | Key::Character("b")
                        | Key::Character("B") => {
                            self.close_browser();
                            return;
                        }
                        Key::Named(NamedKey::ArrowUp) => {
                            self.browser.as_mut().unwrap().move_selection(-1);
                            self.browser_selection_changed();
                            return;
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            self.browser.as_mut().unwrap().move_selection(1);
                            self.browser_selection_changed();
                            return;
                        }
                        Key::Named(NamedKey::PageUp) => {
                            self.browser.as_mut().unwrap().move_selection(-10);
                            self.browser_selection_changed();
                            return;
                        }
                        Key::Named(NamedKey::PageDown) => {
                            self.browser.as_mut().unwrap().move_selection(10);
                            self.browser_selection_changed();
                            return;
                        }
                        Key::Named(NamedKey::Enter) => {
                            self.browser_activate();
                            return;
                        }
                        Key::Named(NamedKey::Backspace) => {
                            self.browser_up();
                            return;
                        }
                        // Delete still deletes, and still asks first.
                        Key::Named(NamedKey::Delete) => {
                            self.request_delete();
                            return;
                        }
                        Key::Named(NamedKey::F1) | Key::Character("?") => {
                            self.open_panel();
                            return;
                        }
                        _ => {}
                    }
                }

                // A pending delete answers only y/n: no navigating away from
                // the question, and no deleting a different file by accident.
                if self.confirming_delete.is_some() {
                    match event.logical_key.as_ref() {
                        Key::Character("y") | Key::Character("Y") => self.perform_delete(),
                        _ => self.cancel_delete(),
                    }
                    return;
                }

                // The browser owns the arrows while it is open.
                if self.browser.is_some()
                    && self.confirming_delete.is_none()
                    && self.panel.is_none()
                    && self.keys.is_none()
                {
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Escape) | Key::Character("q") => {
                            self.close_browser();
                            return;
                        }
                        Key::Named(NamedKey::ArrowUp) => {
                            self.browser.as_mut().unwrap().move_selection(-1);
                            self.browser_selection_changed();
                            return;
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            self.browser.as_mut().unwrap().move_selection(1);
                            self.browser_selection_changed();
                            return;
                        }
                        Key::Named(NamedKey::PageUp) => {
                            self.browser.as_mut().unwrap().move_selection(-10);
                            self.browser_selection_changed();
                            return;
                        }
                        Key::Named(NamedKey::PageDown) => {
                            self.browser.as_mut().unwrap().move_selection(10);
                            self.browser_selection_changed();
                            return;
                        }
                        Key::Named(NamedKey::Enter) => {
                            self.browser_activate();
                            return;
                        }
                        Key::Named(NamedKey::Backspace) => {
                            self.browser_up();
                            return;
                        }
                        // Delete still deletes, and still asks first.
                        Key::Named(NamedKey::Delete) => {
                            self.request_delete();
                            return;
                        }
                        Key::Named(NamedKey::F1) | Key::Character("?") => {
                            self.open_panel();
                            return;
                        }
                        _ => {}
                    }
                }

                // A pending delete answers only y/n: no navigating away from
                // the question, and no deleting a different file by accident.
                if self.confirming_delete.is_some() {
                    match event.logical_key.as_ref() {
                        Key::Character("y") | Key::Character("Y") => self.perform_delete(),
                        _ => self.cancel_delete(),
                    }
                    return;
                }

                // The key page owns the keyboard while it is open, and has
                // to: every key it describes would otherwise fire underneath
                // it while it was being read.
                if self.keys.is_some() {
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Escape) | Key::Character("h") | Key::Character("H") => {
                            self.close_keys()
                        }
                        Key::Character("q") => el.exit(),
                        Key::Named(NamedKey::ArrowUp) => {
                            self.keys.as_mut().unwrap().scroll(-1.0);
                            self.redraw();
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            self.keys.as_mut().unwrap().scroll(1.0);
                            self.redraw();
                        }
                        Key::Named(NamedKey::PageUp) => {
                            self.keys.as_mut().unwrap().scroll(-5.0);
                            self.redraw();
                        }
                        Key::Named(NamedKey::PageDown) => {
                            self.keys.as_mut().unwrap().scroll(5.0);
                            self.redraw();
                        }
                        _ => {}
                    }
                    return;
                }

                // The plugins page owns the keyboard while it is open.
                if self.plugins.is_some() {
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Escape) | Key::Named(NamedKey::F8) => {
                            self.plugins = None;
                            self.redraw();
                        }
                        Key::Character("q") => el.exit(),
                        Key::Named(NamedKey::ArrowUp) => {
                            self.plugins.as_mut().unwrap().move_selection(-1);
                            self.redraw();
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            self.plugins.as_mut().unwrap().move_selection(1);
                            self.redraw();
                        }
                        Key::Named(dir @ (NamedKey::ArrowLeft | NamedKey::ArrowRight)) => {
                            let steps = if dir == NamedKey::ArrowLeft { -1 } else { 1 };
                            let mut config = self.config.clone();
                            self.plugins.as_mut().unwrap().adjust(steps, &mut config);
                            self.config = config;
                            self.save_settings();
                            // Live: the next frame is looked at on the new
                            // schedule, rather than the next film.
                            let (hz, sharpen) = (self.config.look_hz, self.config.sharpen);
                            let limits = self.thresholds();
                            self.plugins.as_mut().unwrap().set_settings(hz, sharpen, limits);
                            if let Some(watcher) = self.watcher.as_ref() {
                                watcher.set_rate(hz);
                                watcher.set_thresholds(limits);
                            }
                            if !sharpen {
                                self.sharp.clear();
                                self.sharp_of = None;
                            }
                            self.build_status_badge();
                            self.redraw();
                        }
                        Key::Named(NamedKey::Enter) | Key::Named(NamedKey::Space) => {
                            self.install_plugin()
                        }
                        Key::Named(NamedKey::Delete) => self.remove_plugin(),
                        Key::Character("v") | Key::Character("V") => self.verify_plugin(),
                        _ => {}
                    }
                    return;
                }

                // The subtitle picker owns the keyboard while it is open, and
                // is checked before everything else so its arrows choose an
                // entry rather than seeking the film underneath.
                if self.menu.is_some() {
                    let count = self.menu.as_ref().map_or(0, |m| m.entries.len());
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Escape) | Key::Named(NamedKey::Alt) => {
                            self.menu = None;
                            self.redraw();
                        }
                        Key::Named(NamedKey::ArrowUp) | Key::Named(NamedKey::ArrowDown) => {
                            let down = event.logical_key == Key::Named(NamedKey::ArrowDown);
                            if let Some(menu) = self.menu.as_mut()
                                && count > 0
                            {
                                // Wraps, because a list this short is quicker
                                // to reach round the end than back through.
                                menu.selected = if down {
                                    (menu.selected + 1) % count
                                } else {
                                    (menu.selected + count - 1) % count
                                };
                            }
                            self.build_menu();
                            self.redraw();
                        }
                        Key::Named(NamedKey::Enter) | Key::Named(NamedKey::Space) => {
                            self.menu_activate()
                        }
                        _ => {}
                    }
                    return;
                }

                // While the panel is up it owns the keyboard, so arrow keys
                // move its selection rather than stepping through the folder.
                if self.panel.is_some() {
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Escape) | Key::Named(NamedKey::F1) => {
                            self.close_panel()
                        }
                        // The row above says H, so H has to work from
                        // here. The page opens over this one rather than
                        // instead of it, so closing it comes back to where
                        // the reader was.
                        Key::Character("h") | Key::Character("H") => self.open_keys(),
                        Key::Character("q") => el.exit(),
                        Key::Named(NamedKey::ArrowUp) => {
                            self.panel.as_mut().unwrap().move_selection(-1);
                            self.redraw();
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            self.panel.as_mut().unwrap().move_selection(1);
                            self.redraw();
                        }
                        Key::Named(dir @ (NamedKey::ArrowLeft | NamedKey::ArrowRight)) => {
                            let steps = if dir == NamedKey::ArrowLeft { -1 } else { 1 };
                            let mut config = self.config.clone();
                            self.panel.as_mut().unwrap().adjust(steps, &mut config);
                            self.config = config;
                            self.save_settings();
                                                self.redraw();
                        }
                        Key::Named(NamedKey::Enter) | Key::Named(NamedKey::Space) => {
                            let mut config = self.config.clone();
                            self.panel.as_mut().unwrap().activate(&mut config);
                            self.config = config;
                            self.save_settings();
                                                self.redraw();
                        }
                        _ => {}
                    }
                    return;
                }

                // The grid has its own navigation: arrows move between cells
                // in two dimensions, not between images in one.
                if self.mosaic.is_some() {
                    let win = self.win_size();
                    let count = self.mosaic.as_ref().unwrap().count(self.playlist.len());
                    let mosaic = self.mosaic.as_mut().unwrap();
                    match event.logical_key.as_ref() {
                        Key::Character("q") => el.exit(),
                        Key::Named(NamedKey::Escape) | Key::Named(NamedKey::F2) => {
                            // A match group is a detour: escape steps back out
                            // of it before it leaves the grid altogether.
                            if !self.mosaic.as_mut().unwrap().leave_match_group() {
                                self.close_mosaic(false);
                            } else {
                                self.request_thumbs();
                                self.redraw();
                            }
                            return;
                        }
                        Key::Named(NamedKey::Enter) => {
                            // In compare mode, enter drills into the matches so
                            // they can be looked at side by side and enlarged.
                            // With no group to enter, it opens the image.
                            let entered = self
                                .mosaic
                                .as_mut()
                                .unwrap()
                                .enter_match_group(win, self.playlist.len());
                            if entered {
                                self.request_thumbs();
                                self.redraw();
                            } else {
                                self.close_mosaic(true);
                            }
                            return;
                        }
                        Key::Named(NamedKey::F6) => {
                            self.start_analysis();
                            return;
                        }
                        Key::Named(NamedKey::F1) | Key::Character("?") => {
                            self.open_panel();
                            return;
                        }
                        Key::Character("h") | Key::Character("H") => {
                            self.open_keys();
                            return;
                        }
                        Key::Character("r") | Key::Character("R") => {
                            self.toggle_recursive();
                            return;
                        }
                        Key::Character("s") | Key::Character("S") => {
                            self.cycle_sort();
                            return;
                        }
                        Key::Named(NamedKey::ArrowLeft) => mosaic.move_selection(-1, win, count),
                        Key::Named(NamedKey::ArrowRight) => mosaic.move_selection(1, win, count),
                        Key::Named(NamedKey::ArrowUp) => {
                            mosaic.move_rows(-1, win, count)
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            mosaic.move_rows(1, win, count)
                        }
                        Key::Named(NamedKey::PageUp) => {
                            mosaic.scroll_by(-win.1, win, count)
                        }
                        Key::Named(NamedKey::PageDown) => mosaic.scroll_by(win.1, win, count),
                        Key::Named(NamedKey::Home) => mosaic.set_selected(0, win, count),
                        Key::Named(NamedKey::End) => {
                            mosaic.set_selected(count.saturating_sub(1), win, count)
                        }
                        _ => return,
                    }
                    self.request_thumbs();
                    self.redraw();
                    return;
                }

                let win = self.win_size();
                let centre = (win.0 / 2.0, win.1 / 2.0);
                let source = self.shown.as_ref().map(|s| s.source);

                match event.logical_key.as_ref() {
                    // Before the arm that quits, which also matches escape:
                    // arms are tried in order, so this one has to come first
                    // or it never fires at all. Escape means "never mind"
                    // everywhere else here, and outlines on screen are the
                    // nearest thing to mind.
                    Key::Named(NamedKey::Escape) if !self.found.is_empty() => {
                        self.forget_found();
                        self.notify("CLEARED");
                    }

                    // Back to the grid this image was opened from, if it was.
                    Key::Named(NamedKey::Escape) if self.grid_behind.is_some() => {
                        self.return_to_grid();
                    }

                    Key::Named(NamedKey::Escape) | Key::Character("q") => el.exit(),

                    Key::Character("f") | Key::Named(NamedKey::F11) => {
                        self.toggle_fullscreen();
                    }

                    Key::Character("0") => {
                        if let Some(source) = source {
                            self.view.fit(source, win);
                            self.build_status_badge();
                            self.redraw();
                        }
                    }
                    Key::Character("1") => {
                        if let Some(source) = source {
                            self.view.set_zoom(1.0, source, win);
                            self.build_status_badge();
                            if self.view.zoom > self.shown.as_ref().map(|s| s.adequate_zoom).unwrap_or(1.0) {
                                self.request_full_res();
                            }
                            self.redraw();
                        }
                    }

                    Key::Character("+") | Key::Character("=") => self.zoom_by(KEY_ZOOM_STEP, centre),
                    Key::Character("-") | Key::Character("_") => self.zoom_by(1.0 / KEY_ZOOM_STEP, centre),

                    Key::Character("i") => {
                        self.smooth_override = Some(!self.smooth());
                        self.redraw();
                    }

                    // Turn the picture or film a quarter: how it is shown,
                    // never the file.
                    Key::Character("[") | Key::Character("{") => self.rotate_by(-1),
                    Key::Character("]") | Key::Character("}") => self.rotate_by(1),

                    // Cycle 100% -> fit -> cover -> 100%. Anything zoomed or
                    // panned by hand counts as 100% for this, so the next
                    // press always lands on a fit. Fit and cover stick, like
                    // autoscale, and are what the next image opens in.
                    Key::Character("z") | Key::Character("Z") => {
                        let Some(source) = source else { return };
                        let label = match (self.view.fitting, self.view.cover) {
                            (true, true) => {
                                self.view.set_zoom(1.0, source, win);
                                if self.view.zoom > self.shown.as_ref().map(|s| s.adequate_zoom).unwrap_or(1.0) {
                                    self.request_full_res();
                                }
                                "ZOOM 100%"
                            }
                            (true, false) => {
                                self.set_cover(true);
                                "ZOOM TO COVER"
                            }
                            (false, _) => {
                                self.set_cover(false);
                                "ZOOM TO FIT"
                            }
                        };
                        self.build_status_badge();
                        self.notify(label);
                        self.redraw();
                    }

                    Key::Named(NamedKey::F1) | Key::Character("?") => self.open_panel(),

                    Key::Named(NamedKey::F2) => self.toggle_mosaic(),

                    Key::Named(NamedKey::F7) => self.toggle_detection(),
                    Key::Named(NamedKey::F9) => self.cycle_depth(),

                    Key::Named(NamedKey::F8) => self.open_plugins(),

                    Key::Character("h") | Key::Character("H") => self.open_keys(),

                    // Type a phrase and have the picture searched for it. The
                    // slash is what every other program uses to mean "find",
                    // and it was the one punctuation key still going spare.
                    Key::Character("/") => {
                        self.typing = Some(String::new());
                        self.build_prompt();
                        self.redraw();
                    }


                    // Alt on its own opens the subtitle picker. It is the one
                    // modifier this program does not use for anything, which
                    // is why it can carry a menu without taking a key away.
                    Key::Named(NamedKey::Alt) if self.video.is_some() => {
                        if self.menu.is_some() {
                            self.menu = None;
                        } else {
                            self.open_menu();
                        }
                        self.redraw();
                    }

                    // Repeat. Only over a film, for the same reason mute is.
                    // If the film has already ended, it starts again now
                    // rather than waiting for an end that has been and gone.
                    // Over a picture, R walks the subfolders too, or stops.
                    Key::Character("r") | Key::Character("R") if self.video.is_none() => {
                        self.toggle_recursive();
                    }

                    Key::Character("r") | Key::Character("R") if self.video.is_some() => {
                        self.repeat = !self.repeat;
                        self.notify(if self.repeat { "REPEAT ON" } else { "REPEAT OFF" });
                        if self.repeat && self.playback_ended {
                            self.seek_to(std::time::Duration::ZERO, false, true);
                        }
                        self.redraw();
                    }

                    // Mute. Only over a film: on a still picture there is
                    // nothing it could mean.
                    Key::Character("m") | Key::Character("M") if self.video.is_some() => {
                        self.toggle_mute()
                    }

                    // Frame at a time, for looking rather than watching.
                    // The comma and full stop keys, as every other player
                    // uses them.
                    // With ctrl, the subtitle size instead: < and > are the
                    // same keys shifted.
                    Key::Character(",") | Key::Character("<") if self.ctrl_held && self.video.is_some() => {
                        self.scale_captions(-1)
                    }
                    Key::Character(".") | Key::Character(">") if self.ctrl_held && self.video.is_some() => {
                        self.scale_captions(1)
                    }
                    Key::Character(",") if self.video.is_some() => self.step_frame(false),
                    Key::Character(".") if self.video.is_some() => self.step_frame(true),

                    // Space pauses a video; on a still image it is next.
                    Key::Named(NamedKey::Space) if self.video.is_some() => self.toggle_pause(),

                    // Enter, or B, opens the detailed view: picture and folder
                    // at once.
                    Key::Named(NamedKey::Enter) | Key::Character("b") | Key::Character("B") => {
                        self.open_browser()
                    }

                    // Subtitles: next track, then off, then round again.
                    Key::Character("v") if !self.subtitles.is_empty() => {
                        let label = match self.subtitles.cycle() {
                            Some(track) => format!("SUBTITLES  {}", track.label()),
                            None => "SUBTITLES OFF".to_string(),
                        };
                        // Force a rebuild: the cue on screen belongs to the
                        // track we just left.
                        self.caption_for = None;
                        self.caption = None;
                        self.tick_subtitles();
                        self.notify(&label);
                        self.redraw();
                    }

                    Key::Named(NamedKey::F6) => self.start_analysis(),

                    Key::Named(NamedKey::F4) => {
                        self.config.show_debug = !self.config.show_debug;
                        if let Err(e) = self.config.save() {
                            eprintln!("show: could not save settings: {e:#}");
                        }
                        self.redraw();
                    }

                    Key::Named(NamedKey::F3) => {
                        self.config.hold_zoom = !self.config.hold_zoom;
                        if let Err(e) = self.config.save() {
                            eprintln!("show: could not save settings: {e:#}");
                        }
                        // Turning it off should not leave the current image
                        // stuck at a zoom the user just stopped asking for.
                        if !self.config.hold_zoom {
                            if let Some(source) = self.shown.as_ref().map(|s| s.source) {
                                self.view.fit(source, win);
                            }
                        }
                        self.build_status_badge();
                        self.redraw();
                    }

                    Key::Named(NamedKey::Delete) => self.request_delete(),

                    Key::Character("t") => {
                        self.osd_visible = !self.osd_visible;
                        timing::report("timings");
                        self.redraw();
                    }

                    // Arrows move through the folder. Panning lives on the
                    // mouse and on WASD, because stepping between images is
                    // what an arrow key means in a viewer.
                    // During a film the arrows scrub rather than change file:
                    // stepping to the next video is what the playlist keys are
                    // for, and moving within this one is the common wish.
                    Key::Named(dir @ (NamedKey::ArrowLeft | NamedKey::ArrowRight))
                        if self.video.is_some() =>
                    {
                        let forward = dir == NamedKey::ArrowRight;
                        // Plain arrows step keyframe to keyframe, which is
                        // both exact and instant. The modifiers keep the
                        // fixed-distance seeks, for when the question is "a
                        // bit further on" rather than "the next cut point".
                        if self.shift_held {
                            self.seek_by(SEEK_LONG, forward);
                        } else if self.ctrl_held {
                            self.seek_by(SEEK_MEDIUM, forward);
                        } else {
                            self.step_keyframe(forward);
                        }
                    }
                    // File navigation that works whatever is on screen. The
                    // arrows cannot do it during a film — they scrub there —
                    // so stepping between files needs keys of its own.
                    Key::Character("n") | Key::Character("N") => self.navigate(1),
                    Key::Character("p") | Key::Character("P") => self.navigate(-1),

                    Key::Named(NamedKey::ArrowLeft) | Key::Named(NamedKey::PageUp) => {
                        self.navigate(-1)
                    }
                    Key::Named(NamedKey::ArrowRight)
                    | Key::Named(NamedKey::PageDown)
                    | Key::Named(NamedKey::Space) => self.navigate(1),
                    Key::Named(NamedKey::Home) => self.jump_to_end(false),
                    Key::Named(NamedKey::End) => self.jump_to_end(true),

                    Key::Character(c @ ("w" | "a" | "s" | "d")) if c != "s" || self.shown.is_some() => {
                        if let Some(source) = source {
                            let (dx, dy) = match c {
                                "a" => (KEY_PAN_PIXELS, 0.0),
                                "d" => (-KEY_PAN_PIXELS, 0.0),
                                "w" => (0.0, KEY_PAN_PIXELS),
                                _ => (0.0, -KEY_PAN_PIXELS),
                            };
                            self.view.pan(dx, dy, source, win);
                            self.redraw();
                        }
                    }

                    _ => {}
                }
            }

            WindowEvent::RedrawRequested => {
                let _ = self.claim_focus();
                self.draw(el);
            }

            _ => {}
        }
    }
}


#[cfg(test)]
mod caption_tests {
    use super::{Corner, caption_px, caption_y, caption_y_scaled};

    #[test]
    fn a_bigger_caption_grows_about_its_middle() {
        // In the band below the picture, and inside it when there is no band.
        for bottom in [1260.0, 1440.0] {
            let usual = caption_y(bottom, 1440.0, 60.0) as f32 + 30.0;
            let grown = caption_y_scaled(bottom, 1440.0, 90.0, 1.5) as f32 + 45.0;
            assert!((usual - grown).abs() <= 1.0, "middle moved from {usual} to {grown}");
        }
        // Never pushed off the bottom of the window.
        assert!(caption_y_scaled(1440.0, 1440.0, 400.0, 3.0) + 400 <= 1440);
    }

    #[test]
    fn captions_scale_with_the_picture_not_the_window() {
        // Zooming in magnifies the film; words left at their old size would
        // shrink against everything around them.
        let fitted = caption_px(1080.0);
        let zoomed = caption_px(1600.0);
        assert!(zoomed > fitted, "{zoomed} is not larger than {fitted}");
    }

    #[test]
    fn the_size_is_bounded_at_both_ends() {
        // A deep zoom would otherwise fill the screen with two words.
        assert_eq!(caption_px(100_000.0), 96.0);
        // And a tiny window would produce text nobody could read.
        assert_eq!(caption_px(10.0), 14.0);
    }

    #[test]
    fn a_deep_letterbox_band_takes_the_caption() {
        // 1080p film letterboxed into a 1440-tall window: 180px of black at
        // the bottom, easily enough for a 60px caption.
        let y = caption_y(1260.0, 1440.0, 60.0);
        assert!(y >= 1260, "the caption must clear the picture: {y}");
        assert!(y + 60 <= 1440, "and stay inside the window: {y}");
    }

    #[test]
    fn a_shallow_band_is_not_used() {
        // Only 20px of black: a caption there would touch both edges, so it
        // goes back inside the picture.
        let y = caption_y(1420.0, 1440.0, 60.0);
        assert!(y < 1420, "should sit inside the picture, not in a sliver: {y}");
    }

    #[test]
    fn a_picture_filling_the_window_keeps_captions_inside_it() {
        let y = caption_y(1440.0, 1440.0, 60.0);
        assert!(y + 60 < 1440, "must not run off the bottom: {y}");
        assert!(y > 1440 / 2, "still low in the frame: {y}");
    }

    #[test]
    fn the_band_placement_keeps_clear_of_both_edges() {
        let (bottom, win, height) = (1000.0f32, 1440.0f32, 100.0f32);
        let y = caption_y(bottom, win, height);
        let margin = Corner::MARGIN;
        assert!(y as f32 >= bottom, "overlaps the picture");
        assert!(y as f32 + height <= win - margin + 1.0, "touches the window edge");
    }
}

#[cfg(test)]
mod click_tests {
    use super::was_click;

    #[test]
    fn a_still_press_and_release_is_a_click() {
        assert!(was_click(Some((100.0, 100.0)), (100.0, 100.0)));
        assert!(was_click(Some((100.0, 100.0)), (102.0, 103.0)), "a small wobble is still a click");
    }

    #[test]
    fn a_drag_is_not_a_click() {
        assert!(!was_click(Some((100.0, 100.0)), (400.0, 100.0)), "panning must not pause");
        assert!(!was_click(Some((100.0, 100.0)), (100.0, 400.0)));
        // Diagonal, just past the threshold on one axis only.
        assert!(!was_click(Some((100.0, 100.0)), (103.0, 110.0)));
    }

    #[test]
    fn a_release_with_no_press_is_not_a_click() {
        // Dragging in from outside the window, or a press swallowed by
        // something that took the gesture first.
        assert!(!was_click(None, (100.0, 100.0)));
    }
}

#[cfg(test)]
mod window_tests {
    use super::window_size_for;

    #[test]
    fn a_film_that_fits_opens_at_its_own_resolution() {
        // One video pixel to one screen pixel: nothing is resampled.
        assert_eq!(window_size_for((1920, 1080), (2560, 1440)), (1920, 1080));
        assert_eq!(window_size_for((1280, 720), (3072, 1728)), (1280, 720));
    }

    #[test]
    fn a_small_film_is_not_blown_up_to_fill_the_screen() {
        assert_eq!(window_size_for((640, 480), (2560, 1440)), (640, 480));
    }

    #[test]
    fn a_film_too_big_for_the_screen_comes_down_to_fit() {
        // 4K on a 1080p screen has to shrink, keeping its shape.
        let (w, h) = window_size_for((3840, 2160), (1920, 1080));
        assert!(w <= 1728 && h <= 972, "{w}x{h} leaves no room for the desktop");
        assert!((w as f32 / h as f32 - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn a_tall_film_is_bound_by_the_height_instead() {
        // Portrait video taller than the screen: the height is what runs out.
        let (w, h) = window_size_for((1080, 1920), (2560, 1440));
        assert!(h <= 1296, "must leave room for the desktop: {h}");
        assert!(w < 800, "the width follows the aspect ratio, not the screen: {w}");
    }

    #[test]
    fn the_aspect_ratio_is_preserved() {
        let (w, h) = window_size_for((1920, 800), (3840, 2160));
        let source = 1920.0 / 800.0;
        let got = w as f32 / h as f32;
        assert!((got - source).abs() < 0.01, "{got} is not {source}");
    }

    #[test]
    fn a_window_never_exceeds_the_screen() {
        let screen = (1366, 768);
        let (w, h) = window_size_for((3840, 2160), screen);
        assert!(w <= screen.0 && h <= screen.1, "{w}x{h} does not fit {screen:?}");
    }

    #[test]
    fn a_tiny_video_still_gets_a_usable_window() {
        let (w, h) = window_size_for((64, 48), (2560, 1440));
        assert!(w >= 320 && h >= 180, "{w}x{h} is too small to use");
    }
}

#[cfg(test)]
mod scrub_tests {
    use super::App;

    /// The mapping a click relies on: the left end of the track is the start
    /// of the file and the right end is the end of it, wherever the overlay
    /// happens to sit in the window.
    #[test]
    fn a_click_maps_to_where_it_landed() {
        // An overlay at x=100 whose track starts 12px in and runs 400px.
        let (x, track_x, track_w) = (100, 12, 400);
        let at = |cursor| App::track_fraction(cursor, x, track_x, track_w);

        assert_eq!(at(112.0), 0.0, "the left end is the start of the file");
        assert_eq!(at(512.0), 1.0, "the right end is the end of it");
        assert!((at(312.0) - 0.5).abs() < 1e-6, "the middle is the middle");
    }

    #[test]
    fn dragging_past_either_end_pins_rather_than_wrapping() {
        let at = |cursor| App::track_fraction(cursor, 100, 12, 400);
        assert_eq!(at(-500.0), 0.0);
        assert_eq!(at(5000.0), 1.0);
    }

    #[test]
    fn a_window_offset_does_not_shift_the_mapping() {
        // The same point on the track, with the overlay moved across the
        // window: a centred bar moves whenever the window is resized.
        let a = App::track_fraction(312.0, 100, 12, 400);
        let b = App::track_fraction(712.0, 500, 12, 400);
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }
}
