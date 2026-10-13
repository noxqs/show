//! `show --screenshot`: pick part of the screen, then copy it or save it.
//!
//! The screen is captured first, whole, and then shown back dimmed in a
//! fullscreen window — so what is selected is exactly what was on screen at
//! the moment it was asked for, and nothing the selection window itself draws
//! can end up in the picture. Dragging marks a rectangle, shown at full
//! brightness; its corners and edge midpoints have handles, it can be dragged
//! as a whole, and buttons beside it copy, save or cancel.
//!
//! X11 only: the capture is a `GetImage` of the root window.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorIcon, Fullscreen, Window, WindowId};

use crate::osd;

/// Brightness of everything outside the selection, out of 256.
const DIM: u32 = 96;
/// Half the side of a resize handle, in pixels.
const HANDLE: i32 = 6;
/// A press within this many pixels of a handle grabs it.
const GRAB: i32 = 10;
/// A drag smaller than this is a click, not a selection.
const MIN_SIZE: i32 = 4;
const ACCENT: u32 = 0x78dca0;

/// The screen as it was, RGBA8.
struct Capture {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

/// Grab the whole of the root window.
fn capture() -> Result<Capture> {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::{ConnectionExt, ImageFormat};

    let (conn, screen_num) = x11rb::xcb_ffi::XCBConnection::connect(None).context("connecting to the X server")?;
    let screen = &conn.setup().roots[screen_num];
    let (w, h) = (screen.width_in_pixels, screen.height_in_pixels);
    let image = conn
        .get_image(ImageFormat::Z_PIXMAP, screen.root, 0, 0, w, h, !0)
        .context("asking for the screen")?
        .reply()
        .context("reading the screen")?;
    let (w, h) = (w as u32, h as u32);
    if image.data.len() < (w * h * 4) as usize {
        bail!("the screen came back as {} bytes, not 32 bits a pixel", image.data.len());
    }
    // Z-pixmap at depth 24 or 32 is BGRX in memory, little-endian.
    let mut rgba = image.data;
    rgba.truncate((w * h * 4) as usize);
    for px in rgba.chunks_exact_mut(4) {
        px.swap(0, 2);
        px[3] = 255;
    }
    Ok(Capture { rgba, width: w, height: h })
}

/// A rectangle in screen pixels, always with `x0 <= x1` and `y0 <= y1`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

impl Rect {
    fn spanning(a: (i32, i32), b: (i32, i32)) -> Rect {
        Rect { x0: a.0.min(b.0), y0: a.1.min(b.1), x1: a.0.max(b.0), y1: a.1.max(b.1) }
    }

    fn width(&self) -> i32 {
        self.x1 - self.x0
    }

    fn height(&self) -> i32 {
        self.y1 - self.y0
    }

    fn contains(&self, (x, y): (i32, i32)) -> bool {
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }

    fn clamped(self, w: i32, h: i32) -> Rect {
        Rect { x0: self.x0.clamp(0, w), y0: self.y0.clamp(0, h), x1: self.x1.clamp(0, w), y1: self.y1.clamp(0, h) }
    }

    /// The eight handles, as which edges each one moves: (left, top, right,
    /// bottom). Corners move two, the midpoints one.
    fn handles(&self) -> [((i32, i32), (bool, bool, bool, bool)); 8] {
        let (mx, my) = ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2);
        [
            ((self.x0, self.y0), (true, true, false, false)),
            ((mx, self.y0), (false, true, false, false)),
            ((self.x1, self.y0), (false, true, true, false)),
            ((self.x1, my), (false, false, true, false)),
            ((self.x1, self.y1), (false, false, true, true)),
            ((mx, self.y1), (false, false, false, true)),
            ((self.x0, self.y1), (true, false, false, true)),
            ((self.x0, my), (true, false, false, false)),
        ]
    }
}

/// What a press started.
#[derive(Clone, Copy, Debug)]
enum Drag {
    None,
    /// A new rectangle, from where the press was.
    Selecting((i32, i32)),
    /// Moving the edges a handle owns.
    Resizing((bool, bool, bool, bool)),
    /// Moving the whole rectangle; the press's offset from its corner.
    Moving((i32, i32)),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Action {
    Copy,
    Save,
    Cancel,
}

/// The buttons beside the selection.
const BUTTONS: [(Action, &str); 3] = [(Action::Copy, "COPY"), (Action::Save, "SAVE"), (Action::Cancel, "CANCEL")];

struct Shot {
    capture: Capture,
    /// The capture packed for the window, at full brightness and dimmed.
    bright: Vec<u32>,
    dim: Vec<u32>,
    window: Option<std::rc::Rc<Window>>,
    surface: Option<softbuffer::Surface<std::rc::Rc<Window>, std::rc::Rc<Window>>>,
    selection: Option<Rect>,
    drag: Drag,
    cursor: (i32, i32),
    /// Button labels, rendered once, and where they were last drawn.
    labels: Vec<osd::Overlay>,
    button_rects: Vec<(Action, Rect)>,
    outcome: Option<Action>,
}

/// Pick a region of the screen; copy it, save it, or let it go.
pub fn run() -> Result<()> {
    let capture = capture()?;
    let pack = |px: &[u8], k: u32| -> u32 {
        let c = |v: u8| (v as u32 * k) >> 8;
        (c(px[0]) << 16) | (c(px[1]) << 8) | c(px[2])
    };
    let bright: Vec<u32> = capture.rgba.chunks_exact(4).map(|p| pack(p, 256)).collect();
    let dim: Vec<u32> = capture.rgba.chunks_exact(4).map(|p| pack(p, DIM)).collect();
    let labels = BUTTONS
        .iter()
        .map(|(_, text)| osd::render_scaled(&[osd::Row::good(*text, "")], osd::SCALE_PANEL))
        .collect();

    let event_loop = EventLoop::new()?;
    let mut shot = Shot {
        capture,
        bright,
        dim,
        window: None,
        surface: None,
        selection: None,
        drag: Drag::None,
        cursor: (0, 0),
        labels,
        button_rects: Vec::new(),
        outcome: None,
    };
    event_loop.run_app(&mut shot)?;

    let (Some(action), Some(sel)) = (shot.outcome, shot.selection) else { return Ok(()) };
    let sel = sel.clamped(shot.capture.width as i32, shot.capture.height as i32);
    if sel.width() < 1 || sel.height() < 1 || action == Action::Cancel {
        return Ok(());
    }
    let (w, h) = (sel.width() as u32, sel.height() as u32);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in sel.y0..sel.y1 {
        let start = (y as usize * shot.capture.width as usize + sel.x0 as usize) * 4;
        rgba.extend_from_slice(&shot.capture.rgba[start..start + w as usize * 4]);
    }
    match action {
        Action::Save => {
            let path = save_path();
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            }
            image::save_buffer(&path, &rgba, w, h, image::ExtendedColorType::Rgba8)
                .with_context(|| format!("saving {}", path.display()))?;
            println!("saved {}", path.display());
        }
        Action::Copy => copy(rgba, w, h)?,
        Action::Cancel => {}
    }
    Ok(())
}

/// `show --copy FILE`: decode it, put it on the clipboard, and stay to serve
/// it until something else is copied. Started by Ctrl+C in a window.
pub fn copy_file(path: &std::path::Path) -> Result<()> {
    let loaded = crate::loader::load(path, crate::loader::Io::Mmap, None)?;
    let d = loaded.decoded;
    copy(d.rgba, d.width, d.height)
}

/// Put the picture on the clipboard and stay to serve it.
///
/// An X11 clipboard holds no data, only the name of the program that has it;
/// pasting asks that program. So this keeps running — window gone — until
/// something else is copied, the way every X11 screenshot tool does.
pub fn copy(rgba: Vec<u8>, width: u32, height: u32) -> Result<()> {
    use arboard::SetExtLinux;
    let mut clipboard = arboard::Clipboard::new().context("opening the clipboard")?;
    println!("copied {width}x{height} to the clipboard");
    clipboard
        .set()
        .wait()
        .image(arboard::ImageData { width: width as usize, height: height as usize, bytes: rgba.into() })
        .context("putting the picture on the clipboard")
}

/// `~/Pictures/Screenshots/Screenshot 2026-10-04 13-05-09.png`, local time.
fn save_path() -> PathBuf {
    let dir = crate::playlist::Playlist::default_start_dir().join("Screenshots");
    let stamp = local_timestamp();
    let mut path = dir.join(format!("Screenshot {stamp}.png"));
    // Two in the same second get a number rather than overwrite each other.
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("Screenshot {stamp} {n}.png"));
        n += 1;
    }
    path
}

/// The local time as `YYYY-MM-DD HH-MM-SS`, from the C library: the one place
/// that knows the user's time zone without reading the database ourselves.
fn local_timestamp() -> String {
    #[repr(C)]
    struct Tm {
        sec: i32,
        min: i32,
        hour: i32,
        mday: i32,
        mon: i32,
        year: i32,
        wday: i32,
        yday: i32,
        isdst: i32,
        gmtoff: i64,
        zone: *const std::ffi::c_char,
    }
    unsafe extern "C" {
        fn localtime_r(time: *const i64, out: *mut Tm) -> *mut Tm;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let mut tm = Tm { sec: 0, min: 0, hour: 0, mday: 0, mon: 0, year: 0, wday: 0, yday: 0, isdst: 0, gmtoff: 0, zone: std::ptr::null() };
    // Safety: both pointers are to locals that outlive the call.
    if unsafe { localtime_r(&now, &mut tm) }.is_null() {
        return now.to_string();
    }
    format!(
        "{:04}-{:02}-{:02} {:02}-{:02}-{:02}",
        tm.year + 1900,
        tm.mon + 1,
        tm.mday,
        tm.hour,
        tm.min,
        tm.sec
    )
}

impl Shot {
    fn redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn screen(&self) -> (i32, i32) {
        (self.capture.width as i32, self.capture.height as i32)
    }

    /// The handle under the pointer, if any.
    fn handle_at(&self, at: (i32, i32)) -> Option<(bool, bool, bool, bool)> {
        let sel = self.selection?;
        sel.handles()
            .into_iter()
            .find(|((hx, hy), _)| (at.0 - hx).abs() <= GRAB && (at.1 - hy).abs() <= GRAB)
            .map(|(_, edges)| edges)
    }

    fn button_at(&self, at: (i32, i32)) -> Option<Action> {
        self.button_rects.iter().find(|(_, r)| r.contains(at)).map(|(a, _)| *a)
    }

    fn finish(&mut self, el: &ActiveEventLoop, action: Action) {
        if action != Action::Cancel && self.selection.is_none() {
            return;
        }
        self.outcome = Some(action);
        // Gone at once: what follows — encoding, or waiting on the clipboard —
        // happens with nothing on screen.
        if let Some(w) = &self.window {
            w.set_visible(false);
        }
        el.exit();
    }

    fn update_cursor(&self) {
        let Some(w) = &self.window else { return };
        let icon = if let Some((l, t, r, b)) = self.handle_at(self.cursor) {
            match (l, t, r, b) {
                (true, true, ..) | (_, _, true, true) => {
                    if l { CursorIcon::NwResize } else { CursorIcon::SeResize }
                }
                (_, true, true, _) | (true, _, _, true) => {
                    if r { CursorIcon::NeResize } else { CursorIcon::SwResize }
                }
                (true, ..) | (_, _, true, _) => CursorIcon::EwResize,
                _ => CursorIcon::NsResize,
            }
        } else if self.button_at(self.cursor).is_some() {
            CursorIcon::Pointer
        } else if self.selection.is_some_and(|s| s.contains(self.cursor)) {
            CursorIcon::Move
        } else {
            CursorIcon::Crosshair
        };
        w.set_cursor(icon);
    }

    /// Where the buttons go: a row under the selection's right edge, or above
    /// it, or inside it, whichever first fits on screen.
    fn place_buttons(&mut self, sel: Rect) {
        const GAP: i32 = 8;
        let widths: Vec<i32> = self.labels.iter().map(|l| l.width as i32).collect();
        let height = self.labels.iter().map(|l| l.height as i32).max().unwrap_or(0);
        let total = widths.iter().sum::<i32>() + GAP * (widths.len() as i32 - 1);
        let (sw, sh) = self.screen();
        let x = (sel.x1 - total).clamp(0, (sw - total).max(0));
        let y = if sel.y1 + GAP + height <= sh {
            sel.y1 + GAP
        } else if sel.y0 - GAP - height >= 0 {
            sel.y0 - GAP - height
        } else {
            sel.y1 - GAP - height
        };
        let mut left = x;
        self.button_rects = BUTTONS
            .iter()
            .zip(&widths)
            .map(|((action, _), w)| {
                let r = Rect { x0: left, y0: y, x1: left + w, y1: y + height };
                left += w + GAP;
                (*action, r)
            })
            .collect();
    }

    fn draw(&mut self) {
        // Buttons placed first: they are hit-tested between frames as well.
        let (sw, sh) = self.screen();
        if let Some(s) = self.selection.map(|s| s.clamped(sw, sh)) {
            self.place_buttons(s);
        }
        let (Some(window), Some(surface)) = (self.window.clone(), self.surface.as_mut()) else { return };
        let size = window.inner_size();
        let (ww, wh) = (size.width.max(1), size.height.max(1));
        let _ = surface.resize(
            std::num::NonZeroU32::new(ww).unwrap(),
            std::num::NonZeroU32::new(wh).unwrap(),
        );
        let Ok(mut buffer) = surface.buffer_mut() else { return };
        let (cw, ch) = (self.capture.width as usize, self.capture.height as usize);
        let (ww, wh) = (ww as usize, wh as usize);

        // The dimmed screen, then the selection at full brightness over it.
        for y in 0..wh.min(ch) {
            let n = ww.min(cw);
            buffer[y * ww..y * ww + n].copy_from_slice(&self.dim[y * cw..y * cw + n]);
        }
        let sel = self.selection.map(|s| s.clamped(ww.min(cw) as i32, wh.min(ch) as i32));
        if let Some(s) = sel.filter(|s| s.width() > 0 && s.height() > 0) {
            for y in s.y0 as usize..s.y1 as usize {
                let (a, b) = (s.x0 as usize, s.x1 as usize);
                buffer[y * ww + a..y * ww + b].copy_from_slice(&self.bright[y * cw + a..y * cw + b]);
            }
            let mut fill = |x0: i32, y0: i32, x1: i32, y1: i32, colour: u32| {
                for y in y0.max(0)..y1.min(wh as i32) {
                    for x in x0.max(0)..x1.min(ww as i32) {
                        buffer[y as usize * ww + x as usize] = colour;
                    }
                }
            };
            // The border, just outside the selection so it covers none of it.
            fill(s.x0 - 1, s.y0 - 1, s.x1 + 1, s.y0, ACCENT);
            fill(s.x0 - 1, s.y1, s.x1 + 1, s.y1 + 1, ACCENT);
            fill(s.x0 - 1, s.y0, s.x0, s.y1, ACCENT);
            fill(s.x1, s.y0, s.x1 + 1, s.y1, ACCENT);
            // Handles, once the rectangle is settled.
            if !matches!(self.drag, Drag::Selecting(_)) {
                for ((hx, hy), _) in s.handles() {
                    fill(hx - HANDLE, hy - HANDLE, hx + HANDLE, hy + HANDLE, 0xffffff);
                    fill(hx - HANDLE + 1, hy - HANDLE + 1, hx + HANDLE - 1, hy + HANDLE - 1, ACCENT);
                }
            }
            // The size, at the top left, and the buttons beside the selection.
            let size = osd::render_scaled(
                &[osd::Row::plain(format!("{} x {}", s.width(), s.height()))],
                osd::SCALE_COMPACT,
            );
            let size_at = (s.x0, (s.y0 - size.height as i32 - 6).max(0));
            blend(&mut buffer, ww, wh, &size, size_at);
            if !matches!(self.drag, Drag::Selecting(_)) {
                for ((_, rect), label) in self.button_rects.iter().zip(&self.labels) {
                    blend(&mut buffer, ww, wh, label, (rect.x0, rect.y0));
                }
            }
            let _ = buffer.present();
            return;
        }
        // Nothing selected yet: say what to do.
        let hint = osd::render_scaled(
            &[osd::Row::plain("DRAG TO SELECT  -  ESC CANCELS")],
            osd::SCALE_PANEL,
        );
        let at = ((ww as i32 - hint.width as i32) / 2, 40);
        blend(&mut buffer, ww, wh, &hint, at);
        let _ = buffer.present();
    }
}

/// Alpha-blend an overlay into the window buffer.
fn blend(dst: &mut [u32], w: usize, h: usize, ov: &osd::Overlay, (x0, y0): (i32, i32)) {
    for oy in 0..ov.height as i32 {
        let y = y0 + oy;
        if y < 0 || y >= h as i32 {
            continue;
        }
        for ox in 0..ov.width as i32 {
            let x = x0 + ox;
            if x < 0 || x >= w as i32 {
                continue;
            }
            let s = ((oy as usize * ov.width as usize) + ox as usize) * 4;
            let a = ov.rgba[s + 3] as u32;
            if a == 0 {
                continue;
            }
            let i = y as usize * w + x as usize;
            let d = dst[i];
            let mix = |sv: u8, shift: u32| ((sv as u32 * a + ((d >> shift) & 0xff) * (255 - a)) / 255) << shift;
            dst[i] = mix(ov.rgba[s], 16) | mix(ov.rgba[s + 1], 8) | mix(ov.rgba[s + 2], 0);
        }
    }
}

impl ApplicationHandler for Shot {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("show — screenshot")
            .with_decorations(false)
            .with_fullscreen(Some(Fullscreen::Borderless(None)))
            .with_window_level(winit::window::WindowLevel::AlwaysOnTop);
        let window = match el.create_window(attrs) {
            Ok(w) => std::rc::Rc::new(w),
            Err(e) => {
                eprintln!("show: no window for the screenshot: {e}");
                el.exit();
                return;
            }
        };
        window.set_cursor(CursorIcon::Crosshair);
        window.focus_window();
        let context = softbuffer::Context::new(window.clone()).ok();
        self.surface = context.and_then(|c| softbuffer::Surface::new(&c, window.clone()).ok());
        self.window = Some(window);
        self.redraw();
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.finish(el, Action::Cancel),
            WindowEvent::RedrawRequested => self.draw(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => self.finish(el, Action::Cancel),
                    Key::Named(NamedKey::Enter) => self.finish(el, Action::Copy),
                    Key::Character("c") | Key::Character("C") => self.finish(el, Action::Copy),
                    Key::Character("s") | Key::Character("S") => self.finish(el, Action::Save),
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let at = (position.x as i32, position.y as i32);
                let (dx, dy) = (at.0 - self.cursor.0, at.1 - self.cursor.1);
                self.cursor = at;
                let (sw, sh) = self.screen();
                match self.drag {
                    Drag::Selecting(anchor) => {
                        self.selection = Some(Rect::spanning(anchor, at));
                        self.redraw();
                    }
                    Drag::Resizing((l, t, r, b)) => {
                        if let Some(s) = self.selection.as_mut() {
                            let (mut x0, mut y0, mut x1, mut y1) = (s.x0, s.y0, s.x1, s.y1);
                            if l { x0 = at.0 }
                            if t { y0 = at.1 }
                            if r { x1 = at.0 }
                            if b { y1 = at.1 }
                            // Dragged past the opposite edge: the rectangle
                            // turns inside out rather than collapsing.
                            *s = Rect::spanning((x0, y0), (x1, y1)).clamped(sw, sh);
                            self.drag = Drag::Resizing((
                                if x0 <= x1 { l } else { r },
                                if y0 <= y1 { t } else { b },
                                if x0 <= x1 { r } else { l },
                                if y0 <= y1 { b } else { t },
                            ));
                        }
                        self.redraw();
                    }
                    Drag::Moving((ox, oy)) => {
                        if let Some(s) = self.selection.as_mut() {
                            let (w, h) = (s.width(), s.height());
                            let x0 = (at.0 - ox).clamp(0, (sw - w).max(0));
                            let y0 = (at.1 - oy).clamp(0, (sh - h).max(0));
                            *s = Rect { x0, y0, x1: x0 + w, y1: y0 + h };
                        }
                        let _ = (dx, dy);
                        self.redraw();
                    }
                    Drag::None => self.update_cursor(),
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => match state {
                ElementState::Pressed => {
                    let at = self.cursor;
                    if let Some(action) = self.button_at(at) {
                        self.finish(el, action);
                    } else if let Some(edges) = self.handle_at(at) {
                        self.drag = Drag::Resizing(edges);
                    } else if let Some(s) = self.selection.filter(|s| s.contains(at)) {
                        self.drag = Drag::Moving((at.0 - s.x0, at.1 - s.y0));
                    } else {
                        self.drag = Drag::Selecting(at);
                        self.selection = None;
                        self.button_rects.clear();
                    }
                    self.redraw();
                }
                ElementState::Released => {
                    if let Drag::Selecting(_) = self.drag
                        && self.selection.is_none_or(|s| s.width() < MIN_SIZE || s.height() < MIN_SIZE)
                    {
                        self.selection = None;
                    }
                    self.drag = Drag::None;
                    self.update_cursor();
                    self.redraw();
                }
            },
            // Right click backs out, as it does in most of these tools.
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Right, .. } => {
                self.finish(el, Action::Cancel)
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rectangle_is_the_same_whichever_way_it_was_dragged() {
        let r = Rect::spanning((50, 80), (10, 20));
        assert_eq!(r, Rect { x0: 10, y0: 20, x1: 50, y1: 80 });
        assert_eq!((r.width(), r.height()), (40, 60));
        assert!(r.contains((10, 20)) && !r.contains((50, 80)));
    }

    #[test]
    fn every_handle_moves_the_edges_it_sits_on() {
        let r = Rect { x0: 0, y0: 0, x1: 100, y1: 60 };
        for ((x, y), (l, t, rt, b)) in r.handles() {
            assert_eq!(l, x == 0, "left at {x},{y}");
            assert_eq!(rt, x == 100, "right at {x},{y}");
            assert_eq!(t, y == 0, "top at {x},{y}");
            assert_eq!(b, y == 60, "bottom at {x},{y}");
        }
    }

    #[test]
    fn the_file_name_stamp_is_fixed_width() {
        let stamp = local_timestamp();
        assert_eq!(stamp.len(), "2026-10-04 13-05-09".len(), "{stamp}");
    }
}
