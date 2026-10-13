//! Keeping the screen on while a video plays.
//!
//! Nobody touches the mouse during a film, so without this the desktop takes
//! the stillness for absence and blanks the screen half way through. Desktops
//! offer a way to say "not now" over the session bus, but not the same one:
//!
//! - `org.freedesktop.ScreenSaver` — KDE, Xfce, MATE, and others.
//! - `org.gnome.SessionManager` with the idle flag — GNOME and Cinnamon.
//!
//! Both are asked, and whichever answer holds the inhibition. Each is tied to
//! the bus connection as well as the cookie, so a viewer that crashes mid-film
//! releases it with its connection rather than leaving the screen on for good.
//!
//! The calls block, so they are made on a thread of their own; the viewer only
//! says whether it wants the screen kept on. Where there is no session bus or
//! neither service answers, nothing happens.

use std::sync::mpsc::{Sender, channel};

use zbus::blocking::{Connection, Proxy};

const APP: &str = "show";
const REASON: &str = "Playing a video";
/// `Inhibit` flag for the session manager: hold off going idle.
const INHIBIT_IDLE: u32 = 8;

pub struct Awake {
    tx: Option<Sender<bool>>,
    wanted: bool,
}

impl Awake {
    pub fn new() -> Awake {
        let (tx, rx) = channel::<bool>();
        let spawned = std::thread::Builder::new().name("awake".into()).spawn(move || {
            let mut held: Option<Held> = None;
            for want in rx {
                if want && held.is_none() {
                    held = Held::take();
                } else if !want && let Some(h) = held.take() {
                    h.release();
                }
            }
            if let Some(h) = held {
                h.release();
            }
        });
        Awake { tx: spawned.ok().map(|_| tx), wanted: false }
    }

    /// Keep the screen on, or stop. Cheap to call every frame: only a change
    /// reaches the bus.
    pub fn set(&mut self, want: bool) {
        if want != self.wanted {
            self.wanted = want;
            if let Some(tx) = &self.tx {
                let _ = tx.send(want);
            }
        }
    }
}

/// An inhibition in force: the connection it lives on, and the cookies to
/// hand back.
struct Held {
    bus: Connection,
    screensaver: Option<u32>,
    session: Option<u32>,
}

impl Held {
    fn take() -> Option<Held> {
        let bus = Connection::session().ok()?;
        let screensaver = Proxy::new(&bus, "org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver", "org.freedesktop.ScreenSaver")
            .and_then(|p| p.call::<_, _, u32>("Inhibit", &(APP, REASON)))
            .ok();
        let session = Proxy::new(&bus, "org.gnome.SessionManager", "/org/gnome/SessionManager", "org.gnome.SessionManager")
            .and_then(|p| p.call::<_, _, u32>("Inhibit", &(APP, 0u32, REASON, INHIBIT_IDLE)))
            .ok();
        if screensaver.is_none() && session.is_none() {
            return None;
        }
        Some(Held { bus, screensaver, session })
    }

    fn release(self) {
        if let Some(cookie) = self.screensaver
            && let Ok(p) = Proxy::new(&self.bus, "org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver", "org.freedesktop.ScreenSaver")
        {
            let _: zbus::Result<()> = p.call("UnInhibit", &(cookie,));
        }
        if let Some(cookie) = self.session
            && let Ok(p) = Proxy::new(&self.bus, "org.gnome.SessionManager", "/org/gnome/SessionManager", "org.gnome.SessionManager")
        {
            let _: zbus::Result<()> = p.call("Uninhibit", &(cookie,));
        }
    }
}
