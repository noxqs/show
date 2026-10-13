//! One tray icon for every open window, with a menu to bring any of them back.
//!
//! Each window is its own process, so the windows find each other on the
//! session bus rather than through anything shared in memory:
//!
//! - Every window takes a name of its own, `dev.show.Window.p<pid>`, and
//!   answers two calls on it: `Describe` (its folder and file, for the menu)
//!   and `Activate` (raise and focus).
//! - Every window also asks for `dev.show.Tray`. The bus gives it to one and
//!   queues the rest; the holder shows the icon. When that window closes, the
//!   bus hands the name to the next in the queue, which puts the icon back —
//!   so there is always exactly one icon while any window is open, and none
//!   after the last.
//!
//! The icon is a StatusNotifierItem through `ksni`: pure Rust over D-Bus, no
//! toolkit and nothing linked. Where there is no session bus or no tray host,
//! nothing happens and the viewer works as before.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use zbus::blocking::Connection;

const TRAY_NAME: &str = "dev.show.Tray";
const WINDOW_PREFIX: &str = "dev.show.Window.p";
const WINDOW_PATH: &str = "/dev/show/Window";
const WINDOW_INTERFACE: &str = "dev.show.Window";
/// How often the icon's owner looks for windows opened or closed. The menu is
/// only read when opened, so this only has to be fresher than a person is
/// quick.
const RESCAN: Duration = Duration::from_millis(1500);

/// Another open window — or this one — as the others see it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Peer {
    /// Its name on the bus, which is how it is asked to come forward.
    pub name: String,
    pub folder: String,
    pub file: String,
}

impl Peer {
    /// The long form, for the tray menu.
    fn label(&self) -> String {
        match (self.folder.is_empty(), self.file.is_empty()) {
            (true, true) => "show".to_string(),
            (false, true) => self.folder.clone(),
            (true, false) => self.file.clone(),
            (false, false) => format!("{}  —  {}", self.folder, self.file),
        }
    }
}

/// This window's presence on the bus. Dropped, it is gone from every menu.
pub struct Presence {
    label: Arc<Mutex<(String, String)>>,
    /// Every window open, this one included, refreshed in the background.
    peers: Arc<Mutex<Vec<Peer>>>,
    own: String,
    connection: Connection,
}

impl Presence {
    /// What the menu says for this window: the folder, and the file in it.
    pub fn describe(&self, folder: &str, file: &str) {
        if let Ok(mut label) = self.label.lock() {
            *label = (folder.to_string(), file.to_string());
        }
    }

    /// Every open window, this one included, in a stable order.
    pub fn peers(&self) -> Vec<Peer> {
        self.peers.lock().map(|p| p.clone()).unwrap_or_default()
    }

    /// This window's own name, to tell it apart in [`Presence::peers`].
    pub fn own_name(&self) -> &str {
        &self.own
    }

    /// Ask another window to come forward — fullscreen, if `fullscreen`, so
    /// stepping between windows does not drop out of fullscreen on the way.
    /// On a thread: the other window may be busy, and this one must not wait.
    pub fn activate(&self, name: &str, fullscreen: bool) {
        let (bus, name) = (self.connection.clone(), name.to_string());
        std::thread::spawn(move || {
            let method = if fullscreen { "ActivateFullscreen" } else { "Activate" };
            if let Ok(proxy) = zbus::blocking::Proxy::new(&bus, name, WINDOW_PATH, WINDOW_INTERFACE) {
                let _: zbus::Result<()> = proxy.call(method, &());
            }
        });
    }
}

/// The calls a window answers.
struct Window {
    label: Arc<Mutex<(String, String)>>,
    /// Called with whether to go fullscreen as well as come forward.
    on_activate: Box<dyn Fn(bool) + Send + Sync>,
}

#[zbus::interface(name = "dev.show.Window")]
impl Window {
    fn activate(&self) {
        (self.on_activate)(false);
    }

    fn activate_fullscreen(&self) {
        (self.on_activate)(true);
    }

    fn describe(&self) -> (String, String) {
        self.label.lock().map(|l| l.clone()).unwrap_or_default()
    }
}

/// Join the bus: answer for this window, keep a list of the others, and queue
/// for the icon. Both callbacks run on bus threads: `on_activate` when this
/// window is picked, with whether it should go fullscreen; `on_peers` when the
/// set of open windows, or what one is showing, changes.
pub fn join(
    on_activate: impl Fn(bool) + Send + Sync + 'static,
    on_peers: impl Fn() + Send + Sync + 'static,
) -> Option<Presence> {
    let connection = Connection::session().ok()?;
    let label = Arc::new(Mutex::new((String::new(), String::new())));
    let window = Window { label: label.clone(), on_activate: Box::new(on_activate) };
    connection.object_server().at(WINDOW_PATH, window).ok()?;
    let own = format!("{WINDOW_PREFIX}{}", std::process::id());
    connection.request_name(own.clone()).ok()?;

    // The others, kept current for the strip of windows across the top.
    let peers = Arc::new(Mutex::new(Vec::new()));
    let (bus, list) = (connection.clone(), peers.clone());
    std::thread::Builder::new()
        .name("peers".into())
        .spawn(move || {
            let Ok(dbus) = zbus::blocking::fdo::DBusProxy::new(&bus) else { return };
            loop {
                let now = list_windows(&bus, &dbus);
                let changed = list.lock().map(|mut l| {
                    let changed = *l != now;
                    *l = now;
                    changed
                });
                if changed.unwrap_or(false) {
                    on_peers();
                }
                std::thread::sleep(RESCAN);
            }
        })
        .ok()?;

    let bus = connection.clone();
    std::thread::Builder::new()
        .name("tray".into())
        .spawn(move || {
            if let Err(e) = keep_the_icon(&bus) {
                eprintln!("show: no tray icon: {e:#}");
            }
        })
        .ok()?;
    Some(Presence { label, peers, own, connection })
}

/// Wait until this window holds the icon's name, then show the icon and keep
/// its menu current for as long as the process lives.
fn keep_the_icon(bus: &Connection) -> anyhow::Result<()> {
    use ksni::blocking::TrayMethods;
    use zbus::fdo::{RequestNameFlags, RequestNameReply};

    let dbus = zbus::blocking::fdo::DBusProxy::new(bus)?;
    // Listening before asking, so a hand-over between the two is not missed.
    let mut acquired = dbus.receive_name_acquired()?;
    let name = zbus::names::WellKnownName::try_from(TRAY_NAME)?;
    // Queued by default: no `DoNotQueue`. Replacement is allowed, but nothing
    // asks for it — every window asks the same plain way.
    if dbus.request_name(name, RequestNameFlags::AllowReplacement.into())? != RequestNameReply::PrimaryOwner {
        // Queued behind another window: ours when it closes.
        loop {
            let signal = acquired.next().ok_or_else(|| anyhow::anyhow!("the bus went away"))?;
            if signal.args()?.name.as_str() == TRAY_NAME {
                break;
            }
        }
    }

    let handle = Tray { windows: Vec::new(), bus: bus.clone() }.spawn()?;
    let mut shown = Vec::new();
    loop {
        let windows = list_windows(bus, &dbus);
        if windows != shown {
            shown = windows.clone();
            handle.update(|tray| tray.windows = windows);
        }
        std::thread::sleep(RESCAN);
    }
}

/// Every window on the bus, in a stable order: by when it opened — its pid —
/// so the buttons across the top do not shuffle as windows change pictures.
fn list_windows(bus: &Connection, dbus: &zbus::blocking::fdo::DBusProxy) -> Vec<Peer> {
    let Ok(names) = dbus.list_names() else { return Vec::new() };
    let mut windows: Vec<Peer> = names
        .into_iter()
        .map(|n| n.to_string())
        .filter(|n| n.starts_with(WINDOW_PREFIX))
        .filter_map(|name| {
            let proxy = zbus::blocking::Proxy::new(bus, name.clone(), WINDOW_PATH, WINDOW_INTERFACE).ok()?;
            let (folder, file): (String, String) = proxy.call("Describe", &()).ok()?;
            Some(Peer { name, folder, file })
        })
        .collect();
    let pid = |p: &Peer| p.name[WINDOW_PREFIX.len()..].parse::<u32>().unwrap_or(u32::MAX);
    windows.sort_by_key(pid);
    windows
}

struct Tray {
    windows: Vec<Peer>,
    bus: Connection,
}

impl ksni::Tray for Tray {
    /// A click opens the list; there is no one window it could mean.
    const MENU_ON_ACTIVATE: bool = true;

    fn id(&self) -> String {
        "show".into()
    }

    fn title(&self) -> String {
        "show".into()
    }

    fn icon_name(&self) -> String {
        "image-x-generic".into()
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        let n = self.windows.len();
        ksni::ToolTip {
            title: format!("show — {n} window{}", if n == 1 { "" } else { "s" }),
            ..Default::default()
        }
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        // The screenshot first, then the windows.
        let mut items: Vec<ksni::MenuItem<Self>> = vec![
            ksni::menu::StandardItem {
                label: "Screenshot".into(),
                icon_name: "applets-screenshooter".into(),
                activate: Box::new(|_: &mut Self| take_screenshot()),
                ..Default::default()
            }
            .into(),
            ksni::MenuItem::Separator,
        ];
        if self.windows.is_empty() {
            items.push(
                ksni::menu::StandardItem { label: "No windows".into(), enabled: false, ..Default::default() }.into(),
            );
            return items;
        }
        items.extend(self.windows
            .iter()
            .map(|peer| {
                let name = peer.name.clone();
                ksni::menu::StandardItem {
                    // Underscores are access keys in menu labels; doubled, they
                    // show as themselves, and file names are full of them.
                    label: peer.label().replace('_', "__"),
                    activate: Box::new(move |tray: &mut Self| {
                        // Off the menu's thread: a window busy decoding must
                        // not freeze the menu while it answers.
                        let (bus, name) = (tray.bus.clone(), name.clone());
                        std::thread::spawn(move || {
                            if let Ok(proxy) = zbus::blocking::Proxy::new(&bus, name, WINDOW_PATH, WINDOW_INTERFACE) {
                                let _: zbus::Result<()> = proxy.call("Activate", &());
                            }
                        });
                    }),
                    ..Default::default()
                }
                .into()
            }));
        items
    }
}

/// Start `show --screenshot` as a process of its own: it takes the screen
/// over, and it may stay on afterwards to serve the clipboard, neither of
/// which should touch the window that happens to hold the icon.
///
/// Not at once: the menu this was picked from is still on screen, closing,
/// and the desktop under it is still being redrawn. Grabbed straight away, the
/// screen came out with the menu in it and the windows beneath half drawn. So
/// it waits for the menu to go and the compositor to catch up — off the menu's
/// own thread, which must not be held.
fn take_screenshot() {
    /// Long enough for a menu's fade and a compositor frame or two; short
    /// enough not to feel like a wait.
    const SETTLE: Duration = Duration::from_millis(400);
    let Ok(exe) = std::env::current_exe() else { return };
    std::thread::spawn(move || {
        std::thread::sleep(SETTLE);
        if let Ok(mut child) = std::process::Command::new(exe).arg("--screenshot").spawn() {
            // Reaped when it ends, so it does not linger as a zombie.
            let _ = child.wait();
        }
    });
}
