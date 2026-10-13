//! Noticing images arrive in, or leave, the folder being shown.
//!
//! inotify, through the four C library calls it takes, on a thread of its
//! own: a watch on the folder — and, when subfolders are being shown, on every
//! folder beneath it, with new ones picked up as they appear. A burst of
//! changes — a copy of two hundred files — is reported once, when the folder
//! has been quiet for a moment, so the list is rebuilt once and not two
//! hundred times.
//!
//! What counts: a file finished being written, or moved in; a file deleted, or
//! moved out; a folder made or removed. A file still being written is not an
//! image yet, so creation alone is not reported — its closing is.

use std::ffi::{CString, c_char, c_int, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const IN_CLOSE_WRITE: u32 = 0x0000_0008;
const IN_MOVED_FROM: u32 = 0x0000_0040;
const IN_MOVED_TO: u32 = 0x0000_0080;
const IN_CREATE: u32 = 0x0000_0100;
const IN_DELETE: u32 = 0x0000_0200;
const IN_DELETE_SELF: u32 = 0x0000_0400;
const IN_MOVE_SELF: u32 = 0x0000_0800;
const IN_ONLYDIR: u32 = 0x0100_0000;
const IN_ISDIR: u32 = 0x4000_0000;
const IN_NONBLOCK: c_int = 0o4000;
const IN_CLOEXEC: c_int = 0o2000000;
const POLLIN: i16 = 0x1;

const WATCHED: u32 =
    IN_CLOSE_WRITE | IN_MOVED_FROM | IN_MOVED_TO | IN_CREATE | IN_DELETE | IN_DELETE_SELF | IN_MOVE_SELF | IN_ONLYDIR;

/// How long the folder has to be quiet before a change is reported.
const SETTLE: Duration = Duration::from_millis(300);

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: i16,
    revents: i16,
}

unsafe extern "C" {
    fn inotify_init1(flags: c_int) -> c_int;
    fn inotify_add_watch(fd: c_int, path: *const c_char, mask: u32) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
    fn close(fd: c_int) -> c_int;
    fn poll(fds: *mut PollFd, nfds: u64, timeout: c_int) -> c_int;
}

/// A folder being watched. Dropped, the watching stops.
pub struct Watcher {
    stop: Arc<AtomicBool>,
}

impl Drop for Watcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Watcher {
    /// Watch `root` — and everything beneath it if `recursive` — calling
    /// `on_change`, on the watcher's thread, after each settled burst of
    /// changes. None where inotify cannot be had.
    pub fn start(root: &Path, recursive: bool, on_change: impl Fn() + Send + 'static) -> Option<Watcher> {
        // Safety: plain system calls; the descriptor is owned by the thread
        // below and closed when it ends.
        let fd = unsafe { inotify_init1(IN_NONBLOCK | IN_CLOEXEC) };
        if fd < 0 {
            return None;
        }
        let mut folders = Vec::new();
        add_watches(fd, root, recursive, &mut folders);
        if folders.is_empty() {
            unsafe { close(fd) };
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let quit = stop.clone();
        std::thread::Builder::new()
            .name("watch".into())
            .spawn(move || {
                run(fd, recursive, folders, &quit, on_change);
                unsafe { close(fd) };
            })
            .ok()?;
        Some(Watcher { stop })
    }
}

/// Watch `dir`, and when `recursive` every folder under it that is not hidden,
/// as the list itself skips hidden ones. Each watch descriptor's folder is
/// kept, so a new subfolder can be found by name when it appears.
fn add_watches(fd: c_int, dir: &Path, recursive: bool, folders: &mut Vec<(c_int, PathBuf)>) {
    let Ok(c) = CString::new(dir.as_os_str().as_bytes()) else { return };
    // Safety: a NUL-terminated path that outlives the call.
    let wd = unsafe { inotify_add_watch(fd, c.as_ptr(), WATCHED) };
    if wd < 0 {
        return;
    }
    folders.push((wd, dir.to_path_buf()));
    if !recursive {
        return;
    }
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let hidden = entry.file_name().as_bytes().first() == Some(&b'.');
        if !hidden && entry.file_type().is_ok_and(|t| t.is_dir()) {
            add_watches(fd, &entry.path(), true, folders);
        }
    }
}

fn run(
    fd: c_int,
    recursive: bool,
    mut folders: Vec<(c_int, PathBuf)>,
    stop: &AtomicBool,
    on_change: impl Fn(),
) {
    let mut buffer = vec![0u8; 64 * 1024];
    let mut pending: Option<Instant> = None;
    while !stop.load(Ordering::Relaxed) {
        // Wakes for events, and every quarter second to notice being dropped
        // and to report a burst once it has gone quiet.
        let mut pfd = PollFd { fd, events: POLLIN, revents: 0 };
        let ready = unsafe { poll(&mut pfd, 1, 250) };
        if ready > 0 {
            // Safety: the buffer is ours and the length is its length.
            let n = unsafe { read(fd, buffer.as_mut_ptr().cast(), buffer.len()) };
            if n > 0 {
                let mut new_dirs = Vec::new();
                for (wd, mask, name) in events(&buffer[..n as usize]) {
                    if !counts(mask, &name) {
                        continue;
                    }
                    pending = Some(Instant::now());
                    // A folder made beneath one being watched: watch it too,
                    // and anything already in it by the time we look.
                    if recursive && mask & IN_ISDIR != 0 && mask & (IN_CREATE | IN_MOVED_TO) != 0 {
                        if let Some((_, parent)) = folders.iter().find(|(w, _)| *w == wd) {
                            new_dirs.push(parent.join(&name));
                        }
                    }
                }
                for dir in new_dirs {
                    add_watches(fd, &dir, true, &mut folders);
                }
            }
        }
        if pending.is_some_and(|at| at.elapsed() >= SETTLE) {
            pending = None;
            on_change();
        }
    }
}

/// Whether an event is one the list cares about: an image coming or going, or
/// a folder — which may hold images — coming or going.
fn counts(mask: u32, name: &str) -> bool {
    if mask & (IN_DELETE_SELF | IN_MOVE_SELF) != 0 {
        return true;
    }
    if mask & IN_ISDIR != 0 {
        return mask & (IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO) != 0 && !name.starts_with('.');
    }
    // Creation alone is a file still being written; its close is what counts.
    mask & (IN_CLOSE_WRITE | IN_MOVED_TO | IN_MOVED_FROM | IN_DELETE) != 0
        && crate::playlist::is_image_extension(Path::new(name))
}

/// The events in one read: watch descriptor, mask and name.
fn events(mut bytes: &[u8]) -> Vec<(c_int, u32, String)> {
    const HEADER: usize = 16;
    let mut out = Vec::new();
    while bytes.len() >= HEADER {
        let int = |at: usize| u32::from_ne_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        let (wd, mask, len) = (int(0) as c_int, int(4), int(12) as usize);
        let Some(raw) = bytes.get(HEADER..HEADER + len) else { break };
        let name = String::from_utf8_lossy(raw.split(|b| *b == 0).next().unwrap_or(&[])).into_owned();
        out.push((wd, mask, name));
        bytes = &bytes[HEADER + len..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("show-watch-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn changed(rx: &mpsc::Receiver<()>) -> bool {
        rx.recv_timeout(Duration::from_secs(3)).is_ok()
    }

    #[test]
    fn an_image_arriving_or_leaving_is_reported_once_settled() {
        let dir = scratch("flat");
        let (tx, rx) = mpsc::channel();
        let watcher = Watcher::start(&dir, false, move || {
            let _ = tx.send(());
        })
        .unwrap();

        // A burst: three images, one report.
        for n in 0..3 {
            std::fs::write(dir.join(format!("{n}.png")), b"x").unwrap();
        }
        assert!(changed(&rx), "new images were not noticed");
        assert!(rx.recv_timeout(Duration::from_millis(600)).is_err(), "one burst, one report");

        // Not an image: nothing.
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        assert!(rx.recv_timeout(Duration::from_millis(800)).is_err(), "a text file is not an image");

        std::fs::remove_file(dir.join("1.png")).unwrap();
        assert!(changed(&rx), "a deleted image was not noticed");
        drop(watcher);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_new_subfolder_is_watched_as_well() {
        let dir = scratch("tree");
        let (tx, rx) = mpsc::channel();
        let _watcher = Watcher::start(&dir, true, move || {
            let _ = tx.send(());
        })
        .unwrap();

        std::fs::create_dir(dir.join("sub")).unwrap();
        assert!(changed(&rx), "a new folder was not noticed");
        // Given a moment to be watched, then an image in it.
        std::thread::sleep(Duration::from_millis(400));
        std::fs::write(dir.join("sub").join("a.jpg"), b"x").unwrap();
        assert!(changed(&rx), "an image in a new subfolder was not noticed");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
