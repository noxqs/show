//! Desktop integration: putting `show` where the system can find it.
//!
//! To be launchable from the applications menu — and to be offered as the
//! handler when someone opens an image — a Linux program needs two things: an
//! executable on `PATH`, and a `.desktop` entry describing it. Both live under
//! `~/.local`, so none of this needs root.
//!
//! Nothing here happens without being asked for. Installing writes files into
//! the user's home and can change which application opens their photos, which
//! is not a decision a viewer should make on its own the first time it runs.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const DESKTOP_FILE: &str = "show.desktop";

/// The window class this program reports, and the `StartupWMClass` the desktop
/// entry advertises. They have to agree for the taskbar to connect a running
/// window to the launcher that started it.
pub const WM_CLASS: &str = "Show";

/// MIME types we claim. Deliberately only those we can actually decode today,
/// so we never become the default handler for a file we would fail to open.
const MIME_TYPES: &[&str] = &[
    // Video, now that it plays.
    "video/mp4",
    "video/x-matroska",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",
    "video/mpeg",
    "video/x-ms-wmv",
    "video/x-flv",
    "video/ogg",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/x-icon",
    "image/vnd.microsoft.icon",
    "image/x-targa",
    "image/x-tga",
    "image/x-portable-pixmap",
    "image/x-portable-anymap",
    "image/vnd.radiance",
    "image/x-exr",
    "image/x-qoi",
    "image/jxl",
    "image/heic",
    "image/avif",
    "image/heif",
];

pub struct Paths {
    pub bin: PathBuf,
    pub desktop: PathBuf,
}

pub fn paths() -> Option<Paths> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    Some(Paths {
        bin: home.join(".local/bin/show"),
        desktop: home.join(".local/share/applications").join(DESKTOP_FILE),
    })
}

#[derive(Debug, PartialEq, Eq)]
pub enum State {
    /// Running from the installed location with a desktop entry in place.
    Installed,
    /// Never installed, or installed somewhere we do not manage.
    NotInstalled,
    /// A newer build than the installed copy is running: a fresh `cargo build`
    /// that has not been deployed yet.
    Stale,
    /// Could not determine, e.g. no HOME.
    Unknown,
}

/// Filesystem timestamps can disagree by a hair across copies and filesystems,
/// so a build only counts as newer when it is newer by a visible margin.
const STALE_TOLERANCE: std::time::Duration = std::time::Duration::from_secs(2);

fn modified(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

pub fn state() -> State {
    let Some(p) = paths() else { return State::Unknown };
    if !p.bin.exists() || !p.desktop.exists() {
        return State::NotInstalled;
    }
    let Ok(current) = std::env::current_exe() else { return State::Unknown };

    // Running the installed copy is the common case and needs no comparison.
    if current == p.bin {
        return State::Installed;
    }

    // Modification time, not size: two builds of the same program are very
    // often byte-for-byte the same length while differing entirely.
    match (modified(&current), modified(&p.bin)) {
        (Some(running), Some(installed)) => {
            match running.duration_since(installed) {
                Ok(ahead) if ahead > STALE_TOLERANCE => State::Stale,
                _ => State::Installed,
            }
        }
        _ => State::Unknown,
    }
}

/// Build timestamps for the running and installed copies, for display.
pub fn build_times() -> (Option<std::time::SystemTime>, Option<std::time::SystemTime>) {
    let installed = paths().and_then(|p| modified(&p.bin));
    let running = std::env::current_exe().ok().as_deref().and_then(modified);
    (running, installed)
}

/// Format a timestamp as `YYYY-MM-DD HH:MM` in UTC.
///
/// Hand-rolled rather than pulling in a date library for one line of a panel.
/// UTC because the alternative is reading the timezone database, and the only
/// job here is telling two builds apart.
pub fn format_time(t: std::time::SystemTime) -> String {
    let secs = match t.duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(_) => return "BEFORE 1970".into(),
    };
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}", rem / 3600, (rem % 3600) / 60)
}

/// Days since the Unix epoch to a calendar date (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // Shift the epoch to 0000-03-01 so leap days land at the end of the cycle.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn desktop_entry(exec: &Path) -> String {
    format!(
        "[Desktop Entry]\n\
         Version=1.0\n\
         Name=Show\n\
         GenericName=Image and Video Viewer\n\
         Comment=Fast fullscreen viewer for images and video\n\
         Keywords=image;photo;picture;viewer;video;player;\n\
         Exec={} %F\n\
         TryExec={}\n\
         Icon=multimedia-photo-viewer\n\
         Terminal=false\n\
         Type=Application\n\
         Categories=Graphics;Viewer;2DGraphics;AudioVideo;Player;\n\
         MimeType={};\n\
         StartupNotify=true\n\
         StartupWMClass={}\n",
        exec.display(),
        exec.display(),
        MIME_TYPES.join(";"),
        WM_CLASS
    )
}

/// Copy this executable into `~/.local/bin` and write the desktop entry.
pub fn install() -> Result<Paths> {
    let p = paths().context("no HOME, so there is nowhere to install to")?;
    let current = std::env::current_exe().context("finding the running executable")?;

    for dir in [p.bin.parent(), p.desktop.parent()].into_iter().flatten() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }

    if current != p.bin {
        // Copying to a temporary name and renaming avoids clobbering a running
        // copy of the binary, which on Linux would otherwise be a live file.
        let staging = p.bin.with_extension("new");
        std::fs::copy(&current, &staging)
            .with_context(|| format!("copying {} to {}", current.display(), staging.display()))?;
        std::fs::rename(&staging, &p.bin)
            .with_context(|| format!("installing {}", p.bin.display()))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p.bin, std::fs::Permissions::from_mode(0o755))
                .with_context(|| format!("making {} executable", p.bin.display()))?;
        }
    }

    std::fs::write(&p.desktop, desktop_entry(&p.bin))
        .with_context(|| format!("writing {}", p.desktop.display()))?;

    // The menu and the MIME database both need telling. Best effort: both are
    // rebuilt at next login anyway, this just avoids waiting for one.
    if let Some(dir) = p.desktop.parent() {
        let quiet = |program: &str, arg: &std::path::Path| {
            let _ = std::process::Command::new(program)
                .arg(arg)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        };
        quiet("update-desktop-database", dir);
        if let Some(share) = dir.parent() {
            quiet("update-mime-database", &share.join("mime"));
        }
    }

    Ok(p)
}

/// Make `show` the default application for the image types it handles.
pub fn set_as_default() -> Result<usize> {
    let mut set = 0;
    for mime in MIME_TYPES {
        let ok = std::process::Command::new("xdg-mime")
            .args(["default", DESKTOP_FILE, mime])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            set += 1;
        }
    }
    if set == 0 {
        anyhow::bail!("xdg-mime could not set any association (is xdg-utils installed?)");
    }
    Ok(set)
}

pub fn is_on_path(bin: &Path) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|d| Some(d.as_path()) == bin.parent()))
        .unwrap_or(false)
}

/// How many of our MIME types currently open with `show`.
///
/// Asking `xdg-mime` per type is a handful of subprocesses, so this is only
/// called when the setup panel is actually shown.
pub fn default_viewer_count() -> (usize, usize) {
    let mut ours = 0;
    for mime in MIME_TYPES {
        let out = std::process::Command::new("xdg-mime")
            .args(["query", "default", mime])
            .output();
        if let Ok(out) = out {
            if String::from_utf8_lossy(&out.stdout).trim() == DESKTOP_FILE {
                ours += 1;
            }
        }
    }
    (ours, MIME_TYPES.len())
}


/// Whether to put the setup menu on screen without being asked.
///
/// Only for a build newer than the installed copy. That happens after a
/// rebuild and nowhere else, so it reaches the person who needs to redeploy and
/// never interrupts anyone just looking at a picture. Not being installed at
/// all is mentioned in one line of the readout instead — it is a standing
/// state, and a menu over the image on every launch would be nagging.
pub fn should_open_setup(state: State) -> bool {
    state == State::Stale
}

/// Mention the option once on stderr. Never prompts: when launched from a file
/// manager there is no terminal to answer on, and a blocking read here would
/// hold the window shut. The in-app setup panel does the asking instead.
pub fn note_if_missing() {
    match state() {
        State::NotInstalled => {
            eprintln!("show: not installed — press F1 to set up, or run `show --install`")
        }
        State::Stale => eprintln!(
            "show: this build is newer than the installed copy — press F1, or run `show --install`"
        ),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_desktop_entry_is_well_formed() {
        let entry = desktop_entry(Path::new("/home/u/.local/bin/show"));
        assert!(entry.starts_with("[Desktop Entry]\n"));
        assert!(entry.contains("Exec=/home/u/.local/bin/show %F\n"));
        assert!(entry.contains("Type=Application\n"));
        // Pinning depends on this matching the window's reported class.
        assert!(
            entry.contains(&format!("StartupWMClass={WM_CLASS}\n")),
            "the taskbar cannot pin a window it cannot match: {entry}"
        );
        // Exactly one Type line, or the entry is invalid. Counted by line,
        // because `MimeType=` contains `Type=` as a substring.
        let type_lines = entry.lines().filter(|l| l.starts_with("Type=")).count();
        assert_eq!(type_lines, 1, "{entry}");
        // Every line after the header must be a key=value pair.
        for line in entry.lines().skip(1).filter(|l| !l.is_empty()) {
            assert!(line.contains('='), "malformed line: {line}");
        }
    }

    #[test]
    fn we_only_claim_types_we_can_open() {
        // Claiming a MIME type we cannot decode would make us the default
        // handler for files we then fail on.
        assert!(MIME_TYPES.contains(&"image/jpeg"));
        assert!(MIME_TYPES.contains(&"image/heic"));
        assert!(MIME_TYPES.contains(&"video/mp4"));
        assert!(MIME_TYPES.contains(&"video/x-matroska"));
        assert!(MIME_TYPES.contains(&"image/avif"));
        assert!(!MIME_TYPES.iter().any(|m| m.contains("svg")), "svg is not implemented");
    }

    #[test]
    fn only_a_stale_build_opens_the_menu_by_itself() {
        assert!(should_open_setup(State::Stale), "a rebuilt binary should prompt to redeploy");
        assert!(!should_open_setup(State::Installed), "the normal case must not interrupt");
        assert!(
            !should_open_setup(State::NotInstalled),
            "not installed is a standing state; nagging every launch is worse than a hint"
        );
        assert!(!should_open_setup(State::Unknown));
    }

    #[test]
    fn dates_convert_correctly_across_leap_years() {
        let at = |secs: u64| {
            format_time(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
        };
        assert_eq!(at(0), "1970-01-01 00:00");
        // 2000 is a leap year despite being divisible by 100.
        assert_eq!(at(951_782_400), "2000-02-29 00:00");
        assert_eq!(at(1_709_164_800), "2024-02-29 00:00");
        assert_eq!(at(1_735_689_599), "2024-12-31 23:59");
    }

    #[test]
    fn mime_list_has_no_duplicates_or_empties() {
        let mut seen = std::collections::HashSet::new();
        for m in MIME_TYPES {
            assert!(!m.is_empty());
            assert!(seen.insert(*m), "duplicate mime type {m}");
        }
    }
}
