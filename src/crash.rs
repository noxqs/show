//! Saying so when something goes wrong, rather than just vanishing.
//!
//! Release builds abort on panic, and launched from the desktop there is no
//! terminal to read the message in: the window simply disappears. So the
//! panic hook writes the message to a log and puts it in front of the user in
//! a window of its own — run as a separate program, because this process is
//! about to abort and cannot be trusted to draw anything itself.

use std::io::Write;
use std::path::PathBuf;

/// Where the last crash is written.
pub fn log_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("show").join("crash.log"))
}

/// Install the hook. The default one still runs first, so a terminal gets the
/// usual message and backtrace as before.
pub fn install() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default(info);

        let what = match info.payload().downcast_ref::<&str>() {
            Some(s) => s.to_string(),
            None => info
                .payload()
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "an unknown error".to_string()),
        };
        let place = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        let thread = std::thread::current().name().unwrap_or("unnamed").to_string();
        let backtrace = std::backtrace::Backtrace::force_capture();

        let log = log_path();
        if let Some(path) = log.as_ref() {
            let _ = std::fs::create_dir_all(path.parent().unwrap_or(path));
            if let Ok(mut file) = std::fs::File::create(path) {
                let _ = writeln!(
                    file,
                    "show {} crashed\nthread: {thread}\nat: {place}\n\n{what}\n\n{backtrace}",
                    env!("CARGO_PKG_VERSION")
                );
            }
        }

        let mut text = format!("show stopped because of an error.\n\n{what}\n\nat {place} (thread {thread})");
        if let Some(path) = log.as_ref() {
            text.push_str(&format!("\n\nDetails: {}", path.display()));
        }
        show_dialog(&text);
    }));
}

/// Put the message on screen with whatever the desktop has. Spawned, not
/// waited for: the dialog outlives this process, which is about to end.
fn show_dialog(text: &str) {
    if std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none() {
        return;
    }
    // Long messages, wgpu's in particular, make an unreadable dialog.
    let text: String = if text.chars().count() > 3000 {
        text.chars().take(3000).chain("…".chars()).collect()
    } else {
        text.to_string()
    };
    let attempts: [(&str, Vec<&str>); 4] = [
        ("zenity", vec!["--error", "--no-markup", "--title=show", "--width=560", "--text", &text]),
        ("yad", vec!["--error", "--title=show", "--width=560", "--text", &text]),
        ("xmessage", vec!["-center", &text]),
        ("notify-send", vec!["-u", "critical", "show crashed", &text]),
    ];
    for (program, args) in attempts {
        let spawned = std::process::Command::new(program)
            .args(&args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        if spawned.is_ok() {
            return;
        }
    }
}
