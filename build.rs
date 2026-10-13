//! Stamps each release build with a number that only goes up.
//!
//! Two builds of this program are otherwise very hard to tell apart: the
//! version in `Cargo.toml` changes when someone remembers to change it, which
//! is never. A counter that increments whenever the source actually changes
//! gives every binary an identity, which matters here because the program
//! compares itself against its installed copy and offers to redeploy.
//!
//! Debug builds do not consume numbers. Only a release build is something that
//! can be installed, so only a release build earns one.

use std::path::Path;

const COUNTER: &str = "build-number";

fn main() {
    // Rerun when the sources change, which is exactly when a new binary would
    // be produced. A build that recompiles nothing does not need a new number.
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");

    let release = std::env::var("PROFILE").as_deref() == Ok("release");
    let previous: u64 = std::fs::read_to_string(COUNTER)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    let number = if release { previous + 1 } else { previous };
    if release && std::fs::write(COUNTER, format!("{number}\n")).is_err() {
        // A read-only checkout should still build; it just cannot count.
        println!("cargo:warning=could not update {COUNTER}");
    }

    println!("cargo:rustc-env=SHOW_BUILD={number}");
    println!("cargo:rustc-env=SHOW_BUILD_DATE={}", today());

    // The commit, when there is one, so a build can be traced back.
    println!("cargo:rustc-env=SHOW_COMMIT={}", commit().unwrap_or_default());
}

/// Today's date as YYYY-MM-DD, from the system clock.
fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);

    // Howard Hinnant's civil-from-days, the same one the viewer uses.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

fn commit() -> Option<String> {
    if !Path::new(".git").exists() {
        return None;
    }
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}
