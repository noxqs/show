//! Finding ONNX Runtime and telling the truth about it.
//!
//! Modelled on `src/heif.rs`. The library is loaded at run time, never linked,
//! so a build with inference compiled in still starts on a machine that has
//! never downloaded it — the plugins page simply says so. That is the property
//! the whole ML design is arranged around, and this module is where it is kept.
//!
//! `ort` does the `dlopen` itself, given a path in `ORT_DYLIB_PATH`. All this
//! does is decide which path, and answer questions about it.

use std::path::PathBuf;

use crate::ml::catalogue::{Capability, spec};

/// What can be said about the runtime on this machine.
#[derive(Clone, Debug, PartialEq)]
pub enum Availability {
    /// Ready to use, at this path.
    Ready(PathBuf),
    /// Not downloaded. The plugins page offers to fetch it.
    NotInstalled,
    /// Present but unusable, with whatever the loader said.
    Broken(String),
}

impl Availability {
    pub fn path(&self) -> Option<&PathBuf> {
        match self {
            Availability::Ready(path) => Some(path),
            _ => None,
        }
    }
}

/// Where the runtime is, if it is anywhere.
///
/// Our own copy first. A system one is accepted as a fallback, but it is not
/// preferred: the version we install is the version the catalogue was written
/// against, and a distribution's may be older or newer in ways nobody has
/// checked.
pub fn locate() -> Availability {
    // The card's runtime first, when it is installed: it is a different build
    // of the same library, and only that one knows how to use a card. It is
    // used whenever it is there, which is what makes installing it the way to
    // ask for it.
    if let Some(path) = gpu_runtime() {
        return Availability::Ready(path);
    }
    let installed = crate::ml::store::directory(spec(Capability::Runtime))
        .map(|dir| dir.join("libonnxruntime.so"))
        .filter(|p| p.is_file());
    if let Some(path) = installed {
        // A truncated download leaves a file that exists and cannot load. The
        // failure would otherwise arrive much later, from inside `dlopen`,
        // where it is far harder to connect to its cause.
        return match std::fs::metadata(&path).map(|m| m.len()) {
            Ok(len) if len > 1 << 20 => Availability::Ready(path),
            Ok(len) => Availability::Broken(format!(
                "{} is only {len} bytes; reinstall it from F8",
                path.display()
            )),
            Err(e) => Availability::Broken(format!("{}: {e}", path.display())),
        };
    }

    for candidate in [
        "/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
        "/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1",
        "/usr/local/lib/libonnxruntime.so",
    ] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Availability::Ready(path);
        }
    }
    Availability::NotInstalled
}

/// The graphics-card build of the runtime, if it is installed.
///
/// The wheel names it with its version, so the file is found rather than
/// assumed: a version bump in the catalogue should not need a change here.
pub fn gpu_runtime() -> Option<PathBuf> {
    let dir = crate::ml::store::directory(spec(Capability::Cuda))?;
    let mut names: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("libonnxruntime.so"))
        })
        .collect();
    names.sort();
    names.pop()
}

/// Make cuDNN findable before the runtime looks for it.
///
/// The CUDA provider is linked against cuDNN, which lives in our own
/// directory rather than anywhere the dynamic linker searches. Opening those
/// libraries first, globally, means the linker finds them already loaded when
/// it comes to resolve the provider — the usual answer to shipping a library
/// somewhere unusual, and the alternative would be telling people to set
/// `LD_LIBRARY_PATH` before starting a picture viewer.
///
/// They are deliberately never closed: the runtime holds pointers into them
/// for as long as it lives, which is the life of the process.
#[cfg(feature = "ml")]
fn preload_cudnn(dir: &std::path::Path) -> Result<(), String> {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};

    // Dependencies before dependants: the dispatcher opens the rest by name,
    // and finds them already there.
    const ORDER: [&str; 10] = [
        "libcudnn_graph.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_tensor_ir.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_ext.so.9",
        "libcudnn.so.9",
    ];
    for name in ORDER {
        let path = dir.join(name);
        if !path.is_file() {
            return Err(format!("{name} is missing; reinstall the card support from F8"));
        }
        let opened = unsafe { Library::open(Some(&path), RTLD_NOW | RTLD_GLOBAL) };
        match opened {
            Ok(library) => std::mem::forget(library),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        }
    }
    Ok(())
}

/// Whether the models should run on the card, and why not when they should not.
///
/// Answered once. Everything it checks is a property of this machine and this
/// installation, neither of which changes while the program runs — except by
/// installing the card support, which is worth a restart to pick up rather
/// than a check on every session.
#[cfg(feature = "ml")]
pub fn on_card() -> Option<&'static str> {
    use std::sync::OnceLock;
    static ANSWER: OnceLock<Option<String>> = OnceLock::new();

    let answer = ANSWER.get_or_init(|| {
        let dir = crate::ml::store::directory(spec(Capability::Cuda))?;
        if !crate::ml::store::installed(Capability::Cuda) {
            return None;
        }
        match preload_cudnn(&dir) {
            Ok(()) => None,
            Err(e) => Some(e),
        }
    });
    match answer {
        // Installed and loaded: nothing to say, and the card is in use.
        None if crate::ml::store::installed(Capability::Cuda) => None,
        None => Some("not installed"),
        Some(why) => Some(why),
    }
}

/// A session builder, with the card if there is one.
///
/// Every session in this program is built here, so "runs on the card" is one
/// decision rather than one per model — and a machine that cannot use the card
/// falls back rather than failing, because falling back is slower and failing
/// is broken.
#[cfg(feature = "ml")]
pub fn session(threads: usize) -> anyhow::Result<ort::session::builder::SessionBuilder> {
    prepare()?;
    let mut builder = ort::session::Session::builder()
        .map_err(|e| anyhow::anyhow!("building a session: {e}"))?;

    if on_card().is_none() {
        // Loudly: without this the runtime shrugs, runs on the processor, and
        // the only sign is that nothing got faster — which is exactly what
        // happened the first time this was wired up.
        let card = ort::ep::CUDA::default().build().error_on_failure();
        match builder.clone().with_execution_providers([card]) {
            Ok(with_card) => return Ok(with_card),
            Err(e) => eprintln!("show: the card was not usable ({e}); running on the processor"),
        }
    }
    builder = builder
        .with_intra_threads(threads)
        .map_err(|e| anyhow::anyhow!("setting the thread count: {e}"))?;
    Ok(builder)
}

/// Load a model, falling back to the processor when the card cannot run it.
///
/// A card's provider is not a superset of the processor's. It has to have a
/// kernel for every operator in the graph *at that graph's opset*, and where
/// it does not, the runtime does not quietly run those nodes on the processor
/// — it refuses to create the session at all. SAM 3's image encoder is the
/// first model here to hit that: `Squeeze` at opset 21, which the installed
/// CUDA build stops short of.
///
/// So whether a model runs on the card is only knowable by trying, and trying
/// has to be allowed to fail without taking the feature with it. Slower is a
/// far better answer than "not installed".
/// Returns the session and whether the card took it, so a caller loading
/// several graphs of one model can stop asking after the first refusal —
/// the runtime prints its own page of diagnostics each time it declines, and
/// three copies of it on every launch is not a thing to make somebody read.
pub fn commit(
    path: &std::path::Path,
    threads: usize,
) -> anyhow::Result<(ort::session::Session, bool)> {
    // The builder first, so only the load itself is inside the measurement —
    // and so a runtime that cannot be configured at all still fails here
    // rather than being taken for a card that refused.
    let mut builder = session(threads)?;
    match crate::vram::charging(path, || builder.commit_from_file(path)) {
        Ok(session) => Ok((session, on_card().is_none())),
        Err(card) => {
            // Only worth saying when there was a card to lose: on a machine
            // without one this path is the only path there was.
            if on_card().is_none() {
                // The runtime has already printed the detail, at length.
                let _ = card;
                eprintln!(
                    "show: the card will not run {}; using the processor for this model",
                    path.file_name().unwrap_or_default().to_string_lossy()
                );
            }
            Ok((commit_on_cpu(path, threads)?, false))
        }
    }
}

/// The same, without asking the card at all.
pub fn commit_on_cpu(
    path: &std::path::Path,
    threads: usize,
) -> anyhow::Result<ort::session::Session> {
    prepare()?;
    ort::session::Session::builder()
        .map_err(|e| anyhow::anyhow!("building a session: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow::anyhow!("setting the thread count: {e}"))?
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))
}

/// Point `ort` at the library, once.
///
/// `ort` reads `ORT_DYLIB_PATH` when it first loads, so this has to happen
/// before any session is built and only needs to happen once.
#[cfg(feature = "ml")]
pub fn prepare() -> anyhow::Result<PathBuf> {
    use std::sync::OnceLock;
    static READY: OnceLock<Option<PathBuf>> = OnceLock::new();

    let path = READY
        .get_or_init(|| {
            let found = locate().path().cloned()?;
            // Safety: called before any inference thread exists, and the
            // variable is set once. `ort` reads it on its first load.
            unsafe { std::env::set_var("ORT_DYLIB_PATH", &found) };
            Some(found)
        })
        .clone();

    path.ok_or_else(|| {
        anyhow::anyhow!(
            "ONNX Runtime is not installed — press F8, or run `show --plugins install runtime`"
        )
    })
}

/// A one-line description for the plugins page and the probe.
pub fn describe() -> String {
    match locate() {
        Availability::Ready(path) => format!("ready at {}", path.display()),
        Availability::NotInstalled => "not installed".to_string(),
        Availability::Broken(why) => format!("broken: {why}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absence_is_reported_rather_than_guessed_at() {
        // Whatever this machine has, `locate` must return something the page
        // can render and never panic.
        let found = locate();
        match &found {
            Availability::Ready(path) => assert!(path.is_file(), "{} is not a file", path.display()),
            Availability::NotInstalled | Availability::Broken(_) => {}
        }
        assert!(!describe().is_empty());
    }

    #[test]
    fn a_truncated_library_is_called_broken_rather_than_ready() {
        // What a half-finished download leaves behind: a file that exists and
        // cannot possibly load.
        let dir = std::env::temp_dir().join(format!("show-rt-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("libonnxruntime.so");
        std::fs::write(&path, b"not really a library").expect("write");
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        assert!(size < (1 << 20), "the test file should be small");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn our_own_copy_is_preferred_to_the_system_one() {
        // The installed version is the one the catalogue was written against.
        // This only checks the ordering when we do have a copy — and the
        // card's build outranks both, so it is only a statement about the two
        // that are on the processor.
        if gpu_runtime().is_some() {
            return;
        }
        let ours = crate::ml::store::directory(spec(Capability::Runtime))
            .map(|d| d.join("libonnxruntime.so"));
        if let Some(ours) = ours.filter(|p| p.is_file()) {
            assert_eq!(locate(), Availability::Ready(ours));
        }
    }

    #[test]
    fn the_cards_runtime_outranks_the_processors() {
        // Installing the card support is how you ask for the card, so its
        // build of the runtime is the one that gets loaded.
        if let Some(card) = gpu_runtime() {
            assert_eq!(locate(), Availability::Ready(card));
        }
    }
}
