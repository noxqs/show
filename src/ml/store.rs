//! Where installed things live, and how they get there.
//!
//! In `$XDG_DATA_HOME`, not the cache. Every other store in this program —
//! thumbnails, similarity hashes — is correctly in the cache, because it is
//! derived and cheap to rebuild. A model somebody chose to download, over a
//! metered connection perhaps, is neither.
//!
//! Nothing is ever installed that has not been checked. A download lands in a
//! `.part` file, is digested, compared against the constant in the catalogue,
//! and only then moved into place. A mismatch leaves the `.part` alone and
//! says so; it never silently retries and never installs.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

use super::catalogue::{Artefact, Capability, Spec, spec};

/// Where everything installed lives.
pub fn root() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(base.join("show"))
}

/// Where one version of one capability lives, under a given root.
///
/// The version is the directory name, so an upgrade downloads beside the
/// working copy and a failed one leaves it untouched.
///
/// Every operation below is written against an explicit root, with a thin
/// wrapper that resolves the real one. That is not ceremony: a test that
/// exercises installing or removing must do it somewhere disposable, and the
/// first version of this file had a test that called `remove` on the user's
/// own data directory and deleted what was actually installed there.
pub fn directory_in(root: &Path, spec: &Spec) -> PathBuf {
    root.join("models").join(spec.id.id()).join(spec.version)
}

pub fn directory(spec: &Spec) -> Option<PathBuf> {
    Some(directory_in(&root()?, spec))
}

/// Whether every file of a capability is present at the catalogued version.
pub fn installed_in(root: &Path, id: Capability) -> bool {
    let spec = spec(id);
    let dir = directory_in(root, spec);
    // An artefact that unpacks into several files is installed when they are
    // all there; its own name is a label for the set, not a file. Checking the
    // label meant the card support reported itself missing while sitting on
    // disk, and the only symptom was that nothing ran any faster.
    spec.files.iter().all(|f| {
        if f.members.is_empty() {
            dir.join(f.name).is_file()
        } else {
            f.members.iter().all(|m| member_present(&dir, m))
        }
    })
}

pub fn installed(id: Capability) -> bool {
    root().is_some_and(|r| installed_in(&r, id))
}

/// Bytes on disk for one capability, at any version.
pub fn disk_used_in(root: &Path, id: Capability) -> u64 {
    let mut total = 0;
    let mut stack = vec![root.join("models").join(id.id())];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(entry.path()),
                Ok(_) => total += entry.metadata().map(|m| m.len()).unwrap_or(0),
                Err(_) => {}
            }
        }
    }
    total
}

pub fn disk_used(id: Capability) -> u64 {
    root().map(|r| disk_used_in(&r, id)).unwrap_or(0)
}

/// Remove every version of a capability.
pub fn remove_in(root: &Path, id: Capability) -> Result<()> {
    let dir = root.join("models").join(id.id());
    if dir.exists() {
        std::fs::remove_dir_all(&dir).with_context(|| format!("removing {}", dir.display()))?;
    }
    // The index must not outlive what it describes, or a reinstall would
    // verify against a record of the file it replaced.
    let path = index_path_in(root);
    if let Ok(text) = std::fs::read_to_string(&path) {
        let kept: Vec<&str> =
            text.lines().filter(|l| !l.starts_with(&format!("{}\t", id.id()))).collect();
        let _ = std::fs::write(&path, kept.join("\n") + "\n");
    }
    Ok(())
}

pub fn remove(id: Capability) -> Result<()> {
    remove_in(&root().context("no data directory")?, id)
}

/// The record of what was installed, and what it hashed to at the time.
///
/// This exists because the catalogue's digest and the installed file's digest
/// are not the same thing and cannot be. The catalogue pins **the download** —
/// a wheel, an archive — and that is what gets checked before anything is
/// unpacked, which is the trust boundary that matters. What lands on disk is
/// something taken *out* of that download, so its digest is only knowable
/// after the fact. It is recorded here.
///
/// So the two guard different things, and both are worth having: the catalogue
/// digest says the bytes came from where they claimed, and this says they have
/// not rotted since — a bad disk, a half-finished copy, an edit.
/// Which model the user chose, when more than one can do the same job.
///
/// Kept beside the models rather than in the settings file: it is a fact about
/// what is installed here, it has to be readable by the headless commands, and
/// a preference for something that was removed should simply stop applying.
fn preference_path_in(root: &Path) -> PathBuf {
    root.join("models").join("preferred")
}

pub fn preferred_in(root: &Path) -> Option<Capability> {
    let id = std::fs::read_to_string(preference_path_in(root)).ok()?;
    Capability::from_id(id.trim())
}

pub fn preferred() -> Option<Capability> {
    preferred_in(&root()?)
}

pub fn set_preferred_in(root: &Path, id: Capability) -> Result<()> {
    let path = preference_path_in(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&path, id.id())
        .with_context(|| format!("writing {}", path.display()))
}

pub fn set_preferred(id: Capability) -> Result<()> {
    let root = root().context("no data directory")?;
    set_preferred_in(&root, id)
}

fn index_path_in(root: &Path) -> PathBuf {
    root.join("models").join("index")
}

/// Note what an installed file hashed to.
fn record(root: &Path, id: Capability, name: &str, digest: &str, bytes: u64) -> Result<()> {
    let path = index_path_in(root);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let version = spec(id).version;
    let mut lines: Vec<String> = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .lines()
        .filter(|l| {
            // Replace any earlier line for this exact file, so reinstalling
            // does not leave two records disagreeing.
            let mut parts = l.split('\t');
            !(parts.next() == Some(id.id())
                && parts.next() == Some(version)
                && parts.next() == Some(name))
        })
        .map(str::to_string)
        .collect();
    lines.push(format!("{}\t{version}\t{name}\t{digest}\t{bytes}", id.id()));
    std::fs::write(&path, lines.join("\n") + "\n")
        .with_context(|| format!("writing {}", path.display()))
}

/// What the index says about one capability at its catalogued version.
fn recorded(root: &Path, id: Capability) -> Vec<(String, String)> {
    let path = index_path_in(root);
    let version = spec(id).version;
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let mut parts = line.split('\t');
            let (who, when, name, digest) =
                (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
            (who == id.id() && when == version)
                .then(|| (name.to_string(), digest.to_string()))
        })
        .collect()
}

/// Check that what is installed is still what was installed.
pub fn verify_in(root: &Path, id: Capability) -> Result<()> {
    let spec = spec(id);
    let dir = directory_in(root, spec);
    let recorded = recorded(root, id);
    anyhow::ensure!(!recorded.is_empty(), "{} is not installed", spec.id.id());

    for file in spec.files {
        if !file.members.is_empty() {
            for member in file.members {
                // By the name it actually landed under: a library carries its
                // version in the archive, and the catalogue names it without.
                let name = installed_name(&dir, member)
                    .with_context(|| format!("{member} is missing"))?;
                verify_one(root, &dir, id, &name)?;
            }
            continue;
        }
        let path = dir.join(file.name);
        anyhow::ensure!(path.is_file(), "{} is missing", file.name);
        let expected = recorded
            .iter()
            .find(|(name, _)| name == file.name)
            .map(|(_, digest)| digest.clone())
            .with_context(|| format!("{} was never recorded as installed", file.name))?;
        let found = crate::sha256::file(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        anyhow::ensure!(
            found == expected,
            "{} has changed since it was installed: expected {expected}, found {found}",
            file.name
        );
    }
    Ok(())
}

pub fn verify(id: Capability) -> Result<()> {
    verify_in(&root().context("no data directory")?, id)
}

/// How an install is getting on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// Bytes fetched of the total expected.
    Fetching(u64, u64),
    /// Reading back what arrived, to check it.
    Verifying,
    /// Taking one file out of an archive.
    Unpacking,
}

/// Download, check and install one capability.
///
/// `progress` is called often enough to drive a bar and rarely enough not to
/// be a cost of its own; it is the caller's job to throttle any redraw.
pub fn install_in(root: &Path, id: Capability, on: &mut dyn FnMut(Step)) -> Result<()> {
    let spec = spec(id);
    let dir = directory_in(root, spec);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    for file in spec.files {
        let final_path = dir.join(file.name);
        // A set of files is done when the last of them is there; a single one
        // when it is.
        let done = if file.members.is_empty() {
            final_path.is_file()
        } else {
            file.members.iter().all(|m| dir.join(basename(m)).is_file())
        };
        if done {
            continue;
        }
        // Named for the artefact, so two interrupted installs cannot collide.
        let part = dir.join(format!("{}.part", file.name));
        fetch(file, &part, on)?;

        on(Step::Verifying);
        let found = crate::sha256::file(&part).with_context(|| format!("reading {}", part.display()))?;
        if found != file.sha256 {
            // Deliberately left in place: a mismatch is worth being able to
            // look at, and removing the evidence would make a mirror serving
            // the wrong bytes indistinguishable from a flaky connection.
            return Err(anyhow!(
                "{} does not match the catalogue — expected {}, got {found}. \
                 The partial download is at {}",
                file.name,
                file.sha256,
                part.display()
            ));
        }

        if !file.members.is_empty() {
            on(Step::Unpacking);
            let mut archive = std::fs::File::open(&part)?;
            let entries = crate::zip::entries(&mut archive)
                .with_context(|| format!("reading {}", part.display()))?;
            for member in file.members {
                let entry = find_member(&entries, member)
                    .with_context(|| format!("{member} is not in {}", file.url))?;
                let name = basename(&entry.name);
                let staged = dir.join(format!("{name}.staged"));
                crate::zip::extract(&mut archive, entry, &staged)?;
                std::fs::rename(&staged, dir.join(&name))?;

                let digest = crate::sha256::file(&dir.join(&name))?;
                let bytes = std::fs::metadata(dir.join(&name)).map(|m| m.len()).unwrap_or(0);
                record(root, id, &name, &digest, bytes)?;
            }
            let _ = std::fs::remove_file(&part);
            continue;
        }

        match file.member {
            None => std::fs::rename(&part, &final_path)?,
            Some(member) => {
                on(Step::Unpacking);
                let mut archive = std::fs::File::open(&part)?;
                let entries = crate::zip::entries(&mut archive)
                    .with_context(|| format!("reading {}", part.display()))?;
                // Matched on the last path element so an archive that nests
                // its payload differently still works, and a name that merely
                // contains the string does not.
                let entry = find_member(&entries, member)
                    .with_context(|| format!("{member} is not in {}", file.url))?;

                let staged = dir.join(format!("{}.staged", file.name));
                crate::zip::extract(&mut archive, entry, &staged)?;
                std::fs::rename(&staged, &final_path)?;
                let _ = std::fs::remove_file(&part);
            }
        }

        // What actually landed, so a later check can tell rot from a bad
        // download.
        let digest = crate::sha256::file(&final_path)?;
        let bytes = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
        record(root, id, file.name, &digest, bytes)?;
    }
    Ok(())
}

pub fn install(id: Capability, on: &mut dyn FnMut(Step)) -> Result<()> {
    install_in(&root().context("no data directory")?, id, on)
}

/// Check one installed file against what was recorded for it.
fn verify_one(root: &Path, dir: &Path, id: Capability, name: &str) -> Result<()> {
    let path = dir.join(name);
    anyhow::ensure!(path.is_file(), "{name} is missing");
    let recorded = recorded(root, id);
    let expected = recorded
        .iter()
        .find(|(had, _)| had == name)
        .map(|(_, digest)| digest.clone())
        .with_context(|| format!("{name} was never recorded as installed"))?;
    let found =
        crate::sha256::file(&path).with_context(|| format!("reading {}", path.display()))?;
    anyhow::ensure!(
        found == expected,
        "{name} has changed since it was installed: expected {expected}, found {found}"
    );
    Ok(())
}

/// Whether a member of an archive landed in `dir`.
///
/// By prefix, exactly as the extraction matches it: a library is named for
/// its version inside the archive — `libonnxruntime.so.1.22.0` for a member
/// written as `libonnxruntime.so` — and the catalogue should not have to be
/// edited for a version bump that the archive already describes.
fn member_present(dir: &Path, member: &str) -> bool {
    installed_name(dir, member).is_some()
}

/// The name a member of an archive actually landed under.
fn installed_name(dir: &Path, member: &str) -> Option<String> {
    let wanted = basename(member);
    let names: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| e.file_name().to_str().map(|n| n.to_string()))
        .collect();
    // Exactly, if anything matches exactly. The prefix is for a library that
    // carries its version in the archive and is named here without it, but
    // once one member's name is a prefix of another's — `x.onnx` beside
    // `x.onnx.data` — a prefix match alone picks whichever the directory
    // happens to list first.
    names
        .iter()
        .find(|name| **name == wanted)
        .or_else(|| names.iter().find(|name| name.starts_with(&wanted)))
        .cloned()
}

/// Find an archive member by name, exactly where possible.
///
/// Same rule, and the same reason: `sam3_image_encoder.onnx` is a prefix of
/// `sam3_image_encoder.onnx.data`, so a first-prefix-match would extract the
/// weights under the graph's name and never extract the graph — a failure that
/// only shows up much later, when the model will not open.
fn find_member<'a>(entries: &'a [crate::zip::Entry], member: &str) -> Option<&'a crate::zip::Entry> {
    let last = |name: &str| basename(name);
    crate::zip::find(entries, |name| last(name) == member)
        .or_else(|| crate::zip::find(entries, |name| last(name).starts_with(member)))
}

/// The last path element, which is what an extracted file is called.
fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

/// Fetch one artefact to `into`.
fn fetch(file: &Artefact, into: &Path, on: &mut dyn FnMut(Step)) -> Result<()> {
    use std::io::{Read, Write};

    let response = ureq::get(file.url)
        .call()
        .with_context(|| format!("fetching {}", file.url))?;
    let expected = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(file.bytes);

    let mut reader = response.into_body().into_reader();
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(into).with_context(|| format!("creating {}", into.display()))?,
    );

    let mut buffer = vec![0u8; 1 << 16];
    let mut done = 0u64;
    let mut last = std::time::Instant::now();
    loop {
        let read = reader.read(&mut buffer).context("reading the download")?;
        if read == 0 {
            break;
        }
        out.write_all(&buffer[..read])?;
        done += read as u64;
        // Often enough to look live, rarely enough not to be a cost.
        if last.elapsed() >= std::time::Duration::from_millis(200) {
            last = std::time::Instant::now();
            on(Step::Fetching(done, expected));
        }
    }
    out.flush()?;
    on(Step::Fetching(done, expected));

    anyhow::ensure!(
        done == expected,
        "{}: expected {expected} bytes, received {done}",
        file.name
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Every member the catalogue names must be in the archive it names.
    ///
    /// A wrong member name costs the user a three-gigabyte download and then
    /// fails at unpacking, which is the worst possible moment to find out. The
    /// archive is not in the repository, so this is opt-in:
    /// `SAM3_ZIP=/path/to/sam3_vit_h.zip cargo test -- --ignored`.
    #[test]
    #[ignore = "needs the SAM 3 archive; set SAM3_ZIP"]
    fn the_archive_holds_every_member_the_catalogue_names() {
        let Ok(path) = std::env::var("SAM3_ZIP") else {
            eprintln!("skipped: set SAM3_ZIP to the downloaded archive");
            return;
        };
        let spec = spec(Capability::TextSegment);
        let mut file = std::fs::File::open(&path).expect("the archive");
        let entries = crate::zip::entries(&mut file).expect("reading the archive");
        for artefact in spec.files {
            for member in artefact.members {
                let found = find_member(&entries, member)
                    .unwrap_or_else(|| panic!("{member} is not in {path}"));
                assert_eq!(
                    basename(&found.name),
                    *member,
                    "{member} resolved to a different file"
                );
            }
        }
    }

    #[test]
    fn a_member_name_that_prefixes_another_still_finds_its_own_file() {
        // `sam3_image_encoder.onnx` is a prefix of `sam3_image_encoder.onnx.data`,
        // and the archive may list either first. A first-prefix-match would
        // extract the weights under the graph's name and never extract the
        // graph — which does not fail until the model is opened, somewhere
        // else entirely.
        let entries = vec![
            crate::zip::Entry::named("sam3_image_encoder.onnx.data"),
            crate::zip::Entry::named("sam3_image_encoder.onnx"),
        ];
        let found = find_member(&entries, "sam3_image_encoder.onnx").expect("found");
        assert_eq!(found.name, "sam3_image_encoder.onnx", "the weights were taken for the graph");

        let data = find_member(&entries, "sam3_image_encoder.onnx.data").expect("found");
        assert_eq!(data.name, "sam3_image_encoder.onnx.data");

        // The prefix rule still applies where nothing matches exactly: a
        // library carries its version in the archive and is named here
        // without it.
        let versioned = vec![crate::zip::Entry::named("libonnxruntime.so.1.16.3")];
        assert_eq!(
            find_member(&versioned, "libonnxruntime.so").expect("found").name,
            "libonnxruntime.so.1.16.3"
        );
    }

    use super::*;

    /// A disposable root. Every mutating test uses one of these, because the
    /// first version of this file did not and deleted what the user had
    /// actually installed.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("show-store-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch root");
        dir
    }

    /// Put a plausible installation in place without downloading anything.
    fn fake_install(root: &Path, id: Capability, contents: &[u8]) {
        let spec = spec(id);
        let dir = directory_in(root, spec);
        std::fs::create_dir_all(&dir).expect("create");
        for file in spec.files {
            let path = dir.join(file.name);
            std::fs::write(&path, contents).expect("write");
            let digest = crate::sha256::hex(contents);
            record(root, id, file.name, &digest, contents.len() as u64).expect("record");
        }
    }

    #[test]
    fn the_store_is_in_data_not_cache() {
        // A model somebody chose to download is not derived and not cheap to
        // rebuild, so it does not belong with the thumbnails.
        let Some(root) = root() else { return };
        let shown = root.display().to_string();
        assert!(shown.contains("share"), "{shown} is not under a data directory");
        assert!(!shown.contains(".cache"), "{shown} is in the cache");
    }

    #[test]
    fn the_version_is_the_directory_so_upgrades_do_not_overwrite() {
        let root = scratch("version");
        let spec = spec(Capability::Runtime);
        let dir = directory_in(&root, spec);
        assert!(dir.ends_with(spec.version), "{} does not end with the version", dir.display());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn removing_something_never_installed_is_not_an_error() {
        // The page offers Del on every row; it must not fail on an empty one.
        let root = scratch("remove-empty");
        assert!(remove_in(&root, Capability::Runtime).is_ok());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn verifying_something_missing_says_so_rather_than_passing() {
        let root = scratch("verify-missing");
        let err = verify_in(&root, Capability::Runtime).expect_err("must not pass when absent");
        let text = format!("{err:#}");
        assert!(
            text.contains("missing") || text.contains("not installed"),
            "unhelpful message: {text}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_installed_file_verifies_and_a_changed_one_does_not() {
        let root = scratch("verify-change");
        fake_install(&root, Capability::Runtime, b"the original bytes");
        assert!(installed_in(&root, Capability::Runtime));
        verify_in(&root, Capability::Runtime).expect("should verify as installed");

        // Rot: the file changes without the index changing with it.
        let spec = spec(Capability::Runtime);
        let path = directory_in(&root, spec).join(spec.files[0].name);
        std::fs::write(&path, b"something else entirely").expect("overwrite");
        let err = verify_in(&root, Capability::Runtime).expect_err("corruption must be caught");
        assert!(format!("{err:#}").contains("changed since"), "{err:#}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn removing_takes_the_index_entry_with_it() {
        let root = scratch("remove-index");
        fake_install(&root, Capability::Runtime, b"bytes");
        remove_in(&root, Capability::Runtime).expect("remove");
        assert!(!installed_in(&root, Capability::Runtime));
        // A stale record would make the next install verify against the file
        // it replaced.
        assert!(
            recorded(&root, Capability::Runtime).is_empty(),
            "the index outlived what it described"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn disk_used_counts_what_is_there() {
        let root = scratch("disk");
        assert_eq!(disk_used_in(&root, Capability::Runtime), 0);
        fake_install(&root, Capability::Runtime, &vec![0u8; 4096]);
        assert_eq!(disk_used_in(&root, Capability::Runtime), 4096);
        let _ = std::fs::remove_dir_all(&root);
    }
}
