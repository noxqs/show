//! A persistent thumbnail cache, so a folder is only ever decoded once.
//!
//! Filling a grid cell costs 2.7 ms for a JPEG and 808 ms for an iPhone HEIC.
//! Across a screenful that is the difference between instant and six seconds,
//! and it is paid again every time the folder is opened. Storing the small
//! version removes it permanently.
//!
//! Thumbnails are stored as JPEG, which looks backwards for a cache until you
//! notice that JPEG is by some distance our fastest decoder — 631 MPix/s
//! against PNG's 163 — and a fifteenth the size on disk. A 512 pixel thumbnail
//! reloads in well under a millisecond.
//!
//! Everything lives under one directory so it can be found, inspected and
//! deleted as a unit.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Longest edge of a stored thumbnail. Serves any grid cell up to this size;
/// larger cells fall back to decoding the real image, which is warranted at
/// that point anyway.
pub const THUMB_SIZE: u32 = 512;

/// How much disk the thumbnail store may occupy before the oldest are dropped.
///
/// At roughly 20 KB a thumbnail this holds around a hundred thousand of them,
/// which is more pictures than most people own and small enough that nobody
/// notices it in their cache directory.
pub const DISK_BUDGET: u64 = 2 << 30;

/// Quality for the stored JPEG. High enough that the grid does not look
/// obviously degraded, low enough to stay around 20 KB.
const QUALITY: u8 = 85;

/// A stable 128-bit hash of a path, for naming cache files.
///
/// FNV-1a rather than the standard library's hasher: `DefaultHasher` is
/// explicitly not stable between releases, and a cache keyed by it would
/// silently invalidate itself on a compiler upgrade.
pub(crate) fn hash_path(path: &Path) -> String {
    fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        h
    }
    let bytes = path.to_string_lossy().into_owned().into_bytes();
    // Two passes in opposite directions give 128 bits from one simple function.
    let a = fnv1a(bytes.iter().copied());
    let b = fnv1a(bytes.iter().rev().copied());
    format!("{a:016x}{b:016x}")
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    modified: u64,
    len: u64,
}

/// A thumbnail as stored: shrunk to [`THUMB_SIZE`] and JPEG-encoded. The slow
/// part of storing one, and needing no cache, so done before taking it.
pub fn encode(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    // Only ever shrink: a small image is already its own thumbnail.
    let factor = crate::scale::factor_for_limit(width, height, THUMB_SIZE);
    let (pixels, w, h) = if factor > 1 {
        crate::scale::box_downscale(rgba, width, height, factor)
    } else {
        (rgba.to_vec(), width, height)
    };

    // JPEG has no alpha, so anything transparent is composited onto a
    // neutral grey rather than being written as garbage.
    let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
    for px in pixels.chunks_exact(4) {
        let a = px[3] as u32;
        for c in 0..3 {
            rgb.push(((px[c] as u32 * a + 128 * (255 - a)) / 255) as u8);
        }
    }

    let mut encoded = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, QUALITY)
        .encode(&rgb, w, h, image::ExtendedColorType::Rgb8)
        .context("encoding a thumbnail")?;
    Ok(encoded)
}

/// Read and decode a stored thumbnail — from a path [`Cache::stored`] gave,
/// with the cache no longer held.
pub fn read(file: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let bytes = std::fs::read(file).ok()?;
    let decoded = crate::decode::decode(&bytes).ok()?;
    Some((decoded.rgba, decoded.width, decoded.height))
}

fn file_stamp(path: &Path) -> Option<Entry> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(Entry { modified, len: meta.len() })
}

/// The desktop's own thumbnail store, per the XDG specification.
///
/// The file manager has usually been through the folder already, so this is a
/// cache we get for nothing: no generation pass, no disk of our own. Files are
/// named by the MD5 of the source's `file://` URI and carry the source's
/// modification time inside, which is what makes a stale one detectable.
///
/// Read only. Writing here means matching the specification's metadata exactly,
/// and a malformed entry would poison the cache the desktop relies on.
pub mod xdg {
    use super::*;

    /// Sizes the specification defines, largest first.
    const DIRS: [(&str, u32); 4] =
        [("xx-large", 1024), ("x-large", 512), ("large", 256), ("normal", 128)];

    fn base() -> Option<PathBuf> {
        let cache = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
        Some(cache.join("thumbnails"))
    }

    /// `file://` URI, escaped the way GLib escapes it.
    ///
    /// The digest is taken over this string, so the escaping has to match the
    /// file manager's byte for byte or nothing is ever found.
    pub fn file_uri(path: &Path) -> String {
        // Everything GLib leaves unescaped in a path: unreserved characters,
        // the sub-delimiters, and the separators.
        const SAFE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~!$&'()*+,;=:@/";

        let mut uri = String::from("file://");
        for byte in path.to_string_lossy().as_bytes() {
            if SAFE.contains(byte) {
                uri.push(*byte as char);
            } else {
                uri.push_str(&format!("%{byte:02X}"));
            }
        }
        uri
    }

    /// Where a thumbnail of `path` would live, at each size.
    pub fn candidates(path: &Path) -> Vec<(PathBuf, u32)> {
        let Some(base) = base() else { return Vec::new() };
        let name = format!("{}.png", crate::md5::hex(file_uri(path).as_bytes()));
        DIRS.iter().map(|(dir, size)| (base.join(dir).join(&name), *size)).collect()
    }

    /// Load the desktop's thumbnail, if there is a valid one at least
    /// `min_size` across.
    pub fn load(path: &Path, min_size: u32) -> Option<(Vec<u8>, u32, u32)> {
        let stamp = file_stamp(path)?;

        for (candidate, size) in candidates(path) {
            if size < min_size || !candidate.exists() {
                continue;
            }
            let Ok(bytes) = std::fs::read(&candidate) else { continue };
            if !mtime_matches(&bytes, stamp.modified) {
                // The source changed since the desktop made this; the file
                // manager will refresh it, but we must not show it.
                continue;
            }
            if let Ok(decoded) = crate::decode::decode(&bytes) {
                return Some((decoded.rgba, decoded.width, decoded.height));
            }
        }
        None
    }

    /// Check the `Thumb::MTime` text chunk against the source's timestamp.
    fn mtime_matches(png_bytes: &[u8], modified: u64) -> bool {
        let decoder = png::Decoder::new(png_bytes);
        let Ok(reader) = decoder.read_info() else { return false };
        let info = reader.info();

        for chunk in &info.uncompressed_latin1_text {
            if chunk.keyword == "Thumb::MTime" {
                return chunk.text.trim().parse::<u64>() == Ok(modified);
            }
        }
        // Without the tag there is no way to tell whether it is current, and a
        // stale thumbnail shown as the current image is worse than a slow one.
        false
    }
}

pub struct Cache {
    dir: PathBuf,
    index: HashMap<PathBuf, Entry>,
    dirty: bool,
}

impl Cache {
    pub fn dir() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
        Some(base.join("show").join("thumbs"))
    }

    pub fn open() -> Option<Cache> {
        let dir = Cache::dir()?;
        let mut index = HashMap::new();

        if let Ok(text) = std::fs::read_to_string(dir.join("index")) {
            for line in text.lines() {
                let mut f = line.split('\t');
                let (Some(m), Some(l), Some(p)) = (f.next(), f.next(), f.next()) else { continue };
                let (Ok(modified), Ok(len)) = (m.parse(), l.parse()) else { continue };
                index.insert(PathBuf::from(p), Entry { modified, len });
            }
        }
        Some(Cache { dir, index, dirty: false })
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// True when this file has a thumbnail that still matches it.
    pub fn has(&self, path: &Path) -> bool {
        match (self.index.get(path), file_stamp(path)) {
            (Some(stored), Some(current)) => *stored == current,
            _ => false,
        }
    }

    fn file_for(&self, path: &Path) -> PathBuf {
        self.dir.join(format!("{}.jpg", hash_path(path)))
    }

    /// Where this file's stored thumbnail is, if it has a valid one. Cheap — a
    /// lookup and a stat — so it is what to do with the cache locked; reading
    /// and decoding the file is not, and is done with [`read`] after the lock
    /// is let go.
    pub fn stored(&self, path: &Path) -> Option<PathBuf> {
        self.has(path).then(|| self.file_for(path))
    }

    /// Load a thumbnail, if one is stored and still valid.
    ///
    /// Returns the decoded RGBA and its size. A miss is not an error: it just
    /// means the image has to be decoded the slow way this once.
    pub fn load(&self, path: &Path) -> Option<(Vec<u8>, u32, u32)> {
        if !self.has(path) {
            return None;
        }
        let bytes = std::fs::read(self.file_for(path)).ok()?;
        let decoded = crate::decode::decode(&bytes).ok()?;
        Some((decoded.rgba, decoded.width, decoded.height))
    }

    /// Store a thumbnail, scaling down to [`THUMB_SIZE`] if needed.
    ///
    /// Encodes here, so the cache has to be held throughout; a worker should
    /// [`encode`] first, unlocked, and only [`Cache::record`] with it held.
    #[allow(dead_code)]
    pub fn store(&mut self, path: &Path, rgba: &[u8], width: u32, height: u32) -> Result<()> {
        let encoded = encode(rgba, width, height)?;
        self.record(path, &encoded)
    }

    /// Install an [`encode`]d thumbnail for `path`: a small file write and an
    /// index entry, which is all that needs the cache to itself.
    pub fn record(&mut self, path: &Path, encoded: &[u8]) -> Result<()> {
        let Some(stamp) = file_stamp(path) else {
            anyhow::bail!("{} vanished before it could be cached", path.display())
        };
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))?;

        // Written beside then renamed, so a cache file is never half-written if
        // the program dies mid-save.
        let target = self.file_for(path);
        let staging = target.with_extension("part");
        std::fs::write(&staging, encoded)
            .with_context(|| format!("writing {}", staging.display()))?;
        std::fs::rename(&staging, &target)
            .with_context(|| format!("installing {}", target.display()))?;

        self.index.insert(path.to_path_buf(), stamp);
        self.dirty = true;
        Ok(())
    }


    pub fn save_index(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir)?;
        let mut out = String::with_capacity(self.index.len() * 64);
        for (path, e) in &self.index {
            out.push_str(&format!("{}\t{}\t{}\n", e.modified, e.len, path.display()));
        }
        std::fs::write(self.dir.join("index"), out).context("writing the thumbnail index")?;
        self.dirty = false;
        Ok(())
    }

    /// Total bytes the stored thumbnails occupy.
    pub fn disk_bytes(&self) -> u64 {
        std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.metadata().ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum()
    }

    /// Delete the oldest thumbnails until the store fits `budget` bytes.
    pub fn trim(&mut self, budget: u64) -> Result<()> {
        if self.disk_bytes() <= budget {
            return Ok(());
        }
        let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "jpg"))
            .filter_map(|e| {
                let m = e.metadata().ok()?;
                Some((m.accessed().or_else(|_| m.modified()).ok()?, m.len(), e.path()))
            })
            .collect();
        files.sort_by_key(|(when, _, _)| *when);

        let mut total: u64 = files.iter().map(|(_, len, _)| len).sum();
        for (_, len, path) in files {
            if total <= budget {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total -= len;
            }
        }
        // The index now names files that are gone; `has` re-checks the file
        // anyway, and the next store rewrites it.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_hashes_are_stable_and_distinct() {
        let a = hash_path(Path::new("/home/u/a.jpg"));
        let b = hash_path(Path::new("/home/u/b.jpg"));
        assert_eq!(a, hash_path(Path::new("/home/u/a.jpg")), "must not vary between calls");
        assert_ne!(a, b);
        assert_eq!(a.len(), 32, "128 bits of hex");

        // Anagrams must not collide, which a single forward pass would allow.
        assert_ne!(hash_path(Path::new("/ab")), hash_path(Path::new("/ba")));
    }

    #[test]
    fn a_missing_file_is_a_miss_not_a_panic() {
        let cache = Cache { dir: PathBuf::from("/nonexistent"), index: HashMap::new(), dirty: false };
        assert!(!cache.has(Path::new("/nonexistent/nope.jpg")));
        assert!(cache.load(Path::new("/nonexistent/nope.jpg")).is_none());
    }

    #[test]
    fn stored_thumbnails_round_trip_and_are_bounded_in_size() {
        let dir = std::env::temp_dir().join(format!("show-thumb-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut cache = Cache { dir: dir.clone(), index: HashMap::new(), dirty: false };

        // A real file on disk, so the mtime validation has something to check.
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.bin");
        std::fs::write(&source, b"placeholder").unwrap();

        // A 1000x600 image is larger than THUMB_SIZE and must be shrunk.
        let rgba: Vec<u8> = (0..1000 * 600 * 4).map(|i| (i % 251) as u8).collect();
        cache.store(&source, &rgba, 1000, 600).unwrap();

        assert!(cache.has(&source));
        let (pixels, w, h) = cache.load(&source).expect("just stored");
        assert!(w <= THUMB_SIZE && h <= THUMB_SIZE, "{w}x{h} exceeds the thumbnail size");
        assert_eq!(pixels.len(), w as usize * h as usize * 4);

        // Editing the file invalidates the thumbnail.
        std::fs::write(&source, b"different content entirely").unwrap();
        assert!(!cache.has(&source), "a changed file must not keep its old thumbnail");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn trimming_respects_the_budget() {
        let dir = std::env::temp_dir().join(format!("show-trim-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut cache = Cache { dir: dir.clone(), index: HashMap::new(), dirty: false };

        for i in 0..10 {
            std::fs::write(dir.join(format!("{i}.jpg")), vec![0u8; 1000]).unwrap();
        }
        assert!(cache.disk_bytes() >= 10_000);

        cache.trim(3_000).unwrap();
        assert!(cache.disk_bytes() <= 3_000, "still {} bytes", cache.disk_bytes());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
