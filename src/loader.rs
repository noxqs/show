//! Getting file bytes in front of a decoder, with the stages kept separate so
//! the profile can tell raw I/O apart from decode work.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::decode::{self, Decoded, Target};
use crate::timing;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Io {
    /// `read(2)` the whole file into a heap buffer. Costs one copy, but the
    /// copy is a single sequential pass and shows up honestly in the profile.
    Read,
    /// `mmap(2)` the file. No copy, but the pages fault in lazily during
    /// decode, so the I/O cost hides inside the decode stage.
    Mmap,
}

pub struct Loaded {
    pub path: PathBuf,
    pub file_size: u64,
    pub decoded: Decoded,
}

/// Backing store for the encoded bytes. Held only for the duration of the
/// decode; the mmap must outlive every borrow of its slice.
enum Bytes {
    Heap(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Heap(v) => v,
            Bytes::Mapped(m) => m,
        }
    }
}

pub fn load(path: &Path, io: Io, target: Option<Target>) -> Result<Loaded> {
    let file = {
        let _s = timing::span("io", "open");
        File::open(path).with_context(|| format!("opening {}", path.display()))?
    };
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);

    let bytes = match io {
        Io::Read => {
            let mut s = timing::span("io", "read");
            s.note(format!("{} read(2)", human_bytes(file_size)));
            Bytes::Heap(std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)
        }
        Io::Mmap => {
            let mut s = timing::span("io", "mmap");
            s.note(format!("{} mmap(2), pages fault during decode", human_bytes(file_size)));
            // Safety: we accept the standard mmap caveat — if another process
            // truncates the file while we hold the mapping, touching the
            // vanished pages raises SIGBUS. That is the same bargain every
            // mmap-based reader makes.
            Bytes::Mapped(unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?)
        }
    };

    // A TIFF too big to decode whole, or one with a pyramid to read a smaller
    // copy from, is assembled from tiles instead.
    if matches!(decode::sniff_path(&bytes, path), Some(decode::Format::Tiff))
        && let Some(decoded) = crate::tiled::overview(path, target)
            .with_context(|| format!("reading tiles of {}", path.display()))?
    {
        return Ok(Loaded { path: path.to_path_buf(), file_size, decoded });
    }

    let decoded =
        decode::decode_path(&bytes, path, target).with_context(|| format!("decoding {}", path.display()))?;

    Ok(Loaded { path: path.to_path_buf(), file_size, decoded })
}

/// A HEIC's embedded thumbnail, to put on screen while [`load`] does the real
/// decode — or `None` for anything else, or where the thumbnail is not worth
/// showing. Never an error: this is a head start, and the real load reports
/// anything actually wrong with the file.
pub fn load_thumbnail(path: &Path, target: Option<Target>) -> Option<Loaded> {
    let target = target?;
    let file = File::open(path).ok()?;
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
    // Safety: as in `load`.
    let bytes = unsafe { memmap2::Mmap::map(&file) }.ok()?;
    let format = decode::sniff_path(&bytes, path)?;
    if format != decode::Format::Heif {
        return None;
    }
    let decoded = crate::heif::thumbnail(&bytes, format, target).ok()??;
    Some(Loaded { path: path.to_path_buf(), file_size, decoded })
}

pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 { format!("{n} B") } else { format!("{v:.1} {}", UNITS[i]) }
}
