//! Just enough ZIP to take one file out of a Python wheel.
//!
//! ONNX Runtime is published as a wheel, which is a ZIP with a shared library
//! inside it. A whole ZIP crate would be a dependency for one operation on one
//! file, and the format's directory is simple enough to read directly — the
//! same reasoning that produced `src/md5.rs` and `src/sha256.rs`.
//!
//! Deliberately narrow. It reads the central directory, finds one entry by a
//! predicate, and inflates it. No writing, no encryption, no ZIP64, no spanned
//! archives, no directory traversal — the one name it is asked for is matched
//! against entries the archive declares, and nothing is written anywhere the
//! caller did not choose.

use anyhow::{Context, Result, anyhow};
use std::io::{Read, Seek, SeekFrom};

/// One file in the archive.
#[derive(Default)]
pub struct Entry {
    pub name: String,
    /// Offset of the local header, which precedes the data.
    offset: u64,
    compressed: u64,
    /// 0 is stored, 8 is deflate. Nothing else is supported, and a wheel uses
    /// only these two.
    method: u16,
}

impl Entry {
    /// An entry that is only a name.
    ///
    /// For callers whose tests are about *choosing* between entries rather
    /// than reading them — the offsets mean nothing here and are never read.
    pub fn named(name: &str) -> Entry {
        Entry { name: name.to_string(), ..Entry::default() }
    }
}

/// Read the central directory.
pub fn entries(file: &mut std::fs::File) -> Result<Vec<Entry>> {
    let size = file.seek(SeekFrom::End(0))?;

    // The end-of-directory record is last, but a trailing comment can push it
    // back by up to 64 KB, so the tail is searched rather than assumed.
    let tail = size.min(66_000);
    file.seek(SeekFrom::Start(size - tail))?;
    let mut buffer = vec![0u8; tail as usize];
    file.read_exact(&mut buffer)?;

    let eocd = (0..buffer.len().saturating_sub(21))
        .rev()
        .find(|&i| buffer[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .context("not a zip: no end-of-directory record")?;

    let count = u16::from_le_bytes([buffer[eocd + 10], buffer[eocd + 11]]) as usize;
    let start = u32::from_le_bytes([
        buffer[eocd + 16],
        buffer[eocd + 17],
        buffer[eocd + 18],
        buffer[eocd + 19],
    ]) as u64;
    anyhow::ensure!(start != u32::MAX as u64, "zip64 archives are not supported");

    file.seek(SeekFrom::Start(start))?;
    let mut directory = Vec::new();
    file.read_to_end(&mut directory)?;

    let mut entries = Vec::with_capacity(count);
    let mut at = 0usize;
    while at + 46 <= directory.len() && directory[at..at + 4] == [0x50, 0x4b, 0x01, 0x02] {
        let word = |o: usize| u16::from_le_bytes([directory[at + o], directory[at + o + 1]]);
        let long = |o: usize| {
            u32::from_le_bytes([
                directory[at + o],
                directory[at + o + 1],
                directory[at + o + 2],
                directory[at + o + 3],
            ]) as u64
        };
        let (method, compressed) = (word(10), long(20));
        let (name_len, extra_len, comment_len) =
            (word(28) as usize, word(30) as usize, word(32) as usize);
        let offset = long(42);

        let name = String::from_utf8_lossy(&directory[at + 46..at + 46 + name_len]).into_owned();
        entries.push(Entry { name, offset, compressed, method });
        at += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

/// Inflate one entry, writing it to `into`.
///
/// Streamed rather than held in memory: the library inside a wheel is tens of
/// megabytes and there is no reason for all of it to be resident at once.
pub fn extract(file: &mut std::fs::File, entry: &Entry, into: &std::path::Path) -> Result<u64> {
    // The local header repeats the name and extra field, at its own lengths —
    // which are not always the directory's, so they must be read here.
    file.seek(SeekFrom::Start(entry.offset))?;
    let mut header = [0u8; 30];
    file.read_exact(&mut header)?;
    anyhow::ensure!(
        header[..4] == [0x50, 0x4b, 0x03, 0x04],
        "{}: local header is missing",
        entry.name
    );
    let name_len = u16::from_le_bytes([header[26], header[27]]) as u64;
    let extra_len = u16::from_le_bytes([header[28], header[29]]) as u64;
    file.seek(SeekFrom::Start(entry.offset + 30 + name_len + extra_len))?;

    let data = file.take(entry.compressed);
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(into).with_context(|| format!("creating {}", into.display()))?,
    );
    let written = match entry.method {
        0 => std::io::copy(&mut std::io::BufReader::new(data), &mut out)?,
        8 => std::io::copy(
            &mut flate2::read::DeflateDecoder::new(std::io::BufReader::new(data)),
            &mut out,
        )?,
        other => return Err(anyhow!("{}: compression method {other} is not supported", entry.name)),
    };
    Ok(written)
}

/// The first entry whose name satisfies `wanted`.
pub fn find<'a>(entries: &'a [Entry], wanted: impl Fn(&str) -> bool) -> Option<&'a Entry> {
    entries.iter().find(|e| wanted(&e.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny archive with the system zipper, so the reader is tested
    /// against something it did not produce.
    fn sample(dir: &std::path::Path, compress: bool) -> Option<std::path::PathBuf> {
        std::fs::create_dir_all(dir).ok()?;
        let inner = dir.join("hello.txt");
        // Repetitive, so deflate genuinely compresses it.
        std::fs::write(&inner, "hello zip ".repeat(500)).ok()?;
        let archive = dir.join("sample.zip");
        let mut cmd = std::process::Command::new("zip");
        cmd.arg(if compress { "-9" } else { "-0" })
            .arg("-j")
            .arg("-q")
            .arg(&archive)
            .arg(&inner);
        cmd.status().ok()?.success().then_some(archive)
    }

    #[test]
    fn a_deflated_entry_comes_back_whole() {
        let dir = std::env::temp_dir().join(format!("show-zip-d-{}", std::process::id()));
        let Some(archive) = sample(&dir, true) else {
            eprintln!("no `zip` command; skipping");
            return;
        };
        let mut file = std::fs::File::open(&archive).expect("open");
        let entries = entries(&mut file).expect("directory");
        let entry = find(&entries, |n| n.ends_with("hello.txt")).expect("entry");
        assert_eq!(entry.method, 8, "expected the archive to be deflated");

        let out = dir.join("out.txt");
        extract(&mut file, entry, &out).expect("extract");
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "hello zip ".repeat(500));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stored_entry_comes_back_whole() {
        let dir = std::env::temp_dir().join(format!("show-zip-s-{}", std::process::id()));
        let Some(archive) = sample(&dir, false) else {
            eprintln!("no `zip` command; skipping");
            return;
        };
        let mut file = std::fs::File::open(&archive).expect("open");
        let entries = entries(&mut file).expect("directory");
        let entry = find(&entries, |n| n.ends_with("hello.txt")).expect("entry");
        assert_eq!(entry.method, 0, "expected the archive to be stored");

        let out = dir.join("out.txt");
        extract(&mut file, entry, &out).expect("extract");
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "hello zip ".repeat(500));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn something_that_is_not_a_zip_is_rejected_rather_than_misread() {
        let path = std::env::temp_dir().join(format!("show-zip-n-{}", std::process::id()));
        std::fs::write(&path, vec![0u8; 4096]).expect("write");
        let mut file = std::fs::File::open(&path).expect("open");
        assert!(entries(&mut file).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
