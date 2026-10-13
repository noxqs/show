//! Pyramids built for TIFFs that came without one, kept on disk.
//!
//! A large TIFF with no reduced levels can only be shown zoomed out by reading
//! every tile of it: the 1955 mosaic is 6,901 tiles at 10.5 ms, some 72 s of
//! decoding however many cores share it. Zooming in from there is no better,
//! because the nearest thing to a coarser level is the full-size one, and just
//! past the overview that is around 1,300 tiles.
//!
//! So the first time such a file is opened, the pass that reads every tile for
//! the overview also writes halved levels of it here, down to a single tile.
//! From then on the file is opened as if it had been written with a pyramid:
//! the overview comes from a small level, and every zoom has a level close to
//! it. The full-size level is still always read from the file itself.
//!
//! One file per image, named by a hash of its path and checked against the
//! source's size and modification time, so an edited image builds afresh. The
//! layout is the tiles, then an index of where each one is, then a footer
//! giving the index's offset — written in one pass, without knowing the tiles'
//! sizes in advance.
//!
//! Tiles are JPEG, as the thumbnails are and for the same reasons: by far the
//! fastest decoder here, and a fraction of the disk. One with any transparency
//! is QOI instead, which keeps it. Neither level ever stands in for the
//! full-size picture, which always comes from the file.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::tiled::{Level, Pyramid, Tile, TileReader};

/// Width and height of a stored tile.
pub const TILE: u32 = 512;

/// How much disk the stored pyramids may take before the least recently used
/// are dropped. A level-by-level copy of the 1955 mosaic is around a tenth of a
/// gigabyte, so this holds dozens of gigapixel images.
pub const DISK_BUDGET: u64 = 8 << 30;

/// JPEG quality. Higher than the thumbnails': these are looked at full screen.
const QUALITY: u8 = 90;

const MAGIC: &[u8; 8] = b"SHOWPYR1";

/// How much of the full-size image is held at once while levels are made: a
/// band of rows, which for the 1955 mosaic is one row of its tiles.
const BAND_BYTES: u64 = 128 << 20;

/// One stored level: its size, and where each of its tiles is in the file.
#[derive(Clone, Debug, PartialEq)]
pub struct CachedLevel {
    pub width: u32,
    pub height: u32,
    offsets: Vec<u64>,
    lengths: Vec<u32>,
}

impl CachedLevel {
    fn grid(&self) -> (u32, u32) {
        (self.width.div_ceil(TILE), self.height.div_ceil(TILE))
    }
}

/// The levels stored for one image, finest first.
#[derive(Clone, Debug, PartialEq)]
pub struct Index {
    pub file: PathBuf,
    pub levels: Vec<CachedLevel>,
}

impl Index {
    /// The stored levels as levels of `pyramid`, to append to its own.
    pub fn as_levels(&self) -> impl Iterator<Item = Level> + '_ {
        self.levels.iter().enumerate().map(|(i, l)| Level {
            ifd: 0,
            width: l.width,
            height: l.height,
            tile: (TILE, TILE),
            cached: Some(i),
        })
    }

    /// Read and decode one stored tile through `file`, a handle on
    /// [`Index::file`].
    pub fn read(&self, file: &mut File, level: usize, tx: u32, ty: u32) -> Result<Tile> {
        let l = self.levels.get(level).context("no such stored level")?;
        let (across, down) = l.grid();
        if tx >= across || ty >= down {
            bail!("tile {tx},{ty} is outside a {across}x{down} grid");
        }
        let i = (ty * across + tx) as usize;
        let mut bytes = vec![0u8; l.lengths[i] as usize];
        file.seek(SeekFrom::Start(l.offsets[i])).context("seeking to a stored tile")?;
        file.read_exact(&mut bytes).context("reading a stored tile")?;
        let decoded = crate::decode::decode(&bytes).context("decoding a stored tile")?;
        Ok(Tile { rgba: decoded.rgba, width: decoded.width, height: decoded.height })
    }
}

/// Where the pyramids are kept.
pub fn dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("show").join("pyramids"))
}

fn file_for(dir: &Path, source: &Path) -> PathBuf {
    dir.join(format!("{}.pyr", crate::thumbs::hash_path(source)))
}

/// Size and modification time, in nanoseconds: what says the stored copy is
/// still of this file.
fn stamp(source: &Path) -> Result<(u64, u64)> {
    let meta = std::fs::metadata(source).with_context(|| format!("reading {}", source.display()))?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    Ok((meta.len(), modified))
}

/// The pyramid stored for `source`, if there is one and it is still current.
pub fn lookup(source: &Path, full: (u32, u32)) -> Option<Index> {
    lookup_in(&dir()?, source, full)
}

fn lookup_in(dir: &Path, source: &Path, full: (u32, u32)) -> Option<Index> {
    let file = file_for(dir, source);
    let index = read_index(&file, stamp(source).ok()?, full).ok()?;
    // Marked as used, for trimming, which drops the least recently used.
    if let Ok(f) = File::options().write(true).open(&file) {
        let _ = f.set_modified(std::time::SystemTime::now());
    }
    Some(index)
}

fn read_index(file: &Path, stamp: (u64, u64), full: (u32, u32)) -> Result<Index> {
    let mut f = File::open(file)?;
    let end = f.seek(SeekFrom::End(0))?;
    if end < 16 {
        bail!("too short");
    }
    let mut footer = [0u8; 16];
    f.seek(SeekFrom::End(-16))?;
    f.read_exact(&mut footer)?;
    if &footer[8..] != MAGIC {
        bail!("not a stored pyramid");
    }
    let at = u64::from_le_bytes(footer[..8].try_into().unwrap());
    if at > end - 16 {
        bail!("index out of range");
    }
    let mut index = vec![0u8; (end - 16 - at) as usize];
    f.seek(SeekFrom::Start(at))?;
    f.read_exact(&mut index)?;

    let mut r = Fields { bytes: &index };
    if (r.u64()?, r.u64()?) != stamp {
        bail!("the source has changed");
    }
    if (r.u32()?, r.u32()?) != full {
        bail!("stored for another size");
    }
    let count = r.u32()?;
    let mut levels = Vec::new();
    for _ in 0..count {
        let (width, height) = (r.u32()?, r.u32()?);
        let tiles = r.u32()? as usize;
        let mut level = CachedLevel { width, height, offsets: Vec::with_capacity(tiles), lengths: Vec::with_capacity(tiles) };
        let (across, down) = level.grid();
        if tiles != (across * down) as usize {
            bail!("tile count does not match the level");
        }
        for _ in 0..tiles {
            let (offset, length) = (r.u64()?, r.u32()?);
            if offset + length as u64 > at {
                bail!("tile out of range");
            }
            level.offsets.push(offset);
            level.lengths.push(length);
        }
        levels.push(level);
    }
    Ok(Index { file: file.to_path_buf(), levels })
}

/// Little-endian fields read off the front of a slice.
struct Fields<'a> {
    bytes: &'a [u8],
}

impl Fields<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.bytes.len() < N {
            bail!("index cut short");
        }
        let (head, rest) = self.bytes.split_at(N);
        self.bytes = rest;
        Ok(head.try_into().unwrap())
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }
}

/// Read every tile of `pyramid`'s full-size level, writing halved levels of it
/// to the store as they go by. `None` when there is nowhere to store them.
pub fn build(pyramid: &Pyramid) -> Result<Option<Index>> {
    let Some(dir) = dir() else { return Ok(None) };
    let index = build_in(&dir, pyramid)?;
    if let Err(e) = trim(&dir, DISK_BUDGET, &index.file) {
        eprintln!("show: trimming {}: {e:#}", dir.display());
    }
    Ok(Some(index))
}

fn build_in(dir: &Path, pyramid: &Pyramid) -> Result<Index> {
    use rayon::prelude::*;

    let source = pyramid.levels.first().copied().context("no levels")?;
    let stamp = stamp(&pyramid.path)?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let file = file_for(dir, &pyramid.path);
    let temp = file.with_extension(format!("pyr.{}", std::process::id()));

    let mut out = Out { w: BufWriter::new(File::create(&temp).with_context(|| format!("creating {}", temp.display()))?), at: 0 };
    let result = (|| -> Result<Vec<Builder>> {
        // Halving down to a level that fits in one tile.
        let mut builders = Vec::new();
        let (mut w, mut h) = (source.width, source.height);
        while w > TILE || h > TILE {
            builders.push(Builder::new(w, h));
            (w, h) = (w.div_ceil(2), h.div_ceil(2));
        }

        let (across, down) = source.grid();
        let (sw, sh) = (source.width as usize, source.height as usize);
        // Enough rows of tiles at once to keep every core busy (one for a
        // tiled file, dozens for one in strips, which is one tile across),
        // but not so many that the band of full-size rows gets large.
        let band = source.tile.1 as u64 * sw as u64 * 4;
        let batch = 64u32.div_ceil(across).min((BAND_BYTES / band.max(1)).max(1) as u32).clamp(1, 64);
        let (mut slots, mut band, mut halves) = (Vec::new(), Vec::new(), Vec::new());
        for first in (0..down).step_by(batch as usize) {
            let rows: Vec<u32> = (first..(first + batch).min(down)).collect();
            // A run of tiles per task, each with one reader: opening a reader
            // parses the file's directory, which per tile costs more than the
            // tile. Each tile is copied into a slot kept from band to band and
            // dropped at once, so the allocator hands the same memory to the
            // next; holding a band's worth of tiles instead, and freeing them
            // together, had them faulted in afresh every band.
            let keys: Vec<(u32, u32)> = rows.iter().flat_map(|&ty| (0..across).map(move |tx| (tx, ty))).collect();
            let run = keys.len().div_ceil(rayon::current_num_threads() * 2).max(1);
            let slot = (source.tile.0 * source.tile.1 * 4) as usize;
            slots.resize(keys.len() * slot, 0);
            let read: Vec<Result<Vec<(u32, u32)>>> = slots
                .par_chunks_mut(run * slot)
                .zip(keys.par_chunks(run))
                .map(|(slots, keys)| {
                    let mut reader = TileReader::open(pyramid)?;
                    keys.iter()
                        .zip(slots.chunks_mut(slot))
                        .map(|(&(tx, ty), slot)| {
                            let tile = reader.read(&source, tx, ty)?;
                            let (w, h) = (tile.width.min(source.tile.0), tile.height.min(source.tile.1));
                            for y in 0..h as usize {
                                let (s, d) = (y * tile.width as usize * 4, y * source.tile.0 as usize * 4);
                                slot[d..d + w as usize * 4].copy_from_slice(&tile.rgba[s..s + w as usize * 4]);
                            }
                            Ok((w, h))
                        })
                        .collect()
                })
                .collect();
            let sizes: Vec<(u32, u32)> = read.into_iter().collect::<Result<Vec<_>>>()?.into_iter().flatten().collect();

            // The band's full-size rows, put together from the slots.
            let y0 = (first * source.tile.1) as usize;
            let y1 = (((first + batch) * source.tile.1) as usize).min(sh);
            band.resize((y1 - y0) * sw * 4, 0);
            band.par_chunks_mut(sw * 4).enumerate().for_each(|(i, line)| {
                let n = i / source.tile.1 as usize;
                let ty_y = i % source.tile.1 as usize;
                line.fill(0);
                for tx in 0..across as usize {
                    let k = n * across as usize + tx;
                    let (w, h) = sizes[k];
                    let x0 = tx * source.tile.0 as usize;
                    let cols = (w as usize).min(sw - x0);
                    if ty_y < h as usize {
                        let s = k * slot + ty_y * source.tile.0 as usize * 4;
                        line[x0 * 4..(x0 + cols) * 4].copy_from_slice(&slots[s..s + cols * 4]);
                    }
                }
            });

            // Halved in pairs across the cores, with a row left over from the
            // last band going first and an odd one out kept for the next.
            let carried = builders[0].pending.take();
            let lines = band.len() / (sw * 4) + carried.is_some() as usize;
            let line = |k: usize| -> &[u8] {
                match &carried {
                    Some(c) if k == 0 => c,
                    Some(_) => &band[(k - 1) * sw * 4..k * sw * 4],
                    None => &band[k * sw * 4..(k + 1) * sw * 4],
                }
            };
            let hw = source.width.div_ceil(2) as usize * 4;
            halves.resize(lines / 2 * hw, 0);
            halves.par_chunks_mut(hw).enumerate().for_each(|(p, half)| halve_into(line(2 * p), line(2 * p + 1), source.width, half));
            if lines % 2 == 1 {
                builders[0].pending = Some(line(lines - 1).to_vec());
            }
            for half in halves.chunks(hw) {
                emit(&mut builders, &mut out, 0, half)?;
            }
        }
        for i in 0..builders.len() {
            if let Some(last) = builders[i].pending.take() {
                // An odd row out at the bottom: halved against itself.
                let half = halve(&last, &last, builders[i].src_width);
                emit(&mut builders, &mut out, i, &half)?;
            }
            if builders[i].filled > 0 {
                builders[i].flush(&mut out)?;
            }
        }
        Ok(builders)
    })();

    let builders = match result {
        Ok(b) => b,
        Err(e) => {
            drop(out);
            let _ = std::fs::remove_file(&temp);
            return Err(e);
        }
    };

    // The index, then the footer pointing at it.
    let at = out.at;
    let mut index = Vec::new();
    index.extend_from_slice(&stamp.0.to_le_bytes());
    index.extend_from_slice(&stamp.1.to_le_bytes());
    index.extend_from_slice(&source.width.to_le_bytes());
    index.extend_from_slice(&source.height.to_le_bytes());
    index.extend_from_slice(&(builders.len() as u32).to_le_bytes());
    for b in &builders {
        index.extend_from_slice(&b.width.to_le_bytes());
        index.extend_from_slice(&b.height.to_le_bytes());
        index.extend_from_slice(&(b.offsets.len() as u32).to_le_bytes());
        for (o, l) in b.offsets.iter().zip(&b.lengths) {
            index.extend_from_slice(&o.to_le_bytes());
            index.extend_from_slice(&l.to_le_bytes());
        }
    }
    index.extend_from_slice(&at.to_le_bytes());
    index.extend_from_slice(MAGIC);
    out.w.write_all(&index)?;
    out.w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&temp, &file).with_context(|| format!("storing {}", file.display()))?;

    Ok(Index {
        file,
        levels: builders
            .into_iter()
            .map(|b| CachedLevel { width: b.width, height: b.height, offsets: b.offsets, lengths: b.lengths })
            .collect(),
    })
}

/// The file being written, and how far into it.
struct Out {
    w: BufWriter<File>,
    at: u64,
}

/// One level being made from the rows of the level above it.
struct Builder {
    /// The size of this level, half the one above.
    width: u32,
    height: u32,
    /// The width of the rows it is fed.
    src_width: u32,
    /// A row waiting for the one below it, to be halved with.
    pending: Option<Vec<u8>>,
    /// A row of tiles, filling up.
    rows: Vec<u8>,
    filled: u32,
    offsets: Vec<u64>,
    lengths: Vec<u32>,
}

impl Builder {
    fn new(src_width: u32, src_height: u32) -> Builder {
        let (width, height) = (src_width.div_ceil(2), src_height.div_ceil(2));
        Builder {
            width,
            height,
            src_width,
            pending: None,
            rows: Vec::with_capacity(width as usize * TILE as usize * 4),
            filled: 0,
            offsets: Vec::new(),
            lengths: Vec::new(),
        }
    }

    /// Cut the rows held into tiles, and write them.
    fn flush(&mut self, out: &mut Out) -> Result<()> {
        use rayon::prelude::*;
        let (w, h) = (self.width as usize, self.filled as usize);
        let rows = &self.rows;
        let encoded: Vec<Result<Vec<u8>>> = (0..self.width.div_ceil(TILE) as usize)
            .into_par_iter()
            .map(|tx| {
                let x0 = tx * TILE as usize;
                let tw = (TILE as usize).min(w - x0);
                encode(&rows[x0 * 4..h * w * 4], tw as u32, h as u32, w * 4)
            })
            .collect();
        for bytes in encoded {
            let bytes = bytes?;
            out.w.write_all(&bytes)?;
            self.offsets.push(out.at);
            self.lengths.push(bytes.len() as u32);
            out.at += bytes.len() as u64;
        }
        self.rows.clear();
        self.filled = 0;
        Ok(())
    }
}

/// Give builder `i` a row of the level above it.
fn feed(builders: &mut [Builder], out: &mut Out, i: usize, line: &[u8]) -> Result<()> {
    let Some(b) = builders.get_mut(i) else { return Ok(()) };
    match b.pending.take() {
        None => b.pending = Some(line.to_vec()),
        Some(above) => {
            let half = halve(&above, line, b.src_width);
            emit(builders, out, i, &half)?;
        }
    }
    Ok(())
}

/// A finished row of builder `i`'s level: kept for its tiles, and passed on
/// to the level below.
fn emit(builders: &mut [Builder], out: &mut Out, i: usize, half: &[u8]) -> Result<()> {
    let b = &mut builders[i];
    b.rows.extend_from_slice(half);
    b.filled += 1;
    if b.filled == TILE {
        b.flush(out)?;
    }
    feed(builders, out, i + 1, half)
}

/// Two rows `width` pixels wide averaged into one half as wide, each output
/// pixel the mean of a 2x2 block. An odd last column is paired with itself.
fn halve(a: &[u8], b: &[u8], width: u32) -> Vec<u8> {
    let mut out = vec![0u8; width.div_ceil(2) as usize * 4];
    halve_into(a, b, width, &mut out);
    out
}

/// As [`halve`], into `out`.
fn halve_into(a: &[u8], b: &[u8], width: u32, out: &mut [u8]) {
    let (w, ow) = (width as usize, width.div_ceil(2) as usize);
    for x in 0..ow {
        let (l, r) = (2 * x, (2 * x + 1).min(w - 1));
        for c in 0..4 {
            let sum = a[l * 4 + c] as u32 + a[r * 4 + c] as u32 + b[l * 4 + c] as u32 + b[r * 4 + c] as u32;
            out[x * 4 + c] = ((sum + 2) / 4) as u8;
        }
    }
}

/// A tile as stored: JPEG when it is opaque, QOI when it is not. `pitch` is
/// the bytes from one row to the next in `rgba`.
fn encode(rgba: &[u8], width: u32, height: u32, pitch: usize) -> Result<Vec<u8>> {
    use image::ImageEncoder;
    let row = |y: usize| &rgba[y * pitch..y * pitch + width as usize * 4];
    if (0..height as usize).all(|y| row(y).chunks_exact(4).all(|p| p[3] == 255)) {
        return crate::turbojpeg::encode(rgba, width, height, pitch, QUALITY);
    }
    let packed: Vec<u8> = (0..height as usize).flat_map(row).copied().collect();
    let mut bytes = Vec::new();
    image::codecs::qoi::QoiEncoder::new(&mut bytes)
        .write_image(&packed, width, height, image::ExtendedColorType::Rgba8)
        .context("encoding a stored tile")?;
    Ok(bytes)
}

/// Drop the least recently used pyramids until the store is within `budget`,
/// never `keep`, the one just made.
fn trim(dir: &Path, budget: u64, keep: &Path) -> Result<()> {
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "pyr"))
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            Some((meta.modified().ok()?, meta.len(), e.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort();
    for (_, len, path) in files {
        if total <= budget {
            break;
        }
        if path != keep && std::fs::remove_file(&path).is_ok() {
            total -= len;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("show-pyramid-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A smooth gradient, so JPEG stays close to it and a wrong tile shows.
    fn source(dir: &Path, w: u32, h: u32) -> (PathBuf, Pyramid) {
        let path = dir.join("big.tif");
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x / 8) as u8, (y / 8) as u8, 128]));
        img.save(&path).unwrap();
        let pyramid = crate::tiled::probe_file(&path).unwrap().unwrap();
        (path, pyramid)
    }

    #[test]
    fn halving_averages_blocks_and_keeps_an_odd_column() {
        let a = [0, 0, 0, 255, 100, 100, 100, 255, 50, 50, 50, 255];
        let b = [0, 0, 0, 255, 100, 100, 100, 255, 50, 50, 50, 255];
        assert_eq!(halve(&a, &b, 3), vec![50, 50, 50, 255, 50, 50, 50, 255]);
    }

    #[test]
    fn a_built_pyramid_halves_to_one_tile_and_matches_the_source() {
        let dir = scratch("build");
        let (path, pyramid) = source(&dir, 1500, 1100);
        let index = build_in(&dir.join("store"), &pyramid).unwrap();

        let sizes: Vec<(u32, u32)> = index.levels.iter().map(|l| (l.width, l.height)).collect();
        assert_eq!(sizes, [(750, 550), (375, 275)], "halved until it fits one tile");

        // Found again, and read back close to the source at that scale.
        let found = lookup_in(&dir.join("store"), &path, (1500, 1100)).unwrap();
        assert_eq!(found.levels, index.levels);
        let mut file = File::open(&found.file).unwrap();
        let tile = found.read(&mut file, 0, 1, 1).unwrap();
        assert_eq!((tile.width, tile.height), (750 - 512, 550 - 512), "the corner tile is short");
        let px = |x: usize, y: usize| &tile.rgba[(y * tile.width as usize + x) * 4..][..3];
        // (512 + 100, 512 + 20) at half size is (1224, 1064) in the source.
        let [r, g, b] = [px(100, 20)[0] as i32, px(100, 20)[1] as i32, px(100, 20)[2] as i32];
        assert!((r - 1224 / 8).abs() <= 3 && (g - 1064 / 8).abs() <= 3 && (b - 128).abs() <= 3, "{r},{g},{b}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_changed_source_is_not_matched() {
        let dir = scratch("stale");
        let (path, pyramid) = source(&dir, 700, 600);
        build_in(&dir.join("store"), &pyramid).unwrap();
        assert!(lookup_in(&dir.join("store"), &path, (700, 600)).is_some());
        assert!(lookup_in(&dir.join("store"), &path, (701, 600)).is_none(), "another size");
        std::fs::write(&path, b"no longer that image").unwrap();
        assert!(lookup_in(&dir.join("store"), &path, (700, 600)).is_none(), "rewritten");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_stored_levels_are_read_as_levels_of_the_file() {
        let dir = scratch("probe");
        let (path, pyramid) = source(&dir, 1100, 700);
        let index = build_in(&dir.join("store"), &pyramid).unwrap();
        let mut with = pyramid.clone();
        with.levels.extend(index.as_levels());
        with.cache = Some(std::sync::Arc::new(index));
        assert_eq!(with.levels.len(), 3, "the file's own level and two stored");
        assert_eq!(with.level_for_zoom(0.5), 1, "half zoom reads the stored half");

        let mut reader = TileReader::open(&with).unwrap();
        let level = with.levels[1];
        let tile = reader.read(&level, 1, 0).unwrap();
        assert_eq!((tile.width, tile.height), (550 - 512, 350));
        let _ = path;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn trimming_drops_the_oldest_but_never_the_newest() {
        let dir = scratch("trim");
        for (i, name) in ["a.pyr", "b.pyr", "c.pyr"].iter().enumerate() {
            let p = dir.join(name);
            std::fs::write(&p, vec![0u8; 100]).unwrap();
            let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1000 + i as u64);
            File::options().write(true).open(&p).unwrap().set_modified(t).unwrap();
        }
        trim(&dir, 200, &dir.join("a.pyr")).unwrap();
        assert!(dir.join("a.pyr").exists(), "kept, though oldest");
        assert!(!dir.join("b.pyr").exists());
        assert!(dir.join("c.pyr").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
