//! Tiled and pyramidal TIFF reading.
//!
//! Some images cannot be decoded whole, and no amount of clever scaling changes
//! that: a 114944x98304 scan is 11.3 gigapixels and 42 GB as RGBA, against a
//! GPU texture limit of 32768 pixels on a side. The only way to show it is to
//! read the part being looked at, at a resolution matching the zoom.
//!
//! TIFF supports exactly that. Tiles are independently compressed rectangles,
//! and a pyramidal file stores the same picture several times at halving
//! resolutions, one per IFD. Choosing a level by zoom and fetching a handful of
//! tiles turns an impossible image into an ordinary one.
//!
//! Files without a pyramid still work. The first time a large one is opened,
//! the pass that reads it for its overview also stores halved levels of it
//! (`src/pyramid.rs`), and from then on those are its pyramid.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tiff::decoder::Decoder;
use tiff::tags::Tag;

/// One resolution of a pyramidal image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level {
    /// Which IFD this is, for seeking back to it.
    pub ifd: usize,
    pub width: u32,
    pub height: u32,
    /// Tile dimensions. For a stripped file this is the full width by the
    /// strip height, which the tile machinery handles unchanged.
    pub tile: (u32, u32),
    /// Set for a level read from a stored pyramid rather than the file: its
    /// index there. See `src/pyramid.rs`.
    pub cached: Option<usize>,
}

impl Level {
    /// Tiles across and down.
    pub fn grid(&self) -> (u32, u32) {
        (self.width.div_ceil(self.tile.0), self.height.div_ceil(self.tile.1))
    }

    /// Scale of this level relative to full resolution, at most 1.0.
    pub fn scale(&self, full_width: u32) -> f32 {
        if full_width == 0 { 1.0 } else { self.width as f32 / full_width as f32 }
    }
}

/// A decoded tile, always RGBA8 and always `width * height * 4` bytes.
pub struct Tile {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// The levels of one file, cheap to obtain: only directory headers are read.
#[derive(Clone, Debug)]
pub struct Pyramid {
    pub path: PathBuf,
    /// Largest first, which is also IFD order in every pyramidal file seen.
    pub levels: Vec<Level>,
    /// The levels stored for a file that came without them, if any.
    pub cache: Option<Arc<crate::pyramid::Index>>,
}

impl Pyramid {
    pub fn full_size(&self) -> (u32, u32) {
        self.levels.first().map(|l| (l.width, l.height)).unwrap_or((0, 0))
    }

    pub fn is_pyramidal(&self) -> bool {
        self.levels.len() > 1
    }

    /// The level to draw at a given zoom.
    ///
    /// Picks the smallest level that still has at least one texel per screen
    /// pixel, so the image is never upscaled from a level coarser than the
    /// display needs. Ties go to the sharper level.
    pub fn level_for_zoom(&self, zoom: f32) -> usize {
        let full = self.full_size().0;
        let mut chosen = 0;
        for (i, level) in self.levels.iter().enumerate() {
            if level.scale(full) >= zoom {
                chosen = i;
            }
        }
        chosen
    }
}

/// Read the level structure without decoding any pixels, along with any levels
/// stored for a file that has none of its own.
///
/// Returns `None` for a TIFF we cannot usefully tile, so the caller can fall
/// back to decoding it whole.
pub fn probe(path: &Path) -> Result<Option<Pyramid>> {
    let Some(mut pyramid) = probe_file(path)? else { return Ok(None) };
    if !pyramid.is_pyramidal()
        && let Some(index) = crate::pyramid::lookup(path, pyramid.full_size())
    {
        pyramid.levels.extend(index.as_levels());
        pyramid.cache = Some(Arc::new(index));
    }
    Ok(Some(pyramid))
}

/// As [`probe`], but only the levels in the file itself.
pub fn probe_file(path: &Path) -> Result<Option<Pyramid>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut decoder = match Decoder::new(BufReader::new(file)) {
        Ok(d) => d,
        // Not a TIFF at all.
        Err(_) => return Ok(None),
    };
    // Gigapixel files exceed every default guard; the whole point here is that
    // we never hold more than a few tiles at a time.
    decoder = decoder.with_limits(tiff::decoder::Limits::unlimited());

    let mut levels = Vec::new();
    let mut ifd = 0;
    loop {
        let (width, height) = match decoder.dimensions() {
            Ok(d) => d,
            Err(_) => break,
        };
        let (tw, th) = decoder.chunk_dimensions();
        if tw > 0 && th > 0 {
            levels.push(Level { ifd, width, height, tile: (tw, th), cached: None });
        }

        if !decoder.more_images() {
            break;
        }
        if decoder.next_image().is_err() {
            break;
        }
        ifd += 1;
        // A malformed file must not spin here.
        if ifd > 64 {
            break;
        }
    }

    if levels.is_empty() {
        return Ok(None);
    }
    // Largest first regardless of how the file ordered its directories.
    levels.sort_by(|a, b| b.width.cmp(&a.width));
    Ok(Some(Pyramid { path: path.to_path_buf(), levels, cache: None }))
}

/// Reads tiles from one file.
///
/// Holds its own decoder, so one of these per worker thread gives parallel tile
/// loading without any shared state.
pub struct TileReader {
    decoder: Decoder<BufReader<File>>,
    at_ifd: usize,
    /// Second handle, for reading compressed tile bytes directly.
    raw: File,
    /// Set when the current level stores WebP tiles, which have to be decoded
    /// here rather than by the tiff crate. See [`WebpLevel`].
    webp: Option<WebpLevel>,
    /// The stored levels, and a handle on their file once one is read.
    cache: Option<(Arc<crate::pyramid::Index>, Option<File>)>,
}

/// Tile locations for a WebP-compressed level.
///
/// The `tiff` crate can nominally read these, but it sizes the output buffer
/// from the TIFF's sample count while `image-webp` demands exactly its own
/// `output_buffer_size()`. In an RGBA file whose individual tiles carry no
/// alpha those disagree, and the tile fails to decode -- which is most of the
/// tiles in the aerial survey this was built for. So the compressed bytes are
/// read straight from the file and decoded here, where the buffer can be sized
/// correctly.
struct WebpLevel {
    ifd: usize,
    offsets: Vec<u64>,
    lengths: Vec<u64>,
}

impl TileReader {
    pub fn open(pyramid: &Pyramid) -> Result<TileReader> {
        let path = pyramid.path.as_path();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let decoder = Decoder::new(BufReader::new(file))
            .with_context(|| format!("reading {} as TIFF", path.display()))?
            .with_limits(tiff::decoder::Limits::unlimited());
        let raw = File::open(path).with_context(|| format!("opening {}", path.display()))?;

        let cache = pyramid.cache.clone().map(|index| (index, None));
        let mut reader = TileReader { decoder, at_ifd: 0, raw, webp: None, cache };
        reader.prepare_level(0)?;
        Ok(reader)
    }

    /// Note whether this level needs the hand-rolled WebP path, and if so read
    /// its tile directory once.
    fn prepare_level(&mut self, ifd: usize) -> Result<()> {
        if self.webp.as_ref().is_some_and(|w| w.ifd == ifd) {
            return Ok(());
        }
        self.webp = None;

        let compression = self
            .decoder
            .find_tag(Tag::Compression)
            .ok()
            .flatten()
            .and_then(|v| v.into_u16().ok());
        // 50001 is WebP, as written by GDAL and libtiff.
        if compression != Some(50001) {
            return Ok(());
        }

        let offsets = self.decoder.get_tag_u64_vec(Tag::TileOffsets).context("tile offsets")?;
        let lengths =
            self.decoder.get_tag_u64_vec(Tag::TileByteCounts).context("tile byte counts")?;
        self.webp = Some(WebpLevel { ifd, offsets, lengths });
        Ok(())
    }

    /// Decode one WebP tile from its compressed bytes.
    fn read_webp(&mut self, index: usize) -> Result<(Vec<u8>, u32, u32)> {
        let level = self.webp.as_ref().expect("only called with a webp level");
        let (Some(&offset), Some(&length)) = (level.offsets.get(index), level.lengths.get(index))
        else {
            bail!("tile {index} is outside the tile directory");
        };
        if length == 0 {
            bail!("tile {index} is empty");
        }

        let mut compressed = vec![0u8; length as usize];
        self.raw.seek(SeekFrom::Start(offset)).context("seeking to a tile")?;
        self.raw.read_exact(&mut compressed).context("reading tile bytes")?;

        let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(compressed))
            .map_err(|e| anyhow::anyhow!("webp tile header: {e}"))?;
        let (w, h) = decoder.dimensions();
        let size = decoder
            .output_buffer_size()
            .ok_or_else(|| anyhow::anyhow!("webp tile reports no output size"))?;

        // Exactly what the decoder asks for: the mismatch here is the whole
        // reason this path exists.
        let mut pixels = vec![0u8; size];
        decoder.read_image(&mut pixels).map_err(|e| anyhow::anyhow!("webp tile: {e}"))?;

        let channels = if decoder.has_alpha() { 4 } else { 3 };
        let mut rgba = vec![0u8; w as usize * h as usize * 4];
        for i in 0..(w as usize * h as usize) {
            let (s, d) = (i * channels, i * 4);
            if s + channels > pixels.len() {
                break;
            }
            rgba[d..d + 3].copy_from_slice(&pixels[s..s + 3]);
            rgba[d + 3] = if channels == 4 { pixels[s + 3] } else { 255 };
        }
        Ok((rgba, w, h))
    }

    /// Fetch one tile, converted to RGBA8.
    pub fn read(&mut self, level: &Level, tx: u32, ty: u32) -> Result<Tile> {
        if let Some(stored) = level.cached {
            let Some((index, file)) = self.cache.as_mut() else { bail!("no stored levels to read") };
            if file.is_none() {
                *file = Some(File::open(&index.file).with_context(|| format!("opening {}", index.file.display()))?);
            }
            return index.read(file.as_mut().unwrap(), stored, tx, ty);
        }
        if self.at_ifd != level.ifd {
            self.decoder
                .seek_to_image(level.ifd)
                .with_context(|| format!("seeking to level {}", level.ifd))?;
            self.at_ifd = level.ifd;
            self.prepare_level(level.ifd)?;
        }

        let (across, down) = level.grid();
        if tx >= across || ty >= down {
            bail!("tile {tx},{ty} is outside a {across}x{down} grid");
        }
        let index = ty * across + tx;

        if self.webp.is_some() {
            let (rgba, w, h) = self.read_webp(index as usize)?;
            return Ok(Tile { rgba, width: w, height: h });
        }

        let colour = self.decoder.colortype().context("reading the colour type")?;
        // Edge tiles carry less data than a full tile; this is the valid part.
        let (data_w, data_h) = self.decoder.chunk_data_dimensions(index);
        let chunk = self.decoder.read_chunk(index).context("reading a tile")?;

        let rgba = to_rgba(chunk, colour, data_w, data_h)?;
        Ok(Tile { rgba, width: data_w, height: data_h })
    }
}

/// Convert whatever the file stores into RGBA8.
fn to_rgba(
    chunk: tiff::decoder::DecodingResult,
    colour: tiff::ColorType,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    use tiff::ColorType;
    use tiff::decoder::DecodingResult;

    let pixels = width as usize * height as usize;
    let mut out = vec![0u8; pixels * 4];

    // 16-bit samples are scaled down rather than truncated so highlights do not
    // wrap; the display path is 8-bit either way.
    let bytes: Vec<u8> = match chunk {
        DecodingResult::U8(v) => v,
        DecodingResult::U16(v) => v.into_iter().map(|s| (s >> 8) as u8).collect(),
        other => bail!("unsupported tile sample format {other:?}"),
    };

    let channels = match colour {
        ColorType::Gray(_) => 1,
        ColorType::GrayA(_) => 2,
        ColorType::RGB(_) => 3,
        ColorType::RGBA(_) | ColorType::CMYK(_) => 4,
        other => bail!("unsupported tile colour type {other:?}"),
    };

    for i in 0..pixels {
        let src = i * channels;
        if src + channels > bytes.len() {
            break;
        }
        let d = i * 4;
        match (colour, channels) {
            (ColorType::CMYK(_), _) => {
                // Stored as ink coverage; 255 means full ink, i.e. no light.
                let k = bytes[src + 3] as u32;
                for c in 0..3 {
                    out[d + c] = ((255 - bytes[src + c] as u32) * k / 255) as u8;
                }
                out[d + 3] = 255;
            }
            (_, 1) => {
                out[d] = bytes[src];
                out[d + 1] = bytes[src];
                out[d + 2] = bytes[src];
                out[d + 3] = 255;
            }
            (_, 2) => {
                out[d] = bytes[src];
                out[d + 1] = bytes[src];
                out[d + 2] = bytes[src];
                out[d + 3] = bytes[src + 1];
            }
            (_, 3) => {
                out[d..d + 3].copy_from_slice(&bytes[src..src + 3]);
                out[d + 3] = 255;
            }
            _ => out[d..d + 4].copy_from_slice(&bytes[src..src + 4]),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pyramid(sizes: &[(u32, u32)]) -> Pyramid {
        Pyramid {
            cache: None,
            path: PathBuf::from("test.tif"),
            levels: sizes
                .iter()
                .enumerate()
                .map(|(ifd, (width, height))| Level {
                    ifd,
                    width: *width,
                    height: *height,
                    tile: (256, 256),
                    cached: None,
                })
                .collect(),
        }
    }

    #[test]
    fn the_tile_grid_covers_every_pixel() {
        let level = Level { ifd: 0, width: 1000, height: 700, tile: (256, 256), cached: None };
        assert_eq!(level.grid(), (4, 3), "partial tiles still need a column and a row");

        let exact = Level { ifd: 0, width: 512, height: 256, tile: (256, 256), cached: None };
        assert_eq!(exact.grid(), (2, 1));
    }

    #[test]
    fn zoom_picks_the_smallest_level_that_is_still_sharp_enough() {
        // A real pyramid: full, half, quarter, eighth.
        let p = pyramid(&[(8000, 6000), (4000, 3000), (2000, 1500), (1000, 750)]);

        // Zoomed right out, the smallest level has enough detail.
        assert_eq!(p.level_for_zoom(0.1), 3);
        // At half scale, the half-size level matches exactly.
        assert_eq!(p.level_for_zoom(0.5), 1);
        // At full zoom nothing but the full level will do.
        assert_eq!(p.level_for_zoom(1.0), 0);
        // Beyond full zoom there is nothing sharper to switch to.
        assert_eq!(p.level_for_zoom(4.0), 0);
    }

    #[test]
    fn a_single_level_file_always_uses_its_only_level() {
        let p = pyramid(&[(2000, 1000)]);
        assert!(!p.is_pyramidal());
        for zoom in [0.01, 0.5, 1.0, 8.0] {
            assert_eq!(p.level_for_zoom(zoom), 0, "zoom {zoom}");
        }
    }

    #[test]
    fn levels_are_ordered_largest_first() {
        let p = pyramid(&[(1000, 750), (8000, 6000), (2000, 1500)]);
        let mut sorted = p.levels.clone();
        sorted.sort_by(|a, b| b.width.cmp(&a.width));
        assert_eq!(sorted[0].width, 8000);
        assert!((sorted[0].scale(8000) - 1.0).abs() < 1e-6);
        assert!((sorted[2].scale(8000) - 0.125).abs() < 1e-6);
    }
}

/// Largest overview built when no target is given: a full-resolution request
/// for an image this size can only ever be answered approximately, and a
/// texture much past this is a gigabyte of memory for no visible gain.
const OVERVIEW_LIMIT: u32 = 8192;

/// A whole-image overview of a TIFF too big to decode whole, or that carries
/// a pyramid to read a smaller copy from. `None` means decode it normally.
///
/// Picks the smallest level still covering `target`, then reads its tiles in
/// parallel, one row of tiles per task, shrinking each by the remaining
/// power-of-two factor as it arrives. So a file without a pyramid never holds
/// more than a row of full-resolution tiles at once, however large it is.
pub fn overview(path: &Path, target: Option<crate::decode::Target>) -> Result<Option<crate::decode::Decoded>> {
    use rayon::prelude::*;

    let Some(mut pyramid) = probe(path)? else { return Ok(None) };
    let (fw, fh) = pyramid.full_size();
    let huge = fw > OVERVIEW_LIMIT || fh > OVERVIEW_LIMIT;
    if !pyramid.is_pyramidal() && !huge {
        return Ok(None);
    }
    let mut span = crate::timing::span("decode", "tiff_overview");

    // Too big to show without levels and with none of its own: this is the
    // one pass over every tile, so it stores levels while it is at it. Where
    // that fails, the overview is made from the full-size tiles as before.
    if !pyramid.is_pyramidal() {
        let mut built = crate::timing::span("decode", "tiff_pyramid");
        match crate::pyramid::build(&pyramid) {
            Ok(Some(index)) => {
                built.note(format!("{} levels stored", index.levels.len()));
                pyramid.levels.extend(index.as_levels());
                pyramid.cache = Some(Arc::new(index));
            }
            Ok(None) => {}
            Err(e) => eprintln!("show: storing levels of {}: {e:#}", path.display()),
        }
    }

    let target = target.unwrap_or(crate::decode::Target { width: OVERVIEW_LIMIT, height: OVERVIEW_LIMIT });
    let factor = crate::decode::factor_for_target(fw, fh, target).max(1) as f32;
    // The smallest level at least 1/factor of full size; levels are largest
    // first, so that is the last one that qualifies.
    let chosen = pyramid
        .levels
        .iter()
        .rposition(|l| l.width as f32 * factor >= fw as f32 * 0.999)
        .unwrap_or(0);
    let level = pyramid.levels[chosen];

    // What is left to shrink, as a power of two no larger than a tile, so
    // every tile shrinks by the same whole factor and the seams line up.
    let left = crate::decode::factor_for_target(level.width, level.height, target).max(1);
    let mut f = 1u32;
    while f * 2 <= left && f * 2 <= level.tile.0.min(level.tile.1) {
        f *= 2;
    }
    let (ow, oh) = (level.width.div_ceil(f), level.height.div_ceil(f));
    let (across, down) = level.grid();
    let tw = level.tile.0 / f;

    let strips: Vec<Result<Vec<u8>>> = (0..down)
        .into_par_iter()
        .map(|ty| {
            let mut reader = TileReader::open(&pyramid)?;
            let rows = (level.height - ty * level.tile.1).min(level.tile.1).div_ceil(f);
            let mut strip = vec![0u8; ow as usize * rows as usize * 4];
            for tx in 0..across {
                let tile = reader.read(&level, tx, ty)?;
                let (px, w, h) = crate::scale::box_downscale(&tile.rgba, tile.width, tile.height, f);
                let x0 = (tx * tw) as usize;
                let (src_w, copy) = (w as usize, (w as usize).min(ow as usize - x0));
                for y in 0..h.min(rows) as usize {
                    let (src, dst) = (y * src_w * 4, (y * ow as usize + x0) * 4);
                    strip[dst..dst + copy * 4].copy_from_slice(&px[src..src + copy * 4]);
                }
            }
            Ok(strip)
        })
        .collect();

    let mut rgba = Vec::with_capacity(ow as usize * oh as usize * 4);
    for strip in strips {
        rgba.extend_from_slice(&strip?);
    }
    rgba.resize(ow as usize * oh as usize * 4, 0);

    span.note(format!(
        "{fw}x{fh}: level {}x{} /{f} -> {ow}x{oh}, {} tiles",
        level.width,
        level.height,
        across * down
    ));
    Ok(Some(crate::decode::Decoded {
        width: ow,
        height: oh,
        source_width: fw,
        source_height: fh,
        format: crate::decode::Format::Tiff,
        rgba,
        path: "tiled-tiff",
        tiles: Some(crate::decode::TileLevels {
            levels: pyramid.levels.iter().map(|l| (l.width, l.height)).collect(),
            chosen,
            shrink: f,
            tile: level.tile,
        }),
    }))
}

