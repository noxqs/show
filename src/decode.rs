//! Format detection and decoding to RGBA8.
//!
//! Every decoder here funnels into one layout — tightly packed RGBA8, top-left
//! origin — because that is what both the GPU texture upload and the software
//! blitter want, and because any per-pixel fixup we skip here is a full pass
//! over the image we don't pay for.
//!
//! Formats take a fast path through a specialised decoder where one exists and
//! fall back to the `image` crate otherwise. The fallback also catches fast
//! path failures, so an exotic-but-valid file still opens.
//!
//! Decoding is *targeted*: a caller that only needs enough pixels to fill a
//! screen says so, and formats that can decode at reduced resolution do. For
//! JPEG that means scaled inverse DCT, which is close to free -- a 416 MPix
//! panorama shrinks from a 2.6 second full decode to a fraction of that,
//! because the work skipped is the expensive part rather than a resize bolted
//! on afterwards. Anything that cannot decode small is decoded whole and box
//! filtered down, which still saves the GPU upload.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::timing;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Format {
    Jpeg,
    Png,
    Gif,
    WebP,
    Bmp,
    Tiff,
    Ico,
    Tga,
    Qoi,
    Dds,
    Hdr,
    OpenExr,
    Pnm,
    Farbfeld,
    Avif,
    Heif,
    Jxl,
    Svg,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Format::Jpeg => "jpeg",
            Format::Png => "png",
            Format::Gif => "gif",
            Format::WebP => "webp",
            Format::Bmp => "bmp",
            Format::Tiff => "tiff",
            Format::Ico => "ico",
            Format::Tga => "tga",
            Format::Qoi => "qoi",
            Format::Dds => "dds",
            Format::Hdr => "hdr",
            Format::OpenExr => "exr",
            Format::Pnm => "pnm",
            Format::Farbfeld => "farbfeld",
            Format::Avif => "avif",
            Format::Heif => "heif",
            Format::Jxl => "jxl",
            Format::Svg => "svg",
        }
    }

    /// Whether this build can decode the format, and if not, why not.
    ///
    /// Optional codecs are the price of a self-contained binary, so a file we
    /// cannot open should say what would change that rather than just failing.
    pub fn support(self) -> Support {
        match self {
            Format::Avif => {
                if cfg!(feature = "avif-system-dav1d") {
                    Support::Yes("libdav1d")
                } else {
                    // Otherwise libheif's AV1 plugin, found at runtime like HEIC.
                    match crate::heif::avif_availability() {
                        a if a.is_ready() => Support::Runtime(a.summary()),
                        a => Support::MissingRuntime(a.summary()),
                    }
                }
            }
            // Probed at runtime rather than decided at build time: libheif is
            // loaded on demand, so support depends on the machine, not the binary.
            Format::Heif => match crate::heif::availability() {
                a if a.is_ready() => Support::Runtime(a.summary()),
                a => Support::MissingRuntime(a.summary()),
            },
            Format::Svg => Support::Missing("SVG rendering is not built in"),
            _ => Support::Yes("built in"),
        }
    }

    /// The `image` crate equivalent, where it has one.
    fn to_image_format(self) -> Option<image::ImageFormat> {
        use image::ImageFormat as F;
        Some(match self {
            Format::Jpeg => F::Jpeg,
            Format::Png => F::Png,
            Format::Gif => F::Gif,
            Format::WebP => F::WebP,
            Format::Bmp => F::Bmp,
            Format::Tiff => F::Tiff,
            Format::Ico => F::Ico,
            Format::Tga => F::Tga,
            Format::Qoi => F::Qoi,
            Format::Dds => F::Dds,
            Format::Hdr => F::Hdr,
            Format::OpenExr => F::OpenExr,
            Format::Pnm => F::Pnm,
            Format::Farbfeld => F::Farbfeld,
            Format::Avif => F::Avif,
            Format::Heif | Format::Jxl | Format::Svg => return None,
        })
    }
}

/// Identify a format, preferring the bytes and falling back to the extension.
///
/// TGA is the reason the fallback exists: it has no signature at the front at
/// all, only an optional footer that plenty of encoders omit. Everything else
/// is decided by magic number, because a viewer that is the system default gets
/// handed plenty of `.jpg` files that are really PNGs.
pub fn sniff_path(bytes: &[u8], path: &std::path::Path) -> Option<Format> {
    if let Some(format) = sniff(bytes) {
        return Some(format);
    }
    // TGA's footer is the only in-band evidence the format leaves.
    if bytes.len() >= 18 && &bytes[bytes.len() - 18..bytes.len() - 2] == b"TRUEVISION-XFILE" {
        return Some(Format::Tga);
    }
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "tga" | "icb" | "vda" | "vst" => Some(Format::Tga),
        "pnm" | "pbm" | "pgm" | "ppm" | "pam" => Some(Format::Pnm),
        _ => None,
    }
}

/// Identify a format from its leading bytes alone.
pub fn sniff(b: &[u8]) -> Option<Format> {
    if b.len() < 12 {
        return None;
    }
    if b.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(Format::Jpeg);
    }
    if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some(Format::Png);
    }
    if b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a") {
        return Some(Format::Gif);
    }
    if b.starts_with(b"RIFF") && &b[8..12] == b"WEBP" {
        return Some(Format::WebP);
    }
    if b.starts_with(b"BM") {
        return Some(Format::Bmp);
    }
    // Classic TIFF, then BigTIFF (version 43), which is what a file past 4 GB
    // — or a mosaic tool expecting one — writes.
    if b.starts_with(b"II*\0") || b.starts_with(b"MM\0*") || b.starts_with(b"II+\0") || b.starts_with(b"MM\0+") {
        return Some(Format::Tiff);
    }
    if b.starts_with(&[0x00, 0x00, 0x01, 0x00]) {
        return Some(Format::Ico);
    }
    if b.starts_with(b"qoif") {
        return Some(Format::Qoi);
    }
    if b.starts_with(b"DDS ") {
        return Some(Format::Dds);
    }
    if b.starts_with(b"#?RADIANCE") || b.starts_with(b"#?RGBE") {
        return Some(Format::Hdr);
    }
    if b.starts_with(&[0x76, 0x2F, 0x31, 0x01]) {
        return Some(Format::OpenExr);
    }
    if b.starts_with(b"farbfeld") {
        return Some(Format::Farbfeld);
    }
    if b[0] == b'P' && (b'1'..=b'7').contains(&b[1]) {
        return Some(Format::Pnm);
    }
    // JPEG XL: raw codestream, or the ISOBMFF container.
    if b.starts_with(&[0xFF, 0x0A]) || b.starts_with(&[0, 0, 0, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A]) {
        return Some(Format::Jxl);
    }
    // ISOBMFF: the brand at offset 8 separates AVIF from the HEIF family.
    if &b[4..8] == b"ftyp" {
        return match &b[8..12] {
            b"avif" | b"avis" => Some(Format::Avif),
            b"heic" | b"heix" | b"hevc" | b"heim" | b"heis" | b"mif1" | b"msf1" => Some(Format::Heif),
            _ => None,
        };
    }
    let head = &b[..b.len().min(1024)];
    if head.starts_with(b"<?xml") || head.starts_with(b"<svg") || head.windows(4).any(|w| w == b"<svg") {
        return Some(Format::Svg);
    }
    None
}

/// The most pixels the caller can actually use. Decoders are free to return
/// fewer than the image holds, never more than it holds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub width: u32,
    pub height: u32,
}

impl Target {
    /// How many times smaller the image may safely get, as a real number.
    ///
    /// An image fitted into the target box is displayed at `min(tw/w, th/h)`,
    /// so the useful texel count is set by whichever axis binds first. The
    /// reciprocal of that is how much resolution we are allowed to discard.
    /// Below 1.0 means the image is already smaller than the box.
    pub fn reduction(&self, w: u32, h: u32) -> f32 {
        let rx = w as f32 / self.width.max(1) as f32;
        let ry = h as f32 / self.height.max(1) as f32;
        rx.max(ry)
    }
}

/// Largest whole factor by which `w`x`h` may shrink and still fill `target`.
///
/// Rounds *down*, and that direction matters: rounding up would satisfy the
/// target by overshooting it, handing the screen fewer texels than it has
/// pixels and making a fitted image visibly soft. Keeping slightly more
/// resolution than needed costs only memory.
pub fn factor_for_target(w: u32, h: u32, target: Target) -> u32 {
    let r = target.reduction(w, h);
    if !r.is_finite() || r < 1.0 { 1 } else { r as u32 }
}

/// Whether a recognised format can actually be decoded by this build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Support {
    Yes(&'static str),
    /// Carries what the user would have to do about it.
    Missing(&'static str),
    /// Available, via a library found on this machine at runtime.
    Runtime(String),
    /// Could work on this machine, but something is missing right now.
    MissingRuntime(String),
}

/// Every format the sniffer recognises, with this build's support for each.
pub const ALL_FORMATS: [Format; 18] = [
    Format::Jpeg,
    Format::Png,
    Format::Gif,
    Format::WebP,
    Format::Bmp,
    Format::Tiff,
    Format::Ico,
    Format::Tga,
    Format::Qoi,
    Format::Dds,
    Format::Hdr,
    Format::OpenExr,
    Format::Pnm,
    Format::Farbfeld,
    Format::Avif,
    Format::Heif,
    Format::Jxl,
    Format::Svg,
];

pub struct Decoded {
    /// Dimensions of the pixels we are actually holding.
    pub width: u32,
    pub height: u32,
    /// Dimensions of the image on disk. Equal to `width`/`height` unless the
    /// decode was scaled down.
    pub source_width: u32,
    pub source_height: u32,
    pub format: Format,
    /// Tightly packed RGBA8, `width * height * 4` bytes.
    pub rgba: Vec<u8>,
    /// Which decoder actually produced this, for the timing report.
    pub path: &'static str,
    /// For an image read from tiles: the pyramid, and which level of it.
    pub tiles: Option<TileLevels>,
}

/// Where a tiled image's pixels came from, for the debug readout.
#[derive(Clone, Debug)]
pub struct TileLevels {
    /// Every level's size, largest first.
    pub levels: Vec<(u32, u32)>,
    /// The level that was read.
    pub chosen: usize,
    /// How much its tiles were shrunk on top of that, as they were read.
    pub shrink: u32,
    pub tile: (u32, u32),
}

impl Decoded {
    /// True when these pixels are a reduced stand-in for a larger image.
    pub fn is_reduced(&self) -> bool {
        self.width < self.source_width || self.height < self.source_height
    }

    /// Largest zoom at which these pixels still have a texel per screen pixel.
    pub fn adequate_zoom(&self) -> f32 {
        if self.source_width == 0 {
            return 1.0;
        }
        self.width as f32 / self.source_width as f32
    }

    fn new(width: u32, height: u32, format: Format, rgba: Vec<u8>, path: &'static str) -> Decoded {
        Decoded::from_rgba(width, height, format, rgba, path)
    }

    /// Build a full-resolution decode from tightly packed RGBA8.
    pub fn from_rgba(
        width: u32,
        height: u32,
        format: Format,
        rgba: Vec<u8>,
        path: &'static str,
    ) -> Decoded {
        Decoded { width, height, source_width: width, source_height: height, format, rgba, path, tiles: None }
    }
}

/// Read an image's dimensions from its header, without decoding it.
///
/// The browser shows the size of whatever row is selected, and that must not
/// cost a decode — moving down a folder of HEICs would take most of a second
/// per keystroke.
pub fn probe_dimensions(path: &std::path::Path) -> Option<(u32, u32)> {
    image::ImageReader::open(path).ok()?.with_guessed_format().ok()?.into_dimensions().ok()
}

/// Decode at full resolution.
pub fn decode(bytes: &[u8]) -> Result<Decoded> {
    decode_to(bytes, None)
}

/// Decode, using no more resolution than `target` needs.
pub fn decode_to(bytes: &[u8], target: Option<Target>) -> Result<Decoded> {
    let format = sniff(bytes).context("unrecognised image format")?;
    decode_as(bytes, format, target)
}

/// As [`decode_to`], but allowed to consult the file name when the bytes are
/// not self-describing.
pub fn decode_path(bytes: &[u8], path: &std::path::Path, target: Option<Target>) -> Result<Decoded> {
    let format = sniff_path(bytes, path).context("unrecognised image format")?;
    decode_as(bytes, format, target)
}

pub fn decode_as(bytes: &[u8], format: Format, target: Option<Target>) -> Result<Decoded> {
    let fast = match format {
        Format::Jpeg => Some(decode_jpeg(bytes, target)),
        Format::Png => Some(decode_png(bytes)),
        Format::Jxl => Some(decode_jxl(bytes)),
        Format::Heif => Some(crate::heif::decode(bytes, format)),
        Format::Avif if !cfg!(feature = "avif-system-dav1d") => {
            Some(crate::heif::decode(bytes, format))
        }
        _ => None,
    };

    let mut decoded = match fast {
        Some(Ok(d)) => d,
        Some(Err(fast_err)) => decode_generic(bytes, format).map_err(|generic_err| {
            fast_err.context(format!("fallback decoder also failed: {generic_err}"))
        })?,
        None => decode_generic(bytes, format)?,
    };

    if let Some(target) = target {
        reduce(&mut decoded, target);
    }
    Ok(decoded)
}

/// Box filter whatever the decoder produced down towards `target`.
///
/// Only whole factors of two or more apply. A residual below 2x is left alone:
/// closing it would need a general resampler, and a texture somewhat larger
/// than the window costs only memory while guaranteeing the fitted image is
/// never soft.
fn reduce(d: &mut Decoded, target: Target) {
    let factor = factor_for_target(d.width, d.height, target);
    if factor < 2 {
        return;
    }

    let mut span = timing::span("decode", "reduce");
    span.note(format!("{}x{} /{factor} for {}x{}", d.width, d.height, target.width, target.height));

    let (px, w, h) = crate::scale::box_downscale(&d.rgba, d.width, d.height, factor);
    d.rgba = px;
    d.width = w;
    d.height = h;
}

/// libjpeg-turbo first. Behind it, two Rust decoders, chosen by how much
/// resolution is wanted.
///
/// zune-jpeg is the faster full-resolution decoder, so it wins whenever we need
/// every pixel. But it cannot decode at reduced scale, and once the target is
/// at least half size a scaled inverse DCT skips so much work that the slower
/// decoder finishes far ahead.
fn decode_jpeg(bytes: &[u8], target: Option<Target>) -> Result<Decoded> {
    // libjpeg-turbo first: its SIMD Huffman decoding is twice as fast as
    // either Rust decoder, full size or scaled. It declines CMYK, and a file
    // it cannot read may still be one the others can.
    match crate::turbojpeg::decode(bytes, target) {
        Ok(d) => return Ok(d),
        Err(e) => timing::mark_note("decode", "turbojpeg_failed", e.to_string()),
    }
    if let Some(target) = target {
        if let Some(dims) = jpeg_dimensions(bytes) {
            if target.reduction(dims.0, dims.1) >= 2.0 {
                match decode_jpeg_scaled(bytes, target, dims) {
                    Ok(d) => return Ok(d),
                    // Scaled decoding is an optimisation; if this particular
                    // file defeats it, full resolution is still correct.
                    Err(e) => timing::mark_note("decode", "scaled_jpeg_failed", e.to_string()),
                }
            }
        }
    }
    decode_jpeg_full(bytes)
}

/// Read just the SOF header for the dimensions.
fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let mut dec = jpeg_decoder::Decoder::new(bytes);
    dec.read_info().ok()?;
    let info = dec.info()?;
    Some((info.width as u32, info.height as u32))
}

/// jpeg-decoder with a reduced inverse DCT: 1/2, 1/4 or 1/8 scale, picked as
/// the smallest reduction that still covers the target.
fn decode_jpeg_scaled(bytes: &[u8], target: Target, source: (u32, u32)) -> Result<Decoded> {
    let mut span = timing::span("decode", "jpeg_scaled");

    let mut dec = jpeg_decoder::Decoder::new(bytes);
    let (w, h) = dec
        .scale(target.width.min(u16::MAX as u32) as u16, target.height.min(u16::MAX as u32) as u16)
        .map_err(|e| anyhow::anyhow!("jpeg scale: {e}"))?;
    let (w, h) = (w as u32, h as u32);

    let pixels = dec.decode().map_err(|e| anyhow::anyhow!("jpeg scaled decode: {e}"))?;
    let info = dec.info().context("jpeg reported no info after decode")?;

    let rgba = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => widen(&pixels, w, h, Channels::Rgb),
        jpeg_decoder::PixelFormat::L8 => widen(&pixels, w, h, Channels::Gray),
        // CMYK and 16-bit greyscale need colour handling we would rather not
        // duplicate; let the full-resolution path deal with them.
        other => bail!("scaled jpeg produced {other:?}"),
    };

    let factor = source.0 as f32 / w.max(1) as f32;
    span.note(format!("{}x{} -> {w}x{h} (1/{:.0} idct)", source.0, source.1, factor));

    Ok(Decoded {
        width: w,
        height: h,
        source_width: source.0,
        source_height: source.1,
        format: Format::Jpeg,
        rgba,
        path: "jpeg-decoder/idct",
        tiles: None,
    })
}

/// zune-jpeg, asked for RGBA directly so there is no separate widening pass.
fn decode_jpeg_full(bytes: &[u8]) -> Result<Decoded> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    let mut span = timing::span("decode", "jpeg");

    // zune defaults to a 16384px cap, which a stitched panorama sails past.
    // A viewer has no business second-guessing how big the user's file is.
    let opts = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGBA)
        .set_max_width(usize::MAX)
        .set_max_height(usize::MAX);
    let mut dec = JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
    dec.decode_headers().map_err(|e| anyhow::anyhow!("jpeg headers: {e:?}"))?;

    let info = dec.info().context("jpeg reported no image info")?;
    let (w, h) = (info.width as u32, info.height as u32);
    let want = w as usize * h as usize * 4;

    let size = dec.output_buffer_size().context("jpeg output size unknown")?;
    if size != want {
        // The requested colourspace conversion didn't happen (some exotic
        // component layouts refuse RGBA); let the generic path handle it.
        bail!("jpeg output is {size} bytes, expected RGBA {want}");
    }

    // Decode straight into our own allocation rather than taking zune's Vec,
    // so the buffer we hand the GPU is one we control the layout of.
    let mut rgba = vec![0u8; want];
    dec.decode_into(&mut rgba).map_err(|e| anyhow::anyhow!("jpeg decode: {e:?}"))?;

    span.note(format!("{w}x{h} zune"));
    Ok(Decoded::new(w, h, Format::Jpeg, rgba, "zune-jpeg"))
}

/// The `png` crate directly, so we can pick the transformations and do the
/// channel widening ourselves in parallel.
fn decode_png(bytes: &[u8]) -> Result<Decoded> {
    let mut span = timing::span("decode", "png");

    let mut dec = png::Decoder::new(bytes);
    // Palette -> RGB, sub-byte depths -> 8, 16-bit -> 8. Leaves us with one of
    // four possible layouts, all of which widen to RGBA cheaply.
    dec.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = dec.read_info()?;

    let mut raw = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut raw)?;
    let (w, h) = (info.width, info.height);
    raw.truncate(info.buffer_size());

    let rgba = match info.color_type {
        png::ColorType::Rgba => raw,
        png::ColorType::Rgb => widen(&raw, w, h, Channels::Rgb),
        png::ColorType::GrayscaleAlpha => widen(&raw, w, h, Channels::GrayAlpha),
        png::ColorType::Grayscale => widen(&raw, w, h, Channels::Gray),
        other => bail!("unexpected png output layout {other:?}"),
    };

    span.note(format!("{w}x{h} png-crate"));
    Ok(Decoded::new(w, h, Format::Png, rgba, "png"))
}

/// jxl-oxide. The `image` crate has no JPEG XL support, so there is no fallback
/// behind this one.
fn decode_jxl(bytes: &[u8]) -> Result<Decoded> {
    let mut span = timing::span("decode", "jxl");

    let image = jxl_oxide::JxlImage::read_with_defaults(bytes)
        .map_err(|e| anyhow::anyhow!("jxl parse: {e}"))?;
    let (w, h) = (image.width(), image.height());

    let render = image.render_frame(0).map_err(|e| anyhow::anyhow!("jxl render: {e}"))?;
    let mut stream = render.stream();
    let channels = stream.channels() as usize;

    let mut buf = vec![0u8; w as usize * h as usize * channels];
    stream.write_to_buffer(&mut buf);

    let rgba = match channels {
        4 => buf,
        3 => widen(&buf, w, h, Channels::Rgb),
        2 => widen(&buf, w, h, Channels::GrayAlpha),
        1 => widen(&buf, w, h, Channels::Gray),
        n => bail!("jxl produced {n} channels"),
    };

    span.note(format!("{w}x{h} jxl-oxide {channels}ch"));
    Ok(Decoded::new(w, h, Format::Jxl, rgba, "jxl-oxide"))
}

/// Everything without a hand-rolled path, plus the fallback for the ones that
/// have one.
fn decode_generic(bytes: &[u8], format: Format) -> Result<Decoded> {
    let mut span = timing::span("decode", "generic");

    match format.support() {
        Support::Missing(why) => bail!("{} is not supported by this build: {why}", format.name()),
        Support::MissingRuntime(why) => bail!("{} is unavailable: {why}", format.name()),
        _ => {}
    }
    let img_format = format
        .to_image_format()
        .with_context(|| format!("no decoder wired up for {}", format.name()))?;

    let img = image::load_from_memory_with_format(bytes, img_format)
        .with_context(|| format!("decoding {}", format.name()))?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());

    span.note(format!("{w}x{h} image-crate {}", format.name()));
    Ok(Decoded::new(w, h, format, rgba.into_raw(), "image"))
}

#[derive(Copy, Clone)]
enum Channels {
    Rgb,
    Gray,
    GrayAlpha,
}

impl Channels {
    fn count(self) -> usize {
        match self {
            Channels::Rgb => 3,
            Channels::Gray => 1,
            Channels::GrayAlpha => 2,
        }
    }
}

/// Widen a narrower layout to RGBA8. Memory-bandwidth bound, so it is worth
/// splitting across cores once the image is big enough to cover the overhead.
fn widen(src: &[u8], w: u32, h: u32, ch: Channels) -> Vec<u8> {
    let px = w as usize * h as usize;
    let n = ch.count();
    let mut out = vec![0u8; px * 4];

    let convert = |src: &[u8], dst: &mut [u8]| {
        for (s, d) in src.chunks_exact(n).zip(dst.chunks_exact_mut(4)) {
            match ch {
                Channels::Rgb => {
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    d[3] = 0xFF;
                }
                Channels::Gray => {
                    d[0] = s[0];
                    d[1] = s[0];
                    d[2] = s[0];
                    d[3] = 0xFF;
                }
                Channels::GrayAlpha => {
                    d[0] = s[0];
                    d[1] = s[0];
                    d[2] = s[0];
                    d[3] = s[1];
                }
            }
        }
    };

    const PARALLEL_THRESHOLD: usize = 256 * 1024;
    if px < PARALLEL_THRESHOLD {
        convert(&src[..px * n], &mut out);
    } else {
        let rows_per_job = (h as usize / rayon::current_num_threads().max(1)).max(1);
        out.par_chunks_mut(rows_per_job * w as usize * 4)
            .zip(src.par_chunks(rows_per_job * w as usize * n))
            .for_each(|(dst, src)| convert(src, dst));
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: Target = Target { width: 3072, height: 1728 };

    #[test]
    fn turbojpeg_agrees_with_the_rust_decoders() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/corpus/base.jpg");
        let Ok(bytes) = std::fs::read(path) else {
            return; // The corpus is fetched separately; nothing to compare.
        };
        let turbo = crate::turbojpeg::decode(&bytes, None).unwrap();
        let zune = decode_jpeg_full(&bytes).unwrap();
        assert_eq!((turbo.width, turbo.height), (zune.width, zune.height));
        // Two correct decoders differ by rounding in the IDCT and upsampling,
        // never by more than a few levels on average.
        let diff: u64 = turbo.rgba.iter().zip(&zune.rgba).map(|(a, b)| a.abs_diff(*b) as u64).sum();
        let mean = diff as f64 / turbo.rgba.len() as f64;
        assert!(mean < 2.0, "mean difference {mean}");

        // And scaled, the size the viewer asked for.
        let half = crate::turbojpeg::decode(&bytes, Some(Target { width: 960, height: 540 })).unwrap();
        assert_eq!((half.width, half.height, half.source_width), (960, 540, 1920));
    }

    #[test]
    fn sniffing_believes_bytes_not_extensions() {
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0];
        jpeg.extend_from_slice(&[0; 16]);
        assert_eq!(sniff(&jpeg), Some(Format::Jpeg));

        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&[0; 16]);
        assert_eq!(sniff(&png), Some(Format::Png));

        let mut avif = vec![0; 4];
        avif.extend_from_slice(b"ftypavif");
        assert_eq!(sniff(&avif), Some(Format::Avif));

        let mut heif = vec![0; 4];
        heif.extend_from_slice(b"ftypheic");
        assert_eq!(sniff(&heif), Some(Format::Heif));

        assert_eq!(sniff(b"not an image at all"), None);
    }

    #[test]
    fn reduction_follows_the_binding_axis() {
        // 8000x6000 fitted into 3072x1728 is limited by height: 6000/1728.
        let r = SCREEN.reduction(8000, 6000);
        assert!((r - 6000.0 / 1728.0).abs() < 1e-4, "{r}");

        // Already smaller than the box.
        assert!(SCREEN.reduction(1920, 1080) < 1.0);
    }

    #[test]
    fn factor_never_drops_below_the_target() {
        // 3.47x is allowed, so 3x is the largest whole step.
        assert_eq!(factor_for_target(8000, 6000, SCREEN), 3);

        // 1.74x allowed: rounding up to 2 would undershoot the screen, so the
        // image is left alone.
        assert_eq!(factor_for_target(4000, 3000, SCREEN), 1);

        // Smaller than the target: never enlarge.
        assert_eq!(factor_for_target(640, 480, SCREEN), 1);
    }

    #[test]
    fn reduced_images_report_the_zoom_they_can_serve() {
        let d = Decoded {
            width: 1000,
            height: 750,
            source_width: 8000,
            source_height: 6000,
            format: Format::Jpeg,
            rgba: Vec::new(),
            path: "test",
            tiles: None,
        };
        assert!(d.is_reduced());
        assert!((d.adequate_zoom() - 0.125).abs() < 1e-6);

        let full = Decoded::new(1920, 1080, Format::Png, Vec::new(), "test");
        assert!(!full.is_reduced());
        assert_eq!(full.adequate_zoom(), 1.0);
    }
}
