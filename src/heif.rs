//! HEIC/HEIF support through a runtime-loaded libheif.
//!
//! libheif is never linked. It is opened with `dlopen` the first time a HEIF
//! file is actually seen, so the binary starts and runs normally on machines
//! that do not have it — the format is simply reported as unavailable, with a
//! way to fix that. This is also the arrangement libheif's LGPL licence is
//! written for: dynamic loading carries no relink obligation, where static
//! linking would.
//!
//! There is no pure-Rust HEVC decoder to use instead. That is the whole reason
//! this module exists.
//!
//! AVIF comes through here too. It is the same container with AV1 inside, and
//! libheif decodes it given its dav1d or aom plugin — which spares the binary
//! the link-time libdav1d dependency the `avif-system-dav1d` feature carries.
//!
//! Because `dlopen` gives up compile-time signature checking, every declaration
//! below was verified against the real library before being written: the
//! enum values, the struct-by-value return of `heif_error`, and the full decode
//! path were exercised end to end first.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;

use anyhow::{Result, bail};
use libloading::{Library, Symbol};

use crate::decode::{Decoded, Format, Target};
use crate::timing;

/// The soname, not the `-dev` symlink: we want whatever runtime is installed,
/// and we never want to require the development package.
const SONAME: &str = "libheif.so.1";

/// `heif_init` and `heif_have_decoder_for_format` arrived in 1.13, along with
/// the loadable-plugin system that supplies the actual HEVC decoder.
const MIN_VERSION: (u8, u8) = (1, 13);

/// `heif_compression_HEVC`. Verified against the library: 1 is HEVC, 2 is AVC.
/// Getting this wrong is silent — it reports "no decoder" rather than failing.
const COMPRESSION_HEVC: c_int = 1;
/// `heif_compression_AV1`.
const COMPRESSION_AV1: c_int = 4;
/// `heif_colorspace_RGB`.
const COLORSPACE_RGB: c_int = 1;
/// `heif_chroma_interleaved_RGBA`, which lets libheif hand us our exact layout.
const CHROMA_INTERLEAVED_RGBA: c_int = 11;
/// `heif_channel_interleaved`.
const CHANNEL_INTERLEAVED: c_int = 10;

/// libheif returns this by value from most calls. Two ints and a pointer, which
/// the SysV ABI passes in registers; `repr(C)` matches.
#[repr(C)]
struct HeifError {
    code: c_int,
    subcode: c_int,
    message: *const c_char,
}

impl HeifError {
    fn check(&self, what: &str) -> Result<()> {
        if self.code == 0 {
            return Ok(());
        }
        let msg = if self.message.is_null() {
            "unknown error".to_string()
        } else {
            // Safety: libheif owns this string and keeps it alive for the
            // process; we only read it.
            unsafe { CStr::from_ptr(self.message) }.to_string_lossy().into_owned()
        };
        bail!("libheif {what}: {msg} (code {}/{})", self.code, self.subcode)
    }
}

type FnInit = unsafe extern "C" fn(*mut c_void) -> HeifError;
type FnVersionNumber = unsafe extern "C" fn() -> u32;
type FnVersion = unsafe extern "C" fn() -> *const c_char;
type FnHaveDecoder = unsafe extern "C" fn(c_int) -> c_int;
type FnContextAlloc = unsafe extern "C" fn() -> *mut c_void;
type FnContextFree = unsafe extern "C" fn(*mut c_void);
type FnSetThreads = unsafe extern "C" fn(*mut c_void, c_int);
type FnHandleDim = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnThumbnailIds = unsafe extern "C" fn(*mut c_void, *mut u32, c_int) -> c_int;
type FnThumbnail = unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> HeifError;

/// Below this many pixels the whole image decodes about as fast as a blurry
/// stand-in would be worth showing.
const THUMBNAIL_WORTH_FROM: u64 = 16_000_000;

/// Whether the embedded thumbnail is worth showing while `primary` decodes:
/// only for a big image, and only if the thumbnail is no worse than a quarter
/// of the size it will be seen at — an iPhone photo's 320x240 stretched across
/// a screen is a smear, not a preview.
fn thumbnail_worth(primary: (u32, u32), thumb: (u32, u32), target: Target) -> bool {
    if (primary.0 as u64 * primary.1 as u64) < THUMBNAIL_WORTH_FROM {
        return false;
    }
    let seen_width = primary.0 as f32 / target.reduction(primary.0, primary.1).max(1.0);
    thumb.0 as f32 * 4.0 >= seen_width
}
type FnReadMemory = unsafe extern "C" fn(*mut c_void, *const u8, usize, *const c_void) -> HeifError;
type FnPrimaryHandle = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> HeifError;
type FnHandleRelease = unsafe extern "C" fn(*mut c_void);
type FnDecodeImage =
    unsafe extern "C" fn(*mut c_void, *mut *mut c_void, c_int, c_int, *const c_void) -> HeifError;
type FnImageRelease = unsafe extern "C" fn(*mut c_void);
type FnImageDim = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type FnPlane = unsafe extern "C" fn(*mut c_void, c_int, *mut c_int) -> *const u8;

/// What this machine can actually do with HEIF, and what to do about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Availability {
    Ready { version: String },
    /// libheif is not installed at all.
    NotInstalled,
    /// Installed, but predates the plugin system we rely on.
    TooOld { version: String },
    /// Installed and new enough, but no decoder plugin for the codec is
    /// present, so it can parse the file and not decode it. This is a real and
    /// confusing state, and a version check alone would miss it entirely.
    NoDecoder { version: String, codec: &'static str },
    /// Present but missing symbols we need, i.e. not the library we expect.
    Unusable { reason: String },
}

impl Availability {
    pub fn is_ready(&self) -> bool {
        matches!(self, Availability::Ready { .. })
    }

    /// One line for `--formats` and error messages.
    pub fn summary(&self) -> String {
        match self {
            Availability::Ready { version } => format!("libheif {version}"),
            Availability::NotInstalled => format!("{SONAME} not installed"),
            Availability::TooOld { version } => {
                format!("libheif {version} < {}.{} needed", MIN_VERSION.0, MIN_VERSION.1)
            }
            Availability::NoDecoder { version, codec } => {
                format!("libheif {version}, no {codec} plugin")
            }
            Availability::Unusable { reason } => format!("{SONAME} unusable: {reason}"),
        }
    }
}

fn library() -> Option<&'static Library> {
    static LIB: OnceLock<Option<Library>> = OnceLock::new();
    LIB.get_or_init(|| {
        let _s = timing::span("heif", "dlopen");
        // Safety: loading a shared library runs its initialisers. libheif is a
        // well-behaved system library; this is the same thing ld.so would do
        // had we linked it.
        unsafe { Library::new(SONAME) }.ok()
    })
    .as_ref()
}

fn sym<T>(lib: &'static Library, name: &[u8]) -> Result<Symbol<'static, T>> {
    // Safety: the signatures these are cast to were verified against the real
    // library before being declared above.
    unsafe { lib.get::<T>(name) }.map_err(|e| {
        anyhow::anyhow!("{} is missing {}: {e}", SONAME, String::from_utf8_lossy(name))
    })
}

/// Probe once and remember. Called lazily — a session that only opens JPEGs
/// never pays for this, and never mentions HEIF.
pub fn availability() -> &'static Availability {
    static STATE: OnceLock<Availability> = OnceLock::new();
    STATE.get_or_init(|| probe(COMPRESSION_HEVC, "HEVC"))
}

/// As [`availability`], for AVIF: the same library, a different plugin.
pub fn avif_availability() -> &'static Availability {
    static STATE: OnceLock<Availability> = OnceLock::new();
    STATE.get_or_init(|| probe(COMPRESSION_AV1, "AV1"))
}

fn availability_of(format: Format) -> &'static Availability {
    match format {
        Format::Avif => avif_availability(),
        _ => availability(),
    }
}

fn probe(compression: c_int, codec: &'static str) -> Availability {
    let Some(lib) = library() else {
        return Availability::NotInstalled;
    };

    let version = match sym::<FnVersion>(lib, b"heif_get_version\0") {
        // Safety: takes nothing, returns a static C string owned by libheif.
        Ok(f) => unsafe { CStr::from_ptr(f()) }.to_string_lossy().into_owned(),
        Err(e) => return Availability::Unusable { reason: e.to_string() },
    };

    let number = match sym::<FnVersionNumber>(lib, b"heif_get_version_number\0") {
        Ok(f) => unsafe { f() },
        Err(e) => return Availability::Unusable { reason: e.to_string() },
    };
    // Packed as 0xMMmmpp00.
    let (major, minor) = (((number >> 24) & 0xFF) as u8, ((number >> 16) & 0xFF) as u8);
    if (major, minor) < MIN_VERSION {
        return Availability::TooOld { version };
    }

    // Plugins are what actually decode; loading them is what makes the
    // capability query below meaningful.
    if let Ok(init) = sym::<FnInit>(lib, b"heif_init\0") {
        unsafe { init(std::ptr::null_mut()) };
    }

    match sym::<FnHaveDecoder>(lib, b"heif_have_decoder_for_format\0") {
        Ok(have) if unsafe { have(compression) } != 0 => Availability::Ready { version },
        Ok(_) => Availability::NoDecoder { version, codec },
        Err(e) => Availability::Unusable { reason: e.to_string() },
    }
}

/// Decode a HEIF/HEIC or AVIF file to RGBA8.
pub fn decode(bytes: &[u8], format: Format) -> Result<Decoded> {
    Ok(decode_with(bytes, format, None)?.expect("the primary image is always decoded"))
}

/// The file's embedded thumbnail, to show while the real decode runs — `None`
/// when there is none, or when it would not be worth looking at (see
/// [`thumbnail_worth`]). Its source size is the primary image's, so the view
/// fits it exactly where the real one will land.
pub fn thumbnail(bytes: &[u8], format: Format, target: Target) -> Result<Option<Decoded>> {
    decode_with(bytes, format, Some(target))
}

/// The primary image's size, from the container alone — nothing is decoded.
pub fn dimensions(bytes: &[u8], format: Format) -> Option<(u32, u32)> {
    if !availability_of(format).is_ready() {
        return None;
    }
    let lib = library()?;
    let context_alloc = sym::<FnContextAlloc>(lib, b"heif_context_alloc\0").ok()?;
    let context_free = sym::<FnContextFree>(lib, b"heif_context_free\0").ok()?;
    let read_memory = sym::<FnReadMemory>(lib, b"heif_context_read_from_memory_without_copy\0").ok()?;
    let primary = sym::<FnPrimaryHandle>(lib, b"heif_context_get_primary_image_handle\0").ok()?;
    let handle_release = sym::<FnHandleRelease>(lib, b"heif_image_handle_release\0").ok()?;
    let width = sym::<FnHandleDim>(lib, b"heif_image_handle_get_width\0").ok()?;
    let height = sym::<FnHandleDim>(lib, b"heif_image_handle_get_height\0").ok()?;

    let ctx = Owned::new(unsafe { context_alloc() }, |p| unsafe { context_free(p) });
    if ctx.ptr.is_null() {
        return None;
    }
    unsafe { read_memory(ctx.ptr, bytes.as_ptr(), bytes.len(), std::ptr::null()) }.check("reading").ok()?;
    let mut handle_ptr = std::ptr::null_mut();
    unsafe { primary(ctx.ptr, &mut handle_ptr) }.check("primary").ok()?;
    let handle = Owned::new(handle_ptr, |p| unsafe { handle_release(p) });
    let (w, h) = unsafe { (width(handle.ptr), height(handle.ptr)) };
    (w > 0 && h > 0).then_some((w as u32, h as u32))
}

fn decode_with(bytes: &[u8], format: Format, thumbnail_for: Option<Target>) -> Result<Option<Decoded>> {
    match availability_of(format) {
        Availability::Ready { .. } => {}
        other => bail!("{}", other.summary()),
    }
    let lib = library().expect("availability() proved the library loads");

    let mut span = timing::span("decode", if thumbnail_for.is_some() { "heif_thumbnail" } else { format.name() });

    let context_alloc = sym::<FnContextAlloc>(lib, b"heif_context_alloc\0")?;
    let context_free = sym::<FnContextFree>(lib, b"heif_context_free\0")?;
    let read_memory =
        sym::<FnReadMemory>(lib, b"heif_context_read_from_memory_without_copy\0")?;
    let primary = sym::<FnPrimaryHandle>(lib, b"heif_context_get_primary_image_handle\0")?;
    let handle_release = sym::<FnHandleRelease>(lib, b"heif_image_handle_release\0")?;
    let decode_image = sym::<FnDecodeImage>(lib, b"heif_decode_image\0")?;
    let image_release = sym::<FnImageRelease>(lib, b"heif_image_release\0")?;
    let get_width = sym::<FnImageDim>(lib, b"heif_image_get_width\0")?;
    let get_height = sym::<FnImageDim>(lib, b"heif_image_get_height\0")?;
    let get_plane = sym::<FnPlane>(lib, b"heif_image_get_plane_readonly\0")?;

    // Every early return past this point has to free what it allocated, so the
    // owning pointers are wrapped in guards rather than freed by hand.
    let ctx = Owned::new(unsafe { context_alloc() }, |p| unsafe { context_free(p) });
    if ctx.ptr.is_null() {
        bail!("libheif could not allocate a context");
    }

    // An iPhone HEIC is a grid of independently coded tiles, which libheif
    // decodes in parallel — on four threads unless told otherwise. Every core
    // took a 61 MP panorama from 950 ms to 400 ms on a 32-thread machine.
    // Optional: a library without the call still decodes, just slower.
    if let Ok(set_threads) = sym::<FnSetThreads>(lib, b"heif_context_set_max_decoding_threads\0") {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        unsafe { set_threads(ctx.ptr, cores.min(64) as c_int) };
    }

    // `_without_copy` borrows our buffer, so `bytes` must outlive the context.
    // It does: the context is dropped at the end of this function.
    unsafe { read_memory(ctx.ptr, bytes.as_ptr(), bytes.len(), std::ptr::null()) }
        .check("reading the file")?;

    let mut handle_ptr = std::ptr::null_mut();
    unsafe { primary(ctx.ptr, &mut handle_ptr) }.check("finding the primary image")?;
    let primary_handle = Owned::new(handle_ptr, |p| unsafe { handle_release(p) });
    let mut source = None;
    let thumb_handle;
    let handle = match thumbnail_for {
        None => primary_handle.ptr,
        Some(target) => {
            let handle_width = sym::<FnHandleDim>(lib, b"heif_image_handle_get_width\0")?;
            let handle_height = sym::<FnHandleDim>(lib, b"heif_image_handle_get_height\0")?;
            let list = sym::<FnThumbnailIds>(lib, b"heif_image_handle_get_list_of_thumbnail_IDs\0")?;
            let get = sym::<FnThumbnail>(lib, b"heif_image_handle_get_thumbnail\0")?;

            let mut id = 0u32;
            if unsafe { list(primary_handle.ptr, &mut id, 1) } < 1 {
                return Ok(None);
            }
            let mut ptr = std::ptr::null_mut();
            unsafe { get(primary_handle.ptr, id, &mut ptr) }.check("finding the thumbnail")?;
            thumb_handle = Owned::new(ptr, |p| unsafe { handle_release(p) });

            let dims = |h: *mut c_void| unsafe { (handle_width(h).max(0) as u32, handle_height(h).max(0) as u32) };
            let (primary, thumb) = (dims(primary_handle.ptr), dims(thumb_handle.ptr));
            if !thumbnail_worth(primary, thumb, target) {
                return Ok(None);
            }
            source = Some(primary);
            thumb_handle.ptr
        }
    };

    let mut image_ptr = std::ptr::null_mut();
    unsafe {
        decode_image(handle, &mut image_ptr, COLORSPACE_RGB, CHROMA_INTERLEAVED_RGBA, std::ptr::null())
    }
    .check("decoding")?;
    let image = Owned::new(image_ptr, |p| unsafe { image_release(p) });

    let width = unsafe { get_width(image.ptr, CHANNEL_INTERLEAVED) };
    let height = unsafe { get_height(image.ptr, CHANNEL_INTERLEAVED) };
    if width <= 0 || height <= 0 {
        bail!("libheif reported a {width}x{height} image");
    }
    let (width, height) = (width as u32, height as u32);

    let mut stride: c_int = 0;
    let plane = unsafe { get_plane(image.ptr, CHANNEL_INTERLEAVED, &mut stride) };
    if plane.is_null() || stride <= 0 {
        bail!("libheif returned no pixel plane");
    }
    let stride = stride as usize;
    let row_bytes = width as usize * 4;
    if stride < row_bytes {
        bail!("libheif stride {stride} is shorter than a {row_bytes}-byte row");
    }

    // Rows are padded to `stride`, so copy row by row rather than wholesale.
    // For the common case where the two are equal this is one memcpy per row,
    // which the allocator-free path below keeps cheap.
    let mut rgba = vec![0u8; row_bytes * height as usize];
    for y in 0..height as usize {
        // Safety: libheif guarantees `stride * height` readable bytes from
        // `plane`, and we bounded `row_bytes <= stride` above.
        let src = unsafe { std::slice::from_raw_parts(plane.add(y * stride), row_bytes) };
        rgba[y * row_bytes..(y + 1) * row_bytes].copy_from_slice(src);
    }

    span.note(format!("{width}x{height} libheif"));
    let mut decoded = Decoded::from_rgba(width, height, format, rgba, "libheif");
    if let Some((w, h)) = source {
        decoded.source_width = w;
        decoded.source_height = h;
        decoded.path = "libheif thumbnail";
    }
    Ok(Some(decoded))
}

/// Frees a libheif pointer on drop, so error paths cannot leak it.
struct Owned<F: FnMut(*mut c_void)> {
    ptr: *mut c_void,
    free: F,
}

impl<F: FnMut(*mut c_void)> Owned<F> {
    fn new(ptr: *mut c_void, free: F) -> Self {
        Owned { ptr, free }
    }
}

impl<F: FnMut(*mut c_void)> Drop for Owned<F> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            (self.free)(self.ptr);
        }
    }
}
