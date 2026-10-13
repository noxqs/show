//! JPEG through libjpeg-turbo, linked statically.
//!
//! Its SIMD Huffman decoder is the part the Rust decoders cannot match, and
//! Huffman decoding is the floor of every JPEG decode: on a 47 MP panorama it
//! took a half-size decode from 154 ms to 66 ms, and a full one from 118 ms to
//! 81 ms. Scaled decoding (1/2, 1/4, 1/8) comes with it.
//!
//! Built from the source `turbojpeg-sys` bundles (see Cargo.toml), so it adds
//! no runtime dependency and is the same speed on every machine. jpeg-decoder
//! and zune-jpeg stay behind it for the files it declines — CMYK — or fails on.

use std::ffi::{CStr, c_int, c_ulong};

use anyhow::{Result, bail};
use turbojpeg_sys as tj;

use crate::decode::{Decoded, Format, Target};
use crate::timing;

/// `TJCS_CMYK` and `TJCS_YCCK`. RGBA output from these needs colour handling
/// the fallback path already has, so they are left to it.
const CS_CMYK: c_int = 3;
const CS_YCCK: c_int = 4;
/// `TJERR_WARNING`: the image decoded, but something about the file was off —
/// typically a truncated scan. The same leniency the other decoders show.
const ERR_WARNING: c_int = 0;

/// A decompressor, destroyed on drop so no error path can leak it. Not shared
/// between threads: a handle is not thread-safe, and making one is cheap.
struct Handle(tj::tjhandle);

impl Handle {
    fn error(&self) -> String {
        // Safety: the string is owned by the handle and valid until its next
        // call, and it is copied out before one is made.
        let msg = unsafe { tj::tjGetErrorStr2(self.0) };
        if msg.is_null() {
            "unknown error".into()
        } else {
            unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned()
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { tj::tjDestroy(self.0) };
    }
}

/// Decode to RGBA8, at the smallest of 1/1, 1/2, 1/4 and 1/8 that still
/// covers `target`.
pub fn decode(bytes: &[u8], target: Option<Target>) -> Result<Decoded> {
    let handle = Handle(unsafe { tj::tjInitDecompress() });
    if handle.0.is_null() {
        bail!("libjpeg-turbo could not make a decompressor");
    }

    let (mut w, mut h, mut subsamp, mut colorspace) = (0, 0, 0, 0);
    let rc = unsafe {
        tj::tjDecompressHeader3(
            handle.0,
            bytes.as_ptr(),
            bytes.len() as c_ulong,
            &mut w,
            &mut h,
            &mut subsamp,
            &mut colorspace,
        )
    };
    if rc != 0 {
        bail!("libjpeg-turbo header: {}", handle.error());
    }
    if w <= 0 || h <= 0 {
        bail!("libjpeg-turbo reported a {w}x{h} image");
    }
    if colorspace == CS_CMYK || colorspace == CS_YCCK {
        bail!("CMYK JPEG, left to the fallback decoder");
    }
    let (w, h) = (w as u32, h as u32);

    // The largest power-of-two reduction that still fills the target. Asking
    // for exactly that size makes the library pick that factor.
    let factor = target.map(|t| crate::decode::factor_for_target(w, h, t)).unwrap_or(1);
    let div = [8u32, 4, 2, 1].into_iter().find(|d| *d <= factor).unwrap_or(1);
    let (ow, oh) = (w.div_ceil(div), h.div_ceil(div));

    let mut span = timing::span("decode", if div > 1 { "jpeg_scaled" } else { "jpeg" });
    let mut rgba = vec![0u8; ow as usize * oh as usize * 4];
    let rc = unsafe {
        tj::tjDecompress2(
            handle.0,
            bytes.as_ptr(),
            bytes.len() as c_ulong,
            rgba.as_mut_ptr(),
            ow as c_int,
            0,
            oh as c_int,
            tj::TJPF_TJPF_RGBA as c_int,
            0,
        )
    };
    if rc != 0 && unsafe { tj::tjGetErrorCode(handle.0) } != ERR_WARNING {
        bail!("libjpeg-turbo decode: {}", handle.error());
    }

    span.note(if div > 1 {
        format!("{w}x{h} -> {ow}x{oh} (1/{div}) turbo")
    } else {
        format!("{w}x{h} turbo")
    });
    Ok(Decoded {
        width: ow,
        height: oh,
        source_width: w,
        source_height: h,
        format: Format::Jpeg,
        rgba,
        path: "libjpeg-turbo",
        tiles: None,
    })
}
