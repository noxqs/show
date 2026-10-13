//! PCM playback through ALSA, loaded at runtime.
//!
//! `cpal` was the obvious alternative and is ruled out for the same reason
//! libheif is: it links `libasound.so.2` at build time, so a binary built with
//! it refuses to start on a machine that lacks the library. Audio is worth
//! having, but not at the cost of the executable running everywhere. Loaded
//! this way, a missing ALSA means silence and a note in the setup panel — the
//! video still plays.
//!
//! Only the handful of entry points a blocking writer needs are bound. The
//! `snd_pcm_set_params` helper does the hardware and software parameter dance
//! in one call, which is most of why this is short.

use anyhow::{Context, Result, anyhow};
use libloading::{Library, Symbol};
use std::ffi::{CStr, CString, c_char, c_int, c_long, c_uint, c_ulong, c_void};

const SND_PCM_STREAM_PLAYBACK: c_int = 0;
const SND_PCM_FORMAT_S16_LE: c_int = 2;
const SND_PCM_ACCESS_RW_INTERLEAVED: c_int = 3;

type OpenFn = unsafe extern "C" fn(*mut *mut c_void, *const c_char, c_int, c_int) -> c_int;
type SetParamsFn = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_uint,
    c_uint,
    c_int,
    c_uint,
) -> c_int;
type WriteiFn = unsafe extern "C" fn(*mut c_void, *const c_void, c_ulong) -> c_long;
type RecoverFn = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> c_int;
type SimpleFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type DelayFn = unsafe extern "C" fn(*mut c_void, *mut c_long) -> c_int;
type StrErrorFn = unsafe extern "C" fn(c_int) -> *const c_char;

struct Api {
    _library: Library,
    open: OpenFn,
    set_params: SetParamsFn,
    writei: WriteiFn,
    recover: RecoverFn,
    prepare: SimpleFn,
    drop_pending: SimpleFn,
    close: SimpleFn,
    delay: DelayFn,
    strerror: StrErrorFn,
}

impl Api {
    fn load() -> Result<Api> {
        // Safety: loading a shared library and reading well-known symbols from
        // it. The signatures are ALSA's documented public API and have been
        // stable for the life of libasound 2.
        unsafe {
            let library = Library::new("libasound.so.2")
                .context("libasound.so.2 could not be loaded; playing without sound")?;
            fn get<T: Copy>(lib: &Library, name: &[u8]) -> Result<T> {
                // Safety: caller states the type; every use below matches the
                // signature in <alsa/pcm.h>.
                unsafe {
                    let symbol: Symbol<T> = lib
                        .get(name)
                        .map_err(|e| anyhow!("{}: {e}", String::from_utf8_lossy(name)))?;
                    Ok(*symbol)
                }
            }
            let open = get(&library, b"snd_pcm_open\0")?;
            let set_params = get(&library, b"snd_pcm_set_params\0")?;
            let writei = get(&library, b"snd_pcm_writei\0")?;
            let recover = get(&library, b"snd_pcm_recover\0")?;
            let prepare = get(&library, b"snd_pcm_prepare\0")?;
            let drop_pending = get(&library, b"snd_pcm_drop\0")?;
            let close = get(&library, b"snd_pcm_close\0")?;
            let delay = get(&library, b"snd_pcm_delay\0")?;
            let strerror = get(&library, b"snd_strerror\0")?;
            Ok(Api {
                _library: library,
                open,
                set_params,
                writei,
                recover,
                prepare,
                drop_pending,
                close,
                delay,
                strerror,
            })
        }
    }

    fn message(&self, code: c_int) -> String {
        // Safety: snd_strerror returns a static string for any input.
        unsafe {
            let ptr = (self.strerror)(code);
            if ptr.is_null() {
                format!("error {code}")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }
}

/// Whether ALSA can be loaded at all, for the setup panel to report.
pub fn available() -> bool {
    api().is_ok()
}

fn api() -> Result<&'static Api> {
    static API: std::sync::OnceLock<Result<Api, String>> = std::sync::OnceLock::new();
    API.get_or_init(|| Api::load().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

/// An open playback device. Interleaved signed 16-bit samples.
pub struct Pcm {
    handle: *mut c_void,
    api: &'static Api,
    channels: u32,
}

// Safety: the handle is only ever touched from the thread that owns the Pcm.
// ALSA permits a handle to be used from another thread so long as it is not
// used from two at once, which ownership already guarantees.
unsafe impl Send for Pcm {}

impl Pcm {
    /// Open the default device. `latency` is a hint, in microseconds.
    pub fn open(rate: u32, channels: u32, latency: u32) -> Result<Pcm> {
        let api = api()?;
        let name = CString::new("default").expect("no interior nul");
        let mut handle: *mut c_void = std::ptr::null_mut();

        // Safety: handle is a valid out-pointer; name outlives the call.
        let code = unsafe {
            (api.open)(&mut handle, name.as_ptr(), SND_PCM_STREAM_PLAYBACK, 0)
        };
        if code < 0 || handle.is_null() {
            return Err(anyhow!("opening the audio device: {}", api.message(code)));
        }
        let pcm = Pcm { handle, api, channels };

        // soft_resample = 1 lets ALSA convert if the hardware will not take
        // this rate, which is the difference between playing and refusing on
        // devices locked to 44.1 kHz.
        // Safety: the handle came from a successful open.
        let code = unsafe {
            (api.set_params)(
                handle,
                SND_PCM_FORMAT_S16_LE,
                SND_PCM_ACCESS_RW_INTERLEAVED,
                channels as c_uint,
                rate as c_uint,
                1,
                latency as c_uint,
            )
        };
        if code < 0 {
            return Err(anyhow!(
                "configuring {rate} Hz, {channels} channels: {}",
                api.message(code)
            ));
        }
        Ok(pcm)
    }

    /// Write interleaved samples, blocking until the device has taken them.
    ///
    /// Underruns are recovered from rather than reported: the speaker has
    /// already missed that audio, and stopping playback over it would turn a
    /// glitch into a failure.
    pub fn write(&mut self, samples: &[i16]) -> Result<()> {
        let frames_total = samples.len() / self.channels as usize;
        let mut written = 0usize;

        while written < frames_total {
            let offset = written * self.channels as usize;
            // Safety: the slice covers frames_total frames; offset is inside it.
            let produced = unsafe {
                (self.api.writei)(
                    self.handle,
                    samples[offset..].as_ptr() as *const c_void,
                    (frames_total - written) as c_ulong,
                )
            };

            if produced < 0 {
                // Safety: recovering a handle we own.
                let code = unsafe { (self.api.recover)(self.handle, produced as c_int, 1) };
                if code < 0 {
                    return Err(anyhow!("audio write: {}", self.api.message(code)));
                }
                continue;
            }
            written += produced as usize;
        }
        Ok(())
    }

    /// Frames already handed over but not yet heard. Subtracting this from what
    /// has been written is what makes the audio clock reflect the speaker
    /// rather than the buffer.
    pub fn delay_frames(&self) -> i64 {
        let mut frames: c_long = 0;
        // Safety: frames is a valid out-pointer.
        let code = unsafe { (self.api.delay)(self.handle, &mut frames) };
        if code < 0 { 0 } else { frames as i64 }
    }

    /// Discard anything buffered and ready the device for more.
    pub fn reset(&mut self) {
        // Safety: both act on a handle we own; failures leave it usable.
        unsafe {
            (self.api.drop_pending)(self.handle);
            (self.api.prepare)(self.handle);
        }
    }
}

impl Drop for Pcm {
    fn drop(&mut self) {
        // Safety: closing a handle we own, exactly once.
        unsafe {
            (self.api.drop_pending)(self.handle);
            (self.api.close)(self.handle);
        }
    }
}
