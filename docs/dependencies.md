# What `show` depends on, and what it refuses to

The binary runs on a stock Linux install with nothing added. That is a property,
not an accident, and it has decided several designs in this project. This file
records what the rule is, where it has been bent, and why each refusal was made
— so the next decision is argued against the same standard rather than from
memory.

## The rule

A default build links against libc and nothing else. Anything more must be
either compiled in, or loaded at runtime and optional.

Three shapes satisfy it:

1. **Compiled in.** Pure-Rust crates, and C libraries vendored and linked
   statically (FFmpeg, via `tools/build-ffmpeg.sh`).
2. **Loaded at runtime.** `dlopen` through `libloading`, so a missing library
   means a feature is absent and says so, not a binary that will not start.
   `src/heif.rs` (libheif) and `src/alsa.rs` (libasound) are the two examples.
3. **Written ourselves.** When the library is large and the part we need is
   small: `src/md5.rs`, `src/typeface.rs`, `src/software.rs`.

## What is loaded at runtime

| Library | Module | Absent means |
|---|---|---|
| `libheif.so.1` | `src/heif.rs` | HEIC and AVIF do not open; the panel says so. AVIF also needs its AV1 plugin (dav1d or aom) |
| `libasound.so.2` | `src/alsa.rs` | video plays silently; the panel says so |

Both are reported on the F1 panel as facts about *this machine*, not about the
build.

## The one exception

`avif-system-dav1d` links libdav1d at build time. It is behind a feature flag
and off by default, and it is the only thing that can give the executable a
non-libc runtime dependency. Enabled, the binary refuses to start where
`libdav1d.so.7` is absent.

A related gap, worth knowing: **AV1 has no software decoder in this build.**
FFmpeg's native AV1 decoder is hardware-only, so AV1 files need VA-API. Measured
at 0 frames/s in software against 159 on the GPU. Pure-Rust AV1 via `rav1d`
would close this and is on the backlog.

## What was refused, and why

**`cpal`** for audio output. It link-binds `libasound.so.2`, which would cost
the binary its run-anywhere property for the sake of a device-opening
convenience. Replaced by `src/alsa.rs`, which `dlopen`s the nine symbols a
blocking writer needs.

**`libass`** for subtitles. It brings freetype, fontconfig and harfbuzz.
Replaced by embedded Noto Sans and the pure-Rust `ab_glyph`. The cost is real
and stated: no ASS positioning, no per-line styling, no text shaping — no
ligatures, no contextual forms, no bidirectional reordering. Latin subtitles
lose nothing visible; Arabic and Devanagari would.

**`nvidia-vaapi-driver`.** Exposes H.264/HEVC/AV1/VP9 through NVDEC but no JPEG,
and the discrete card had already lost the adapter benchmark to the integrated
one by 2.2x.

**`rsmpeg`** for FFmpeg bindings. Sits on `rusty_ffmpeg`, which stops at
FFmpeg 8.1.

**Re-encoding cached frames to HEVC** to save memory. This build has no encoders
at all (`--disable-encoders`), and more decisively, a lossily compressed frame
shows the compressor's work rather than the film's — which defeats the purpose
of a cache that exists so frames can be *inspected*.

## Machine learning

ML follows shape 2, and is worth spelling out because it looks like an
exception and is not.

- The **`ort` crate** is compiled in behind an `ml` feature, which is **on by
  default**. The flag exists because the crate tree is not free, **not**
  because the binary stops running anywhere — and that was checked rather than
  assumed before it was defaulted on: with `ml` compiled in and no runtime
  installed, `show` starts, plays, and says what is missing.
- **`libonnxruntime.so` is `dlopen`ed** via `ort`'s `load-dynamic` feature,
  exactly as libheif is. Absent, the plugins page says so and offers to fetch
  it; nothing else changes.
- **Model weights are downloaded on request only**, verified against a SHA-256
  compiled into the binary. `show` never executes a graph it fetched — only
  opaque blobs at pinned URLs.

- **Licences differ per model and are shown before the download starts.** The
  segmentation weights are AGPL-3.0, the runtime is MIT; nothing is fetched
  without the licence having been on screen first.

An `ml` build starts on a machine with no runtime and no models, and says so.
That is the test any future ML dependency has to pass.

**One hazard found by doing it.** A `Session` must never be destroyed. With a
`dlopen`ed ONNX Runtime, dropping one leaves the process to segfault inside the
library at exit — and the drop itself completes normally, so the crash arrives
later and looks unrelated. `vision::Detector` therefore holds its session in a
`ManuallyDrop` and leaks it. That is affordable because a detector is
long-lived, one per process; opening and dropping them in a loop would leak in
earnest.

## Adding one

Ask, in order:

1. Can it be pure Rust?
2. If not, can it be vendored and linked statically?
3. If not, can it be `dlopen`ed and the feature simply absent without it?
4. If not, is the part we need small enough to write?
5. If none of those, it needs a feature flag, off by default, documented here.

## Built in from source

libjpeg-turbo 3.1 is compiled from the source `turbojpeg-sys` bundles and
linked statically (`src/turbojpeg.rs`), so every JPEG gets its SIMD decoder
with no runtime dependency. Building needs `cmake` and `nasm`; `TURBOJPEG_STATIC`
in `.cargo/config.toml` keeps it from ever picking up the system's shared copy.
