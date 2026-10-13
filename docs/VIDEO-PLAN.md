# Playing video

A plan for mp4/mkv playback with subtitles, via FFmpeg 8.

The linking question is the one that gets asked first, but it is the smaller
half of the work. The larger half is that this program is currently built around
an assumption video breaks: that an image is decoded once and then sits still.

## 1. Which FFmpeg, and how it gets in

FFmpeg **9.0 "Lei"** (August 2026) is current, with **9.0.1** as the fix
release. This machine has **6.1.1** from Ubuntu 24.04, with development headers
present. Using 9 therefore means building it — the distribution will not have it
for a long while.

Note that `rsmpeg` currently pins itself to FFmpeg 8 (`0.18.0+ffmpeg.8.0`) and
`ffmpeg-next` 9.0 covers FFmpeg 4-7. Neither advertises 9.0 support yet, so the
first task is finding out whether either builds against it, and if not, whether
the gap is small enough to patch. That is a real risk to check before the
toolchain work, not after.

Three ways to attach it, and they are not equally good here.

| | binary | runtime deps | licence burden | build cost |
|---|---|---|---|---|
| **Static** | +20-35 MB | none | LGPL relink obligation | a day of toolchain work |
| **Dynamic** | small | 5 shared libs, version-locked | none | trivial |
| **`dlopen`** | small | optional, degrades gracefully | none | large: ~80 hand-written signatures |

**Decided: static, and that is the shipping form.** It matches the rule this
project has held to — the binary links libc and nothing else — and it is the
only way a viewer built here runs on a machine with a different FFmpeg.

Still worth starting dynamic against the system 6.1 for the first day or two.
It gets a frame on screen and proves the binding while the static build is
being sorted out, and swapping to static afterwards is a build-system change
rather than a code change. Not a different plan, just a different first step.

`dlopen` — which is how HEIC works here — is the wrong tool this time. libheif
needed eight functions; a video player needs demux, decode, seek, rescale and
resample, which is dozens of signatures hand-asserted against a C ABI that
changes between major versions. The safety argument that justified it for one
optional format does not survive being multiplied by ten.

### Licensing, which is the part that bites

FFmpeg is **LGPL-2.1+** provided `--enable-gpl` is left off. Decoding H.264,
HEVC, AV1, VP9, AAC and Opus needs no GPL component, so staying LGPL is
comfortable — but it must be checked, because a single GPL filter pulled in by
habit relicenses the whole binary.

Static linking LGPL code carries the relink obligation: recipients must be able
to relink against a modified FFmpeg. For an open-source project that is
satisfied by publishing the build. It is worth knowing before choosing static,
not after.

### Configure flags to aim for

    --disable-everything --disable-programs --disable-doc --disable-network
    --enable-decoder=h264,hevc,av1,vp9,mpeg4,aac,mp3,opus,flac,ac3
    --enable-demuxer=mov,matroska,avi
    --enable-parser=h264,hevc,av1,vp9,aac
    --enable-protocol=file
    --enable-vaapi          # this machine's iGPU decodes H.264/HEVC/AV1/VP9
    --enable-pic --enable-static --disable-shared

Everything off, then only what is needed back on: the difference between a
7 MB library and a 35 MB one.

## 2. What has to change in this program

This is the real work. Four assumptions break at once.

**Decoding is no longer one-shot.** Today a file is decoded once on a worker
thread and the result becomes a texture. Video needs a decode thread producing
frames continuously into a bounded queue, with backpressure, and a presentation
thread pulling them on a clock.

**The event loop must be paced.** It currently sleeps on `Wait` and redraws only
when something happens — which is why an idle viewer costs nothing. Video needs
a frame deadline. `ControlFlow::WaitUntil` already does this; the idle
thumbnail work has to yield to it.

**Audio is a separate problem FFmpeg does not solve — and it is required.** It
decodes audio; it does not play it. That needs `cpal` (portable, pulls ALSA on
Linux) plus a resampler and a ring buffer. And once there is audio there is an
audio clock, which is what video syncs to — not the other way round. This is
confirmed as in scope, so the sync work in step 4 is not optional and the design
should assume it from the start rather than bolt it on.

**Colour conversion should not go through swscale.** FFmpeg hands back YUV
planes. Converting to RGBA on the CPU costs a full pass per frame. Upload the
three planes as three textures and convert in the fragment shader instead: the
existing renderer already draws one textured quad, so this is a second pipeline
rather than a new architecture. `swscale` stays available for the awkward
formats.

## 3. Subtitles, honestly

Two kinds, and they are not the same job.

**Bitmap subtitles** (PGS from Blu-ray, VobSub from DVD) decode to images.
Those composite exactly like the existing overlays — nearly free.

**Text subtitles** (SRT, and ASS/SSA which is what mkv usually carries) need a
real text renderer. The bitmap font in this program is 5x7, uppercase-only, and
ASCII: fine for a timing readout, useless for dialogue in any language. That
means either

* `fontdue` or `ab_glyph` plus an embedded font — pure Rust, handles SRT well,
  but ASS positioning, styling and karaoke would have to be reimplemented; or
* **libass**, which does ASS properly and is what every other player uses, at
  the cost of another C dependency (and it pulls fontconfig, freetype, harfbuzz).

**Recommendation: SRT first with a pure-Rust renderer, ASS via libass later if
it is actually needed.** Most mkv files carry ASS, but most of those use it for
plain positioned dialogue that a simple renderer handles acceptably. Find out
before taking on freetype.

## 3a. Hardware decode: measured, and it matters

Tested by building FFmpeg 9.0.1 and decoding 300 frames of 3840x1920 HEVC
(real 360-degree footage from the T5 drive):

| path | CPU consumed | cores busy | wall |
|---|---|---|---|
| software, all threads | 25.7 s | 920% | 2.82 s |
| software, 4 threads | 21.5 s | 363% | 5.96 s |
| **VA-API on the AMD iGPU** | **1.6 s** | **58%** | 2.76 s |

Software decode is fast enough — twice realtime — but it saturates nine cores
to do it. Hardware decode does the same work in the same wall time for a
sixteenth of the CPU. On a machine that is also running an editor, a browser and
this program's own background thumbnailing, that is the whole argument.

**So VA-API is in by default**, and the cost is three shared libraries:

    libva.so.2   libva-drm.so.2   libdrm.so.2

Ubuntu ships no static libva, so `--enable-vaapi` links them dynamically and the
libc-only property is lost. The fix is to build libva and libdrm from source as
well: both are MIT-licensed and libva itself needs only libc, so there is no
cascade. Meson takes `-Ddefault_library=static`. Needs verifying, but it is a
build-script problem rather than a design one.

Note that libva `dlopen`s the system's VA driver (`radeonsi_drv_video.so`)
whatever we do, so hardware decode always depends on something outside the
binary — it simply falls back to software rather than failing to start, which is
how Vulkan already behaves here.

**Decided: hardware decode, and no further deliberation about it.** The
remaining question is not whether but how frames reach the screen. Decoding
happens on the AMD iGPU while the display is on the NVIDIA card, so either

* **present from the iGPU too** — it already won the adapter benchmark by
  2.2x, so the decoder and the display living on the same device is the
  straightforward answer and avoids the crossing entirely; or
* **export the decoded surface as a DMA-BUF** and import it into the display
  GPU, which is what a cross-vendor path would need.

The first is preferable and is what to build. It is an implementation choice to
make while writing the decode path, not a decision to defer.

## 4. Order of work

1. **Demux and decode one frame.** `rsmpeg` (targets FFmpeg 8) or `ffmpeg-next`
   (9.0, FFmpeg 4-7). Open an mp4, pull the first video frame, show it with the
   existing image path. Proves the binding and the build.
2. **YUV shader.** Three-plane upload, colour convert on the GPU. Still a single
   frame.
3. **Playback without audio.** Decode thread, frame queue, paced redraw,
   play/pause, seek. Video is watchable at this point.
4. **Audio and sync.** `cpal` output, audio clock, video synced to it. This is
   where a player stops feeling broken.
5. **Subtitles.** Bitmap first, then SRT.
6. **Hardware decode.** VA-API through FFmpeg — sixteen times less CPU, see
   above. Decided, not optional. Build it in from the point the decode path
   exists rather than retrofitting it, since it changes where frames live.
7. **Static FFmpeg 8.** Build script, minimal configure, licence check.

Steps 1-3 give something watchable. Step 4 is what makes it a player.

## 5. Decisions needed before starting

* ~~Static or dynamic?~~ **Static.** Accepts the LGPL relink obligation and the
  build complexity.
* ~~Does audio have to work?~~ **Yes.** Step 4 is not optional.
* ~~Hardware or software decode?~~ **Hardware.** VA-API, with libva and libdrm
  built statically alongside FFmpeg so the binary stays self-contained.
* ~~Does FFmpeg 9 have working Rust bindings?~~ **Yes, verified.**
  `ffmpeg-next` 9.0 builds and runs against a static FFmpeg 9.0.1 with no
  shared FFmpeg libraries: h264, hevc, av1, vp9, aac and opus decoders all
  resolve, 7.5 MB binary. `rsmpeg` is out — it sits on `rusty_ffmpeg`, which
  stops at FFmpeg 8.1.
* **How much does ASS matter?** The only decision still open. It decides
  whether libass — and freetype, fontconfig and harfbuzz with it — comes into
  the tree, which would be the largest dependency this project has taken on.

## 6. What this costs

Video roughly doubles the program. The image viewer is ~6,000 lines; a
competent player with sync, seeking and subtitles is comparable. It is worth
being deliberate about whether this is the same program or a sibling that
shares the decoding and rendering code.
