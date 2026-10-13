# Backlog

Two lists. **Deferred** is scope deliberately parked. **Measured** is work this
project's own profiling has already justified, roughly in payoff order.

## Deferred (KIV)

| Item | Note |
|---|---|
| ~~Video playback~~ | **Built.** 25 codecs, hardware decode, YUV shader, frame-rate chart, sound, seeking, subtitles. See `docs/VIDEO-PLAN.md`. |
| Plugin: upscaler | Post-decode stage. Wants the pipeline to have a stable "decoded frame" hand-off point first. |
| Plugin: local-LLM tagging | Analyse image contents, attach tags. Entirely off the display path. |
| Basic edits: crop, rotate, convert, rename | Turns a viewer into an editor: needs undo, a save path, and format re-encoding. |
| Batch mode | Depends on the edit operations existing first. |
| GeoTIFF georeferencing + OpenStreetMap overlay | The map TIFFs carry ModelPixelScale/ModelTiepoint/GeoKeyDirectory tags. Explicitly an addon, not core. |
| Backwards compatibility with older Linux Mint | Mint 22 is glibc 2.39, Mint 21 is 2.35, Mint 20 is 2.31. Measured conclusion: **costs nothing in speed** — every hot path is pure Rust with `cpuid`-dispatched SIMD, and glibc only supplies memcpy/malloc. Build in an old-glibc container or use `cargo-zigbuild --target x86_64-unknown-linux-gnu.2.31`. Avoid static musl unless mimalloc comes with it; musl's allocator is slow under threads. The real risk is old **Mesa**, not glibc — and the software fallback already covers "no Vulkan" (measured: 10.6 ms). |

**Explicitly not wanted:** a reduced-quality stand-in shown *in place of* the
real image — no embedded-preview first paint, no thumbnail substituted for the
picture you asked for. The mosaic view is not that: it is a deliberate
multi-image mode you switch into, where small is the point.

## Ruled out

**Embedded EXIF/preview-image first paint.** Ruled out on two counts. It shows a
low-resolution stand-in rather than the real image, which is not what this
viewer is for. And it would not have worked anyway: exiftool confirms the
336 MB panorama, the DNG and the iPhone HEIC all carry **no** embedded preview,
so the files that motivated it are exactly the ones it could not have helped.

**`nvidia-vaapi-driver` / NVIDIA hardware decode.** Decided against, not merely
deferred. It exposes H.264/HEVC/AV1/VP9 through NVDEC but no JPEG — NVIDIA's
JPEG engine is not reachable through VA-API, and GeForce parts lack the nvJPEG
hardware decoder that A100-class cards have. So hardware JPEG is AMD-only on
this machine regardless. The only thing it would have bought is a cross-vendor
AVIF/HEIC decode comparison, against a discrete card that already lost the
adapter benchmark to the integrated Radeon by 2.2x.

## Next, in the order I would take them

1. ~~**Audio.**~~ **Built.** `src/alsa.rs` loads libasound at runtime and
   `src/audio.rs` holds the ring buffer, writer thread and clock; the decode
   thread resamples to 48 kHz stereo through swresample. Video is presented
   against the audio clock whenever a file has sound, and falls back to the
   wall clock when it does not. The wheel is the volume knob during playback.

   `cpal` was the obvious choice and was rejected: it links `libasound.so.2`
   at build time, which would have cost the binary its "runs anywhere"
   property for the sake of a device-opening convenience. Verified with
   `show --audio FILE`, which plays at zero gain and reports the clock —
   3.05 s of audio in 3.01 s of wall clock.

   The clock is re-anchored on a seek, and the device flushed. Still open:
   multichannel sources are downmixed to stereo by swresample rather than by a
   mixing matrix we choose.
2. ~~**Seeking.**~~ **Built.** Plain arrows step keyframe to keyframe, ctrl
   moves 10 s and shift 30 s; both ends are walls rather than doors.

   A keyframe step is exact and immediate — it lands where decoding can start,
   so nothing has to be decoded and discarded to get there — but *how far* it
   moves is the file's business, not ours. Measured: 0.56 s per press on one
   encode and 3.3 s on another. That is inherent, and the reason the
   fixed-distance seeks are kept on the modifiers. Both ends of a file are
   walls: there is no keyframe before the first or after the last, and staying
   put on reaching one is the right answer rather than a fault. The decode thread takes seek
   requests on a channel and coalesces a burst into its last request, so
   holding a key down costs one seek rather than twenty. Frames carry a
   generation stamp and stale ones are dropped, which is what stops the old
   position appearing after the new one. The seek bar shows on pointer
   movement and fades after 2.5 s.
3. **Zero-copy hardware frames.** Hardware decode currently reads frames back
   from the GPU, which costs about what the decode saved — measured at 24% CPU
   against software's 21% on 4K HEVC. It only pays once the VA surface is
   exported as a DMA-BUF and imported straight into Vulkan, or once presenting
   happens on the same device that decoded.
4. ~~**Subtitles.**~~ **Built**, in `src/subtitles.rs`. Both families: text
   (SRT, ASS/SSA, WebVTT, mov_text) via libavcodec, which hands every one of
   them over as ASS dialogue so there is a single parser; and bitmap (PGS,
   VobSub, DVB, XSUB) composited from their palettised rectangles. Files
   beside the video are found too, including the `film.en.srt` convention and
   a `Subs/` directory, and they outrank an embedded stream. `V` cycles
   tracks and off.

   **libass was rejected**, not deferred: it would bring freetype, fontconfig
   and harfbuzz to a binary that has gone to some trouble to need no system
   libraries. The cost is real and worth stating: ASS positioning, per-line
   styling, karaoke and typesetting are dropped, and what survives is the
   dialogue, its timing and its line breaks.

   Captions are set in **Noto Sans, embedded** (`assets/`, OFL 1.1) and
   rasterised by `ab_glyph`, which is pure Rust — see `src/typeface.rs`. That
   is deliberately the same face libass ends up using on this desktop, where
   it asks fontconfig for `sans-serif` and gets Noto; embedding it keeps the
   result without the three libraries. The 5x7 bitmap font stays for the
   interface, where it belongs, and keeps the lowercase and accent folding it
   grew for this.

   What that still lacks against libass is text *shaping*: no ligatures, no
   contextual forms, no bidirectional reordering. Latin subtitles lose nothing
   visible; Arabic and Devanagari would, and want a shaper rather than a
   better font.

   Characters the face cannot draw are stripped rather than shown as empty
   boxes — Noto Sans has no musical notes, and subtitle files put those around
   song lyrics, so the boxes turned up at the start and end of every sung
   line. Captions sit in the letterbox band below the picture when there is
   one deep enough, where they cover nothing.

   Still open: no styling, so a sign and a line of dialogue look alike; and
   cues are retired 30 s behind the play position, which becomes a cache bound
   to revisit now that seeking exists.

## Why playback is three threads

Reading, decoding pictures and decoding sound block on different things, and a
single thread doing all three stops reading whenever any one of them is full.
That is not a theoretical worry: a container may place a stretch of video
packets well ahead of the audio that belongs with them, and reading forward is
the only way to reach that audio. With one thread, a full frame queue — eight
raw pictures — stopped the reading, the sound ran dry, and the timeline video
is shown against jerked every time it did.

Measured on a 2.6 GB webm that stuttered visibly:

| | underruns | spread | hitches |
|---|---|---|---|
| one thread | 103 of 472 blocks (21.8%) | 22.6 ms (68% of a frame) | 69 |
| demux + video + audio | 0 of 279 (0.0%) | 2.1 ms (6%) | 0 |

The fix is what is buffered, not how much. The demuxer now blocks on *packet*
queues, and packets are compressed — a few kilobytes each, so `PACKET_QUEUE`
holds many seconds of a file for a handful of megabytes, where the same depth
in decoded frames would be gigabytes.

## The window draws its own frame

`with_decorations(false)`, and a title bar and two-pixel border of our own in
one dark grey. Everything the desktop's decorations did has to be provided
instead: dragging the bar moves the window (`drag_window`), the edges resize it
(`drag_resize_window`, with a grab area three times the drawn border because a
two-pixel target is a poor thing to have to hit), and the mark at the right
closes it.

The picture is inset *inside* the frame rather than drawn under it, and every
badge that hangs off an edge hangs off the inside — a readout half-hidden by
our own title bar looks like a fault, because it is one. Fullscreen draws no
frame at all: there is no window edge to mark and no room to spare.

## Frame-by-frame stepping, and why there is no background decoder

`,` and `.` move one frame and stop playback. Forward is easy — the next frame
is already decoded and waiting. Backwards cannot be decoded directly, and the
usual answer is to seek to the previous keyframe and decode forward
discarding, which on a file with ten-second keyframes means decoding 250
frames to move back one.

The window is three seconds each way. Backwards, `src/framecache.rs` keeps the
frames the player has **already shown**. Forwards, the decoder simply waits three
seconds further on than it used to — a deeper queue, not a background task, and
bounded by a gigabyte so a 4K film gets fewer seconds rather than more memory.

Playing forwards fills the backward window for nothing, since those frames were
just shown. A **jump** empties it, and that is the one case worth a second
decoder: `BackFill` covers the three seconds behind while the main decoder
covers the three ahead. A decoder cannot run backwards, so it seeks to the
keyframe before and decodes forwards — the same direction of work on a
different range, which is exactly why splitting it in two halves the wait.
Measured: **385 ms** to fill three seconds of 1080p h264, **1.96 s** for 4K AV1.

**AV1 has no software decoder in this build** — 0 frames/s against 159 on the
GPU, because FFmpeg's native AV1 decoder is hardware-only and libdav1d is not
vendored. The fill prefers software so the GPU stays free for drawing, and
retries on the GPU when software produces nothing, which is the only way an
AV1 file gets filled at all. Worth knowing more widely: on a machine without
VA-API AV1, these files will not play.

Stepping shows the frame number, padded to the width of the last frame in the
film so the readout does not shift while counting, and the moment to the
millisecond — at twenty-five frames a second, a timecode in whole seconds
cannot tell two dozen frames apart.

`src/framecache.rs` keeps the frames the player has **already shown**. Its
virtue is what it does not cost: no second decoder, no extra CPU, nothing
asked of the GPU that is busy drawing the film. The frames were decoded
anyway; only the memory is new. Beyond the cache it does the expensive thing properly: seek to the keyframe
before, decode forward discarding, keep the last frame short of the target.
That is bounded by a 400 ms budget so the window keeps answering, and landing
early beats not responding.

It would have been cheaper to fall back to the previous *keyframe*, and that
is what it did at first — but a key that means "one frame" near the play head
and "somewhere between half a second and ten seconds" away from it is not one
key, it is two wearing the same label. Measured cost of doing it properly:
40 ms on a 0.56 s group, 79 ms on a 3.3 s one.

A background pre-decoder was measured before being rejected as unnecessary:

| filler | rate | effect on playback |
|---|---|---|
| software | 232 fps (9.3x real time), 687 MB/s | spread 1.3 -> 1.4 ms, 0 hitches |
| hardware | 500 fps (20x real time), 1.4 GB/s | spread 1.4 ms, 0 hitches |

Neither disturbs the film on this machine, so the choice was free — and the
rear-view cache made both unnecessary, since it fills from frames that already
exist. `show --cache-bench FILE` reproduces those numbers.

**Compressing the cache was considered and rejected.** Re-encoding to HEVC
would need an encoder, and this build has none (`--disable-encoders`); adding
one brings a GPL dependency and a per-frame cost on the machine playing the
film. More decisively, the feature exists so a frame can be *inspected*, and a
lossily compressed frame shows the compressor's work rather than the film's.
Memory is bounded by a span instead — 10 s each way, roughly 1.5 GB at 1080p
— with a byte budget behind it for larger formats.

## Drawing what was found: a pipeline, not an overlay

Boxes are drawn by a third render pipeline (`src/mask.wgsl`), beside the image
and YUV ones, from an **`R8Uint` label grid** — one byte per cell holding an
instance id, zero meaning nothing. 480x270 is 130 KB against 8.3 MB for a
screen-sized RGBA overlay, and it is written once per *result* rather than once
per frame.

That is the whole reason it is not an overlay. `sync_overlays` re-uploads
**every** overlay whenever any one of them changes, so a per-frame mask would
re-upload the title bar, the borders, the badges and the caption 25 times a
second alongside itself.

It is drawn with the **same rect as the video quad**, so it letterboxes, pans
and zooms with the picture while knowing nothing about any of that. The texture
is `R8Uint` rather than `R8Unorm` deliberately: an id is not a brightness, and
a filtered sample between two ids would mean a third thing.

The shader finds edges by comparing a cell with its four neighbours, so what it
wants is the *region*, not a picture of a border — the grid is filled, and the
outline is the shader's business. The inside is left at 16% rather than filled:
a film is still being watched, and colouring in every found thing hides more
than it explains.

The gate held with drawing added: spread 1.1 ms without, 1.4 ms with, no
hitches either way, ~16 ms an inference.

## Detection on video, and the gate it had to pass

`F7` turns it on. A bar along the top names what is in view, each tag in the
colour that thing's box will be drawn in — so the bar **is** the legend, and
"which colour is which" needs no separate key.

The scheduling is one thread, one session, and a queue of **one**. Frames are
*offered*: if the worker is busy the frame is dropped and the next one taken.
That is the whole policy and it is the right one — this is a readout, and a
readout one frame behind is invisible, while a decoder made to wait for a
readout is a stutter anybody can see. Looking happens five times a second,
because what is on screen changes with the shot, not the frame.

The gate, measured twice on a 25 fps film:

| | spread | hitches |
|---|---|---|
| nothing looking | 0.7 – 1.1 ms | 0 |
| detection running | 1.4 ms | 0 |

26 frames examined per six seconds, none skipped, ~20 ms an inference. The
tally is rebuilt only when it would *read* differently, so the bar rides the
ordinary overlay path where re-uploading the whole thing is correct and cheap.

The bar **steals** space through `image_viewport` rather than floating over the
picture, and every badge that hangs off the top is offset below it — the
codebase's own rule that a readout half-hidden by our own furniture looks like
a fault.

## Outlines instead of boxes

Segmentation is a second model in the same slot: install it from `F8` and the
watcher opens that instead, `look()` instead of `run()`, outlines instead of
rectangles. Nothing else in the app asks which is running — a plain detection
comes back as an outline that fills its own box, so there is one shape
downstream rather than two.

Measured on a 919x696 photograph, three people: **33 ms an inference**
against detection's 23, and the outlines cover 54–73% of their boxes — a
person-shaped fraction rather than a filled rectangle, which is how the
numbers say the prototypes decoded into something real.

The gate, on the same 25 fps film as before:

| | spread | hitches |
|---|---|---|
| nothing looking | 1.3 ms | 0 |
| segmentation running | 1.2 ms | 0 |

27 frames examined, none skipped, 23 ms an inference. The pacing line now says
**which** model it measured, because the choice depends on what is installed
and a measurement that does not name its subject is worth little.

The whole video corpus was re-swept afterwards — **213 files, zero uneven
pacing, zero underruns, worst spread 2.4 ms** — because a change to what runs
beside the decoder is exactly the kind that shows up on one file in a hundred.

**The box units are not stated anywhere in the file.** This export gives
`cx,cy,w,h` as fractions of the input; the same architecture elsewhere gives
them in model pixels. Read one as the other and every person lands in the
top-left corner — which is exactly what happened, and it looked like a broken
letterbox rather than a units mistake. It is now measured instead of assumed:
no real detection in a 640-pixel square has a side under two pixels, and every
fraction is at most one, so the maximum over the four box fields decides. A
test pins that both conventions produce the same box.

**Prototypes are mixed only inside each box.** A mask is 32 coefficients over
32 basis images; doing that for every cell of a 160x160 grid, for every
instance, is most of a frame budget. Doing it inside the box is the same answer
for a fraction of the work, because everything outside the box is discarded.

**A mask cell is bigger than a label cell**, so writing one label per mask cell
leaves gaps, and the shader's edge detector reads gaps as more edges — a dotted
outline that looks like a rendering fault. Each mask cell fills its whole
footprint instead, pinned by a test that measures the worst gap.

## Following things between inferences

`src/ml/track.rs` is ByteTrack without the Kalman filter: greedy IoU matching,
same class only, three frames of patience, ids never reused. No model, no
runtime, no feature gate — which is why it is the most thoroughly tested part
of the ML work.

It earns its place twice. A thing keeps its **identity** across inferences,
which is what click-to-follow will need. And a detection is drawn only from its
**second** sighting: a model fires at a shadow now and then, and at five
inferences a second the cost of waiting is a fifth of a second while the gain
is that boxes stop flickering.

Colour stays **by class**, not by track. The tag bar is a legend, and a legend
whose colours change as things come and go is not one.

**Verified without a window**, because the desktop is in use: `--detect FILE
--dump-ui DIR` writes what the mask shader would paint — nothing at zero, an
opaque edge where a neighbour differs, a faint fill inside — over the picture
it came from. Sampling it against the original showed the people's pixels moved
and the sky's did not, which is the only way to tell a correct mask from an
inverted one by looking.

## Running on the card

`F8` has a row for it: **NVIDIA ACCELERATION**, 986 MB, off until installed.
Installing it is how you ask for it — there is no second switch, because a
card that is present and unused is a card nobody asked to leave idle.

Measured on this machine, an RTX 4070 SUPER, warm:

| | processor | card |
|---|---|---|
| yolo26n-seg | 32 ms | **7 ms** |
| yolo26m-seg (largest) | 125 ms | **12 ms** |
| MobileSAM encode (a click) | 250 ms | **34 ms** |
| sharpening one box | 20 ms | **6 ms** |

The first inference on the card is ~320 ms while it decides how to do the
work; every one after it is the number above. And it costs playback nothing
measurable, which was the open question — the same card is drawing the film.

### What is actually downloaded

Two archives, both pinned by digest like everything else: the **CUDA build of
ONNX Runtime** (the ordinary one cannot be told to use a card it was not built
for) and **cuDNN**. Nothing else — CUDA itself is expected to be on the machine
already, because this pins what is specific to running these models, not a
driver stack.

An archive can now yield several files rather than one, which cuDNN needs: it
arrives as ten libraries that only work as a set. Fetching one archive and
taking them all is the difference between one download and ten.

### Three ways this silently did nothing

Worth writing down, because each looked like "the GPU just is not faster":

- **The provider was never registered.** Registering an execution provider
  warns and carries on by default. `error_on_failure` makes it say so.
- **`installed` looked for the wrong name.** An archive that unpacks into
  several files has no file of its own; the check was for the label. So the
  card support reported itself missing while sitting on disk.
- **cuDNN was short a piece.** It is a dispatcher that opens the other nine by
  name at run time, so a missing one is not a link error — it is a sentence
  about a library it could not find, arriving in the middle of an inference.

## Sharpening what was found

The detector says *what* and roughly *where*. The cut-out model says exactly
where. Turning `SHARPEN OUTLINES` on at `F8` puts them in series: every box the
detector finds is handed to the cut-out model as a prompt, and what comes back
is a 1024-resolution outline with the detector's own class still attached — so
a person stays a person and keeps its colour.

A box is given to that model as two points labelled 2 and 3, which is its own
convention for a corner pair, against 1 for "this is the thing". Measured on a
photograph of three people: **61 ms for all three, 20 ms each**, on top of one
encode of the picture. The encode is the expensive half and is shared by every
box in the frame, which is what makes doing this for everything affordable at
all.

It costs the film nothing measurable: sharpening five outlines leaves frame
pacing at 1.4 ms of spread, the same as doing nothing. The gate measures it.

What it looks like: the small model bleeds around a person and traces their
outline in steps about seven pixels wide; sharpened, the same person is
outlined around their hair, between their shoulder and the person behind, and
around the badge hanging from their lanyard.

## How often to look, and why a bigger model does not sharpen the edge

The rate is a slider on `F8`, one to ten times a second, default one. It
changes while looking rather than on the next film, and it is on that page
rather than the settings one because what it costs depends entirely on which
model above it is in use: ten times a second with the largest is a second and
a quarter of work per second of film.

**A larger model does not give a finer outline, and it was reasonable to
expect that it would.** All three emit their masks as 32 prototypes at
**160x160 for the whole frame** — the same grid, whichever size. What grows
with the model is what it finds and where it says the thing is, not the
resolution of the saying. So the small model bleeds around a person and the
large one traces the arm, but neither has a sharper *edge*: both are drawn
from cells about seven pixels across on a 1080p frame.

Pixel-accurate edges come from a model that looks at one thing at a time.
That is what `CTRL` click already does — MobileSAM at 1024 across rather than
160 — and it is the reason both exist.

**The running model is now in the F4 readout**, with the rate and the time of
the last look. "I chose a different model and see no difference" should be
answerable by looking rather than by guessing whether the choice took.

## What is in the card's memory, and who has it

A twelve-gigabyte card is mostly not ours. Before `show` starts, X and the
compositor and a browser are already holding about a gigabyte of it, and when a
model fails to load that is usually the reason rather than anything about the
model. Nothing in the program could see this: `wgpu` knows what `wgpu`
allocated, the inference runtime knows what it allocated, and neither can see
the other or anybody else. Only the driver sees all of it.

So there is now a card-memory readout in the bottom right, under F4 and its own
setting — what is free, what each graph took when it loaded, what this process
holds, then every other process biggest first. It is asked once a second rather
than per frame: the figures move because *another* program allocated, which has
nothing to do with our frame rate.

Three things that cost time and are worth not rediscovering:

* **`nvmlDeviceGetMemoryInfo` returns zeroes on a current driver** — and
  returns *success* while doing it. The `_v2` form is the one that answers, and
  it wants its own size and version written into the struct before the call.
* **Asking only for compute processes explains nothing.** That is what
  `nvidia-smi --query-compute-apps` does, and on an idle desktop it returns an
  empty list. X, the compositor and the browser are *graphics* processes, and
  they are the ones holding the memory somebody is looking for.
* **Absent from the driver's list means nothing held, not "unknown".** Read as
  unknown, the first graph of the session — the biggest, the one somebody
  wanted to see — is charged to nothing and never appears. That is exactly
  where we stand before the first model loads.

Our own process only appears in the driver's list once inference is on. Playing
a film, neither this readout nor `nvidia-smi` attributes the video textures to
us, and the difference shows up in the gap between the total and the rows. The
first row states used against total for that reason: the rows beneath account
for most of it and never all of it.

The charge per graph is a difference, measured either side of the load, because
nothing reports it directly — the runtime does not say what a session cost. It
is taken in one wrapper that all three loading sites share, which they did not
before: the detector and the cut-out built their sessions directly and only SAM
3 went through `runtime::commit`, so the first version of this charged SAM 3
and silently ignored every other model.

## Bugs found by using it

**A page saved the settings, so the tests overwrote them.** `Panel::adjust`
and the new rate slider both called `config.save()`, and their tests pass a
real `Config` — so running `cargo test` wrote the settings file, replacing
whatever the person using this program had chosen. Pages change settings; only
the app writes them. There is now a test that reads the page sources and fails
if any of them saves.

This is the second time a test has written over real data: `store` once removed
models from the actual data directory. Both had the same shape — a unit that
knows how to persist itself, and a test that exercises it for its return
value.

**Clicking the seek bar landed at the wrong place.** A seek can only start at a
keyframe, so the click seeked to the keyframe *before* it and showed the frames
from there — and the test films in hand have exactly one keyframe, the first,
so every click landed at 0:00. The frames between the keyframe and the click
are now decoded past rather than shown. It applies to the settled seek only:
during a drag the keyframes are the preview, and being exact on every mouse
move would make dragging crawl.

**Frame stepping forwards stopped after a few hundred frames.** A paused film
stops draining its sound, so the audio queue fills, so the demuxer blocks
putting a packet into it — and no more video is read either. Stepping worked
until the decoded frames ran out and then did nothing at all. Two answers: the
window *ahead* is now filled while paused, by the same spare decoder that fills
the window behind; and a step that finds nothing waiting seeks and decodes to
the next frame itself, which is what the backward step has always done. The
probe reproduces the stall on purpose and checks the recovery moves exactly
one frame.

**A measurement that was not comparable.** The new gate for following a cut-out
reported the *maximum* pacing error where the other two report the standard
deviation, which made it look ten times worse than detection — 13.4 ms against
1.4. Two hours of hunting a load that was not there. Following a cut-out costs
the film nothing measurable, and the harness now reports the same statistic as
its neighbours.

**Choosing a model was not a choice.** The largest installed one won, so the
only way to use a different one was to uninstall the better. `ENTER` on an
installed model now makes it the one in use, the row says `IN USE`, and the
choice is remembered beside the models rather than in the settings — it is a
fact about what is installed here. `DEL` is labelled `UNINSTALL`, because that
is what it does. Headless: `show --plugins use segment`.

**The setup menu grew on the first keypress, and my first fix did not work.**
`invalidate` cleared the window size the page had been laid out for, so every
change of rows looked like a resize and started the fit over. The test I wrote
restored that field by hand, which is exactly the thing the code was failing to
keep — so it passed while the menu still grew. The size is now settled once per
window size and left alone until the window itself changes, and the test drives
the real flow.

**A windowed film ignored the keyboard until it was clicked, and asking for
focus did not fix it.** Images open fullscreen and every window manager gives a
fullscreen window focus; a film opens windowed and undecorated, and that one is
not always granted it. So the window asked — from `resumed`, the moment it was
created, where X11 throws the request away. winit guards `focus_window` behind
its own `Visibility::Yes`, which is only reached when the server's
`VisibilityNotify` arrives, and that cannot happen until the event loop starts
pumping, which is after `resumed` returns. The request was a no-op everywhere,
not just where focus had been given, and winit reports that refusal by doing
nothing at all. The window now *owes* a focus request rather than making one,
and pays it from the redraw path until the keyboard arrives, stopping the
moment `has_focus` is true so a window deliberately clicked away from stays
away.

The first version of that counted frames, and passed — under a debug build,
which is slow enough that ten frames outlast the round trip to the server. A
release build draws ten frames in a few milliseconds, spent every attempt
before the server had answered, and gave up before it could have worked. What
is being waited for is a round trip, so the budget is now time: a second and a
half, which no attempt has ever needed more than a fraction of. **A retry
budget counted in frames is a budget that means something different in every
build.**

Measured rather than reasoned about, and measured from the window's own
`_NET_WM_STATE` rather than from guessing which process owns the active window
— the first harness compared pids, found the IDE's window was called `show`
too, and reported failures that were its own: the installed build, 0 of 5
launches focused; the fixed build, 5 of 5, every one inside 250 ms.

**A page resized itself when its rows changed.** Pressing enter in the setup
menu made the whole page grow: activating a row adds a status line and can
make another row shorter, the page then fitted itself to the new content, and
a shorter page allows a larger text size. The automatic fit now only ever
steps *down*, and only when the content genuinely does not fit; a new window
height starts it over. The same applies sideways — the plugins page changed
width as the highlight moved down it, because the widest row changed. A page
now keeps the widest it has needed while it is open.

**Half a rule is not a rule: the page still grew sideways.** A remembered
widest stops a page narrowing and leaves it entirely free to widen, which is
the direction the setup menu actually moved: pressing enter on `INSTALL` adds a
status line carrying a full path, longer than anything else on the page, and
the panel jumped wider on the keypress. The settled width is now a ceiling as
well as a floor, and a row that says more than the page has room for is elided
into it — the renderer already had that ability, it was simply never given a
limit to elide against. Feeding a width back in as a limit turned out to need
the character arithmetic to round trip, which it did not: the last character on
a line carries no trailing spacing, so a budget counted as `chars * advance` came
out one character short of the width that produced it, and the widest row would
have picked up an ellipsis it did not have on a page that had not changed. The
test for that compares the pixels of two layouts of identical rows.

Both are the same lesson: **layout that follows content is layout that moves
under the reader.** It is right for a page being built and wrong for a page
being read.

**Four clock tests asserted to a microsecond against a clock that carries
milliseconds.** `Clock::position` deliberately adds the wall time since the
last device update, bounded at one block, so the picture keeps moving between
updates. The tests demanded the exact arithmetic to within 1e-6 s, which meant
they failed whenever the machine took longer than a microsecond to reach the
next line — so `cargo test` was permanently, meaninglessly red, and a real
failure would have had nowhere to show. The tolerance is now one block, the
slack the clock is documented to have, and asymmetric: the carry is only ever
*added*, so a clock running behind its anchor still fails. Checked both ways —
25 runs idle and 15 under eight spinning cores, all clean; and with the latency
subtraction flipped to an addition, the one test that pins it still fails.
**A tolerance is a statement about the code's slack, not a knob to turn until
the test passes.**

**Every person on screen was the same colour, and seven of the fifteen colours
were never used.** Mask ids were palette slots taken from the *class*, so two
people were both drawn in the person colour and telling them apart meant
clicking one. They are now taken from the tracker's id, which is the thing that
already knows one person from another — and taken from the id rather than from
position in the list, so a colour stays with a thing as it moves instead of
flickering whenever the model returns its answers in a different order.

The palette had eight colours and the slots ran to fifteen, so `index %
PALETTE.len()` quietly made slots nine to fifteen repeats of one to seven: two
objects could come out identical while seven distinct colours sat unused. It
now has fifteen, which is what the shader can tell apart — sixteen entries, one
of them background.

The bar had to follow. It is the legend — a tag is drawn in the colour that
thing's mask is — so per-object colours made a per-kind bar a lie: one PERSON x3
tag in one colour, naming three things drawn in three. It lists things now, not
kinds, numbered within each kind and only when there is more than one of them.
Both sides read the colour from `labels::colour_of`, so they cannot drift
apart; the cut-out, which has no tracker id, gets a reserved slot rather than
falling on slot one and clashing with whatever the tracker numbered fifteenth.

**Two constants decided what counted as a thing, and neither could be
reached.** `MIN_CONFIDENCE` and `DUPLICATE` were consts: everything far off or
half-hidden was dropped, and the only way to see it was to edit and rebuild.
The second was the worse of the two — a crowd looks exactly like a model
repeating itself, so people standing close were discarded as duplicates of each
other. Both are sliders on the plugins page now, read by the worker once a
frame so they take effect in the film being watched rather than the next one.
**A number that decides what the user sees is a setting, whether or not it was
written as one.**

**The CLIP tokenizer, and why three examples are not a test.** SAM 3's language
encoder takes token ids, not text, so text prompting needs CLIP's byte-level
BPE written out here — a regex engine and a JSON crate to read two files once
was not the trade. The first version passed its unit tests, matched the
canonical `a photo of a cat` ids exactly, and was wrong.

What found it was a differential harness: 3000 lines of random Unicode through
this and through a reference tokenizer, compared id by id. 393 disagreed. The
cause was one word — `char::is_alphabetic` is the Unicode *Alphabetic*
property, and CLIP's pattern asks for `\p{L}`. Alphabetic additionally takes in
every combining mark carrying Other_Alphabetic, letter-numbers like Ⅱ, and some
enclosed symbols: 1694 codepoints that split a prompt differently than the
model was trained to. They are a table now, generated by walking all of Unicode
and comparing the property against the general category.

That left 54. Thirty-eight were NFC, which the reference normalises with and
this does not — recorded as a known gap with a test pinning it, because closing
it means decomposition tables, combining classes, composition exclusions and
Hangul, and every realistic prompt is composed already. The last sixteen are
thirteen codepoints unassigned in Unicode 15 and assigned as letters in the
newer tables Rust carries: two Unicode versions disagreeing, not two
implementations.

**A tokenizer that is merely close does not degrade gracefully.** It hands the
model a different sentence from the one that was typed, silently, and the model
answers it — so the only way to know is to run it against another
implementation and compare every id, and the corpus has to be adversarial
because the realistic one agreed 462 out of 462 while a bug was sitting in it.

**SAM 3 found nothing, six times over, and every explanation was wrong.** The
three graphs loaded, the token ids were accepted, masks came back at the source
resolution — and the decoder returned an empty array for every prompt. Guessed
at it for a while: letterbox against squash, six words against six other words,
the language mask as given, inverted, all true, all false. Every one returned
nothing, which should have been the clue much earlier — a knob that changes
nothing is not the knob.

Two things were wrong, and neither was guessable. The prompt was padded with
CLIP's end token, which is what CLIP does; SAM 3 derives its padding mask from
the ids and reads a tail of end tokens as *content*, so a three-word prompt
arrived as thirty-two words. And the empty geometric slot was passed with
`box_masks` false, which does not mean "no box" — it means "this box is real",
so every query was anchored to a box at the origin. True is the empty one.

Both came out of the exporter's own inference code in twenty minutes, after an
hour of guessing. **When a model answers nothing to everything, stop varying
the inputs and go and read the caller.** The visible measure of the difference:
zero detections in every configuration before, and after, four people in
`bus.jpg` at 0.94 and above — which is the exhaustive-instance behaviour the
duplicate-overlap slider was invented to work around.

Two things fell out of reading it. The vocabulary is *derived* from the merges
list rather than shipped beside it, so `vocab.json` and the JSON parser written
to read it both went — checked entry by entry against the published table
first, all 49408. And the decoder's external weights are marked
`referenced_by_graph: false`; it loads without them, so the catalogue takes
five members out of the archive instead of six and nobody downloads the other
116 MB.

**Outlines had never been drawn on a still picture, and nobody had noticed.**
Wiring SAM 3 into the viewer produced no outlines. The grid was right — 111,494
cells set, twelve per cent of the frame, four people — the answer arrived, the
upload ran. The thing that found it was refusing to believe it was mine:
pressing F7 on the same photograph drew nothing either, and F7 has been in for
months.

The render pass drew the mask *before* `layers`, and for a still the picture is
in `layers`. So the outlines were drawn and then painted over by the
photograph. On video the picture goes through the YUV pipeline earlier in the
same pass and is already down, which is why every use of this feature that
anybody had tried — on a film — worked perfectly. **A comment can be true of
one path and false of the other: "after the picture and before the readouts"
was describing video.**

A second one behind it: clearing what was found emptied the grid on the next
tick, and nothing about that tick asked for a frame, so the last mask uploaded
stayed on the card. The outlines sat there after being dismissed, which reads
as a key that did nothing. Dismissing now clears the grid and the card's copy
itself rather than leaving it to be noticed later.

**"LOOKING..." for ever.** A picture with nothing in it that the model has a
name for produced an empty tally, and an empty tally rendered as the same
"LOOKING..." shown before the first answer. Three states now: nothing yet,
nothing it knows, and what is there. The middle one is the honest answer to a
photograph of a landscape, and it is the one that looked like a fault.

## A block of code in the wrong function

`tick_volume_osd` — which exists to take the volume readout off the screen
after 1.4 seconds — also contained the whole of `stop_video`'s reset: it
cleared the frame cache, the subtitles and the detection results, closed the
menu and zeroed the position. Every volume change did that a second and a half
later.

It was a text splice: an edit meant for the reset landed after the wrong
`self.volume_osd = None;`, of which there are two. It compiled, because every
statement in it is valid in both places, and nothing tested it. It has been
in every archived build going back to the oldest one kept.

Worth stating plainly because the same accident has now happened three times
in this codebase, and it is always the same shape: an anchor that appears
twice, and a replacement that takes the wrong one.

## Pointing at one thing

Clicking a thing that is already outlined **follows** it: it keeps its
identity as the shot moves, everything else stops being drawn, and the bar
says which one. Clicking it again lets it go. It is a click rather than a key
because the things are visibly on screen — a target you can see is a target
you can point at — and it falls through to pause-and-resume when the click
lands on the film rather than on something.

**Ctrl-click asks a different question.** Not "what is in this picture" from a
list of eighty things, but "what is *that*", for anything at all. That is a
different model — MobileSAM, prompted by a point — and it answers with a mask
of its own at 1024 pixels rather than the detector's 160, which is why the
edges are sharper: on a photograph it traced a shirt around a lanyard and a
badge.

### Which model, measured rather than assumed

| | encoder | decoder | licence |
|---|---|---|---|
| MobileSAM (TinyViT) | **188 ms** | 15 ms | Apache-2.0 upstream, mirror silent |
| SlimSAM-77 (pruned ViT-B) | 957 ms | 11 ms | Apache-2.0 |

Five times the wait for the first click decided it. Both were downloaded and
run on the same picture before either was written into the catalogue.

### What the split buys

The encoder turns a picture into an embedding and is the slow half; the
decoder answers a point against it and is not. So the embedding is kept: a
second click on the same picture skips the encoder and comes back in about
twenty milliseconds, which is what makes "that was the shoulder, I meant the
person" a usable gesture rather than another quarter-second wait.

The whole thing runs on a thread of its own, because a fifth of a second on
the display thread is five dropped frames. A film is **paused** first: by the
time the answer arrives a playing picture is five frames further on, and an
outline of what used to be there is worse than none.

### A bigger model, when there is room for it

Three segmentation models rather than one, same family and licence, chosen by
installing them — the largest installed one wins, and removing it falls back
to the next:

| | weights | an inference | what it buys |
|---|---|---|---|
| yolo26n-seg | 11 MB | ~20 ms | the outline |
| yolo26s-seg | 40 MB | ~40 ms | the arm, and the gap between two people |
| yolo26m-seg | 90 MB | ~125 ms | a little more again |

All three pass the gate on this machine — 125 ms an inference still costs the
film nothing measurable, because looking happens five times a second on a
thread of its own and a frame offered while it is busy is skipped rather than
waited for. On a machine with fewer cores that skipping is what will happen,
and the readout will lag the picture rather than the picture stuttering. That
is the right way round, and it is why the larger rows say what they cost.

### Sharper edges, without a bigger model

An outline is only as fine as the grid it is written into, and the old code
walked the *mask* and stamped a square into the grid for each of its cells —
so the finest edge possible was one mask cell across, which at 160 cells for a
whole frame is coarse and stepped.

It now walks the **grid** and asks the mask what is under each cell,
interpolating between the model's own numbers. That means keeping the mask as
the model's values rather than thresholding it to yes and no on the way in:
the boundary lies between cells, and only the numbers on either side say
where. With the grid also raised from 480 to 960 wide — half a megabyte a
frame, against a picture that is many times that — the same model produces a
smooth outline instead of a staircase.

Worth being clear about what this is not: it does not invent detail the model
did not have. A larger segmentation model would, and that is a catalogue row
rather than a code change.

### Two things this turned up

**A photograph reaches a model differently from a film.** The app gives a
still's pixels to the GPU and frees them, so there is nothing to hand over
later. Rather than keep a second copy of every picture on the chance that
someone points at it, both workers now take a *subject* — a frame, or a path
they read themselves. Which also fixed a quiet gap: `F7` on a photograph used
to say `LOOKING...` for ever, because only films were ever offered.

**Tracking has to be told when there is nothing to track.** A detection is
drawn from its second sighting, which is what stops one-frame false positives
flickering. A photograph is looked at once and never changes, so it never had
a second sighting and always came back empty. It is exempt now, and the
reason is written where the exemption is.

## The first inference

`show --detect FILE` loads a model and prints what it finds. Measured on a
700x922 photograph: model loaded in 33 ms, preparation 3.2 ms, **first
inference 24 ms, warm 23 ms** — so the first one is not a special case worth
excluding here, unlike a codec's first frame.

Two things this slice existed to discover, both found:

**The model does not emit what its name suggests.** `yolo26n-ONNX` outputs
`logits [1,300,80]` and `pred_boxes [1,300,4]` — the DETR family's layout, with
sigmoid scores, no background class and no suppression to do — not the
end-to-end `[1,300,6]` the YOLO name implies. The first decoder was written for
the wrong shape and silently found nothing, which is exactly how a
misunderstood tensor fails. Outputs are now looked up **by name**, because
reading boxes as scores produces confident nonsense rather than an error.

**A session must never be dropped.** With a `dlopen`ed runtime, destroying one
segfaults the process at exit — and the drop itself completes and execution
carries on, so the crash looks unrelated to what caused it. See
`docs/dependencies.md`.

## Plugins: what is installed, and what installs it

`F8` lists what can be installed, what is, and how much disk it takes; `ENTER`
fetches, `DEL` removes, `V` re-checks. `show --plugins list|install|remove|verify`
is the same logic headless, which is how the download path was built and tested
before any window was involved.

**The manifest is compiled in** (`src/ml/catalogue.rs`). `show` never runs a
graph described by something it fetched — it fetches opaque blobs, from URLs it
already knew, and checks them against digests it already knew. A new model
therefore means a new release. That is deliberate: the trust boundary stays at
the binary.

**Two digests, guarding different things.** The catalogue pins *the download* —
a wheel, an archive — and that is checked before anything is unpacked. What
lands on disk is something taken *out* of that download, so its digest is only
knowable afterwards; it is recorded in `models/index` and is what `verify`
checks. The first version conflated them and reported a freshly installed file
as damaged.

Artefacts come from PyPI rather than GitHub releases, because PyPI publishes a
digest for every file it serves — so the constant in the catalogue has a
published source rather than being one we computed once and hoped about.
Reading one file out of a wheel needed a zip reader, which is 120 lines in
`src/zip.rs` against a dependency for one operation on one file; `flate2` was
already in the tree via `png`, so inflate cost nothing.

**A test deleted the user's data.** `store`'s tests originally called `remove`
on the real data directory and destroyed what had just been installed. Every
operation now takes an explicit root and the mutating tests use a disposable
one — verified by installing, running the whole suite, and checking it survived.

## Feeding a model: measured before anything was downloaded

`show --ml-probe FILE` does the work that surrounds an inference — letterbox,
YUV to RGB, planar f32 — and reports what it costs. It exists because that work
happens for every frame whether or not the model turns out to be fast, and if
it does not fit in the budget no choice of model rescues it.

| source | to 640x640 | share of the frame budget |
|---|---|---|
| 1920x1080 at 25 fps | 0.87 ms | 2.2% |
| 1440x2560 at 30 fps | 1.08 ms | 3.2% |

So preparation is free in practice and the whole budget is available to the
model. The picture is letterboxed rather than stretched: a detector trained on
undistorted boxes places undistorted boxes on a distorted picture, which is
wrong in a way that is hard to see and impossible to correct afterwards.

One number in the first version of this probe was a lie worth recording. It
reported "decode 0.00 ms", which flattered the pipeline by hiding where the work
is — the decoder runs on its own thread and stays ahead, so what was being timed
was only the cost of taking a frame already waiting. It says `fetch` now.

## Diagnosing playback without a screen

`show --probe FILE` opens no window and plays at zero gain, so it can run while
the machine is in use. It reports the audio clock against the wall clock, frame
pacing, subtitle tracks and their cues, and a seek. Pacing is the one worth
knowing about: **spread** is how unevenly frames land, **starved** is how often
nothing was decoded in time, and **underruns** is how often the sound ran dry. They look identical on screen and have
nothing to do with each other — uneven spacing is a clock fault, starvation is
a throughput fault, and a fix for one does nothing for the other.

The measurement that justified the interpolated clock, on a 25 fps file:

| clock | spread | worst |
|---|---|---|
| staircase, one step per write block | 7.1 ms (18% of a frame) | 19.1 ms |
| interpolated between writes | 0.5 ms (1% of a frame) | 4.7 ms |

Five things it learned the hard way.

It plays a film to its end and then seeks back into it. Reaching the end used
to be the end of the *pipeline* as well — the threads read the last packet and
exited — so a seek afterwards had nobody left to act on it, and a film paused
at the end could not be restarted by dragging the bar back. Reaching the end
is not the same as being finished with it.

It checks that a film **starts on its own**, before anything starts the device
by hand. Getting that wrong deadlocks: the device starts when the first
picture is shown, the first picture waits for the clock, and a stopped clock
never moves — so nothing ever happens until the film is seeked. Every other
check here started the device itself, which is exactly why none of them
noticed. The reproduction is a file whose video begins after its audio, which
Opus produces routinely through its pre-skip:
`ffmpeg -i in.mp4 -itsoffset 1 -i in.mp4 -map 1:v -map 0:a -c copy out.mp4`.

Pacing is measured against **each frame's own timestamp**, not against the
average frame rate. Screen recordings and anything else with a variable frame
rate space their frames unevenly on purpose, and showing them at their own
timestamps is correct — measured against a nominal they never had, two such
files read as 24% and 12% out; measured properly, both are 0.2%. Pacing is measured **first**, before any
other check touches the frame queue: the later ones drain frames as fast as
they decode, which drives the decoder far ahead of the sound and makes the
measurement about the probe rather than the player. And the **first second is
excluded** — the codec is building state and the queue is empty, and judging
steady playback by it would condemn every file.

What it does **not** cover is presentation: upload, compositing and vsync all
happen after this point. A file that probes clean and still judders on screen
is a rendering problem, and this is how the two are told apart.

## Measured backlog

Ordered by payoff, with the measurement that justifies each.

1. ~~**Tiled / pyramidal rendering**~~ — **built.** `src/tiled.rs` reads
   pyramid levels and individual tiles, and is verified against all three real
   file shapes via `show --tiles <file>`:

   | file | geometry | tile read |
   |---|---|---|
   | 1950 aerial survey | 114944x98304, 9 levels, WebP tiles | 0.8-2 ms |
   | 1955 mosaic | 52416x34080, no pyramid, deflate | 10.5 ms |
   | 1982 Soviet plan | 23585x18939, no pyramid, deflate | 16.6 ms |

   `src/tiles.rs` plans which level and tiles a view needs, reads them on a
   pool of threads (each new plan replacing the queue), and keeps them on the
   card under a budget, least recently used out first. Missing tiles borrow
   from finer or coarser ones already there. The debug badge says which level
   is drawn and how many of its tiles are in.

   **Two things to remember when picking this up.**

   *An upstream bug is worked around here.* `tiff` 0.11.3's WebP path sizes its
   output buffer as `samples x w x h` from the TIFF's sample count, while
   `image-webp` demands exactly its own `output_buffer_size()`. Tiles with no
   alpha channel in an RGBA file therefore fail with `ImageTooLarge` — which was
   most of the tiles in the aerial survey. We read `TileOffsets`/`TileByteCounts`
   and decode those tiles ourselves. Worth reporting upstream: it makes
   GDAL-written WebP COGs largely unreadable. If a fixed `tiff` ships, the
   workaround in `TileReader::read_webp` can go.

   *Files without a pyramid get one, once.* Zoomed out, the 1955 mosaic means
   all 6,901 tiles at 10.5 ms — about 72 seconds of decoding — and zooming in
   from there had only the full-size level to drop to. Subsampling was weighed
   and dropped: it would show a gappy stand-in. Instead the first open, which
   reads every tile for the overview anyway, also stores halved levels in
   `~/.cache/show/pyramids/` (`src/pyramid.rs`), and every later open reads
   those as the file's own pyramid. Checked against the source by size and
   modification time; 8 GB budget, least recently used dropped first.

   Measured on a synthetic 24000x16000 deflate file, 4 cores: the first open
   goes from 1.3 s to 1.8 s, and memory from 395 MB to 590 MB. Not yet measured
   on the real files: `show --tiles <file>` reports the first-open time and the
   stored levels.
2. ~~**Multi-file navigation**~~ — built, along with the detailed view
   (`browser.rs`), the mosaic grid, similarity grouping (`similarity.rs`) and a
   persistent thumbnail cache that also reads the desktop's own
   (`thumbs.rs`). Neighbour **prefetch** for the single-image view is still
   open; the grid has its own.
3. **Parallel JPEG entropy decode.** Entropy decoding is the floor that DCT
   scaling cannot cut: 84 ms -> 38 ms is the whole range on a 48 MP file.
   Baseline JPEGs with restart markers can be split across cores.
4. **Pure-Rust AVIF** via `avif-parse` + `rav1d`. Deletes the one optional
   system dependency we have. Needs nothing installed.
5. **HEIC via runtime-loaded libheif.** `libheif.so.1` and `libde265` are
   already present on the dev machine, and the binary keeps working where they
   are not.
6. **Evaluate `zune-png` and `libwebp`.** PNG runs at 163 MPix/s and WebP at
   94, against JPEG's 631.
7. **Staging buffer pool + transfer queue.** Upload is *not* free — 183 MB cost
   41 ms — so it should not block a frame.
8. **Mipmaps.** Linear minification without mips aliases when a large image is
   fitted to a window.
9. **Chase the ~10 ms window-map gap.** On a 2 MP JPEG the decode finishes at
    9.7 ms but rasterising starts at 19.6 ms. Suspected compositor latency;
    it is now a larger slice of cold start than the decode.
10. **Question whether the GPU earns its startup at all.** The software
    rasteriser costs ~1.7 ms/frame and scales with *screen* pixels, not image
    pixels — the same cost for 2 MP as for 400 MP. That is 60 fps. Worth
    testing zoom and pan on a gigapixel texture before believing it, since
    cache behaviour will be far worse there.
