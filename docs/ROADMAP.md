# Roadmap

The working list. One line per item; the reasoning and measurements live in
`BACKLOG.md`, and each item names the section it comes from.

**How it is used.** The commit that finishes an item ticks it here in the same
commit. When every item in a release section is ticked, bump `version` in
`Cargo.toml` and merge to main: CI sees a version with no tag yet, tags it
`vX.Y.Z` and publishes the release. A version with a dash (`1.3.0-rc1`) is
published as a pre-release.

Linux only: no other platform is planned.

Released: **v1.2.0** (2026-10-15), **v1.1.0** (2026-10-13).

## v1.2.0 (released 2026-10-15)

- [x] Tiled / pyramidal renderer: tile cache, draw tiles instead of one texture,
      switch level on zoom, tile readout in the badge (Measured #1).
- [x] Files without a pyramid: store halved levels on first open, so zooming
      out or in no longer reads every full-size tile (Measured #1).
- [x] Measure the stored pyramid on the real files on K11: first open, later
      opens, disk size (`show --tiles <file>`).
- [x] Subtitle cue retention: a backward seek of more than 30 s loses captions;
      replace the fixed `KEEP` window in `src/subtitles.rs` (Next #4).
- [x] Neighbour prefetch for the single-image view (Measured #2).
- [x] Check on K11 that stepping through a folder feels instant one way and
      the other.
- [x] Keep the screensaver off while a video plays (`src/awake.rs`).
- [x] Check on K11 that the screen stays on through a video, and blanks again
      when paused.
- [x] Subtitle size on ctrl < / >, kept between sessions, growing about the
      caption's middle.
- [x] Docs: audio clock already re-anchors on seek; update the stale note in
      `BACKLOG.md` (Next #1).
- [x] Windowed, a plain drag anywhere moves the window; shift + drag pans the
      picture or film. Fullscreen, a plain drag still pans.
- [x] Releases publish themselves when a new version reaches main.

## v1.3.0 (next)

- [ ] Zero-copy hardware frames: VA surface → DMA-BUF → Vulkan, so hardware
      decode stops costing more CPU than software (Next #3).
- [x] Mipmaps, so a large image fitted to the window does not alias
      (Measured #8).
- [x] Uploads do not block a frame: the full-resolution picture is uploaded
      on a thread of its own (Measured #7). wgpu has no transfer queue to
      ask for; a staging pool waits until the copy itself measures slow.

## Later

- [ ] Subtitle styling: tell signs from dialogue (Next #4).
- [ ] Text shaping for Arabic / Devanagari subtitles (Next #4).
- [ ] Multichannel audio downmix with our own matrix (Next #1).
- [ ] Parallel JPEG entropy decode for baseline files with restart markers
      (Measured #3).
- [ ] Pure-Rust AVIF (`avif-parse` + `rav1d`), removing the need for libheif's
      AV1 plugin (Measured #4).
- [ ] Evaluate `zune-png` and `libwebp` (Measured #6).
- [ ] Chase the ~10 ms window-map gap at cold start (Measured #9).
- [ ] Test whether the GPU earns its startup against the software rasteriser on
      a gigapixel image (Measured #10).
- [ ] Report the `tiff` 0.11.3 WebP buffer-size bug upstream; drop the
      workaround once fixed (Measured #1).

## Parked

Deliberately out of scope for now; see **Deferred** in `BACKLOG.md`: upscaler
and LLM-tagging plugins, basic edits, batch mode, GeoTIFF + OpenStreetMap
overlay, builds for older Linux Mint.
