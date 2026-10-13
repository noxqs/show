# Roadmap

The working list. One line per item; the reasoning and measurements live in
`BACKLOG.md`, and each item names the section it comes from.

**How it is used.** The commit that finishes an item ticks it here in the same
commit. When every item in a release section is ticked, bump `version` in
`Cargo.toml`, move the section's items into the release notes, and push a tag
`vX.Y.Z`; CI builds and publishes the release from the tag.

Released: **v1.1.0** (2026-10-13).

## v1.2.0 (next)

- [ ] Tiled / pyramidal renderer: tile cache, draw tiles instead of one texture,
      switch level on zoom, tile readout in the badge (Measured #1). *In progress
      on K11.*
- [ ] Files without a pyramid: subsample tiles when zoomed out, so the 1955
      mosaic does not take ~72 s (Measured #1).
- [ ] Subtitle cue retention: a backward seek of more than 30 s loses captions;
      replace the fixed `KEEP` window in `src/subtitles.rs` (Next #4).
- [ ] Neighbour prefetch for the single-image view (Measured #2).
- [ ] Docs: audio clock already re-anchors on seek; update the stale note in
      `BACKLOG.md` (Next #1).

## v1.3.0

- [ ] Zero-copy hardware frames: VA surface → DMA-BUF → Vulkan, so hardware
      decode stops costing more CPU than software (Next #3).
- [ ] Mipmaps, so a large image fitted to the window does not alias
      (Measured #8).
- [ ] Staging buffer pool + transfer queue, so uploads do not block a frame
      (Measured #7).

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
