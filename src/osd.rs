//! On-screen display: the load breakdown, drawn over the image.
//!
//! Rendered on the CPU into a small RGBA overlay, which both presentation paths
//! then composite the same way — the software blitter blends it directly, the
//! GPU draws it as a second quad. One rasteriser, one appearance, no divergence
//! between the two.

use std::time::Duration;

use crate::font;
use crate::timing;

/// Pixel scale for the 5x7 font.
///
/// Fractional, deliberately. Drawing each font pixel as an N x N block only
/// allows whole-number sizes, and stepping 2 -> 3 is a fifty percent jump that
/// feels like a lurch rather than a zoom. Sampling the glyph per destination
/// pixel instead gives any scale, and stays crisp because the sampling is
/// nearest-neighbour — a pixel font blurred by interpolation would look worse
/// at every size.
pub const SCALE_COMPACT: f32 = 2.0;
pub const SCALE_PANEL: f32 = 3.0;
/// The debug readouts — zoom badge, load breakdown — half as large again as
/// compact: they are read, not glanced at, and at compact size they were hard
/// to make out on a large screen.
pub const SCALE_DEBUG: f32 = SCALE_COMPACT * 1.5;
/// Smallest and largest the interactive overlays may be scaled to.
pub const SCALE_MIN: f32 = 1.0;
pub const SCALE_MAX: f32 = 6.0;
/// One wheel notch. Small enough that zooming reads as continuous, large
/// enough that a notch is always visible at the smallest size.
pub const SCALE_STEP: f32 = 0.125;

/// Subtitle colours: white on a black edge, which is what every player has
/// settled on because it survives both a snowfield and a night scene.
/// The window frame: dark grey rather than black, so the edge of the window
/// is visible against a dark desktop and against a dark film.
pub const CHROME: [u8; 4] = [44, 46, 50, 255];

const CAPTION_TEXT: [u8; 4] = [255, 255, 255, 255];
const CAPTION_EDGE: [u8; 4] = [0, 0, 0, 235];
const LINE_GAP: usize = 3;
/// Minimum gap between a label and its right-aligned value.
const COLUMN_GAP: usize = 4;
/// Longest label and value a *compact* overlay will lay out, in characters.
///
/// These apply to the readouts that sit over the picture, where one verbose row
/// must not stretch a status badge across the screen. The panel is bounded by
/// the window instead — see [`render_within`] — because there the text is the
/// thing being read and there is no reason not to use the room.
const MAX_LABEL_CHARS: usize = 38;
const MAX_VALUE_CHARS: usize = 34;

/// Background for readouts that sit over the picture: translucent, so the
/// image is still visible behind them.
const PANEL: [u8; 4] = [16, 16, 20, 210];
/// Background for the menu, which is being read rather than glanced at, and
/// where the picture behind is a distraction.
const PANEL_SOLID: [u8; 4] = [14, 14, 18, 246];
pub const TEXT: [u8; 4] = [235, 235, 240, 255];
pub const DIM: [u8; 4] = [150, 155, 165, 255];
pub const ACCENT: [u8; 4] = [120, 220, 160, 255];
pub const WARN: [u8; 4] = [240, 170, 110, 255];
pub const TEXT_COLOUR: [u8; 4] = TEXT;
const SELECTION: [u8; 4] = [60, 90, 130, 255];

/// How a row's current value should read.
///
/// A setting being off is not a problem — it is a setting. Only states the
/// user would actually want to change are coloured, which is why this exists
/// rather than the caller picking a colour directly.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Tone {
    /// Working, or the safe choice.
    Good,
    /// Simply a value; neither good nor bad.
    Neutral,
    /// Worth attention: not installed, or a safety net switched off.
    Warn,
}

impl Tone {
    /// The row this tone calls for.
    pub fn row(self, label: impl Into<String>, value: impl Into<String>) -> Row {
        match self {
            Tone::Good => Row::good(label, value),
            Tone::Neutral => Row::new(label, value, TEXT),
            Tone::Warn => Row::bad(label, value),
        }
    }
}

pub struct Row {
    label: String,
    value: String,
    colour: [u8; 4],
    /// Drawn with a selection bar behind it.
    highlighted: bool,
}

impl Row {
    pub fn new(label: impl Into<String>, value: impl Into<String>, colour: [u8; 4]) -> Row {
        Row { label: label.into(), value: value.into(), colour, highlighted: false }
    }

    pub fn plain(label: impl Into<String>) -> Row {
        Row::new(label, "", TEXT)
    }

    pub fn dim(label: impl Into<String>, value: impl Into<String>) -> Row {
        Row::new(label, value, DIM)
    }

    pub fn good(label: impl Into<String>, value: impl Into<String>) -> Row {
        Row::new(label, value, ACCENT)
    }

    pub fn bad(label: impl Into<String>, value: impl Into<String>) -> Row {
        Row::new(label, value, WARN)
    }

    pub fn selected(mut self, yes: bool) -> Row {
        self.highlighted = yes;
        self
    }

    /// Exact-match on the label, for tests asserting panel contents.
    #[cfg(test)]
    pub fn label_is(&self, label: &str) -> bool {
        self.label.eq_ignore_ascii_case(label)
    }

    #[cfg(test)]
    pub fn label_text(&self) -> &str {
        &self.label
    }

    #[cfg(test)]
    pub fn value_text(&self) -> &str {
        &self.value
    }
}

/// A rasterised overlay, tightly packed RGBA8 with premultiplied-safe alpha.
///
/// Each carries a unique id. That is what lets the draw loop notice the set of
/// overlays has changed without anyone having to remember to say so — a flag
/// that must be set by hand at every call site is a flag that will be missed at
/// one of them.
pub struct Overlay {
    pub id: u64,
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl Overlay {
    pub fn new(rgba: Vec<u8>, width: u32, height: u32) -> Overlay {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Overlay { id, rgba, width, height }
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Turn the recorded timeline into display rows.
///
/// Stages are shown in the order they started, so overlap is visible as a
/// stage whose duration is smaller than the gap to the next one — the same
/// story the waterfall tells, in the space an overlay has.
pub fn rows() -> Vec<Row> {
    let events = timing::events();
    if events.is_empty() {
        return vec![Row::new("LOAD", "", TEXT)];
    }

    let base = events[0].start;
    let end = events.iter().map(|e| e.start + e.dur).max().unwrap_or_default();
    let total = end.saturating_sub(base);

    // The filename has its own badge above this one, so it is not repeated.
    let mut rows = vec![Row::new("LOAD", format!("{:.1} MS", ms(total)), ACCENT)];

    for e in &events {
        // Instants (first_present and friends) have no duration to report and
        // would just be noise in a list of costs.
        if e.dur.is_zero() {
            continue;
        }
        rows.push(Row::new(
            format!("{} {}", e.lane, e.name),
            format!("{:.1}", ms(e.dur)),
            if ms(e.dur) >= ms(total) / 8.0 { TEXT } else { DIM },
        ));
    }

    // The rows will not add up to the elapsed time, and that is the point:
    // stages run on different threads. Saying so beats leaving the reader to
    // wonder whether the numbers are wrong.
    let summed: Duration = events.iter().map(|e| e.dur).sum();
    if summed > total + Duration::from_micros(200) {
        rows.push(Row::new(
            "OVERLAPPED",
            format!("-{:.1}", ms(summed - total)),
            DIM,
        ));
    } else if total > summed + Duration::from_micros(200) {
        rows.push(Row::new("UNTRACKED", format!("{:.1}", ms(total - summed)), DIM));
    }
    rows.push(Row::new("ELAPSED", format!("{:.1} MS", ms(total)), ACCENT));

    // Surfaced here rather than in a panel over the picture: visible on every
    // launch until it is dealt with, intrusive on none.
    if crate::install::state() == crate::install::State::NotInstalled {
        rows.push(Row::new("F1  SETUP - NOT INSTALLED", "", WARN));
    }
    rows
}

/// A labelled bar chart, for a value watched over time.
///
/// A frame rate printed once a second says almost nothing about smoothness: a
/// steady thirty and a stuttering thirty read identically. The shape over the
/// last few seconds is the thing worth seeing, so it gets drawn.
/// `None` samples are gaps rather than zeroes: before playback began there is
/// no rate to report, and drawing that as a floor makes every film look as
/// though it started by stalling.
pub fn chart(title: &str, value: &str, samples: &[Option<f32>], target: f32, scale: f32) -> Overlay {
    let scale = scale.clamp(SCALE_MIN, SCALE_MAX);
    let pad = px(3.0 * scale);
    let chart_h = px(30.0 * scale);
    let gap = px(2.0 * scale);

    let header = render_scaled(&[Row::new(title, value, ACCENT)], scale);
    // The overlay is as wide as the header needs, and the plot then fills that
    // width. Sizing the bars first leaves the chart ending mid-panel with dead
    // space beside it, which reads as the data starting in the wrong place.
    let width = (header.width as usize).max(samples.len() * px(3.0 * scale) + pad * 2);
    let plot_w = width - pad * 2;
    let bar_w = (plot_w / samples.len().max(1)).max(1);
    // Any remainder from the division goes to the left, so the newest sample
    // sits flush against the right edge.
    let plot_x = width - pad - bar_w * samples.len().max(1);
    let height = header.height as usize + gap + chart_h + pad;

    let mut rgba = Vec::with_capacity(width * height * 4);
    for _ in 0..width * height {
        rgba.extend_from_slice(&PANEL);
    }

    // The header, left-aligned into the wider canvas.
    for y in 0..header.height as usize {
        for x in 0..(header.width as usize).min(width) {
            let src = (y * header.width as usize + x) * 4;
            let dst = (y * width + x) * 4;
            rgba[dst..dst + 4].copy_from_slice(&header.rgba[src..src + 4]);
        }
    }

    // Scale to the target rather than to the data, so a rate that falls short
    // looks short rather than being silently renormalised to fill the box.
    let ceiling = target.max(samples.iter().flatten().copied().fold(0.0f32, f32::max)).max(1.0);
    let base = header.height as usize + gap;

    // The target line, so "is it keeping up" is answerable at a glance.
    if target > 0.0 {
        let y = base + chart_h - ((target / ceiling) * chart_h as f32) as usize;
        fill_at(&mut rgba, width, height, plot_x, y.min(base + chart_h - 1), plot_w, px(scale).max(1), DIM);
    }

    // A line rather than bars: the question is how the rate moves over time,
    // and a run of bars hides a dip inside its own gaps.
    let thickness = px(scale).max(1);
    let y_of = |sample: f32| -> usize {
        let h = ((sample / ceiling) * chart_h as f32).round() as usize;
        base + chart_h - h.min(chart_h)
    };
    // Sample centres, so the first and last points sit inside the plot rather
    // than half-clipped at the edges.
    let x_of = |i: usize| plot_x + i * bar_w + bar_w / 2;

    for (i, sample) in samples.iter().enumerate() {
        let Some(sample) = *sample else { continue };
        // Amber when it is not keeping up: the whole question this answers.
        let colour = if target > 0.0 && sample < target * 0.92 { WARN } else { ACCENT };
        let (x1, y1) = (x_of(i), y_of(sample));

        match samples.get(i + 1).copied().flatten() {
            // Draw each segment from its own sample, so the colour changes
            // exactly where the rate crosses the threshold.
            Some(next) => {
                let (x2, y2) = (x_of(i + 1), y_of(next));
                let span = x2.saturating_sub(x1).max(1);
                for step in 0..span {
                    let t = step as f32 / span as f32;
                    let y = y1 as f32 + (y2 as f32 - y1 as f32) * t;
                    // Span the gap to the next row so a steep climb stays a
                    // connected line instead of a dotted diagonal.
                    let y_next = y1 as f32 + (y2 as f32 - y1 as f32) * ((step + 1) as f32 / span as f32);
                    let (lo, hi) = if y <= y_next { (y, y_next) } else { (y_next, y) };
                    let lo = lo.round() as usize;
                    let hi = (hi.round() as usize).max(lo);
                    fill_at(&mut rgba, width, height, x1 + step, lo, thickness,
                              (hi - lo) + thickness, colour);
                }
            }
            // The newest sample gets a marker so "now" is findable.
            None => {
                let dot = thickness * 2;
                fill_at(&mut rgba, width, height, x1.saturating_sub(dot / 2),
                          y1.saturating_sub(dot / 2), dot, dot, colour);
            }
        }
    }

    Overlay::new(rgba, width as u32, height as u32)
}

/// A standalone notice, e.g. for when there is nothing to show.
/// The window's own title bar, drawn because the desktop's is not wanted.
///
/// Thinner than the system's, and in the same grey as the border so the whole
/// frame reads as one edge rather than a bar with a line under it.
pub fn title_bar(title: &str, width: u32, height: u32, scale: f32) -> Overlay {
    let (w, h) = (width.max(1) as usize, height.max(1) as usize);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for _ in 0..w * h {
        rgba.extend_from_slice(&CHROME);
    }

    let text_h = px(font::HEIGHT as f32 * scale);
    let y = h.saturating_sub(text_h) / 2;
    let pad = px(8.0 * scale);

    // The close mark first, so the title can be shortened to clear it.
    let close = "X";
    let close_w = text_width_at(close, scale);
    let close_x = w.saturating_sub(pad + close_w);
    draw_text_into(&mut rgba, w, h, close, close_x, y, TEXT, scale);

    let room = close_x.saturating_sub(pad * 2);
    let mut shown = title.to_string();
    while text_width_at(&shown, scale) > room && shown.chars().count() > 1 {
        shown.pop();
    }
    draw_text_into(&mut rgba, w, h, &shown, pad, y, DIM, scale);

    Overlay::new(rgba, w as u32, h as u32)
}

/// A plain block of the chrome colour, for the window's border.
pub fn chrome_strip(width: u32, height: u32) -> Overlay {
    let (w, h) = (width.max(1) as usize, height.max(1) as usize);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for _ in 0..w * h {
        rgba.extend_from_slice(&CHROME);
    }
    Overlay::new(rgba, w as u32, h as u32)
}

/// Subtitle text: centred lines, no panel behind them, outlined.
///
/// Set in real type rather than the interface font — see `typeface`. The
/// outline is the whole trick: a caption has to be legible over a white sky
/// and a black night in the same film, and a box behind it would hide the
/// picture it is captioning. Each line is rasterised once and the resulting
/// coverage stamped around itself in black before being stamped in white.
pub fn caption(lines: &[String], px: f32) -> Overlay {
    let px = px.clamp(10.0, 200.0);
    // Proportional to the type size, so it holds up at any window size.
    let outline = (px * 0.09).round().max(1.0) as usize;
    let line_h = crate::typeface::line_height(px).ceil() as usize;

    let masks: Vec<crate::typeface::Mask> =
        lines.iter().map(|l| crate::typeface::mask(l, px)).collect();
    let content_w = masks.iter().map(|m| m.width).max().unwrap_or(0);
    if masks.is_empty() || content_w == 0 {
        return Overlay::new(Vec::new(), 0, 0);
    }

    let width = content_w + outline * 2;
    let height = masks.len() * line_h + outline * 2;
    // Transparent: everything not drawn stays picture.
    let mut rgba = vec![0u8; width * height * 4];

    for (i, mask) in masks.iter().enumerate() {
        let left = outline + (content_w - mask.width) / 2;
        let top = outline + i * line_h;

        // The edge first, stamped all round. A square of offsets rather than a
        // circle: the corners cost nothing and the join stays solid.
        let r = outline as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx == 0 && dy == 0 {
                    continue;
                }
                stamp(&mut rgba, width, height, mask, left as i32 + dx, top as i32 + dy, CAPTION_EDGE);
            }
        }
        stamp(&mut rgba, width, height, mask, left as i32, top as i32, CAPTION_TEXT);
    }

    Overlay::new(rgba, width as u32, height as u32)
}

/// Composite a coverage mask into the buffer in one colour.
///
/// Source-over, because the glyphs are anti-aliased now: writing the colour
/// flat wherever there is any coverage would put a hard edge back on type that
/// has just been given a soft one.
fn stamp(
    buf: &mut [u8],
    buf_w: usize,
    buf_h: usize,
    mask: &crate::typeface::Mask,
    x0: i32,
    y0: i32,
    colour: [u8; 4],
) {
    for y in 0..mask.height {
        let py = y0 + y as i32;
        if py < 0 || py as usize >= buf_h {
            continue;
        }
        for x in 0..mask.width {
            let px = x0 + x as i32;
            if px < 0 || px as usize >= buf_w {
                continue;
            }
            let coverage = mask.at(x, y);
            if coverage == 0 {
                continue;
            }
            let a = coverage as f32 / 255.0 * (colour[3] as f32 / 255.0);
            let i = (py as usize * buf_w + px as usize) * 4;
            let dst_a = buf[i + 3] as f32 / 255.0;
            let out_a = a + dst_a * (1.0 - a);
            if out_a <= 0.0 {
                continue;
            }
            for c in 0..3 {
                let src = colour[c] as f32 / 255.0;
                let dst = buf[i + c] as f32 / 255.0;
                let v = (src * a + dst * dst_a * (1.0 - a)) / out_a;
                buf[i + c] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            buf[i + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Break `text` so no line is wider than `max_width` pixels.
///
/// Breaks between words, and only inside one when a single word is wider than
/// the screen — which for dialogue means a URL or a run of dashes.
pub fn wrap(lines: &[String], px: f32, max_width: u32) -> Vec<String> {
    let max_width = max_width.max(1) as f32;
    let mut out = Vec::new();

    for line in lines {
        if crate::typeface::measure(line, px) <= max_width {
            out.push(line.clone());
            continue;
        }
        let mut current = String::new();
        for word in line.split_whitespace() {
            let candidate =
                if current.is_empty() { word.to_string() } else { format!("{current} {word}") };
            if crate::typeface::measure(&candidate, px) <= max_width {
                current = candidate;
                continue;
            }
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            // A word too long for a line of its own is cut rather than
            // overflowing the frame.
            let mut rest = word.to_string();
            while crate::typeface::measure(&rest, px) > max_width && rest.chars().count() > 1 {
                let keep = rest.chars().count().saturating_sub(1);
                let head: String = rest.chars().take(keep).collect();
                if crate::typeface::measure(&head, px) <= max_width {
                    out.push(head);
                    rest = rest.chars().skip(keep).collect();
                    break;
                }
                rest = head;
            }
            current = rest;
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}

/// A seek bar: elapsed on the left, remaining on the right, a track between.
///
/// Sized to a width the caller chooses rather than to its text, because a
/// scrubber whose length changed with the timecode would be unreadable as a
/// position.
pub fn scrubber(elapsed: &str, total: &str, fraction: f32, width: u32, scale: f32) -> Overlay {
    let scale = scale.clamp(SCALE_MIN, SCALE_MAX);
    let pad = px(6.0 * scale);
    let text_h = px(font::HEIGHT as f32 * scale);
    let track_h = px(4.0 * scale);
    let gap = px(4.0 * scale);

    let width = width.max(160) as usize;
    let height = pad * 2 + text_h + gap + track_h;

    let mut rgba = Vec::with_capacity(width * height * 4);
    for _ in 0..width * height {
        rgba.extend_from_slice(&PANEL);
    }

    draw_text_into(&mut rgba, width, height, elapsed, pad, pad, ACCENT, scale);
    let total_w = text_width_at(total, scale);
    draw_text_into(
        &mut rgba,
        width,
        height,
        total,
        width.saturating_sub(pad + total_w),
        pad,
        DIM,
        scale,
    );

    let track_y = pad + text_h + gap;
    let track_w = width - pad * 2;
    fill_at(&mut rgba, width, height, pad, track_y, track_w, track_h, DIM);

    let filled = (track_w as f32 * fraction.clamp(0.0, 1.0)).round() as usize;
    fill_at(&mut rgba, width, height, pad, track_y, filled, track_h, ACCENT);

    // A head on the filled end, so the exact position is readable at a glance
    // rather than being wherever the colour happens to stop.
    let head = px(3.0 * scale).max(2);
    let head_x = (pad + filled).saturating_sub(head / 2).min(width - head);
    fill_at(&mut rgba, width, height, head_x,
        track_y.saturating_sub(head),
        head,
        track_h + head * 2,
        TEXT,
    );

    Overlay::new(rgba, width as u32, height as u32)
}

/// Where the track sits inside a scrubber of this width: `(left, length)`.
///
/// Exported so a click can be turned back into a position. Deriving it in the
/// caller would mean two copies of the same padding, and a bar that seeks to
/// somewhere other than where it was clicked is worse than one that cannot be
/// clicked at all.
pub fn scrubber_track(width: u32, scale: f32) -> (u32, u32) {
    let pad = px(6.0 * scale.clamp(SCALE_MIN, SCALE_MAX)) as u32;
    (pad, width.saturating_sub(pad * 2).max(1))
}

/// Position as `mm.ss.mmm`, for reading a single frame's moment.
///
/// Milliseconds because that is the resolution a frame has: at twenty-five a
/// second they are forty apart, and a timecode rounded to seconds cannot tell
/// two dozen of them apart.
pub fn timecode_ms(at: std::time::Duration) -> String {
    let total = at.as_secs();
    let (m, s, ms) = (total / 60, total % 60, at.subsec_millis());
    format!("{m:02}.{s:02}.{ms:03}")
}

/// Seconds as `H:MM:SS`, or `M:SS` for anything under an hour.
pub fn timecode(at: std::time::Duration) -> String {
    let total = at.as_secs();
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 { format!("{h}:{m:02}:{s:02}") } else { format!("{m}:{s:02}") }
}

pub fn message(lines: &[&str]) -> Overlay {
    let rows: Vec<Row> = lines
        .iter()
        .enumerate()
        .map(|(i, l)| Row::new(*l, "", if i == 0 { ACCENT } else { DIM }))
        .collect();
    render_scaled(&rows, SCALE_PANEL)
}

/// A badge for the grid's sort order: what it sorts by, and an arrow.
pub fn sort_icon(by: crate::playlist::Sort, scale: f32) -> Overlay {
    use crate::playlist::Sort;
    // '#' is the accent, '+' dim. Eleven units square.
    let art: [&str; 11] = match by {
        Sort::Folder => [
            "++++.......",
            "+++++++++++",
            "+.........+",
            "+...###...+",
            "+..#...#..+",
            "+..#####..+",
            "+..#...#..+",
            "+..#...#..+",
            "+.........+",
            "+++++++++++",
            "...........",
        ],
        Sort::Name => [
            "....###....",
            "...#...#...",
            "...#####...",
            "...#...#...",
            "...#...#...",
            "...........",
            "...#####...",
            "......#....",
            ".....#.....",
            "....#......",
            "...#####...",
        ],
        Sort::NthPerFolder(1) => [
            "++++.......",
            "+++++++++++",
            "+.........+",
            "+....#....+",
            "+...##....+",
            "+....#....+",
            "+....#....+",
            "+...###...+",
            "+.........+",
            "+++++++++++",
            "...........",
        ],
        Sort::NthPerFolder(2) => [
            "++++.......",
            "+++++++++++",
            "+.........+",
            "+...###...+",
            "+......#..+",
            "+....##...+",
            "+...#.....+",
            "+...####..+",
            "+.........+",
            "+++++++++++",
            "...........",
        ],
        Sort::NthPerFolder(_) => [
            "++++.......",
            "+++++++++++",
            "+.........+",
            "+...###...+",
            "+......#..+",
            "+....##...+",
            "+......#..+",
            "+...###...+",
            "+.........+",
            "+++++++++++",
            "...........",
        ],
        Sort::Date => [
            "...#####...",
            "..#.....#..",
            ".#...#...#.",
            "#....#....#",
            "#....#....#",
            "#....###..#",
            "#.........#",
            "#.........#",
            ".#.......#.",
            "..#.....#..",
            "...#####...",
        ],
    };
    let u = px(scale.clamp(SCALE_MIN, SCALE_MAX)).max(1);
    // Two units of margin, the symbol, a gap, and a three-wide arrow.
    let (w, h) = (20 * u, 15 * u);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for _ in 0..w * h {
        rgba.extend_from_slice(&PANEL);
    }
    let mut dot = |x: usize, y: usize, colour: [u8; 4]| {
        for py in y * u..(y + 1) * u {
            for pxl in x * u..(x + 1) * u {
                let i = (py * w + pxl) * 4;
                rgba[i..i + 4].copy_from_slice(&colour);
            }
        }
    };
    for (y, row) in art.iter().enumerate() {
        for (x, c) in row.chars().enumerate() {
            match c {
                '#' => dot(x + 2, y + 2, ACCENT),
                '+' => dot(x + 2, y + 2, DIM),
                _ => {}
            }
        }
    }
    // Ascending: an arrow pointing down the grid.
    for y in 2..10 {
        dot(16, y, TEXT);
    }
    for (y, half) in [(9, 2), (10, 1), (11, 0)] {
        for x in 16 - half..=16 + half {
            dot(x, y, TEXT);
        }
    }
    Overlay::new(rgba, w as u32, h as u32)
}

/// Where the grid is scrolled to: a faint strip as tall as a line of panel
/// text, with a bright mark at the position. Not a progress bar — nothing is
/// filled in behind the mark, because nothing has been completed.
pub fn scroll_marker(width: u32, fraction: f32, scale: f32) -> Overlay {
    let scale = scale.clamp(SCALE_MIN, SCALE_MAX);
    let w = width.max(1) as usize;
    let h = line_height_at(scale) + px(3.0 * scale) * 2;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for _ in 0..w * h {
        rgba.extend_from_slice(&[16, 16, 20, 110]);
    }
    let line = px(scale).max(2);
    let x0 = (fraction.clamp(0.0, 1.0) * (w - line.min(w)) as f32).round() as usize;
    for y in 0..h {
        for x in x0..(x0 + line).min(w) {
            let i = (y * w + x) * 4;
            rgba[i..i + 4].copy_from_slice(&ACCENT);
        }
    }
    Overlay::new(rgba, w as u32, h as u32)
}

/// A map with a pin in it: this image knows where on the earth it belongs.
pub fn geo_icon(scale: f32) -> Overlay {
    // '#' the pin, in the accent; '+' and 'o' the folded panels of the map,
    // alternating so the folds read.
    const ART: [&str; 12] = [
        "....###....",
        "...#####...",
        "...##.##...",
        "...#####...",
        "....###....",
        "++++o#oo+++",
        "++++oooo+++",
        "+++ooo+++oo",
        "+++ooo+++oo",
        "++ooo+++ooo",
        "++ooo+++ooo",
        "+ooo+++oooo",
    ];
    let u = px(scale.clamp(SCALE_MIN, SCALE_MAX)).max(1);
    let (w, h) = (15 * u, 16 * u);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for _ in 0..w * h {
        rgba.extend_from_slice(&PANEL);
    }
    for (y, row) in ART.iter().enumerate() {
        for (x, c) in row.chars().enumerate() {
            let colour = match c {
                '#' => ACCENT,
                '+' => DIM,
                'o' => TEXT,
                _ => continue,
            };
            for py in (y + 2) * u..(y + 3) * u {
                for pxl in (x + 2) * u..(x + 3) * u {
                    let i = (py * w + pxl) * 4;
                    rgba[i..i + 4].copy_from_slice(&colour);
                }
            }
        }
    }
    Overlay::new(rgba, w as u32, h as u32)
}

/// Overlays one above another, left-aligned, as wide as the widest — padded
/// with the panel colour so a list reads as one block.
pub fn vstack(parts: &[&Overlay]) -> Overlay {
    let w = parts.iter().map(|p| p.width).max().unwrap_or(0).max(1);
    let h = parts.iter().map(|p| p.height).sum::<u32>().max(1);
    let mut rgba = Vec::with_capacity(w as usize * h as usize * 4);
    for _ in 0..w as usize * h as usize {
        rgba.extend_from_slice(&PANEL_SOLID);
    }
    let mut y0 = 0usize;
    for p in parts {
        for y in 0..p.height as usize {
            let src = y * p.width as usize * 4;
            let dst = ((y0 + y) * w as usize) * 4;
            let run = p.width as usize * 4;
            rgba[dst..dst + run].copy_from_slice(&p.rgba[src..src + run]);
        }
        y0 += p.height as usize;
    }
    Overlay::new(rgba, w, h)
}

/// Badges side by side, top-aligned, with a clear gap between them.
pub fn hstack(parts: &[&Overlay], gap: u32) -> Overlay {
    let w = parts.iter().map(|p| p.width).sum::<u32>() + gap * parts.len().saturating_sub(1) as u32;
    let h = parts.iter().map(|p| p.height).max().unwrap_or(0);
    let (w, h) = (w.max(1), h.max(1));
    let mut rgba = vec![0u8; w as usize * h as usize * 4];
    let mut x0 = 0usize;
    for p in parts {
        for y in 0..p.height as usize {
            let src = y * p.width as usize * 4;
            let dst = (y * w as usize + x0) * 4;
            let run = p.width as usize * 4;
            rgba[dst..dst + run].copy_from_slice(&p.rgba[src..src + run]);
        }
        x0 += (p.width + gap) as usize;
    }
    Overlay::new(rgba, w, h)
}

/// A small badge of two nested folders: the grid is showing subfolders too.
pub fn recursive_icon(scale: f32) -> Overlay {
    let u = px(scale.clamp(SCALE_MIN, SCALE_MAX)).max(1);
    let (w, h) = (22 * u, 18 * u);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for _ in 0..w * h {
        rgba.extend_from_slice(&PANEL);
    }
    // Drawn on a grid of `u`-pixel units, so it scales without blurring.
    let fill = |rgba: &mut [u8], (x0, y0, x1, y1): (usize, usize, usize, usize), colour: [u8; 4]| {
        for y in y0 * u..y1 * u {
            for x in x0 * u..x1 * u {
                let i = (y * w + x) * 4;
                rgba[i..i + 4].copy_from_slice(&colour);
            }
        }
    };
    // A folder: a tab on the left, then the body.
    let folder = |rgba: &mut [u8], x: usize, y: usize, colour: [u8; 4]| {
        fill(rgba, (x, y, x + 5, y + 2), colour);
        fill(rgba, (x, y + 1, x + 12, y + 9), colour);
    };
    folder(&mut rgba, 7, 3, DIM);
    // A panel-coloured margin separates the front folder from the one behind.
    fill(&mut rgba, (2, 6, 16, 17), PANEL);
    folder(&mut rgba, 3, 7, ACCENT);
    Overlay::new(rgba, w as u32, h as u32)
}

/// Rasterise rows at the compact scale.
pub fn render(rows: &[Row]) -> Overlay {
    render_scaled(rows, SCALE_COMPACT)
}

/// Rasterise rows into an overlay sized to its content.
pub fn render_scaled(rows: &[Row], scale: f32) -> Overlay {
    render_inner(rows, scale, None, 0, PANEL)
}

/// Rasterise rows, using as much width as they need up to `max_width` pixels.
///
/// Nothing is shortened unless it genuinely does not fit the window, so a full
/// sentence of explanation stays a full sentence wherever there is room for it.
pub fn render_within(rows: &[Row], scale: f32, max_width: u32) -> Overlay {
    render_inner(rows, scale, Some(max_width), 0, PANEL)
}

/// Render into a buffer of exactly `size`, padding rather than shrinking.
///
/// Used by the browser panel, which is a fixed region of the screen rather than
/// a label sized to its text.
pub fn render_fixed(rows: &[Row], scale: f32, size: (u32, u32)) -> Overlay {
    let content = render_inner(rows, scale, Some(size.0), 0, PANEL_SOLID);
    let (w, h) = (size.0.max(1), size.1.max(1));

    let mut rgba = Vec::with_capacity(w as usize * h as usize * 4);
    for _ in 0..w as usize * h as usize {
        rgba.extend_from_slice(&PANEL_SOLID);
    }
    // Top-left aligned: a file list reads from the top.
    for y in 0..content.height.min(h) as usize {
        let src = y * content.width as usize * 4;
        let dst = y * w as usize * 4;
        let run = (content.width.min(w) as usize) * 4;
        rgba[dst..dst + run].copy_from_slice(&content.rgba[src..src + run]);
    }
    Overlay::new(rgba, w, h)
}

/// As [`render_within`], with the near-opaque background the menu uses.
pub fn render_menu(rows: &[Row], scale: f32, max_width: u32) -> Overlay {
    render_inner(rows, scale, Some(max_width), 0, PANEL_SOLID)
}

/// A menu that will not be narrower than `min_width` pixels.
///
/// A page whose rows change while it is open — a status line appearing, a
/// value that reads shorter than it did — would otherwise change width under
/// the reader, and a menu that resizes as the highlight moves down it looks
/// broken. The floor is a remembered width, not a guess: it is whatever this
/// page has already needed.
pub fn render_menu_at_least(rows: &[Row], scale: f32, max_width: u32, min_width: u32) -> Overlay {
    render_inner(rows, scale, Some(max_width), min_width, PANEL_SOLID)
}

fn render_inner(
    rows: &[Row],
    scale: f32,
    max_width: Option<u32>,
    min_width: u32,
    background: [u8; 4],
) -> Overlay {
    let scale = scale.clamp(SCALE_MIN, SCALE_MAX);
    // Padding tracks the text size so a compact overlay is compact all over.
    let pad = px(3.0 * scale);

    let advance = px((font::WIDTH + font::SPACING) as f32 * scale).max(1);

    // Each row is allotted width on its own terms. A shared budget makes one
    // row's short value reserve a column that every other row's label then has
    // to squeeze beside — which is how filenames ended up truncated with half
    // the panel empty beside them.
    let (clipped, content_w) = match max_width {
        Some(limit) => {
            // The last character on a line carries no trailing spacing, so the
            // content is one gap narrower than `chars * advance`. Counting
            // without adding it back makes the budget one character short of
            // the width that produced it — which matters once a settled width
            // is fed back in as the limit: the widest row would lose a
            // character to an ellipsis on every re-render.
            let usable = ((limit as usize).saturating_sub(pad * 2)
                + px(font::SPACING as f32 * scale))
                / advance.max(1);
            let budget = usable.max(8);

            // The value column is as wide as its widest entry, up to half.
            let value_cap = rows
                .iter()
                .map(|r| r.value.chars().count())
                .max()
                .unwrap_or(0)
                .min(budget / 2);

            let clipped: Vec<(String, String)> = rows
                .iter()
                .map(|r| {
                    let value = elide(&r.value, value_cap);
                    // A row with no value has the whole width to itself; one
                    // with a value gets everything the value does not need.
                    let label_cap = if value.is_empty() {
                        budget
                    } else {
                        budget.saturating_sub(value_cap + COLUMN_GAP).max(4)
                    };
                    (elide(&r.label, label_cap), value)
                })
                .collect();

            // Only as wide as the content actually is. The limit is a ceiling,
            // not a target: a menu stretched to the window with half of it
            // empty is harder to read than one sized to its longest line.
            let label_w = clipped.iter().map(|(l, _)| l.chars().count()).max().unwrap_or(0);
            let value_w = clipped.iter().map(|(_, v)| v.chars().count()).max().unwrap_or(0);
            let natural =
                label_w + if value_w > 0 { COLUMN_GAP + value_w } else { 0 };
            (clipped, natural.min(budget))
        }
        None => {
            let clipped: Vec<(String, String)> = rows
                .iter()
                .map(|r| (elide(&r.label, MAX_LABEL_CHARS), elide(&r.value, MAX_VALUE_CHARS)))
                .collect();
            let label_w = clipped.iter().map(|(l, _)| l.chars().count()).max().unwrap_or(0);
            let value_w = clipped.iter().map(|(_, v)| v.chars().count()).max().unwrap_or(0);
            (clipped, label_w + COLUMN_GAP + value_w)
        }
    };

    let content_w = (content_w * (font::WIDTH + font::SPACING)).saturating_sub(font::SPACING);
    let line_h = px((font::HEIGHT + LINE_GAP) as f32 * scale).max(1);
    let content_h = rows.len() * line_h - (line_h - px(font::HEIGHT as f32 * scale));

    // The floor widens the content itself rather than padding the edge, so the
    // right-aligned values stay against the right edge of the wider page.
    let ceiling = max_width.map(|w| (w as usize).saturating_sub(pad * 2)).unwrap_or(usize::MAX);
    let content_px = px(content_w as f32 * scale)
        .max((min_width as usize).saturating_sub(pad * 2))
        .min(ceiling);

    let width = (content_px + pad * 2).max(1);
    let height = (content_h + pad * 2).max(1);

    let mut rgba = Vec::with_capacity(width * height * 4);
    for _ in 0..width * height {
        rgba.extend_from_slice(&background);
    }

    for (i, row) in rows.iter().enumerate() {
        let y = pad + i * line_h;
        if row.highlighted {
            // A full-width bar, so the selected row reads as a target rather
            // than just differently coloured text.
            fill_at(&mut rgba, width, height, pad / 2,
                y.saturating_sub(px(LINE_GAP as f32 * scale) / 2),
                width - pad,
                line_h,
                SELECTION,
            );
        }
        let (label, value) = &clipped[i];
        draw_text(&mut rgba, width, height, label, pad, y, row.colour, scale);
        // Values are right-aligned so the decimal points line up into a column.
        let x = pad + content_px - px(font::text_width(value) as f32 * scale);
        draw_text(&mut rgba, width, height, value, x, y, row.colour, scale);
    }

    Overlay::new(rgba, width as u32, height as u32)
}

/// Trim to `max` characters, marking the cut so a truncated value never reads
/// Fill a rectangle, clipped to the buffer.
///
/// Signed coordinates on purpose: a box drawn round something at the edge of a
/// picture runs off it, and clipping here is what lets every caller ignore
/// that. Lifted from `mosaic.rs`, which needed exactly this and had it
/// privately, as did `osd` in a version that could not go negative.
pub fn outline(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    x: i64,
    y: i64,
    w: u32,
    h: u32,
    thickness: u32,
    colour: [u8; 4],
) {
    let t = thickness as i64;
    let (ow, oh) = (w + thickness * 2, h + thickness * 2);
    // Top, bottom, left, right.
    fill(dst, dst_w, dst_h, x - t, y - t, ow, thickness, colour);
    fill(dst, dst_w, dst_h, x - t, y + h as i64, ow, thickness, colour);
    fill(dst, dst_w, dst_h, x - t, y - t, thickness, oh, colour);
    fill(dst, dst_w, dst_h, x + w as i64, y - t, thickness, oh, colour);
}

/// Trim to `max` characters, marking the cut so a truncated value never reads
/// as a complete one.
fn elide(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max.saturating_sub(2)).collect::<String>() + ".."
}

/// `fill` for callers already working in buffer coordinates.
///
/// Everything inside this module indexes in `usize`, where a coordinate cannot
/// be negative; the signed version exists for callers placing something
/// relative to a picture, which can.
fn fill_at(
    buf: &mut [u8],
    buf_w: usize,
    buf_h: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    colour: [u8; 4],
) {
    fill(buf, buf_w as u32, buf_h as u32, x as i64, y as i64, w as u32, h as u32, colour);
}

/// A solid rectangle, clipped to the buffer.
pub fn fill(dst: &mut [u8], dst_w: u32, dst_h: u32, x: i64, y: i64, w: u32, h: u32, colour: [u8; 4]) {
    for row in 0..h as i64 {
        let py = y + row;
        if py < 0 || py >= dst_h as i64 {
            continue;
        }
        for col in 0..w as i64 {
            let px = x + col;
            if px < 0 || px >= dst_w as i64 {
                continue;
            }
            let d = ((py as u32 * dst_w + px as u32) * 4) as usize;
            dst[d..d + 4].copy_from_slice(&colour);
        }
    }
}

/// Draw text into any RGBA buffer. Used by the grid, which composites its own
/// labels rather than stacking an overlay per cell.
pub fn draw_text_into(
    buf: &mut [u8],
    buf_w: usize,
    buf_h: usize,
    text: &str,
    x: usize,
    y: usize,
    colour: [u8; 4],
    scale: f32,
) {
    draw_text(buf, buf_w, buf_h, text, x, y, colour, scale);
}

/// Round a scaled length to whole pixels, never to nothing.
fn px(value: f32) -> usize {
    value.round().max(0.0) as usize
}

/// Width in pixels of `text` at `scale`.
pub fn text_width_at(text: &str, scale: f32) -> usize {
    px(font::text_width(text) as f32 * scale)
}

/// Height in pixels of one line at `scale`.
pub fn line_height_at(scale: f32) -> usize {
    px(font::HEIGHT as f32 * scale)
}

fn draw_text(
    buf: &mut [u8],
    buf_w: usize,
    buf_h: usize,
    text: &str,
    x0: usize,
    y0: usize,
    colour: [u8; 4],
    scale: f32,
) {
    let scale = scale.max(0.5);
    let glyph_w = px(font::WIDTH as f32 * scale).max(1);
    let glyph_h = px(font::HEIGHT as f32 * scale).max(1);
    let advance = (font::WIDTH + font::SPACING) as f32 * scale;

    // The pen advances by a float and is rounded only when a glyph is placed,
    // so spacing does not drift across a long line.
    let mut pen = 0.0f32;
    for ch in text.chars() {
        let left = x0 + px(pen);
        if let Some(bits) = font::glyph(ch) {
            // Walk the destination and sample the glyph, rather than expanding
            // each source pixel — which is what allows a fractional scale.
            for dy in 0..glyph_h {
                let row = ((dy as f32 + 0.5) / scale) as usize;
                let Some(byte) = bits.get(row.min(font::HEIGHT - 1)) else { continue };
                let py = y0 + dy;
                if py >= buf_h {
                    break;
                }
                for dx in 0..glyph_w {
                    let col = ((dx as f32 + 0.5) / scale) as usize;
                    if col >= font::WIDTH || byte & (0x80 >> col) == 0 {
                        continue;
                    }
                    let x = left + dx;
                    if x < buf_w {
                        let i = (py * buf_w + x) * 4;
                        buf[i..i + 4].copy_from_slice(&colour);
                    }
                }
            }
        }
        pen += advance;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_overlay_is_sized_to_its_widest_row() {
        let narrow = render(&[Row::new("A", "1", TEXT)]);
        let wide = render(&[Row::new("AAAAAAAAAA", "1", TEXT)]);
        assert!(wide.width > narrow.width);
        assert_eq!(wide.height, narrow.height, "one row is one row either way");
    }

    #[test]
    fn more_rows_make_it_taller() {
        let one = render(&[Row::new("A", "1", TEXT)]);
        let three = render(&[
            Row::new("A", "1", TEXT),
            Row::new("B", "2", TEXT),
            Row::new("C", "3", TEXT),
        ]);
        assert!(three.height > one.height);
    }

    #[test]
    fn text_is_actually_drawn_onto_the_panel() {
        let o = render(&[Row::new("W", "", TEXT)]);
        assert_eq!(o.rgba.len(), o.width as usize * o.height as usize * 4);
        let lit = o.rgba.chunks_exact(4).filter(|p| p[0] > 200).count();
        assert!(lit > 0, "no text pixels were written");
    }

    #[test]
    fn a_wide_window_keeps_text_a_compact_overlay_would_shorten() {
        let long = "REBUILD WITH --FEATURES AVIF-SYSTEM-DAV1D, WHICH ADDS A LIBDAV1D RUNTIME DEPENDENCY";
        let rows = [Row::new("AVIF", long, TEXT)];

        let compact = render_scaled(&rows, SCALE_COMPACT);
        let wide = render_within(&rows, SCALE_COMPACT, 3072);
        assert!(wide.width > compact.width, "a wide window should show more: {} vs {}", wide.width, compact.width);
        assert!(wide.width <= 3072, "but never more than the window: {}", wide.width);
    }

    #[test]
    fn the_chart_reaches_the_right_edge_whatever_the_header_says() {
        // A long header used to make the overlay wide while the plot stayed
        // narrow, leaving the newest samples stranded in the middle.
        let samples = vec![10.0f32; 20];
        let samples: Vec<Option<f32>> = samples.iter().map(|v| Some(*v)).collect();
        let ov = chart("A VERY LONG CODEC NAME HERE", "12.3 / 29.97 FPS", &samples, 30.0, 2.0);

        // The rightmost bar column must contain drawn pixels.
        let column = ov.width as usize - 6 - 1;
        let lit = (0..ov.height as usize).any(|y| {
            let i = (y * ov.width as usize + column) * 4;
            ov.rgba[i] != PANEL[0] || ov.rgba[i + 1] != PANEL[1]
        });
        assert!(lit, "the chart does not reach the right edge");
    }

    #[test]
    fn a_partly_filled_window_still_ends_at_the_right() {
        // Early in playback most buckets are empty; the ones with data are the
        // newest and belong against the right edge.
        let mut samples = vec![0.0f32; 20];
        for s in samples.iter_mut().skip(15) {
            *s = 30.0;
        }
        let samples: Vec<Option<f32>> = samples.iter().map(|v| Some(*v)).collect();
        let ov = chart("HEVC", "30.0 / 29.97 FPS", &samples, 30.0, 2.0);
        let column = ov.width as usize - 6 - 1;
        let lit = (0..ov.height as usize).any(|y| {
            let i = (y * ov.width as usize + column) * 4;
            ov.rgba[i] != PANEL[0] || ov.rgba[i + 1] != PANEL[1]
        });
        assert!(lit, "the newest sample should be at the right edge");
    }

    #[test]
    fn short_content_makes_a_narrow_overlay_however_much_room_there_is() {
        let rows = [Row::new("A", "1", TEXT), Row::new("BB", "22", TEXT)];
        let narrow = render_within(&rows, SCALE_COMPACT, 3072);
        assert!(
            narrow.width < 200,
            "a two-word menu should not span the screen: {}",
            narrow.width
        );
    }

    #[test]
    fn a_long_name_runs_to_the_edge_even_beside_rows_that_have_values() {
        // The case from the browser: an info block with short values above a
        // list of long filenames. The names must not be cut to fit beside a
        // value column they do not use.
        let name = "A VERY LONG PHOTOGRAPH FILENAME THAT SHOULD REACH THE RIGHT EDGE.JPG";
        let rows = [
            Row::new("SIZE", "1920 X 1080", TEXT),
            Row::new("FILE", "179.2 KB", TEXT),
            Row::plain(name),
        ];
        let ov = render_fixed(&rows, SCALE_COMPACT, (900, 400));

        // Count how much of the name survived by rendering it alone at the
        // same width; the two should agree.
        let alone = render_fixed(&[Row::plain(name)], SCALE_COMPACT, (900, 400));
        let lit = |o: &Overlay| o.rgba.chunks_exact(4).filter(|p| p[0] > 200).count();
        assert!(
            lit(&ov) >= lit(&alone) / 2,
            "the name lost most of its width to a value column it does not use"
        );
    }

    /// The exported track geometry has to agree with what `scrubber` draws,
    /// or a click seeks to somewhere other than where it landed. Checked by
    /// finding the drawn boundary rather than by recomputing the padding,
    /// which would only prove the test and the code share an assumption.
    #[test]
    fn the_track_geometry_matches_what_is_drawn() {
        let width = 800u32;
        let ov = scrubber("0:30", "1:00", 0.5, width, SCALE_COMPACT);
        let (track_x, track_w) = scrubber_track(ov.width, SCALE_COMPACT);

        // The row through the middle of the track.
        let row = (ov.height as usize) - px(4.0 * SCALE_COMPACT) / 2 - px(6.0 * SCALE_COMPACT);
        let at = |x: usize| {
            let i = (row * ov.width as usize + x) * 4;
            [ov.rgba[i], ov.rgba[i + 1], ov.rgba[i + 2]]
        };

        // Just inside each end of the track is track, not panel.
        assert_ne!(at(track_x as usize + 1), [PANEL[0], PANEL[1], PANEL[2]], "track starts late");
        let last = (track_x + track_w - 2) as usize;
        assert_ne!(at(last), [PANEL[0], PANEL[1], PANEL[2]], "track ends early");

        // Half filled: the accent runs out around the middle of the track.
        let filled = (track_x as usize..(track_x + track_w) as usize)
            .filter(|x| at(*x) == [ACCENT[0], ACCENT[1], ACCENT[2]])
            .count();
        let expected = track_w as usize / 2;
        assert!(
            filled.abs_diff(expected) <= px(3.0 * SCALE_COMPACT).max(2),
            "half of a {track_w}px track drew {filled}px of fill, wanted about {expected}"
        );
    }

    #[test]
    fn a_row_with_no_value_gets_the_whole_width() {
        let name = "A RATHER LONG PHOTOGRAPH FILENAME FROM A CAMERA 2026.JPG   3/32";
        let ov = render_within(&[Row::plain(name)], SCALE_PANEL, 3072);
        // Wide enough to hold every character, so none should be dropped.
        let needed = text_width_at(name, SCALE_PANEL);
        assert!(
            ov.width as usize >= needed,
            "{} is too narrow for {} pixels of text",
            ov.width,
            needed
        );
    }

    #[test]
    fn a_narrow_window_still_shortens() {
        let long = "A".repeat(300);
        let narrow = render_within(&[Row::new("X", long, TEXT)], SCALE_COMPACT, 600);
        assert!(narrow.width <= 600, "must fit the window: {}", narrow.width);
    }

    #[test]
    fn one_long_row_does_not_stretch_the_whole_panel() {
        let short = render(&[Row::new("A", "B", TEXT), Row::new("C", "D", TEXT)]);
        let with_essay = render(&[
            Row::new("A", "B", TEXT),
            Row::new("C", "D".repeat(400), TEXT),
        ]);
        // Bounded growth: wider, but not by four hundred characters.
        let cap = (MAX_VALUE_CHARS * (font::WIDTH + font::SPACING)) as f32 * SCALE_COMPACT;
        let cap = cap as u32 + 40;
        assert!(
            with_essay.width < short.width + cap,
            "{} is not bounded by {} + {cap}",
            with_essay.width,
            short.width
        );
    }

    #[test]
    fn elision_marks_the_cut() {
        assert_eq!(elide("SHORT", 10), "SHORT");
        let cut = elide("ABCDEFGHIJ", 5);
        assert_eq!(cut, "ABC..");
        assert_eq!(cut.chars().count(), 5);
    }

    #[test]
    fn buffer_writes_stay_in_bounds_for_oversized_text() {
        // Long text must clip rather than panic or corrupt memory.
        let o = render(&[Row::new("W".repeat(200), "9".repeat(50), TEXT)]);
        assert_eq!(o.rgba.len(), o.width as usize * o.height as usize * 4);
    }
}
