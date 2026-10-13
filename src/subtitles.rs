//! Subtitles: finding them, decoding them, and knowing which one is due.
//!
//! Two families arrive by very different routes and meet at [`Cue`].
//!
//! * **Text** — SRT, ASS/SSA, WebVTT, mov_text. libavcodec turns all of them
//!   into ASS dialogue lines, so there is one parser here rather than four.
//! * **Bitmap** — PGS, VobSub, DVB, XSUB. These are already pictures, in a
//!   palette of their own, positioned in a canvas that is usually the
//!   authoring resolution rather than the video's.
//!
//! Rendering is deliberately not libass. It would bring freetype, fontconfig
//! and harfbuzz — three system libraries — to a program that has gone to some
//! trouble to need none, and it would draw captions in a typeface with nothing
//! to do with the rest of the interface. The 5x7 font the OSD already uses
//! grew a lowercase for this instead.
//!
//! What that costs is honest to state: ASS positioning, per-line styling,
//! karaoke and typesetting are dropped. What survives is the dialogue, its
//! timing, and its line breaks, which is what subtitles are for.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// Which rasteriser a stream needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Text,
    Bitmap,
}

impl Kind {
    /// Classify by codec name, which is how FFmpeg names these to us.
    pub fn of(codec: &str) -> Option<Kind> {
        Some(match codec {
            "ass" | "ssa" | "subrip" | "srt" | "text" | "webvtt" | "mov_text" | "eia_608"
            | "subviewer" | "microdvd" | "sami" | "realtext" | "stl" | "jacosub" | "mpl2"
            | "pjs" | "vplayer" | "ttml" => Kind::Text,
            "hdmv_pgs_subtitle" | "dvd_subtitle" | "dvb_subtitle" | "xsub"
            | "hdmv_text_subtitle" | "dvb_teletext" => Kind::Bitmap,
            _ => return None,
        })
    }
}

/// A subtitle stream the user can choose.
#[derive(Clone, Debug)]
pub struct Track {
    /// Stream index within its own container.
    pub index: usize,
    pub kind: Kind,
    pub codec: String,
    /// Language tag, when the file bothered to say.
    pub language: Option<String>,
    pub title: Option<String>,
    /// Set when the track came from a file beside the video.
    pub external: Option<PathBuf>,
}

impl Track {
    /// A short name for the menu.
    pub fn label(&self) -> String {
        let mut parts = Vec::new();
        if let Some(lang) = &self.language {
            parts.push(lang.to_uppercase());
        }
        if let Some(title) = &self.title {
            parts.push(title.clone());
        }
        if parts.is_empty() {
            parts.push(self.codec.to_uppercase());
        }
        if self.kind == Kind::Bitmap {
            // Worth saying: a picture track cannot be restyled or resized, and
            // looks different from the rest for reasons that are not a bug.
            parts.push("(PICTURE)".into());
        }
        if self.external.is_some() {
            parts.push("(FILE)".into());
        }
        parts.join(" ")
    }
}

/// One picture belonging to a bitmap caption.
#[derive(Clone)]
pub struct Image {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Straight (not premultiplied) RGBA.
    pub rgba: Vec<u8>,
}

#[derive(Clone)]
pub enum Payload {
    Lines(Vec<String>),
    /// Pictures, and the canvas their coordinates are relative to.
    Images(Vec<Image>, (u32, u32)),
}

/// A caption and the window it occupies.
#[derive(Clone)]
pub struct Cue {
    /// Which stream it came from, so a track change can filter without
    /// re-decoding what has already been read.
    pub track: usize,
    pub start: Duration,
    pub end: Duration,
    pub payload: Payload,
}

/// How long a caption stays up when nothing says otherwise.
const DEFAULT_DURATION: Duration = Duration::from_secs(3);

/// Take the spoken text out of an ASS dialogue line.
///
/// libavcodec hands every text format over in this shape, so this is the only
/// text parser in the program. The line is nine comma-separated fields and
/// then the dialogue, which may itself contain commas — hence splitting a
/// fixed number of times rather than on every comma.
pub fn text_from_ass(line: &str) -> Vec<String> {
    // Two shapes reach here, and the field counts differ by one.
    //
    // A decoder hands over the event body alone:
    //     ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text
    // A script file line carries the times instead of the read order:
    //     Dialogue: Layer,Start,End,Style,Name,ML,MR,MV,Effect,Text
    //
    // Either way the text is the last field and may contain commas of its own,
    // so the split is bounded rather than exhaustive. Getting this off by one
    // truncates every caption at its first comma, which reads as the subtitle
    // itself being wrong rather than the parser.
    let (body, fields) = match line.strip_prefix("Dialogue:") {
        Some(rest) => (rest.trim_start(), 10),
        None => (line, 9),
    };
    match body.splitn(fields, ',').nth(fields - 1) {
        Some(text) => strip_ass_markup(text),
        // Not an event line at all; show it rather than dropping it.
        None => strip_ass_markup(body),
    }
}

/// Remove override blocks and turn ASS escapes into real line breaks.
fn strip_ass_markup(text: &str) -> Vec<String> {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            // {\pos(..)}, {\i1}, and the rest of the styling we do not honour.
            '{' => {
                for c in chars.by_ref() {
                    if c == '}' {
                        break;
                    }
                }
            }
            '\\' => match chars.peek() {
                // \N is a hard break, \n a soft one; both end the line here.
                Some('N') | Some('n') => {
                    chars.next();
                    out.push('\n');
                }
                // A non-breaking space.
                Some('h') => {
                    chars.next();
                    out.push(' ');
                }
                _ => out.push(c),
            },
            _ => out.push(c),
        }
    }

    out.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// Extensions worth looking for beside a video, best first.
///
/// ASS before SRT: if someone shipped both, the styled one is the one they
/// meant, and we get the same dialogue out of either.
const EXTENSIONS: &[&str] = &["ass", "ssa", "srt", "vtt", "sub", "ttml"];

/// Every subtitle file in the video's folder, best first.
///
/// Wider than [`find_beside`] on purpose: that answers "what should load
/// automatically", which has to be conservative, while this answers "what
/// could I pick", where offering a file whose name does not match is exactly
/// the point — subtitles downloaded separately rarely arrive named to match.
pub fn find_in_folder(video: &Path) -> Vec<PathBuf> {
    let mut found = find_beside(video);
    let Some(parent) = video.parent() else { return found };

    let mut others = Vec::new();
    for dir in [parent.to_path_buf(), parent.join("Subs"), parent.join("subs")] {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() || found.contains(&path) || others.contains(&path) {
                continue;
            }
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            if EXTENSIONS.contains(&ext.as_str()) {
                others.push(path);
            }
        }
    }
    // The ones that match the film's name first, since they are the likeliest
    // choice; everything else after, in a stable order.
    others.sort();
    found.extend(others);
    found
}

/// Subtitle files belonging to `video`.
///
/// Matches `film.srt`, the `film.en.srt` convention releases use, and the
/// `Subs/` directory box sets hide them in. Case-insensitive throughout: a
/// file copied off a Windows share is as likely to be `Film.SRT`.
pub fn find_beside(video: &Path) -> Vec<PathBuf> {
    let Some(stem) = video.file_stem().map(|s| s.to_string_lossy().to_lowercase()) else {
        return Vec::new();
    };

    let mut dirs = Vec::new();
    if let Some(parent) = video.parent() {
        dirs.push(parent.to_path_buf());
        for sub in ["Subs", "subs", "Subtitles", "subtitles"] {
            let dir = parent.join(sub);
            if dir.is_dir() {
                dirs.push(dir);
            }
        }
    }

    let mut found = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            if !EXTENSIONS.contains(&ext.as_str()) {
                continue;
            }
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            // `film` matches and so does `film.en`, but `film2` must not.
            if name == stem || name.starts_with(&format!("{stem}.")) {
                found.push(path);
            }
        }
    }

    // Preferred extension first, then by name so the order is stable run to run.
    found.sort_by_key(|p| {
        let ext = p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        (EXTENSIONS.iter().position(|e| *e == ext).unwrap_or(99), p.to_string_lossy().into_owned())
    });
    found
}

/// Every caption read so far, and which track is being shown.
pub struct Subtitles {
    cues: Vec<Cue>,
    pub tracks: Vec<Track>,
    /// Index into `tracks`; `None` means subtitles are off.
    selected: Option<usize>,
}

impl Default for Subtitles {
    fn default() -> Self {
        Subtitles::new()
    }
}

impl Subtitles {
    pub fn new() -> Subtitles {
        Subtitles { cues: Vec::new(), tracks: Vec::new(), selected: None }
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    pub fn selected(&self) -> Option<&Track> {
        self.selected.and_then(|i| self.tracks.get(i))
    }

    pub fn selected_index(&self) -> Option<usize> {
        self.selected
    }

    /// Show a particular track, or none at all.
    pub fn select(&mut self, track: Option<usize>) {
        self.selected = track.filter(|i| *i < self.tracks.len());
    }

    /// Add a track, and select it if nothing is selected yet.
    ///
    /// A file beside the video wins over an embedded stream: someone who went
    /// to the trouble of downloading subtitles meant to use them.
    pub fn add_track(&mut self, track: Track) {
        let external = track.external.is_some();
        self.tracks.push(track);
        let last = self.tracks.len() - 1;
        match self.selected {
            None => self.selected = Some(last),
            Some(current) if external && self.tracks[current].external.is_none() => {
                self.selected = Some(last)
            }
            _ => {}
        }
    }

    pub fn push(&mut self, cue: Cue) {
        // Seeking backwards makes the demuxer read the same packets again, so
        // the same caption arrives more than once. Without this they pile up
        // for as long as someone keeps scrubbing over one stretch of a film.
        if self
            .cues
            .iter()
            .any(|c| c.track == cue.track && c.start == cue.start)
        {
            return;
        }
        self.cues.push(cue);
    }

    /// Step to the next track, then to off, then round again.
    ///
    /// Off is part of the cycle rather than a separate key: a caption you do
    /// not want gone is the same gesture as one you want changed.
    pub fn cycle(&mut self) -> Option<&Track> {
        if self.tracks.is_empty() {
            return None;
        }
        self.selected = match self.selected {
            None => Some(0),
            Some(i) if i + 1 < self.tracks.len() => Some(i + 1),
            Some(_) => None,
        };
        self.selected()
    }

    /// The caption due at `now` on the selected track, if any.
    pub fn at(&self, now: Duration) -> Option<&Cue> {
        let track = self.selected()?.index;
        self.cues
            .iter()
            .filter(|c| c.track == track && c.start <= now && now < c.end)
            // A later cue wins an overlap, which is how a correction or a
            // second speaker replaces what is on screen.
            .next_back()
    }

    /// Forget captions well behind the play position.
    ///
    /// Without this a long film with two picture-based tracks would keep every
    /// caption it ever decoded. There is no seeking yet, so anything this far
    /// back cannot be asked for again; when seeking lands this becomes a cache
    /// bound rather than a discard.
    pub fn retire(&mut self, now: Duration) {
        const KEEP: Duration = Duration::from_secs(30);
        let Some(cutoff) = now.checked_sub(KEEP) else { return };
        self.cues.retain(|c| c.end >= cutoff);
    }

    pub fn clear(&mut self) {
        self.cues.clear();
        self.tracks.clear();
        self.selected = None;
    }
}

/// Work out a cue's window from what the decoder reported.
///
/// `start`/`end` are milliseconds relative to the packet, and formats disagree
/// about which of the three available timings they fill in — so each is tried
/// in turn rather than trusting one.
pub fn window(pts: Duration, start_ms: u32, end_ms: u32, packet: Option<Duration>) -> (Duration, Duration) {
    let start = pts + Duration::from_millis(start_ms as u64);
    if end_ms > start_ms {
        return (start, pts + Duration::from_millis(end_ms as u64));
    }
    match packet {
        Some(d) if d > Duration::ZERO => (start, start + d),
        _ => (start, start + DEFAULT_DURATION),
    }
}

/// The open decoders for every subtitle stream in a container.
///
/// All of them, not just the chosen one. A subtitle stream costs nothing until
/// a caption actually appears, and decoding the lot means switching tracks is
/// instant rather than a re-read of the file from the beginning.
pub struct Decoders {
    entries: Vec<Entry>,
}

struct Entry {
    stream: usize,
    time_base: f64,
    decoder: ffmpeg_next::decoder::Subtitle,
    canvas: (u32, u32),
}

impl Decoders {
    pub fn handles(&self, stream: usize) -> bool {
        self.entries.iter().any(|e| e.stream == stream)
    }

    /// Decode one packet into however many cues it carries.
    pub fn decode(&mut self, stream: usize, packet: &ffmpeg_next::codec::packet::Packet) -> Vec<Cue> {
        let Some(entry) = self.entries.iter_mut().find(|e| e.stream == stream) else {
            return Vec::new();
        };
        let mut subtitle = ffmpeg_next::codec::subtitle::Subtitle::new();
        match entry.decoder.decode(packet, &mut subtitle) {
            Ok(true) => {}
            _ => return Vec::new(),
        }

        let pts = Duration::from_secs_f64(
            packet.pts().or(packet.dts()).unwrap_or(0).max(0) as f64 * entry.time_base,
        );
        let packet_duration = (packet.duration() > 0)
            .then(|| Duration::from_secs_f64(packet.duration() as f64 * entry.time_base));
        let (start, end) = window(pts, subtitle.start(), subtitle.end(), packet_duration);

        let mut lines = Vec::new();
        let mut images = Vec::new();
        for rect in subtitle.rects() {
            match rect {
                ffmpeg_next::codec::subtitle::Rect::Ass(ass) => {
                    lines.extend(text_from_ass(ass.get()))
                }
                ffmpeg_next::codec::subtitle::Rect::Text(text) => {
                    lines.extend(strip_ass_markup(text.get()))
                }
                ffmpeg_next::codec::subtitle::Rect::Bitmap(bitmap) => {
                    // Safety: the rect belongs to `subtitle`, which outlives
                    // this loop, and is a bitmap because the enum says so.
                    if let Some(image) = unsafe { bitmap_to_rgba(bitmap.as_ptr()) } {
                        images.push(image);
                    }
                }
                ffmpeg_next::codec::subtitle::Rect::None(_) => {}
            }
        }

        let payload = if !images.is_empty() {
            Payload::Images(images, entry.canvas)
        } else if !lines.is_empty() {
            Payload::Lines(lines)
        } else {
            // An empty caption is how PGS and DVB clear the screen. Keeping it
            // as an empty cue is what makes the previous one disappear.
            Payload::Lines(Vec::new())
        };
        vec![Cue { track: stream, start, end, payload }]
    }
}

/// Turn one palettised rectangle into straight RGBA.
///
/// FFmpeg gives an 8-bit index plane and a 256-entry BGRA palette. The
/// alternative would be uploading both and doing the lookup in a shader, which
/// is more machinery than a picture that changes every few seconds deserves.
///
/// # Safety
/// `ptr` must be a live `AVSubtitleRect` of bitmap type.
unsafe fn bitmap_to_rgba(ptr: *const ffmpeg_next::sys::AVSubtitleRect) -> Option<Image> {
    unsafe {
        let rect = &*ptr;
        let (w, h) = (rect.w.max(0) as usize, rect.h.max(0) as usize);
        if w == 0 || h == 0 || rect.data[0].is_null() || rect.data[1].is_null() {
            return None;
        }
        let stride = rect.linesize[0].max(0) as usize;
        if stride < w {
            return None;
        }
        let indices = std::slice::from_raw_parts(rect.data[0], stride * h);
        let colours = rect.nb_colors.clamp(0, 256) as usize;
        let palette = std::slice::from_raw_parts(rect.data[1] as *const u32, colours.max(1));

        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let index = indices[y * stride + x] as usize;
                // A palette entry is BGRA packed into a native-endian word.
                let entry = palette.get(index).copied().unwrap_or(0);
                let out = (y * w + x) * 4;
                rgba[out] = (entry >> 16) as u8;
                rgba[out + 1] = (entry >> 8) as u8;
                rgba[out + 2] = entry as u8;
                rgba[out + 3] = (entry >> 24) as u8;
            }
        }
        Some(Image {
            x: rect.x.max(0) as u32,
            y: rect.y.max(0) as u32,
            width: w as u32,
            height: h as u32,
            rgba,
        })
    }
}

/// Metadata lookup that tolerates the tag being absent or empty.
fn tag(stream: &ffmpeg_next::format::stream::Stream, key: &str) -> Option<String> {
    stream
        .metadata()
        .get(key)
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

/// Open every subtitle stream in an already-open container.
///
/// `canvas` is the video's size, used as the fallback coordinate space for
/// picture subtitles that do not declare their own.
pub fn open_embedded(
    input: &ffmpeg_next::format::context::Input,
    canvas: (u32, u32),
) -> (Vec<Track>, Decoders) {
    let mut tracks = Vec::new();
    let mut entries = Vec::new();

    for stream in input.streams() {
        let parameters = stream.parameters();
        if parameters.medium() != ffmpeg_next::media::Type::Subtitle {
            continue;
        }
        let Ok(context) = ffmpeg_next::codec::context::Context::from_parameters(parameters) else {
            continue;
        };
        let codec = context.id().name().to_string();
        let Some(kind) = Kind::of(&codec) else { continue };
        let Ok(decoder) = context.decoder().subtitle() else { continue };

        // A picture subtitle is positioned in the canvas it was authored for,
        // which is only the video's size by coincidence. The codec parameters
        // carry it when the stream declared one.
        // Safety: reading two ints from parameters we still hold.
        let declared = unsafe {
            let raw = stream.parameters().as_ptr();
            ((*raw).width.max(0) as u32, (*raw).height.max(0) as u32)
        };
        let canvas = if declared.0 > 0 && declared.1 > 0 { declared } else { canvas };

        tracks.push(Track {
            index: stream.index(),
            kind,
            codec,
            language: tag(&stream, "language"),
            title: tag(&stream, "title"),
            external: None,
        });
        entries.push(Entry {
            stream: stream.index(),
            time_base: f64::from(stream.time_base()),
            decoder,
            canvas,
        });
    }
    (tracks, Decoders { entries })
}

/// Read a subtitle file sitting beside the video, in full.
///
/// Demuxed with libavformat rather than parsed here, which is why SRT, WebVTT,
/// ASS and the rest cost one code path between them. They are small enough
/// that reading the whole file up front is simpler than streaming it, and it
/// means a track is complete the moment it is switched to.
///
/// `index` is the identifier this track gets; external tracks are numbered
/// above the container's own streams so the two cannot collide.
pub fn load_external(path: &Path, index: usize) -> anyhow::Result<(Track, Vec<Cue>)> {
    use anyhow::Context as _;

    let mut input = ffmpeg_next::format::input(path)
        .map_err(|e| anyhow::anyhow!("opening {}: {e}", path.display()))?;

    let stream = input
        .streams()
        .best(ffmpeg_next::media::Type::Subtitle)
        .context("no subtitle stream")?;
    let stream_index = stream.index();
    let time_base = f64::from(stream.time_base());
    let language = tag(&stream, "language");

    let context = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|e| anyhow::anyhow!("subtitle parameters: {e}"))?;
    let codec = context.id().name().to_string();
    let kind = Kind::of(&codec).context("unsupported subtitle codec")?;
    let decoder = context.decoder().subtitle().map_err(|e| anyhow::anyhow!("decoder: {e}"))?;

    let mut decoders = Decoders {
        entries: vec![Entry { stream: stream_index, time_base, decoder, canvas: (0, 0) }],
    };

    let mut cues = Vec::new();
    for (stream, packet) in input.packets() {
        if stream.index() != stream_index {
            continue;
        }
        for mut cue in decoders.decode(stream_index, &packet) {
            cue.track = index;
            cues.push(cue);
        }
    }

    let track = Track {
        index,
        kind,
        codec,
        language: language.or_else(|| language_from_name(path)),
        title: None,
        external: Some(path.to_path_buf()),
    };
    Ok((track, cues))
}

/// Read the language out of a `film.en.srt` style name.
fn language_from_name(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_string_lossy().to_lowercase();
    let suffix = stem.rsplit('.').next()?;
    // Two- and three-letter codes only; anything longer is part of the title.
    (suffix.len() == 2 || suffix.len() == 3)
        .then(|| suffix.to_string())
        .filter(|s| s.chars().all(|c| c.is_ascii_alphabetic()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialogue_text_survives_its_own_commas() {
        let line = "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Well, hello there";
        assert_eq!(text_from_ass(line), vec!["Well, hello there"]);
    }

    /// What libavcodec actually emits: the event body with a read order in
    /// place of the timestamps, and one field fewer than a script line.
    #[test]
    fn the_decoder_form_has_one_field_fewer() {
        let line = "0,0,Default,,0,0,0,,This is the first caption,\\Non two lines.";
        assert_eq!(text_from_ass(line), vec!["This is the first caption,", "on two lines."]);
    }

    #[test]
    fn a_decoder_line_keeps_commas_in_the_dialogue() {
        let line = "2,0,Default,,0,0,0,,Accented: cafe, naive, jalapeno.";
        assert_eq!(text_from_ass(line), vec!["Accented: cafe, naive, jalapeno."]);
    }

    #[test]
    fn override_blocks_are_dropped_but_the_words_are_not() {
        let line = "Dialogue: 0,0,0,Default,,0,0,0,,{\\pos(190,270)}{\\i1}Softly{\\i0} spoken";
        assert_eq!(text_from_ass(line), vec!["Softly spoken"]);
    }

    #[test]
    fn hard_breaks_become_separate_lines() {
        let line = "Dialogue: 0,0,0,Default,,0,0,0,,First line\\NSecond line";
        assert_eq!(text_from_ass(line), vec!["First line", "Second line"]);
    }

    #[test]
    fn plain_text_without_an_ass_header_is_kept() {
        assert_eq!(text_from_ass("just some words"), vec!["just some words"]);
    }

    #[test]
    fn an_empty_caption_yields_no_lines() {
        let line = "Dialogue: 0,0,0,Default,,0,0,0,,{\\p1}";
        assert!(text_from_ass(line).is_empty());
    }

    #[test]
    fn a_missing_end_time_falls_back_to_the_packet_then_a_default() {
        let pts = Duration::from_secs(10);
        let (s, e) = window(pts, 0, 2000, None);
        assert_eq!((s, e), (pts, pts + Duration::from_secs(2)));

        let (s, e) = window(pts, 0, 0, Some(Duration::from_secs(5)));
        assert_eq!((s, e), (pts, pts + Duration::from_secs(5)));

        let (s, e) = window(pts, 0, 0, None);
        assert_eq!((s, e), (pts, pts + DEFAULT_DURATION));
    }

    #[test]
    fn files_beside_the_video_match_the_stem_and_its_language_suffixes() {
        let dir = std::env::temp_dir().join(format!("show-subs-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        for name in ["film.srt", "film.en.ass", "film2.srt", "other.srt", "film.txt"] {
            let _ = std::fs::write(dir.join(name), "");
        }
        let found = find_beside(&dir.join("film.mkv"));
        let names: Vec<String> =
            found.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(names, vec!["film.en.ass", "film.srt"], "ass is preferred, film2 excluded");
    }

    #[test]
    fn the_folder_listing_includes_names_that_do_not_match() {
        let dir = std::env::temp_dir().join(format!("show-folder-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        for name in ["film.srt", "downloaded-elsewhere.srt", "notes.txt"] {
            let _ = std::fs::write(dir.join(name), "");
        }
        let found = find_in_folder(&dir.join("film.mkv"));
        let names: Vec<String> =
            found.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(names, vec!["film.srt", "downloaded-elsewhere.srt"], "matching name first");
    }

    #[test]
    fn cycling_ends_at_off_and_comes_back_round() {
        let mut subs = Subtitles::new();
        let track = |index| Track {
            index,
            kind: Kind::Text,
            codec: "subrip".into(),
            language: None,
            title: None,
            external: None,
        };
        subs.add_track(track(0));
        subs.add_track(track(1));
        // The first added is selected.
        assert_eq!(subs.selected().map(|t| t.index), Some(0));
        assert_eq!(subs.cycle().map(|t| t.index), Some(1));
        assert!(subs.cycle().is_none(), "off is part of the cycle");
        assert_eq!(subs.cycle().map(|t| t.index), Some(0));
    }

    #[test]
    fn a_file_beside_the_video_outranks_an_embedded_stream() {
        let mut subs = Subtitles::new();
        subs.add_track(Track {
            index: 0,
            kind: Kind::Text,
            codec: "subrip".into(),
            language: None,
            title: None,
            external: None,
        });
        subs.add_track(Track {
            index: 1,
            kind: Kind::Text,
            codec: "ass".into(),
            language: None,
            title: None,
            external: Some(PathBuf::from("/tmp/film.ass")),
        });
        assert_eq!(subs.selected().map(|t| t.index), Some(1));
    }

    #[test]
    fn only_the_selected_track_is_shown() {
        let mut subs = Subtitles::new();
        for index in 0..2 {
            subs.add_track(Track {
                index,
                kind: Kind::Text,
                codec: "subrip".into(),
                language: None,
                title: None,
                external: None,
            });
        }
        for track in 0..2 {
            subs.push(Cue {
                track,
                start: Duration::ZERO,
                end: Duration::from_secs(10),
                payload: Payload::Lines(vec![format!("track {track}")]),
            });
        }
        let shown = |subs: &Subtitles| match &subs.at(Duration::from_secs(1)).unwrap().payload {
            Payload::Lines(l) => l[0].clone(),
            _ => unreachable!(),
        };
        assert_eq!(shown(&subs), "track 0");
        subs.cycle();
        assert_eq!(shown(&subs), "track 1");
    }

    /// Scrubbing back and forth over one stretch re-reads the same packets;
    /// the captions must not accumulate a copy each time.
    #[test]
    fn the_same_caption_arriving_twice_is_kept_once() {
        let mut subs = Subtitles::new();
        subs.add_track(Track {
            index: 0,
            kind: Kind::Text,
            codec: "subrip".into(),
            language: None,
            title: None,
            external: None,
        });
        let cue = || Cue {
            track: 0,
            start: Duration::from_secs(5),
            end: Duration::from_secs(8),
            payload: Payload::Lines(vec!["once".into()]),
        };
        for _ in 0..10 {
            subs.push(cue());
        }
        assert_eq!(subs.cues.len(), 1, "a re-read caption must not pile up");

        // A different moment on the same track is still a different caption.
        subs.push(Cue { start: Duration::from_secs(9), ..cue() });
        assert_eq!(subs.cues.len(), 2);
    }

    #[test]
    fn retiring_keeps_what_is_on_screen() {
        let mut subs = Subtitles::new();
        subs.add_track(Track {
            index: 0,
            kind: Kind::Text,
            codec: "subrip".into(),
            language: None,
            title: None,
            external: None,
        });
        for at in [0u64, 100, 200] {
            subs.push(Cue {
                track: 0,
                start: Duration::from_secs(at),
                end: Duration::from_secs(at + 2),
                payload: Payload::Lines(vec![format!("{at}")]),
            });
        }
        subs.retire(Duration::from_secs(200));
        assert!(subs.at(Duration::from_secs(201)).is_some(), "the current cue must survive");
        assert!(subs.at(Duration::from_secs(1)).is_none(), "the old one is gone");
    }
}
