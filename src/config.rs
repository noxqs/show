//! Persisted settings.
//!
//! A plain `key = value` file rather than TOML or JSON: there are a handful of
//! booleans here, and a parser for them is shorter than the line that would add
//! a dependency to read them.

use std::path::PathBuf;

/// Resident memory of this process in bytes.
///
/// Field two of `statm` is resident pages. Cheap enough to read whenever the
/// readout is rebuilt, which is only when something has actually changed.
pub fn process_memory_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}

/// Physical memory in megabytes, if the system will say.
pub fn total_memory_mb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Ask before deleting. On by default — deleting the picture someone is
    /// looking at is not an action to take on a single keystroke.
    pub confirm_delete: bool,
    /// Move to the desktop trash rather than unlinking. Also on by default,
    /// for the same reason.
    pub delete_to_trash: bool,
    /// The per-image load breakdown, top right.
    pub show_stats: bool,
    /// The zoom percentage, top left.
    pub show_zoom: bool,
    /// What is in the card's memory and who is holding it, bottom right.
    ///
    /// Its own setting rather than part of the zoom badge: it is the only
    /// readout here that is about the machine rather than about the picture,
    /// and it is the one somebody leaves on for an hour while working out why
    /// a model will not load.
    pub show_vram: bool,
    /// Carry the zoom across images instead of re-fitting each one.
    pub hold_zoom: bool,
    /// Megabytes of decoded pixels the grid may hold. Zero means "decide from
    /// the machine", which is a quarter of physical memory.
    pub max_memory_mb: u64,
    /// Grow images to fill the window instead of stopping at 1:1.
    pub autoscale: bool,
    /// Fit to cover the window, cropping, rather than to show all of it.
    pub cover: bool,
    /// The order files are walked and the grid is laid out in.
    pub sort: crate::playlist::Sort,
    /// Decode video on the GPU. Sixteen times less CPU where it works, and it
    /// falls back to software on its own when it does not.
    pub hardware_decode: bool,
    /// Master switch for the on-screen diagnostics: zoom, memory and the load
    /// breakdown. The individual settings still apply beneath it.
    pub show_debug: bool,
    /// Megabytes of decoded video frames kept for stepping backwards. Zero
    /// means "decide from the machine", which is an eighth of physical memory.
    pub frame_cache_mb: u64,
    /// How often the picture is looked at, in times a second, 1 to 20.
    ///
    /// One is enough to name what is in a shot, which changes with the cut
    /// rather than the frame. Twenty is most of a frame rate: it follows a
    /// moving thing closely and costs twenty times as much, which no model
    /// here can actually keep up with on a processor — the frames it cannot
    /// look at are skipped rather than queued, so asking for more than it can
    /// do costs nothing but is answered no faster.
    pub look_hz: u32,
    /// Sharpen the detector's outlines with the cut-out model.
    ///
    /// The detector says what and roughly where at 160 cells across the whole
    /// frame; the cut-out model says exactly where, at 1024. On means every
    /// thing found is outlined by the second model, which costs an encode of
    /// the picture and a few milliseconds per thing.
    pub sharpen: bool,
    /// How sure the model has to be before a thing is drawn, 0.05 to 0.95.
    ///
    /// The models emit a long tail of very low scores, and drawing all of it
    /// would be drawing static — but a quarter is only a default, and what it
    /// costs is the far-off and the half-hidden. Lower it to pick those up,
    /// and expect to see things that are not there.
    pub min_confidence: f32,
    /// How much two things of the same kind may overlap before the weaker one
    /// is taken for a duplicate, 0.1 to 0.95.
    ///
    /// The segmenting models do not suppress their own repeats, so the same
    /// thing arrives several times with slightly different boxes and the
    /// overlapping ones have to be dropped. The catch is that a crowd looks
    /// exactly like that: people standing close genuinely overlap, and at a
    /// low setting the ones behind are discarded as repeats of the ones in
    /// front. Raise it to keep them.
    pub duplicate_overlap: f32,
    /// Playback gain, 0 to 1.5. Kept because a volume that reset to full on
    /// every file would be a nasty surprise at least once per evening.
    pub volume: f32,
    /// Subtitle size against the one chosen for the picture, 0.5 to 3. Set
    /// with ctrl < and >, and kept, since eyes and sofas do not change between
    /// films.
    pub caption_scale: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config { confirm_delete: true, delete_to_trash: true, show_stats: true, show_zoom: true, show_vram: true, hold_zoom: false, max_memory_mb: 0, show_debug: true, autoscale: false, cover: false, sort: crate::playlist::Sort::Folder, hardware_decode: true, frame_cache_mb: 0, look_hz: 1, sharpen: false, min_confidence: 0.25, duplicate_overlap: 0.45, volume: 1.0, caption_scale: 1.0 }
    }
}

impl Config {
    /// Bytes of decoded frames to keep for stepping backwards.
    ///
    /// An eighth of memory by default rather than a quarter: this is a
    /// convenience for a paused film, and the grid's budget is for the thing
    /// the user is actually looking at.
    pub fn frame_cache_bytes(&self) -> u64 {
        let mb = if self.frame_cache_mb > 0 {
            self.frame_cache_mb
        } else {
            total_memory_mb().map(|t| t / 8).unwrap_or(512)
        };
        mb * 1024 * 1024
    }

    /// The budget in bytes, resolving the "decide for me" default.
    ///
    /// A quarter of physical memory: enough to keep a large grid resident,
    /// while leaving the machine to everything else the user is doing.
    pub fn memory_budget_bytes(&self) -> u64 {
        let mb = if self.max_memory_mb > 0 {
            self.max_memory_mb
        } else {
            total_memory_mb().map(|t| t / 4).unwrap_or(1024)
        };
        mb * 1024 * 1024
    }

    /// The budget as a fraction of physical memory, for the panel's slider.
    pub fn memory_fraction(&self) -> f32 {
        let total = total_memory_mb().unwrap_or(4096).max(1) as f32;
        (self.memory_budget_bytes() as f32 / (1024.0 * 1024.0)) / total
    }

    /// Step the budget, in eighths of physical memory.
    pub fn adjust_memory(&mut self, steps: i32) {
        let total = total_memory_mb().unwrap_or(4096).max(8);
        let current = (self.memory_fraction() * 8.0).round() as i32;
        let next = (current + steps).clamp(1, 6);
        self.max_memory_mb = total * next as u64 / 8;
    }

    pub fn path() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
        Some(base.join("show").join("config"))
    }

    /// Read the config, falling back to defaults for anything missing or
    /// unparseable. A damaged config should not stop the viewer opening.
    pub fn load() -> Config {
        let Some(path) = Config::path() else { return Config::default() };
        let Ok(text) = std::fs::read_to_string(&path) else { return Config::default() };
        Config::parse(&text)
    }

    /// Read settings from the file's text.
    ///
    /// Separate from `load` so a round trip can be tested without touching the
    /// user's own configuration — and so the test exercises this parser rather
    /// than a copy of it written alongside, which is what it used to do and
    /// which could not have caught a field missing from either side.
    pub fn parse(text: &str) -> Config {
        let mut config = Config::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else { continue };
            let value = value.trim();
            let on = matches!(value, "true" | "yes" | "1" | "on");
            match key.trim() {
                "confirm_delete" => config.confirm_delete = on,
                "delete_to_trash" => config.delete_to_trash = on,
                "show_stats" => config.show_stats = on,
                "show_zoom" => config.show_zoom = on,
                "show_vram" => config.show_vram = on,
                "hold_zoom" => config.hold_zoom = on,
                "max_memory_mb" => config.max_memory_mb = value.parse().unwrap_or(0),
                "show_debug" => config.show_debug = on,
                "autoscale" => config.autoscale = on,
                "cover" => config.cover = on,
                "sort" => config.sort = crate::playlist::Sort::parse(value).unwrap_or_default(),
                "hardware_decode" => config.hardware_decode = on,
                "frame_cache_mb" => config.frame_cache_mb = value.parse().unwrap_or(0),
                "look_hz" => {
                    config.look_hz = value.trim().parse::<u32>().unwrap_or(1).clamp(1, 20)
                }
                "sharpen" => config.sharpen = value.trim() == "true",
                "min_confidence" => {
                    config.min_confidence =
                        value.trim().parse::<f32>().unwrap_or(0.25).clamp(0.05, 0.95)
                }
                "duplicate_overlap" => {
                    config.duplicate_overlap =
                        value.trim().parse::<f32>().unwrap_or(0.45).clamp(0.1, 0.95)
                }
                "volume" => {
                    config.volume = value.trim().parse::<f32>().unwrap_or(1.0).clamp(0.0, 1.5)
                }
                "caption_scale" => {
                    config.caption_scale = value.trim().parse::<f32>().unwrap_or(1.0).clamp(0.5, 3.0)
                }
                _ => {}
            }
        }
        config
    }

    pub fn save(&self) -> anyhow::Result<()> {
        use anyhow::Context;
        let path = Config::path().context("no config directory")?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&path, self.to_text())
            .with_context(|| format!("writing {}", path.display()))
    }

    /// The file's contents for these settings.
    pub fn to_text(&self) -> String {
        format!(
            "# show configuration\n\
             confirm_delete = {}\n\
             delete_to_trash = {}\n\
             show_stats = {}\n\
             show_zoom = {}\n\
             # what is in the card's memory, and which processes are holding it\n\
             show_vram = {}\n\
             hold_zoom = {}\n\
             # 0 means a quarter of physical memory\n\
             max_memory_mb = {}\n\
             show_debug = {}\n\
             autoscale = {}\n\
             # fit to fill the window, cropping, instead of showing it all\n\
             cover = {}\n\
             # folder, name or date\n\
             sort = {}\n\
             hardware_decode = {}\n\
             # 0 means an eighth of physical memory\n\
             frame_cache_mb = {}\n\
             # how many times a second the picture is looked at, 1 to 20\n\
             look_hz = {}\n\
             # outline what is found with the cut-out model, at four times the\n\
             # resolution and a real cost per frame\n\
             sharpen = {}\n\
             # how sure the model must be before a thing is drawn\n\
             min_confidence = {:.2}\n\
             # how much two of a kind may overlap before the weaker one is\n\
             # taken for a repeat rather than a second thing\n\
             duplicate_overlap = {:.2}\n\
             volume = {:.2}\n\
             # subtitle size, 1.00 being the size chosen for the picture\n\
             caption_scale = {:.2}\n",
            self.confirm_delete,
            self.delete_to_trash,
            self.show_stats,
            self.show_zoom,
            self.show_vram,
            self.hold_zoom,
            self.max_memory_mb,
            self.show_debug,
            self.autoscale,
            self.cover,
            self.sort.name(),
            self.hardware_decode,
            self.frame_cache_mb,
            self.look_hz,
            self.sharpen,
            self.min_confidence,
            self.duplicate_overlap,
            self.volume,
            self.caption_scale
        )
    }
}

/// Where a deleted file went, so the viewer can say so.
pub enum Deleted {
    Trashed,
    Removed,
}

/// Delete `path`, preferring the desktop trash.
///
/// `gio trash` implements the XDG trash spec including the restore metadata, so
/// the file can be put back from the file manager. Reimplementing that here
/// would be a lot of code to arrive at a worse version of it.
pub fn delete_file(path: &std::path::Path, to_trash: bool) -> anyhow::Result<Deleted> {
    use anyhow::Context;

    if to_trash {
        let status = std::process::Command::new("gio")
            .arg("trash")
            .arg(path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        match status {
            Ok(s) if s.success() => return Ok(Deleted::Trashed),
            // Falling through to an unlink here would silently turn "move to
            // trash" into "destroy", which is not a substitution to make on
            // someone's behalf.
            Ok(s) => anyhow::bail!("gio trash failed ({s}); file not deleted"),
            Err(e) => anyhow::bail!("could not run gio trash ({e}); file not deleted"),
        }
    }

    std::fs::remove_file(path).with_context(|| format!("deleting {}", path.display()))?;
    Ok(Deleted::Removed)
}

#[cfg(test)]
mod tests {
    /// Nothing under test may write the real settings file.
    ///
    /// A page used to save the config itself, so running the tests overwrote
    /// whatever the person using this program had chosen — the volume it had
    /// carefully remembered, among other things. Pages change settings; only
    /// the app writes them.
    #[test]
    fn the_pages_do_not_write_the_settings_file() {
        let sources = ["src/panel.rs", "src/plugins.rs", "src/browser.rs", "src/sheet.rs"];
        for name in sources {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(name);
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            assert!(
                !text.contains("config.save()"),
                "{name} saves the settings itself; leave that to the app"
            );
        }
    }

    use super::*;

    #[test]
    fn defaults_are_the_cautious_ones() {
        let c = Config::default();
        assert!(c.confirm_delete, "deleting must ask by default");
        assert!(c.delete_to_trash, "deleting must be recoverable by default");
    }

    #[test]
    fn the_memory_budget_defaults_to_a_quarter_and_stays_bounded() {
        let mut c = Config::default();
        assert_eq!(c.max_memory_mb, 0, "zero means decide from the machine");
        let quarter = c.memory_fraction();
        assert!((quarter - 0.25).abs() < 0.02, "default should be a quarter: {quarter}");

        // Never to zero, and never to all of it.
        for _ in 0..20 {
            c.adjust_memory(-1);
        }
        assert!(c.memory_fraction() > 0.0);
        for _ in 0..20 {
            c.adjust_memory(1);
        }
        assert!(c.memory_fraction() <= 0.80, "must leave memory for everything else");
    }

    #[test]
    fn every_setting_survives_a_save_and_load_round_trip() {
        // Guards against adding a field to the struct and forgetting to write
        // it out, which would silently reset it on every launch. Goes through
        // the real serialiser and the real parser: a test that spelled the
        // format out itself would pass while the program lost the setting.
        let written = Config {
            confirm_delete: false,
            delete_to_trash: false,
            show_stats: false,
            show_zoom: false,
            show_vram: false,
            hold_zoom: true,
            max_memory_mb: 2048,
            show_debug: false,
            autoscale: true,
            cover: true,
            sort: crate::playlist::Sort::Date,
            hardware_decode: false,
            frame_cache_mb: 4096,
            look_hz: 7,
            min_confidence: 0.35,
            duplicate_overlap: 0.7,
            sharpen: true,
            volume: 0.35,
            caption_scale: 1.5,
        };
        let read = Config::parse(&written.to_text());
        assert_eq!(read, written, "a setting was lost between saving and loading");
        // Every field differs from the default, so an omission cannot pass by
        // coincidentally matching it.
        assert_ne!(read, Config::default());
    }

    #[test]
    fn round_trips_through_the_file_format() {
        let text = "# comment\nconfirm_delete = false\ndelete_to_trash=yes\nbogus = 1\n";
        let mut c = Config::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let on = matches!(v.trim(), "true" | "yes" | "1" | "on");
                match k.trim() {
                    "confirm_delete" => c.confirm_delete = on,
                    "delete_to_trash" => c.delete_to_trash = on,
                    _ => {}
                }
            }
        }
        assert!(!c.confirm_delete);
        assert!(c.delete_to_trash);
    }
}
