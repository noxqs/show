//! The in-app setup and status panel.
//!
//! Everything here could be a command-line flag, and is — but a viewer launched
//! from an applications menu has no terminal to print to and no stdin to read
//! from, so a question asked there is a question nobody can answer. The panel
//! is how the program talks to someone who never opened a shell.
//!
//! It is drawn with the same bitmap font as the timing overlay rather than a
//! widget toolkit, which keeps the binary's dependency list at libc.

use crate::config::Config;
use crate::decode::{self, Support};
use crate::install::{self, State};
use crate::osd::{self, Overlay, Row, Tone};

/// Something the user can do from the panel.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Action {
    Install,
    SetDefault,
    ToggleConfirmDelete,
    ToggleDeleteToTrash,
    ToggleStats,
    ToggleZoom,
    ToggleHoldZoom,
    MemoryBudget,
    ToggleDebug,
}


impl Action {
    fn label(self, ctx: &Context) -> (String, String, Tone) {
        // A plain toggle: on reads as active, off reads as ordinary.
        let toggle = |on: bool| {
            (if on { "ON".to_string() } else { "OFF".to_string() },
             if on { Tone::Good } else { Tone::Neutral })
        };
        match self {
            Action::ToggleDebug => {
                let (v, tone) = toggle(ctx.config.show_debug);
                ("DEBUG INFO  (F4)".into(), v, tone)
            }
            Action::ToggleStats => {
                let (v, tone) = toggle(ctx.config.show_stats);
                ("SHOW LOADING STATS".into(), v, tone)
            }
            Action::ToggleZoom => {
                let (v, tone) = toggle(ctx.config.show_zoom);
                ("SHOW ZOOM LEVEL".into(), v, tone)
            }
            Action::MemoryBudget => {
                let fraction = ctx.config.memory_fraction();
                let mb = ctx.config.memory_budget_bytes() / (1024 * 1024);
                // A text slider: the bar shows the share of physical memory,
                // the number shows what that actually amounts to.
                let filled = (fraction * SLIDER_WIDTH as f32).round() as usize;
                let bar: String = (0..SLIDER_WIDTH)
                    .map(|i| if i < filled { '=' } else { '-' })
                    .collect();
                let size = if mb >= 1024 {
                    format!("{:.1} GB", mb as f32 / 1024.0)
                } else {
                    format!("{mb} MB")
                };
                ("MEMORY BUDGET  < >".into(), format!("[{bar}] {size}"), Tone::Neutral)
            }
            Action::ToggleHoldZoom => {
                let (v, tone) = toggle(ctx.config.hold_zoom);
                ("HOLD ZOOM  (F3)".into(), v, tone)
            }
            // These two are safety nets, so off really is worth flagging.
            Action::ToggleConfirmDelete => (
                "CONFIRM BEFORE DELETE".into(),
                if ctx.config.confirm_delete { "ON".into() } else { "OFF - NO PROMPT".into() },
                if ctx.config.confirm_delete { Tone::Good } else { Tone::Warn },
            ),
            Action::ToggleDeleteToTrash => (
                "DELETE TO TRASH".into(),
                if ctx.config.delete_to_trash { "ON".into() } else { "OFF - PERMANENT".into() },
                if ctx.config.delete_to_trash { Tone::Good } else { Tone::Warn },
            ),
            Action::Install => match ctx.install_state {
                State::Installed => ("INSTALLED".into(), ctx.bin.clone(), Tone::Good),
                State::Stale => (
                    "REDEPLOY THIS BUILD".into(),
                    format!("INSTALLED {}", ctx.installed_at.as_deref().unwrap_or("?")),
                    Tone::Warn,
                ),
                State::NotInstalled => {
                    ("INSTALL TO APPLICATIONS MENU".into(), "NOT INSTALLED".into(), Tone::Warn)
                }
                State::Unknown => ("INSTALL".into(), "UNKNOWN".into(), Tone::Neutral),
            },
            Action::SetDefault => {
                let (ours, total) = ctx.defaults;
                if ours == total {
                    ("DEFAULT IMAGE VIEWER".into(), "YES".into(), Tone::Good)
                } else if ours == 0 {
                    ("SET AS DEFAULT IMAGE VIEWER".into(), "NO".into(), Tone::Warn)
                } else {
                    (
                        "SET AS DEFAULT IMAGE VIEWER".into(),
                        format!("{ours} OF {total}"),
                        Tone::Warn,
                    )
                }
            }
        }
    }
}

/// A snapshot of what the panel is describing. Gathered once when the panel
/// opens, and again after an action, because both queries shell out.
struct Context {
    install_state: State,
    bin: String,
    defaults: (usize, usize),
    config: Config,
    running_at: Option<String>,
    installed_at: Option<String>,
}

impl Context {
    fn gather(config: &Config) -> Context {
        let (running, installed) = install::build_times();
        Context {
            install_state: install::state(),
            bin: install::paths()
                .map(|p| p.bin.display().to_string())
                .unwrap_or_else(|| "NO HOME".into()),
            defaults: install::default_viewer_count(),
            config: config.clone(),
            running_at: running.map(install::format_time),
            installed_at: installed.map(install::format_time),
        }
    }
}

pub struct Panel {
    context: Context,
    /// Highlight, text size, drag offset and the rasterised overlay — the
    /// parts every full-screen page has, in `sheet.rs`. `sheet.selected`
    /// indexes `ACTIONS`, not the rows, since headings are not selectable.
    sheet: crate::sheet::Sheet,
    /// Result of the last action, shown at the bottom.
    status: Option<String>,
}

/// Characters in the memory slider's bar.
const SLIDER_WIDTH: usize = 8;

const ACTIONS: [Action; 9] = [
    Action::Install,
    Action::SetDefault,
    Action::ToggleDebug,
    Action::ToggleStats,
    Action::ToggleZoom,
    Action::ToggleHoldZoom,
    Action::MemoryBudget,
    Action::ToggleConfirmDelete,
    Action::ToggleDeleteToTrash,
];

impl Panel {
    pub fn open(config: &Config) -> Panel {
        Panel {
            context: Context::gather(config),
            sheet: crate::sheet::Sheet::new(),
            status: None,
        }
    }

    pub fn move_selection(&mut self, delta: isize) {
        self.sheet.move_selection(delta, ACTIONS.len());
    }

    /// Where the panel sits relative to centred, after dragging.
    pub fn offset(&self) -> (i32, i32) {
        self.sheet.offset()
    }

    /// Wheel: step the text size.
    pub fn zoom(&mut self, steps: f32, win: (f32, f32)) {
        self.sheet.zoom(steps, win);
    }

    pub fn pan(&mut self, dx: i32, dy: i32) {
        self.sheet.pan(dx, dy);
    }

    pub fn adjust(&mut self, steps: i32, config: &mut Config) {
        if ACTIONS[self.sheet.selected] != Action::MemoryBudget {
            return;
        }
        config.adjust_memory(steps);
        self.status = Some("SAVED".to_string());
        self.context = Context::gather(config);
        self.sheet.invalidate();
    }

    /// Run the highlighted action and refresh what we are showing.
    ///
    /// The setting is changed here and written by the caller, immediately: a
    /// toggle that only takes effect after a clean exit is a toggle that will
    /// be reported as broken. Writing it *here* would mean these tests writing
    /// to the real settings file, which is how a test once overwrote the
    /// volume this program had carefully remembered.
    pub fn activate(&mut self, config: &mut Config) {
        self.status = Some(match ACTIONS[self.sheet.selected] {
            // A range, not a switch: enter does nothing, left and right do.
            Action::MemoryBudget => "USE LEFT AND RIGHT".to_string(),
            action @ (Action::ToggleConfirmDelete
            | Action::ToggleDeleteToTrash
            | Action::ToggleStats
            | Action::ToggleZoom
            | Action::ToggleHoldZoom
            | Action::ToggleDebug) => {
                match action {
                    Action::ToggleDebug => config.show_debug = !config.show_debug,
                    Action::ToggleConfirmDelete => config.confirm_delete = !config.confirm_delete,
                    Action::ToggleDeleteToTrash => config.delete_to_trash = !config.delete_to_trash,
                    Action::ToggleStats => config.show_stats = !config.show_stats,
                    Action::ToggleZoom => config.show_zoom = !config.show_zoom,
                    _ => config.hold_zoom = !config.hold_zoom,
                }
                "SAVED".to_string()
            }
            Action::Install => match install::install() {
                Ok(p) => {
                    let mut msg = format!("INSTALLED {}", p.bin.display());
                    if !install::is_on_path(&p.bin) {
                        msg.push_str(" (NOT ON PATH)");
                    }
                    msg
                }
                Err(e) => format!("FAILED: {e}"),
            },
            Action::SetDefault => {
                // Setting a default before installing would point the system at
                // a desktop entry that does not exist yet.
                if install::state() == State::NotInstalled {
                    "INSTALL FIRST".to_string()
                } else {
                    match install::set_as_default() {
                        Ok(n) => format!("DEFAULT FOR {n} IMAGE TYPES"),
                        Err(e) => format!("FAILED: {e}"),
                    }
                }
            }
        });

        self.context = Context::gather(config);
        self.sheet.invalidate();
    }

    /// Rasterise if anything has changed since the last frame.
    ///
    /// The panel grows with the number of formats and settings, and a panel
    /// taller than the screen is a panel with its footer cut off. So the text
    /// size is chosen to fit rather than fixed: large where there is room,
    /// smaller on a short screen.
    pub fn ensure_overlay(&mut self, win: (f32, f32)) {
        let rows = self.rows();
        self.sheet.ensure(&rows, win);
    }

    /// Rasterise now and return it, for callers outside the draw loop.
    pub fn rendered(&mut self, win: (f32, f32)) -> &Overlay {
        self.ensure_overlay(win);
        self.cached().expect("ensure_overlay just ran")
    }

    /// The rasterised panel, valid after [`Panel::ensure_overlay`].
    pub fn cached(&self) -> Option<&Overlay> {
        self.sheet.cached()
    }


    fn rows(&self) -> Vec<Row> {
        let mut rows = vec![
            Row::good("SHOW - SETUP", crate::version_short().to_uppercase()),
            Row::plain(""),
            Row::dim("DESKTOP INTEGRATION", ""),
        ];

        for (i, action) in ACTIONS.iter().enumerate() {
            // The settings live under their own heading, not under desktop
            // integration, so the list gets a break in the middle.
            if *action == Action::ToggleDebug {
                rows.push(Row::plain(""));
                rows.push(Row::dim("ON SCREEN", ""));
            }
            if *action == Action::MemoryBudget {
                rows.push(Row::plain(""));
                rows.push(Row::dim("MEMORY", ""));
            }
            if *action == Action::ToggleConfirmDelete {
                rows.push(Row::plain(""));
                rows.push(Row::dim("DELETING", ""));
            }
            let (label, value, tone) = action.label(&self.context);
            let row = tone.row(label, value);
            rows.push(row.selected(i == self.sheet.selected));

            // The running build's date belongs beside the row that compares
            // against it, not adrift at the end of the section.
            if *action == Action::Install && self.context.install_state == State::Stale {
                rows.push(Row::dim(
                    "THIS BUILD",
                    self.context.running_at.clone().unwrap_or_else(|| "?".into()),
                ));
            }
        }

        if let Some(status) = &self.status {
            rows.push(Row::plain(""));
            rows.push(Row::dim(status.clone(), ""));
        }

        rows.push(Row::plain(""));
        rows.push(Row::dim("KEYS", ""));
        // One row, not the whole list. The bindings have a page of their own,
        // and carrying a second copy here made this page long enough to need
        // shrinking and gave the list two places to drift apart in.
        rows.push(Row::new("H", "EVERY KEYBOARD SHORTCUT", osd::TEXT_COLOUR));

        rows.push(Row::plain(""));
        rows.push(Row::dim("IMAGE FORMATS", ""));

        // Grouped by how each format is provided. Eighteen rows all saying
        // "BUILT IN" is a list to scroll past; three rows is a list to read.
        let mut built_in: Vec<&str> = Vec::new();
        let mut runtime: Vec<Row> = Vec::new();
        let mut missing: Vec<Row> = Vec::new();
        for format in decode::ALL_FORMATS {
            match format.support() {
                // Upper-cased here rather than in `name()`: the panel is set
                // in capitals throughout, and the font gained a lowercase for
                // subtitles, so these stopped being folded and started
                // shouting the odd word in the wrong case.
                Support::Yes(_) => built_in.push(format.name()),
                Support::Runtime(via) => {
                    runtime.push(Row::good(format.name().to_uppercase(), via))
                }
                Support::Missing(why) => {
                    missing.push(Row::bad(format.name().to_uppercase(), why))
                }
                Support::MissingRuntime(why) => {
                    missing.push(Row::bad(format.name().to_uppercase(), why))
                }
            }
        }
        let available = built_in.len() + runtime.len();
        let unavailable = missing.len();

        for (i, chunk) in built_in.chunks(6).enumerate() {
            let label = if i == 0 { "BUILT IN" } else { "" };
            rows.push(Row::good(label, chunk.join("  ").to_uppercase()));
        }
        rows.extend(runtime);
        rows.extend(missing);
        // Worth naming: it is why a JPEG opens twice as fast here as through
        // the Rust decoders, and it is part of the binary, not the machine.
        rows.push(Row::good("JPEG DECODER", "LIBJPEG-TURBO, BUILT IN"));

        // Video too, now that it plays it. Compact: the full list is behind
        // `--video-formats`, and this page is already long.
        let codecs = crate::video::capabilities();
        let playable: Vec<&str> =
            codecs.iter().filter(|(_, ok)| *ok).map(|(name, _)| *name).collect();
        if !playable.is_empty() {
            rows.push(Row::plain(""));
            rows.push(Row::dim("VIDEO", ""));
            for (i, chunk) in playable.chunks(6).enumerate() {
                let label = if i == 0 { "DECODERS" } else { "" };
                rows.push(Row::good(label, chunk.join("  ").to_uppercase()));
            }
            // Sound is loaded at runtime, so whether it is there is a fact
            // about this machine rather than about the build.
            if crate::alsa::available() {
                rows.push(Row::good("SOUND", format!("ALSA  {} HZ STEREO", crate::audio::RATE)));
            } else {
                rows.push(Row::bad("SOUND", "NO LIBASOUND - VIDEO PLAYS SILENT"));
            }
        }

        rows.push(Row::plain(""));
        rows.push(Row::dim(
            format!("{available} IMAGE FORMATS, {unavailable} UNAVAILABLE"),
            format!("{} VIDEO CODECS", playable.len()),
        ));
        rows.push(Row::dim("UP/DOWN SELECT", "LEFT/RIGHT ADJUST"));
        rows.push(Row::dim("WHEEL RESIZES", "DRAG MOVES"));
        rows.push(Row::dim("ENTER ACTIVATE", "F1 OR ESC CLOSE"));
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel() -> Panel {
        Panel {
            context: Context {
                install_state: State::NotInstalled,
                bin: "/home/u/.local/bin/show".into(),
                defaults: (0, 18),
                config: Config::default(),
                running_at: Some("2026-09-05 10:00".into()),
                installed_at: Some("2026-09-04 09:00".into()),
            },
            sheet: crate::sheet::Sheet::new(),
            status: None,
        }
    }

    #[test]
    fn selection_wraps_within_the_action_list() {
        let mut p = panel();
        assert_eq!(p.sheet.selected, 0);
        p.move_selection(-1);
        assert_eq!(p.sheet.selected, ACTIONS.len() - 1, "moving up from the first wraps to the last");
        p.move_selection(1);
        assert_eq!(p.sheet.selected, 0);
    }

    #[test]
    fn every_format_is_accounted_for_somewhere() {
        let rows = panel().rows();
        let all: String = rows
            .iter()
            .map(|r| format!("{} {}", r.label_text(), r.value_text()))
            .collect::<Vec<_>>()
            .join(" ")
            .to_uppercase();
        for format in decode::ALL_FORMATS {
            assert!(
                all.contains(&format.name().to_uppercase()),
                "{} is missing from the panel",
                format.name()
            );
        }
    }

    #[test]
    fn an_ordinary_setting_being_off_is_not_a_warning() {
        let mut p = panel();
        p.context.config.hold_zoom = false;
        let (_, _, tone) = Action::ToggleHoldZoom.label(&p.context);
        assert_eq!(tone, Tone::Neutral, "off is a setting, not a problem");

        // A safety net being off is different.
        p.context.config.delete_to_trash = false;
        let (_, _, tone) = Action::ToggleDeleteToTrash.label(&p.context);
        assert_eq!(tone, Tone::Warn, "permanent deletion should be flagged");
    }

    #[test]
    fn the_wheel_overrides_the_automatic_fit() {
        let win = (1920.0, 1728.0);
        let mut p = panel();
        p.ensure_overlay(win);
        let fitted = p.cached().unwrap().height;

        p.zoom(1.0, win);
        p.ensure_overlay(win);
        assert!(p.cached().unwrap().height > fitted, "wheel up must enlarge past the auto fit");

        p.pan(40, -20);
        assert_eq!(p.offset(), (40, -20));
        // Resizing recentres, so a zoom cannot strand the panel off-screen.
        p.zoom(-1.0, win);
        assert_eq!(p.offset(), (0, 0));
    }

    #[test]
    fn the_panel_shrinks_to_fit_the_screen() {
        let mut tall = panel();
        tall.ensure_overlay((1920.0, 4000.0));
        let big = tall.cached().unwrap().height;

        // A real desktop: must fit. It may not need to shrink at all, which is
        // the better outcome and what collapsing the format list achieved.
        let mut desktop = panel();
        desktop.ensure_overlay((3072.0, 1728.0));
        let fitted = desktop.cached().unwrap().height;
        assert!(fitted as f32 <= 1728.0 * 0.96, "must fit a real screen: {fitted}");
        assert!(fitted <= big, "never larger than the unconstrained layout");

        // A very short screen gets the smallest text we have, even though the
        // panel may still overflow it.
        let mut short = panel();
        short.ensure_overlay((1920.0, 400.0));
        assert!(short.cached().unwrap().height < fitted, "must keep shrinking");
    }

    #[test]
    fn the_memory_slider_shows_a_bar_and_a_size() {
        let p = panel();
        let (label, value, _) = Action::MemoryBudget.label(&p.context);
        assert!(label.contains("MEMORY"), "{label}");
        assert!(value.starts_with('['), "{value}");
        assert_eq!(
            value.chars().filter(|c| *c == '=' || *c == '-').count(),
            SLIDER_WIDTH,
            "the bar must always be the same width: {value}"
        );
        assert!(value.contains("GB") || value.contains("MB"), "{value}");
    }

    #[test]
    fn adjusting_only_applies_to_the_memory_row() {
        let mut p = panel();
        let mut config = Config::default();
        // Selection is on Install, which has no range to adjust.
        p.adjust(1, &mut config);
        assert_eq!(config.max_memory_mb, 0, "a switch row must ignore left/right");

        p.sheet.selected = ACTIONS.iter().position(|a| *a == Action::MemoryBudget).unwrap();
        p.adjust(1, &mut config);
        assert!(config.max_memory_mb > 0, "the memory row must respond");
    }

    #[test]
    fn the_panel_points_at_the_key_page() {
        // The list itself lives in `keys.rs`; what this page owes the reader
        // is the one binding that gets them there.
        let rows = panel().rows();
        assert!(
            rows.iter().any(|r| r.label_is("H")),
            "nothing on the setup page says where the keys are"
        );
    }

    #[test]
    fn a_stale_install_offers_a_redeploy_with_dates() {
        let mut p = panel();
        p.context.install_state = State::Stale;
        let (label, value, tone) = Action::Install.label(&p.context);
        assert_eq!(tone, Tone::Warn);
        assert!(label.contains("REDEPLOY"), "{label}");
        assert!(value.contains("2026-09-04"), "should name when it was installed: {value}");

        let rows = p.rows();
        let this_build = rows.iter().position(|r| r.label_is("THIS BUILD"));
        let deleting = rows.iter().position(|r| r.label_is("DELETING"));
        assert!(this_build.is_some(), "a stale install should show the running build's date");
        assert!(
            this_build < deleting,
            "the build date belongs with the redeploy row, not in the deleting section"
        );
    }

    #[test]
    fn delete_settings_show_their_current_state() {
        let mut p = panel();
        let (_, value, tone) = Action::ToggleConfirmDelete.label(&p.context);
        assert_eq!(value, "ON");
        assert_eq!(tone, Tone::Good);

        p.context.config.delete_to_trash = false;
        let (_, value, tone) = Action::ToggleDeleteToTrash.label(&p.context);
        assert_eq!(value, "OFF - PERMANENT", "permanent deletion must say so");
        assert_eq!(tone, Tone::Warn, "and must not be styled as the safe state");
    }

    #[test]
    fn an_uninstalled_panel_offers_to_install() {
        let p = panel();
        let (label, value, tone) = Action::Install.label(&p.context);
        assert_eq!(tone, Tone::Warn);
        assert!(label.contains("INSTALL"), "{label}");
        assert_eq!(value, "NOT INSTALLED");
    }

    #[test]
    fn a_partially_applied_default_is_not_reported_as_done() {
        let mut p = panel();
        p.context.defaults = (9, 18);
        let (_, value, tone) = Action::SetDefault.label(&p.context);
        assert_eq!(tone, Tone::Warn, "half-applied defaults must not read as complete");
        assert_eq!(value, "9 OF 18");
    }
}
