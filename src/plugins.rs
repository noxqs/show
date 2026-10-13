//! The plugins page: what can be installed, what is, and how much of it.
//!
//! Shares its presentation with the setup panel through `sheet.rs` — the
//! highlight, the shrink-to-fit, the drag — and differs only in what it says
//! and what choosing a row does.
//!
//! Nothing here downloads anything on its own. Every fetch is a key press, the
//! size is shown before it starts, and the licence is shown with it, because a
//! viewer that quietly pulls two gigabytes has misunderstood what it is.

use crate::ml::catalogue::{CATALOGUE, Capability};
use crate::ml::store;
use crate::osd::{self, Overlay, Row, Tone};
use crate::sheet::Sheet;

/// What a row is doing, as far as the page knows.
#[derive(Clone, PartialEq)]
pub enum Status {
    NotInstalled { bytes: u64 },
    Working(String),
    Installed,
    /// Installed, and the one of its kind actually being used. Only the models
    /// that do the looking are alternatives to each other; the rest are simply
    /// installed or not.
    InUse,
    /// Present but no longer what was installed, or a download that did not
    /// match. The text says which.
    Failed(String),
}

impl Status {
    fn tone(&self) -> Tone {
        match self {
            // Not having something is not a fault, it is a choice not yet
            // made — the same rule the setup panel applies to a setting that
            // is off.
            Status::NotInstalled { .. } | Status::Working(_) => Tone::Neutral,
            Status::Installed | Status::InUse => Tone::Good,
            Status::Failed(_) => Tone::Warn,
        }
    }

    fn value(&self) -> String {
        match self {
            Status::NotInstalled { bytes } => {
                format!("NOT INSTALLED  {}", crate::loader::human_bytes(*bytes))
            }
            Status::Working(what) => what.clone(),
            Status::Installed => "INSTALLED".into(),
            Status::InUse => "INSTALLED - IN USE".into(),
            Status::Failed(why) => why.to_uppercase(),
        }
    }
}

/// A slider reading as a bar and a percentage, for the fractions on this page.
fn threshold_bar(value: f32) -> String {
    const WIDTH: usize = 10;
    let filled = (value.clamp(0.0, 1.0) * WIDTH as f32).round() as usize;
    let bar: String = (0..WIDTH).map(|i| if i < filled { '=' } else { '-' }).collect();
    format!("[{bar}] {:>3}%", (value * 100.0).round() as i32)
}

/// A progress bar of the same shape the rest of the program draws.
pub fn bar(fraction: f32) -> String {
    const WIDTH: usize = 12;
    let filled = (fraction.clamp(0.0, 1.0) * WIDTH as f32).round() as usize;
    (0..WIDTH).map(|i| if i < filled { '=' } else { '-' }).collect()
}

pub struct Plugins {
    sheet: Sheet,
    /// One per catalogue entry, in the same order.
    status: Vec<Status>,
    /// Result of the last action, shown at the bottom.
    note: Option<String>,
    /// The looking rate, in times a second. The setting lives in the config;
    /// this is the page's copy of it for drawing.
    look_hz: u32,
    /// Whether outlines are sharpened by the cut-out model. Likewise a copy.
    sharpen: bool,
    /// What counts as a thing. The page's copy, as above.
    limits: crate::ml::vision::Thresholds,
}

/// The most times a second the picture can be looked at.
///
/// Twenty is past what any of these models can do on a processor; it is there
/// so the ceiling is the machine's rather than this number's.
const RATE_MAX: usize = 20;

/// Characters in the rate slider. Fewer than the range, so the bar stays a
/// readable width and each character is worth two.
const RATE_BAR: usize = 10;

/// One step of a threshold slider. Twentieths: fine enough to find the point
/// where a thing appears, coarse enough to cross the useful range in a few
/// presses.
const THRESHOLD_STEP: f32 = 0.05;

/// The settings that are sliders rather than plugins, in the order they are
/// reached below the models.
const SLIDERS: usize = 4;

impl Default for Plugins {
    fn default() -> Self {
        Plugins::new()
    }
}

impl Plugins {
    pub fn new() -> Plugins {
        let mut page =
            Plugins {
                sheet: Sheet::new(),
                status: Vec::new(),
                note: None,
                look_hz: 1,
                sharpen: false,
                limits: crate::ml::vision::Thresholds::default(),
            };
        page.refresh();
        page
    }

    /// What a plugin needs that is not there yet.
    pub fn missing_requirement(&self, id: Capability) -> Option<Capability> {
        crate::ml::catalogue::spec(id)
            .requires
            .iter()
            .copied()
            .find(|needed| !store::installed(*needed))
    }

    /// Re-read what is on disk.
    ///
    /// Digests are not checked here — that reads every installed byte, which
    /// is far too much work for opening a page. `verify` is a deliberate act.
    pub fn refresh(&mut self) {
        let looking = crate::ml::watcher::chosen();
        self.status = CATALOGUE
            .iter()
            .map(|spec| {
                if !store::installed(spec.id) {
                    Status::NotInstalled { bytes: spec.bytes() }
                } else if spec.id == looking {
                    Status::InUse
                } else if spec.id == Capability::Cuda {
                    // Installed is not the same as working here: the card
                    // support can be on disk and still not usable, and the
                    // difference is the whole point of installing it.
                    match crate::ml::runtime::on_card() {
                        None => Status::InUse,
                        Some(why) => Status::Failed(why.to_uppercase()),
                    }
                } else {
                    Status::Installed
                }
            })
            .collect();
        self.sheet.invalidate();
    }

    /// Make the highlighted model the one that does the looking.
    ///
    /// Only the alternatives — the models that all answer the same question —
    /// can be chosen between; everything else is installed or not.
    pub fn use_selected(&mut self) -> Result<Capability, String> {
        let id = self.selected();
        if !id.is_looker() {
            return Err("that one is not a choice — it is used when it is installed".into());
        }
        if !store::installed(id) {
            return Err("not installed yet".into());
        }
        store::set_preferred(id).map_err(|e| format!("{e:#}"))?;
        self.refresh();
        Ok(id)
    }

    /// The highlighted model, or `None` on the row that is not one.
    pub fn selected_model(&self) -> Option<Capability> {
        CATALOGUE.get(self.sheet.selected).map(|s| s.id)
    }

    pub fn selected(&self) -> Capability {
        CATALOGUE[self.sheet.selected.min(CATALOGUE.len() - 1)].id
    }

    /// Whether the highlight is on the looking-rate row.
    pub fn on_rate(&self) -> bool {
        self.sheet.selected == CATALOGUE.len()
    }

    /// Whether the highlight is on the sharpening row.
    pub fn on_sharpen(&self) -> bool {
        self.sheet.selected == CATALOGUE.len() + 1
    }

    /// Whether the highlight is on the confidence slider.
    pub fn on_confidence(&self) -> bool {
        self.sheet.selected == CATALOGUE.len() + 2
    }

    /// Whether the highlight is on the overlap slider.
    pub fn on_overlap(&self) -> bool {
        self.sheet.selected == CATALOGUE.len() + 3
    }

    pub fn move_selection(&mut self, delta: isize) {
        // One past the models: the rate is a setting rather than a plugin, but
        // it belongs to them, so it lives on the same page and is reached the
        // same way.
        self.sheet.move_selection(delta, CATALOGUE.len() + SLIDERS);
    }

    /// Change whichever setting is highlighted.
    pub fn adjust(&mut self, steps: i32, config: &mut crate::config::Config) {
        if self.on_rate() {
            config.look_hz = (config.look_hz as i32 + steps).clamp(1, RATE_MAX as i32) as u32;
            self.look_hz = config.look_hz;
            self.note(format!("looking {} times a second", config.look_hz));
            return;
        }
        if self.on_sharpen() {
            if !store::installed(Capability::PointSegment) {
                self.note("install cut out first — it is the model that does the sharpening");
                return;
            }
            config.sharpen = steps > 0;
            self.sharpen = config.sharpen;
            self.note(if config.sharpen {
                "outlines sharpened by the cut-out model"
            } else {
                "outlines as the detector draws them"
            });
            return;
        }
        if self.on_confidence() {
            config.min_confidence = (config.min_confidence
                + steps as f32 * THRESHOLD_STEP)
                .clamp(0.05, 0.95);
            self.limits =
                crate::ml::vision::Thresholds::new(config.min_confidence, config.duplicate_overlap);
            self.note(format!(
                "drawing what the model is at least {}% sure of",
                (config.min_confidence * 100.0).round() as i32
            ));
            return;
        }
        if self.on_overlap() {
            config.duplicate_overlap = (config.duplicate_overlap
                + steps as f32 * THRESHOLD_STEP)
                .clamp(0.1, 0.95);
            self.limits =
                crate::ml::vision::Thresholds::new(config.min_confidence, config.duplicate_overlap);
            self.note(format!(
                "two of a kind may overlap by {}% before the weaker is dropped",
                (config.duplicate_overlap * 100.0).round() as i32
            ));
        }
    }


    pub fn zoom(&mut self, steps: f32, win: (f32, f32)) {
        self.sheet.zoom(steps, win);
    }

    pub fn pan(&mut self, dx: i32, dy: i32) {
        self.sheet.pan(dx, dy);
    }

    pub fn offset(&self) -> (i32, i32) {
        self.sheet.offset()
    }

    /// Whether an install is in flight, so the page knows not to start another.
    pub fn busy(&self) -> bool {
        self.status.iter().any(|s| matches!(s, Status::Working(_)))
    }

    pub fn set_status(&mut self, id: Capability, status: Status) {
        if let Some(at) = CATALOGUE.iter().position(|s| s.id == id) {
            if self.status.get(at) != Some(&status) {
                self.status[at] = status;
                self.sheet.invalidate();
            }
        }
    }

    pub fn note(&mut self, text: impl Into<String>) {
        self.note = Some(text.into());
        self.sheet.invalidate();
    }

    /// Tell the page what the settings say, since it does not own them.
    pub fn set_settings(
        &mut self,
        hz: u32,
        sharpen: bool,
        limits: crate::ml::vision::Thresholds,
    ) {
        if self.look_hz != hz || self.sharpen != sharpen || self.limits != limits {
            self.look_hz = hz;
            self.sharpen = sharpen;
            self.limits = limits;
            self.sheet.invalidate();
        }
    }

    pub fn rows(&self) -> Vec<Row> {
        let mut rows = vec![Row::good("SHOW - PLUGINS", crate::version_short().to_uppercase())];

        rows.push(Row::plain(""));
        rows.push(Row::dim("MODELS AND RUNTIME", ""));
        for (i, spec) in CATALOGUE.iter().enumerate() {
            let status = self.status.get(i).cloned().unwrap_or(Status::Installed);
            let row = status.tone().row(spec.name, status.value());
            rows.push(row.selected(i == self.sheet.selected));
        }

        // How often the picture is looked at. A slider rather than a number
        // to type, and on this page rather than the settings one, because what
        // it costs depends entirely on which model above it is in use.
        let filled = (self.look_hz as usize).div_ceil(RATE_MAX / RATE_BAR).min(RATE_BAR);
        let bar: String = (0..RATE_BAR).map(|i| if i < filled { '=' } else { '-' }).collect();
        rows.push(
            Row::dim("HOW OFTEN TO LOOK  < >", format!("[{bar}] {} HZ", self.look_hz))
                .selected(self.on_rate()),
        );
        rows.push(
            Row::dim(
                "SHARPEN OUTLINES  < >",
                if self.sharpen { "ON  (CUT OUT MODEL)" } else { "OFF (AS DETECTED)" },
            )
            .selected(self.on_sharpen()),
        );

        // The two numbers that decide what counts as a thing. They were
        // constants, which made them arguments nobody could reach: everything
        // far off or half-hidden was dropped and the only way to see it was to
        // edit and rebuild.
        rows.push(
            Row::dim("HOW SURE BEFORE DRAWING  < >", threshold_bar(self.limits.confidence))
                .selected(self.on_confidence()),
        );
        rows.push(
            Row::dim("OVERLAP BEFORE A REPEAT  < >", threshold_bar(self.limits.duplicate))
                .selected(self.on_overlap()),
        );

        // What the highlighted row actually is, so a choice is informed before
        // it is made rather than explained after.
        if let Some(spec) = CATALOGUE.get(self.sheet.selected) {
            rows.push(Row::plain(""));
            rows.push(Row::dim("VERSION", spec.version));
            rows.push(Row::dim("LICENCE", spec.licence));
            // Said before it is tried, not after it fails.
            if let Some(needed) = self.missing_requirement(spec.id) {
                rows.push(Row::bad("NEEDS FIRST", needed.id().to_uppercase()));
            }
            // Plain, not a value: the value column is capped so one long entry
            // cannot stretch the page, and a sentence hits that cap. A row
            // with no value gets the whole width.
            rows.push(Row::plain(spec.what.to_uppercase()));
        }

        rows.push(Row::plain(""));
        let used: u64 = CATALOGUE.iter().map(|s| store::disk_used(s.id)).sum();
        rows.push(Row::dim(
            "ON DISK",
            match store::root() {
                Some(root) => format!("{}  {}", crate::loader::human_bytes(used), root.display()),
                None => "NO DATA DIRECTORY".to_string(),
            },
        ));
        if let Some(note) = &self.note {
            rows.push(Row::plain(note.to_uppercase()));
        }

        rows.push(Row::plain(""));
        // What enter does depends on the row, so it says which.
        let enter = match self.selected_model() {
            None => "LEFT AND RIGHT CHANGE IT",
            Some(id) if !store::installed(id) => "ENTER DOWNLOAD",
            Some(id) if id.is_looker() => "ENTER USE THIS ONE",
            _ => "ENTER DOWNLOAD",
        };
        rows.push(Row::dim(enter, "DEL UNINSTALL"));
        rows.push(Row::dim("V VERIFY", "F8 OR ESC CLOSE"));
        rows
    }

    pub fn ensure_overlay(&mut self, win: (f32, f32)) {
        let rows = self.rows();
        self.sheet.ensure(&rows, win);
    }

    pub fn cached(&self) -> Option<&Overlay> {
        self.sheet.cached()
    }

}

/// The whole page as it would look, for `--dump-ui`.
pub fn specimen(win: (f32, f32)) -> Overlay {
    let page = Plugins::new();
    let rows = page.rows();
    // Rasterised directly rather than through the sheet, so the specimen
    // borrows nothing and the caller can keep it.
    osd::render_menu(&rows, osd::SCALE_PANEL, (win.0 * 0.96) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_exists_for_every_catalogue_entry() {
        let page = Plugins::new();
        assert_eq!(page.status.len(), CATALOGUE.len());
        let rows = page.rows();
        for spec in CATALOGUE {
            assert!(
                rows.iter().any(|r| r.label_is(spec.name)),
                "{} has no row",
                spec.name
            );
        }
    }

    #[test]
    fn not_installed_is_not_a_fault() {
        // The setup panel's rule, applied here: a thing you have not chosen to
        // install is a choice not yet made, not a problem to be flagged.
        assert_eq!(Status::NotInstalled { bytes: 1 }.tone(), Tone::Neutral);
        assert_eq!(Status::Installed.tone(), Tone::Good);
        assert_eq!(Status::Failed("x".into()).tone(), Tone::Warn);
    }

    #[test]
    fn the_licence_and_size_are_shown_before_anything_is_fetched() {
        let page = Plugins::new();
        let rows = page.rows();
        let spec = &CATALOGUE[0];
        assert!(
            rows.iter().any(|r| r.value_text() == spec.licence),
            "the licence is not on the page"
        );
        // And the size, when it is not installed.
        if !store::installed(spec.id) {
            assert!(
                rows.iter().any(|r| r.value_text().contains("NOT INSTALLED")),
                "the size is not shown"
            );
        }
    }

    #[test]
    fn the_selection_wraps_and_stays_in_range() {
        let mut page = Plugins::new();
        // Up from the first row lands on the last of the sliders that follow
        // the models, and none of those is a model.
        page.move_selection(-1);
        assert!(page.on_overlap(), "the last row should be the overlap slider");
        assert_eq!(page.selected_model(), None);
        page.move_selection(-1);
        assert!(page.on_confidence(), "then the confidence slider");
        page.move_selection(-1);
        assert!(page.on_sharpen(), "then the sharpening switch");
        page.move_selection(-1);
        assert!(page.on_rate(), "and the one before that the rate");
        for _ in 0..SLIDERS {
            page.move_selection(1);
        }
        assert_eq!(page.sheet.selected, 0);
        assert!(page.selected_model().is_some());
    }

    #[test]
    fn the_thresholds_can_be_adjusted_and_are_bounded() {
        let mut page = Plugins::new();
        let mut config = crate::config::Config::default();

        page.move_selection(-2);
        assert!(page.on_confidence());
        for _ in 0..40 {
            page.adjust(-1, &mut config);
        }
        assert!(config.min_confidence >= 0.05, "zero would be every anchor the model has");
        for _ in 0..40 {
            page.adjust(1, &mut config);
        }
        assert!(config.min_confidence <= 0.95, "and it must stop below certainty");

        page.move_selection(1);
        assert!(page.on_overlap());
        let before = config.min_confidence;
        for _ in 0..40 {
            page.adjust(1, &mut config);
        }
        assert!(config.duplicate_overlap <= 0.95);
        assert_eq!(config.min_confidence, before, "the wrong slider moved");
    }

    #[test]
    fn the_rate_can_be_adjusted_and_is_bounded() {
        let mut page = Plugins::new();
        let mut config = crate::config::Config::default();
        config.look_hz = 1;

        // Not on the rate row: nothing happens.
        page.adjust(1, &mut config);
        assert_eq!(config.look_hz, 1, "a model row is not a slider");

        page.move_selection(-(SLIDERS as isize));
        assert!(page.on_rate());
        for _ in 0..40 {
            page.adjust(1, &mut config);
        }
        assert_eq!(config.look_hz, RATE_MAX as u32, "it should stop at the ceiling");
        for _ in 0..40 {
            page.adjust(-1, &mut config);
        }
        assert_eq!(config.look_hz, 1, "and at one");
    }

    #[test]
    fn a_working_row_reports_busy_so_a_second_install_cannot_start() {
        let mut page = Plugins::new();
        assert!(!page.busy());
        page.set_status(CATALOGUE[0].id, Status::Working("FETCHING".into()));
        assert!(page.busy());
    }
}
