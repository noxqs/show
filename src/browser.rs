//! The file list shown beside the image in detailed view.
//!
//! A viewer that can only walk one folder is a viewer you have to quit to get
//! out of. This is the way out: folders as well as images, a `..` that always
//! leads back, and the same keyboard the rest of the program uses.
//!
//! It is deliberately not a general file manager. Only directories and images
//! are listed, because everything else is something this program cannot open
//! and would only be offering falsely.

use std::path::{Path, PathBuf};

/// One row of the list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    /// The parent directory. Always first, so going back is always in the same
    /// place whatever the folder contains.
    Parent,
    Folder(PathBuf),
    Image(PathBuf),
}

impl Entry {
    pub fn label(&self) -> String {
        match self {
            Entry::Parent => "..".to_string(),
            Entry::Folder(p) => {
                format!("{}/", p.file_name().unwrap_or_default().to_string_lossy())
            }
            Entry::Image(p) => p.file_name().unwrap_or_default().to_string_lossy().into_owned(),
        }
    }

    pub fn path(&self) -> Option<&Path> {
        match self {
            Entry::Parent => None,
            Entry::Folder(p) | Entry::Image(p) => Some(p),
        }
    }

    pub fn is_image(&self) -> bool {
        matches!(self, Entry::Image(_))
    }
}

/// What the list shows about an image beside its name, read from its header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FileInfo {
    pub size: Option<(u32, u32)>,
    /// Resolutions stored, for a TIFF: one, or a pyramid's worth.
    pub levels: Option<usize>,
}

impl FileInfo {
    /// Read what can be had without decoding: the header, and for a TIFF its
    /// directory of levels. Milliseconds a file, even for a gigapixel map.
    pub fn probe(path: &Path) -> FileInfo {
        use crate::decode::Format;
        let format = std::fs::File::open(path).ok().and_then(|f| {
            // Safety: read-only, and only the first bytes are looked at.
            let map = unsafe { memmap2::Mmap::map(&f) }.ok()?;
            crate::decode::sniff_path(&map, path).map(|format| (format, map))
        });
        match format {
            Some((Format::Tiff, _)) => match crate::tiled::probe(path) {
                Ok(Some(pyramid)) => {
                    FileInfo { size: Some(pyramid.full_size()), levels: Some(pyramid.levels.len()) }
                }
                _ => FileInfo { size: crate::decode::probe_dimensions(path), levels: None },
            },
            Some((format @ (Format::Heif | Format::Avif), map)) => {
                FileInfo { size: crate::heif::dimensions(&map, format), levels: None }
            }
            _ => FileInfo { size: crate::decode::probe_dimensions(path), levels: None },
        }
    }

    /// The columns before the name: size, then levels. Fixed width, so they
    /// line up in the monospaced font; blank where there is nothing known yet.
    pub fn columns(info: Option<&FileInfo>) -> String {
        let size = info.and_then(|i| i.size).map(|(w, h)| format!("{w}x{h}")).unwrap_or_default();
        let levels = info.and_then(|i| i.levels).map(|n| n.to_string()).unwrap_or_default();
        format!("{size:>11} {levels:>2}  ")
    }
}

/// What activating the selected row should do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Activate {
    /// Move into a directory.
    Enter(PathBuf),
    /// Show an image.
    Show(PathBuf),
    /// Nothing to do.
    None,
}

pub struct Browser {
    dir: PathBuf,
    entries: Vec<Entry>,
    selected: usize,
}

impl Browser {
    /// List `dir`, selecting `current` if it is in there.
    pub fn open(dir: &Path, current: Option<&Path>) -> Browser {
        let mut folders: Vec<PathBuf> = Vec::new();
        let mut images: Vec<PathBuf> = Vec::new();

        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            // Hidden entries are noise in a picture folder, and the ones that
            // matter (`..`) are provided explicitly.
            if path.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')) {
                continue;
            }
            if path.is_dir() {
                folders.push(path);
            } else if crate::playlist::is_image_extension(&path) {
                images.push(path);
            }
        }

        let by_name = |a: &PathBuf, b: &PathBuf| {
            crate::playlist::natural_order(
                &a.file_name().unwrap_or_default().to_string_lossy(),
                &b.file_name().unwrap_or_default().to_string_lossy(),
            )
        };
        folders.sort_by(by_name);
        images.sort_by(by_name);

        // Folders before images: navigation above content, so moving around is
        // always at the top of the list rather than buried past the pictures.
        let mut entries = vec![Entry::Parent];
        entries.extend(folders.into_iter().map(Entry::Folder));
        entries.extend(images.into_iter().map(Entry::Image));

        let selected = current
            .and_then(|c| entries.iter().position(|e| e.path() == Some(c)))
            .unwrap_or(0);

        Browser { dir: dir.to_path_buf(), entries, selected }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn selected(&self) -> usize {
        self.selected
    }

    pub fn selected_entry(&self) -> Option<&Entry> {
        self.entries.get(self.selected)
    }

    /// Move the selection, stopping at the ends rather than wrapping: a list
    /// you are reading should not jump from bottom to top under your hand.
    pub fn move_selection(&mut self, delta: isize) {
        if self.entries.is_empty() {
            return;
        }
        let last = self.entries.len() as isize - 1;
        self.selected = (self.selected as isize + delta).clamp(0, last) as usize;
    }

    pub fn activate(&self) -> Activate {
        match self.selected_entry() {
            Some(Entry::Parent) => match self.dir.parent() {
                Some(parent) => Activate::Enter(parent.to_path_buf()),
                // Already at the root; there is nowhere above it.
                None => Activate::None,
            },
            Some(Entry::Folder(p)) => Activate::Enter(p.clone()),
            Some(Entry::Image(p)) => Activate::Show(p.clone()),
            None => Activate::None,
        }
    }

    /// The parent directory, for backspace.
    pub fn parent(&self) -> Option<PathBuf> {
        self.dir.parent().map(|p| p.to_path_buf())
    }

    /// Images in this folder, in list order — the sequence the arrow keys walk
    /// when the browser is closed again.
    pub fn images(&self) -> Vec<PathBuf> {
        self.entries
            .iter()
            .filter_map(|e| match e {
                Entry::Image(p) => Some(p.clone()),
                _ => None,
            })
            .collect()
    }

    /// Which rows to draw, given room for `visible` of them.
    ///
    /// Keeps the selection in view with a margin, so moving down a long folder
    /// scrolls before the cursor reaches the very bottom edge.
    pub fn window(&self, visible: usize) -> std::ops::Range<usize> {
        let total = self.entries.len();
        if visible >= total {
            return 0..total;
        }
        const MARGIN: usize = 2;
        let half = visible / 2;
        let start = self
            .selected
            .saturating_sub(half)
            .min(total.saturating_sub(visible))
            .min(self.selected.saturating_sub(MARGIN.min(self.selected)));
        let start = start.min(total - visible);
        start..(start + visible).min(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("show-browser-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        for f in ["b.jpg", "a.png", "c10.png", "c2.png", "notes.txt"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        std::fs::write(dir.join(".hidden.jpg"), b"x").unwrap();
        dir
    }

    #[test]
    fn lists_parent_then_folders_then_images_and_nothing_else() {
        let dir = scratch("list");
        let b = Browser::open(&dir, None);
        let labels: Vec<String> = b.entries().iter().map(|e| e.label()).collect();

        assert_eq!(labels[0], "..", "the way back is always first");
        assert_eq!(labels[1], "sub/", "folders before images");
        // Images in natural order, and only images.
        assert_eq!(&labels[2..], &["a.png", "b.jpg", "c2.png", "c10.png"]);
        assert!(!labels.iter().any(|l| l.contains("notes")), "non-images must not be offered");
        assert!(!labels.iter().any(|l| l.contains("hidden")), "hidden files are noise here");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn activating_each_kind_does_the_right_thing() {
        let dir = scratch("activate");
        let mut b = Browser::open(&dir, None);

        assert_eq!(b.activate(), Activate::Enter(dir.parent().unwrap().to_path_buf()));
        b.move_selection(1);
        assert_eq!(b.activate(), Activate::Enter(dir.join("sub")));
        b.move_selection(1);
        assert_eq!(b.activate(), Activate::Show(dir.join("a.png")));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn selection_stops_at_the_ends() {
        let dir = scratch("clamp");
        let mut b = Browser::open(&dir, None);
        let last = b.entries().len() - 1;

        b.move_selection(-5);
        assert_eq!(b.selected(), 0, "must not wrap off the top");
        b.move_selection(1000);
        assert_eq!(b.selected(), last, "must not wrap off the bottom");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn opening_selects_the_image_already_being_viewed() {
        let dir = scratch("select");
        let b = Browser::open(&dir, Some(&dir.join("c2.png")));
        assert_eq!(b.selected_entry().unwrap().label(), "c2.png");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_visible_window_always_contains_the_selection() {
        let dir = scratch("window");
        let mut b = Browser::open(&dir, None);
        for _ in 0..b.entries().len() {
            let w = b.window(3);
            assert!(w.contains(&b.selected()), "selection {} not in {w:?}", b.selected());
            assert!(w.len() <= 3);
            b.move_selection(1);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unreadable_folder_still_offers_a_way_out() {
        let b = Browser::open(Path::new("/nonexistent-folder-for-tests"), None);
        assert_eq!(b.entries().len(), 1);
        assert_eq!(b.entries()[0], Entry::Parent);
    }
}


