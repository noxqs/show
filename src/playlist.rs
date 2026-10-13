//! The set of images the arrow keys move through.
//!
//! A viewer launched on one file is nearly always launched from a folder of
//! them, so the folder is the unit of navigation. The list is built once, by
//! extension rather than by reading every file's magic bytes: a directory of
//! several thousand photos should not cost several thousand opens before the
//! first one is on screen. A file whose extension lies still opens correctly
//! when you reach it, because the decoder sniffs content regardless.

use std::path::{Path, PathBuf};

/// Extensions we will list. Deliberately broader than what we can decode today
/// — a file we cannot open yet still deserves a place in the sequence, and it
/// reports why when you land on it.
const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "jpe", "jfif", "png", "gif", "webp", "bmp", "dib", "tif", "tiff", "ico", "tga",
    "icb", "vda", "vst", "qoi", "dds", "hdr", "exr", "pnm", "pbm", "pgm", "ppm", "pam", "ff",
    "avif", "avifs", "heic", "heif", "hif", "jxl", "svg",
    // Video, played rather than shown, but reached the same way.
    "mp4", "m4v", "mkv", "webm", "mov", "avi", "mpg", "mpeg", "ts", "m2ts", "wmv", "flv", "ogv",
];

pub fn is_image_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let lower = e.to_ascii_lowercase();
            IMAGE_EXTENSIONS.contains(&lower.as_str())
        })
        .unwrap_or(false)
}

/// The order the files are walked in, and laid out in the grid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sort {
    /// Folder by folder, then by name within each: the order of the tree.
    #[default]
    Folder,
    /// By file name alone, wherever the file lives.
    Name,
    /// Oldest first, by when the file was created. Where the filesystem does
    /// not record that, when it was last written stands in for it.
    Date,
    /// One image from each folder — its first, second or third by name —
    /// folders in tree order. A folder with fewer images than that has none
    /// to show and is left out. A filter as much as a sort: the rest are set
    /// aside, not dropped.
    NthPerFolder(u8),
}

impl Sort {
    pub fn next(self) -> Sort {
        match self {
            Sort::Folder => Sort::Name,
            Sort::Name => Sort::Date,
            Sort::Date => Sort::NthPerFolder(1),
            Sort::NthPerFolder(n) if n < 3 => Sort::NthPerFolder(n + 1),
            Sort::NthPerFolder(_) => Sort::Folder,
        }
    }

    /// As written in the settings file.
    pub fn name(self) -> &'static str {
        match self {
            Sort::Folder => "folder",
            Sort::Name => "name",
            Sort::Date => "date",
            Sort::NthPerFolder(1) => "first",
            Sort::NthPerFolder(2) => "second",
            Sort::NthPerFolder(_) => "third",
        }
    }

    pub fn parse(text: &str) -> Option<Sort> {
        // Written by one build that briefly meant "the first two or three".
        match text.trim() {
            "first2" => return Some(Sort::NthPerFolder(2)),
            "first3" => return Some(Sort::NthPerFolder(3)),
            _ => {}
        }
        [
            Sort::Folder,
            Sort::Name,
            Sort::Date,
            Sort::NthPerFolder(1),
            Sort::NthPerFolder(2),
            Sort::NthPerFolder(3),
        ]
        .into_iter()
        .find(|s| s.name() == text.trim())
    }

    /// As said on screen.
    pub fn label(self) -> &'static str {
        match self {
            Sort::Folder => "SORT BY FOLDER, THEN NAME",
            Sort::Name => "SORT BY NAME",
            Sort::Date => "SORT BY DATE CREATED",
            Sort::NthPerFolder(1) => "FIRST IMAGE OF EACH FOLDER",
            Sort::NthPerFolder(2) => "SECOND IMAGE OF EACH FOLDER",
            Sort::NthPerFolder(_) => "THIRD IMAGE OF EACH FOLDER",
        }
    }
}

#[derive(Clone)]
pub struct Playlist {
    entries: Vec<PathBuf>,
    index: usize,
    /// Files a filtering sort has set aside, to come back when it is left.
    hidden: Vec<PathBuf>,
}

impl Playlist {
    /// Build from the folder containing `start`, positioned on `start`.
    ///
    /// If the folder cannot be read, the playlist is just the one file, which
    /// keeps the viewer working on paths the user can open but not enumerate.
    pub fn from_file(start: &Path) -> Playlist {
        // A bare file name has an empty parent, which is not a path read_dir
        // accepts; it means the current folder.
        let dir = match start.parent() {
            Some(d) if d.as_os_str().is_empty() => Some(Path::new(".")),
            other => other,
        };
        let mut entries = dir
            .and_then(|dir| std::fs::read_dir(dir).ok())
            .map(|dir| {
                dir.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && is_image_extension(p))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if entries.is_empty() {
            return Playlist { entries: vec![start.to_path_buf()], index: 0, hidden: Vec::new() };
        }

        entries.sort_by(|a, b| natural_order(&file_name(a), &file_name(b)));

        // `start` may have been given as a relative path while the listing is
        // absolute, so match on the file name rather than the whole path.
        // A file the listing left out — an extension it does not know — was
        // still asked for by name, and is already decoding: it joins the list
        // rather than a different image being shown in its place.
        let index = match entries.iter().position(|p| p.file_name() == start.file_name()) {
            Some(i) => i,
            None => {
                let name = file_name(start);
                let i = entries.partition_point(|p| natural_order(&file_name(p), &name).is_lt());
                entries.insert(i, start.to_path_buf());
                i
            }
        };
        Playlist { entries, index, hidden: Vec::new() }
    }

    /// Several files named on the command line: those, in the order given.
    pub fn from_list(files: &[PathBuf]) -> Playlist {
        Playlist { entries: files.to_vec(), index: 0, hidden: Vec::new() }
    }

    /// Every image in a folder. Used when launched with no arguments at all,
    /// e.g. from the applications menu.
    pub fn from_dir(dir: &Path) -> Playlist {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && is_image_extension(p))
            .collect();
        entries.sort_by(|a, b| natural_order(&file_name(a), &file_name(b)));
        Playlist { entries, index: 0, hidden: Vec::new() }
    }

    /// Every image in a folder and all the folders beneath it.
    ///
    /// Each folder's own images come first, then its subfolders in order, so
    /// the grid reads as the tree does. Hidden folders are skipped, and
    /// symlinked ones are not followed: a link back up the tree would
    /// otherwise never end.
    pub fn from_dir_recursive(root: &Path) -> Playlist {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(read) = std::fs::read_dir(dir) else { return };
            let (mut files, mut dirs) = (Vec::new(), Vec::new());
            for entry in read.flatten() {
                let Ok(kind) = entry.file_type() else { continue };
                let path = entry.path();
                if kind.is_dir() {
                    if !file_name(&path).starts_with('.') {
                        dirs.push(path);
                    }
                } else if path.is_file() && is_image_extension(&path) {
                    files.push(path);
                }
            }
            files.sort_by(|a, b| natural_order(&file_name(a), &file_name(b)));
            dirs.sort_by(|a, b| natural_order(&file_name(a), &file_name(b)));
            out.extend(files);
            for d in dirs {
                walk(&d, out);
            }
        }
        let mut entries = Vec::new();
        walk(root, &mut entries);
        Playlist { entries, index: 0, hidden: Vec::new() }
    }

    /// Reorder, staying on the same file — or, if the new order leaves it
    /// out, on the one standing in for its folder.
    pub fn sort(&mut self, by: Sort) {
        let current = self.current().map(|p| p.to_path_buf());
        self.entries.append(&mut self.hidden);
        // Folder order first in every mode: the sorts below are stable, so it
        // is what decides between two files with the same name or time.
        self.entries.sort_by(|a, b| folder_order(a, b));
        match by {
            Sort::Folder => {}
            Sort::Name => self.entries.sort_by(|a, b| natural_order(&file_name(a), &file_name(b))),
            // Cached: one stat per file, not one per comparison.
            Sort::Date => self.entries.sort_by_cached_key(|p| created(p)),
            Sort::NthPerFolder(n) => {
                // Folder order already puts each folder's files together, by
                // name within each, so keep the nth of each run.
                let mut kept: Vec<PathBuf> = Vec::new();
                let (mut folder, mut run) = (None::<PathBuf>, 0u8);
                for p in std::mem::take(&mut self.entries) {
                    let here = p.parent().map(|d| d.to_path_buf());
                    run = if here == folder { run.saturating_add(1) } else { 1 };
                    folder = here;
                    if run == n {
                        kept.push(p);
                    } else {
                        self.hidden.push(p);
                    }
                }
                self.entries = kept;
            }
        }
        self.index = 0;
        if let Some(current) = current {
            self.index = self.position_near(&current).unwrap_or(0);
        }
    }

    /// Take a file out wherever it is, keeping the current one current.
    /// Returns whether it was there.
    pub fn remove(&mut self, path: &Path) -> bool {
        self.hidden.retain(|p| p != path);
        let Some(at) = self.entries.iter().position(|p| p == path) else { return false };
        self.entries.remove(at);
        if at < self.index || self.index >= self.entries.len() {
            self.index = self.index.saturating_sub(1);
        }
        true
    }

    /// Where `path` is, or failing that the first file from its folder.
    pub fn position_near(&self, path: &Path) -> Option<usize> {
        self.entries
            .iter()
            .position(|p| p == path)
            .or_else(|| self.entries.iter().position(|p| p.parent() == path.parent()))
    }

    /// Put the selection on `path`, if it is in the list.
    pub fn select(&mut self, path: &Path) -> bool {
        match self.entries.iter().position(|p| p == path) {
            Some(i) => self.go_to(i),
            None => false,
        }
    }

    /// Where to start when given nothing: the user's Pictures folder if the
    /// desktop defines one, else the home directory, else wherever we are.
    pub fn default_start_dir() -> PathBuf {
        let home = std::env::var_os("HOME").map(PathBuf::from);

        if let Some(home) = &home {
            // XDG records the localised Pictures folder here; the value is a
            // shell-style string like "$HOME/Pictures".
            if let Ok(conf) = std::fs::read_to_string(home.join(".config/user-dirs.dirs")) {
                for line in conf.lines() {
                    if let Some(rest) = line.trim().strip_prefix("XDG_PICTURES_DIR=") {
                        let raw = rest.trim().trim_matches('"');
                        let expanded = raw.replace("$HOME", &home.to_string_lossy());
                        let path = PathBuf::from(expanded);
                        if path.is_dir() {
                            return path;
                        }
                    }
                }
            }
            let pictures = home.join("Pictures");
            if pictures.is_dir() {
                return pictures;
            }
            return home.clone();
        }
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }

    /// `None` when the folder held no images at all.
    pub fn current(&self) -> Option<&Path> {
        self.entries.get(self.index).map(|p| p.as_path())
    }

    pub fn at(&self, index: usize) -> Option<&Path> {
        self.entries.get(index).map(|p| p.as_path())
    }

    /// Jump straight to an index, as the grid does on selection.
    pub fn go_to(&mut self, index: usize) -> bool {
        if index >= self.entries.len() || index == self.index {
            return false;
        }
        self.index = index;
        true
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn position(&self) -> usize {
        self.index
    }

    /// Step by `delta`, wrapping at both ends. Returns false if there is
    /// nowhere to go, so the caller can skip a pointless reload.
    pub fn step(&mut self, delta: isize) -> bool {
        if self.entries.len() < 2 {
            return false;
        }
        let n = self.entries.len() as isize;
        self.index = (((self.index as isize + delta) % n + n) % n) as usize;
        true
    }

    /// Drop the current entry, staying at the same position so the next image
    /// slides into view. Returns false when nothing remains.
    pub fn remove_current(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        self.entries.remove(self.index);
        if self.entries.is_empty() {
            self.index = 0;
            return false;
        }
        // Deleting the last entry leaves the cursor past the end.
        if self.index >= self.entries.len() {
            self.index = self.entries.len() - 1;
        }
        true
    }

    pub fn go_to_end(&mut self, last: bool) -> bool {
        let target = if last { self.entries.len() - 1 } else { 0 };
        let moved = target != self.index;
        self.index = target;
        moved
    }
}

/// When the file was created, or failing that last written.
fn created(p: &Path) -> std::time::SystemTime {
    std::fs::metadata(p)
        .and_then(|m| m.created().or_else(|_| m.modified()))
        .unwrap_or(std::time::UNIX_EPOCH)
}

/// Tree order: a folder's own files before its subfolders', each level in
/// natural order.
fn folder_order(a: &Path, b: &Path) -> std::cmp::Ordering {
    let parts = |p: &Path| -> Vec<String> {
        p.parent()
            .map(|d| d.iter().map(|c| c.to_string_lossy().into_owned()).collect())
            .unwrap_or_default()
    };
    let (pa, pb) = (parts(a), parts(b));
    for (x, y) in pa.iter().zip(&pb) {
        match natural_order(x, y) {
            std::cmp::Ordering::Equal => {}
            other => return other,
        }
    }
    pa.len().cmp(&pb.len()).then_with(|| natural_order(&file_name(a), &file_name(b)))
}

fn file_name(p: &Path) -> String {
    p.file_name().unwrap_or_default().to_string_lossy().into_owned()
}

/// Compare names so `img2` sorts before `img10`.
///
/// Plain lexicographic order puts `img10` first, which is wrong every time a
/// camera or a export script numbers its output.
pub(crate) fn natural_order(a: &str, b: &str) -> std::cmp::Ordering {
    let (mut ai, mut bi) = (a.chars().peekable(), b.chars().peekable());
    loop {
        match (ai.peek().copied(), bi.peek().copied()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(x), Some(y)) if x.is_ascii_digit() && y.is_ascii_digit() => {
                let take_number = |it: &mut std::iter::Peekable<std::str::Chars>| {
                    let mut n = String::new();
                    while it.peek().is_some_and(|c| c.is_ascii_digit()) {
                        n.push(it.next().unwrap());
                    }
                    // Trimming leading zeros keeps `007` and `7` comparable.
                    n.trim_start_matches('0').to_string()
                };
                let (x, y) = (take_number(&mut ai), take_number(&mut bi));
                match x.len().cmp(&y.len()).then_with(|| x.cmp(&y)) {
                    std::cmp::Ordering::Equal => {}
                    other => return other,
                }
            }
            (Some(x), Some(y)) => {
                ai.next();
                bi.next();
                let (lx, ly) = (x.to_ascii_lowercase(), y.to_ascii_lowercase());
                if lx != ly {
                    return lx.cmp(&ly);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_a_file_keeps_the_current_one_current() {
        let mut list = Playlist::from_list(&["/a", "/b", "/c", "/d"].map(PathBuf::from));
        list.go_to(2);
        assert!(list.remove(Path::new("/a")));
        assert_eq!(list.current(), Some(Path::new("/c")));
        assert!(list.remove(Path::new("/d")));
        assert_eq!(list.current(), Some(Path::new("/c")));
        assert!(list.remove(Path::new("/c")));
        assert_eq!(list.current(), Some(Path::new("/b")));
        assert!(!list.remove(Path::new("/zz")));
    }

    #[test]
    fn each_sort_orders_by_what_it_says() {
        let paths = ["/r/b/a2.jpg", "/r/a1.jpg", "/r/b/c/a0.jpg", "/r/z.jpg"];
        let mut list = Playlist::from_list(&paths.map(PathBuf::from));
        list.go_to(1);
        let names = |l: &Playlist| (0..l.len()).map(|i| l.at(i).unwrap().to_string_lossy().into_owned()).collect::<Vec<_>>();

        list.sort(Sort::Folder);
        assert_eq!(names(&list), ["/r/a1.jpg", "/r/z.jpg", "/r/b/a2.jpg", "/r/b/c/a0.jpg"]);
        assert_eq!(list.current(), Some(Path::new("/r/a1.jpg")), "stays on the same file");

        list.sort(Sort::Name);
        assert_eq!(names(&list), ["/r/b/c/a0.jpg", "/r/a1.jpg", "/r/b/a2.jpg", "/r/z.jpg"]);
        assert_eq!(list.current(), Some(Path::new("/r/a1.jpg")));

        // On z.jpg, which the filter sets aside: its folder's first stands in.
        list.go_to(3);
        list.sort(Sort::NthPerFolder(1));
        assert_eq!(names(&list), ["/r/a1.jpg", "/r/b/a2.jpg", "/r/b/c/a0.jpg"]);
        assert_eq!(list.current(), Some(Path::new("/r/a1.jpg")));

        // And leaving it brings everything back.
        list.sort(Sort::Folder);
        assert_eq!(list.len(), 4);

        // The second of each folder: only the root has two; the rest drop out.
        list.sort(Sort::NthPerFolder(2));
        assert_eq!(names(&list), ["/r/z.jpg"]);
        let mut three = Playlist::from_list(&["/f/1", "/f/2", "/f/3", "/f/4", "/g/1", "/g/2", "/g/3"].map(PathBuf::from));
        three.sort(Sort::NthPerFolder(3));
        assert_eq!(names(&three), ["/f/3", "/g/3"]);
        // And back: nothing was lost on the way.
        three.sort(Sort::Folder);
        assert_eq!(three.len(), 7);
    }

    #[test]
    fn subfolders_follow_their_parents_images_in_order() {
        let root = std::env::temp_dir().join(format!("show-recursive-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for dir in ["b10", "b2/deep", ".hidden"] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        for file in ["z.jpg", "a.png", "notes.txt", "b10/x.jpg", "b2/y.jpg", "b2/deep/w.jpg", ".hidden/h.jpg"] {
            std::fs::write(root.join(file), b"").unwrap();
        }

        let list = Playlist::from_dir_recursive(&root);
        let names: Vec<String> = (0..list.len())
            .map(|i| list.at(i).unwrap().strip_prefix(&root).unwrap().to_string_lossy().into_owned())
            .collect();
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(names, ["a.png", "z.jpg", "b2/y.jpg", "b2/deep/w.jpg", "b10/x.jpg"]);
    }

    #[test]
    fn numbers_sort_by_value_not_by_text() {
        let mut names = vec!["img10.jpg", "img2.jpg", "img1.jpg", "img20.jpg"];
        names.sort_by(|a, b| natural_order(a, b));
        assert_eq!(names, ["img1.jpg", "img2.jpg", "img10.jpg", "img20.jpg"]);
    }

    #[test]
    fn leading_zeros_do_not_change_the_order() {
        let mut names = vec!["p007.png", "p10.png", "p0009.png"];
        names.sort_by(|a, b| natural_order(a, b));
        assert_eq!(names, ["p007.png", "p0009.png", "p10.png"]);
    }

    #[test]
    fn an_empty_folder_yields_no_current_image() {
        let p = Playlist::from_dir(Path::new("/nonexistent-directory-for-tests"));
        assert!(p.is_empty());
        assert!(p.current().is_none());
    }

    #[test]
    fn stepping_wraps_at_both_ends() {
        let files: Vec<PathBuf> = ["a.png", "b.png", "c.png"].iter().map(PathBuf::from).collect();
        let mut p = Playlist::from_list(&files);
        assert_eq!(p.position(), 0);
        assert!(p.current().is_some());
        assert!(p.step(-1));
        assert_eq!(p.position(), 2, "stepping back from the first wraps to the last");
        assert!(p.step(1));
        assert_eq!(p.position(), 0);
        assert!(p.step(5));
        assert_eq!(p.position(), 2, "a step larger than the list still lands in range");
    }

    #[test]
    fn removing_advances_to_the_next_image() {
        let files: Vec<PathBuf> = ["a.png", "b.png", "c.png"].iter().map(PathBuf::from).collect();
        let mut p = Playlist::from_list(&files);
        p.step(1);
        assert_eq!(p.current().unwrap().file_name().unwrap(), "b.png");
        assert!(p.remove_current());
        assert_eq!(p.current().unwrap().file_name().unwrap(), "c.png", "the next one slides in");
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn removing_the_last_entry_steps_back() {
        let files: Vec<PathBuf> = ["a.png", "b.png"].iter().map(PathBuf::from).collect();
        let mut p = Playlist::from_list(&files);
        p.go_to_end(true);
        assert!(p.remove_current());
        assert_eq!(p.current().unwrap().file_name().unwrap(), "a.png");
    }

    #[test]
    fn removing_the_only_entry_empties_the_list() {
        let mut p = Playlist::from_list(&[PathBuf::from("only.png")]);
        assert!(!p.remove_current());
        assert!(p.is_empty());
        assert!(p.current().is_none());
    }

    #[test]
    fn a_single_image_has_nowhere_to_step() {
        let mut p = Playlist::from_list(&[PathBuf::from("only.png")]);
        assert!(!p.step(1));
        assert_eq!(p.position(), 0);
    }

    #[test]
    fn extensions_are_matched_case_insensitively() {
        assert!(is_image_extension(Path::new("IMG_0001.HEIC")));
        assert!(is_image_extension(Path::new("a.JpG")));
        assert!(!is_image_extension(Path::new("notes.txt")));
        assert!(!is_image_extension(Path::new("noextension")));
    }
}
