//! Grid view: many images at once.
//!
//! The whole grid is composited on the CPU into one window-sized RGBA buffer
//! and uploaded as a single texture. The alternative — a quad and a texture per
//! cell — means hundreds of bind groups and draw calls to show pictures that
//! are, individually, a few hundred pixels across. Compositing is a few
//! milliseconds and only happens when something actually changes.
//!
//! Cells are decoded *at cell resolution*, which is where the targeted decoding
//! the rest of the viewer uses pays off most: a folder of 40 megapixel photos
//! becomes a grid of 200-pixel thumbnails without ever materialising a full
//! frame.

use std::collections::HashMap;

use crate::osd::Overlay;

/// Gap between cells, and around the edge of the grid.
const GAP: u32 = 8;
/// Border drawn around the selected cell.
const SELECTION_BORDER: u32 = 3;

/// Bounds on how many images sit across the window.
///
/// The grid is sized by *column count* rather than by cell size. Choosing the
/// cell first leaves whatever does not divide evenly as a dead strip down the
/// right-hand side, and makes each zoom step an unpredictable jump. Choosing
/// columns means every step is "one more per row" and the cells always fill the
/// width exactly.
const MIN_COLUMNS: usize = 1;
const MAX_COLUMNS: usize = 24;
const DEFAULT_COLUMNS: usize = 6;
const MIN_CELL: u32 = 48;

const BACKGROUND: [u8; 4] = [18, 18, 22, 255];
const CELL_EMPTY: [u8; 4] = [32, 32, 38, 255];
const SELECTION: [u8; 4] = [120, 220, 160, 255];
/// Cells the comparison considers the same picture as the selected one.
const MATCH: [u8; 4] = [235, 170, 90, 255];
const BAR: [u8; 4] = [30, 34, 44, 255];
const BAR_TEXT: [u8; 4] = [235, 235, 240, 255];
const LABEL_TEXT: [u8; 4] = [245, 245, 250, 255];
const LABEL_SHADOW: [u8; 4] = [0, 0, 0, 255];

/// Text scale for the compare bar and the per-cell percentages.
const BAR_SCALE: f32 = 3.0;
const LABEL_SCALE: f32 = 2.0;
/// Cells smaller than this get no percentage: the text would cover the picture.
const LABEL_MIN_CELL: u32 = 110;

/// A decoded cell image, already close to cell size.
pub struct Thumb {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct Mosaic {
    /// Images across the window. The wheel changes this, and the cell size
    /// follows from it.
    columns: usize,
    /// Vertical scroll in pixels.
    scroll: f32,
    selected: usize,
    thumbs: HashMap<usize, Thumb>,
    /// Indices already handed to a decoder, so they are not requested twice.
    requested: std::collections::HashSet<usize>,
    /// Decoded bytes currently held, tracked rather than recomputed.
    bytes: u64,
    /// Grid position to playlist index. Set when the folder has been analysed
    /// and reordered; otherwise positions and indices are the same thing.
    order: Option<Vec<usize>>,
    /// Per-image hashes, once the folder has been compared. Their presence is
    /// what puts the grid into compare mode.
    hashes: Option<Vec<Option<u64>>>,
    /// The order to return to when a match group is left. One level deep: the
    /// group is a detour, not a directory tree.
    previous: Option<(Option<Vec<usize>>, usize, usize)>,
    /// Positions that start a new row even when the one above has room: the
    /// first image of each folder, so a tree reads as one block per folder.
    /// Ignored while `order` is set, since a reordered grid has no folders.
    breaks: Vec<usize>,
    dirty: bool,
}

/// Where the grid puts things, for a given window and cell size.
pub struct Layout {
    pub columns: usize,
    pub pitch: u32,
    #[allow(dead_code)] // Completes the layout; read by tests and future paging.
    pub rows: usize,
    pub content_height: f32,
}

impl Mosaic {
    pub fn new(selected: usize) -> Mosaic {
        Mosaic {
            columns: DEFAULT_COLUMNS,
            scroll: 0.0,
            selected,
            thumbs: HashMap::new(),
            requested: std::collections::HashSet::new(),
            bytes: 0,
            order: None,
            hashes: None,
            previous: None,
            breaks: Vec::new(),
            dirty: true,
        }
    }

    /// The list under the grid changed — images added or gone. Each held
    /// thumbnail moves to its image's new index, or goes with its image;
    /// what was still decoding is asked for again; `selected` is the new
    /// selection. Scroll and size stay, so the grid does not jump about
    /// because a file arrived. An ordering from a comparison is dropped: it
    /// was of the old list.
    pub fn remap(&mut self, new_index: impl Fn(usize) -> Option<usize>, selected: usize) {
        let thumbs = std::mem::take(&mut self.thumbs);
        self.bytes = 0;
        for (old, thumb) in thumbs {
            if let Some(new) = new_index(old) {
                self.bytes += thumb.rgba.len() as u64;
                self.thumbs.insert(new, thumb);
            }
        }
        self.requested = self.thumbs.keys().copied().collect();
        self.order = None;
        self.hashes = None;
        self.previous = None;
        self.selected = selected;
        self.dirty = true;
    }

    pub fn breaks(&self) -> &[usize] {
        &self.breaks
    }

    /// Start a new row at each of these positions. Must be ascending.
    pub fn set_breaks(&mut self, breaks: Vec<usize>) {
        self.breaks = breaks;
        self.dirty = true;
    }

    /// Each row's first position and how many cells it holds.
    ///
    /// Without breaks every row is full but the last. With them a row also
    /// ends where a folder does, and the next folder starts a row of its own.
    fn rows(&self, count: usize) -> Vec<(usize, usize)> {
        let columns = self.columns.max(1);
        let breaks: &[usize] = if self.order.is_none() { &self.breaks } else { &[] };
        let mut rows = Vec::with_capacity(count.div_ceil(columns) + breaks.len());
        let mut start = 0;
        let mut next_break = breaks.iter().copied().filter(|b| *b > 0 && *b < count).peekable();
        while start < count {
            let segment_end = loop {
                match next_break.peek() {
                    Some(&b) if b <= start => {
                        next_break.next();
                    }
                    Some(&b) => break b,
                    None => break count,
                }
            };
            let len = (segment_end - start).min(columns);
            rows.push((start, len));
            start += len;
        }
        rows
    }

    /// Row and column of a grid position.
    fn cell_at(rows: &[(usize, usize)], position: usize) -> (usize, usize) {
        let row = rows.partition_point(|r| r.0 <= position).saturating_sub(1);
        let start = rows.get(row).map(|r| r.0).unwrap_or(0);
        (row, position - start)
    }

    /// Playlist index of the selected cell.
    pub fn selected(&self) -> usize {
        self.index_at(self.selected)
    }

    /// Playlist index shown at a grid position.
    fn index_at(&self, position: usize) -> usize {
        match &self.order {
            Some(order) => order.get(position).copied().unwrap_or(position),
            None => position,
        }
    }

    /// Grid position showing a playlist index.
    fn position_of(&self, index: usize) -> usize {
        match &self.order {
            Some(order) => order.iter().position(|i| *i == index).unwrap_or(index),
            None => index,
        }
    }

    /// How many cells the grid is showing, which is fewer than the folder when
    /// it has been narrowed to a match group.
    pub fn count(&self, playlist_len: usize) -> usize {
        self.order.as_ref().map(|o| o.len()).unwrap_or(playlist_len)
    }

    /// True when the grid is showing a subset rather than the whole folder.
    pub fn is_group(&self) -> bool {
        self.previous.is_some()
    }

    /// Narrow the grid to the selected image and everything matching it.
    ///
    /// Cells are enlarged to fill the window, because comparing near-identical
    /// pictures is the entire reason for being here and a thumbnail is no use
    /// for it.
    pub fn enter_match_group(&mut self, win: (f32, f32), playlist_len: usize) -> bool {
        if self.previous.is_some() || !self.is_comparing() {
            return false;
        }
        let selected = self.index_at(self.selected);
        let mut group = vec![selected];
        group.extend(self.matches(self.count(playlist_len)));
        if group.len() < 2 {
            return false;
        }

        // Most alike first, so the closest comparison is adjacent.
        let hashes = self.hashes.clone().unwrap_or_default();
        let key = |i: &usize| match (hashes.get(selected), hashes.get(*i)) {
            (Some(Some(a)), Some(Some(b))) => crate::similarity::distance(*a, *b),
            _ => u32::MAX,
        };
        group.sort_by_key(key);

        self.previous = Some((self.order.clone(), self.selected, self.columns));
        self.order = Some(group.clone());
        self.selected = 0;
        self.scroll = 0.0;
        self.columns = columns_to_fill(win, group.len());
        self.dirty = true;
        true
    }

    /// Return to the full grid. Returns false when not in a group.
    pub fn leave_match_group(&mut self) -> bool {
        let Some((order, selected, columns)) = self.previous.take() else { return false };
        self.order = order;
        self.selected = selected;
        self.columns = columns;
        self.scroll = 0.0;
        self.dirty = true;
        true
    }

    /// Rearrange the grid. Positions are remapped so the selected image stays
    /// selected rather than whatever now happens to sit at its old position.
    pub fn set_order(&mut self, order: Vec<usize>) {
        let selected_index = self.selected();
        self.order = Some(order);
        self.selected = self.position_of(selected_index);
        self.scroll = 0.0;
        self.dirty = true;
    }

    #[cfg(test)]
    pub fn is_ordered(&self) -> bool {
        self.order.is_some()
    }

    pub fn is_comparing(&self) -> bool {
        self.hashes.is_some()
    }

    pub fn set_hashes(&mut self, hashes: Vec<Option<u64>>) {
        self.hashes = Some(hashes);
        self.dirty = true;
    }

    /// How alike an image is to the selected one, if both were hashed.
    fn similarity_to_selected(&self, index: usize) -> Option<f32> {
        let hashes = self.hashes.as_ref()?;
        let a = (*hashes.get(self.index_at(self.selected))?)?;
        let b = (*hashes.get(index)?)?;
        Some(crate::similarity::similarity(a, b))
    }

    /// Playlist indices the comparison groups with the selected image.
    pub fn matches(&self, count: usize) -> Vec<usize> {
        let Some(hashes) = self.hashes.as_ref() else { return Vec::new() };
        let selected = self.index_at(self.selected);
        let Some(Some(a)) = hashes.get(selected) else { return Vec::new() };
        (0..count)
            .filter(|i| *i != selected)
            .filter(|i| {
                matches!(hashes.get(*i), Some(Some(b))
                    if crate::similarity::distance(*a, *b) <= crate::similarity::SAME_IMAGE)
            })
            .collect()
    }

    /// Vertical space above the grid: the compare bar, when there is one.
    fn top_inset(&self) -> u32 {
        if self.is_comparing() {
            GAP + (crate::osd::line_height_at(BAR_SCALE) as u32 + GAP * 2)
        } else {
            GAP
        }
    }

    /// Cell edge for the current column count, filling the window exactly.
    pub fn cell_size(&self, win: (f32, f32)) -> u32 {
        let columns = self.columns.max(1) as f32;
        let usable = win.0 - GAP as f32 * (columns + 1.0);
        ((usable / columns) as u32).max(MIN_CELL)
    }

    #[cfg(test)]
    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    pub fn layout(&self, win: (f32, f32), count: usize) -> Layout {
        let columns = self.columns.max(1);
        let pitch = self.cell_size(win) + GAP;
        let rows = self.rows(count).len().max(1);
        Layout {
            columns,
            pitch,
            rows,
            content_height: (rows as u32 * pitch + self.top_inset()) as f32,
        }
    }

    fn max_scroll(&self, win: (f32, f32), count: usize) -> f32 {
        (self.layout(win, count).content_height - win.1).max(0.0)
    }

    /// How far down the grid is scrolled, 0 to 1, or `None` when it all fits
    /// and there is nowhere to scroll.
    pub fn scroll_fraction(&self, win: (f32, f32), count: usize) -> Option<f32> {
        let max = self.max_scroll(win, count);
        (max > 0.0).then(|| (self.scroll / max).clamp(0.0, 1.0))
    }

    pub fn scroll_by(&mut self, delta: f32, win: (f32, f32), count: usize) {
        self.scroll = (self.scroll + delta).clamp(0.0, self.max_scroll(win, count));
        self.dirty = true;
    }

    /// Add or remove a column, keeping roughly the same images on screen.
    ///
    /// Wheel up means bigger pictures, so it takes a column away.
    pub fn resize_cells(&mut self, steps: f32, win: (f32, f32), count: usize) {
        let before = self.layout(win, count).content_height.max(1.0);
        let anchor = (self.scroll + win.1 / 2.0) / before;

        let delta = if steps > 0.0 { -1isize } else { 1 };
        let columns = (self.columns as isize + delta).clamp(MIN_COLUMNS as isize, MAX_COLUMNS as isize)
            as usize;
        // Never so many columns that the cells fall below a useful size.
        if columns == self.columns {
            return;
        }
        let previous = self.columns;
        self.columns = columns;
        if self.cell_size(win) <= MIN_CELL && columns > previous {
            self.columns = previous;
            return;
        }

        let after = self.layout(win, count).content_height;
        self.scroll = (anchor * after - win.1 / 2.0).clamp(0.0, self.max_scroll(win, count));
        self.dirty = true;
    }

    /// Move the selection, scrolling to follow it.
    pub fn move_selection(&mut self, delta: isize, win: (f32, f32), count: usize) {
        if count == 0 {
            return;
        }
        let next = (self.selected as isize + delta).clamp(0, count as isize - 1) as usize;
        if next == self.selected {
            return;
        }
        self.selected = next;
        self.reveal(win, count);
        self.dirty = true;
    }

    /// Move up or down whole rows, keeping the column where the row is long
    /// enough to have it. Rows are not all full once folders break them.
    pub fn move_rows(&mut self, delta: isize, win: (f32, f32), count: usize) {
        if count == 0 {
            return;
        }
        let rows = self.rows(count);
        let (row, col) = Self::cell_at(&rows, self.selected);
        let target = (row as isize + delta).clamp(0, rows.len() as isize - 1) as usize;
        let (start, len) = rows[target];
        let next = start + col.min(len - 1);
        if next == self.selected {
            return;
        }
        self.selected = next;
        self.reveal(win, count);
        self.dirty = true;
    }

    /// Coming back to a grid set aside earlier: select what is on screen now,
    /// scrolling only if it is out of view, and re-request anything that was
    /// still decoding when the grid was left — those answers were discarded.
    pub fn resume(&mut self, index: usize, win: (f32, f32), count: usize, same_list: bool) {
        if !same_list {
            // Positions moved under the cached cells; they would be wrong.
            self.thumbs.clear();
            self.bytes = 0;
        }
        self.requested = self.thumbs.keys().copied().collect();
        if self.order.is_none() {
            self.selected = index.min(count.saturating_sub(1));
        }
        self.scroll = self.scroll.clamp(0.0, self.max_scroll(win, count));
        self.reveal(win, count);
        self.dirty = true;
    }

    /// Select by grid position, as a click does.
    pub fn set_selected(&mut self, position: usize, win: (f32, f32), count: usize) {
        self.selected = position.min(count.saturating_sub(1));
        self.reveal(win, count);
        self.dirty = true;
    }

    /// Scroll the minimum distance that brings the selected cell fully on
    /// screen. Scrolling further would move the grid under the user for no
    /// reason.
    fn reveal(&mut self, win: (f32, f32), count: usize) {
        let l = self.layout(win, count);
        let row = Self::cell_at(&self.rows(count), self.selected).0 as f32;
        let top = self.top_inset() as f32 + row * l.pitch as f32;
        let bottom = top + self.cell_size(win) as f32;

        if top < self.scroll {
            self.scroll = top - GAP as f32;
        } else if bottom > self.scroll + win.1 {
            self.scroll = bottom - win.1 + GAP as f32;
        }
        self.scroll = self.scroll.clamp(0.0, self.max_scroll(win, count));
    }

    /// Playlist index of the image under a window-space point, if any.
    pub fn index_under(&self, at: (f32, f32), win: (f32, f32), count: usize) -> Option<usize> {
        self.hit_test(at, win, count).map(|p| self.index_at(p))
    }

    /// Grid position under a window-space point, if any.
    pub fn hit_test(&self, at: (f32, f32), win: (f32, f32), count: usize) -> Option<usize> {
        let l = self.layout(win, count);
        let x = at.0 - GAP as f32;
        let y = at.1 + self.scroll - self.top_inset() as f32;
        if x < 0.0 || y < 0.0 {
            return None;
        }
        let (col, row) = (x / l.pitch as f32, y / l.pitch as f32);
        // Points in the gap between cells belong to neither.
        let cell = self.cell_size(win) as f32;
        if col.fract() * l.pitch as f32 > cell || row.fract() * l.pitch as f32 > cell
        {
            return None;
        }
        let (col, row) = (col as usize, row as usize);
        if col >= l.columns {
            return None;
        }
        let (start, len) = *self.rows(count).get(row)?;
        (col < len).then_some(start + col)
    }

    /// Indices currently on screen, plus a row above and below so scrolling
    /// does not reveal empty cells that only then start decoding.
    pub fn visible(&self, win: (f32, f32), count: usize) -> std::ops::Range<usize> {
        let l = self.layout(win, count);
        let above = (self.scroll - self.top_inset() as f32).max(0.0);
        let first_row = ((above / l.pitch as f32).floor() as usize).saturating_sub(1);
        let last_row = ((self.scroll + win.1) / l.pitch as f32).ceil() as usize + 1;
        let rows = self.rows(count);
        let start = rows.get(first_row).map(|r| r.0).unwrap_or(count);
        let end = rows.get(last_row).map(|r| r.0).unwrap_or(count);
        start.min(end)..end
    }

    /// Cells on screen that have neither a thumbnail nor a decode in flight.
    pub fn take_pending(&mut self, win: (f32, f32), count: usize) -> Vec<usize> {
        // Resolved first: the filter below needs `&mut self`, so the mapping
        // cannot still be borrowing it.
        let indices: Vec<usize> =
            self.visible(win, count).map(|pos| self.index_at(pos)).collect();
        indices
            .into_iter()
            .filter(|i| !self.thumbs.contains_key(i) && self.requested.insert(*i))
            .collect()
    }

    #[cfg(test)]
    pub fn bytes_held(&self) -> u64 {
        self.bytes
    }

    pub fn insert(&mut self, index: usize, thumb: Thumb) {
        self.bytes += thumb.rgba.len() as u64;
        if let Some(old) = self.thumbs.insert(index, thumb) {
            self.bytes -= old.rgba.len() as u64;
        }
        self.dirty = true;
    }

    /// Note that a cell failed to decode, so it is not retried forever.
    pub fn mark_failed(&mut self, index: usize) {
        let _ = index;
        self.dirty = true;
    }

    /// Drop thumbnails until the cache fits `budget` bytes.
    ///
    /// Furthest from the visible range goes first: scrolling back a little
    /// should not have to re-decode, but a folder of fifty thousand images must
    /// not be allowed to fill memory.
    pub fn evict(&mut self, win: (f32, f32), count: usize, budget: u64) {
        if self.bytes <= budget {
            return;
        }
        let visible: std::collections::HashSet<usize> =
            self.visible(win, count).map(|pos| self.index_at(pos)).collect();
        let centre = self.index_at(self.selected) as i64;

        let mut by_distance: Vec<(i64, usize)> = self
            .thumbs
            .keys()
            .map(|i| ((*i as i64 - centre).abs(), *i))
            .collect();
        // Furthest first, so draining from the front sheds the least useful.
        by_distance.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        for (_, index) in by_distance {
            if self.bytes <= budget {
                break;
            }
            if visible.contains(&index) {
                continue;
            }
            if let Some(thumb) = self.thumbs.remove(&index) {
                self.bytes -= thumb.rgba.len() as u64;
                self.requested.remove(&index);
            }
        }
        self.dirty = true;
    }

    /// Cells just outside the visible range, for filling in while idle.
    ///
    /// Visible cells are always requested first and separately; this is the
    /// "everything else in the folder, eventually" pass, and it stops as soon
    /// as the budget is reached so background work cannot evict what is on
    /// screen.
    pub fn take_prefetch(&mut self, win: (f32, f32), count: usize, budget: u64, limit: usize) -> Vec<usize> {
        if self.bytes >= budget || count == 0 {
            return Vec::new();
        }
        let visible = self.visible(win, count);
        let mut out = Vec::new();

        // Walk outwards from the visible range in both directions at once.
        for step in 1..count {
            for candidate in [visible.end.checked_add(step - 1), visible.start.checked_sub(step)] {
                let Some(pos) = candidate else { continue };
                if pos >= count {
                    continue;
                }
                let i = self.index_at(pos);
                if self.thumbs.contains_key(&i) {
                    continue;
                }
                if self.requested.insert(i) {
                    out.push(i);
                    if out.len() >= limit {
                        return out;
                    }
                }
            }
        }
        out
    }

    /// Paint the grid into a window-sized RGBA buffer.
    pub fn compose(&self, win: (f32, f32), count: usize) -> Overlay {
        let (w, h) = (win.0.max(1.0) as u32, win.1.max(1.0) as u32);
        let l = self.layout(win, count);

        let cell = self.cell_size(win);
        let mut rgba = Vec::with_capacity(w as usize * h as usize * 4);
        for _ in 0..w as usize * h as usize {
            rgba.extend_from_slice(&BACKGROUND);
        }

        let rows = self.rows(count);
        for position in self.visible(win, count) {
            let index = self.index_at(position);
            let (row, col) = Self::cell_at(&rows, position);
            let x = GAP as i64 + col as i64 * l.pitch as i64;
            let y = self.top_inset() as i64 + row as i64 * l.pitch as i64 - self.scroll as i64;

            // The selected cell, and in compare mode everything the comparison
            // groups with it, are outlined. Two colours so it is obvious which
            // one you chose and which ones followed.
            let similarity = self.similarity_to_selected(index);
            let is_match = position != self.selected
                && similarity.is_some_and(|s| s >= crate::similarity::SAME_IMAGE_SIMILARITY);

            if position == self.selected || is_match {
                let colour = if position == self.selected { SELECTION } else { MATCH };
                crate::osd::outline(&mut rgba, w, h, x, y, cell, cell, SELECTION_BORDER, colour);
            }

            match self.thumbs.get(&index) {
                Some(thumb) => self.draw_thumb(&mut rgba, w, h, x, y, thumb, cell),
                // A cell that has not decoded yet is a placeholder, not a hole:
                // the grid should not reflow when images arrive.
                None => crate::osd::fill(&mut rgba, w, h, x, y, cell, cell, CELL_EMPTY),
            }

            // How alike this one is to the selected image, on the cell itself.
            if let Some(similarity) = similarity {
                if cell >= LABEL_MIN_CELL && position != self.selected {
                    let text = format!("{:.0}%", similarity * 100.0);
                    let tw = crate::osd::text_width_at(&text, LABEL_SCALE) as i64;
                    let th = crate::osd::line_height_at(LABEL_SCALE) as i64;
                    let lx = x + cell as i64 - tw - 4;
                    let ly = y + cell as i64 - th - 4;
                    if lx >= 0 && ly >= 0 {
                        // A shadow, because the text lands on the picture and
                        // any single colour will vanish against some of them.
                        crate::osd::draw_text_into(
                            &mut rgba, w as usize, h as usize, &text,
                            lx as usize + 1, ly as usize + 1, LABEL_SHADOW, LABEL_SCALE,
                        );
                        crate::osd::draw_text_into(
                            &mut rgba, w as usize, h as usize, &text,
                            lx as usize, ly as usize,
                            if is_match { MATCH } else { LABEL_TEXT }, LABEL_SCALE,
                        );
                    }
                }
            }
        }

        if self.is_comparing() {
            self.draw_compare_bar(&mut rgba, w, h, count);
        }
        Overlay::new(rgba, w, h)
    }

    /// A bar across the top saying what mode this is and what is selected.
    fn draw_compare_bar(&self, dst: &mut [u8], w: u32, h: u32, count: usize) {
        let bar_h = crate::osd::line_height_at(BAR_SCALE) as u32 + GAP * 2;
        crate::osd::fill(dst, w, h, 0, 0, w, bar_h, BAR);

        let text = if self.is_group() {
            format!("MATCH GROUP  -  {} IMAGES  -  ENTER OPENS  -  ESC GOES BACK", count)
        } else {
            format!(
                "COMPARE  -  {} SIMILAR TO THE SELECTED IMAGE  -  ENTER COMPARES THEM  -  ESC LEAVES",
                self.matches(count).len()
            )
        };
        crate::osd::draw_text_into(
            dst,
            w as usize,
            h as usize,
            &text,
            GAP as usize,
            GAP as usize,
            BAR_TEXT,
            BAR_SCALE,
        );
    }

    /// Draw one thumbnail, letterboxed inside its cell so nothing is cropped
    /// and nothing is stretched.
    #[allow(clippy::too_many_arguments)]
    fn draw_thumb(
        &self,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        x: i64,
        y: i64,
        thumb: &Thumb,
        cell: u32,
    ) {
        if thumb.width == 0 || thumb.height == 0 {
            return;
        }
        let scale = (cell as f32 / thumb.width as f32).min(cell as f32 / thumb.height as f32);
        let dw = ((thumb.width as f32 * scale) as u32).max(1);
        let dh = ((thumb.height as f32 * scale) as u32).max(1);
        let ox = x + ((cell - dw.min(cell)) / 2) as i64;
        let oy = y + ((cell - dh.min(cell)) / 2) as i64;

        for row in 0..dh {
            let py = oy + row as i64;
            if py < 0 || py >= dst_h as i64 {
                continue;
            }
            // Nearest sampling: the cell was decoded near this size already, so
            // the ratio is close to 1 and a filter would buy little.
            let sy = (row as u64 * thumb.height as u64 / dh as u64).min(thumb.height as u64 - 1);
            for col in 0..dw {
                let px = ox + col as i64;
                if px < 0 || px >= dst_w as i64 {
                    continue;
                }
                let sx = (col as u64 * thumb.width as u64 / dw as u64).min(thumb.width as u64 - 1);
                let s = ((sy * thumb.width as u64 + sx) * 4) as usize;
                let d = ((py as u32 * dst_w + px as u32) * 4) as usize;
                dst[d..d + 4].copy_from_slice(&thumb.rgba[s..s + 4]);
            }
        }
    }
}

/// Column count that shows `count` images at the largest size without scrolling.
fn columns_to_fill(win: (f32, f32), count: usize) -> usize {
    let n = count.max(1);
    let mut best = (MIN_COLUMNS, 0.0f32);
    for columns in MIN_COLUMNS..=n.min(MAX_COLUMNS) {
        let rows = n.div_ceil(columns) as f32;
        let by_width = (win.0 - GAP as f32 * (columns as f32 + 1.0)) / columns as f32;
        let by_height = (win.1 - GAP as f32 * (rows + 1.0)) / rows;
        let cell = by_width.min(by_height);
        if cell > best.1 {
            best = (columns, cell);
        }
    }
    best.0
}

/// Draw a frame around a cell rather than a filled block behind it.
///
/// A filled rectangle shows through wherever the thumbnail is letterboxed,
/// which for a 4:3 picture in a square cell is most of it — the cell then reads
/// as coloured rather than outlined.
#[allow(clippy::too_many_arguments)]


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_changed_list_keeps_the_thumbnails_of_the_images_still_in_it() {
        let mut m = Mosaic::new(0);
        for i in 0..4 {
            m.insert(i, Thumb { rgba: vec![i as u8; 4], width: 1, height: 1 });
        }
        // Image 1 deleted, a new one at the front: 0->1, 2->3, 3->4.
        m.remap(|old| [Some(1), None, Some(3), Some(4)][old], 3);
        assert_eq!(m.selected(), 3);
        let mut held: Vec<usize> = m.thumbs.keys().copied().collect();
        held.sort();
        assert_eq!(held, [1, 3, 4]);
        assert_eq!(m.thumbs[&3].rgba[0], 2, "thumbnails follow their images");
        assert_eq!(m.bytes_held(), 12);
    }

    #[test]
    fn each_folder_starts_a_row_of_its_own() {
        // Six columns; folders of 3, 8 and 2 images.
        let mut m = Mosaic::new(0);
        m.set_breaks(vec![3, 11]);
        let win = (1000.0, 800.0);
        assert_eq!(m.rows(13), [(0, 3), (3, 6), (9, 2), (11, 2)]);
        assert_eq!(m.layout(win, 13).rows, 4);

        // Down from the third cell of a short row keeps the column.
        m.set_selected(2, win, 13);
        m.move_rows(1, win, 13);
        assert_eq!(m.selected(), 5);
        // And clamps to the end of a shorter row.
        m.move_rows(1, win, 13);
        assert_eq!(m.selected(), 10);
        m.move_rows(1, win, 13);
        assert_eq!(m.selected(), 12);

        // The empty end of a short row is not a cell.
        let pitch = m.layout(win, 13).pitch as f32;
        let top = m.top_inset() as f32;
        assert_eq!(m.hit_test((GAP as f32 + 3.5 * pitch, top + 1.0), win, 13), None);
        assert_eq!(m.hit_test((GAP as f32 + 1.0, top + pitch + 1.0), win, 13), Some(3));
    }

    const WIN: (f32, f32) = (1000.0, 600.0);

    #[test]
    fn cells_always_fill_the_width() {
        // Whatever the column count, the row spans the window with only the
        // gaps left over — no dead strip down the right-hand side.
        let mut m = Mosaic::new(0);
        for _ in 0..6 {
            let l = m.layout(WIN, 100);
            let used = l.columns as f32 * (m.cell_size(WIN) + GAP) as f32 + GAP as f32;
            let slack = WIN.0 - used;
            assert!(
                (0.0..l.columns as f32).contains(&slack),
                "{slack} px unused with {} columns",
                l.columns
            );
            m.resize_cells(-1.0, WIN, 100);
        }
    }

    #[test]
    fn zooming_changes_the_column_count_one_at_a_time() {
        let mut m = Mosaic::new(0);
        let start = m.columns();
        m.resize_cells(1.0, WIN, 100);
        assert_eq!(m.columns(), start - 1, "wheel up means bigger, so one fewer");
        m.resize_cells(-1.0, WIN, 100);
        assert_eq!(m.columns(), start);
    }

    #[test]
    fn the_column_count_is_bounded() {
        let mut m = Mosaic::new(0);
        for _ in 0..50 {
            m.resize_cells(1.0, WIN, 100);
        }
        assert_eq!(m.columns(), MIN_COLUMNS);
        for _ in 0..80 {
            m.resize_cells(-1.0, WIN, 100);
        }
        assert!(m.columns() <= MAX_COLUMNS);
        assert!(m.cell_size(WIN) >= MIN_CELL, "cells must stay usable");
    }

    #[test]
    fn selection_stays_in_range_and_scrolls_into_view() {
        let mut m = Mosaic::new(0);
        m.move_selection(-1, WIN, 50);
        assert_eq!(m.selected(), 0, "cannot select before the first");
        m.move_selection(10_000, WIN, 50);
        assert_eq!(m.selected(), 49, "cannot select past the last");
        assert!(m.scroll > 0.0, "selecting the last cell must scroll to it");
    }

    #[test]
    fn scrolling_is_clamped_to_the_content() {
        let mut m = Mosaic::new(0);
        m.scroll_by(-500.0, WIN, 50);
        assert_eq!(m.scroll, 0.0);
        m.scroll_by(100_000.0, WIN, 50);
        assert_eq!(m.scroll, m.max_scroll(WIN, 50));
    }

    #[test]
    fn a_short_grid_does_not_scroll_at_all() {
        let m = Mosaic::new(0);
        // Four cells across one row, in a window far taller than one row.
        assert_eq!(m.max_scroll(WIN, 3), 0.0);
    }

    #[test]
    fn hit_test_finds_cells_and_rejects_the_gaps() {
        let m = Mosaic::new(0);
        let pitch = m.layout(WIN, 20).pitch as f32;
        // Middle of the first cell.
        assert_eq!(m.hit_test((GAP as f32 + 10.0, GAP as f32 + 10.0), WIN, 20), Some(0));
        // Second column.
        assert_eq!(m.hit_test((GAP as f32 + pitch + 10.0, GAP as f32 + 10.0), WIN, 20), Some(1));
        // Above the grid entirely.
        assert_eq!(m.hit_test((2.0, 2.0), WIN, 20), None);
    }

    #[test]
    fn only_visible_cells_are_requested_and_never_twice() {
        let mut m = Mosaic::new(0);
        let first = m.take_pending(WIN, 10_000);
        assert!(!first.is_empty());
        assert!(first.len() < 10_000, "must not request the whole folder at once");
        assert!(m.take_pending(WIN, 10_000).is_empty(), "already in flight");
    }

    #[test]
    fn eviction_frees_memory_but_keeps_what_is_on_screen() {
        let mut m = Mosaic::new(0);
        let thumb = || Thumb { rgba: vec![0; 64 * 64 * 4], width: 64, height: 64 };
        for i in 0..300 {
            m.insert(i, thumb());
        }
        let before = m.bytes_held();
        assert!(before > 0);

        let budget = before / 4;
        m.evict(WIN, 300, budget);
        assert!(m.bytes_held() <= budget, "{} > {budget}", m.bytes_held());

        // Everything on screen survived.
        for pos in m.visible(WIN, 300) {
            let i = m.index_at(pos);
            assert!(m.thumbs.contains_key(&i), "evicted a visible cell {i}");
        }
    }

    #[test]
    fn prefetch_stops_at_the_budget() {
        let mut m = Mosaic::new(0);
        let _ = m.take_pending(WIN, 1000);
        assert!(!m.take_prefetch(WIN, 1000, u64::MAX, 16).is_empty(), "should look ahead");

        for i in 0..50 {
            m.insert(i, Thumb { rgba: vec![0; 1024 * 1024], width: 512, height: 512 });
        }
        assert!(
            m.take_prefetch(WIN, 1000, 1024, 16).is_empty(),
            "background loading must stop once the budget is reached"
        );
    }

    #[test]
    fn prefetch_never_returns_a_cell_twice() {
        let mut m = Mosaic::new(0);
        let first = m.take_prefetch(WIN, 500, u64::MAX, 32);
        let second = m.take_prefetch(WIN, 500, u64::MAX, 32);
        for i in &first {
            assert!(!second.contains(i), "cell {i} requested twice");
        }
    }

    #[test]
    fn entering_a_match_group_narrows_the_grid_and_enlarges_the_cells() {
        let mut m = Mosaic::new(0);
        let a = 0u64;
        // Three alike, two not.
        m.set_hashes(vec![Some(a), Some(a | 0b11), Some(0xFFFF_FFFF_FFFF_FFFF), Some(a | 0b111), Some(0x0F0F_0F0F_0F0F_0F0F)]);
        let before_cell = m.cell_size(WIN);

        assert!(m.enter_match_group(WIN, 5));
        assert!(m.is_group());
        assert_eq!(m.count(5), 3, "the selected image and its two matches");
        assert!(m.cell_size(WIN) > before_cell, "cells should grow to fill the window");
        assert_eq!(m.selected(), 0, "the image drilled into stays selected");

        // And back out again, restoring what was there.
        assert!(m.leave_match_group());
        assert!(!m.is_group());
        assert_eq!(m.count(5), 5);
        assert_eq!(m.cell_size(WIN), before_cell);
    }

    #[test]
    fn a_lone_image_has_no_group_to_enter() {
        let mut m = Mosaic::new(0);
        m.set_hashes(vec![Some(0), Some(0xFFFF_FFFF_FFFF_FFFF)]);
        assert!(!m.enter_match_group(WIN, 2), "nothing matches, so there is nothing to compare");
        assert!(!m.is_group());
        assert!(!m.leave_match_group(), "and nothing to leave");
    }

    #[test]
    fn a_highlight_frames_the_cell_without_filling_it() {
        let mut m = Mosaic::new(0);
        // A cell whose thumbnail is a black 1:1 square, so any colour showing
        // inside it came from the highlight rather than the picture.
        let cell = m.cell_size(WIN) as usize;
        m.insert(0, Thumb { rgba: vec![0, 0, 0, 255].repeat(cell * cell), width: cell as u32, height: cell as u32 });
        let ov = m.compose(WIN, 1);

        let at = |x: usize, y: usize| {
            let i = (y * ov.width as usize + x) * 4;
            [ov.rgba[i], ov.rgba[i + 1], ov.rgba[i + 2], ov.rgba[i + 3]]
        };
        // The frame sits just outside the cell, so probe one pixel to its left.
        assert_eq!(
            at(GAP as usize - 1, GAP as usize + cell / 2),
            SELECTION,
            "the frame should be drawn outside the cell"
        );
        assert_eq!(
            at(GAP as usize + cell / 2, GAP as usize + cell / 2),
            [0, 0, 0, 255],
            "the middle of the cell must still be the picture"
        );
    }

    #[test]
    fn compare_mode_finds_what_matches_the_selection() {
        let mut m = Mosaic::new(0);
        assert!(!m.is_comparing());

        // Two near-identical hashes and two unrelated ones.
        let a = 0u64;
        m.set_hashes(vec![Some(a), Some(a | 0b11), Some(0xFFFF_FFFF_FFFF_FFFF), None]);
        assert!(m.is_comparing());

        let matches = m.matches(4);
        assert_eq!(matches, vec![1], "only the near-identical one, and never itself");

        // The percentage is reported relative to the selection.
        assert_eq!(m.similarity_to_selected(0), Some(1.0));
        assert!(m.similarity_to_selected(1).is_some_and(|s| s > 0.9));
        assert!(m.similarity_to_selected(2).is_some_and(|s| s < 0.1));
        assert_eq!(m.similarity_to_selected(3), None, "an unhashed image has no score");
    }

    #[test]
    fn the_compare_bar_takes_space_above_the_grid() {
        let mut plain = Mosaic::new(0);
        let before = plain.layout(WIN, 100).content_height;
        plain.set_hashes(vec![Some(0); 100]);
        let after = plain.layout(WIN, 100).content_height;
        assert!(after > before, "the bar must push the grid down, not cover it");

        // And a click still lands on the cell it looks like it lands on.
        let inset = plain.top_inset() as f32;
        assert_eq!(plain.hit_test((GAP as f32 + 10.0, inset + 10.0), WIN, 100), Some(0));
    }

    #[test]
    fn reordering_keeps_the_same_image_selected() {
        let mut m = Mosaic::new(0);
        m.set_selected(3, WIN, 10);
        assert_eq!(m.selected(), 3);

        // Reverse the grid: the selected image moves position but stays chosen.
        m.set_order((0..10).rev().collect());
        assert_eq!(m.selected(), 3, "selection follows the image, not the slot");
        assert_eq!(m.index_at(0), 9, "the grid really was reordered");
    }

    #[test]
    fn an_ordered_grid_requests_the_reordered_cells() {
        let mut m = Mosaic::new(0);
        m.set_order((0..200).rev().collect());
        let pending = m.take_pending(WIN, 200);
        // The first visible slot now holds the last playlist entry.
        assert!(pending.contains(&199), "should fetch what the order puts on screen");
        assert!(!pending.contains(&0), "and not what it moved off screen");
    }

    #[test]
    fn composing_fills_the_window_exactly() {
        let m = Mosaic::new(0);
        let ov = m.compose(WIN, 12);
        assert_eq!(ov.width, WIN.0 as u32);
        assert_eq!(ov.height, WIN.1 as u32);
        assert_eq!(ov.rgba.len(), (WIN.0 * WIN.1) as usize * 4);
    }

    #[test]
    fn a_thumbnail_keeps_its_aspect_ratio_inside_the_cell() {
        let mut m = Mosaic::new(0);
        // A wide thumbnail: 4 wide, 1 tall, all white.
        m.insert(0, Thumb { rgba: vec![255; 4 * 1 * 4], width: 4, height: 1 });
        let ov = m.compose(WIN, 1);

        // The cell must contain white pixels, and they must not fill its height.
        let cell = m.cell_size(WIN) as usize;
        let mut white_rows = 0;
        for row in 0..cell {
            let y = GAP as usize + row;
            let start = (y * ov.width as usize + GAP as usize) * 4;
            if ov.rgba[start..start + cell * 4].chunks_exact(4).any(|p| p[0] > 200) {
                white_rows += 1;
            }
        }
        assert!(white_rows > 0, "the thumbnail was not drawn");
        assert!(white_rows < cell / 2, "a 4:1 image must not fill a square cell");
    }
}
