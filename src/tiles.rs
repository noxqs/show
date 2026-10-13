//! Which tiles of a tiled image to show, and getting them read.
//!
//! The first stage of the tile renderer: everything that decides *what* is
//! needed, none of the drawing. Three parts:
//!
//! - [`plan`] turns a zoom and a pan into a pyramid level and the tiles of it
//!   the window can see, centre first, plus a ring around them to read ahead.
//!   Pure arithmetic, so it is tested without a file or a GPU.
//! - [`TileCache`] holds decoded tiles under a byte budget, dropping the least
//!   recently used first — but never one the current plan still needs.
//! - [`TileLoader`] reads tiles on a small pool of threads. Each new plan
//!   *replaces* the queue rather than adding to it, so panning away from a
//!   region stops its tiles being read instead of finishing them for nothing.
//!
//! Where a tile is not loaded yet, [`fallback`] finds the closest coarser tile
//! that covers the same ground, and the part of it that does: drawn scaled up,
//! it is blurry rather than missing.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

use crate::tiled::{Level, Pyramid, Tile, TileReader};

/// One tile of one level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    /// Index into [`Pyramid::levels`], 0 being full resolution.
    pub level: usize,
    pub tx: u32,
    pub ty: u32,
}

/// What a view of the image needs.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub level: usize,
    /// On screen, nearest the centre of the window first.
    pub visible: Vec<TileKey>,
    /// One tile's width around the visible ones: what a pan reveals next.
    pub ahead: Vec<TileKey>,
}

impl Plan {
    /// Everything the plan wants, in the order to fetch it.
    pub fn wanted(&self) -> impl Iterator<Item = TileKey> + '_ {
        self.visible.iter().chain(&self.ahead).copied()
    }
}

/// The level and tiles to draw `center` (in full-resolution pixels) at `zoom`
/// (screen pixels per full-resolution pixel) in a `window`-sized viewport.
pub fn plan(pyramid: &Pyramid, zoom: f32, center: [f32; 2], window: (f32, f32)) -> Plan {
    let level = pyramid.level_for_zoom(zoom);
    let mut plan = plan_at(pyramid, level, zoom, center, window);
    // Then the next level down for the same view, last: what zooming out
    // will want, so that it is usually already here when it does.
    if level + 1 < pyramid.levels.len() {
        let coarser = plan_at(pyramid, level + 1, zoom, center, window);
        plan.ahead.extend(coarser.visible);
    }
    plan
}

/// As [`plan`], at a given level rather than the one the zoom calls for.
pub fn plan_at(pyramid: &Pyramid, level_index: usize, zoom: f32, center: [f32; 2], window: (f32, f32)) -> Plan {
    let level = pyramid.levels[level_index];
    let (fw, _) = pyramid.full_size();
    let scale = level.scale(fw);

    // The window's footprint, in this level's pixels.
    let half = (window.0 / 2.0 / zoom * scale, window.1 / 2.0 / zoom * scale);
    let mid = (center[0] * scale, center[1] * scale);
    let (x0, x1) = (mid.0 - half.0, mid.0 + half.0);
    let (y0, y1) = (mid.1 - half.1, mid.1 + half.1);

    let (across, down) = level.grid();
    let (tw, th) = (level.tile.0 as f32, level.tile.1 as f32);
    let clamp = |v: f32, n: u32| -> i64 { (v.floor() as i64).clamp(0, n as i64 - 1) };
    let range = |lo: f32, hi: f32, size: f32, n: u32| -> Option<(i64, i64)> {
        // Entirely off the image on this axis: nothing to see.
        if hi <= 0.0 || lo >= n as f32 * size {
            return None;
        }
        Some((clamp(lo / size, n), clamp((hi - f32::EPSILON) / size, n)))
    };

    let (Some((cx0, cx1)), Some((cy0, cy1))) = (range(x0, x1, tw, across), range(y0, y1, th, down)) else {
        return Plan { level: level_index, visible: Vec::new(), ahead: Vec::new() };
    };

    // Centre out: distance from each tile's middle to the window's, so what
    // the eye is on arrives first and the edges fill in after.
    let by_distance = |keys: &mut Vec<TileKey>| {
        keys.sort_by(|a, b| {
            let d = |k: &TileKey| {
                let (kx, ky) = ((k.tx as f32 + 0.5) * tw - mid.0, (k.ty as f32 + 0.5) * th - mid.1);
                kx * kx + ky * ky
            };
            d(a).total_cmp(&d(b)).then(a.cmp(b))
        });
    };

    let key = |tx: i64, ty: i64| TileKey { level: level_index, tx: tx as u32, ty: ty as u32 };
    let mut visible: Vec<TileKey> =
        (cy0..=cy1).flat_map(|ty| (cx0..=cx1).map(move |tx| key(tx, ty))).collect();
    by_distance(&mut visible);

    let (ax0, ax1) = ((cx0 - 1).max(0), (cx1 + 1).min(across as i64 - 1));
    let (ay0, ay1) = ((cy0 - 1).max(0), (cy1 + 1).min(down as i64 - 1));
    let mut ahead: Vec<TileKey> = (ay0..=ay1)
        .flat_map(|ty| (ax0..=ax1).map(move |tx| (tx, ty)))
        .filter(|&(tx, ty)| tx < cx0 || tx > cx1 || ty < cy0 || ty > cy1)
        .map(|(tx, ty)| key(tx, ty))
        .collect();
    by_distance(&mut ahead);

    Plan { level: level_index, visible, ahead }
}

/// The tiles one level finer that cover the same ground as `key` — sharper
/// stand-ins than any coarser tile, and on the card already when the view has
/// just zoomed out past them.
pub fn finer(pyramid: &Pyramid, key: TileKey) -> Vec<TileKey> {
    if key.level == 0 {
        return Vec::new();
    }
    let (fw, _) = pyramid.full_size();
    let (coarse, fine) = (pyramid.levels[key.level], pyramid.levels[key.level - 1]);
    // The coarse tile's rectangle, in the finer level's pixels.
    let ratio = fine.scale(fw) / coarse.scale(fw);
    let x0 = key.tx as f32 * coarse.tile.0 as f32 * ratio;
    let y0 = key.ty as f32 * coarse.tile.1 as f32 * ratio;
    let x1 = ((key.tx + 1) * coarse.tile.0).min(coarse.width) as f32 * ratio;
    let y1 = ((key.ty + 1) * coarse.tile.1).min(coarse.height) as f32 * ratio;
    let (across, down) = fine.grid();
    let (tw, th) = (fine.tile.0 as f32, fine.tile.1 as f32);
    let cols = (x0 / tw).floor() as u32..((x1 / tw).ceil() as u32).min(across);
    let rows = (y0 / th).floor() as u32..((y1 / th).ceil() as u32).min(down);
    rows.flat_map(|ty| cols.clone().map(move |tx| TileKey { level: key.level - 1, tx, ty })).collect()
}

/// Where a missing tile can borrow its picture from: a coarser tile covering
/// the same ground, and the part of it that does, as fractions of that tile
/// `[x0, y0, x1, y1]`. Nearest level first; the first one loaded wins.
pub fn fallback(pyramid: &Pyramid, key: TileKey) -> Vec<(TileKey, [f32; 4])> {
    let (fw, _) = pyramid.full_size();
    let fine = pyramid.levels[key.level];
    let fine_scale = fine.scale(fw);
    // The tile's rectangle in full-resolution pixels.
    let x0 = key.tx as f32 * fine.tile.0 as f32 / fine_scale;
    let y0 = key.ty as f32 * fine.tile.1 as f32 / fine_scale;
    let x1 = ((key.tx + 1) * fine.tile.0).min(fine.width) as f32 / fine_scale;
    let y1 = ((key.ty + 1) * fine.tile.1).min(fine.height) as f32 / fine_scale;

    (key.level + 1..pyramid.levels.len())
        .map(|coarse_index| {
            let coarse = pyramid.levels[coarse_index];
            let s = coarse.scale(fw);
            let (tw, th) = (coarse.tile.0 as f32, coarse.tile.1 as f32);
            // The coarse tile holding the fine tile's top-left corner. A fine
            // tile never straddles two coarse ones when levels halve and tiles
            // are the same size, which is how every pyramid seen is built;
            // where one does, the corner's tile still covers most of it.
            let (cx, cy) = (((x0 * s) / tw).floor() as u32, ((y0 * s) / th).floor() as u32);
            let (ox, oy) = (cx as f32 * tw, cy as f32 * th);
            // The coarse tile's actual extent, which is short at the edges.
            let (cw, ch) = (
                ((cx + 1) * coarse.tile.0).min(coarse.width) as f32 - ox,
                ((cy + 1) * coarse.tile.1).min(coarse.height) as f32 - oy,
            );
            let frac = [
                ((x0 * s - ox) / cw).clamp(0.0, 1.0),
                ((y0 * s - oy) / ch).clamp(0.0, 1.0),
                ((x1 * s - ox) / cw).clamp(0.0, 1.0),
                ((y1 * s - oy) / ch).clamp(0.0, 1.0),
            ];
            (TileKey { level: coarse_index, tx: cx, ty: cy }, frac)
        })
        .collect()
}

/// Tiles — decoded pixels, or their textures on the card — bounded by bytes.
pub struct TileCache<T> {
    /// Each with its size and when it was last used.
    tiles: HashMap<TileKey, (T, u64, u64)>,
    bytes: u64,
    budget: u64,
    /// A clock for recency: bumped on every touch, so no wall time is needed.
    tick: u64,
}

impl<T> TileCache<T> {
    pub fn new(budget: u64) -> TileCache<T> {
        TileCache { tiles: HashMap::new(), bytes: 0, budget, tick: 0 }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    pub fn contains(&self, key: &TileKey) -> bool {
        self.tiles.contains_key(key)
    }

    /// A tile, marked as just used.
    pub fn get(&mut self, key: &TileKey) -> Option<&T> {
        self.tick += 1;
        let tick = self.tick;
        self.tiles.get_mut(key).map(|(tile, _, used)| {
            *used = tick;
            &*tile
        })
    }

    /// A tile without marking it used, for drawing from after the frame's
    /// tiles have been touched.
    pub fn peek(&self, key: &TileKey) -> Option<&T> {
        self.tiles.get(key).map(|(tile, _, _)| tile)
    }

    pub fn insert(&mut self, key: TileKey, tile: T, bytes: u64) {
        self.tick += 1;
        if let Some((_, old, _)) = self.tiles.insert(key, (tile, bytes, self.tick)) {
            self.bytes -= old;
        }
        self.bytes += bytes;
    }

    pub fn clear(&mut self) {
        self.tiles.clear();
        self.bytes = 0;
    }

    /// Drop the least recently used tiles until under budget, sparing `keep`
    /// — what is on screen now. If `keep` alone is over budget, it stays over:
    /// a screen with holes in it is worse than using the memory.
    pub fn evict(&mut self, keep: &HashSet<TileKey>) {
        if self.bytes <= self.budget {
            return;
        }
        let mut by_age: Vec<(u64, TileKey)> = self
            .tiles
            .iter()
            .filter(|(k, _)| !keep.contains(k))
            .map(|(k, (_, _, used))| (*used, *k))
            .collect();
        by_age.sort_unstable();
        for (_, key) in by_age {
            if self.bytes <= self.budget {
                break;
            }
            if let Some((_, size, _)) = self.tiles.remove(&key) {
                self.bytes -= size;
            }
        }
    }
}

/// The work queue the loader's threads share.
struct Queue {
    pending: VecDeque<TileKey>,
    /// Being read right now, so a replanned queue does not ask for them again.
    in_flight: HashSet<TileKey>,
    closed: bool,
}

/// Reads tiles on a pool of threads, each with its own file handle.
pub struct TileLoader {
    shared: Arc<(Mutex<Queue>, Condvar)>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl TileLoader {
    /// `on_tile` is called on a worker thread with each tile as it is read —
    /// in the viewer, to post it to the event loop.
    pub fn new(
        pyramid: Arc<Pyramid>,
        threads: usize,
        on_tile: impl Fn(TileKey, anyhow::Result<Tile>) + Send + Sync + 'static,
    ) -> TileLoader {
        let shared = Arc::new((
            Mutex::new(Queue { pending: VecDeque::new(), in_flight: HashSet::new(), closed: false }),
            Condvar::new(),
        ));
        let on_tile = Arc::new(on_tile);
        let workers = (0..threads.max(1))
            .map(|i| {
                let (shared, on_tile, pyramid) = (shared.clone(), on_tile.clone(), pyramid.clone());
                std::thread::Builder::new()
                    .name(format!("tiles-{i}"))
                    .spawn(move || worker(&pyramid, &shared, &*on_tile))
                    .expect("spawning a tile reader")
            })
            .collect();
        TileLoader { shared, workers }
    }

    /// Replace what is waiting with `wanted`, in that order, leaving out what
    /// `have` already holds and what is already being read.
    pub fn request(&self, wanted: impl IntoIterator<Item = TileKey>, have: impl Fn(&TileKey) -> bool) {
        let (queue, ready) = &*self.shared;
        let mut q = queue.lock().unwrap();
        let busy = q.in_flight.clone();
        q.pending = wanted.into_iter().filter(|k| !have(k) && !busy.contains(k)).collect();
        ready.notify_all();
    }

    /// How many tiles are waiting or being read.
    pub fn outstanding(&self) -> usize {
        let q = self.shared.0.lock().unwrap();
        q.pending.len() + q.in_flight.len()
    }
}

impl Drop for TileLoader {
    fn drop(&mut self) {
        let (queue, ready) = &*self.shared;
        queue.lock().unwrap().closed = true;
        ready.notify_all();
        // Not joined: a worker inside a large tile's decode would hold up
        // whatever dropped the loader — typically moving to the next image.
        // It finds the queue closed when it next looks, and ends.
        self.workers.clear();
    }
}

fn worker(
    pyramid: &Pyramid,
    shared: &(Mutex<Queue>, Condvar),
    on_tile: &(dyn Fn(TileKey, anyhow::Result<Tile>) + Send + Sync),
) {
    // Opened on first use, not up front: a pool of eight handles is wasted on
    // a screen that needs two tiles.
    let mut reader: Option<TileReader> = None;
    loop {
        let key = {
            let (queue, ready) = shared;
            let mut q = queue.lock().unwrap();
            loop {
                if q.closed {
                    return;
                }
                if let Some(key) = q.pending.pop_front() {
                    q.in_flight.insert(key);
                    break key;
                }
                q = ready.wait(q).unwrap();
            }
        };

        let level: Level = pyramid.levels[key.level];
        let result = match reader.as_mut() {
            Some(r) => r.read(&level, key.tx, key.ty),
            None => TileReader::open(pyramid).and_then(|mut r| {
                let tile = r.read(&level, key.tx, key.ty);
                reader = Some(r);
                tile
            }),
        };
        shared.0.lock().unwrap().in_flight.remove(&key);
        on_tile(key, result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// 1024 x 512 at full size, halving three times, 256-pixel tiles.
    fn pyramid() -> Pyramid {
        let levels = (0..4)
            .map(|i| Level { ifd: i, width: 1024 >> i, height: 512 >> i, tile: (256, 256), cached: None })
            .collect();
        Pyramid { path: PathBuf::from("test.tif"), levels, cache: None }
    }

    #[test]
    fn the_level_is_the_smallest_with_a_texel_per_pixel() {
        let p = pyramid();
        // 1:1 needs full resolution; a quarter size can use the level that is
        // a quarter size; in between rounds towards sharper.
        assert_eq!(plan(&p, 1.0, [512.0, 256.0], (800.0, 600.0)).level, 0);
        assert_eq!(plan(&p, 0.25, [512.0, 256.0], (800.0, 600.0)).level, 2);
        assert_eq!(plan(&p, 0.3, [512.0, 256.0], (800.0, 600.0)).level, 1);
        // Past the smallest level there is nothing smaller to use.
        assert_eq!(plan(&p, 0.01, [512.0, 256.0], (800.0, 600.0)).level, 3);
    }

    #[test]
    fn visible_tiles_are_what_the_window_covers_centre_first() {
        let p = pyramid();
        // At 1:1 a 300x200 window centred on (512, 256) spans x 362..662 and
        // y 156..356: tile columns 1 and 2, rows 0 and 1.
        let plan = plan(&p, 1.0, [512.0, 256.0], (300.0, 200.0));
        let mut seen: Vec<(u32, u32)> = plan.visible.iter().map(|k| (k.tx, k.ty)).collect();
        seen.sort();
        assert_eq!(seen, [(1, 0), (1, 1), (2, 0), (2, 1)]);

        // Off-centre: the tile under the middle of the window comes first.
        let plan = super::plan(&p, 1.0, [300.0, 100.0], (600.0, 400.0));
        assert_eq!((plan.visible[0].tx, plan.visible[0].ty), (1, 0));
    }

    #[test]
    fn read_ahead_is_the_ring_around_then_the_next_level_down() {
        let p = pyramid();
        let plan = plan(&p, 1.0, [512.0, 256.0], (300.0, 200.0));
        let visible: HashSet<_> = plan.visible.iter().collect();
        assert!(plan.ahead.iter().all(|k| !visible.contains(k)));
        // Columns 0..=3 by rows 0..=1, less the four visible: four more at
        // this level, then level 1's tiles for the same view.
        let (ring, down): (Vec<&TileKey>, Vec<&TileKey>) = plan.ahead.iter().partition(|k| k.level == 0);
        assert_eq!(ring.len(), 4);
        assert!(!down.is_empty() && down.iter().all(|k| k.level == 1));
        assert!(plan.ahead.iter().position(|k| k.level == 1) > plan.ahead.iter().rposition(|k| k.level == 0));
    }

    #[test]
    fn the_finer_tiles_under_a_coarse_one_are_the_four_it_halves_into() {
        let p = pyramid();
        let mut under = finer(&p, TileKey { level: 1, tx: 1, ty: 0 });
        under.sort();
        let at = |tx, ty| TileKey { level: 0, tx, ty };
        assert_eq!(under, [at(2, 0), at(2, 1), at(3, 0), at(3, 1)]);
        // Full resolution has nothing finer.
        assert!(finer(&p, TileKey { level: 0, tx: 0, ty: 0 }).is_empty());
    }

    #[test]
    fn a_view_off_the_image_needs_nothing_and_edges_are_clamped() {
        let p = pyramid();
        let plan = plan(&p, 1.0, [5000.0, 5000.0], (300.0, 200.0));
        assert!(plan.visible.is_empty() && plan.ahead.is_empty());

        // Zoomed right out, the whole image is one screen: every tile of the
        // smallest level, and no index past the grid.
        let plan = super::plan(&p, 0.05, [512.0, 256.0], (1920.0, 1080.0));
        assert_eq!(plan.level, 3);
        assert_eq!(plan.visible, [TileKey { level: 3, tx: 0, ty: 0 }]);
    }

    #[test]
    fn a_missing_tile_falls_back_to_the_part_of_a_coarser_one_it_covers() {
        let p = pyramid();
        // Full-resolution tile (3, 1) is the bottom-right quarter of the
        // image's right half: at level 1 it is the bottom-right quarter of
        // tile (1, 0); at level 2 an eighth of the only tile.
        let chain = fallback(&p, TileKey { level: 0, tx: 3, ty: 1 });
        assert_eq!(chain[0], (TileKey { level: 1, tx: 1, ty: 0 }, [0.5, 0.5, 1.0, 1.0]));
        assert_eq!(chain[1], (TileKey { level: 2, tx: 0, ty: 0 }, [0.75, 0.5, 1.0, 1.0]));
        assert_eq!(chain.len(), 3);
    }

    #[test]
    fn the_loader_reads_what_is_asked_and_skips_what_is_held() {
        // A real file: 300x200, written in strips, which the reader treats as
        // full-width tiles.
        let path = std::env::temp_dir().join(format!("show-tiles-{}.tif", std::process::id()));
        let img = image::RgbImage::from_fn(300, 200, |x, y| image::Rgb([x as u8, y as u8, 7]));
        img.save(&path).unwrap();
        let pyramid = Arc::new(crate::tiled::probe(&path).unwrap().unwrap());
        let (across, down) = pyramid.levels[0].grid();

        let (tx, rx) = std::sync::mpsc::channel();
        let loader = TileLoader::new(pyramid.clone(), 3, move |key, tile| {
            let _ = tx.send((key, tile));
        });
        let all: Vec<TileKey> =
            (0..down).flat_map(|ty| (0..across).map(move |tx| TileKey { level: 0, tx, ty })).collect();
        // Pretend the first is already cached: it must not be read again.
        let held = all[0];
        loader.request(all.clone(), |k| *k == held);

        let mut got: Vec<TileKey> = (1..all.len())
            .map(|_| {
                let (key, tile) = rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
                let tile = tile.unwrap();
                assert_eq!(tile.width, 300);
                assert_eq!(tile.rgba.len(), (tile.width * tile.height * 4) as usize);
                key
            })
            .collect();
        got.sort();
        assert_eq!(got, all[1..]);
        assert!(rx.recv_timeout(std::time::Duration::from_millis(100)).is_err(), "nothing extra");
        drop(loader);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_new_request_replaces_what_was_waiting() {
        // No threads get a look in: the queue is inspected directly.
        let shared = Arc::new((
            Mutex::new(Queue { pending: VecDeque::new(), in_flight: HashSet::new(), closed: true }),
            Condvar::new(),
        ));
        let loader = TileLoader { shared: shared.clone(), workers: Vec::new() };
        let key = |tx| TileKey { level: 0, tx, ty: 0 };
        loader.request([key(0), key(1), key(2)], |_| false);
        shared.0.lock().unwrap().in_flight.insert(key(3));
        loader.request([key(3), key(4)], |_| false);
        let q = shared.0.lock().unwrap();
        assert_eq!(q.pending, [key(4)], "old requests gone, the one in flight not asked again");
    }


    #[test]
    fn the_cache_drops_the_oldest_but_never_what_is_on_screen() {
        let key = |tx| TileKey { level: 0, tx, ty: 0 };
        let mut cache = TileCache::new(300);
        for tx in 0..4 {
            cache.insert(key(tx), (), 100);
        }
        // Touch 0, so 1 is now the oldest; keep 1 anyway, as if on screen.
        cache.get(&key(0));
        cache.evict(&HashSet::from([key(1)]));
        assert!(cache.bytes() <= 300);
        assert!(cache.contains(&key(1)), "on screen, so kept");
        assert!(cache.contains(&key(0)), "used recently, so kept");
        assert!(!cache.contains(&key(2)), "the oldest not on screen goes");
    }

    #[test]
    fn a_screen_bigger_than_the_budget_keeps_every_tile() {
        let key = |tx| TileKey { level: 0, tx, ty: 0 };
        let mut cache = TileCache::new(100);
        cache.insert(key(0), (), 100);
        cache.insert(key(1), (), 100);
        cache.evict(&HashSet::from([key(0), key(1)]));
        assert_eq!(cache.len(), 2);
    }
}

