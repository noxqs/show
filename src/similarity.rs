//! Perceptual hashing, so a folder can be sorted by how alike its images are.
//!
//! The hash is pHash: reduce to 32x32 grey, take the two-dimensional DCT, keep
//! the low-frequency 8x8 corner minus the DC term, and threshold at the median.
//! Sixty-four bits that survive rescaling, re-compression and gamma shifts,
//! because all of those move high-frequency detail and leave the coarse
//! structure the low corner describes.
//!
//! It is written out rather than pulled from a crate for two reasons: it is
//! sixty lines of well-specified arithmetic, and this program's dependency list
//! is a feature. What it cannot do is survive rotation or a heavy crop — the
//! hash is computed on a fixed grid, so moving the content moves the bits. That
//! is a property of the method, not a tuning problem, and feature matching is
//! the answer if it ever matters.

// Built and tested, not yet wired to a control. `visible_similarity` answers
// "how different do these look" — the ranking that says which of two
// duplicates went through the worse encoder. Waiting on a place to show it.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Edge of the grid the DCT runs on.
const GRID: usize = 32;
/// Edge of the low-frequency corner kept from it.
const KEEP: usize = 8;

/// Similarity at or above which two images are treated as the same picture.
///
/// Expressed as a percentage because that is what the grid shows and what a
/// person reasons about; the Hamming threshold below is derived from it so the
/// two can never drift apart.
pub const SAME_IMAGE_SIMILARITY: f32 = 0.75;

/// The equivalent distance over a 64-bit hash.
pub const SAME_IMAGE: u32 = ((1.0 - SAME_IMAGE_SIMILARITY) * 64.0) as u32;

/// How alike two hashes are, from 0.0 to 1.0.
pub fn similarity(a: u64, b: u64) -> f32 {
    1.0 - (a ^ b).count_ones() as f32 / 64.0
}

pub fn distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Perceptual hash of an RGBA8 image of any size.
pub fn phash(rgba: &[u8], width: u32, height: u32) -> u64 {
    let grey = to_grey_grid(rgba, width, height);
    let coeffs = dct_2d(&grey);

    // The DC term is the average brightness; keeping it would make the hash
    // depend on exposure, which is exactly what it should ignore.
    let mut low = Vec::with_capacity(KEEP * KEEP - 1);
    for y in 0..KEEP {
        for x in 0..KEEP {
            if (x, y) != (0, 0) {
                low.push(coeffs[y * GRID + x]);
            }
        }
    }

    let mut sorted = low.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    // Compare with a small tolerance rather than exactly. Coefficients that sit
    // on the median differ only by float noise between two encodings of the
    // same picture, and a bare `>` turns that noise into flipped bits.
    let spread = sorted[sorted.len() - 1] - sorted[0];
    let epsilon = spread * 1e-4;

    let mut hash = 0u64;
    for (i, v) in low.iter().enumerate() {
        if *v > median + epsilon {
            hash |= 1 << i;
        }
    }
    hash
}

/// Box-average down to a GRID x GRID luma grid.
fn to_grey_grid(rgba: &[u8], width: u32, height: u32) -> Vec<f32> {
    let mut grid = vec![0.0f32; GRID * GRID];
    if width == 0 || height == 0 {
        return grid;
    }

    for (cell, out) in grid.iter_mut().enumerate() {
        let (gx, gy) = (cell % GRID, cell / GRID);
        // Source rectangle for this cell, at least one pixel wide and tall.
        let x0 = (gx as u64 * width as u64 / GRID as u64) as u32;
        let x1 = (((gx + 1) as u64 * width as u64 / GRID as u64) as u32).max(x0 + 1).min(width);
        let y0 = (gy as u64 * height as u64 / GRID as u64) as u32;
        let y1 = (((gy + 1) as u64 * height as u64 / GRID as u64) as u32).max(y0 + 1).min(height);

        let mut sum = 0.0f32;
        let mut n = 0u32;
        for y in y0..y1 {
            for x in x0..x1 {
                let i = ((y as usize * width as usize) + x as usize) * 4;
                if i + 2 >= rgba.len() {
                    continue;
                }
                // Rec. 601 luma: the weights perceptual hashing has always used.
                sum += 0.299 * rgba[i] as f32
                    + 0.587 * rgba[i + 1] as f32
                    + 0.114 * rgba[i + 2] as f32;
                n += 1;
            }
        }
        *out = if n > 0 { sum / n as f32 } else { 0.0 };
    }
    grid
}

/// Separable 2D DCT-II. Naive, and that is fine: 32x32 is a few tens of
/// thousands of multiply-adds, far below the cost of decoding the image.
fn dct_2d(input: &[f32]) -> Vec<f32> {
    let cos: Vec<f32> = (0..GRID * GRID)
        .map(|i| {
            let (x, u) = (i % GRID, i / GRID);
            (((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI) / (2.0 * GRID as f32)).cos()
        })
        .collect();

    let scale = |u: usize| if u == 0 { (1.0f32 / GRID as f32).sqrt() } else { (2.0f32 / GRID as f32).sqrt() };

    // Rows, then columns.
    let mut rows = vec![0.0f32; GRID * GRID];
    for y in 0..GRID {
        for u in 0..GRID {
            let mut sum = 0.0;
            for x in 0..GRID {
                sum += input[y * GRID + x] * cos[u * GRID + x];
            }
            rows[y * GRID + u] = sum * scale(u);
        }
    }

    let mut out = vec![0.0f32; GRID * GRID];
    for x in 0..GRID {
        for v in 0..GRID {
            let mut sum = 0.0;
            for y in 0..GRID {
                sum += rows[y * GRID + x] * cos[v * GRID + y];
            }
            out[v * GRID + x] = sum * scale(v);
        }
    }
    out
}

/// How visible the difference between two images is, from 0.0 to 1.0.
///
/// This answers a different question from [`phash`]. The hash asks "are these
/// the same picture"; this asks "how different do they look" — which is what
/// tells you, of two copies of the same photo, which one has been through a
/// worse encoder.
///
/// Both images are reduced to the smaller one's dimensions first, because SSIM
/// is only defined on aligned images of equal size, and shrinking the larger
/// loses less than stretching the smaller invents.
pub fn visible_similarity(
    a: &[u8],
    a_size: (u32, u32),
    b: &[u8],
    b_size: (u32, u32),
) -> f32 {
    let w = a_size.0.min(b_size.0);
    let h = a_size.1.min(b_size.1);
    if w == 0 || h == 0 {
        return 0.0;
    }
    let ga = grey_at(a, a_size, (w, h));
    let gb = grey_at(b, b_size, (w, h));
    ssim(&ga, &gb, w as usize, h as usize)
}

/// Box-average an RGBA image to a luma buffer of exactly `to`.
fn grey_at(rgba: &[u8], from: (u32, u32), to: (u32, u32)) -> Vec<f32> {
    let (tw, th) = (to.0 as usize, to.1 as usize);
    let mut out = vec![0.0f32; tw * th];
    for (i, cell) in out.iter_mut().enumerate() {
        let (gx, gy) = (i % tw, i / tw);
        let x0 = (gx as u64 * from.0 as u64 / tw as u64) as u32;
        let x1 = (((gx + 1) as u64 * from.0 as u64 / tw as u64) as u32).max(x0 + 1).min(from.0);
        let y0 = (gy as u64 * from.1 as u64 / th as u64) as u32;
        let y1 = (((gy + 1) as u64 * from.1 as u64 / th as u64) as u32).max(y0 + 1).min(from.1);

        let (mut sum, mut n) = (0.0f32, 0u32);
        for y in y0..y1 {
            for x in x0..x1 {
                let p = ((y as usize * from.0 as usize) + x as usize) * 4;
                if p + 2 < rgba.len() {
                    sum += 0.299 * rgba[p] as f32
                        + 0.587 * rgba[p + 1] as f32
                        + 0.114 * rgba[p + 2] as f32;
                    n += 1;
                }
            }
        }
        *cell = if n > 0 { sum / n as f32 } else { 0.0 };
    }
    out
}

/// Mean structural similarity over 8x8 windows.
fn ssim(a: &[f32], b: &[f32], w: usize, h: usize) -> f32 {
    const WINDOW: usize = 8;
    // The usual stabilisers, for an 8-bit dynamic range.
    const C1: f32 = 6.5025;
    const C2: f32 = 58.5225;

    let (mut total, mut windows) = (0.0f32, 0u32);
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let (x1, y1) = ((x + WINDOW).min(w), (y + WINDOW).min(h));
            let n = ((x1 - x) * (y1 - y)) as f32;
            if n < 4.0 {
                x += WINDOW;
                continue;
            }

            let (mut sa, mut sb) = (0.0f32, 0.0f32);
            for yy in y..y1 {
                for xx in x..x1 {
                    sa += a[yy * w + xx];
                    sb += b[yy * w + xx];
                }
            }
            let (ma, mb) = (sa / n, sb / n);

            let (mut va, mut vb, mut cov) = (0.0f32, 0.0f32, 0.0f32);
            for yy in y..y1 {
                for xx in x..x1 {
                    let (da, db) = (a[yy * w + xx] - ma, b[yy * w + xx] - mb);
                    va += da * da;
                    vb += db * db;
                    cov += da * db;
                }
            }
            let (va, vb, cov) = (va / n, vb / n, cov / n);

            total += ((2.0 * ma * mb + C1) * (2.0 * cov + C2))
                / ((ma * ma + mb * mb + C1) * (va + vb + C2));
            windows += 1;
            x += WINDOW;
        }
        y += WINDOW;
    }
    if windows == 0 { 0.0 } else { (total / windows as f32).clamp(0.0, 1.0) }
}

/// A hash plus what it was computed from, so a changed file is not trusted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Entry {
    pub hash: u64,
    pub modified: u64,
    pub len: u64,
}

/// Hashes remembered between runs.
///
/// Keyed by path, validated by size and modification time: a file edited in
/// place must not keep the old hash, and re-hashing a folder is expensive
/// enough that not caching it would make the feature unusable twice.
#[derive(Default)]
pub struct Cache {
    entries: HashMap<PathBuf, Entry>,
    dirty: bool,
}

impl Cache {
    pub fn path() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
        Some(base.join("show").join("hashes"))
    }

    pub fn load() -> Cache {
        let mut cache = Cache::default();
        let Some(path) = Cache::path() else { return cache };
        let Ok(text) = std::fs::read_to_string(&path) else { return cache };

        for line in text.lines() {
            let mut fields = line.split('\t');
            let (Some(hash), Some(modified), Some(len), Some(file)) =
                (fields.next(), fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            let (Ok(hash), Ok(modified), Ok(len)) = (
                u64::from_str_radix(hash, 16),
                modified.parse::<u64>(),
                len.parse::<u64>(),
            ) else {
                continue;
            };
            cache.entries.insert(PathBuf::from(file), Entry { hash, modified, len });
        }
        cache
    }

    pub fn save(&self) -> anyhow::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let path = Cache::path().ok_or_else(|| anyhow::anyhow!("no cache directory"))?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        // Tab separated with the path last, because a path may contain
        // anything except a tab or a newline.
        let mut out = String::with_capacity(self.entries.len() * 64);
        for (file, e) in &self.entries {
            out.push_str(&format!("{:016x}\t{}\t{}\t{}\n", e.hash, e.modified, e.len, file.display()));
        }
        std::fs::write(&path, out)?;
        Ok(())
    }

    /// The stored hash, if the file still looks like the one it came from.
    pub fn get(&self, path: &Path) -> Option<u64> {
        let entry = self.entries.get(path)?;
        let meta = std::fs::metadata(path).ok()?;
        let modified = meta
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs();
        (entry.len == meta.len() && entry.modified == modified).then_some(entry.hash)
    }

    pub fn insert(&mut self, path: &Path, hash: u64) {
        let Ok(meta) = std::fs::metadata(path) else { return };
        let modified = meta
            .modified()
            .ok()
            .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.entries.insert(path.to_path_buf(), Entry { hash, modified, len: meta.len() });
        self.dirty = true;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Hash every file, using the cache where it is still valid.
///
/// Deliberately does *not* use the global rayon pool. Hashing a folder is a
/// background chore, and taking every core for it would stall the decoding that
/// keeps the grid responsive while it runs — so it gets a quarter of them.
pub fn analyse(
    paths: &[PathBuf],
    io: crate::loader::Io,
    on_progress: impl Fn(usize, usize) + Sync,
) -> (Vec<Option<u64>>, Cache) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let cache = Cache::load();
    let total = paths.len();
    let done = AtomicUsize::new(0);

    let threads = std::thread::available_parallelism()
        .map(|n| (n.get() / 4).max(1))
        .unwrap_or(1);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build();

    let hash_one = |path: &PathBuf| -> (Option<u64>, bool) {
        if let Some(cached) = cache.get(path) {
            return (Some(cached), false);
        }
        // A 32x32 target is the cheapest decode there is: for JPEG it is a
        // one-eighth scaled inverse DCT and almost no pixels.
        let target = crate::decode::Target { width: GRID as u32, height: GRID as u32 };
        let hash = crate::loader::load(path, io, Some(target))
            .ok()
            .map(|l| phash(&l.decoded.rgba, l.decoded.width, l.decoded.height));
        (hash, hash.is_some())
    };

    let run = |paths: &[PathBuf]| -> Vec<(Option<u64>, bool)> {
        paths
            .par_iter()
            .map(|path| {
                let result = hash_one(path);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                // Reporting every file would flood the event loop with
                // redraws; every sixteenth still looks continuous.
                if n % 16 == 0 || n == total {
                    on_progress(n, total);
                }
                result
            })
            .collect()
    };

    let results = match &pool {
        Ok(pool) => pool.install(|| run(paths)),
        // A pool we could not build is not a reason to refuse the work.
        Err(_) => run(paths),
    };

    let mut cache = cache;
    for (path, (hash, computed)) in paths.iter().zip(&results) {
        if *computed {
            if let Some(h) = hash {
                cache.insert(path, *h);
            }
        }
    }
    (results.into_iter().map(|(h, _)| h).collect(), cache)
}

/// Order `hashes` so alike images sit together, alike first, odd ones last.
///
/// Images within [`SAME_IMAGE`] of each other are grouped, groups are emitted
/// largest and tightest first, and everything with no neighbour follows in
/// order of how far its nearest neighbour was — so the most singular image in
/// the folder ends up last.
pub fn order_by_similarity(hashes: &[Option<u64>]) -> Vec<usize> {
    let n = hashes.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    // Nearest neighbour distance per image, and the union-find groups.
    let mut nearest = vec![u32::MAX; n];
    for i in 0..n {
        let Some(a) = hashes[i] else { continue };
        for j in (i + 1)..n {
            let Some(b) = hashes[j] else { continue };
            let d = distance(a, b);
            nearest[i] = nearest[i].min(d);
            nearest[j] = nearest[j].min(d);
            if d <= SAME_IMAGE {
                let (ra, rb) = (find(&mut parent, i), find(&mut parent, j));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        if hashes[i].is_some() {
            let root = find(&mut parent, i);
            groups.entry(root).or_default().push(i);
        }
    }

    let mut clustered: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();
    // Biggest groups first; ties broken by tightness so the most obviously
    // duplicated pair leads.
    clustered.sort_by(|a, b| {
        let tightness = |g: &Vec<usize>| g.iter().map(|i| nearest[*i]).min().unwrap_or(u32::MAX);
        b.len().cmp(&a.len()).then_with(|| tightness(a).cmp(&tightness(b)))
    });

    let mut order: Vec<usize> = Vec::with_capacity(n);
    for mut group in clustered {
        group.sort_unstable();
        order.extend(group);
    }

    // Everything else, least singular first, so the bottom of the grid is the
    // image least like anything around it.
    let mut singles: Vec<usize> = (0..n).filter(|i| !order.contains(i)).collect();
    singles.sort_by_key(|i| (nearest[*i], *i));
    order.extend(singles);
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An RGBA image built from a function of x and y.
    fn make(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Vec<u8> {
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = f(x, y);
                px.extend_from_slice(&[v, v, v, 255]);
            }
        }
        px
    }

    /// The same picture at any resolution: sampled from normalised
    /// coordinates, so two sizes really are the same content rescaled rather
    /// than two different patterns that happen to share a formula.
    fn scene(w: u32, h: u32, brightness: f32) -> Vec<u8> {
        make(w, h, |x, y| {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            // Amplitudes chosen so a brightness shift of up to +/-40 still
            // fits in range: clipping would change the structure, not just the
            // exposure, and the test would then be measuring the wrong thing.
            let value = 110.0
                + 45.0 * (3.0 * std::f32::consts::PI * u).sin()
                + 30.0 * (2.0 * std::f32::consts::PI * v).cos()
                + brightness;
            value.clamp(0.0, 255.0) as u8
        })
    }

    #[test]
    fn identical_images_hash_identically() {
        let a = make(64, 64, |x, y| ((x * 3 + y * 5) % 256) as u8);
        assert_eq!(phash(&a, 64, 64), phash(&a, 64, 64));
        assert_eq!(similarity(phash(&a, 64, 64), phash(&a, 64, 64)), 1.0);
    }

    #[test]
    fn rescaling_barely_moves_the_hash() {
        // Surviving a resize is the whole point of the method.
        let big = scene(400, 300, 0.0);
        let small = scene(100, 75, 0.0);
        let d = distance(phash(&big, 400, 300), phash(&small, 100, 75));
        assert!(d <= SAME_IMAGE, "distance {d} exceeds the duplicate threshold");
    }

    #[test]
    fn brightness_shifts_barely_change_the_hash() {
        // The DC term is dropped precisely so exposure does not matter.
        let base = scene(128, 128, 0.0);
        let brighter = scene(128, 128, 40.0);
        let d = distance(phash(&base, 128, 128), phash(&brighter, 128, 128));
        assert!(d <= 2, "a brightness change should barely move the hash: {d}");
    }

    #[test]
    fn different_images_are_far_apart() {
        let stripes = make(64, 64, |x, _| if (x / 4) % 2 == 0 { 255 } else { 0 });
        let rings = make(64, 64, |x, y| {
            let (dx, dy) = (x as f32 - 32.0, y as f32 - 32.0);
            (((dx * dx + dy * dy).sqrt() * 8.0) as u32 % 256) as u8
        });
        let d = distance(phash(&stripes, 64, 64), phash(&rings, 64, 64));
        assert!(d > SAME_IMAGE, "unrelated images should not look like duplicates: {d}");
    }

    #[test]
    fn visible_similarity_reads_one_for_identical_and_falls_for_noise() {
        let a = scene(128, 96, 0.0);
        assert!((visible_similarity(&a, (128, 96), &a, (128, 96)) - 1.0).abs() < 1e-3);

        // Same picture, coarsely quantised: still similar, but not identical.
        let quantised: Vec<u8> = a.iter().map(|v| v & 0b1110_0000).collect();
        let degraded = visible_similarity(&a, (128, 96), &quantised, (128, 96));
        assert!(degraded < 0.999, "quantisation must be visible: {degraded}");
        assert!(degraded > 0.3, "but it is still the same picture: {degraded}");

        // A different picture entirely scores far lower.
        let other = make(128, 96, |x, _| if (x / 3) % 2 == 0 { 255 } else { 0 });
        let unrelated = visible_similarity(&a, (128, 96), &other, (128, 96));
        assert!(unrelated < degraded, "unrelated {unrelated} vs degraded {degraded}");
    }

    #[test]
    fn visible_similarity_compares_at_the_smaller_size() {
        // The same scene at two resolutions is the same scene.
        let big = scene(400, 300, 0.0);
        let small = scene(100, 75, 0.0);
        let s = visible_similarity(&big, (400, 300), &small, (100, 75));
        assert!(s > 0.8, "a rescaled copy should still look alike: {s}");
    }

    #[test]
    fn the_threshold_agrees_with_the_percentage_it_is_derived_from() {
        assert_eq!(SAME_IMAGE, 16, "75% of 64 bits");
        // A pair exactly at the threshold counts as the same picture.
        let at_threshold = (1u64 << SAME_IMAGE) - 1;
        assert_eq!(distance(0, at_threshold), SAME_IMAGE);
        assert!((similarity(0, at_threshold) - SAME_IMAGE_SIMILARITY).abs() < 1e-6);
    }

    #[test]
    fn similarity_is_a_zero_to_one_index() {
        assert_eq!(similarity(0, 0), 1.0);
        assert_eq!(similarity(0, u64::MAX), 0.0);
        assert!((similarity(0, 0xFFFF_FFFF) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ordering_puts_groups_first_and_singletons_last() {
        // Three near-identical hashes, then two unrelated ones.
        let a = 0x0000_0000_0000_0000u64;
        let group = [Some(a), Some(a | 0b11), Some(a | 0b1100)];
        let odd_one = Some(0xFFFF_FFFF_FFFF_FFFFu64);
        let other = Some(0x0F0F_0F0F_0F0F_0F0Fu64);

        let hashes = vec![odd_one, group[0], other, group[1], group[2]];
        let order = order_by_similarity(&hashes);

        assert_eq!(order.len(), hashes.len(), "every image must appear once");
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);

        // The three-image group leads, contiguously.
        let leading: std::collections::HashSet<usize> = order[..3].iter().copied().collect();
        assert_eq!(leading, [1, 3, 4].into_iter().collect());
    }

    #[test]
    fn images_that_failed_to_hash_still_appear() {
        let hashes = vec![Some(0u64), None, Some(0u64)];
        let order = order_by_similarity(&hashes);
        assert_eq!(order.len(), 3);
        assert!(order.contains(&1), "an unhashable image must not vanish from the grid");
    }
}
