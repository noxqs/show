//! Integer box downscaling.
//!
//! Needed for two reasons: images wider than the GPU's maximum texture
//! dimension cannot be uploaded at all, and images far larger than the display
//! are pure waste to keep at full resolution. A box filter over an integer
//! block is both the cheapest correct downscale and the one that vectorises,
//! which matters when the input is a few hundred megapixels.

use rayon::prelude::*;

/// Smallest integer factor that brings both dimensions within `limit`.
pub fn factor_for_limit(w: u32, h: u32, limit: u32) -> u32 {
    if limit == 0 {
        return 1;
    }
    let need = |d: u32| d.div_ceil(limit).max(1);
    need(w).max(need(h))
}

/// Average `factor` x `factor` blocks of RGBA8 down to one pixel each.
/// Returns the new `(pixels, width, height)`.
pub fn box_downscale(src: &[u8], w: u32, h: u32, factor: u32) -> (Vec<u8>, u32, u32) {
    assert!(factor >= 1);
    if factor == 1 {
        return (src.to_vec(), w, h);
    }

    // Round up so edge blocks that are only partly covered still produce a
    // pixel; the loops below clamp their reads to the source bounds.
    let ow = w.div_ceil(factor);
    let oh = h.div_ceil(factor);
    let (w, h, f) = (w as usize, h as usize, factor as usize);
    let (ow_us, oh_us) = (ow as usize, oh as usize);

    let mut out = vec![0u8; ow_us * oh_us * 4];
    out.par_chunks_mut(ow_us * 4).enumerate().for_each(|(oy, row)| {
        let y0 = oy * f;
        let y1 = (y0 + f).min(h);
        for ox in 0..ow_us {
            let x0 = ox * f;
            let x1 = (x0 + f).min(w);
            let n = ((y1 - y0) * (x1 - x0)) as u32;
            if n == 0 {
                continue;
            }
            let mut acc = [0u32; 4];
            for y in y0..y1 {
                let base = (y * w + x0) * 4;
                for px in src[base..base + (x1 - x0) * 4].chunks_exact(4) {
                    acc[0] += px[0] as u32;
                    acc[1] += px[1] as u32;
                    acc[2] += px[2] as u32;
                    acc[3] += px[3] as u32;
                }
            }
            let d = &mut row[ox * 4..ox * 4 + 4];
            for c in 0..4 {
                d[c] = (acc[c] / n) as u8;
            }
        }
        let _ = oh_us;
    });

    (out, ow, oh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_covers_the_limit() {
        assert_eq!(factor_for_limit(1000, 500, 8192), 1);
        assert_eq!(factor_for_limit(16384, 500, 8192), 2);
        assert_eq!(factor_for_limit(20000, 500, 8192), 3);
        assert_eq!(factor_for_limit(114944, 98304, 16384), 8);
        // Exactly divisible: no extra step.
        assert_eq!(factor_for_limit(16384, 16384, 8192), 2);
    }

    #[test]
    fn averages_a_uniform_block() {
        // 4x4, left half 0, right half 200.
        let mut src = vec![0u8; 4 * 4 * 4];
        for y in 0..4 {
            for x in 2..4 {
                let i = (y * 4 + x) * 4;
                src[i..i + 4].copy_from_slice(&[200, 200, 200, 255]);
            }
        }
        let (out, w, h) = box_downscale(&src, 4, 4, 2);
        assert_eq!((w, h), (2, 2));
        assert_eq!(&out[0..4], &[0, 0, 0, 0]);
        assert_eq!(&out[4..8], &[200, 200, 200, 255]);
    }

    #[test]
    fn handles_partial_edge_blocks() {
        // 3x3 downscaled by 2 gives 2x2, with the right/bottom blocks partial.
        let src = vec![128u8; 3 * 3 * 4];
        let (out, w, h) = box_downscale(&src, 3, 3, 2);
        assert_eq!((w, h), (2, 2));
        assert!(out.iter().all(|&b| b == 128), "{out:?}");
    }
}
