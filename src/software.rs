//! CPU presentation for the first frame.
//!
//! Bringing up Vulkan costs ~45 ms before it can show anything, while decoding
//! a typical JPEG takes ~4 ms. Waiting for the GPU to be ready before putting
//! *any* pixels on screen therefore spends an order of magnitude more time than
//! producing them did. So the first frame goes out through the CPU straight
//! into the window, and the GPU takes over once it has finished starting up.
//!
//! The two never share the window. Softbuffer draws until the moment the wgpu
//! surface is created, and is dropped first; a Vulkan swapchain and an X11
//! image blitter on one window at the same time is not a fight worth having.
//!
//! This rasteriser is deliberately the same *view* as the GPU path -- same
//! transform, same filter rule -- so the handover is not visible.

use std::num::NonZeroU32;
use std::sync::Arc;

use anyhow::{Context, Result};
use rayon::prelude::*;
use winit::window::Window;

use crate::view::View;

pub struct Software {
    // The context has to outlive the surface built from it.
    _context: softbuffer::Context<Arc<Window>>,
    surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
    width: u32,
    height: u32,
}

impl Software {
    pub fn new(window: Arc<Window>) -> Result<Software> {
        let _s = crate::timing::span("cpu", "surface");
        let context = softbuffer::Context::new(window.clone())
            .map_err(|e| anyhow::anyhow!("softbuffer context: {e}"))?;
        let surface = softbuffer::Surface::new(&context, window.clone())
            .map_err(|e| anyhow::anyhow!("softbuffer surface: {e}"))?;

        let size = window.inner_size();
        let mut sw = Software {
            _context: context,
            surface,
            width: size.width.max(1),
            height: size.height.max(1),
        };
        sw.resize(size.width, size.height)?;
        Ok(sw)
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        let (w, h) = (width.max(1), height.max(1));
        self.surface
            .resize(NonZeroU32::new(w).unwrap(), NonZeroU32::new(h).unwrap())
            .map_err(|e| anyhow::anyhow!("softbuffer resize: {e}"))?;
        self.width = w;
        self.height = h;
        Ok(())
    }


    /// Draw `pixels` (RGBA8, `tex` sized) through `view` and put it on screen.
    ///
    /// `source` is the *original* image size, which may be larger than `tex`
    /// when the decode was scaled down; the view works in source coordinates,
    /// so the sampler rescales into texture space itself.
    pub fn present(
        &mut self,
        pixels: Option<(&[u8], (u32, u32))>,
        view: &View,
        source: (f32, f32),
        smooth: bool,
        overlays: &[(&crate::osd::Overlay, (i32, i32))],
        viewport: (f32, f32, f32, f32),
    ) -> Result<()> {
        let (width, height) = (self.width, self.height);
        let mut buffer = self
            .surface
            .buffer_mut()
            .map_err(|e| anyhow::anyhow!("softbuffer buffer: {e}"))?;

        match pixels {
            Some((px, tex)) => {
                let _s = crate::timing::span("cpu", "raster");
                rasterise(&mut buffer, width, height, px, tex, view, source, smooth, viewport);
            }
            None => buffer.fill(0),
        }

        for (ov, at) in overlays {
            blend(&mut buffer, width, height, ov, *at);
        }

        buffer.present().map_err(|e| anyhow::anyhow!("softbuffer present: {e}")).context("presenting")
    }
}

/// Alpha-blend an RGBA overlay over the window buffer at `(x, y)`.
///
/// Straight `over` compositing in 8-bit: the overlay is a handful of thousand
/// pixels, so there is nothing here worth vectorising.
fn blend(dst: &mut [u32], dst_w: u32, dst_h: u32, ov: &crate::osd::Overlay, at: (i32, i32)) {
    for oy in 0..ov.height as i32 {
        let py = at.1 + oy;
        if py < 0 || py >= dst_h as i32 {
            continue;
        }
        for ox in 0..ov.width as i32 {
            let px = at.0 + ox;
            if px < 0 || px >= dst_w as i32 {
                continue;
            }
            let s = ((oy as usize * ov.width as usize) + ox as usize) * 4;
            let (sr, sg, sb, sa) = (
                ov.rgba[s] as u32,
                ov.rgba[s + 1] as u32,
                ov.rgba[s + 2] as u32,
                ov.rgba[s + 3] as u32,
            );
            if sa == 0 {
                continue;
            }
            let i = py as usize * dst_w as usize + px as usize;
            let d = dst[i];
            let (dr, dg, db) = ((d >> 16) & 0xFF, (d >> 8) & 0xFF, d & 0xFF);
            let mix = |s: u32, d: u32| ((s * sa + d * (255 - sa)) / 255) & 0xFF;
            dst[i] = pack(mix(sr, dr) as u8, mix(sg, dg) as u8, mix(sb, db) as u8);
        }
    }
}

/// Sample the image into the window buffer under the view transform.
///
/// Runs backwards -- for each destination pixel, find the source point -- which
/// is what makes arbitrary zoom and pan one code path instead of separate
/// scale, crop and blit stages.
#[allow(clippy::too_many_arguments)]
fn rasterise(
    dst: &mut [u32],
    dst_w: u32,
    dst_h: u32,
    src: &[u8],
    tex: (u32, u32),
    view: &View,
    source: (f32, f32),
    smooth: bool,
    viewport: (f32, f32, f32, f32),
) {
    let (vx, vy, vw, vh) = viewport;
    // Source pixels per texture pixel: 1.0 unless the decode was scaled down.
    let tex_scale = (tex.0 as f32 / source.0.max(1.0), tex.1 as f32 / source.1.max(1.0));
    // Screen pixel -> texture pixel, folding the view transform and the
    // decode's own reduction into one step per axis.
    let step = (tex_scale.0 / view.zoom, tex_scale.1 / view.zoom);
    // Measured from the viewport's top-left, not the window's.
    let origin = (
        (view.center[0] - vw / 2.0 / view.zoom) * tex_scale.0 - vx * step.0,
        (view.center[1] - vh / 2.0 / view.zoom) * tex_scale.1 - vy * step.1,
    );

    let (tw, th) = (tex.0 as i32, tex.1 as i32);

    let (x_lo, x_hi) = (vx.max(0.0) as usize, (vx + vw).min(dst_w as f32) as usize);
    let (y_lo, y_hi) = (vy.max(0.0) as usize, (vy + vh).min(dst_h as f32) as usize);

    // Turned, a row of the screen is a column of the picture, so nothing can
    // be shared between rows: each pixel finds its own way back. Slower, and
    // only ever asked for by hand.
    if view.turn % 4 != 0 {
        let left = view.center[0] - vw / 2.0 / view.zoom;
        let top = view.center[1] - vh / 2.0 / view.zoom;
        dst.par_chunks_mut(dst_w as usize).enumerate().for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                if y < y_lo || y >= y_hi || x < x_lo || x >= x_hi {
                    *out = 0;
                    continue;
                }
                let shown = [
                    left + (x as f32 - vx + 0.5) / view.zoom,
                    top + (y as f32 - vy + 0.5) / view.zoom,
                ];
                let p = crate::view::unturn_point(view.turn, shown, source);
                let (tx, ty) = (p[0] * tex_scale.0 - 0.5, p[1] * tex_scale.1 - 0.5);
                *out = if tx < -1.0 || ty < -1.0 || tx > tw as f32 || ty > th as f32 {
                    0
                } else if smooth {
                    sample_bilinear(src, tw, th, tx, ty)
                } else {
                    sample_nearest(src, tw, th, tx, ty)
                };
            }
        });
        return;
    }

    // Every row samples the same columns, so where each lands in the texture
    // is worked out once rather than per pixel.
    let columns: Vec<f32> = (0..dst_w as usize).map(|x| origin.0 + (x as f32 + 0.5) * step.0 - 0.5).collect();

    dst.par_chunks_mut(dst_w as usize).enumerate().for_each(|(y, row)| {
        // Outside the viewport belongs to whatever else is on screen.
        if y < y_lo || y >= y_hi {
            row.fill(0);
            return;
        }
        let ty = origin.1 + (y as f32 + 0.5) * step.1 - 0.5;

        // Rows entirely outside the image are a memset, which is most of the
        // work when a portrait image sits on a landscape screen.
        if ty < -1.0 || ty > th as f32 {
            row.fill(0);
            return;
        }

        for (x, (out, &tx)) in row.iter_mut().zip(&columns).enumerate() {
            if x < x_lo || x >= x_hi {
                *out = 0;
                continue;
            }
            *out = if smooth {
                sample_bilinear(src, tw, th, tx, ty)
            } else {
                sample_nearest(src, tw, th, tx, ty)
            };
        }
    });
}

#[inline]
fn texel(src: &[u8], tw: i32, th: i32, x: i32, y: i32) -> [u8; 4] {
    // Clamp to edge, matching the GPU sampler's address mode.
    let x = x.clamp(0, tw - 1) as usize;
    let y = y.clamp(0, th - 1) as usize;
    let i = (y * tw as usize + x) * 4;
    [src[i], src[i + 1], src[i + 2], src[i + 3]]
}

#[inline]
fn pack(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) << 16 | (g as u32) << 8 | b as u32
}

#[inline]
fn sample_nearest(src: &[u8], tw: i32, th: i32, tx: f32, ty: f32) -> u32 {
    let (x, y) = ((tx + 0.5).floor() as i32, (ty + 0.5).floor() as i32);
    if x < 0 || y < 0 || x >= tw || y >= th {
        return 0;
    }
    let p = texel(src, tw, th, x, y);
    pack(p[0], p[1], p[2])
}

#[inline]
fn sample_bilinear(src: &[u8], tw: i32, th: i32, tx: f32, ty: f32) -> u32 {
    if tx < -1.0 || ty < -1.0 || tx > tw as f32 || ty > th as f32 {
        return 0;
    }
    let (x0, y0) = (tx.floor(), ty.floor());
    let (fx, fy) = (tx - x0, ty - y0);
    let (x0, y0) = (x0 as i32, y0 as i32);

    let p00 = texel(src, tw, th, x0, y0);
    let p10 = texel(src, tw, th, x0 + 1, y0);
    let p01 = texel(src, tw, th, x0, y0 + 1);
    let p11 = texel(src, tw, th, x0 + 1, y0 + 1);

    // Fixed point in 8.8: the weights are all in [0,1] and the channels are
    // bytes, so this stays exact and avoids four float round-trips per pixel.
    let wx = (fx * 256.0) as u32;
    let wy = (fy * 256.0) as u32;
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as u32 * (256 - wx) + p10[c] as u32 * wx;
        let bot = p01[c] as u32 * (256 - wx) + p11[c] as u32 * wx;
        out[c] = ((top * (256 - wy) + bot * wy) >> 16) as u8;
    }
    pack(out[0], out[1], out[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2x2 image: red, green / blue, white.
    fn quad() -> Vec<u8> {
        vec![
            255, 0, 0, 255, 0, 255, 0, 255, // row 0
            0, 0, 255, 255, 255, 255, 255, 255, // row 1
        ]
    }

    #[test]
    fn nearest_picks_the_covering_texel() {
        let src = quad();
        assert_eq!(sample_nearest(&src, 2, 2, 0.0, 0.0), pack(255, 0, 0));
        assert_eq!(sample_nearest(&src, 2, 2, 1.0, 0.0), pack(0, 255, 0));
        assert_eq!(sample_nearest(&src, 2, 2, 0.0, 1.0), pack(0, 0, 255));
        // Outside the image reads as background, not as a clamped edge texel.
        assert_eq!(sample_nearest(&src, 2, 2, -2.0, 0.0), 0);
    }

    #[test]
    fn bilinear_midpoint_averages_all_four() {
        let src = quad();
        // Dead centre of a 2x2 image averages red, green, blue and white.
        let got = sample_bilinear(&src, 2, 2, 0.5, 0.5);
        let (r, g, b) = ((got >> 16) & 0xFF, (got >> 8) & 0xFF, got & 0xFF);
        for (name, v) in [("r", r), ("g", g), ("b", b)] {
            assert!((v as i32 - 127).abs() <= 2, "{name}={v} in {got:#08x}");
        }
    }

    #[test]
    fn a_fitted_image_lands_where_the_gpu_would_put_it() {
        // 2x2 image scaled to fill a 4x4 window: zoom 2, centred.
        let src = quad();
        let view = View { zoom: 2.0, center: [1.0, 1.0], autoscale: true, ..View::default() };
        let mut dst = vec![0u32; 16];
        rasterise(&mut dst, 4, 4, &src, (2, 2), &view, (2.0, 2.0), false, (0.0, 0.0, 4.0, 4.0));

        // Corners of the window should hold the corners of the image.
        assert_eq!(dst[0], pack(255, 0, 0), "top-left");
        assert_eq!(dst[3], pack(0, 255, 0), "top-right");
        assert_eq!(dst[12], pack(0, 0, 255), "bottom-left");
        assert_eq!(dst[15], pack(255, 255, 255), "bottom-right");
    }

    #[test]
    fn a_viewport_leaves_the_rest_of_the_buffer_alone() {
        let src = quad();
        let view = View { zoom: 2.0, center: [1.0, 1.0], autoscale: true, ..View::default() };
        let mut dst = vec![0u32; 8 * 4];
        // Draw into the left half only.
        rasterise(&mut dst, 8, 4, &src, (2, 2), &view, (2.0, 2.0), false, (0.0, 0.0, 4.0, 4.0));

        for y in 0..4 {
            assert_ne!(dst[y * 8], 0, "the viewport should have been drawn");
            for x in 4..8 {
                assert_eq!(dst[y * 8 + x], 0, "outside the viewport must stay clear");
            }
        }
    }

    #[test]
    fn a_reduced_decode_still_covers_the_same_rectangle() {
        // A 2x2 texture standing in for a 200x200 source, fitted to 4x4.
        let src = quad();
        let view =
            View { zoom: 0.02, center: [100.0, 100.0], autoscale: true, ..View::default() };
        let mut dst = vec![0u32; 16];
        rasterise(&mut dst, 4, 4, &src, (2, 2), &view, (200.0, 200.0), false, (0.0, 0.0, 4.0, 4.0));

        assert_eq!(dst[0], pack(255, 0, 0), "top-left");
        assert_eq!(dst[15], pack(255, 255, 255), "bottom-right");
    }
}
