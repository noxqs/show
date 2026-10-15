//! Headless end-to-end profiling: file on disk to pixels in a framebuffer,
//! without ever showing a window.
//!
//! This runs the same `Core`/`Renderer` code the viewer does, rendering into an
//! offscreen texture the size of the display instead of a swapchain. What it
//! cannot measure is the compositor's own present latency; everything upstream
//! of that -- read, decode, GPU bring-up, texture upload, draw -- is exactly
//! what the viewer pays.
//!
//! It also times a CPU-only path over the same decoded pixels, which is the
//! honest way to answer "does the GPU actually help here".

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use rayon::prelude::*;

use crate::gpu::{Core, Renderer};
use crate::loader::{self, Io, human_bytes};
use crate::scale;
use crate::timing;

/// Offscreen target format, matching what a real swapchain hands us.
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

/// Primary monitor size, for sizing the offscreen target.
///
/// Monitor enumeration lives on `ActiveEventLoop`, so we have to enter the
/// event loop to ask -- but we exit from `resumed` without creating a window,
/// so nothing appears on screen.
pub fn detect_screen() -> Option<(u32, u32)> {
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::window::WindowId;

    struct Probe(Option<(u32, u32)>);

    impl ApplicationHandler for Probe {
        fn resumed(&mut self, el: &ActiveEventLoop) {
            self.0 = el
                .primary_monitor()
                .or_else(|| el.available_monitors().next())
                .map(|m| (m.size().width, m.size().height));
            el.exit();
        }
        fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
    }

    let mut probe = Probe(None);
    EventLoop::new().ok()?.run_app(&mut probe).ok()?;
    probe.0
}

pub struct Options {
    pub iterations: usize,
    pub io: Io,
    pub adapter: Option<String>,
    pub screen: (u32, u32),
    pub backends: wgpu::Backends,
    pub preview: Option<crate::decode::Target>,
    /// Write a machine-readable summary to stdout, for `--compare` to collect.
    /// Everything human-facing goes to stderr, so stdout stays clean.
    pub emit_summary: bool,
}

/// The phases `--compare` ranks adapters on.
pub struct Measured {
    pub file: String,
    pub adapter: String,
    /// Instance, adapter, device and pipeline: paid once, before any image.
    pub init: Duration,
    /// Read and decode. Independent of the adapter, but reported for context.
    pub decode: Duration,
    /// Texture allocation, the copy, and waiting for it to land.
    pub upload: Duration,
    pub draw: Duration,
    /// Scale and blit on the CPU with no graphics API involved at all.
    pub cpu: Duration,
}

impl Measured {

    fn summary_line(&self) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        format!(
            "show-summary\tfile={}\tadapter={}\tinit={:.3}\tdecode={:.3}\tupload={:.3}\tdraw={:.3}\tcpu={:.3}",
            self.file,
            self.adapter,
            ms(self.init),
            ms(self.decode),
            ms(self.upload),
            ms(self.draw),
            ms(self.cpu)
        )
    }
}

pub fn run(paths: &[std::path::PathBuf], opts: &Options) -> Result<()> {
    eprintln!(
        "profiling {} file(s) headless, target {}x{}, {} repeat(s)",
        paths.len(),
        opts.screen.0,
        opts.screen.1,
        opts.iterations
    );

    let mut failures = 0;
    for path in paths {
        match profile_one(path, opts) {
            Ok(m) if opts.emit_summary => println!("{}", m.summary_line()),
            Ok(_) => {}
            Err(e) => {
                eprintln!("  {}: {e:#}", path.display());
                failures += 1;
            }
        }
    }
    if failures > 0 && failures == paths.len() {
        anyhow::bail!("every file failed to profile");
    }
    Ok(())
}

fn profile_one(path: &Path, opts: &Options) -> Result<Measured> {
    let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
    eprintln!("\n═══ {name} ═══");

    timing::reset();

    // --- the cold path, exactly as the viewer runs it ---
    let t = Instant::now();
    let loaded = loader::load(path, opts.io, opts.preview)?;
    let decode_t = t.elapsed();
    let d = &loaded.decoded;

    let t = Instant::now();
    let core = Core::new_default(opts.adapter.as_deref(), opts.backends)?;
    let renderer = Renderer::new(&core.device, TARGET_FORMAT);
    let init_t = t.elapsed();
    let target = offscreen_target(&core.device, opts.screen);

    let t = Instant::now();
    let uploaded = core.upload_mipmapped(&d.rgba, d.width, d.height);
    let bound = renderer.bind(&core.device, &uploaded.view);
    let upload_t = t.elapsed();

    let t = Instant::now();
    {
        // Draw and wait for the GPU to actually finish, so this is real work
        // rather than a queue submission returning immediately.
        let _s = timing::span("gpu", "draw");
        renderer.draw(&core, &target, &[crate::gpu::Layer { bound: &bound, rect: [-1.0, -1.0, 1.0, 1.0], uv: crate::gpu::FULL_UV, smooth: true, turn: 0 }]);
        let _ = core.device.poll(wgpu::PollType::Wait);
    }
    let draw_t = t.elapsed();

    timing::report(&format!("{name} — cold start"));

    eprintln!(
        "  {}x{} {} · {} on disk · {} decoded · {} · uploaded {}x{}{}",
        d.width,
        d.height,
        d.format.name(),
        human_bytes(loaded.file_size),
        human_bytes(d.rgba.len() as u64),
        core.adapter_name,
        uploaded.uploaded.0,
        uploaded.uploaded.1,
        if uploaded.downscaled_by > 1 {
            format!(" (downscaled 1/{}, over the {}px texture limit)", uploaded.downscaled_by, core.max_texture)
        } else {
            String::new()
        }
    );

    // --- repeats, with the GPU already warm ---
    if opts.iterations > 1 {
        repeat(path, opts, &core, &renderer, &target)?;
    }

    // --- the CPU-only comparison over the same pixels ---
    let cpu_t = cpu_comparison(&d.rgba, d.width, d.height, opts, &uploaded.uploaded);

    Ok(Measured {
        file: name,
        adapter: core.adapter_name.clone(),
        init: init_t,
        decode: decode_t,
        upload: upload_t,
        draw: draw_t,
        cpu: cpu_t,
    })
}

fn offscreen_target(device: &wgpu::Device, screen: (u32, u32)) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d { width: screen.0, height: screen.1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

fn repeat(
    path: &Path,
    opts: &Options,
    core: &Core,
    renderer: &Renderer,
    target: &wgpu::TextureView,
) -> Result<()> {
    let mut decode_t = Vec::new();
    let mut upload_t = Vec::new();
    let mut draw_t = Vec::new();

    for _ in 1..opts.iterations {
        // Timing is suppressed for the repeats so the cold waterfall above
        // stays the only thing in the event log.
        let t = Instant::now();
        let loaded = loader::load(path, opts.io, opts.preview)?;
        decode_t.push(t.elapsed());
        let d = &loaded.decoded;

        let t = Instant::now();
        let uploaded = core.upload_mipmapped(&d.rgba, d.width, d.height);
        let bound = renderer.bind(&core.device, &uploaded.view);
        upload_t.push(t.elapsed());

        let t = Instant::now();
        renderer.draw(core, target, &[crate::gpu::Layer { bound: &bound, rect: [-1.0, -1.0, 1.0, 1.0], uv: crate::gpu::FULL_UV, smooth: true, turn: 0 }]);
        let _ = core.device.poll(wgpu::PollType::Wait);
        draw_t.push(t.elapsed());
    }

    let (io_decode, upload, draw) = (median(decode_t), median(upload_t), median(draw_t));
    eprintln!("\n  warm medians over {} repeat(s)", opts.iterations - 1);
    eprintln!("    io + decode           {:>9.3} ms", io_decode.as_secs_f64() * 1e3);
    eprintln!("    alloc + upload + wait {:>9.3} ms", upload.as_secs_f64() * 1e3);
    eprintln!("    draw + wait           {:>9.3} ms", draw.as_secs_f64() * 1e3);
    eprintln!(
        "    ── total to pixels    {:>9.3} ms",
        (io_decode + upload + draw).as_secs_f64() * 1e3
    );
    Ok(())
}

/// What the same frame would cost with no GPU at all: scale the decoded image
/// to fit the display, then compose it into a screen-sized BGRA framebuffer.
///
/// The scale here is an integer box filter, which is the *cheapest* correct
/// downscale available; a real software renderer needing a fractional factor
/// would pay more. So this is a lower bound on the CPU path, deliberately
/// generous to the CPU side of the comparison.
fn cpu_comparison(rgba: &[u8], w: u32, h: u32, opts: &Options, gpu_uploaded: &(u32, u32)) -> Duration {
    let (sw, sh) = opts.screen;
    let factor = scale::factor_for_limit(w, h, sw.max(sh));

    let t = Instant::now();
    let (scaled, scaled_w, scaled_h) = scale::box_downscale(rgba, w, h, factor);
    let scale_t = t.elapsed();

    let t = Instant::now();
    let mut framebuffer = vec![0u8; sw as usize * sh as usize * 4];
    blit_bgra(&scaled, scaled_w, scaled_h, &mut framebuffer, sw, sh);
    let blit_t = t.elapsed();
    std::hint::black_box(&framebuffer);

    eprintln!("\n  cpu-only path over the same decoded pixels (lower bound)");
    eprintln!(
        "    box downscale /{factor} to {scaled_w}x{scaled_h}  {:>9.3} ms",
        scale_t.as_secs_f64() * 1e3
    );
    eprintln!(
        "    blit into {sw}x{sh} bgra framebuffer  {:>9.3} ms",
        blit_t.as_secs_f64() * 1e3
    );
    eprintln!(
        "    ── cpu total          {:>9.3} ms   (gpu holds {}x{} and rescales for free on every zoom)",
        (scale_t + blit_t).as_secs_f64() * 1e3,
        gpu_uploaded.0,
        gpu_uploaded.1
    );
    scale_t + blit_t
}

/// Centre `src` in `dst`, converting RGBA to BGRA on the way.
fn blit_bgra(src: &[u8], sw: u32, sh: u32, dst: &mut [u8], dw: u32, dh: u32) {
    let copy_w = sw.min(dw) as usize;
    let copy_h = sh.min(dh) as usize;
    let x0 = (dw as usize - copy_w) / 2;
    let y0 = (dh as usize - copy_h) / 2;

    dst.par_chunks_mut(dw as usize * 4)
        .skip(y0)
        .take(copy_h)
        .enumerate()
        .for_each(|(row, dst_row)| {
            let src_row = &src[row * sw as usize * 4..][..copy_w * 4];
            let out = &mut dst_row[x0 * 4..(x0 + copy_w) * 4];
            for (s, d) in src_row.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            }
        });
}
