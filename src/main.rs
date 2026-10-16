mod alsa;
mod app;
mod audio;
mod awake;
mod bench;
mod browser;
mod cli;
mod config;
mod crash;
mod compare;
mod decode;
mod dmabuf;
mod font;
mod geo;
mod framecache;
mod gpu;
mod heif;
mod install;
mod keys;
mod ml;
mod loader;
mod md5;
mod mosaic;
mod osd;
mod panel;
mod playlist;
mod plugins;
mod profile;
mod pyramid;
mod scale;
mod screenshot;
mod sha256;
mod sheet;
mod similarity;
mod software;
mod subtitles;
mod thumbs;
mod typeface;
mod tiled;
mod tray;
mod tiles;
mod timing;
mod turbojpeg;
mod video;
mod view;
mod vram;
mod watch;
mod zip;

use std::process::ExitCode;

/// How this binary identifies itself: package version, then a build number
/// that increases with every release build, then when it was built.
pub fn version() -> String {
    let commit = env!("SHOW_COMMIT");
    format!(
        "show {} build {} ({}){}",
        env!("CARGO_PKG_VERSION"),
        env!("SHOW_BUILD"),
        env!("SHOW_BUILD_DATE"),
        if commit.is_empty() { String::new() } else { format!(" {commit}") },
    )
}

/// The version without the program's name, for places that already say it.
pub fn version_short() -> String {
    let commit = env!("SHOW_COMMIT");
    format!(
        "{} build {} ({}){}",
        env!("CARGO_PKG_VERSION"),
        env!("SHOW_BUILD"),
        env!("SHOW_BUILD_DATE"),
        if commit.is_empty() { String::new() } else { format!(" {commit}") },
    )
}

/// Just the number, for comparing two builds.
pub fn build_number() -> u64 {
    env!("SHOW_BUILD").parse().unwrap_or(0)
}

/// Run the detector over one picture and say what it found.
///
/// The smallest thing that exercises the whole path — runtime, model, prepare,
/// infer, decode — with no window, no video and no timing to confuse a failure
/// with. If this works, everything after it is plumbing.
#[cfg(feature = "ml")]
fn detect_in(path: &std::path::Path, args: &cli::Args) -> anyhow::Result<()> {
    use std::time::Instant;

    println!("onnx runtime: {}", ml::runtime::describe());
    match ml::runtime::on_card() {
        None => println!("graphics card: in use"),
        Some(why) => println!("graphics card: not used — {why}"),
    }

    // The same choice the viewer makes, made in the same place, so this
    // reports on what the app would actually run.
    let which = ml::watcher::chosen();
    let outlines = which != ml::catalogue::Capability::Detect;
    let opened = Instant::now();
    let mut detector = ml::vision::Detector::open_for(which)?;
    println!(
        "model loaded in {:.0} ms — {} — {}",
        opened.elapsed().as_secs_f64() * 1000.0,
        ml::catalogue::spec(which).version,
        detector.describe()
    );

    // A still picture, through the same loader the viewer uses.
    let loaded = loader::load(path, args.io, args.preview)?;
    let picture = &loaded.decoded;
    let frame = ml::prep::from_pixels(&picture.rgba, picture.width, picture.height)
        .ok_or_else(|| anyhow::anyhow!("the picture has no size"))?;
    println!("\n{}: {}x{}", path.display(), frame.width, frame.height);

    let mut input = ml::prep::Input::new(ml::prep::DEFAULT_SIZE);
    let prepared = Instant::now();
    ml::prep::letterbox(&frame, &mut input);
    let prep_ms = prepared.elapsed().as_secs_f64() * 1000.0;

    // The first inference includes graph optimisation and arena allocation, so
    // it is reported separately rather than averaged into a number that would
    // flatter or damn the model depending on how many pictures were run.
    let one = |detector: &mut ml::vision::Detector| -> anyhow::Result<Vec<ml::segment::Segmented>> {
        // The saved thresholds, so a benchmark counts what the viewer would
        // have drawn rather than what the defaults would have.
        let saved = config::Config::load();
        let limits =
            ml::vision::Thresholds::new(saved.min_confidence, saved.duplicate_overlap);
        if outlines {
            Ok(detector.look(&input, limits)?)
        } else {
            Ok(detector
                .run(&input, limits)?
                .into_iter()
                .map(ml::segment::Segmented::from_box)
                .collect())
        }
    };

    let first = Instant::now();
    let mut found = one(&mut detector)?;
    let first_ms = first.elapsed().as_secs_f64() * 1000.0;

    let again = Instant::now();
    let _ = one(&mut detector)?;
    let warm_ms = again.elapsed().as_secs_f64() * 1000.0;

    println!("prepare {prep_ms:.1} ms · first inference {first_ms:.0} ms · warm {warm_ms:.0} ms");

    if found.is_empty() {
        println!("\nnothing found above {:.0}% confidence", ml::vision::MIN_CONFIDENCE * 100.0);
        return Ok(());
    }
    println!(
        "\n{:>5}  {:<16} {:<38} {}",
        "CONF", "WHAT", "WHERE (left top right bottom)", "OUTLINE"
    );
    for item in &found {
        let (x1, y1, x2, y2) = item.found.bounds;
        // How much of the box the outline actually covers: a mask that filled
        // its whole box would be no better than the box, and the number says
        // straight away whether the prototypes decoded into anything.
        let outline = if item.has_outline() {
            let cells = item.size.0 * item.size.1;
            let inside = item.mask.iter().filter(|c| **c > ml::segment::INSIDE).count();
            format!("{}x{} · {:.0}% of box", item.size.0, item.size.1,
                    inside as f32 / cells.max(1) as f32 * 100.0)
        } else {
            "box only".to_string()
        };
        println!(
            "{:>4.0}%  {:<16} {x1:>6.0} {y1:>6.0} {x2:>6.0} {y2:>6.0}{:<8} {outline}",
            item.found.confidence * 100.0,
            item.found.label(),
            ""
        );
    }
    println!("\n{} found", found.len());

    if let Some(dir) = args.dump_ui.as_deref() {
        let source = (frame.width, frame.height);

        // Ids before drawing, not after: the colour a thing is drawn in comes
        // from its tracker id, so a dump made before the tracker had run would
        // put every object in the same colour and show nothing of what the
        // viewer actually draws. The tracker is run twice because one sighting
        // is never followed — which is also what a click needs.
        let mut tracker = ml::track::Tracker::new();
        let boxes: Vec<_> = found.iter().map(|s| s.found.clone()).collect();
        tracker.update(&boxes);
        let ids = tracker.update(&boxes);
        for (item, id) in found.iter_mut().zip(ids) {
            item.id = id;
        }

        let mut grid = ml::labels::Grid::for_picture(source);
        if found.iter().any(|s| s.has_outline()) {
            grid.draw_outlines(&found, &input, source);
        } else {
            grid.draw(&found, source);
        }
        dump_labels(dir, "outlines.png", &loaded.decoded, &grid)?;

        // And what clicking one of them leaves on screen.
        if let Some(first) = found.first() {
            let b = first.found.bounds;
            let middle = ((b.0 + b.2) / 2.0, (b.1 + b.3) / 2.0);
            match ml::segment::pick(&found, middle) {
                Some(chosen) => {
                    println!(
                        "clicking ({:.0}, {:.0}) follows {} #{}",
                        middle.0,
                        middle.1,
                        chosen.found.label(),
                        chosen.id
                    );
                    let mut only = ml::labels::Grid::for_picture(source);
                    only.draw_outlines_of(std::iter::once(chosen), &input, source);
                    dump_labels(dir, "following.png", &loaded.decoded, &only)?;

                    // And what pointing at the same spot cuts out, which is a
                    // different model answering a different question.
                    if ml::store::installed(ml::catalogue::Capability::PointSegment) {
                        cut_out_at(dir, &frame, &loaded.decoded, middle)?;
                        sharpen_all(dir, &frame, &loaded.decoded, &found)?;
                    }
                }
                None => println!("clicking ({:.0}, {:.0}) follows nothing", middle.0, middle.1),
            }
        }
    }
    Ok(())
}

/// Outline everything the detector found, with the cut-out model.
///
/// The detector says what and roughly where at 160 cells across the whole
/// frame; this says exactly where, at 1024, for each of them — one encode of
/// the picture and a few milliseconds per thing.
#[cfg(feature = "ml")]
fn sharpen_all(
    dir: &std::path::Path,
    frame: &video::Frame,
    picture: &decode::Decoded,
    found: &[ml::segment::Segmented],
) -> anyhow::Result<()> {
    use std::time::Instant;

    let boxes: Vec<_> = found.iter().map(|s| s.found.clone()).collect();
    if boxes.is_empty() {
        return Ok(());
    }
    let source = (frame.width, frame.height);
    let mut cutter = ml::cutout::Cutter::open()?;
    let square = ml::cutout::square(frame);
    let embedding = cutter.embed(&square)?;

    let started = Instant::now();
    let sharp = cutter.refine(&embedding, &boxes, source)?;
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "\nsharpened {} of {} boxes in {ms:.0} ms ({:.0} ms each)",
        sharp.len(),
        boxes.len(),
        ms / boxes.len().max(1) as f64
    );
    for cut in &sharp {
        let inside = cut.mask.iter().filter(|c| **c > ml::segment::INSIDE).count();
        println!(
            "  {:<16} {}x{} cells, {:.0}% of them inside",
            cut.found.label(),
            cut.size.0,
            cut.size.1,
            inside as f32 / (cut.size.0 * cut.size.1).max(1) as f32 * 100.0
        );
    }

    let geometry = ml::cutout::geometry(source);
    let mut grid = ml::labels::Grid::for_picture(source);
    grid.add_outlines(sharp.iter(), &geometry, source);
    dump_labels(dir, "sharpened.png", picture, &grid)
}

/// Cut out whatever is at one point, and write the result as a picture.
///
/// The same three steps the click does — square, embed, decode — timed
/// separately, because the split between them is the whole reason a second
/// click is instant and the first is not.
#[cfg(feature = "ml")]
fn cut_out_at(
    dir: &std::path::Path,
    frame: &video::Frame,
    picture: &decode::Decoded,
    at: (f32, f32),
) -> anyhow::Result<()> {
    use std::time::Instant;

    let source = (frame.width, frame.height);
    let mut cutter = ml::cutout::Cutter::open()?;

    let squared = Instant::now();
    let square = ml::cutout::square(frame);
    let square_ms = squared.elapsed().as_secs_f64() * 1000.0;

    let embedded = Instant::now();
    let embedding = cutter.embed(&square)?;
    let embed_ms = embedded.elapsed().as_secs_f64() * 1000.0;

    let cutting = Instant::now();
    let cut = cutter.cut(&embedding, at, source)?;
    let cut_ms = cutting.elapsed().as_secs_f64() * 1000.0;

    println!(
        "cut out: prepare {square_ms:.0} ms · encode {embed_ms:.0} ms · point {cut_ms:.0} ms"
    );
    let Some(cut) = cut else {
        println!("  nothing separable at ({:.0}, {:.0})", at.0, at.1);
        return Ok(());
    };
    let (x1, y1, x2, y2) = cut.found.bounds;
    let inside = cut.mask.iter().filter(|c| **c > ml::segment::INSIDE).count();
    println!(
        "  {:.0}% sure · {x1:.0} {y1:.0} {x2:.0} {y2:.0} · {}x{} cells, {:.0}% of them inside",
        cut.found.confidence * 100.0,
        cut.size.0,
        cut.size.1,
        inside as f32 / (cut.size.0 * cut.size.1).max(1) as f32 * 100.0
    );

    let geometry = ml::cutout::geometry(source);
    let mut grid = ml::labels::Grid::for_picture(source);
    grid.draw_outlines_of(std::iter::once(&cut), &geometry, source);
    dump_labels(dir, "cutout.png", picture, &grid)
}

/// Write what the mask shader would paint, over the picture it belongs to.
///
/// The shader itself cannot be inspected without a window, and the user is
/// working on this desktop. This runs the same three rules in software —
/// nothing where the grid is zero, an opaque edge where a cell's neighbour
/// differs, a faint fill inside — so a mask landing in the wrong place is
/// visible in a file rather than argued about.
#[cfg(feature = "ml")]
fn dump_labels(
    dir: &std::path::Path,
    name: &str,
    picture: &decode::Decoded,
    grid: &ml::labels::Grid,
) -> anyhow::Result<()> {
    use anyhow::Context;

    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let palette = ml::labels::palette();
    let (gw, gh) = (grid.size.0 as usize, grid.size.1 as usize);
    let (w, h) = (picture.width as usize, picture.height as usize);
    let mut rgba = picture.rgba.clone();

    let cell = |x: usize, y: usize| -> u8 {
        if x < gw && y < gh { grid.cells[y * gw + x] } else { 0 }
    };
    for py in 0..h {
        let gy = py * gh / h.max(1);
        for px in 0..w {
            let gx = px * gw / w.max(1);
            let here = cell(gx, gy);
            if here == 0 {
                continue;
            }
            let edge = [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)].iter().any(|(dx, dy)| {
                let (nx, ny) = (gx as i64 + dx, gy as i64 + dy);
                nx < 0 || ny < 0 || cell(nx as usize, ny as usize) != here
            });
            let colour = palette[here as usize % palette.len()];
            let alpha = if edge { 1.0 } else { FILL };
            let at = (py * w + px) * 4;
            for c in 0..3 {
                let was = rgba[at + c] as f32 / 255.0;
                let now = colour[c] * alpha + was * (1.0 - alpha);
                rgba[at + c] = (now * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }

    let path = dir.join(name);
    let file = std::fs::File::create(&path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&rgba)?;
    println!("\n{}  {w}x{h}  from a {gw}x{gh} grid", path.display());
    Ok(())
}

/// How much of the picture a mask's interior shows through; the shader's own
/// figure, kept here so the specimen and the screen agree.
#[cfg(feature = "ml")]
const FILL: f32 = 0.16;

/// Without inference compiled in, say so rather than pretending.
#[cfg(not(feature = "ml"))]
fn detect_in(_path: &std::path::Path, _args: &cli::Args) -> anyhow::Result<()> {
    anyhow::bail!("this build has no inference; rebuild with `--features ml`")
}

/// The plugins page, without the page.
///
/// Everything F8 does, scriptable and headless — which is how the download,
/// digest and install path gets tested without a window, and how a puzzling
/// failure gets diagnosed without one either.
fn plugins_command(words: &[String]) -> anyhow::Result<()> {
    use anyhow::Context;
    use ml::catalogue::{CATALOGUE, Capability, spec};
    use ml::store;

    let verb = words.first().map(String::as_str).unwrap_or("list");
    let named = |what: &str| -> anyhow::Result<Capability> {
        let id = words.get(1).map(String::as_str).unwrap_or_default();
        Capability::from_id(id).with_context(|| {
            let known: Vec<&str> = CATALOGUE.iter().map(|s| s.id.id()).collect();
            format!("{what} what? known plugins: {}", known.join(", "))
        })
    };

    match verb {
        "list" => {
            // Wide enough for the longest identifier there is, worked out
            // rather than guessed: a column that fits today's names and not
            // tomorrow's is a column that will be wrong again.
            let id_w = CATALOGUE.iter().map(|s| s.id.id().len()).max().unwrap_or(8).max(2);
            let name_w = CATALOGUE.iter().map(|s| s.name.len()).max().unwrap_or(20).max(4);
            println!("{:<id_w$} {:<name_w$} {:>10}  {}", "ID", "NAME", "SIZE", "STATE");
            for spec in CATALOGUE {
                let state = if store::installed(spec.id) {
                    match store::verify(spec.id) {
                        Ok(()) => "installed".to_string(),
                        Err(e) => format!("DAMAGED: {e:#}"),
                    }
                } else {
                    "not installed".to_string()
                };
                println!(
                    "{:<id_w$} {:<name_w$} {:>10}  {state}{}",
                    spec.id.id(),
                    spec.name,
                    loader::human_bytes(spec.bytes()),
                    if spec.id == ml::watcher::chosen() { " - in use" } else { "" }
                );
            }
            if let Some(root) = store::root() {
                println!("\ninto {}", root.display());
            }
        }
        "install" => {
            let id = named("install")?;
            let spec = spec(id);
            // Said before anything is fetched, not after.
            println!(
                "{} {} — {}\nlicence: {}\n{} to download",
                spec.name,
                spec.version,
                spec.what,
                spec.licence,
                loader::human_bytes(spec.bytes())
            );
            let mut last = String::new();
            store::install(id, &mut |step| {
                let line = match step {
                    store::Step::Fetching(done, total) => format!(
                        "  {} of {} ({:.0}%)",
                        loader::human_bytes(done),
                        loader::human_bytes(total),
                        done as f64 / total.max(1) as f64 * 100.0
                    ),
                    store::Step::Verifying => "  checking the digest".to_string(),
                    store::Step::Unpacking => "  unpacking".to_string(),
                };
                if line != last {
                    println!("{line}");
                    last = line;
                }
            })?;
            println!("installed {}", store::directory(spec).unwrap_or_default().display());
        }
        "remove" => {
            let id = named("remove")?;
            let freed = store::disk_used(id);
            store::remove(id)?;
            println!("removed {}, freeing {}", id.id(), loader::human_bytes(freed));
        }
        "verify" => {
            let id = named("verify")?;
            store::verify(id)?;
            println!("{} matches the catalogue", id.id());
        }
        "use" => {
            let id = named("use")?;
            anyhow::ensure!(id.is_looker(), "{} is not one of the alternatives", id.id());
            anyhow::ensure!(store::installed(id), "{} is not installed", id.id());
            store::set_preferred(id)?;
            println!("looking with {}", id.id());
        }
        other => {
            anyhow::bail!("unknown verb {other:?}; try list, install, use, remove or verify")
        }
    }
    Ok(())
}

/// Report what feeding a vision model would cost, before there is one.
///
/// Every frame a detector sees has to be letterboxed, converted from YUV and
/// written out as planar floats first, and that happens whether or not the
/// model itself turns out to be fast. If the preparation alone does not fit in
/// the frame budget, no choice of model rescues it — so it is worth knowing
/// before a byte of weights is downloaded.
fn probe_ml(path: &std::path::Path) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    // A quarter, matching what similarity.rs reserves and for the same reason:
    // taking every core would stall the decoding.
    println!("machine: {cores} cores, {} for inference", (cores / 4).max(1));
    println!(
        "onnx runtime: {}",
        match find_onnxruntime() {
            Some(found) => format!("found at {}", found.display()),
            None => "not found — F8 would offer to install it".to_string(),
        }
    );

    let mut player = video::Player::open_video_only(path, true)?;
    let (w, h) = (player.info.width, player.info.height);
    let rate = player.info.frame_rate.max(1.0);
    let size = ml::prep::DEFAULT_SIZE;
    println!(
        "\n{w}x{h} at {rate:.2} fps -> {size}x{size} planar rgb, {} per input",
        loader::human_bytes(ml::prep::bytes_per_input(size))
    );

    let mut input = ml::prep::Input::new(size);
    let mut prepared = 0u32;
    // Not decode time: the decoder runs on its own thread and stays ahead, so
    // this is only what it costs to take a frame that is already waiting.
    // Calling it decode would flatter the pipeline by hiding where the work is.
    let mut fetch_only = Duration::ZERO;
    let mut prep_only = Duration::ZERO;

    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(6) {
        let before = Instant::now();
        let Some(frame) = player.try_next() else {
            if player.finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
            continue;
        };
        fetch_only += before.elapsed();

        let before = Instant::now();
        ml::prep::letterbox(&frame, &mut input);
        prep_only += before.elapsed();
        prepared += 1;
    }

    anyhow::ensure!(prepared > 10, "too few frames to measure ({prepared})");
    let each = prep_only.as_secs_f64() / prepared as f64;
    let budget = 1.0 / rate;

    println!("\npreparing {prepared} frames");
    println!(
        "  fetch       {:>7.2} ms each  (a frame already decoded on its own thread)",
        fetch_only.as_secs_f64() / prepared as f64 * 1000.0
    );
    println!("  letterbox   {:>7.2} ms each", each * 1000.0);
    println!(
        "  together    {:>7.0} frames/s  ({:.1}x real time)",
        prepared as f64 / (fetch_only + prep_only).as_secs_f64(),
        prepared as f64 / (fetch_only + prep_only).as_secs_f64() / rate
    );
    println!(
        "  the frame budget at {rate:.2} fps is {:.1} ms; preparation is {:.1}% of it",
        budget * 1000.0,
        each / budget * 100.0
    );

    // What matters is whether preparation alone eats the budget. The model
    // still has to fit in what is left.
    if each > budget * 0.25 {
        println!("  TOO SLOW — preparation alone takes a quarter of the budget");
    } else {
        println!("  preparation leaves the budget essentially intact");
    }
    Ok(())
}

/// Where a system ONNX Runtime might already be, for reporting only.
fn find_onnxruntime() -> Option<std::path::PathBuf> {
    let mut roots = vec![
        std::path::PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        std::path::PathBuf::from("/usr/local/lib"),
    ];
    if let Some(data) = std::env::var_os("XDG_DATA_HOME").map(std::path::PathBuf::from) {
        roots.push(data.join("show/runtime"));
    } else if let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) {
        roots.push(home.join(".local/share/show/runtime"));
    }
    for root in roots {
        for name in ["libonnxruntime.so", "libonnxruntime.so.1"] {
            let candidate = root.join(name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Measure what a background frame cache would cost and buy.
///
/// Two questions, both answerable only by measurement: how fast frames can be
/// decoded into memory, and whether doing that behind a playing film disturbs
/// it. Guessing at either would be guessing at the whole feature.
fn bench_cache(path: &std::path::Path) -> anyhow::Result<()> {
    let probe = video::Player::open_video_only(path, true)?;
    let (w, h) = (probe.info.width, probe.info.height);
    // 4:2:0 planar: one byte of luma per pixel, half that again for chroma.
    let frame_bytes = (w as u64 * h as u64) * 3 / 2;
    drop(probe);

    println!("{w}x{h}, {} per frame", loader::human_bytes(frame_bytes));

    // --- how fast can frames be decoded into memory, each way ---
    println!("\ndecoding into memory");
    let mut source_fps = 25.0;
    let mut best_fps = 0.0f64;
    for hardware in [false, true] {
        let (fps, rate, actually_hardware, rate_fps) = fill_rate(path, hardware, frame_bytes)?;
        source_fps = rate_fps;
        best_fps = best_fps.max(fps);
        let how = match (hardware, actually_hardware) {
            (false, _) => "software",
            (true, true) => "hardware",
            // Asking for it is not getting it: some codecs have no VA-API
            // path on this machine and fall back without saying so.
            (true, false) => "hardware (fell back to software)",
        };
        println!(
            "  {how:<32} {fps:>6.0} frames/s  {:>5.1}x real time  {}/s",
            fps / rate_fps,
            loader::human_bytes(rate as u64)
        );
    }
    let fps = best_fps;

    for budget_gb in [1u64, 4, 16] {
        let budget = budget_gb << 30;
        let held = budget / frame_bytes.max(1);
        println!(
            "  {budget_gb:>2} GB holds {held} frames — {:.1} min of film, {:.0} s to fill",
            held as f64 / source_fps / 60.0,
            held as f64 / fps
        );
    }

    // --- does looking at the film disturb watching it? ---
    #[cfg(feature = "ml")]
    {
        println!("\npacing with detection running");
        let quiet = measure_pacing(path, None)?;
        let watching = measure_pacing_watched(path)?;
        let pointed = measure_pacing_pointed(path, 0)?;
        let sharpened = measure_pacing_pointed(path, 5)?;
        println!("  nothing looking          spread {:>5.1} ms   hitches {}", quiet.0 * 1000.0, quiet.1);
        println!("  detection running        spread {:>5.1} ms   hitches {}", watching.0 * 1000.0, watching.1);
        println!("  following a cut-out      spread {:>5.1} ms   hitches {}", pointed.0 * 1000.0, pointed.1);
        println!("  sharpening five outlines spread {:>5.1} ms   hitches {}", sharpened.0 * 1000.0, sharpened.1);
        // The gate: a readout is not worth a stutter.
        let nominal = 1.0 / source_fps;
        if watching.0 > nominal * 0.1 || watching.1 > quiet.1 + 2 {
            println!("  DETECTION DISTURBS THE FILM");
        } else {
            println!("  detection costs the film nothing measurable");
        }
    }

    // --- does filling a cache disturb the film being watched? ---
    println!("\npacing while a cache fills behind it");
    let quiet = measure_pacing(path, None)?;
    let soft = measure_pacing(path, Some(false))?;
    let hard = measure_pacing(path, Some(true))?;
    println!("  nothing behind it        spread {:>5.1} ms   hitches {}", quiet.0 * 1000.0, quiet.1);
    println!("  software filler          spread {:>5.1} ms   hitches {}", soft.0 * 1000.0, soft.1);
    println!("  hardware filler          spread {:>5.1} ms   hitches {}", hard.0 * 1000.0, hard.1);

    // A spread that grows into a visible fraction of a frame is the failure
    // this is looking for: a cache nobody asked for is not worth a stutter.
    let nominal = 1.0 / source_fps;
    for (name, (sd, hitches)) in [("software", soft), ("hardware", hard)] {
        if sd > nominal * 0.1 || hitches > quiet.1 + 2 {
            println!("  {name} filling DISTURBS the film");
        } else {
            println!("  {name} filling costs the film nothing measurable");
        }
    }
    Ok(())
}

/// Decode flat out for a few seconds and report the rate.
fn fill_rate(
    path: &std::path::Path,
    hardware: bool,
    frame_bytes: u64,
) -> anyhow::Result<(f64, f64, bool, f64)> {
    use std::time::{Duration, Instant};

    let mut filler = video::Player::open_video_only(path, hardware)?;
    let actually_hardware = filler.is_hardware();
    let source_fps = filler.info.frame_rate.max(1.0);

    let start = Instant::now();
    let mut frames = 0u64;
    // Kept, so the measurement includes the cost of retaining them rather than
    // only of decoding and dropping.
    let mut held: Vec<video::Frame> = Vec::new();
    while start.elapsed() < Duration::from_secs(5) {
        match filler.try_next() {
            Some(frame) => {
                frames += 1;
                held.push(frame);
            }
            None if filler.finished() => break,
            None => std::thread::sleep(Duration::from_micros(200)),
        }
    }
    let fps = frames as f64 / start.elapsed().as_secs_f64();
    Ok((fps, fps * frame_bytes as f64, actually_hardware, source_fps))
}

/// Play a file with detection running, and report the same (spread, hitches).
///
/// The gate for putting inference on the display path at all: a readout is not
/// worth a stutter, and the only way to know is to measure the same thing with
/// The same, with a cut-out being followed through the film.
///
/// A different shape of load from detection: the encoder is four times the
/// work of the largest detector and it runs back to back for as long as the
/// thing is followed. If anything in this design disturbs playback it is this,
/// so it is measured rather than assumed.
#[cfg(feature = "ml")]
fn measure_pacing_pointed(path: &std::path::Path, boxes: usize) -> anyhow::Result<(f64, usize)> {
    use std::time::{Duration, Instant};

    let mut player = video::Player::open(path, true)?;
    if let Some(track) = player.audio() {
        track.volume.set(0.0);
    }
    let pointer = Some(ml::pointer::Pointer::start());
    let mut asked = false;
    let mut last: Option<Instant> = None;
    let mut answered = 0usize;

    let mut next: Option<video::Frame> = None;
    let mut origin: Option<Instant> = None;
    let mut shown: Vec<(Instant, Duration)> = Vec::new();
    let mut cutter = pointer;
    let mut cutter = cutter.take();

    // Opening two models is a tenth of a second of heavy work, and measuring
    // it as though it were the steady state blames the wrong thing: in the
    // viewer it happens once, on the click, and is what the notice on screen
    // is for. So it is done first and waited for.
    let warm = Instant::now();
    while let Some(c) = cutter.as_ref()
        && warm.elapsed() < Duration::from_secs(20)
    {
        if let Some(frame) = player.try_next() {
            c.ask(
                ml::pointer::Subject::Frame(frame),
                0,
                Duration::ZERO,
                ml::pointer::Prompt::Point((0.5, 0.5)),
            );
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    while let Some(c) = cutter.as_mut()
        && warm.elapsed() < Duration::from_secs(20)
    {
        if !c.collect().is_empty() || c.failed().is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let start = Instant::now();

    while start.elapsed() < Duration::from_secs(6) {
        if next.is_none() {
            next = player.try_next();
        }
        let due = match (&next, player.audio_position()) {
            (Some(frame), Some(now)) => frame.at <= now,
            (Some(frame), None) => match origin {
                Some(origin) => Instant::now().duration_since(origin) >= frame.at,
                None => {
                    origin = Some(Instant::now() - frame.at);
                    true
                }
            },
            (None, _) => false,
        };
        if due {
            let frame = next.take().expect("checked above");
            // Exactly what the app does: one question at a time, asked again
            // about the newest frame as soon as the last one is answered.
            // The same pause between questions the viewer leaves.
            const BREATHE: Duration = Duration::from_millis(200);
            // Either one point, as following does, or a detector's worth of
            // boxes, as sharpening does.
            let prompt = if boxes == 0 {
                ml::pointer::Prompt::Point((0.5, 0.5))
            } else {
                let found = (0..boxes)
                    .map(|i| ml::vision::Found {
                        class: 0,
                        confidence: 0.9,
                        bounds: {
                            let x = i as f32 * 40.0;
                            (x, 40.0, x + 200.0, 400.0)
                        },
                    })
                    .collect();
                ml::pointer::Prompt::Boxes { found, size: (frame.width, frame.height) }
            };
            if let Some(cutter) = cutter.as_ref()
                && !asked
                && last.is_none_or(|t: Instant| t.elapsed() >= BREATHE)
            {
                last = Some(Instant::now());
                cutter.ask(
                    ml::pointer::Subject::Frame(frame.clone_ref()),
                    0,
                    frame.at,
                    prompt,
                );
                asked = true;
            }
            let back = cutter.as_mut().map(|c| c.collect()).unwrap_or_default();
            if !back.is_empty() {
                answered += back.len();
                asked = false;
            }
            shown.push((Instant::now(), frame.at));
            player.set_playing(true);
            continue;
        }
        std::thread::sleep(Duration::from_micros(500));
    }

    if let Some(why) = cutter.as_ref().and_then(|c| c.failed()) {
        println!("  cut out did not run: {}", why.lines().next().unwrap_or(why));
    } else {
        let what = if boxes == 0 { "followed a cut-out".into() } else { format!("sharpened {boxes} boxes") };
        println!("  {what}, answered {answered} times in six seconds");
    }

    let settled = shown.first().map(|(t, _)| *t + Duration::from_secs(1));
    let shown: Vec<(Instant, Duration)> =
        shown.into_iter().filter(|(t, _)| Some(*t) >= settled).collect();
    anyhow::ensure!(shown.len() > 10, "too few frames to judge");

    let errors: Vec<f64> = shown
        .windows(2)
        .map(|w| {
            w[1].0.duration_since(w[0].0).as_secs_f64()
                - w[1].1.saturating_sub(w[0].1).as_secs_f64()
        })
        .collect();
    // The same statistic the other two report. It was the maximum here at
    // first, which made following a cut-out look ten times worse than
    // detection when the two were in fact indistinguishable — a measurement
    // that is not comparable is not a measurement.
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let sd = (errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / errors.len() as f64).sqrt();
    let nominal = 1.0 / player.info.frame_rate.max(1.0);
    let hitches = errors.iter().filter(|e| e.abs() > nominal * 0.5).count();
    Ok((sd, hitches))
}

/// Play a file and report (spread, hitches), optionally with a second decoder
/// running flat out behind it.
#[cfg(feature = "ml")]
fn measure_pacing_watched(path: &std::path::Path) -> anyhow::Result<(f64, usize)> {
    use std::time::{Duration, Instant};

    let mut player = video::Player::open(path, true)?;
    if let Some(track) = player.audio() {
        track.volume.set(0.0);
    }
    let mut watcher = ml::watcher::Watcher::start();

    let nominal = 1.0 / player.info.frame_rate.max(1.0);
    let mut next: Option<video::Frame> = None;
    let mut origin: Option<Instant> = None;
    let mut shown: Vec<(Instant, Duration)> = Vec::new();
    let start = Instant::now();

    while start.elapsed() < Duration::from_secs(6) {
        if next.is_none() {
            next = player.try_next();
        }
        let due = match (&next, player.audio_position()) {
            (Some(frame), Some(now)) => frame.at <= now,
            (Some(frame), None) => match origin {
                Some(origin) => Instant::now().duration_since(origin) >= frame.at,
                None => {
                    origin = Some(Instant::now() - frame.at);
                    true
                }
            },
            (None, _) => false,
        };
        if due {
            let frame = next.take().expect("checked above");
            // Exactly where the player offers it, and on the same thread.
            watcher.offer(&frame, 0);
            shown.push((Instant::now(), frame.at));
            player.set_playing(true);
            continue;
        }
        std::thread::sleep(Duration::from_micros(500));
    }

    let stats = watcher.stats();
    if let Some(why) = watcher.failed() {
        println!("  detection did not run: {}", why.lines().next().unwrap_or(why));
    } else {
        println!(
            "  {} · examined {} frames, skipped {}, last inference {:.0} ms",
            watcher.kind(),
            stats.examined,
            stats.skipped,
            stats.last.as_secs_f64() * 1000.0
        );
    }

    let settled = shown.first().map(|(t, _)| *t + Duration::from_secs(1));
    let shown: Vec<(Instant, Duration)> =
        shown.into_iter().filter(|(t, _)| Some(*t) >= settled).collect();
    anyhow::ensure!(shown.len() > 10, "too few frames to judge");

    let errors: Vec<f64> = shown
        .windows(2)
        .map(|w| {
            w[1].0.duration_since(w[0].0).as_secs_f64()
                - w[1].1.saturating_sub(w[0].1).as_secs_f64()
        })
        .collect();
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let sd = (errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / errors.len() as f64).sqrt();
    let hitches = errors.iter().filter(|e| e.abs() > nominal * 0.5).count();
    Ok((sd, hitches))
}

/// Play a file and report (spread, hitches), optionally with a second decoder
/// running flat out behind it.
fn measure_pacing(
    path: &std::path::Path,
    filler_hardware: Option<bool>,
) -> anyhow::Result<(f64, usize)> {
    use std::time::{Duration, Instant};

    let mut player = video::Player::open(path, true)?;
    if let Some(track) = player.audio() {
        track.volume.set(0.0);
    }

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let filler = filler_hardware.map(|hardware| {
        let path = path.to_path_buf();
        let stop = stop.clone();
        std::thread::spawn(move || {
            // Exactly what a cache would do: decode as fast as it can and keep
            // what it decodes.
            let Ok(mut p) = video::Player::open_video_only(&path, hardware) else { return };
            let mut held: Vec<video::Frame> = Vec::new();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                match p.try_next() {
                    Some(frame) => held.push(frame),
                    None if p.finished() => break,
                    None => std::thread::sleep(Duration::from_micros(200)),
                }
            }
        })
    });

    let nominal = 1.0 / player.info.frame_rate.max(1.0);
    let mut next: Option<video::Frame> = None;
    let mut origin: Option<Instant> = None;
    let mut shown: Vec<(Instant, Duration)> = Vec::new();
    let start = Instant::now();

    while start.elapsed() < Duration::from_secs(6) {
        if next.is_none() {
            next = player.try_next();
        }
        let due = match (&next, player.audio_position()) {
            (Some(frame), Some(now)) => frame.at <= now,
            (Some(frame), None) => match origin {
                Some(origin) => Instant::now().duration_since(origin) >= frame.at,
                None => {
                    origin = Some(Instant::now() - frame.at);
                    true
                }
            },
            (None, _) => false,
        };
        if due {
            let at = next.as_ref().map(|f| f.at).unwrap_or_default();
            shown.push((Instant::now(), at));
            next = None;
            player.set_playing(true);
            continue;
        }
        std::thread::sleep(Duration::from_micros(500));
    }

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Some(filler) = filler {
        let _ = filler.join();
    }

    let settled = shown.first().map(|(t, _)| *t + Duration::from_secs(1));
    let shown: Vec<(Instant, Duration)> =
        shown.into_iter().filter(|(t, _)| Some(*t) >= settled).collect();
    anyhow::ensure!(shown.len() > 10, "too few frames to judge");

    let errors: Vec<f64> = shown
        .windows(2)
        .map(|w| {
            w[1].0.duration_since(w[0].0).as_secs_f64()
                - w[1].1.saturating_sub(w[0].1).as_secs_f64()
        })
        .collect();
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let sd = (errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / errors.len() as f64).sqrt();
    let hitches = errors.iter().filter(|e| e.abs() > nominal * 0.5).count();
    Ok((sd, hitches))
}

/// Play a file with no window at all and report how playback behaves.
///
/// Silent and invisible on purpose: it opens no window and plays at zero
/// gain, so it can run while the machine is being used for something else.
/// Everything it checks is mechanical — whether packets decode and resample,
/// whether the device accepts them, whether the clock tracks what has actually
/// been heard, whether frames land evenly, whether seeking arrives where it
/// was asked to. Whether it *looks* and *sounds* right still needs eyes and
/// ears; what this catches is the timing faults underneath.
fn probe_playback(path: &std::path::Path) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    println!("alsa: {}", if alsa::available() { "loaded" } else { "NOT AVAILABLE" });

    // A device, so hardware frames can go straight onto the card as they do
    // in the viewer; held for the whole probe, so every check below runs on
    // the frames the viewer would get.
    let _card = check_card(path)?;

    let mut player = video::Player::open(path, true)?;
    // A silent file is still worth every check that does not need sound —
    // which is most of them, and more than half the films on this machine are
    // silent. Bailing here left them completely unmeasured.
    let has_audio = match player.audio() {
        Some(track) => {
            println!(
                "audio: {} · {} ch · {} Hz -> {} Hz stereo",
                track.codec, track.channels, track.rate, audio::RATE
            );
            // Zero gain: the device still consumes, so the clock still
            // advances.
            track.volume.set(0.0);
            true
        }
        None => {
            println!("audio: none — pacing runs against the wall clock");
            false
        }
    };
    check_autostart(&mut player)?;

    // Pacing first, and before anything else touches the frame queue. The
    // checks below drain frames as fast as they decode, which drives the
    // decoder far ahead of the sound — a state the player never reaches, and
    // one that makes a pacing measurement meaningless.
    check_pacing(&mut player)?;

    if !has_audio {
        // Everything except the pause check, which is the only one that needs
        // a sound clock to measure. Skipping the rest left half the films on
        // this machine covered by nothing but a pacing figure.
        check_keyframes(&mut player)?;
        check_seek_after_end(&mut player)?;
        check_uncached_step(&mut player)?;
        check_backfill(&mut player, path)?;
        check_subtitles(&mut player, path)?;
        check_seek(&mut player)?;
        return Ok(());
    }

    check_seek_sync(&mut player)?;
    let start = Instant::now();
    let mut frames = 0u32;
    let mut last_video = Duration::ZERO;
    println!(
        "\n  {:>8} {:>10} {:>12} {:>10}",
        "wall s", "audio s", "decoded to", "lead ms"
    );

    while start.elapsed() < Duration::from_secs(4) {
        while let Some(frame) = player.try_next() {
            last_video = frame.at;
            frames += 1;
        }
        if start.elapsed().as_millis() % 500 < 20 {
            if let Some(at) = player.audio_position() {
                // The lead is the decoder running ahead into the queue, not a
                // sync error: this loop drains frames as fast as they appear,
                // whereas the player shows each one when the clock reaches it.
                println!(
                    "  {:>8.2} {:>10.3} {:>12.3} {:>10.1}",
                    start.elapsed().as_secs_f64(),
                    at.as_secs_f64(),
                    last_video.as_secs_f64(),
                    (last_video.as_secs_f64() - at.as_secs_f64()) * 1000.0,
                );
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    let played = player.audio_position().unwrap_or_default();
    println!("\n{frames} video frames decoded, audio clock reached {:.2}s", played.as_secs_f64());
    anyhow::ensure!(played > Duration::from_secs(1), "the audio clock did not advance");
    println!("audio pipeline works");

    check_pause(&mut player)?;
    check_keyframes(&mut player)?;
    check_uncached_step(&mut player)?;
    check_backfill(&mut player, path)?;
    check_seek_after_end(&mut player)?;
    check_subtitles(&mut player, path)?;
    check_seek(&mut player)?;
    Ok(())
}

/// Whether hardware frames reach the card without a readback, and whether
/// what arrives there is the right picture: the first frame, read back from
/// the card, against the same frame decoded in software. Two decoders differ
/// by rounding; a wrongly imported surface (the wrong tiling, pitch or plane
/// order) differs by far more than that.
fn check_card(path: &std::path::Path) -> anyhow::Result<Option<gpu::Core>> {
    use std::time::{Duration, Instant};

    let core = match gpu::Core::new_default(None, wgpu::Backends::VULKAN) {
        Ok(core) => core,
        Err(e) => {
            println!("card: no Vulkan device ({e:#}), frames are read back");
            return Ok(None);
        }
    };
    println!("card: {}", core.adapter_name);

    let first = |hardware: bool| -> anyhow::Result<video::Frame> {
        let mut player = video::Player::open_video_only(path, hardware)?;
        let start = Instant::now();
        loop {
            if let Some(frame) = player.try_next() {
                return Ok(frame);
            }
            anyhow::ensure!(start.elapsed() < Duration::from_secs(5), "no frame within 5 s");
            std::thread::sleep(Duration::from_millis(2));
        }
    };

    let started = Instant::now();
    let hardware = first(true)?;
    let Some(card) = hardware.planes.on_card() else {
        println!("card: frames are read back (not hardware-decoded, or not the GPU that draws)");
        return Ok(Some(core));
    };
    println!(
        "card: frames imported, {}x{} {:?} + {:?}, first in {:.0} ms",
        hardware.width,
        hardware.height,
        card.y.format(),
        card.uv.format(),
        started.elapsed().as_secs_f64() * 1000.0
    );

    let read = Instant::now();
    let from_card = [hardware.planes[0].clone(), hardware.planes[1].clone(), hardware.planes[2].clone()];
    println!("card: read back in {:.1} ms", read.elapsed().as_secs_f64() * 1000.0);

    let software = first(false)?;
    let differ = |a: &[u8], b: &[u8]| -> f64 {
        let n = a.len().min(b.len()).max(1);
        a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / n as f64
    };
    let names = ["luma", "cb", "cr"];
    let mut worst = 0f64;
    for i in 0..3 {
        anyhow::ensure!(
            from_card[i].len() == software.planes[i].len(),
            "{} is {} bytes from the card, {} in software",
            names[i],
            from_card[i].len(),
            software.planes[i].len()
        );
        let d = differ(&from_card[i], &software.planes[i]);
        worst = worst.max(d);
        println!("card: {} differs from software by {d:.2} levels on average", names[i]);
    }
    anyhow::ensure!(worst < 4.0, "the frame on the card is not the decoded picture");
    println!("card: the imported frame matches");
    Ok(Some(core))
}

/// Confirm a film starts playing on its own, with nothing pressed.
///
/// Deliberately does not start the device by hand: the player starts it when
/// it shows the first picture, and that is the rule under test. Getting it
/// wrong deadlocks — the frame waits for a clock that waits for the frame —
/// and the symptom is a film that sits on its first frame until it is seeked.
/// Every other check here used to start the device itself, which is precisely
/// why none of them noticed.
fn check_autostart(player: &mut video::Player) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    let start = Instant::now();
    let mut next: Option<video::Frame> = None;
    let mut origin: Option<Instant> = None;
    let mut shown = 0u32;

    while start.elapsed() < Duration::from_secs(3) && shown < 5 {
        if next.is_none() {
            next = player.try_next();
        }
        let Some(frame) = next.as_ref() else {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        };
        // The player's own rule, copied: the sound leads once it is running,
        // and the wall clock stands in until then.
        let due = match player.audio_position() {
            Some(now) => frame.at <= now,
            None => {
                let origin = *origin.get_or_insert_with(|| Instant::now() - frame.at);
                Instant::now() >= origin + frame.at
            }
        };
        if due {
            next = None;
            shown += 1;
            // Exactly what the player does when it puts up a picture.
            player.set_playing(true);
        } else {
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    anyhow::ensure!(
        shown >= 5,
        "playback did not start on its own: {shown} frames in {:.1}s with nothing pressed",
        start.elapsed().as_secs_f64()
    );
    println!("starts on its own: {shown} frames in {:.0} ms", start.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

/// Present frames the way the player does and measure the spacing.
///
/// Judder is a timing fault, not a throughput one: a player can decode every
/// frame on time and still look wrong if it shows them at uneven moments. So
/// this measures the interval between presentations rather than counting them.
fn check_pacing(player: &mut video::Player) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    let nominal = 1.0 / player.info.frame_rate.max(1.0);
    let mut next: Option<video::Frame> = None;
    // When each frame was shown, and the moment it claims to be. Screen
    // recordings and anything else with a variable frame rate space their
    // frames unevenly on purpose, and showing them at their own timestamps is
    // correct — so error is measured against each frame's own spacing, not
    // against an average that such a file never had.
    let mut shown: Vec<(Instant, Duration)> = Vec::new();
    // Times round the loop with nothing decoded to show. Separates "the clock
    // handed frames out unevenly" from "the decoder could not keep up", which
    // look identical on screen and have nothing to do with each other.
    let mut starved = 0u64;
    let mut polls = 0u64;
    // Only used when the file is silent.
    let mut origin: Option<Instant> = None;

    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(6) {
        if next.is_none() {
            next = player.try_next();
        }
        polls += 1;
        if next.is_none() {
            starved += 1;
        }
        // With sound the speaker sets the timeline; without it the wall
        // clock does, anchored to the first frame — exactly what the player
        // falls back to.
        let due = match (&next, player.audio_position()) {
            (Some(frame), Some(now)) => frame.at <= now,
            (Some(frame), None) => match origin {
                Some(origin) => Instant::now().duration_since(origin) >= frame.at,
                None => {
                    origin = Some(Instant::now() - frame.at);
                    true
                }
            },
            (None, _) => false,
        };
        if due {
            let at = next.as_ref().map(|f| f.at).unwrap_or_default();
            shown.push((Instant::now(), at));
            next = None;
            continue;
        }
        std::thread::sleep(Duration::from_micros(500));
    }

    if shown.len() < 12 {
        // A clip shorter than the measurement is not a fault; say so and let
        // the rest of the checks run.
        println!("\ntoo short to judge pacing ({} frames)", shown.len());
        return Ok(());
    }

    // The first second is the pipeline filling: the codec has state to build,
    // the queue is empty and the clock has just been anchored. Judging steady
    // playback by it would condemn every file.
    let settled = shown[0].0 + Duration::from_secs(1);
    let shown: Vec<(Instant, Duration)> =
        shown.into_iter().filter(|(t, _)| *t >= settled).collect();
    if shown.len() < 12 {
        println!("\ntoo short to judge pacing ({} frames after start-up)", shown.len());
        return Ok(());
    }

    let gaps: Vec<f64> = shown.windows(2).map(|w| w[1].0.duration_since(w[0].0).as_secs_f64()).collect();
    // How late each frame was against its own timestamp, which is the only
    // question that means anything for a variable frame rate.
    let errors: Vec<f64> = shown
        .windows(2)
        .map(|w| {
            let wall = w[1].0.duration_since(w[0].0).as_secs_f64();
            let media = w[1].1.saturating_sub(w[0].1).as_secs_f64();
            wall - media
        })
        .collect();

    let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
    let error_mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let variance =
        errors.iter().map(|e| (e - error_mean).powi(2)).sum::<f64>() / errors.len() as f64;
    let sd = variance.sqrt();
    let worst = errors.iter().cloned().fold(0.0f64, |a, e| a.max(e.abs()));

    // Frames far enough from their own slot to be seen as a hitch.
    let late = errors.iter().filter(|e| e.abs() > nominal * 0.5).count();

    println!("\npacing over {} frames", shown.len());
    println!("  nominal   {:>7.1} ms", nominal * 1000.0);
    println!("  mean      {:>7.1} ms", mean * 1000.0);
    println!("  spread    {:>7.1} ms  ({:.0}% of a frame)", sd * 1000.0, sd / nominal * 100.0);
    println!("  worst     {:>7.1} ms off its own timestamp", worst * 1000.0);
    println!("  hitches   {late:>7}  (more than half a frame out)");
    println!("  starved   {:>7.1}%  of polls had no frame ready", starved as f64 / polls.max(1) as f64 * 100.0);
    if let Some(track) = player.audio() {
        use std::sync::atomic::Ordering;
        let under = track.clock.underruns.load(Ordering::Relaxed);
        let blocks = track.clock.blocks.load(Ordering::Relaxed);
        println!(
            "  audio     {under:>7} underruns in {blocks} blocks ({:.1}%)",
            under as f64 / blocks.max(1) as f64 * 100.0
        );
    }

    // A tenth of a frame of spread is about where uneven spacing stops being
    // merely measurable. Measured against the staircase this replaced: that
    // read 7.1 ms on a 40 ms frame, which was plainly visible.
    if sd > nominal * 0.1 {
        println!("  UNEVEN — spread is more than a tenth of a frame");
    } else {
        println!("  pacing is even");
    }
    if starved as f64 / polls.max(1) as f64 > 0.5 {
        println!("  STARVED — the decoder is not keeping up; pacing is not the problem");
    }
    Ok(())
}

/// Pause, confirm the timeline stops, resume, confirm it carries on.
///
/// The failure this is looking for is a resume that jumps: the device keeps
/// running while paused, so a clock that counts what the device consumed
/// rather than what it was given would skip the whole pause on resume.
fn check_pause(player: &mut video::Player) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    // Read the clock directly rather than through `audio_position`, which
    // reports nothing while the device is stopped — that is the correct answer
    // for deciding when to show a frame, and the wrong one for asking where
    // the timeline is sitting.
    let at = |p: &video::Player| {
        p.audio().and_then(|a| a.clock.position()).unwrap_or_default()
    };

    let before = at(player);
    player.set_playing(false);

    // Long enough that a clock which kept running would be obvious.
    let paused_for = Duration::from_millis(1200);
    let start = Instant::now();
    while start.elapsed() < paused_for {
        // Keep draining frames: a real pause holds the picture, and the queue
        // filling must not be what stops the clock.
        while player.try_next().is_some() {}
        std::thread::sleep(Duration::from_millis(20));
    }
    let during = at(player);

    player.set_playing(true);
    std::thread::sleep(Duration::from_millis(600));
    let after = at(player);

    let drift = during.as_secs_f64() - before.as_secs_f64();
    let resumed = after.as_secs_f64() - during.as_secs_f64();
    println!("\npause: {:.2}s -> {:.2}s while paused for {:.2}s, then {:.2}s after resuming 0.60s",
        before.as_secs_f64(), during.as_secs_f64(), paused_for.as_secs_f64(), after.as_secs_f64());
    println!("  drift while paused  {:>6.0} ms", drift * 1000.0);
    println!("  advance on resume   {:>6.0} ms", resumed * 1000.0);

    anyhow::ensure!(
        drift.abs() < 0.15,
        "the clock ran on while paused, by {:.0} ms — resuming would skip that far",
        drift * 1000.0
    );
    anyhow::ensure!(resumed > 0.3, "the clock did not restart: only {:.0} ms", resumed * 1000.0);
    println!("pause and resume work");
    Ok(())
}

/// Step keyframe to keyframe and report where each press lands.
///
/// Two things have to hold: each step moves, and it moves the right way. A
/// seek that rounds the wrong way lands back on the keyframe it started from,
/// which on screen looks like the key doing nothing at all.
fn check_keyframes(player: &mut video::Player) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    const NUDGE: Duration = Duration::from_millis(10);

    // Mirrors the player: nudge off the current position so the seek cannot
    // round back onto the keyframe already showing.
    // Mirrors the player: forwards measures from the play position, backwards
    // from where the picture on screen began. Returns where it landed, and the
    // keyframe that picture belongs to.
    let step = |player: &mut video::Player,
                position: Duration,
                keyframe: Duration,
                forward: bool|
     -> Option<(Duration, Duration)> {
        let target = if forward {
            position + NUDGE
        } else {
            keyframe.min(position).saturating_sub(NUDGE)
        };
        player.seek(target, forward);
        let deadline = Instant::now() + Duration::from_secs(4);
        while Instant::now() < deadline {
            if let Some(frame) = player.try_next() {
                let key = if frame.key { frame.at } else { keyframe };
                return Some((frame.at, key));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        None
    };

    // Let playback run on between presses, which is the case that matters:
    // paused, the position and the keyframe are the same instant, and a step
    // that measures from the wrong one still looks right.
    //
    // Advances by media time rather than wall time. Taking frames as fast as
    // they decode is not playback — it races seconds ahead in a moment, and
    // then measures a state the player never reaches.
    let drain = |player: &mut video::Player, state: &mut (Duration, Duration)| {
        let target = state.0 + Duration::from_millis(400);
        let deadline = Instant::now() + Duration::from_secs(3);
        while state.0 < target && Instant::now() < deadline {
            if let Some(frame) = player.try_next() {
                state.0 = frame.at;
                if frame.key {
                    state.1 = frame.at;
                }
            } else {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    };

    let mut state = step(player, Duration::from_secs(2), Duration::from_secs(2), false)
        .unwrap_or_default();
    let mut forward_steps = Vec::new();
    for _ in 0..4 {
        drain(player, &mut state);
        let Some(next) = step(player, state.0, state.1, true) else { break };
        forward_steps.push(next.0);
        state = next;
    }
    let mut back_steps = Vec::new();
    for _ in 0..3 {
        // Playing on between presses is what made this fail: the position
        // moves past the keyframe, and a step measured from the position
        // lands back on the keyframe already showing.
        drain(player, &mut state);
        let Some(prev) = step(player, state.0, state.1, false) else { break };
        back_steps.push(prev.0);
        state = prev;
    }

    let show = |v: &[Duration]| {
        v.iter().map(|d| format!("{:.2}", d.as_secs_f64())).collect::<Vec<_>>().join("  ")
    };
    println!("\nkeyframes forward: {}", show(&forward_steps));
    println!("keyframes back:    {}", show(&back_steps));

    // Both ends are walls: there is no keyframe before the first or after the
    // last, and staying put on reaching one is the right answer rather than a
    // fault. Only movement away from a wall has to go the right way.
    const START: Duration = Duration::from_millis(100);
    let end = player.info.duration.unwrap_or(Duration::MAX).saturating_sub(Duration::from_secs(2));

    anyhow::ensure!(forward_steps.len() >= 2, "stepping forward did not move");
    anyhow::ensure!(
        forward_steps.windows(2).all(|w| w[1] > w[0] || w[0] >= end),
        "stepping forward did not always advance: {}",
        show(&forward_steps)
    );
    anyhow::ensure!(
        back_steps.windows(2).all(|w| w[1] < w[0] || w[0] <= START),
        "stepping back did not always retreat: {}",
        show(&back_steps)
    );
    if back_steps.last().is_some_and(|d| *d <= START) {
        println!("(reached the start of the file)");
    }
    let gaps: Vec<f64> =
        forward_steps.windows(2).map(|w| (w[1] - w[0]).as_secs_f64()).collect();
    let mean = gaps.iter().sum::<f64>() / gaps.len().max(1) as f64;
    println!("keyframe spacing:  {mean:.2}s average");
    println!("keyframe stepping works");
    Ok(())
}

/// Run the film to its end, then seek back into it.
///
/// The end used to be the end of the pipeline as well: the threads reached the
/// last packet and exited, so a seek afterwards had nobody left to act on it
/// and the film could not be restarted without reopening the file. Reaching
/// the end is not the same as being finished with it.
fn check_seek_after_end(player: &mut video::Player) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    let Some(duration) = player.info.duration else {
        println!("\nno duration: skipping the seek-after-end check");
        return Ok(());
    };

    // Right up to the end, then drain until the decoder says it is done.
    player.seek(duration, false);
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && !player.finished() {
        while player.try_next().is_some() {}
        std::thread::sleep(Duration::from_millis(10));
    }
    anyhow::ensure!(player.finished(), "the film never reported reaching its end");
    println!("\nreached the end");

    // Now back to the front, as dragging the seek bar would.
    let target = duration / 4;
    player.seek(target, false);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut first = None;
    while Instant::now() < deadline {
        if let Some(frame) = player.try_next() {
            first = Some(frame.at);
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let Some(at) = first else {
        anyhow::bail!("nothing decoded after seeking back from the end — playback cannot restart");
    };
    println!("seeking back from the end works: resumed at {:.1}s", at.as_secs_f64());
    anyhow::ensure!(!player.finished(), "still reports finished after seeking back");
    Ok(())
}

/// Time the fill of the window behind the play head.
///
/// This is what a jump costs before stepping back is instant again, and it is
/// the number that decides whether the second decoder is worth having.
fn check_backfill(player: &mut video::Player, path: &std::path::Path) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    let span = framecache::DEFAULT_SPAN;
    let Some(at) = player.info.duration.map(|d| (d / 2).max(span)) else {
        println!("\nno duration: skipping the backfill check");
        return Ok(());
    };

    let started = Instant::now();
    let Some(mut job) = framecache::BackFill::start(path, at, span) else {
        anyhow::bail!("could not start the backward fill");
    };
    let mut frames = Vec::new();
    while started.elapsed() < Duration::from_secs(20) {
        if let Some(done) = job.collect() {
            frames = done;
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let took = started.elapsed();

    anyhow::ensure!(!frames.is_empty(), "the backward fill produced nothing");
    let covered = frames
        .last()
        .zip(frames.first())
        .map(|(l, f)| l.at.saturating_sub(f.at))
        .unwrap_or_default();
    println!(
        "\nbackward fill: {:.0} ms for {:.1}s of film ({} frames, {:.1}x real time)",
        took.as_secs_f64() * 1000.0,
        covered.as_secs_f64(),
        frames.len(),
        covered.as_secs_f64() / took.as_secs_f64().max(0.001)
    );
    Ok(())
}

/// Step back one frame with nothing cached, and time it.
///
/// This is the path the frame cache exists to avoid, so its cost is the
/// argument for the cache. It has to land on the frame *before* the one asked
/// for — not on the keyframe before it, which would be a seek wearing a step's
/// clothes.
fn check_uncached_step(player: &mut video::Player) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    // Somewhere with a full group of pictures ahead of it.
    let from = player.info.duration.map(|d| d / 3).unwrap_or(Duration::from_secs(10));
    player.seek(from, false);
    // Well inside the group of pictures, not on its first frame: a seek lands
    // on a keyframe, and there is nothing before a keyframe to step back to.
    // Measuring from there would test the easiest case and call it the worst.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut at = None;
    let mut seen = 0;
    while Instant::now() < deadline && seen < 12 {
        if let Some(frame) = player.try_next() {
            at = Some(frame.at);
            seen += 1;
        } else {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    let Some(target) = at else {
        anyhow::bail!("no frame to step back from");
    };

    let started = Instant::now();
    player.seek(target.saturating_sub(Duration::from_millis(1)), false);
    let mut best: Option<Duration> = None;
    let mut decoded = 0u32;
    while started.elapsed() < Duration::from_secs(5) {
        match player.try_next() {
            Some(frame) if frame.at >= target => break,
            Some(frame) => {
                best = Some(frame.at);
                decoded += 1;
            }
            None if player.finished() => break,
            None => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    let took = started.elapsed();

    let Some(landed) = best else {
        anyhow::bail!("stepping back with an empty cache found nothing");
    };
    let gap = target.as_secs_f64() - landed.as_secs_f64();
    let frame_time = 1.0 / player.info.frame_rate.max(1.0);
    println!(
        "\nuncached step back: {:.0} ms, {decoded} frames decoded, landed {:.0} ms earlier",
        took.as_secs_f64() * 1000.0,
        gap * 1000.0
    );
    // Within a frame and a half: it must be the previous frame, not the
    // previous keyframe.
    anyhow::ensure!(
        gap <= frame_time * 1.5,
        "landed {:.0} ms back, which is {:.1} frames — that is a seek, not a step",
        gap * 1000.0,
        gap / frame_time
    );
    println!("the uncached step moves exactly one frame");
    check_stepping_forward_forever(player)?;
    Ok(())
}

/// Step forward until the decoder stops supplying, then keep going.
///
/// A paused film stops draining its sound, the audio queue fills, and the
/// demuxer blocks putting a packet into it — so no more video is read either.
/// Stepping forward then works until whatever was decoded runs out and stops
/// dead, which is what this reproduces: drain the queue on purpose, then check
/// that the next step still moves.
fn check_stepping_forward_forever(player: &mut video::Player) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    let from = player.info.duration.map(|d| d / 4).unwrap_or(Duration::from_secs(5));
    player.seek(from, false);

    // Take frames as fast as they come until there are none: the decoder is
    // then in the state a paused film reaches by itself.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut at = Duration::ZERO;
    let mut drained = 0u32;
    let mut idle = 0u32;
    while Instant::now() < deadline {
        match player.try_next() {
            Some(frame) => {
                at = frame.at;
                drained += 1;
                idle = 0;
            }
            None if player.finished() => break,
            None => {
                idle += 1;
                // Twenty consecutive empty tries is a stall, not a pause for
                // breath.
                if idle > 20 {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
    println!("\ndrained {drained} frames, stalled at {:.2}s", at.as_secs_f64());
    if player.finished() {
        println!("the file ended before the queue did; nothing to prove here");
        return Ok(());
    }

    // Exactly what the viewer does when a step finds nothing waiting.
    let started = Instant::now();
    player.seek(at, false);
    let mut landed = None;
    while started.elapsed() < Duration::from_secs(5) {
        match player.try_next() {
            Some(frame) if frame.at > at => {
                landed = Some(frame.at);
                break;
            }
            Some(_) => {}
            None if player.finished() => break,
            None => std::thread::sleep(Duration::from_millis(2)),
        }
    }
    let Some(landed) = landed else {
        anyhow::bail!("stepping forward from a dry decoder found nothing");
    };
    let gap = landed.as_secs_f64() - at.as_secs_f64();
    let frame_time = 1.0 / player.info.frame_rate.max(1.0);
    println!(
        "stepped on to {:.2}s — {:.0} ms later, in {:.0} ms",
        landed.as_secs_f64(),
        gap * 1000.0,
        started.elapsed().as_secs_f64() * 1000.0
    );
    anyhow::ensure!(
        gap <= frame_time * 1.5,
        "landed {:.1} frames on, which is a jump rather than a step",
        gap / frame_time
    );
    println!("stepping forward keeps working after the decoder stops");
    Ok(())
}

/// Report what subtitle tracks a file has and how many cues arrived.
fn check_subtitles(player: &mut video::Player, path: &std::path::Path) -> anyhow::Result<()> {
    use std::time::Duration;

    let mut subs = subtitles::Subtitles::new();
    for track in player.subtitle_tracks.clone() {
        println!("subtitle track {}: {} [{}]", track.index, track.label(), track.codec);
        subs.add_track(track);
    }
    for (i, file) in subtitles::find_beside(path).into_iter().enumerate() {
        match subtitles::load_external(&file, (1 << 20) + i) {
            Ok((track, cues)) => {
                println!("subtitle file {}: {} cues", file.display(), cues.len());
                subs.add_track(track);
                for cue in cues {
                    subs.push(cue);
                }
            }
            Err(e) => println!("subtitle file {}: {e:#}", file.display()),
        }
    }
    for cue in player.try_cues() {
        subs.push(cue);
    }
    if subs.is_empty() {
        println!("no subtitles");
        return Ok(());
    }
    // Show what is due at a few points, which exercises selection and timing
    // rather than merely counting what was decoded.
    for at in [2u64, 6, 31, 121] {
        let at = Duration::from_secs(at);
        match subs.at(at).map(|c| &c.payload) {
            Some(subtitles::Payload::Lines(lines)) => {
                println!("  at {:>4}s  {}", at.as_secs(), lines.join(" / "))
            }
            Some(subtitles::Payload::Images(images, canvas)) => println!(
                "  at {:>4}s  {} picture(s) in a {}x{} canvas",
                at.as_secs(),
                images.len(),
                canvas.0,
                canvas.1
            ),
            None => println!("  at {:>4}s  -", at.as_secs()),
        }
    }
    Ok(())
}

/// Seek and confirm the frames that come back belong to the new position.
fn check_seek(player: &mut video::Player) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    let Some(duration) = player.info.duration else {
        println!("\nno duration: skipping the seek check");
        return Ok(());
    };
    let target = duration / 2;
    println!("\nseeking to {:.1}s of {:.1}s", target.as_secs_f64(), duration.as_secs_f64());

    let landed = player.seek(target, false);
    let start = Instant::now();
    let mut first = None;
    while start.elapsed() < Duration::from_secs(5) {
        if let Some(frame) = player.try_next() {
            first = Some(frame.at);
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let Some(at) = first else {
        anyhow::bail!("no frame arrived within five seconds of seeking");
    };
    let drift = (at.as_secs_f64() - landed.as_secs_f64()).abs();
    println!("  first frame at {:.2}s, {:.2}s from the target", at.as_secs_f64(), drift);
    // A seek lands on the keyframe at or before the target, so being early is
    // expected; being far past it is not.
    anyhow::ensure!(
        at <= landed + Duration::from_secs(1),
        "the seek overshot: landed at {at:?} for a target of {landed:?}"
    );
    println!("seeking works");
    Ok(())
}

/// Write the overlays to PNGs so their layout can be inspected directly.
///
/// The alternative is describing a picture to someone who cannot see it, which
/// is a poor way to find out that a panel is twice as wide as the screen.
/// Segment everything matching a phrase, headless.
///
/// The same three graphs the viewer will use, driven from the command line so
/// the pipeline can be checked against a known answer without a window — and
/// so a wrong answer is a wrong answer rather than a wrong answer somewhere
/// inside a frame loop.
#[cfg(feature = "ml")]
fn find_command(phrase: &str, path: &std::path::Path, args: &cli::Args) -> anyhow::Result<()> {
    use std::time::Instant;

    let opened = Instant::now();
    let mut model = ml::sam3::Sam3::open()?;
    println!("opened the model in {:.1}s", opened.elapsed().as_secs_f64());

    let loaded = loader::load(path, args.io, args.preview)?;
    let picture = &loaded.decoded;
    let frame = ml::prep::from_pixels(&picture.rgba, picture.width, picture.height)
        .ok_or_else(|| anyhow::anyhow!("the picture has no size"))?;

    let looked = Instant::now();
    let embedding = model.look(&frame)?;
    let look_ms = looked.elapsed().as_secs_f64() * 1000.0;

    let asked = Instant::now();
    let found = model.find(&embedding, phrase, ml::sam3::KEEP)?;
    let ask_ms = asked.elapsed().as_secs_f64() * 1000.0;

    // Again, to show what a second phrase costs: the picture is already
    // encoded and the answer to *this* phrase is already cached, so what is
    // left is the decoder.
    let again = Instant::now();
    let _ = model.find(&embedding, phrase, ml::sam3::KEEP)?;
    let again_ms = again.elapsed().as_secs_f64() * 1000.0;

    println!(
        "{}: {}x{} · encoded in {look_ms:.0} ms · first answer {ask_ms:.0} ms · same phrase again {again_ms:.0} ms",
        path.display(),
        picture.width,
        picture.height
    );
    println!("\n{:?}: {} found", phrase, found.len());
    for item in &found {
        let (x1, y1, x2, y2) = item.found.bounds;
        let inside = item.mask.iter().filter(|v| **v > 0.0).count();
        println!(
            "  {:>4.0}%  [{x1:>6.0} {y1:>6.0} {x2:>6.0} {y2:>6.0}]  outline {:.1}% of the grid",
            item.found.confidence * 100.0,
            inside as f32 / item.mask.len().max(1) as f32 * 100.0
        );
    }

    if let Some(dir) = args.dump_ui.as_deref() {
        let source = (picture.width, picture.height);
        let mut grid = ml::labels::Grid::for_picture(source);
        grid.clear();
        // Ids so each thing is drawn in its own colour, exactly as the viewer
        // would draw them.
        let mut numbered = found;
        for (i, item) in numbered.iter_mut().enumerate() {
            item.id = i as u32 + 1;
        }
        grid.add_cells(numbered.iter());
        dump_labels(dir, "found.png", &loaded.decoded, &grid)?;
    }
    Ok(())
}

#[cfg(not(feature = "ml"))]
fn find_command(_: &str, _: &std::path::Path, _: &cli::Args) -> anyhow::Result<()> {
    anyhow::bail!("this build has no inference; rebuild with the `ml` feature")
}

fn dump_ui(dir: &std::path::Path, args: &cli::Args) -> anyhow::Result<()> {
    use anyhow::Context;

    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

    // Run a real load so the breakdown has real stages in it.
    let title = match args.files.first() {
        Some(path) => {
            loader::load(path, args.io, args.preview)?;
            path.file_name().unwrap_or_default().to_string_lossy().into_owned()
        }
        None => "NO FILE".to_string(),
    };

    let screen = args.screen.or_else(profile::detect_screen).unwrap_or((1920, 1080));

    let write = |name: &str, ov: &osd::Overlay| -> anyhow::Result<()> {
        let path = dir.join(name);
        let file = std::fs::File::create(&path)?;
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), ov.width, ov.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(&ov.rgba)?;
        println!("{}  {}x{}", path.display(), ov.width, ov.height);
        Ok(())
    };

    let _ = &title;
    // A specimen, so a wrong pixel in a glyph is visible rather than found
    // later in the middle of a caption.
    write(
        "font.png",
        &osd::render_scaled(
            &[
                osd::Row::plain("ABCDEFGHIJKLM  NOPQRSTUVWXYZ"),
                osd::Row::plain("abcdefghijklm  nopqrstuvwxyz"),
                osd::Row::plain("0123456789  .,:;!?'\"-()[]"),
                osd::Row::plain(""),
                osd::Row::plain("The quick brown fox jumps"),
                osd::Row::plain("over the lazy dog. Cafe,"),
                osd::Row::plain("naive, jalapeno - 75% of it."),
            ],
            osd::SCALE_PANEL,
        ),
    )?;
    write(
        "caption.png",
        &{
            let px = (screen.1 as f32 * 0.045).clamp(14.0, 96.0);
            // Through the same stripping the player applies, so the specimen
            // shows what a caption really looks like — notes and all, which
            // this face has no glyph for and which must not become boxes.
            let lines: Vec<String> = [
                "\u{266a} I'm going to make him an offer he can't refuse \u{266a}",
                "(quietly) It's not personal, Sonny \u{2014} it's strictly business.",
                "Caf\u{e9}, na\u{ef}ve, jalape\u{f1}o; \u{201c}typographic quotes\u{201d}\u{2026}",
            ]
            .iter()
            .map(|l| typeface::strip_unsupported(l))
            .filter(|l| !l.is_empty())
            .collect();
            osd::caption(&osd::wrap(&lines, px, (screen.0 as f32 * 0.86) as u32), px)
        },
    )?;
    write(
        "scrubber.png",
        &osd::scrubber(
            &osd::timecode(std::time::Duration::from_secs(1462)),
            &osd::timecode(std::time::Duration::from_secs(6231)),
            1462.0 / 6231.0,
            (screen.0 as f32 * 0.6) as u32,
            osd::SCALE_COMPACT,
        ),
    )?;
    write(
        "scrubber_paused.png",
        &osd::scrubber(
            &format!("{}  PAUSED", osd::timecode(std::time::Duration::from_secs(1462))),
            &osd::timecode(std::time::Duration::from_secs(6231)),
            1462.0 / 6231.0,
            (screen.0 as f32 * 0.6) as u32,
            osd::SCALE_COMPACT,
        ),
    )?;
    write(
        "menu.png",
        &osd::render_menu(
            &[
                osd::Row::dim("SUBTITLES", ""),
                osd::Row::new("OFF", "", osd::TEXT),
                osd::Row::good("ENG ENGLISH", "SHOWING").selected(true),
                osd::Row::new("NLD DUTCH (PICTURE)", "LOADED", osd::TEXT),
                osd::Row::new("THE.FILM.2026.FORCED.SRT", "FILE", osd::TEXT),
                osd::Row::new("DOWNLOADED-ELSEWHERE.SRT", "FILE", osd::TEXT),
                osd::Row::plain(""),
                osd::Row::dim("UP/DOWN CHOOSE", "ENTER SELECT"),
                osd::Row::dim("ALT OR ESC", "CLOSE"),
            ],
            osd::SCALE_PANEL,
            (screen.0 as f32 * 0.8) as u32,
        ),
    )?;
    write("plugins.png", &plugins::specimen((screen.0 as f32, screen.1 as f32)))?;
    write("osd.png", &osd::render(&osd::rows()))?;
    // The card's memory, from the real driver. Skipped rather than faked where
    // there is no card to ask: a specimen full of invented numbers would be
    // checked against nothing.
    match vram::sample() {
        Some(usage) => {
            let figures = vram::figures(&usage, &vram::charges());
            write("vram.png", &osd::render_scaled(&vram::rows(&usage, &figures), osd::SCALE_COMPACT))?;
        }
        None => println!("no NVIDIA card to ask about memory; skipping vram.png"),
    }
    write(
        "filename.png",
        &osd::render_within(
            &[osd::Row::plain(format!("{title}   3/32"))],
            osd::SCALE_PANEL,
            (screen.0 as f32 * 0.9) as u32,
        ),
    )?;
    write(
        "status.png",
        &osd::render(&[
            osd::Row::good("ZOOM", "100%"),
            osd::Row::dim("AUTOSCALE", "ON"),
            osd::Row::dim(
                "MEM",
                format!(
                    "{:.2} / {:.1} GB",
                    config::process_memory_bytes().unwrap_or(0) as f64 / 1073741824.0,
                    config::Config::load().memory_budget_bytes() as f64 / 1073741824.0
                ),
            ),
        ]),
    )?;
    let win = (screen.0 as f32, screen.1 as f32);
    write("panel.png", panel::Panel::open(&config::Config::load()).rendered(win))?;
    write(
        "notice.png",
        &osd::message(&["NO IMAGES FOUND", "LOOKED IN /HOME/USER/PICTURES", "", "ESC OR Q TO QUIT"]),
    )?;
    // The grid in compare mode, with real thumbnails, so its layout can be
    // checked the same way as everything else.
    {
        let dir = args.files.first().and_then(|f| f.parent().map(|p| p.to_path_buf()));
        if let Some(dir) = dir {
            let list = playlist::Playlist::from_dir(&dir);
            let count = list.len().min(24);
            let mut grid = mosaic::Mosaic::new(0);
            let mut hashes = Vec::new();
            for i in 0..count {
                let Some(path) = list.at(i) else { continue };
                let target = decode::Target { width: 200, height: 200 };
                match loader::load(path, args.io, Some(target)) {
                    Ok(l) => {
                        let d = l.decoded;
                        hashes.push(Some(similarity::phash(&d.rgba, d.width, d.height)));
                        grid.insert(
                            i,
                            mosaic::Thumb { rgba: d.rgba, width: d.width, height: d.height },
                        );
                    }
                    Err(_) => hashes.push(None),
                }
            }
            // Ordered exactly as F6 does, so the dump shows the real grouping.
            grid.set_order(similarity::order_by_similarity(&hashes));
            grid.set_hashes(hashes);
            grid.set_selected(0, win, count);
            write("mosaic.png", &grid.compose(win, count))?;
        }
    }

    // The browser panel, with the long names it has to cope with.
    {
        let rows = vec![
            osd::Row::dim("SELECTED", ""),
            osd::Row::good("a rather long photograph filename from a camera 2026.jpg", ""),
            osd::Row::new("SIZE", "1920 x 1080", osd::TEXT_COLOUR),
            osd::Row::new("FILE", "179.2 KB", osd::TEXT_COLOUR),
            osd::Row::dim("ENTER", "SHOW THIS IMAGE"),
            osd::Row::plain(""),
            osd::Row::dim("FOLDER", "corpus"),
            osd::Row::good("..", "").selected(false),
            osd::Row::good("sub-folder-with-a-long-name/", ""),
            osd::Row::plain("another very long photograph filename indeed 2026.jpg").selected(true),
            osd::Row::plain("short.png"),
        ];
        write("browser.png", &osd::render_fixed(&rows, osd::SCALE_COMPACT, (460, 420)))?;
    }

    {
        // A rate that starts healthy and falls behind, so both colours show.
        let target = 30.0f32;
        let samples: Vec<f32> = (0..20)
            .map(|i| if i < 12 { 30.0 } else { 30.0 - (i - 11) as f32 * 1.8 })
            .collect();
        // The first quarter is a gap, as it is for the first second of any
        // film: no rate is known before the first frame.
        let samples: Vec<Option<f32>> = samples
            .iter()
            .enumerate()
            .map(|(i, v)| (i >= samples.len() / 4).then_some(*v))
            .collect();
        write(
            "fps.png",
            &osd::chart("HEVC HW", "19.4 / 29.97 FPS", &samples, target, osd::SCALE_PANEL / 2.0),
        )?;
    }

    write(
        "confirm.png",
        &osd::message(&[
            "MOVE TO TRASH",
            &title,
            "",
            "Y  CONFIRM        N OR ESC  CANCEL",
        ]),
    )?;
    Ok(())
}

/// Hash a folder and print how its images group, without opening a window.
///
/// The same code path F6 runs, so a grouping that looks wrong on screen can be
/// examined here instead of by squinting at a grid.
fn compare_folder(dir: &std::path::Path, args: &cli::Args) -> anyhow::Result<()> {
    let playlist = playlist::Playlist::from_dir(dir);
    let paths: Vec<std::path::PathBuf> = (0..playlist.len())
        .filter_map(|i| playlist.at(i).map(|p| p.to_path_buf()))
        .collect();
    if paths.is_empty() {
        anyhow::bail!("no images in {}", dir.display());
    }

    let start = std::time::Instant::now();
    let (hashes, cache) = similarity::analyse(&paths, args.io, |_, _| {});
    let elapsed = start.elapsed();
    cache.save()?;

    let order = similarity::order_by_similarity(&hashes);
    let name = |i: usize| {
        paths[i].file_name().unwrap_or_default().to_string_lossy().into_owned()
    };

    println!(
        "{} images in {:.0} ms ({} cached)\n",
        paths.len(),
        elapsed.as_secs_f64() * 1e3,
        cache.len()
    );

    // Walk the ordering and break it wherever consecutive images stop being
    // alike, which is exactly where the grid's grouping visually ends.
    let mut group = 1;
    let mut printed_header = false;
    for (place, &i) in order.iter().enumerate() {
        let prev = place.checked_sub(1).map(|p| order[p]);
        let alike = match (prev.and_then(|p| hashes[p]), hashes[i]) {
            (Some(a), Some(b)) => similarity::distance(a, b) <= similarity::SAME_IMAGE,
            _ => false,
        };
        if !alike {
            if printed_header {
                println!();
            }
            println!("group {group}:");
            group += 1;
            printed_header = true;
        }
        match (prev.and_then(|p| hashes[p]), hashes[i]) {
            (Some(a), Some(b)) if alike => {
                println!("  {:<28} {:.0}% like the one above", name(i), similarity::similarity(a, b) * 100.0)
            }
            (_, Some(_)) => println!("  {:<28}", name(i)),
            (_, None) => println!("  {:<28} (could not be hashed)", name(i)),
        }
    }
    Ok(())
}

/// Describe a tiled image's pyramid and time reading a tile from each level.
fn report_tiles(path: &std::path::Path) -> anyhow::Result<()> {
    let start = std::time::Instant::now();
    let Some(mut pyramid) = tiled::probe(path)? else {
        anyhow::bail!("{} is not a tiled image", path.display());
    };
    println!("probed in {:.1} ms", start.elapsed().as_secs_f64() * 1e3);

    // No pyramid: open it the way the viewer does, which stores levels for a
    // file big enough to need them, and time that first open.
    if !pyramid.is_pyramidal() {
        let t = std::time::Instant::now();
        if tiled::overview(path, None)?.is_some() {
            println!("first open, storing levels: {:.1} s", t.elapsed().as_secs_f64());
            pyramid = tiled::probe(path)?.unwrap_or(pyramid);
        }
    }

    let (fw, fh) = pyramid.full_size();
    let stored = pyramid.levels.iter().filter(|l| l.cached.is_some()).count();
    println!(
        "{}\n  {fw}x{fh}, {} level(s){}\n",
        path.display(),
        pyramid.levels.len(),
        match (pyramid.is_pyramidal(), stored) {
            (false, _) => " (no pyramid)".to_string(),
            (true, 0) => String::new(),
            (true, n) => format!(" ({n} stored)"),
        }
    );
    println!("  {:>5} {:>16} {:>12} {:>10} {:>9} {:>10}", "level", "size", "tiles", "scale", "read ms", "tile");

    let mut reader = tiled::TileReader::open(&pyramid)?;
    for (i, level) in pyramid.levels.iter().enumerate() {
        let (across, down) = level.grid();
        // A tile from the middle, where a partial edge tile cannot flatter the
        // timing by being mostly empty.
        let (tx, ty) = (across / 2, down / 2);
        let t = std::time::Instant::now();
        let result = reader.read(level, tx, ty);
        let ms = t.elapsed().as_secs_f64() * 1e3;

        match result {
            Ok(tile) => println!(
                "  {i:>5} {:>16} {:>12} {:>10.4} {ms:>9.1} {:>10}",
                format!("{}x{}", level.width, level.height),
                format!("{across}x{down}"),
                level.scale(fw),
                format!("{}x{}", tile.width, tile.height),
            ),
            Err(e) => println!(
                "  {i:>5} {:>16} {:>12} {:>10.4} {:>9}  FAILED: {e:#}",
                format!("{}x{}", level.width, level.height),
                format!("{across}x{down}"),
                level.scale(fw),
                "-"
            ),
        }
    }

    // The tile renderer's view of it: which level and tiles a screen needs,
    // fitted and at 1:1 in the middle, and how long the loader takes to read
    // them. What opening and zooming will cost, before anything is drawn.
    let screen = (3072.0f32, 1728.0f32);
    let fit = (screen.0 / fw as f32).min(screen.1 / fh as f32);
    let pyramid = std::sync::Arc::new(pyramid);
    println!();
    for (what, zoom) in [("fit", fit), ("1/4", 0.26), ("1:1", 1.0)] {
        let plan = tiles::plan(&pyramid, zoom, [fw as f32 / 2.0, fh as f32 / 2.0], screen);
        let (tx, rx) = std::sync::mpsc::channel();
        let loader = tiles::TileLoader::new(pyramid.clone(), 8, move |key, tile| {
            let _ = tx.send((key, tile.is_ok()));
        });
        let t = std::time::Instant::now();
        loader.request(plan.visible.iter().copied(), |_| false);
        let ok = (0..plan.visible.len())
            .filter_map(|_| rx.recv_timeout(std::time::Duration::from_secs(60)).ok())
            .filter(|(_, ok)| *ok)
            .count();
        println!(
            "  {what:>4}: level {} ({}x{}), {} tiles on screen + {} ahead, {ok} read in {:.1} ms",
            plan.level,
            pyramid.levels[plan.level].width,
            pyramid.levels[plan.level].height,
            plan.visible.len(),
            plan.ahead.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let pyramid = std::sync::Arc::try_unwrap(pyramid).unwrap_or_else(|p| (*p).clone());

    // What a screenful actually costs at the coarsest useful level.
    if let Some(level) = pyramid.levels.last() {
        let (across, down) = level.grid();
        let t = std::time::Instant::now();
        let mut ok = 0;
        for ty in 0..down.min(8) {
            for tx in 0..across.min(8) {
                if reader.read(level, tx, ty).is_ok() {
                    ok += 1;
                }
            }
        }
        println!(
            "\n  {ok} tiles from the smallest level in {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(())
}

/// Report where a folder's thumbnails would come from, and how fast.
fn report_thumbs(dir: &std::path::Path) -> anyhow::Result<()> {
    let list = playlist::Playlist::from_dir(dir);
    let paths: Vec<std::path::PathBuf> =
        (0..list.len()).filter_map(|i| list.at(i).map(|p| p.to_path_buf())).collect();
    if paths.is_empty() {
        anyhow::bail!("no images in {}", dir.display());
    }

    let cache = thumbs::Cache::open().ok_or_else(|| anyhow::anyhow!("no cache directory"))?;
    let (mut ours, mut desktop, mut neither) = (0, 0, 0);
    let (mut ours_ms, mut desktop_ms, mut decode_ms) = (0.0f64, 0.0f64, 0.0f64);

    for path in &paths {
        let t = std::time::Instant::now();
        if cache.load(path).is_some() {
            ours_ms += t.elapsed().as_secs_f64() * 1e3;
            ours += 1;
            continue;
        }
        let t = std::time::Instant::now();
        if thumbs::xdg::load(path, 128).is_some() {
            desktop_ms += t.elapsed().as_secs_f64() * 1e3;
            desktop += 1;
            continue;
        }
        neither += 1;
        // What it costs to fill this cell without any thumbnail at all.
        if neither <= 12 {
            let t = std::time::Instant::now();
            let target = decode::Target { width: 200, height: 200 };
            if loader::load(path, loader::Io::Mmap, Some(target)).is_ok() {
                decode_ms += t.elapsed().as_secs_f64() * 1e3;
            }
        }
    }

    let mean = |total: f64, n: usize| if n == 0 { 0.0 } else { total / n as f64 };
    println!("{} images in {}\n", paths.len(), dir.display());
    println!("  {:<26} {:>6}  {:>10}", "source", "count", "mean ms");
    println!("  {:<26} {:>6}  {:>10.2}", "show's own cache", ours, mean(ours_ms, ours));
    println!("  {:<26} {:>6}  {:>10.2}", "desktop (nemo) cache", desktop, mean(desktop_ms, desktop));
    println!(
        "  {:<26} {:>6}  {:>10.2}  (full decode, first {} sampled)",
        "no thumbnail yet",
        neither,
        mean(decode_ms, neither.min(12)),
        neither.min(12)
    );

    if let Some(sample) = paths.first() {
        println!("\n  uri for {}\n    {}", sample.display(), thumbs::xdg::file_uri(sample));
    }
    Ok(())
}

fn main() -> ExitCode {
    // Before anything else, and before any thread: see the doc comment there.
    app::capture_activation_token();
    // A crash says what happened, in a window, rather than the program just
    // disappearing.
    crash::install();

    let args = match cli::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("show: {e}\n");
            eprint!("{}", cli::USAGE);
            return ExitCode::from(2);
        }
    };

    // A profile that records nothing is not a profile.
    timing::init(args.timing || args.profile.is_some());
    if let Some(d) = timing::pre_main() {
        timing::record("main", "pre_main", std::time::Duration::ZERO, d, Some("exec -> main".into()));
    }

    if let Some(dir) = args.thumbs.clone() {
        return match report_thumbs(&dir) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(file) = args.detect.clone() {
        return match detect_in(&file, &args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some((phrase, file)) = args.find.clone() {
        return match find_command(&phrase, &file, &args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(words) = args.plugins.clone() {
        return match plugins_command(&words) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(file) = args.ml_probe.clone() {
        return match probe_ml(&file) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(file) = args.cache_bench.clone() {
        return match bench_cache(&file) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(file) = args.probe.clone() {
        return match probe_playback(&file) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(file) = args.tiles.clone() {
        return match report_tiles(&file) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(dir) = args.similar.clone() {
        return match compare_folder(&dir, &args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(dir) = args.dump_ui.clone() {
        return match dump_ui(&dir, &args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(file) = args.copy.clone() {
        return match screenshot::copy_file(&file) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if args.screenshot {
        return match screenshot::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if args.install {
        return match install::install() {
            Ok(p) => {
                println!("installed {}", p.bin.display());
                println!("installed {}", p.desktop.display());
                if !install::is_on_path(&p.bin) {
                    println!(
                        "note: {} is not on your PATH",
                        p.bin.parent().unwrap_or(&p.bin).display()
                    );
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if args.set_default {
        return match install::set_as_default() {
            Ok(n) => {
                println!("show is now the default application for {n} image types");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    if args.list_formats {
        for f in decode::ALL_FORMATS {
            match f.support() {
                decode::Support::Yes(via) => println!("  {:<9} yes  ({via})", f.name()),
                decode::Support::Runtime(via) => println!("  {:<9} yes  ({via})", f.name()),
                decode::Support::Missing(why) => println!("  {:<9} NO   {why}", f.name()),
                decode::Support::MissingRuntime(why) => println!("  {:<9} NO   {why}", f.name()),
            }
        }
        return ExitCode::SUCCESS;
    }

    if args.show_version {
        println!("{}", version());
        return ExitCode::SUCCESS;
    }

    if args.list_formats_video {
        let caps = video::capabilities();
        let (yes, no): (Vec<_>, Vec<_>) = caps.iter().partition(|(_, ok)| *ok);
        println!("  containers  {}", video::containers().join(" "));
        println!();
        println!("  decoders    {}", yes.iter().map(|(n, _)| *n).collect::<Vec<_>>().join("  "));
        if !no.is_empty() {
            println!("  missing     {}", no.iter().map(|(n, _)| *n).collect::<Vec<_>>().join("  "));
        }
        return ExitCode::SUCCESS;
    }

    if args.list_adapters {
        for (i, a) in gpu::describe_adapters().iter().enumerate() {
            println!("  {i}: {a}");
        }
        return ExitCode::SUCCESS;
    }

    if args.compare {
        return match compare::run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("show: {e:#}");
                ExitCode::FAILURE
            }
        };
    }

    let result = match (args.bench, args.profile) {
        (Some(iterations), _) => bench::run(&args.files, iterations, args.io, args.preview),
        (_, Some(iterations)) => {
            let screen = args.screen.or_else(profile::detect_screen).unwrap_or((1920, 1080));
            profile::run(
                &args.files,
                &profile::Options {
                    iterations,
                    io: args.io,
                    adapter: args.adapter.clone(),
                    screen,
                    backends: args.backends,
                    preview: args.preview,
                    emit_summary: args.emit_summary,
                },
            )
        }
        _ => {
            // Mentioned once, acted on only if asked. Never for the headless
            // modes, and never when there is no terminal to answer on.
            install::note_if_missing();
            let result = app::run(args);
            leave_now(result)
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("show: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// End the viewer without running library destructors.
///
/// Decodes run on detached threads, and quitting while one is inside libheif
/// — which `--once` does every time on a HEIC, and a user can do by closing
/// the window straight after opening one — had the C++ runtime tear the
/// library down under it: a segfault on the way out. Everything of ours is
/// already saved and dropped by the time `app::run` returns, so there is
/// nothing left that needs the orderly version.
fn leave_now(result: anyhow::Result<()>) -> ! {
    use std::io::Write;
    let code = match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("show: {e:#}");
            1
        }
    };
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    unsafe extern "C" {
        fn _exit(code: i32) -> !;
    }
    // Safety: ends the process at once, which is the point.
    unsafe { _exit(code) }
}

/// Seek while playing, the way a click on the timeline does, and measure how
/// far the picture is from the sound for the next few seconds.
///
/// Mirrors the player's loop: frames short of the target are decoded past,
/// the rest are shown when the audio clock reaches them, or against the wall
/// clock while there is no audio timeline.
fn check_seek_sync(player: &mut video::Player) -> anyhow::Result<()> {
    use std::time::{Duration, Instant};

    let Some(duration) = player.info.duration else { return Ok(()) };
    for (i, target) in [duration / 3, duration / 5, duration * 2 / 3].into_iter().enumerate() {
        // Let it play a moment first, so the device is full of the old place.
        let settle = Instant::now();
        player.set_playing(true);
        while settle.elapsed() < Duration::from_millis(800) {
            while player.try_next().is_some() {}
            std::thread::sleep(Duration::from_millis(10));
        }
        let landed = player.seek_exact(target, false);
        player.set_playing(false);
        let start = Instant::now();
        let mut origin: Option<Instant> = None;
        let mut next: Option<video::Frame> = None;
        let mut worst = 0.0f64;
        let mut last = 0.0f64;
        let mut shown = 0;
        let mut first: Option<(f64, f64)> = None;
        let mut released = false;
        let (mut arrived, mut reached, mut times) = (None, None, Vec::new());
        while start.elapsed() < Duration::from_secs(3) {
            if !released && start.elapsed() > Duration::from_millis(90) {
                // The release lands where the press did, which the player
                // recognises and does not seek again.
                released = true;
            }
            if next.is_none() {
                next = player.try_next();
            }
            let Some(frame) = next.as_ref() else {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            };
            if arrived.is_none() {
                arrived = Some(start.elapsed().as_secs_f64());
            }
            if frame.at + Duration::from_millis(20) < landed {
                next = None;
                continue;
            }
            if reached.is_none() {
                reached = Some(start.elapsed().as_secs_f64());
            }
            let audio = player.audio_position();
            let due = match audio {
                Some(now) => frame.at <= now,
                None => {
                    let o = *origin.get_or_insert_with(|| Instant::now() - frame.at);
                    o + frame.at <= Instant::now()
                }
            };
            if !due {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let frame = next.take().unwrap();
            player.set_playing(true);
            if shown < 4 {
                times.push(start.elapsed().as_secs_f64());
            }
            shown += 1;
            if let Some(now) = player.audio().and_then(|a| a.clock.position()) {
                let off = (now.as_secs_f64() - frame.at.as_secs_f64()) * 1000.0;
                if first.is_none() {
                    first = Some((start.elapsed().as_secs_f64(), off));
                }
                if start.elapsed() > Duration::from_millis(500) {
                    worst = worst.max(off.abs());
                    last = off;
                }
            }
        }
        println!(
            "seek {i} to {:.1}s: {shown} frames, first after {:.2}s; then worst {:.0} ms, last {:.0} ms apart",
            landed.as_secs_f64(), first.map_or(0.0, |f| f.0), worst, last
        );
        println!("    first new frame {arrived:?}, at target {reached:?}, first shown {times:.3?}");
        anyhow::ensure!(shown > 0, "nothing was shown in the three seconds after seeking");
        anyhow::ensure!(worst < 50.0, "picture and sound drifted {worst:.0} ms apart after seeking");
    }
    println!("seeking keeps sound and picture together");
    Ok(())
}
