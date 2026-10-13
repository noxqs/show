//! Argument parsing.
//!
//! Hand-rolled rather than pulled from a crate: this binary is meant to be the
//! system default image opener, so it is started constantly and every
//! millisecond of startup is worth keeping.

use std::path::PathBuf;

use crate::loader::Io;

pub const USAGE: &str = "\
show — a fast fullscreen image viewer

USAGE:
    show [OPTIONS] [FILE]...

With no FILE, opens your Pictures folder — the same window you get by opening
an image from the file manager.

VIEW OPTIONS:
    -w, --windowed          Start windowed instead of fullscreen
        --fullscreen        Force fullscreen, including for video
        --detect FILE       List what a model finds in FILE, with outlines if a
                            segmentation model is installed
        --plugins [VERB ID] List, install, remove or verify plugins
        --find TEXT FILE    Segment everything matching TEXT in FILE, and say
                            where. Needs the sam3 plugin
        --ml-probe FILE     Report what preparing frames for a model costs
        --probe FILE        Play FILE with no window and report audio, frame
                            pacing, subtitles and seeking
        --upscale           Enlarge images smaller than the window to fit
        --vsync             Force vsync (Fifo) instead of the lowest-latency
                            present mode the surface offers

PERFORMANCE:
    -t, --timing            Print a stage-by-stage timing waterfall
        --bench [N]         Headless decode benchmark, N iterations (default 5)
        --profile [N]       Headless end-to-end profile (io, decode, GPU upload,
                            draw) with no window shown; N runs (default 3)
        --screen <WxH>      Offscreen target size for --profile (default: the
                            primary monitor)
        --io <read|mmap>    How to get bytes off disk (default: mmap)
        --preview <WxH|off> Resolution to decode for the first frame. Formats
                            that decode at reduced scale (JPEG) do so directly.
                            Full resolution loads behind it. Default 3840x2160
        --adapter <NAME|N>  Pick a GPU by name substring or index
        --backends <LIST>   Graphics backends to initialise: vulkan, gl, all
                            (default: vulkan, falling back to all)
        --screenshot        Select part of the screen to copy or save
        --copy FILE         Put FILE on the clipboard, and stay to serve it
        --install           Install to ~/.local/bin and add an applications
                            menu entry, then exit
        --set-default       Make show the default application for images
        --dump-ui <DIR>     Render the overlays to PNGs in DIR and exit. Lets
                            the on-screen layout be checked without a display.
                            With --detect, also writes the outlines over the
                            picture they were found in
        --similar <DIR>     Hash every image in DIR, print how they group, exit
        --tiles <FILE>      Report a tiled image's pyramid and time a tile read
        --thumbs <DIR>      Report thumbnail coverage for DIR and time a load
        --formats           List the image formats this build can decode
        --video-formats     List the video containers and codecs it can play
        --list-adapters     Show every GPU wgpu can see, then exit
        --compare           Profile every adapter -- discrete, integrated and
                            software -- in a fresh process each, and rank them
        --no-software       Wait for the GPU instead of drawing the first frame
                            on the CPU. For measuring what that costs
        --once              Quit as soon as the first frame is on screen. Makes
                            the real windowed path measurable, surface setup and
                            present included, which --profile cannot see

    -V, --version           Print the version and build number
    -h, --help              Show this help

KEYS:
    esc, q          quit                 wheel / +, -    zoom
    f               toggle fullscreen    drag / wasd     pan
    0               fit to window        1               100%
    i               toggle smoothing     t               print timings
    space           pause video          m               mute
    r               repeat this film     /               find what you type
    h               every key there is
";

pub struct Args {
    pub files: Vec<PathBuf>,
    pub timing: bool,
    pub bench: Option<usize>,
    pub profile: Option<usize>,
    pub screen: Option<(u32, u32)>,
    pub preview: Option<crate::decode::Target>,
    pub io: Io,
    pub windowed: bool,
    pub upscale: bool,
    pub fullscreen: bool,
    pub vsync: bool,
    pub adapter: Option<String>,
    pub backends: wgpu::Backends,
    pub list_adapters: bool,
    pub list_formats: bool,
    pub list_formats_video: bool,
    pub show_version: bool,
    pub install: bool,
    pub screenshot: bool,
    pub copy: Option<PathBuf>,
    pub dump_ui: Option<std::path::PathBuf>,
    pub similar: Option<PathBuf>,
    pub tiles: Option<PathBuf>,
    pub thumbs: Option<PathBuf>,
    pub probe: Option<PathBuf>,
    pub cache_bench: Option<PathBuf>,
    pub ml_probe: Option<PathBuf>,
    pub detect: Option<PathBuf>,
    pub plugins: Option<Vec<String>>,
    /// A phrase and a picture, for the model that segments what it is told.
    pub find: Option<(String, std::path::PathBuf)>,
    pub set_default: bool,
    pub compare: bool,
    pub once: bool,
    pub software_first_frame: bool,
    /// Internal: emit a machine-readable summary for `--compare` to collect.
    pub emit_summary: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            files: Vec::new(),
            timing: false,
            bench: None,
            profile: None,
            screen: None,
            preview: Some(crate::decode::Target { width: 3840, height: 2160 }),
            io: Io::Mmap,
            windowed: false,
            upscale: false,
            fullscreen: false,
            vsync: false,
            adapter: None,
            // Bringing up OpenGL alongside Vulkan costs tens of milliseconds on
            // a cold start and buys nothing, so ask for one backend by default.
            backends: wgpu::Backends::VULKAN,
            list_adapters: false,
            list_formats: false,
            list_formats_video: false,
            show_version: false,
            install: false,
            screenshot: false,
            copy: None,
            dump_ui: None,
            similar: None,
            tiles: None,
            thumbs: None,
            probe: None,
            cache_bench: None,
            ml_probe: None,
            detect: None,
            plugins: None,
            find: None,
            set_default: false,
            compare: false,
            once: false,
            software_first_frame: true,
            emit_summary: false,
        }
    }
}

pub fn parse() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args_os().skip(1).peekable();

    while let Some(arg) = it.next() {
        let s = arg.to_string_lossy().into_owned();
        match s.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "-t" | "--timing" => args.timing = true,
            "-w" | "--windowed" => args.windowed = true,
            "--upscale" => args.upscale = true,
            "--fullscreen" => args.fullscreen = true,
            "--vsync" => args.vsync = true,
            "--list-adapters" => args.list_adapters = true,
            "-V" | "--version" => args.show_version = true,
            "--formats" => args.list_formats = true,
            "--video-formats" => args.list_formats_video = true,
            "--install" => args.install = true,
            "--screenshot" => args.screenshot = true,
            "--copy" => {
                let v = it.next().ok_or("--copy needs a file")?;
                args.copy = Some(PathBuf::from(v));
            }
            "--thumbs" => {
                let v = it.next().ok_or("--thumbs needs a directory")?;
                args.thumbs = Some(PathBuf::from(v));
            }
            "--detect" => {
                let v = it.next().ok_or("--detect needs a file")?;
                args.detect = Some(PathBuf::from(v));
            }
            "--find" => {
                let phrase = it.next().ok_or("--find needs a phrase and a picture")?;
                let file = it.next().ok_or("--find needs a picture after the phrase")?;
                args.find = Some((
                    phrase.to_string_lossy().into_owned(),
                    PathBuf::from(file),
                ));
            }
            "--plugins" => {
                // Everything after the flag belongs to it: a verb and maybe a
                // name. Taking the rest avoids inventing a second flag for the
                // argument and reads the way the command is spoken.
                args.plugins =
                    Some(it.by_ref().map(|a| a.to_string_lossy().into_owned()).collect());
            }
            "--ml-probe" => {
                let v = it.next().ok_or("--ml-probe needs a video file")?;
                args.ml_probe = Some(PathBuf::from(v));
            }
            "--cache-bench" => {
                let v = it.next().ok_or("--cache-bench needs a video file")?;
                args.cache_bench = Some(PathBuf::from(v));
            }
            "--probe" => {
                let v = it.next().ok_or("--probe needs a video file")?;
                args.probe = Some(PathBuf::from(v));
            }
            "--tiles" => {
                let v = it.next().ok_or("--tiles needs a file")?;
                args.tiles = Some(PathBuf::from(v));
            }
            "--similar" => {
                let v = it.next().ok_or("--similar needs a directory")?;
                args.similar = Some(PathBuf::from(v));
            }
            "--dump-ui" => {
                let v = it.next().ok_or("--dump-ui needs a directory")?;
                args.dump_ui = Some(PathBuf::from(v));
            }
            "--set-default" => args.set_default = true,
            "--compare" => args.compare = true,
            "--once" => args.once = true,
            "--no-software" => args.software_first_frame = false,
            "--emit-summary" => args.emit_summary = true,
            "--bench" => {
                // The count is optional, so only consume the next token if it
                // actually looks like a number.
                match it.peek().and_then(|v| v.to_string_lossy().parse::<usize>().ok()) {
                    Some(n) => {
                        it.next();
                        args.bench = Some(n.max(1));
                    }
                    None => args.bench = Some(5),
                }
            }
            "--profile" => {
                match it.peek().and_then(|v| v.to_string_lossy().parse::<usize>().ok()) {
                    Some(n) => {
                        it.next();
                        args.profile = Some(n.max(1));
                    }
                    None => args.profile = Some(3),
                }
            }
            "--preview" => {
                let v = it.next().ok_or("--preview needs WxH or 'off'")?;
                let v = v.to_string_lossy().into_owned();
                args.preview = if v.eq_ignore_ascii_case("off") {
                    None
                } else {
                    let (w, h) = v.split_once(['x', 'X']).ok_or("--preview wants WxH or 'off'")?;
                    let parse = |s: &str| s.trim().parse::<u32>().map_err(|_| format!("bad --preview value {v:?}"));
                    Some(crate::decode::Target { width: parse(w)?, height: parse(h)? })
                };
            }
            "--screen" => {
                let v = it.next().ok_or("--screen needs a value like 1920x1080")?;
                let v = v.to_string_lossy().into_owned();
                let (w, h) = v.split_once(['x', 'X']).ok_or("--screen wants WxH")?;
                let parse = |s: &str| s.trim().parse::<u32>().map_err(|_| format!("bad --screen value {v:?}"));
                args.screen = Some((parse(w)?, parse(h)?));
            }
            "--io" => {
                let v = it.next().ok_or("--io needs a value")?;
                args.io = match v.to_string_lossy().as_ref() {
                    "read" => Io::Read,
                    "mmap" => Io::Mmap,
                    other => return Err(format!("unknown --io value {other:?}")),
                };
            }
            "--backends" => {
                let v = it.next().ok_or("--backends needs a value")?;
                args.backends = match v.to_string_lossy().to_lowercase().as_str() {
                    "vulkan" | "vk" => wgpu::Backends::VULKAN,
                    "gl" | "opengl" => wgpu::Backends::GL,
                    "all" => wgpu::Backends::all(),
                    other => return Err(format!("unknown --backends value {other:?}")),
                };
            }
            "--adapter" => {
                let v = it.next().ok_or("--adapter needs a value")?;
                args.adapter = Some(v.to_string_lossy().into_owned());
            }
            _ if s.starts_with('-') && s.len() > 1 => return Err(format!("unknown option {s:?}")),
            _ => args.files.push(PathBuf::from(arg)),
        }
    }

    // A missing file is only an error for the headless modes, which have
    // nothing to work on without one. The viewer falls back to a folder.
    if args.files.is_empty() && (args.bench.is_some() || args.profile.is_some()) {
        return Err("--bench and --profile need at least one file".into());
    }
    Ok(args)
}
