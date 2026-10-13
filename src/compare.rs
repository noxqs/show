//! Rank every available graphics device on the same work.
//!
//! Each adapter is measured in a *fresh process*. That matters: driver
//! libraries, shader caches and kernel-side GPU contexts all warm up on first
//! use, so measuring four adapters in one process would flatter every one after
//! the first — which is precisely the number we care about, since a viewer
//! launched from a file manager is always paying cold-start cost.
//!
//! The child does the real work via `--profile`; this side only spawns, parses
//! one summary line per file, and tabulates.

use std::collections::BTreeMap;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::cli::Args;
use crate::gpu;

struct Row {
    label: String,
    init: Duration,
    upload: Duration,
    draw: Duration,
    /// Populated from whichever run reported it; identical across adapters.
    decode: Duration,
    cpu: Duration,
}

impl Row {
    fn gpu_total(&self) -> Duration {
        self.init + self.upload + self.draw
    }
}

pub fn run(args: &Args) -> Result<()> {
    let adapters = gpu::list_adapters();
    if adapters.is_empty() {
        anyhow::bail!("no graphics adapters found");
    }

    let exe = std::env::current_exe().context("finding our own executable")?;
    eprintln!("comparing {} adapter(s), one process each:", adapters.len());
    for a in &adapters {
        eprintln!("  {}", a.label());
    }

    // file -> adapter label -> row
    let mut results: BTreeMap<String, Vec<Row>> = BTreeMap::new();

    for adapter in &adapters {
        // The GL backend cannot create a device without a window, and the
        // profiler is deliberately headless. Rather than report that as a
        // failure every run, say plainly that it is out of scope here.
        if adapter.backend == "gl" {
            eprintln!("  skipping {} (GL needs a window; use --once to time it)", adapter.label());
            continue;
        }
        eprint!("  running {} ... ", adapter.label());
        match measure(&exe, args, adapter) {
            Ok(rows) => {
                eprintln!("ok");
                for (file, row) in rows {
                    results.entry(file).or_default().push(row);
                }
            }
            Err(e) => eprintln!("failed: {e:#}"),
        }
    }

    if results.is_empty() {
        anyhow::bail!("no adapter produced a usable measurement");
    }
    report(&results);
    Ok(())
}

fn measure(
    exe: &std::path::Path,
    args: &Args,
    adapter: &gpu::AdapterEntry,
) -> Result<Vec<(String, Row)>> {
    let mut cmd = Command::new(exe);
    cmd.arg("--profile")
        .arg("1")
        .arg("--emit-summary")
        .arg("--backends")
        .arg(adapter.backend)
        // Name rather than index: indices are relative to the backends the
        // child initialises, and we are pinning the backend too.
        .arg("--adapter")
        .arg(&adapter.name)
        .arg("--io")
        .arg(match args.io {
            crate::loader::Io::Read => "read",
            crate::loader::Io::Mmap => "mmap",
        });

    match args.preview {
        Some(t) => cmd.arg("--preview").arg(format!("{}x{}", t.width, t.height)),
        None => cmd.arg("--preview").arg("off"),
    };
    if let Some((w, h)) = args.screen {
        cmd.arg("--screen").arg(format!("{w}x{h}"));
    }
    cmd.args(&args.files);

    // Keep the child's own waterfall out of the way unless it fails.
    let out = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).output().context("spawning child")?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("{}", err.lines().rev().take(3).collect::<Vec<_>>().join("; "));
    }

    let label = adapter.label();
    let rows = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| parse_summary(line, &label))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        anyhow::bail!("child produced no summary");
    }
    Ok(rows)
}

fn parse_summary(line: &str, label: &str) -> Option<(String, Row)> {
    let mut fields = line.split('\t');
    if fields.next()? != "show-summary" {
        return None;
    }

    let mut kv = BTreeMap::new();
    for field in fields {
        if let Some((k, v)) = field.split_once('=') {
            kv.insert(k, v.to_string());
        }
    }
    let ms = |key: &str| -> Duration {
        Duration::from_secs_f64(kv.get(key).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) / 1e3)
    };

    Some((
        kv.get("file")?.clone(),
        Row {
            label: label.to_string(),
            init: ms("init"),
            upload: ms("upload"),
            draw: ms("draw"),
            decode: ms("decode"),
            cpu: ms("cpu"),
        },
    ))
}

fn report(results: &BTreeMap<String, Vec<Row>>) {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;

    for (file, rows) in results {
        let mut rows: Vec<&Row> = rows.iter().collect();
        rows.sort_by_key(|r| r.gpu_total());

        let decode = rows.first().map(|r| r.decode).unwrap_or_default();
        let cpu = rows.first().map(|r| r.cpu).unwrap_or_default();

        println!();
        println!("═══ {file} ═══");
        println!(
            "  decode is adapter-independent at {:.1} ms; the table ranks what happens after it",
            ms(decode)
        );
        println!(
            "  {:<34} {:>9} {:>9} {:>9} {:>10}  {}",
            "adapter", "init ms", "upload ms", "draw ms", "gpu ms", "vs best"
        );

        let best = rows.first().map(|r| r.gpu_total()).unwrap_or_default();
        for r in &rows {
            let ratio = if best.is_zero() { 1.0 } else { r.gpu_total().as_secs_f64() / best.as_secs_f64() };
            println!(
                "  {:<34} {:>9.1} {:>9.1} {:>9.1} {:>10.1}  {:>5.2}x",
                r.label,
                ms(r.init),
                ms(r.upload),
                ms(r.draw),
                ms(r.gpu_total()),
                ratio
            );
        }
        println!(
            "  {:<34} {:>9} {:>9} {:>9} {:>10.1}  {:>5.2}x  (scale + blit, no graphics API)",
            "cpu only",
            "-",
            "-",
            "-",
            ms(cpu),
            if best.is_zero() { 1.0 } else { cpu.as_secs_f64() / best.as_secs_f64() }
        );

        if let Some(winner) = rows.first() {
            println!(
                "  fastest to pixels: {} at {:.1} ms total ({:.1} ms decode + {:.1} ms gpu)",
                winner.label,
                ms(decode + winner.gpu_total()),
                ms(decode),
                ms(winner.gpu_total())
            );
        }
    }
}
