//! Headless decode benchmark. No window, no GPU — just the I/O and decode
//! stages, repeated, so their cost can be compared across files and formats
//! without frame pacing or compositor latency in the way.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::decode::{self, Target};
use crate::loader::{Io, human_bytes};

struct Stats {
    min: Duration,
    median: Duration,
    mean: Duration,
}

fn stats(mut samples: Vec<Duration>) -> Stats {
    samples.sort_unstable();
    let sum: Duration = samples.iter().sum();
    Stats {
        min: samples[0],
        median: samples[samples.len() / 2],
        mean: sum / samples.len() as u32,
    }
}

pub fn run(paths: &[std::path::PathBuf], iterations: usize, io: Io, target: Option<Target>) -> Result<()> {
    println!(
        "{:<28} {:>8} {:>11} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "file", "format", "pixels", "size", "io ms", "decode ms", "total ms", "MPix/s"
    );
    println!("{}", "-".repeat(104));

    let mut failures = Vec::new();

    for path in paths {
        match bench_one(path, iterations, io, target) {
            Ok(()) => {}
            Err(e) => failures.push((path.clone(), e)),
        }
    }

    if !failures.is_empty() {
        println!();
        println!("{} file(s) failed:", failures.len());
        for (path, e) in &failures {
            println!("  {}: {e:#}", path.display());
        }
    }
    Ok(())
}

fn bench_one(path: &Path, iterations: usize, io: Io, target: Option<Target>) -> Result<()> {
    let file_size = std::fs::metadata(path)?.len();

    let mut io_samples = Vec::with_capacity(iterations);
    let mut decode_samples = Vec::with_capacity(iterations);
    let mut dims = (0u32, 0u32);
    let mut format = None;
    let mut path_name = "";

    for _ in 0..iterations {
        let t0 = Instant::now();
        // Both modes are re-done every iteration: the point is to measure the
        // warm-cache cost of the whole acquire step, mapping included.
        let bytes: Box<dyn std::ops::Deref<Target = [u8]>> = match io {
            Io::Read => Box::new(std::fs::read(path)?),
            Io::Mmap => {
                let f = std::fs::File::open(path)?;
                let m = unsafe { memmap2::Mmap::map(&f)? };
                // Fault the pages in now so the cost lands in the io column
                // instead of leaking into decode.
                let _ = m.iter().step_by(4096).fold(0u8, |a, b| a ^ b);
                Box::new(m)
            }
        };
        io_samples.push(t0.elapsed());

        let t1 = Instant::now();
        let d = decode::decode_path(&bytes, path, target)?;
        decode_samples.push(t1.elapsed());

        dims = (d.width, d.height);
        format = Some(d.format);
        path_name = d.path;
        std::hint::black_box(&d.rgba);
    }

    let io_s = stats(io_samples);
    let dec_s = stats(decode_samples);
    let total = io_s.median + dec_s.median;
    let mpix = (dims.0 as f64 * dims.1 as f64) / 1e6;
    let throughput = mpix / total.as_secs_f64();

    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let name = if name.len() > 27 { format!("{}…", &name[..26]) } else { name.to_string() };

    println!(
        "{:<28} {:>8} {:>11} {:>9} {:>9.3} {:>9.3} {:>9.3} {:>10.1}  [{}]",
        name,
        format.map(|f| f.name()).unwrap_or("?"),
        format!("{}x{}", dims.0, dims.1),
        human_bytes(file_size),
        io_s.median.as_secs_f64() * 1e3,
        dec_s.median.as_secs_f64() * 1e3,
        total.as_secs_f64() * 1e3,
        throughput,
        path_name,
    );
    let _ = (io_s.min, io_s.mean, dec_s.min, dec_s.mean);
    Ok(())
}
