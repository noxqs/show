//! What is in the card's memory, and who put it there.
//!
//! The processor's memory has a number everyone agrees on; the card's does
//! not. `wgpu` knows what it allocated, the inference runtime knows what it
//! allocated, and neither knows what the compositor and the browser took
//! before either of them started — which on a twelve-gigabyte card is most of
//! the reason a model fails to load. Only the driver sees all of it, so this
//! asks the driver.
//!
//! NVML is loaded at runtime, for the same reason ALSA and libheif are: it
//! ships with the driver, and a viewer that refused to start without one
//! would be a viewer that refuses to start on every machine with an AMD card.
//! Absent, this reports nothing and the readout simply does not appear.
//!
//! Two things here are worth knowing before changing them:
//!
//! * `nvmlDeviceGetMemoryInfo` — the original, un-suffixed call — returns
//!   **zeroes** on a current driver (595 here) while reporting success. The
//!   `_v2` form is the one that answers, and it wants its own size and version
//!   written into the struct before the call. The v1 path is kept only as a
//!   fallback for drivers old enough to lack v2.
//! * the per-process list is two lists. Compute processes are the ones running
//!   inference; *graphics* processes are X, the compositor and the browser, and
//!   they are the ones holding the memory somebody is looking for. Asking only
//!   for compute processes — which is what `nvidia-smi --query-compute-apps`
//!   does — returns an empty list on an idle machine and explains nothing.
//!
//! The first card only. A second one would need a choice of which to report,
//! and the question this answers is about the card the pictures are on.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};

use libloading::{Library, Symbol};

/// NVML's success code. Everything else is a failure we report as "unknown".
const OK: c_int = 0;

/// `nvmlMemory_v2_t`. The version field is the struct's own size with the
/// version number in the top byte, which is how NVML tells the two layouts
/// apart over one symbol.
#[repr(C)]
#[derive(Default)]
struct MemoryV2 {
    version: c_uint,
    total: u64,
    reserved: u64,
    free: u64,
    used: u64,
}

/// `nvmlMemory_t`, the original three-field form.
#[repr(C)]
#[derive(Default)]
struct MemoryV1 {
    total: u64,
    free: u64,
    used: u64,
}

/// `nvmlProcessInfo_v2_t`, which is what the `_v3` calls fill in.
///
/// The trailing instance ids are for partitioned cards and are read but not
/// used; they are here because leaving them out would make the struct the
/// wrong size and the list would come back as rubbish.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: c_uint,
    used: u64,
    gpu_instance: c_uint,
    compute_instance: c_uint,
}

type InitFn = unsafe extern "C" fn() -> c_int;
type HandleFn = unsafe extern "C" fn(c_uint, *mut *mut c_void) -> c_int;
type MemoryV2Fn = unsafe extern "C" fn(*mut c_void, *mut MemoryV2) -> c_int;
type MemoryV1Fn = unsafe extern "C" fn(*mut c_void, *mut MemoryV1) -> c_int;
type ProcessesFn = unsafe extern "C" fn(*mut c_void, *mut c_uint, *mut ProcessInfo) -> c_int;
type NameFn = unsafe extern "C" fn(c_uint, *mut c_char, c_uint) -> c_int;

struct Api {
    _library: Library,
    device: *mut c_void,
    memory_v2: Option<MemoryV2Fn>,
    memory_v1: Option<MemoryV1Fn>,
    graphics: Option<ProcessesFn>,
    compute: Option<ProcessesFn>,
    name: Option<NameFn>,
}

// The handle is an opaque driver pointer, used only as an argument to the
// library that produced it, and NVML's own documentation calls its interface
// thread-safe.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    fn load() -> Option<Api> {
        // Safety: opening a shared library and reading documented symbols from
        // it. Every signature below is NVML's published C API, and the structs
        // are the published layouts — checked against the library's own
        // answers rather than assumed, which is how the v1 memory call was
        // found to be silently empty.
        unsafe {
            let library = Library::new("libnvidia-ml.so.1").ok()?;
            fn get<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
                // Safety: the caller names the type, and each use below matches
                // the documented signature for that symbol.
                unsafe { lib.get::<T>(name).ok().map(|s: Symbol<T>| *s) }
            }

            // `nvmlInit_v2` is the one every driver this decade has. Without
            // it nothing else in the library may be called.
            let init: InitFn = get(&library, b"nvmlInit_v2\0")?;
            if init() != OK {
                return None;
            }
            let handle: HandleFn = get(&library, b"nvmlDeviceGetHandleByIndex_v2\0")?;
            let mut device = std::ptr::null_mut();
            if handle(0, &mut device) != OK {
                return None;
            }
            Some(Api {
                memory_v2: get(&library, b"nvmlDeviceGetMemoryInfo_v2\0"),
                memory_v1: get(&library, b"nvmlDeviceGetMemoryInfo\0"),
                graphics: get(&library, b"nvmlDeviceGetGraphicsRunningProcesses_v3\0"),
                compute: get(&library, b"nvmlDeviceGetComputeRunningProcesses_v3\0"),
                name: get(&library, b"nvmlSystemGetProcessName\0"),
                device,
                _library: library,
            })
        }
    }

    /// Total, free, used and reserved, in bytes.
    fn memory(&self) -> Option<(u64, u64, u64, u64)> {
        if let Some(call) = self.memory_v2 {
            let mut out = MemoryV2 {
                // Size and version in one word, as the header's macro builds
                // it. Wrong, and the call refuses rather than misreading.
                version: (std::mem::size_of::<MemoryV2>() as c_uint) | (2 << 24),
                ..MemoryV2::default()
            };
            // Safety: the struct matches the documented v2 layout and is
            // initialised before the call.
            if unsafe { call(self.device, &mut out) } == OK && out.total > 0 {
                return Some((out.total, out.free, out.used, out.reserved));
            }
        }
        let call = self.memory_v1?;
        let mut out = MemoryV1::default();
        // Safety: as above, for the three-field form.
        if unsafe { call(self.device, &mut out) } == OK && out.total > 0 {
            return Some((out.total, out.free, out.used, 0));
        }
        None
    }

    /// Every process holding card memory, graphics and compute together.
    fn processes(&self) -> Vec<(u32, u64)> {
        /// Generous: a desktop with a browser open runs to a handful, and the
        /// call tells us if it needed more room than it was given.
        const ROOM: usize = 64;

        let mut out: Vec<(u32, u64)> = Vec::new();
        for call in [self.graphics, self.compute].into_iter().flatten() {
            let mut list = [ProcessInfo::default(); ROOM];
            let mut count = ROOM as c_uint;
            // Safety: `count` says how many entries the buffer holds, which is
            // the contract; on return it says how many were written.
            if unsafe { call(self.device, &mut count, list.as_mut_ptr()) } != OK {
                continue;
            }
            for info in list.iter().take((count as usize).min(ROOM)) {
                // A process can appear in both lists — a browser that also
                // decodes video does — and it is one consumer either way.
                if let Some(seen) = out.iter_mut().find(|(pid, _)| *pid == info.pid) {
                    seen.1 = seen.1.max(info.used);
                } else {
                    out.push((info.pid, info.used));
                }
            }
        }
        out
    }

    /// What to call a process.
    ///
    /// `/proc` first, because it gives the short name — `Xorg`, `chrome` —
    /// where NVML gives the full command line, and a readout eighteen
    /// characters wide cannot show a Chrome command line. NVML is the fallback
    /// for a process owned by somebody else, whose `/proc` entry we may not be
    /// allowed to read.
    fn name_of(&self, pid: u32) -> String {
        if let Ok(comm) = std::fs::read_to_string(format!("/proc/{pid}/comm")) {
            let comm = comm.trim();
            if !comm.is_empty() {
                return comm.to_string();
            }
        }
        if let Some(call) = self.name {
            let mut buf = [0u8; 128];
            // Safety: the buffer's own length is passed, and NVML writes a
            // null-terminated string within it.
            if unsafe { call(pid, buf.as_mut_ptr() as *mut c_char, buf.len() as c_uint) } == OK {
                let text = CStr::from_bytes_until_nul(&buf).ok().and_then(|c| c.to_str().ok());
                if let Some(text) = text {
                    // The last path component, for the same width reason.
                    let short = text.rsplit('/').next().unwrap_or(text);
                    if !short.is_empty() {
                        return short.to_string();
                    }
                }
            }
        }
        format!("PID {pid}")
    }
}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(Api::load).as_ref()
}

/// One process holding card memory.
pub struct Process {
    pub name: String,
    pub bytes: u64,
    /// Whether this is us, so the readout can say so.
    pub ours: bool,
}

/// What the card's memory is doing, at one moment.
pub struct Usage {
    pub total: u64,
    pub free: u64,
    pub used: u64,
    /// The driver's own overhead, which is neither free nor anybody's.
    pub reserved: u64,
    /// Biggest consumer first.
    pub processes: Vec<Process>,
}

/// Ask the driver. `None` when there is no NVIDIA card or no driver to ask.
pub fn sample() -> Option<Usage> {
    let api = api()?;
    let (total, free, used, reserved) = api.memory()?;
    let us = std::process::id();
    let mut processes: Vec<Process> = api
        .processes()
        .into_iter()
        // The driver lists a process with no attributable memory now and then;
        // a row reading zero is noise.
        .filter(|(_, bytes)| *bytes > 0)
        .map(|(pid, bytes)| Process { name: api.name_of(pid), bytes, ours: pid == us })
        .collect();
    processes.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    Some(Usage { total, free, used, reserved, processes })
}

/// What this process is holding, in bytes.
///
/// Absent from the driver's list means nothing held, not "unknown" — and the
/// difference matters, because that is exactly where we are standing when the
/// first graph is about to be loaded. Reading it as unknown there loses the
/// largest charge of the session, which is the one somebody wanted to see.
/// `None` is reserved for having no card to ask.
pub fn ours() -> Option<u64> {
    let api = api()?;
    let us = std::process::id();
    Some(api.processes().into_iter().find(|(pid, _)| *pid == us).map_or(0, |(_, bytes)| bytes))
}

/// What each graph cost when it went onto the card.
static CHARGED: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());

/// Record what loading `label` added to our own usage.
///
/// Measured as a difference rather than asked for directly, because nothing
/// reports it directly: the runtime does not say what a session took, and the
/// driver only counts per process. The first graph onto the card is charged
/// with the card's context as well — several hundred megabytes that are paid
/// once, not per model — which is honest about what switching it on costs, and
/// worth remembering when reading the first row.
pub fn charge(label: &str, before: Option<u64>) {
    let (Some(before), Some(after)) = (before, ours()) else { return };
    let grew = after.saturating_sub(before);
    if grew == 0 {
        return;
    }
    let mut charged = CHARGED.lock().expect("vram charges");
    match charged.iter_mut().find(|(had, _)| had == label) {
        // Re-loaded, rather than two of them: the older figure is stale.
        Some(seen) => seen.1 = grew,
        None => charged.push((label.to_string(), grew)),
    }
}

/// Load a graph, and charge the card memory it takes against its name.
///
/// A wrapper rather than something the loaders each remember to call around
/// their own code: there are three places that open a session, they open them
/// in three different ways for three different reasons, and the one thing they
/// must agree on is that the measurement brackets the load exactly. A model
/// that lands on the processor costs no card memory and so records nothing,
/// which is the right answer without anywhere having to ask where it landed.
pub fn charging<T, E>(
    path: &std::path::Path,
    load: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let before = ours();
    let loaded = load();
    if loaded.is_ok() {
        charge(&label_for(path), before);
    }
    loaded
}

/// The charges so far, in the order they were incurred.
pub fn charges() -> Vec<(String, u64)> {
    CHARGED.lock().expect("vram charges").clone()
}

/// A name for a graph, from the file it was loaded from.
///
/// Three shapes of name arrive here and each needs something different. SAM 3's
/// files say what they are — `sam3_image_encoder` — and need nothing added.
/// Every detector ships as the bare `model.onnx`, where the directory is the
/// only part that says which model it is. The cut-out's are `encoder.onnx` and
/// `decoder.onnx`, which say what the graph does but not whose it is, and a row
/// reading only `ENCODER` on a panel that may also hold SAM 3's encoder names
/// nothing at all.
pub fn label_for(path: &std::path::Path) -> String {
    let tidy = |s: String| s.replace(['_', '-'], " ").to_uppercase();
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let folder = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    if stem == "model" || stem.is_empty() {
        return tidy(folder);
    }
    // Whether the file already says whose it is: the first word of the
    // directory — `mobile` of `mobile-sam`, `sam3` of `sam3-vit-h` — appearing
    // in the file's own name is what makes the directory redundant.
    let whose = folder.split(['-', '_']).next().unwrap_or_default();
    if whose.is_empty() || stem.contains(whose) {
        return tidy(stem);
    }
    tidy(format!("{folder} {stem}"))
}

/// The readout's lines, in the order they are shown: what is left, what our
/// own graphs took, and who else is holding some.
///
/// Free first, because that is the number being looked for — the question this
/// answers is "will the next model fit", and the answer is the top line. Then
/// the graphs, each charged what it added when it loaded, then every other
/// process biggest first.
///
/// Separate from the drawing so it can be tested: the lumping and the ordering
/// are the parts with decisions in them, and a test that had to rasterise text
/// to check them would be testing the font.
pub fn figures(usage: &Usage, charges: &[(String, u64)]) -> Vec<(String, u64)> {
    /// Other processes worth a row of their own. Beyond this they are a sum: a
    /// browser with eight helpers would otherwise push everything that matters
    /// off the top of the panel.
    const OTHERS: usize = 5;

    let mut out = vec![("FREE".to_string(), usage.free)];
    out.extend(charges.iter().cloned());
    let ours: u64 = usage.processes.iter().filter(|p| p.ours).map(|p| p.bytes).sum();
    if ours > 0 {
        out.push(("SHOW".to_string(), ours));
    }
    let mut lumped = 0;
    for (i, process) in usage.processes.iter().filter(|p| !p.ours).enumerate() {
        if i >= OTHERS {
            lumped += process.bytes;
            continue;
        }
        out.push((process.name.to_uppercase(), process.bytes));
    }
    if lumped > 0 {
        out.push(("OTHERS".to_string(), lumped));
    }
    // The driver's own overhead is neither free nor anybody's, and on a
    // twelve-gigabyte card it is four hundred megabytes of the difference
    // somebody would otherwise go looking for.
    if usage.reserved > 0 {
        out.push(("DRIVER".to_string(), usage.reserved));
    }
    out
}

/// Those lines as the overlay draws them.
pub fn rows(usage: &Usage, figures: &[(String, u64)]) -> Vec<crate::osd::Row> {
    use crate::osd::Row;

    // Used against total on the first line: the rows beneath account for most
    // of it but never all of it, and seeing both figures is what stops the
    // difference reading as a mistake here.
    let mut rows = vec![Row::dim(
        "CARD MEMORY",
        format!("{} / {}", human(usage.used), human(usage.total)),
    )];
    // A tenth left is the one thing here worth colouring, because it is the one
    // thing that stops the next model loading.
    let low = usage.free * 10 < usage.total;
    for (i, (label, bytes)) in figures.iter().enumerate() {
        let value = human(*bytes);
        rows.push(match (i, low) {
            (0, true) => Row::bad(label, value),
            (0, false) => Row::good(label, value),
            _ => Row::dim(label, value),
        });
    }
    rows
}

/// Bytes as the readout shows them: gigabytes once there are gigabytes to
/// show, megabytes below that, and never more than three significant figures
/// because the column has to stay still while the numbers move.
pub fn human(bytes: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    let mb = bytes as f64 / MB;
    if mb >= 1024.0 {
        format!("{:.2} GB", mb / 1024.0)
    } else {
        format!("{mb:.0} MB")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_structs_match_the_layouts_nvml_publishes() {
        // Both are passed to the driver by pointer, so a wrong size is read as
        // a wrong field and reported as plausible nonsense rather than as an
        // error. The v2 memory struct is 40 bytes and the process entry 24.
        assert_eq!(std::mem::size_of::<MemoryV2>(), 40);
        assert_eq!(std::mem::size_of::<ProcessInfo>(), 24);
        assert_eq!(std::mem::size_of::<MemoryV1>(), 24);
    }

    #[test]
    fn a_graph_is_named_after_whichever_part_of_its_path_says_something() {
        assert_eq!(
            label_for(std::path::Path::new("/m/sam3/sam3_image_encoder.onnx")),
            "SAM3 IMAGE ENCODER"
        );
        // The detectors all ship as `model.onnx`; the directory is the name.
        assert_eq!(label_for(std::path::Path::new("/m/segment/yolo26m-seg/model.onnx")), "YOLO26M SEG");
        // The cut-out's graphs say what they do but not whose they are, and
        // `ENCODER` beside SAM 3's encoder would name nothing.
        assert_eq!(
            label_for(std::path::Path::new("/m/cutout/mobile-sam/encoder.onnx")),
            "MOBILE SAM ENCODER"
        );
    }

    #[test]
    fn sizes_read_in_the_unit_that_suits_them() {
        assert_eq!(human(479 * 1024 * 1024), "479 MB");
        assert_eq!(human(1024 * 1024 * 1024 + 512 * 1024 * 1024), "1.50 GB");
        assert_eq!(human(0), "0 MB");
    }

    fn usage_of(processes: Vec<Process>) -> Usage {
        Usage {
            total: 12 << 30,
            free: 10 << 30,
            used: 1 << 30,
            reserved: 400 << 20,
            processes,
        }
    }

    fn process(name: &str, mb: u64, ours: bool) -> Process {
        Process { name: name.to_string(), bytes: mb << 20, ours }
    }

    #[test]
    fn what_is_left_is_the_first_line() {
        let figures = figures(&usage_of(vec![process("Xorg", 479, false)]), &[]);
        assert_eq!(figures[0].0, "FREE");
        assert_eq!(figures[0].1, 10 << 30);
    }

    #[test]
    fn our_graphs_come_before_other_peoples_processes() {
        let charges = [("SAM3 IMAGE ENCODER".to_string(), 1900 << 20)];
        let figures = figures(
            &usage_of(vec![process("show", 2100, true), process("Xorg", 479, false)]),
            &charges,
        );
        let labels: Vec<&str> = figures.iter().map(|(l, _)| l.as_str()).collect();
        assert_eq!(labels, ["FREE", "SAM3 IMAGE ENCODER", "SHOW", "XORG", "DRIVER"]);
    }

    #[test]
    fn a_long_tail_of_helpers_is_summed_rather_than_listed() {
        // Eight browser helpers must not push the interesting rows off the top.
        let mut processes = vec![process("Xorg", 479, false)];
        for _ in 0..8 {
            processes.push(process("chrome", 10, false));
        }
        let figures = figures(&usage_of(processes), &[]);
        let lumped = figures.iter().find(|(l, _)| l == "OTHERS").expect("a sum of the rest");
        // Five listed, one of which is Xorg, so four helpers listed and four
        // summed.
        assert_eq!(lumped.1, 4 * (10 << 20));
        assert_eq!(figures.iter().filter(|(l, _)| l == "CHROME").count(), 4);
    }

    #[test]
    fn nothing_of_ours_means_no_row_claiming_otherwise() {
        let figures = figures(&usage_of(vec![process("Xorg", 479, false)]), &[]);
        assert!(figures.iter().all(|(l, _)| l != "SHOW"));
    }

    #[test]
    fn holding_nothing_yet_is_zero_rather_than_unknown() {
        // Only meaningful with a driver present; without one the whole readout
        // is absent and there is nothing to get wrong.
        if api().is_some() {
            assert!(ours().is_some(), "a card we can see must answer, even with nothing of ours on it");
        }
    }

    #[test]
    fn a_charge_without_a_before_is_not_recorded() {
        // The card may not be there, in which case `ours()` is `None` and a
        // subtraction would be invented rather than measured.
        charge("NOTHING AT ALL", None);
        assert!(charges().iter().all(|(label, _)| label != "NOTHING AT ALL"));
    }
}
