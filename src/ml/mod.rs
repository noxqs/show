//! Machine learning: preparing frames, running models, and what comes back.
//!
//! Deliberately layered so the parts that need no model can be built, tested
//! and measured on their own. `prep` is arithmetic over a decoded frame and
//! depends on nothing; the inference that will sit on top of it arrives later
//! and behind a feature flag, so a default build is unchanged.

/// Identity colours.
///
/// One colour means one identity everywhere it appears: the box round a thing,
/// the tag naming it, and later the face and the speaker it belongs to. Chosen
/// to stay apart from each other and from the interface's own green and amber.
/// Fifteen, because that is how many the mask shader can tell apart: ids are
/// palette slots, slot zero means nothing-here, and the shader's palette holds
/// sixteen. Eight left the last seven slots as repeats of the first seven, so
/// two objects could come out the same colour while seven colours went unused.
///
/// All light and unsaturated on purpose: they are drawn over a picture, at
/// full strength on the outline and at a sixth inside it, and a dark or
/// saturated colour there reads as part of the picture rather than as a mark
/// on it.
pub const PALETTE: [[u8; 4]; 15] = [
    [120, 190, 255, 255], // blue
    [255, 170, 120, 255], // orange
    [170, 230, 140, 255], // green
    [235, 150, 210, 255], // pink
    [235, 220, 130, 255], // yellow
    [150, 210, 225, 255], // cyan
    [200, 170, 245, 255], // violet
    [245, 160, 160, 255], // red
    [160, 235, 200, 255], // mint
    [225, 195, 150, 255], // sand
    [190, 200, 250, 255], // periwinkle
    [230, 180, 175, 255], // clay
    [180, 225, 165, 255], // moss
    [215, 165, 230, 255], // mauve
    [165, 215, 240, 255], // sky
];

/// The colour for a class or cluster index.
pub fn colour(index: usize) -> [u8; 4] {
    PALETTE[index % PALETTE.len()]
}

pub mod bar;
pub mod catalogue;
pub mod clip;
pub mod cutout;
pub mod depth;
pub mod finder;
pub mod labels;
pub mod pointer;
pub mod prep;
pub mod runtime;
pub mod sam3;
pub mod segment;
pub mod timeline;
pub mod track;
pub mod watcher;
pub mod vision;
pub mod store;
pub mod subject;
