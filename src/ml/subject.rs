//! What a model is being pointed at.
//!
//! Two kinds, because the app holds them differently. A film keeps its frames
//! — they are decoded anyway and shared by reference — so it hands one over.
//! A still gives its pixels to the GPU and frees them, so it hands over its
//! path instead and whoever wants the pixels reads them again. Holding a
//! second copy of every photograph on the chance that a model might want it
//! would cost more than the occasional re-decode does.

use crate::video::Frame;

pub enum Subject {
    Frame(Frame),
    Picture(std::path::PathBuf),
}

/// Enough resolution for any of these models, and no more.
///
/// They all look at a thousand pixels across or fewer, so decoding a 48
/// megapixel photograph whole would be time spent on detail that is thrown
/// away in the next step.
const ENOUGH: crate::decode::Target = crate::decode::Target { width: 2048, height: 2048 };

impl Subject {
    /// The pixels, in the planar form every model here expects.
    pub fn frame(&self) -> Option<Frame> {
        match self {
            Subject::Frame(frame) => Some(frame.clone_ref()),
            Subject::Picture(path) => {
                let loaded = crate::loader::load(path, crate::loader::Io::Mmap, Some(ENOUGH)).ok()?;
                let picture = &loaded.decoded;
                super::prep::from_pixels(&picture.rgba, picture.width, picture.height)
            }
        }
    }
}
