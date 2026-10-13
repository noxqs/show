//! What can be installed, pinned at build time.
//!
//! The manifest is compiled in rather than fetched, which is the whole trust
//! model: `show` never runs a graph described by something it downloaded. It
//! downloads opaque blobs, from a URL it already knew, and checks them against
//! a digest it already knew. An adversary who controls the mirror can withhold
//! a file or serve a wrong one, and gets a `DIGEST MISMATCH` for the trouble.
//!
//! The cost is that a new model means a new release. That is deliberate: the
//! trust boundary stays at the binary, and it matches how the F1 panel already
//! reports formats — as facts about this build.

/// Something that can be installed.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Capability {
    /// ONNX Runtime itself. Everything else needs it.
    Runtime,
    /// Boxes and labels: what is in the picture.
    Detect,
    /// Per-pixel outlines of the things found.
    Segment,
    /// The same, from a larger model: better outlines, more time each.
    SegmentBetter,
    /// The largest of them, for a machine with the cores to spare.
    SegmentBest,
    /// Run the models on an NVIDIA card rather than the processor.
    Cuda,
    /// The outline of whatever was clicked, whether or not it has a name.
    PointSegment,
    /// The outline of everything matching a phrase that was typed.
    TextSegment,
    /// How far away everything is.
    Depth,
}

impl Capability {
    /// The identifier used on the command line.
    pub fn id(self) -> &'static str {
        match self {
            Capability::Runtime => "runtime",
            Capability::Detect => "detect",
            Capability::Segment => "segment",
            Capability::SegmentBetter => "segment-better",
            Capability::SegmentBest => "segment-best",
            Capability::Cuda => "cuda",
            Capability::PointSegment => "cutout",
            Capability::TextSegment => "sam3",
            Capability::Depth => "depth",
        }
    }

    /// Whether this is one of the models that can do the looking.
    ///
    /// They are alternatives to each other, so exactly one is used and which
    /// one is a choice rather than an accident of what was installed last.
    pub fn is_looker(self) -> bool {
        matches!(
            self,
            Capability::Detect
                | Capability::Segment
                | Capability::SegmentBetter
                | Capability::SegmentBest
        )
    }

    pub fn from_id(id: &str) -> Option<Capability> {
        CATALOGUE.iter().map(|s| s.id).find(|c| c.id() == id)
    }
}

/// One file to fetch.
pub struct Artefact {
    pub url: &'static str,
    /// Lowercase hex, 64 characters.
    pub sha256: &'static str,
    pub bytes: u64,
    /// Inside an archive, the entry to take; the file is used whole when this
    /// is `None`.
    pub member: Option<&'static str>,
    /// Several entries to take from one archive, each keeping its own name.
    ///
    /// For a download that carries a set of libraries that only work
    /// together — a runtime and its providers, or cuDNN and the half-dozen
    /// pieces it is split into. Fetching the archive once and taking them all
    /// is the difference between one download and six.
    pub members: &'static [&'static str],
    /// What it is called once installed. With `members`, the name recorded for
    /// the set — the files themselves keep their own names.
    pub name: &'static str,
}

pub struct Spec {
    pub id: Capability,
    /// Shown on the plugins page.
    pub name: &'static str,
    /// Which build of the thing, and the directory it installs into.
    pub version: &'static str,
    /// What it is, in one line.
    pub what: &'static str,
    /// Shown before anything is downloaded.
    pub licence: &'static str,
    pub requires: &'static [Capability],
    pub files: &'static [Artefact],
}

impl Spec {
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }
}

pub const CATALOGUE: &[Spec] = &[
    Spec {
        id: Capability::Runtime,
        name: "ONNX RUNTIME",
        version: "1.22.0",
        what: "runs every model; nothing else works without it",
        licence: "MIT",
        requires: &[],
        // Taken from the Python wheel rather than the GitHub tarball: PyPI
        // publishes a digest for every file it serves, so the constant below
        // has a published source rather than one we computed once and hoped
        // about.
        files: &[Artefact {
            url: "https://files.pythonhosted.org/packages/8c/60/16d219b8868cc8e8e51a68519873bdb9f5f24af080b62e917a13fff9989b/onnxruntime-1.22.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            sha256: "6964a975731afc19dc3418fad8d4e08c48920144ff590149429a5ebe0d15fb3c",
            bytes: 16_406_377,
            member: Some("libonnxruntime.so"),
        members: &[],
            name: "libonnxruntime.so",
        }],
    },
    Spec {
        id: Capability::Cuda,
        name: "NVIDIA ACCELERATION",
        version: "cuda-12",
        what: "runs the models on the graphics card: ten times faster, a gigabyte to install",
        licence: "MIT (RUNTIME) + NVIDIA CUDNN LICENCE",
        requires: &[Capability::Runtime],
        // Two archives, and everything needed comes out of them. The runtime
        // itself has to be the build that knows about CUDA — the plain one
        // cannot be told to use a card it was not built for — and cuDNN comes
        // in seven pieces that only work as a set. Everything else CUDA needs
        // is expected to be on the machine already: this pins what is specific
        // to running these models, not a driver stack.
        files: &[
            Artefact {
                url: "https://files.pythonhosted.org/packages/b5/5c/3f9700ba277d52c121dd2cebc8a672fb60b53e888972fc6682b6692a766c/onnxruntime_gpu-1.22.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
                sha256: "86b064c8f6cbe6da03f51f46351237d985f8fd5eb907d3f9997ea91881131a13",
                bytes: 283_199_528,
                member: None,
                members: &[
                    "libonnxruntime.so",
                    "libonnxruntime_providers_shared.so",
                    "libonnxruntime_providers_cuda.so",
                ],
                name: "onnxruntime-gpu",
            },
            Artefact {
                url: "https://files.pythonhosted.org/packages/74/27/450c690b2ed878fd323dce0fcc6ed1746013210bc9c395f30eaf6fb03b09/nvidia_cudnn_cu12-9.25.1.1-py3-none-manylinux_2_27_x86_64.whl",
                sha256: "515df94e4060bb58c88588e46fb39a5df310659d2d98733d1697ed667b4895d1",
                bytes: 751_246_851,
                member: None,
                // All ten of them. cuDNN is a dispatcher that opens the rest
                // by name at run time, so a missing one is not a link error
                // but a message about a library it could not find, arriving in
                // the middle of the first inference.
                members: &[
                    "libcudnn.so.9",
                    "libcudnn_graph.so.9",
                    "libcudnn_ops.so.9",
                    "libcudnn_adv.so.9",
                    "libcudnn_cnn.so.9",
                    "libcudnn_ext.so.9",
                    "libcudnn_heuristic.so.9",
                    "libcudnn_engines_precompiled.so.9",
                    "libcudnn_engines_runtime_compiled.so.9",
                    "libcudnn_engines_tensor_ir.so.9",
                ],
                name: "cudnn",
            },
        ],
    },
    Spec {
        id: Capability::Detect,
        name: "OBJECT DETECTION",
        version: "yolo26n",
        what: "finds people, vehicles and 78 other everyday things",
        // Ultralytics' weights are AGPL-3.0. Shown here because a licence
        // nobody was told about is a licence nobody agreed to.
        licence: "AGPL-3.0",
        requires: &[Capability::Runtime],
        // Pinned to a commit rather than a branch, and to a digest the host
        // publishes: Hugging Face records a sha256 for every large file, so
        // the constant below has a source rather than being one we computed
        // once and hoped about.
        files: &[Artefact {
            url: "https://huggingface.co/onnx-community/yolo26n-ONNX/resolve/a8dc7e14743e1cea8ccd493bd99b4c2827de1acf/onnx/model.onnx",
            sha256: "cda08d9440217e243e075ee839f40383c59b3f973e493f9f6c7452922a69436e",
            bytes: 9_890_454,
            member: None,
            members: &[],
            name: "model.onnx",
        }],
    },
    Spec {
        id: Capability::Segment,
        name: "SEGMENTATION",
        version: "yolo26n-seg",
        what: "outlines what it finds, rather than boxing it",
        licence: "AGPL-3.0",
        requires: &[Capability::Runtime],
        files: &[Artefact {
            url: "https://huggingface.co/EdgeFirst/yolo26-seg/resolve/bf07618a0f12bbfe3f26eca3d7bce456dec26b86/onnx/yolo26n-seg-fp32.onnx",
            sha256: "50b9866809b1fd88cf81ab3625c8de4eabd760b8f2ddcd85e2204a8f6013ff72",
            bytes: 11_213_068,
            member: None,
            members: &[],
            name: "model.onnx",
        }],
    },
    Spec {
        id: Capability::SegmentBetter,
        name: "SEGMENTATION (BETTER)",
        version: "yolo26s-seg",
        // Same family, same output, same licence — four times the weights and
        // roughly twice the time per look. Installed alongside the small one
        // rather than instead of it, and preferred while it is there.
        what: "the same, from a larger model: closer outlines, twice the time",
        licence: "AGPL-3.0",
        requires: &[Capability::Runtime],
        files: &[Artefact {
            url: "https://huggingface.co/EdgeFirst/yolo26-seg/resolve/bf07618a0f12bbfe3f26eca3d7bce456dec26b86/onnx/yolo26s-seg-fp32.onnx",
            sha256: "1acf6a989e2b8aaaeb9231925b98da0424ba842f63f45d11b66c7b8d7bb45901",
            bytes: 41_906_639,
            member: None,
            members: &[],
            name: "model.onnx",
        }],
    },
    Spec {
        id: Capability::SegmentBest,
        name: "SEGMENTATION (BEST)",
        version: "yolo26m-seg",
        // Four times the time of the small model. It still costs the film
        // nothing, because looking happens five times a second on a thread of
        // its own and a frame offered while it is busy is simply skipped —
        // but on fewer cores than this was measured on, that skipping is what
        // will happen, and the readout will lag the picture.
        what: "the largest of them: closest outlines, four times the time",
        licence: "AGPL-3.0",
        requires: &[Capability::Runtime],
        files: &[Artefact {
            url: "https://huggingface.co/EdgeFirst/yolo26-seg/resolve/bf07618a0f12bbfe3f26eca3d7bce456dec26b86/onnx/yolo26m-seg-fp32.onnx",
            sha256: "f63b12b193af4221f7081a1830165aee1d400a4c5ad5c9270ea23ed12494d9a0",
            bytes: 94_607_708,
            member: None,
            members: &[],
            name: "model.onnx",
        }],
    },
    Spec {
        id: Capability::TextSegment,
        name: "SEGMENT WHAT I TYPE",
        version: "sam3-vit-h",
        what: "finds every one of whatever you name, named or not by the others",
        // The exporter's own manifest names this, and it is Meta's licence
        // rather than the exporter's: the MIT wrapper does not relicence the
        // weights, which it says itself. Stated as it is rather than rounded
        // to something friendlier.
        licence: "META SAM LICENSE",
        requires: &[Capability::Runtime],
        files: &[
            // One archive, and only the parts that are read. The decoder's
            // external data is in there and deliberately not taken: the
            // manifest marks it `referenced_by_graph: false`, and the decoder
            // was confirmed to load without it — 116 MB nobody has to fetch.
            Artefact {
                url: "https://huggingface.co/vietanhdev/segment-anything-3-onnx-models/resolve/main/sam3_vit_h.zip",
                sha256: "1617436f9db7eb9dd69029c2c465d54b316c36c21143f0fde0d1be513f15b7d5",
                bytes: 3_413_511_219,
                member: None,
                members: &[
                    "sam3_image_encoder.onnx",
                    "sam3_image_encoder.onnx.data",
                    "sam3_language_encoder.onnx",
                    "sam3_language_encoder.onnx.data",
                    "sam3_decoder.onnx",
                ],
                name: "sam3",
            },
            // The tokenizer's merges, which the archive does not carry. CLIP's
            // own, from a repository that is not behind a licence click, and
            // pinned to a commit rather than to a branch. The vocabulary is
            // not fetched: it is derived from this file, and the derivation
            // was checked against the published table entry for entry.
            Artefact {
                url: "https://huggingface.co/openai/clip-vit-base-patch32/resolve/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/merges.txt",
                sha256: "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
                bytes: 524_657,
                member: None,
                members: &[],
                name: "merges.txt",
            },
        ],
    },
    Spec {
        id: Capability::PointSegment,
        name: "CUT OUT WHAT I CLICK",
        version: "mobile-sam",
        what: "outlines anything you point at, named or not",
        // MobileSAM's own weights are Apache-2.0. This mirror publishes no
        // licence of its own, which is said plainly rather than papered over:
        // the pin is to a commit and a digest, so what arrives is fixed even
        // where the mirror's terms are not stated.
        licence: "APACHE-2.0 (MIRROR STATES NONE)",
        requires: &[Capability::Runtime],
        // Two files, and both are needed: the encoder turns a picture into an
        // embedding once, the decoder answers a click against that embedding
        // in milliseconds. The split is what makes a second click instant.
        files: &[
            Artefact {
                url: "https://huggingface.co/Acly/MobileSAM/resolve/0d3b403339b4674a82493d5e97964dd78089ddc8/mobile_sam_image_encoder.onnx",
                sha256: "580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749",
                bytes: 28_157_093,
                member: None,
            members: &[],
                name: "encoder.onnx",
            },
            Artefact {
                url: "https://huggingface.co/Acly/MobileSAM/resolve/0d3b403339b4674a82493d5e97964dd78089ddc8/sam_mask_decoder_multi.onnx",
                sha256: "8976b90a87ba50a6a72217a5ff994f7d25ce16f2229fcc1ed259e1294c622ffe",
                bytes: 16_496_559,
                member: None,
            members: &[],
                name: "decoder.onnx",
            },
        ],
    },
    Spec {
        id: Capability::Depth,
        name: "DEPTH",
        version: "depth-anything-v2-small",
        what: "how near or far everything is, as colour over the picture",
        // The small model only: the base and large ones are CC-BY-NC.
        licence: "APACHE-2.0",
        requires: &[Capability::Runtime],
        files: &[Artefact {
            url: "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx",
            sha256: "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
            bytes: 99_060_839,
            member: None,
            members: &[],
            name: "model.onnx",
        }],
    },
];

pub fn spec(id: Capability) -> &'static Spec {
    CATALOGUE.iter().find(|s| s.id == id).expect("every capability is in the catalogue")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalogue is the trust boundary, so every field of it is checked.
    /// A typo in a digest is a plugin that can never install, and a typo in a
    /// URL is worse — it is a download nobody vetted.
    #[test]
    fn every_entry_is_pinned_properly() {
        for spec in CATALOGUE {
            assert!(!spec.name.is_empty(), "{:?} has no name", spec.id);
            assert!(!spec.licence.is_empty(), "{} has no licence", spec.name);
            assert!(!spec.what.is_empty(), "{} does not say what it is", spec.name);
            assert!(!spec.files.is_empty(), "{} downloads nothing", spec.name);

            for file in spec.files {
                assert!(
                    file.url.starts_with("https://"),
                    "{}: {} is not https",
                    spec.name,
                    file.url
                );
                assert_eq!(
                    file.sha256.len(),
                    64,
                    "{}: digest is {} characters, not 64",
                    spec.name,
                    file.sha256.len()
                );
                assert!(
                    file.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                    "{}: digest is not lowercase hex",
                    spec.name
                );
                assert!(file.bytes > 0, "{}: zero-byte artefact", spec.name);
                assert!(!file.name.contains('/'), "{}: name must not be a path", spec.name);
            }
        }
    }

    #[test]
    fn every_requirement_names_something_that_exists() {
        for spec in CATALOGUE {
            for needed in spec.requires {
                assert!(
                    CATALOGUE.iter().any(|s| s.id == *needed),
                    "{} requires {:?}, which is not in the catalogue",
                    spec.name,
                    needed
                );
                assert_ne!(*needed, spec.id, "{} requires itself", spec.name);
            }
        }
    }

    #[test]
    fn identifiers_are_unique_and_round_trip() {
        let mut seen = Vec::new();
        for spec in CATALOGUE {
            let id = spec.id.id();
            assert!(!seen.contains(&id), "duplicate identifier {id}");
            assert_eq!(Capability::from_id(id), Some(spec.id));
            seen.push(id);
        }
        assert_eq!(Capability::from_id("nothing-like-this"), None);
    }
}