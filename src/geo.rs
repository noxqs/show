//! Whether an image carries a placement on the map.
//!
//! Only detection: nothing here projects or reprojects anything. The viewer
//! shows a map as the picture it is, and says when it knows where that picture
//! belongs on the earth — a scanned map someone has georeferenced is a
//! different object from a scan of the same map, and worth marking.
//!
//! Three places the placement is kept, all read without touching a pixel:
//! GeoTIFF tags in the file, a GIS sidecar (world file or GDAL's `.aux.xml`),
//! and histomap's own `.histomap.json`.

use std::path::{Path, PathBuf};

/// Where a placement was found. More than one is common: a GeoTIFF that has
/// also been placed by hand in histomap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    GeoTiff,
    WorldFile,
    GdalAux,
    Histomap,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::GeoTiff => "GEOTIFF",
            Source::WorldFile => "WORLD FILE",
            Source::GdalAux => "GDAL",
            Source::Histomap => "HISTOMAP",
        }
    }
}

/// ModelPixelScale, ModelTiepoint, ModelTransformation, GeoKeyDirectory.
/// Any one of the first three places the raster; the last says in what.
const GEOTIFF_TAGS: [u16; 4] = [33550, 33922, 34264, 34735];

/// Every placement `path` has, in a fixed order; empty for none.
pub fn detect(path: &Path) -> Vec<Source> {
    let mut found = Vec::new();
    if geotiff(path) {
        found.push(Source::GeoTiff);
    }
    if world_file(path).is_some() {
        found.push(Source::WorldFile);
    }
    if gdal_aux(path) {
        found.push(Source::GdalAux);
    }
    if histomap(path) {
        found.push(Source::Histomap);
    }
    found
}

/// Reads the first directory only: a pyramid repeats its tags on every level,
/// and the first is enough to know.
fn geotiff(path: &Path) -> bool {
    let is_tiff = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "tif" | "tiff"));
    if !is_tiff {
        return false;
    }
    let Ok(file) = std::fs::File::open(path) else { return false };
    let Ok(mut decoder) = tiff::decoder::Decoder::new(std::io::BufReader::new(file)) else {
        return false;
    };
    GEOTIFF_TAGS
        .iter()
        .any(|t| matches!(decoder.find_tag(tiff::tags::Tag::Unknown(*t)), Ok(Some(_))))
}

/// The six-line affine sidecar GIS tools write: `map.tfw` beside `map.tif`,
/// `photo.jgw` beside `photo.jpg`, or the generic `.wld`.
fn world_file(path: &Path) -> Option<PathBuf> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    // The convention: first and last letter of the extension, then `w`; also
    // the whole extension plus `w`, and `.wld` for anything.
    let mut candidates = vec![format!("{ext}w"), "wld".to_string()];
    if ext.len() >= 2 {
        let short: String = [ext.chars().next()?, ext.chars().last()?, 'w'].iter().collect();
        candidates.insert(0, short);
    }
    candidates.into_iter().map(|e| path.with_extension(e)).find(|p| p.is_file())
}

/// GDAL writes a placement it was given into `map.tif.aux.xml`; the file also
/// holds statistics and the like, so only a transform or control points count.
fn gdal_aux(path: &Path) -> bool {
    let aux = sidecar(path, ".aux.xml");
    std::fs::read_to_string(aux)
        .is_ok_and(|xml| xml.contains("<GeoTransform>") || xml.contains("<GCPList"))
}

/// histomap's sidecar always exists once the map has been opened there; it is
/// a placement only once `placement` is an object rather than null.
fn histomap(path: &Path) -> bool {
    let Ok(json) = std::fs::read_to_string(sidecar(path, ".histomap.json")) else {
        return false;
    };
    // A look for `"placement": {` rather than a JSON parser: this is the one
    // thing wanted from the file, and whitespace is all that can vary.
    let Some(at) = json.find("\"placement\"") else { return false };
    json[at + "\"placement\"".len()..]
        .trim_start()
        .strip_prefix(':')
        .is_some_and(|rest| rest.trim_start().starts_with('{'))
}

/// `name.ext` + `suffix`, keeping the image's own extension.
fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("show-geo-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_histomap_placement_counts_and_an_empty_one_does_not() {
        let dir = scratch("histomap");
        let (placed, empty) = (dir.join("a.tif"), dir.join("b.tif"));
        std::fs::write(sidecar(&placed, ".histomap.json"), r#"{"pins": [], "placement": {"center": [1, 2]}}"#)
            .unwrap();
        std::fs::write(sidecar(&empty, ".histomap.json"), r#"{"pins": [], "placement": null}"#).unwrap();
        assert_eq!(detect(&placed), [Source::Histomap]);
        assert!(detect(&empty).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn world_files_are_found_by_every_naming_convention() {
        let dir = scratch("world");
        for (image, world) in [("a.tif", "a.tfw"), ("b.jpg", "b.jpgw"), ("c.png", "c.wld")] {
            std::fs::write(dir.join(world), "1\n0\n0\n-1\n0\n0\n").unwrap();
            assert_eq!(detect(&dir.join(image)), [Source::WorldFile], "{image}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

