#!/usr/bin/env bash
# Build a test corpus under testdata/.
#
# Two halves:
#   sources/  real photographs at a spread of resolutions, downloaded once
#   corpus/   one base image transcoded into every format we claim to support,
#             plus the awkward variants that trip viewers up (progressive JPEG,
#             CMYK, 16-bit, interlaced PNG, palette, alpha, grayscale)
#
# Re-running is cheap: downloads are skipped if the file is already there.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/testdata/sources"
OUT="$ROOT/testdata/corpus"
mkdir -p "$SRC" "$OUT"

UA='show-image-viewer/0.1 (test corpus fetcher)'
CONVERT="${CONVERT:-convert}"

have() { command -v "$1" >/dev/null 2>&1; }

fetch() { # fetch <url> <dest>
    local url="$1" dest="$2"
    if [[ -s "$dest" ]]; then
        echo "  have $(basename "$dest")"
        return 0
    fi
    if curl -fsSL --max-time 120 -A "$UA" -o "$dest.part" "$url"; then
        mv "$dest.part" "$dest"
        echo "  got  $(basename "$dest") ($(stat -c%s "$dest") bytes)"
    else
        rm -f "$dest.part"
        echo "  FAIL $(basename "$dest") <- $url" >&2
    fi
}

echo "==> sources"
# Lorem Picsum serves real photos cropped to any size, so we get a clean
# resolution sweep from the same pipeline. Seeded URLs keep runs reproducible.
fetch "https://picsum.photos/seed/small/640/480"    "$SRC/photo-640x480.jpg"
fetch "https://picsum.photos/seed/hd/1920/1080"     "$SRC/photo-1920x1080.jpg"
fetch "https://picsum.photos/seed/four/4000/3000"   "$SRC/photo-4000x3000.jpg"
fetch "https://picsum.photos/seed/huge/8000/6000"   "$SRC/photo-8000x6000.jpg"

# The Kodak suite: the reference set for lossless image work, uncompressed PNG.
for n in 05 13 19 23; do
    fetch "https://r0k.us/graphics/kodak/kodak/kodim$n.png" "$SRC/kodim$n.png"
done

BASE="$SRC/photo-1920x1080.jpg"
if [[ ! -s "$BASE" ]]; then
    echo "no base image; cannot transcode" >&2
    exit 1
fi

if ! have "$CONVERT"; then
    echo "ImageMagick not found; skipping transcode step" >&2
    exit 0
fi

echo "==> formats (from $(basename "$BASE"))"
tc() { # tc <name> <convert args...>
    local name="$1"; shift
    if [[ -s "$OUT/$name" ]]; then echo "  have $name"; return; fi
    if "$CONVERT" "$BASE" "$@" "$OUT/$name" 2>/dev/null && [[ -s "$OUT/$name" ]]; then
        echo "  made $name ($(stat -c%s "$OUT/$name") bytes)"
    else
        rm -f "$OUT/$name"
        echo "  skip $name (encoder unavailable)"
    fi
}

tc base.png
tc base.jpg   -quality 85
tc base.webp  -quality 85
tc base.bmp
tc base.tif   -compress lzw
tc base.tga
tc base.ppm
tc base.pgm   -colorspace Gray
tc base.gif
tc base.ico   -resize 256x256
tc base.dds
tc base.hdr
tc base.exr
tc base.qoi
tc base.ff

echo "==> awkward variants"
tc progressive.jpg   -quality 85 -interlace Plane
tc cmyk.jpg          -quality 85 -colorspace CMYK
tc gray.jpg          -quality 85 -colorspace Gray
tc interlaced.png    -interlace PNG
tc palette.png       -colors 256 -type Palette
tc gray16.png        -colorspace Gray -depth 16
tc rgb16.png         -depth 16

# Alpha needs a real transparent region, not just an alpha channel of 0xFF.
if [[ ! -s "$OUT/alpha.png" ]]; then
    "$CONVERT" "$BASE" -alpha set -channel A -evaluate multiply 0.5 +channel \
        "$OUT/alpha.png" 2>/dev/null && echo "  made alpha.png" || echo "  skip alpha.png"
fi
tc alpha.webp -alpha set -channel A -evaluate multiply 0.5 +channel

# Tiny and degenerate sizes: the classic off-by-one bait for a renderer.
tc tiny-1x1.png     -resize 1x1\!
tc thin-1x2000.png  -resize 1x2000\!
tc wide-2000x1.png  -resize 2000x1\!

echo "==> jpeg xl"
if have cjxl && [[ ! -s "$OUT/base.jxl" ]]; then
    cjxl "$BASE" "$OUT/base.jxl" -q 85 >/dev/null 2>&1 \
        && echo "  made base.jxl" || echo "  skip base.jxl"
else
    echo "  have/skip base.jxl"
fi

echo "==> avif / heif"
if have ffmpeg && [[ ! -s "$OUT/base.avif" ]]; then
    ffmpeg -y -loglevel error -i "$BASE" -c:v libaom-av1 -still-picture 1 "$OUT/base.avif" \
        >/dev/null 2>&1 && echo "  made base.avif" || { rm -f "$OUT/base.avif"; echo "  skip base.avif"; }
else
    echo "  have/skip base.avif"
fi

echo "==> large-format stress (from $(basename "$SRC/photo-8000x6000.jpg"))"
if [[ -s "$SRC/photo-8000x6000.jpg" ]]; then
    [[ -s "$OUT/huge.png" ]] || { "$CONVERT" "$SRC/photo-8000x6000.jpg" "$OUT/huge.png" 2>/dev/null && echo "  made huge.png ($(stat -c%s "$OUT/huge.png") bytes)"; }
    [[ -s "$OUT/huge.jpg" ]] || cp "$SRC/photo-8000x6000.jpg" "$OUT/huge.jpg"
fi

echo
echo "corpus: $(find "$OUT" -type f | wc -l) files, $(du -sh "$OUT" | cut -f1)"
