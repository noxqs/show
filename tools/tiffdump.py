#!/usr/bin/env python3
"""Minimal TIFF/BigTIFF IFD walker.

Prints the geometry of every page: pyramid levels, tile grid, compression.
ImageMagick refuses to open gigapixel files at all and vipsheader drowns the
answer in GeoTIFF warnings, so we read the directory structure ourselves. Only
the header is touched -- no pixel data is decoded.
"""
import struct
import sys

TAGS = {
    256: "ImageWidth", 257: "ImageLength", 258: "BitsPerSample",
    259: "Compression", 262: "PhotometricInterp", 277: "SamplesPerPixel",
    278: "RowsPerStrip", 284: "PlanarConfig", 322: "TileWidth",
    323: "TileLength", 324: "TileOffsets", 325: "TileByteCounts",
    273: "StripOffsets", 279: "StripByteCounts", 254: "SubfileType",
    317: "Predictor", 347: "JPEGTables",
}
COMPRESSION = {
    1: "none", 2: "ccitt-rle", 3: "ccitt-g3", 4: "ccitt-g4", 5: "lzw",
    6: "jpeg(old)", 7: "jpeg", 8: "deflate", 32773: "packbits",
    32946: "deflate(old)", 34712: "jpeg2000", 50000: "zstd", 50001: "webp",
}
PHOTOMETRIC = {0: "white-is-zero", 1: "black-is-zero", 2: "rgb", 3: "palette",
               4: "mask", 5: "cmyk", 6: "ycbcr", 8: "cielab"}
# TIFF type -> (struct code, byte size)
TYPES = {1: ("B", 1), 2: ("c", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8),
         6: ("b", 1), 7: ("B", 1), 8: ("h", 2), 9: ("i", 4), 10: ("ii", 8),
         11: ("f", 4), 12: ("d", 8), 16: ("Q", 8), 17: ("q", 8), 18: ("Q", 8)}


def walk(path):
    f = open(path, "rb")
    head = f.read(16)
    endian = "<" if head[:2] == b"II" else ">"
    magic = struct.unpack(endian + "H", head[2:4])[0]

    if magic == 42:
        big = False
        off_fmt, off_size = endian + "I", 4
        next_off = struct.unpack(endian + "I", head[4:8])[0]
        entry_size, count_size = 12, 2
    elif magic == 43:
        big = True
        off_fmt, off_size = endian + "Q", 8
        next_off = struct.unpack(endian + "Q", head[8:16])[0]
        entry_size, count_size = 20, 8
    else:
        raise SystemExit(f"{path}: not a TIFF (magic {magic})")

    print(f"{path}")
    print(f"  {'BigTIFF' if big else 'classic TIFF'}, {'little' if endian == '<' else 'big'}-endian")

    page = 0
    while next_off and page < 64:
        f.seek(next_off)
        n = struct.unpack(endian + ("Q" if big else "H"), f.read(count_size))[0]
        entries = f.read(n * entry_size)
        tags = {}
        for i in range(n):
            e = entries[i * entry_size : (i + 1) * entry_size]
            tag, typ = struct.unpack(endian + "HH", e[:4])
            count = struct.unpack(endian + ("Q" if big else "I"), e[4 : 4 + off_size])[0]
            valbytes = e[4 + off_size :]
            if typ not in TYPES:
                continue
            code, tsize = TYPES[typ]
            total = tsize * count
            if total > len(valbytes):
                ptr = struct.unpack(off_fmt, valbytes[:off_size])[0]
                keep = f.tell()
                f.seek(ptr)
                raw = f.read(min(total, 4096))
                f.seek(keep)
            else:
                raw = valbytes[:total]
            if code in ("II", "ii"):
                tags[tag] = ("rational", count)
            else:
                fitting = len(raw) // tsize
                try:
                    tags[tag] = struct.unpack(endian + code * fitting, raw[: fitting * tsize])
                except struct.error:
                    tags[tag] = ()

        def one(tag, default=None):
            v = tags.get(tag)
            return v[0] if v else default

        w, h = one(256), one(257)
        tw, th = one(322), one(323)
        comp = one(259, 1)
        spp = one(277, 1)
        bps = one(258, 8)
        photo = one(262)
        ntiles = len(tags.get(324, ())) or len(tags.get(273, ()))
        layout = f"tiled {tw}x{th}" if tw else f"stripped rows={one(278)}"
        grid = ""
        if tw and w:
            grid = f" grid={-(-w // tw)}x{-(-h // th)}"
        mp = w * h / 1e6 if w and h else 0
        print(
            f"  page {page}: {w}x{h} ({mp:8.1f} MPix)  {layout}{grid}"
            f"  comp={COMPRESSION.get(comp, comp)}  {spp}x{bps}bit"
            f"  {PHOTOMETRIC.get(photo, photo)}  chunks={ntiles}"
        )
        page += 1
        f.seek(next_off + count_size + n * entry_size)
        next_off = struct.unpack(off_fmt, f.read(off_size))[0]
    f.close()


for p in sys.argv[1:]:
    try:
        walk(p)
    except Exception as e:
        print(f"{p}: {type(e).__name__}: {e}")
    print()
