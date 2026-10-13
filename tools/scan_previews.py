#!/usr/bin/env python3
"""Report every embedded preview image a file carries.

An embedded preview is the only thing that makes a multi-second decode feel
instant, but only if one is actually there and big enough to be worth showing.
Cameras, phones and stitching tools all disagree about that, so ask the files.

Covers JPEG (EXIF IFD1 thumbnail, MPF full-size previews) and TIFF/DNG
(reduced-resolution SubIFDs).
"""
import struct
import sys


def u16(b, o, e):
    return struct.unpack(e + "H", b[o : o + 2])[0]


def u32(b, o, e):
    return struct.unpack(e + "I", b[o : o + 4])[0]


def read_ifd(buf, base, off, e):
    """Yield (tag, type, count, value_offset) for one IFD, plus the next offset."""
    n = u16(buf, base + off, e)
    entries = []
    for i in range(n):
        p = base + off + 2 + i * 12
        tag, typ, count = u16(buf, p, e), u16(buf, p + 2, e), u32(buf, p + 4, e)
        entries.append((tag, typ, count, u32(buf, p + 8, e)))
    nxt = u32(buf, base + off + 2 + n * 12, e)
    return entries, nxt


def jpeg_dims(buf):
    i = 2
    while i < len(buf) - 9:
        if buf[i] != 0xFF:
            i += 1
            continue
        m = buf[i + 1]
        if m in (0xC0, 0xC1, 0xC2):
            return u16(buf, i + 7, ">"), u16(buf, i + 5, ">")
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack(">H", buf[i + 2 : i + 4])[0]
    return None


def scan_jpeg(data, name):
    found = []
    i = 2
    while i < len(data) - 4:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xDA:  # start of scan; metadata is all behind us
            break
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        seglen = struct.unpack(">H", data[i + 2 : i + 4])[0]
        seg = data[i + 4 : i + 2 + seglen]

        if m == 0xE1 and seg.startswith(b"Exif\0\0"):
            tiff = seg[6:]
            e = "<" if tiff[:2] == b"II" else ">"
            try:
                entries, nxt = read_ifd(tiff, 0, u32(tiff, 4, e), e)
                if nxt:  # IFD1 holds the thumbnail
                    thumb, _ = read_ifd(tiff, 0, nxt, e)
                    off = length = None
                    for tag, _t, _c, v in thumb:
                        if tag == 0x0201:
                            off = v
                        elif tag == 0x0202:
                            length = v
                    if off and length:
                        blob = tiff[off : off + length]
                        found.append(("EXIF IFD1 thumbnail", jpeg_dims(blob), length))
            except Exception as ex:
                found.append((f"EXIF parse failed: {ex}", None, 0))

        if m == 0xE2 and seg.startswith(b"MPF\0"):
            mpf = seg[4:]
            e = "<" if mpf[:2] == b"II" else ">"
            try:
                entries, _ = read_ifd(mpf, 0, u32(mpf, 4, e), e)
                for tag, _t, count, v in entries:
                    if tag == 0xB002:  # MP Entry: 16 bytes per image
                        for k in range(count // 16):
                            p = v + k * 16
                            size = u32(mpf, p + 4, e)
                            offset = u32(mpf, p + 8, e)
                            if offset:  # 0 means "this is the primary image"
                                found.append((f"MPF image {k}", None, size))
            except Exception as ex:
                found.append((f"MPF parse failed: {ex}", None, 0))
        i += 2 + seglen
    return found


def scan_tiff(data, name):
    e = "<" if data[:2] == b"II" else ">"
    magic = u16(data, 2, e)
    if magic != 42:
        return [("BigTIFF: not scanned by this tool", None, 0)]
    found = []
    off = u32(data, 4, e)
    page = 0
    while off and page < 8:
        entries, off = read_ifd(data, 0, off, e)
        tags = {t: (typ, c, v) for t, typ, c, v in entries}
        w = tags.get(256, (0, 0, 0))[2]
        h = tags.get(257, (0, 0, 0))[2]
        sub = tags.get(254, (0, 0, 0))[2]
        comp = tags.get(259, (0, 0, 1))[2]
        kind = "reduced-resolution" if sub & 1 else "full"
        found.append((f"IFD{page} {kind} comp={comp}", (w, h), 0))
        page += 1
    # SubIFDs (tag 330) are where DNG usually hides the real image and previews
    return found


for path in sys.argv[1:]:
    data = open(path, "rb").read()
    name = path.split("/")[-1]
    print(f"\n{name}  ({len(data)/1e6:.1f} MB)")
    if data[:3] == b"\xff\xd8\xff":
        main = jpeg_dims(data)
        print(f"  primary image: {main[0]}x{main[1]}" if main else "  primary: ?")
        results = scan_jpeg(data, name)
    elif data[:2] in (b"II", b"MM"):
        results = scan_tiff(data, name)
    else:
        results = [("not a JPEG or TIFF container", None, 0)]
    if not results:
        print("  no embedded preview found")
    for label, dims, size in results:
        d = f"{dims[0]}x{dims[1]}" if dims else "?"
        print(f"  {label}: {d}  ({size/1024:.0f} KB)" if size else f"  {label}: {d}")
