#!/usr/bin/env python3
"""Encode QOI and farbfeld from a binary PPM (P6).

ImageMagick 6 knows neither format and, worse, silently writes a JPEG under the
requested extension instead of failing. Both formats are simple enough to emit
directly, so we do.
"""
import struct
import sys


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    # Header: P6 <w> <h> <maxval>, whitespace-separated, '#' comments allowed.
    fields, pos = [], 2
    while len(fields) < 3:
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":
            while data[pos : pos + 1] not in (b"\n", b""):
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        fields.append(int(data[start:pos]))
    pos += 1  # a single whitespace byte terminates the header
    w, h, maxval = fields
    if maxval != 255:
        raise SystemExit(f"{path}: only 8-bit PPM supported (maxval={maxval})")
    return w, h, data[pos : pos + w * h * 3]


def write_farbfeld(path, w, h, rgb):
    out = bytearray(b"farbfeld" + struct.pack(">II", w, h))
    # farbfeld is 16-bit RGBA big-endian; 8-bit values scale by 0x0101.
    for i in range(0, len(rgb), 3):
        out += struct.pack(">HHHH", rgb[i] * 257, rgb[i + 1] * 257, rgb[i + 2] * 257, 65535)
    open(path, "wb").write(bytes(out))


QOI_OP_INDEX, QOI_OP_DIFF, QOI_OP_LUMA, QOI_OP_RUN = 0x00, 0x40, 0x80, 0xC0
QOI_OP_RGB, QOI_OP_RGBA = 0xFE, 0xFF


def write_qoi(path, w, h, rgb):
    out = bytearray(b"qoif" + struct.pack(">IIBB", w, h, 4, 0))
    index = [(0, 0, 0, 0)] * 64
    pr, pg, pb, pa = 0, 0, 0, 255
    run = 0
    npix = w * h
    for i in range(npix):
        r, g, b, a = rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 255
        if (r, g, b, a) == (pr, pg, pb, pa):
            run += 1
            if run == 62 or i == npix - 1:
                out.append(QOI_OP_RUN | (run - 1))
                run = 0
        else:
            if run:
                out.append(QOI_OP_RUN | (run - 1))
                run = 0
            hashpos = (r * 3 + g * 5 + b * 7 + a * 11) % 64
            if index[hashpos] == (r, g, b, a):
                out.append(QOI_OP_INDEX | hashpos)
            else:
                index[hashpos] = (r, g, b, a)
                dr, dg, db = r - pr, g - pg, b - pb
                dg_r, dg_b = dr - dg, db - dg
                if a != pa:
                    out += bytes((QOI_OP_RGBA, r, g, b, a))
                elif -2 <= dr <= 1 and -2 <= dg <= 1 and -2 <= db <= 1:
                    out.append(QOI_OP_DIFF | (dr + 2) << 4 | (dg + 2) << 2 | (db + 2))
                elif -32 <= dg <= 31 and -8 <= dg_r <= 7 and -8 <= dg_b <= 7:
                    out += bytes((QOI_OP_LUMA | (dg + 32), (dg_r + 8) << 4 | (dg_b + 8)))
                else:
                    out += bytes((QOI_OP_RGB, r, g, b))
        pr, pg, pb, pa = r, g, b, a
    out += bytes(7) + b"\x01"
    open(path, "wb").write(bytes(out))


if __name__ == "__main__":
    ppm, qoi_out, ff_out = sys.argv[1:4]
    w, h, rgb = read_ppm(ppm)
    write_qoi(qoi_out, w, h, rgb)
    write_farbfeld(ff_out, w, h, rgb)
    print(f"  made {qoi_out} + {ff_out} ({w}x{h})")
