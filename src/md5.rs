//! MD5, purely to name files in the desktop's thumbnail cache.
//!
//! The XDG thumbnail specification names each file by the MD5 of the source's
//! `file://` URI, so reading the cache the file manager has already filled
//! means computing exactly that digest. Nothing here is security-relevant —
//! MD5 is thoroughly broken for that and is used here only as a fixed naming
//! scheme defined by someone else.

/// Lowercase hex digest of `data`.
pub fn hex(data: &[u8]) -> String {
    let d = digest(data);
    let mut out = String::with_capacity(32);
    for byte in d {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn digest(data: &[u8]) -> [u8; 16] {
    // Per-round left-rotation amounts.
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    // K[i] = floor(2^32 * abs(sin(i + 1))).
    let k: [u32; 64] = std::array::from_fn(|i| {
        ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32
    });

    let mut state: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];

    // Message, a 0x80 byte, zero padding to 56 mod 64, then the bit length.
    let mut message = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_le_bytes());

    for chunk in message.chunks_exact(64) {
        let m: [u32; 16] = std::array::from_fn(|i| {
            u32::from_le_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]])
        });

        let [mut a, mut b, mut c, mut d] = state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let tmp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]).rotate_left(S[i]),
            );
            a = tmp;
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
    }

    let mut out = [0u8; 16];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_published_test_vectors() {
        // From RFC 1321.
        assert_eq!(hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(b"a"), "0cc175b9c0f1b6a831c399e269772661");
        assert_eq!(hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(hex(b"message digest"), "f96b697d7cb7938d525a2f31aaf161d0");
        assert_eq!(
            hex(b"abcdefghijklmnopqrstuvwxyz"),
            "c3fcd3d76192e4007dfb496cca67e13b"
        );
        assert_eq!(
            hex(b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    #[test]
    fn matches_a_real_thumbnail_name() {
        // Taken from this machine's own thumbnail cache, which is the only
        // thing that proves the naming scheme is implemented correctly.
        let uri = "file:///home/morgan/proj/claude_research/girl_ref/reject_dups/girl_ref_066_4x.png";
        assert_eq!(hex(uri.as_bytes()), "0000b357fea98c981e6137cb0ee18d2a");
    }

    #[test]
    fn handles_input_spanning_the_block_boundary() {
        // 55, 56 and 64 bytes exercise every padding case.
        for len in [55usize, 56, 63, 64, 65] {
            let data = vec![b'x'; len];
            assert_eq!(hex(&data).len(), 32, "length {len}");
        }
        assert_eq!(hex(&vec![b'a'; 64]), "014842d480b571495a4a0363793f7367");
    }
}
