// Planar YUV drawn directly, without converting to RGB on the CPU first.
//
// A 4:2:0 frame is 1.5 bytes per pixel; the RGBA it would become is 4. Sending
// the planes as they are and converting here saves both the swscale pass and
// three-fifths of the upload, and the conversion itself is free — the GPU is
// doing a matrix multiply per pixel it was already going to shade.

struct Uniforms {
    // xy = bottom-left in NDC, zw = top-right.
    rect: vec4<f32>,
    // Rows of the YUV to RGB matrix. w is unused.
    row_r: vec4<f32>,
    row_g: vec4<f32>,
    row_b: vec4<f32>,
    // x subtracted from luma, y from both chroma planes, before the matrix.
    offsets: vec4<f32>,
    // x: quarter turns clockwise.
    turn: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var plane_y: texture_2d<f32>;
@group(0) @binding(2) var plane_u: texture_2d<f32>;
@group(0) @binding(3) var plane_v: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let corner = vec2<f32>(f32(i & 1u), f32((i >> 1u) & 1u));
    var out: VsOut;
    out.pos = vec4<f32>(mix(u.rect.xy, u.rect.zw, corner), 0.0, 1.0);
    out.uv = unturn(vec2<f32>(corner.x, 1.0 - corner.y), u.turn.x);
    return out;
}

// Which point of the texture belongs at `d`, a point of the quad as shown
// (0..1, top-left origin), when the picture is turned `turn` quarter turns
// clockwise. Turning is only ever a matter of how it is shown.
fn unturn(d: vec2<f32>, turn: f32) -> vec2<f32> {
    let q = u32(turn + 0.5) % 4u;
    if q == 1u { return vec2<f32>(d.y, 1.0 - d.x); }
    if q == 2u { return vec2<f32>(1.0 - d.x, 1.0 - d.y); }
    if q == 3u { return vec2<f32>(1.0 - d.y, d.x); }
    return d;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // The chroma planes are half size in 4:2:0; sampling them with the same
    // coordinates and a linear filter is the upsample.
    let y = textureSample(plane_y, samp, in.uv).r - u.offsets.x;
    let cb = textureSample(plane_u, samp, in.uv).r - u.offsets.y;
    let cr = textureSample(plane_v, samp, in.uv).r - u.offsets.y;

    let ycc = vec3<f32>(y, cb, cr);
    let rgb = vec3<f32>(
        dot(u.row_r.xyz, ycc),
        dot(u.row_g.xyz, ycc),
        dot(u.row_b.xyz, ycc),
    );
    let clamped = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

    // The matrix above produces gamma-encoded RGB, because that is what the
    // samples in the file are. The surface is sRGB, so the hardware will
    // encode whatever we return on the way out — encoding it a second time,
    // which lifts the midtones and washes the picture out. Undoing the
    // encoding here means the hardware's own encoding lands back on the
    // values the file actually holds.
    //
    // The image path needs none of this: its texture is Rgba8UnormSrgb, so
    // sampling decodes and the surface re-encodes, and the two cancel. The
    // luma and chroma planes are R8Unorm and cannot do that, since only the
    // matrix above knows they are colour at all.
    return vec4<f32>(srgb_to_linear(clamped), 1.0);
}

/// The sRGB electro-optical transfer function.
///
/// Video is BT.709 rather than sRGB, and the two transfer curves differ
/// slightly at the dark end. sRGB is the right inverse to use regardless:
/// what is being undone here is the surface's own sRGB encoding, not the
/// camera's.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = c <= vec3<f32>(0.04045);
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, cutoff);
}
