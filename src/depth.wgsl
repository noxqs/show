// How far away things are, as colour over the picture.
//
// The input is one byte per cell, 255 nearest. Filtered, unlike the mask: a
// distance between two distances is a distance, where an id between two ids
// is nonsense. Drawn with the video quad's rect, as the mask is, so it pans
// and zooms with the picture.

struct Uniforms {
    // Clip-space rectangle: x0, y0, x1, y1.
    rect: vec4<f32>,
    // Opacity, then quarter turns clockwise, as the picture beneath.
    style: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var depth: texture_2d<f32>;
@group(0) @binding(2) var filtered: sampler;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    let corner = vec2<f32>(f32(index & 1u), f32((index >> 1u) & 1u));
    var out: VsOut;
    out.position = vec4<f32>(
        mix(u.rect.x, u.rect.z, corner.x),
        mix(u.rect.w, u.rect.y, corner.y),
        0.0,
        1.0,
    );
    out.uv = unturn(corner, u.style.y);
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

// Turbo, Google's rainbow without the false edges of jet: perceptually
// smoother, and dark at both ends so near and far both read against it.
// Polynomial fit by Ruofei Du, public domain.
fn turbo(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let v4 = vec4<f32>(1.0, x, x * x, x * x * x);
    let v2 = v4.zw * v4.z;
    let r = dot(v4, vec4<f32>(0.13572138, 4.61539260, -42.66032258, 132.13108234))
        + dot(v2, vec2<f32>(-152.94239396, 59.28637943));
    let g = dot(v4, vec4<f32>(0.09140261, 2.19418839, 4.84296658, -14.18503333))
        + dot(v2, vec2<f32>(4.27729857, 2.82956604));
    let b = dot(v4, vec4<f32>(0.10667330, 12.64194608, -60.58204836, 110.36276771))
        + dot(v2, vec2<f32>(-89.90310912, 27.34824973));
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let near = textureSample(depth, filtered, in.uv).r;
    return vec4<f32>(turbo(near), u.style.x);
}
