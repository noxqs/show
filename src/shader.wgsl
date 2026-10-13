// One textured quad. The vertex shader places it in NDC from a rect the CPU
// computes each frame, so zoom and pan cost a 16-byte uniform write and
// nothing else -- no vertex buffer, no geometry, no per-pixel transform.

struct Uniforms {
    // xy = bottom-left in NDC, zw = top-right.
    rect: vec4<f32>,
    // The part of the texture to show: xy = top-left, zw = bottom-right, in
    // texture coordinates. The whole of it, except when a coarser tile stands
    // in for a missing finer one and only its corner covers the same ground.
    uv: vec4<f32>,
    // x: quarter turns clockwise. Zero for everything but the picture.
    turn: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    // Triangle strip corners: (0,0) (1,0) (0,1) (1,1).
    let corner = vec2<f32>(f32(i & 1u), f32((i >> 1u) & 1u));

    var out: VsOut;
    out.pos = vec4<f32>(mix(u.rect.xy, u.rect.zw, corner), 0.0, 1.0);
    // NDC y points up, texture v points down.
    out.uv = mix(u.uv.xy, u.uv.zw, unturn(vec2<f32>(corner.x, 1.0 - corner.y), u.turn.x));
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
    return textureSample(tex, samp, in.uv);
}
