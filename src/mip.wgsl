// One mip level from the level above it. Drawn as a single triangle covering
// the target; a bilinear sample at each target pixel's centre lands on the
// corner shared by the four source pixels it replaces, so it averages them.
//
// The texture is sRGB: sampling decodes to linear light and the write encodes
// again, so the average is of light rather than of gamma-encoded values, and
// fine bright detail does not darken as the image shrinks.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    // (0,0) (2,0) (0,2): one triangle that covers the whole target.
    let corner = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VsOut;
    out.pos = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src, samp, in.uv);
}
