// Drawing what a model found, over the picture it found it in.
//
// The input is a label texture: one byte per pixel holding an instance id,
// zero meaning nothing. That is 130 KB for a 480x270 grid against 8.3 MB for a
// screen-sized RGBA overlay, and it is uploaded once per result rather than
// once per frame — which is the whole reason this is a pipeline rather than
// another entry in the overlay stack.
//
// Drawn with the same rect the video quad uses, so it letterboxes, pans and
// zooms with the picture without knowing anything about any of that.

struct Uniforms {
    // Clip-space rectangle: x0, y0, x1, y1.
    rect: vec4<f32>,
    // Outline thickness in label pixels, fill opacity, and the label texture's
    // size, which the outline test needs to step by one texel.
    style: vec4<f32>,
    // One colour per instance id. Index zero is never read; a label of zero is
    // background and returns before the lookup.
    palette: array<vec4<f32>, 16>,
    // x: quarter turns clockwise, as the picture beneath.
    turn: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var labels: texture_2d<u32>;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    // The same four-vertex strip the other pipelines use.
    let corner = vec2<f32>(f32(index & 1u), f32((index >> 1u) & 1u));
    var out: VsOut;
    out.position = vec4<f32>(
        mix(u.rect.x, u.rect.z, corner.x),
        mix(u.rect.w, u.rect.y, corner.y),
        0.0,
        1.0,
    );
    out.uv = unturn(corner, u.turn.x);
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

/// The label at a texel offset from here, or zero off the edge.
fn label_at(base: vec2<i32>, offset: vec2<i32>, size: vec2<i32>) -> u32 {
    let at = base + offset;
    if at.x < 0 || at.y < 0 || at.x >= size.x || at.y >= size.y {
        return 0u;
    }
    return textureLoad(labels, at, 0).r;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let size = vec2<i32>(i32(u.style.z), i32(u.style.w));
    // Nearest by construction: these are ids, and interpolating between two
    // of them would produce a third that means something else entirely.
    let at = vec2<i32>(in.uv * vec2<f32>(size));
    let here = label_at(at, vec2<i32>(0, 0), size);

    if here == 0u {
        discard;
    }

    let colour = u.palette[here % 16u];

    // An edge is a texel whose neighbours are not all the same as it. Checking
    // the four cardinal directions is enough for shapes this size, and cheaper
    // than eight.
    let thickness = i32(u.style.x);
    var edge = false;
    for (var d = 1; d <= thickness; d = d + 1) {
        if label_at(at, vec2<i32>( d,  0), size) != here { edge = true; }
        if label_at(at, vec2<i32>(-d,  0), size) != here { edge = true; }
        if label_at(at, vec2<i32>( 0,  d), size) != here { edge = true; }
        if label_at(at, vec2<i32>( 0, -d), size) != here { edge = true; }
    }

    // The outline is drawn solid and the inside is left mostly alone: a film
    // is still being watched, and filling every found thing with colour hides
    // more of it than it explains.
    if edge {
        return vec4<f32>(colour.rgb, 1.0);
    }
    return vec4<f32>(colour.rgb, u.style.y);
}
