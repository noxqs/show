//! wgpu setup and rendering: one texture, one quad, one uniform.
//!
//! Displaying an image is the easiest thing a GPU will ever be asked to do, so
//! the interesting question is not "can it draw this" but "what does it cost to
//! get the pixels there". Every stage of that -- instance, adapter, device,
//! texture allocation, the upload itself -- is timed separately, because on a
//! cold start those dwarf the draw.
//!
//! [`Core`] and [`Renderer`] deliberately know nothing about windows, so the
//! headless profiler can measure exactly the same code the viewer runs.

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

use crate::timing;

/// What the CPU hands the GPU each frame.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    rect: [f32; 4],
    uv: [f32; 4],
    /// Quarter turns clockwise, then nothing.
    turn: [f32; 4],
}

/// The whole texture.
pub const FULL_UV: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Card memory the tiles of one image may hold. Far more than a screen needs —
/// a 4K screen of 256-pixel tiles is under 40 MB — so zooming back to a place
/// already seen is instant, while a long session over a gigapixel map still
/// cannot fill the card.
const TILE_BUDGET: u64 = 512 * 1024 * 1024;

/// Build an instance over just the requested backends.
///
/// This matters more than it looks: `Backends::all()` makes wgpu bring up
/// Vulkan *and* OpenGL (and probe for more), and on a cold start that costs
/// tens of milliseconds -- more than decoding the image. Naming the one backend
/// we intend to use skips all of it.
pub fn new_instance(backends: wgpu::Backends) -> wgpu::Instance {
    let _s = timing::span("gpu", "instance");
    wgpu::Instance::new(&wgpu::InstanceDescriptor { backends, ..Default::default() })
}

/// One graphics device wgpu can reach.
#[derive(Clone, Debug)]
pub struct AdapterEntry {
    pub name: String,
    /// The `--backends` value that reaches this adapter.
    pub backend: &'static str,
    /// "DiscreteGpu", "IntegratedGpu", "Cpu", ...
    pub device_type: String,
    pub driver: String,
}

impl AdapterEntry {
    /// Short label for comparison tables.
    pub fn label(&self) -> String {
        let kind = match self.device_type.as_str() {
            "DiscreteGpu" => "discrete",
            "IntegratedGpu" => "integrated",
            "Cpu" => "software",
            other => other,
        };
        // Trim the marketing words that make every row the same width.
        let mut name = self
            .name
            .replace("NVIDIA GeForce ", "")
            .replace("AMD Radeon Graphics (RADV ", "Radeon RADV/")
            .replace("llvmpipe (", "llvmpipe (");
        if let Some(cut) = name.find(" (LLVM") {
            name.truncate(cut);
        }
        // The vendor strings we rewrote above leave an unbalanced paren behind.
        if name.ends_with(')') && !name.contains('(') {
            name.pop();
        }
        format!("{name} [{kind}, {}]", self.backend)
    }
}

/// Every adapter wgpu can see. Worth knowing on a hybrid machine: the discrete
/// card has the throughput, but the integrated one shares system memory and so
/// may win on upload latency -- and the software rasteriser is the floor.
pub fn list_adapters() -> Vec<AdapterEntry> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .iter()
        .map(|a| {
            let i = a.get_info();
            AdapterEntry {
                name: i.name.clone(),
                backend: match i.backend {
                    wgpu::Backend::Vulkan => "vulkan",
                    wgpu::Backend::Gl => "gl",
                    _ => "all",
                },
                device_type: format!("{:?}", i.device_type),
                driver: i.driver_info.clone(),
            }
        })
        .collect()
}

pub fn describe_adapters() -> Vec<String> {
    list_adapters()
        .iter()
        .map(|a| format!("{} {} [{}, driver: {}]", a.device_type, a.name, a.backend, a.driver))
        .collect()
}

/// Instance, adapter, device and queue — everything that exists before any
/// particular image or window does.
pub struct Core {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub max_texture: u32,
    pub adapter_name: String,
    /// Built the first time an image wants mipmaps, not at startup.
    mips: std::sync::Arc<std::sync::OnceLock<MipGen>>,
}

impl Core {
    /// `surface` constrains adapter selection to one that can actually present
    /// to that window; headless callers pass `None`.
    pub fn new(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'static>>,
        filter: Option<&str>,
    ) -> Result<Core> {
        let adapter = {
            let mut s = timing::span("gpu", "adapter");
            let adapter = pick_adapter(&instance, surface, filter)?;
            s.note(adapter.get_info().name.clone());
            adapter
        };

        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend);

        // Ask for the largest textures the hardware will give us rather than
        // wgpu's conservative 8192 default: a 20k-pixel-wide panorama is an
        // ordinary file and fits comfortably on real hardware.
        let limits = adapter.limits();
        let max_texture = limits.max_texture_dimension_2d;

        let (device, queue) = {
            let mut s = timing::span("gpu", "device");
            s.note(format!("max_texture={max_texture}"));
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("show"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
            .context("requesting device")?
        };

        Ok(Core { instance, adapter, device, queue, max_texture, adapter_name, mips: Default::default() })
    }

    /// Try `backends`, and if nothing usable turns up, try everything.
    ///
    /// Restricting to one backend is a large startup win, but it must never be
    /// the reason the viewer refuses to open a file on a machine without it.
    pub fn new_default(filter: Option<&str>, backends: wgpu::Backends) -> Result<Core> {
        match Core::new(new_instance(backends), None, filter) {
            Ok(core) => Ok(core),
            Err(first) if backends != wgpu::Backends::all() => {
                timing::mark_note("gpu", "backend_fallback", first.to_string());
                Core::new(new_instance(wgpu::Backends::all()), None, filter)
            }
            Err(e) => Err(e),
        }
    }

    /// Move RGBA8 pixels into a texture, downscaling first if the image
    /// exceeds the hardware's maximum texture dimension.
    pub fn upload(&self, rgba: &[u8], width: u32, height: u32) -> Uploaded {
        self.upload_inner(rgba, width, height, true, false)
    }

    /// As [`Core::upload`], with a full mip chain built on the card. For the
    /// picture itself, which is routinely shown at a fraction of its size:
    /// linear filtering alone reads only four texels per screen pixel, so a
    /// large image fitted to the window would alias and shimmer.
    pub fn upload_mipmapped(&self, rgba: &[u8], width: u32, height: u32) -> Uploaded {
        self.upload_inner(rgba, width, height, true, true)
    }

    /// As [`Core::upload`], without waiting for the copy to land: it is
    /// ordered before the next frame's submit anyway. For tiles, which arrive
    /// by the dozen, where waiting for each one in turn would stall the loop.
    pub fn upload_unflushed(&self, rgba: &[u8], width: u32, height: u32) -> Uploaded {
        self.upload_inner(rgba, width, height, false, false)
    }

    fn upload_inner(&self, rgba: &[u8], width: u32, height: u32, flush: bool, mipmapped: bool) -> Uploaded {
        self.uploader().upload(rgba, width, height, flush, mipmapped)
    }

    /// What an upload needs, detached from the window so it can run on a
    /// thread of its own.
    fn uploader(&self) -> Uploader {
        Uploader {
            device: self.device.clone(),
            queue: self.queue.clone(),
            max_texture: self.max_texture,
            mips: self.mips.clone(),
        }
    }
}

/// Everything an upload touches. The device and queue are shared handles,
/// so this can be sent to another thread while the window keeps drawing.
#[derive(Clone)]
struct Uploader {
    device: wgpu::Device,
    queue: wgpu::Queue,
    max_texture: u32,
    mips: std::sync::Arc<std::sync::OnceLock<MipGen>>,
}

impl Uploader {
    fn upload(&self, rgba: &[u8], width: u32, height: u32, flush: bool, mipmapped: bool) -> Uploaded {
        let factor = crate::scale::factor_for_limit(width, height, self.max_texture);

        let (owned, w, h) = if factor > 1 {
            let mut s = timing::span("gpu", "downscale");
            s.note(format!("{width}x{height} /{factor} (max texture {})", self.max_texture));
            let (px, w, h) = crate::scale::box_downscale(rgba, width, height, factor);
            (Some(px), w, h)
        } else {
            (None, width, height)
        };
        let pixels: &[u8] = owned.as_deref().unwrap_or(rgba);
        let levels = if mipmapped { mip_levels(w, h) } else { 1 };

        let texture = {
            let mut s = timing::span("gpu", "texture_alloc");
            s.note(format!("{w}x{h} = {}", crate::loader::human_bytes(w as u64 * h as u64 * 4)));
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("image"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: if levels > 1 {
                    wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                } else {
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
                },
                view_formats: &[],
            })
        };

        {
            let mut s = timing::span("gpu", "upload");
            s.note(crate::loader::human_bytes(w as u64 * h as u64 * 4));
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels[..(w as usize * h as usize * 4)],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }

        if levels > 1 {
            let mut s = timing::span("gpu", "mips");
            s.note(format!("{levels} levels"));
            let generate = self.mips.get_or_init(|| MipGen::new(&self.device));
            generate.run(&self.device, &self.queue, &texture, levels);
        }

        if flush {
            // write_texture only stages the copy; the transfer happens on
            // submit. Waiting here is what makes the number above mean
            // anything, and it must happen before the first draw regardless.
            let _s = timing::span("gpu", "upload_flush");
            self.queue.submit(std::iter::empty());
            let _ = self.device.poll(wgpu::PollType::Wait);
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Uploaded { texture, view, source: (width, height), uploaded: (w, h), downscaled_by: factor }
    }
}

pub struct Uploaded {
    /// Held so the view below stays valid; nothing reads it directly.
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    /// Dimensions of the decoded image.
    #[allow(dead_code)] // Reported by --profile.
    pub source: (u32, u32),
    /// Dimensions actually living on the GPU.
    pub uploaded: (u32, u32),
    pub downscaled_by: u32,
}

/// Levels in a full mip chain for `w` x `h`, down to 1x1.
fn mip_levels(w: u32, h: u32) -> u32 {
    32 - w.max(h).max(1).leading_zeros()
}

/// Builds each mip level from the one above with a render pass.
struct MipGen {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl MipGen {
    fn new(device: &wgpu::Device) -> MipGen {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mip"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mip.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mip"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mip"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mip"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mip"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        MipGen { pipeline, layout, sampler }
    }

    /// Fill levels 1.. of `texture` from level 0, which must already be
    /// written. Ordered after that write on the queue, so nothing waits here.
    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, levels: u32) {
        let view = |level| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mip") });
        for level in 1..levels {
            let src = view(level - 1);
            let dst = view(level);
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mip"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mip"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

/// Uniforms for the planar-YUV pipeline: where to draw, and how to convert.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct YuvUniforms {
    rect: [f32; 4],
    row_r: [f32; 4],
    row_g: [f32; 4],
    row_b: [f32; 4],
    offsets: [f32; 4],
    turn: [f32; 4],
}

/// What the mask pipeline needs to know.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MaskUniforms {
    rect: [f32; 4],
    /// Outline thickness, fill opacity, and the label grid's size.
    style: [f32; 4],
    palette: [[f32; 4]; 16],
    turn: [f32; 4],
}

/// What the depth pipeline needs to know.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DepthUniforms {
    rect: [f32; 4],
    /// Opacity, quarter turns clockwise, then nothing.
    style: [f32; 4],
}

/// A depth map, a byte per cell, and where to draw it.
pub struct DepthMap {
    texture: wgpu::Texture,
    size: (u32, u32),
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// A grid of instance ids, and the colours to draw them in.
///
/// One byte per cell rather than four: a screen-sized RGBA overlay would be
/// 8.3 MB a frame, and this is 130 KB. Allocated once and rewritten, in the
/// same way `YuvFrame` is.
pub struct MaskGrid {
    labels: wgpu::Texture,
    size: (u32, u32),
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// Three single-channel planes and the transform that turns them into colour.
pub struct YuvFrame {
    y: wgpu::Texture,
    u: wgpu::Texture,
    v: wgpu::Texture,
    size: (u32, u32),
    chroma: (u32, u32),
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// The pipeline, samplers and uniform buffer for one target format.
pub struct Renderer {
    depth_pipeline: wgpu::RenderPipeline,
    depth_layout: wgpu::BindGroupLayout,
    mask_pipeline: wgpu::RenderPipeline,
    mask_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    yuv_pipeline: wgpu::RenderPipeline,
    yuv_layout: wgpu::BindGroupLayout,
    linear: wgpu::Sampler,
    nearest: wgpu::Sampler,
}

/// A texture bound for drawing, with both filter modes ready to switch between.
///
/// Each carries its own uniform buffer so several can be drawn in one pass with
/// different rectangles — the image and the overlay on top of it.
pub struct Bound {
    uniforms: wgpu::Buffer,
    linear: wgpu::BindGroup,
    nearest: wgpu::BindGroup,
}

/// One quad to draw this frame.
pub struct Layer<'a> {
    pub bound: &'a Bound,
    /// Position in NDC: `[min_x, min_y, max_x, max_y]`.
    pub rect: [f32; 4],
    /// Which part of the texture: see [`FULL_UV`].
    pub uv: [f32; 4],
    pub smooth: bool,
    /// Quarter turns clockwise to show the texture at, within `rect`.
    pub turn: u8,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Renderer {
        let _s = timing::span("gpu", "pipeline");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quad"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("image"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("image"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = |mag, label| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: mag,
                min_filter: wgpu::FilterMode::Linear,
                // Between mip levels too, so a zoom that crosses one does not
                // visibly step. Textures without mips are unaffected.
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        };

        let (yuv_pipeline, yuv_layout) = build_yuv_pipeline(device, format);
        let (mask_pipeline, mask_layout) = build_mask_pipeline(device, format);
        let (depth_pipeline, depth_layout) = build_depth_pipeline(device, format);

        Renderer {
            depth_pipeline,
            depth_layout,
            mask_pipeline,
            mask_layout,
            pipeline,
            layout,
            yuv_pipeline,
            yuv_layout,
            linear: sampler(wgpu::FilterMode::Linear, "linear"),
            nearest: sampler(wgpu::FilterMode::Nearest, "nearest"),
        }
    }

    /// Allocate the three plane textures for a frame size.
    /// Allocate the label grid and its uniforms.
    pub fn make_mask(&self, core: &Core, size: (u32, u32)) -> MaskGrid {
        let labels = core.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("labels"),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // One byte per cell, read as an integer. Not `R8Unorm`: an id is
            // not a brightness, and sampling one as a float would quantise it.
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = labels.create_view(&wgpu::TextureViewDescriptor::default());

        let uniforms = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask uniforms"),
            size: std::mem::size_of::<MaskUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mask"),
            layout: &self.mask_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });

        MaskGrid { labels, size, uniforms, bind }
    }

    /// Allocate a depth texture and its uniforms.
    pub fn make_depth(&self, core: &Core, size: (u32, u32)) -> DepthMap {
        let texture = core.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // A brightness this time, and filtered: the map is a fraction of
            // the picture's size, and stretched nearest it would be blocks.
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniforms = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depth uniforms"),
            size: std::mem::size_of::<DepthUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth"),
            layout: &self.depth_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear),
                },
            ],
        });
        DepthMap { texture, size, uniforms, bind }
    }

    pub fn make_yuv(&self, core: &Core, size: (u32, u32), chroma: (u32, u32)) -> YuvFrame {
        let plane = |w: u32, h: u32, label| {
            core.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // One byte per sample: that is what a plane is.
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let (y, u, v) = (
            plane(size.0, size.1, "luma"),
            plane(chroma.0, chroma.1, "cb"),
            plane(chroma.0, chroma.1, "cr"),
        );
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());

        let uniforms = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yuv uniforms"),
            size: std::mem::size_of::<YuvUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuv"),
            layout: &self.yuv_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view(&y)) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&view(&u)) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&view(&v)) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.linear) },
            ],
        });

        YuvFrame { y, u, v, size, chroma, uniforms, bind }
    }

    pub fn bind(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> Bound {
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layer uniforms"),
            contents: bytemuck::bytes_of(&Uniforms { rect: [-1.0, -1.0, 1.0, 1.0], uv: FULL_UV, turn: [0.0; 4] }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let make = |sampler, label| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            })
        };
        let (linear, nearest) = (make(&self.linear, "linear"), make(&self.nearest, "nearest"));
        Bound { uniforms, linear, nearest }
    }

    /// Record and submit one frame, drawing `layers` back to front.
    pub fn draw(&self, core: &Core, target: &wgpu::TextureView, layers: &[Layer]) {
        self.draw_with_yuv(core, target, None, None, None, layers, layers.len().min(1))
    }

    /// As [`Renderer::draw`], with an optional planar-YUV frame underneath.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_with_yuv(
        &self,
        core: &Core,
        target: &wgpu::TextureView,
        yuv: Option<&wgpu::BindGroup>,
        depth: Option<&wgpu::BindGroup>,
        mask: Option<&wgpu::BindGroup>,
        layers: &[Layer],
        // How many of `layers`, from the front, are the picture: the image,
        // or its tiles. The rest is interface, drawn over the outlines.
        pictures: usize,
    ) {
        for layer in layers {
            core.queue.write_buffer(
                &layer.bound.uniforms,
                0,
                bytemuck::bytes_of(&Uniforms { rect: layer.rect, uv: layer.uv, turn: [layer.turn as f32, 0.0, 0.0, 0.0] }),
            );
        }

        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("image"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if let Some(bind) = yuv {
                pass.set_pipeline(&self.yuv_pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..4, 0..1);
            }

            // The picture, then what was found on it, then the readouts.
            //
            // The order is the whole point and it was wrong: the mask used to
            // be drawn before `layers`, and for a *still* the picture is in
            // `layers` — so the outlines were drawn and then painted over by
            // the photograph. It looked right on video, where the picture goes
            // through the YUV pipeline above and is already down by this
            // point, which is why nothing but a still ever showed the fault.
            let (picture, interface) = layers.split_at(pictures.min(layers.len()));
            if !picture.is_empty() {
                pass.set_pipeline(&self.pipeline);
                for layer in picture {
                    let group = if layer.smooth { &layer.bound.linear } else { &layer.bound.nearest };
                    pass.set_bind_group(0, group, &[]);
                    pass.draw(0..4, 0..1);
                }
            }
            let layers = interface.iter();

            // Depth colours the picture, so it goes on before the outlines:
            // they are drawn on top of whatever the picture looks like.
            if let Some(bind) = depth {
                pass.set_pipeline(&self.depth_pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..4, 0..1);
            }

            if let Some(bind) = mask {
                pass.set_pipeline(&self.mask_pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..4, 0..1);
            }

            let rest: Vec<&Layer> = layers.collect();
            if !rest.is_empty() {
                pass.set_pipeline(&self.pipeline);
                for layer in rest {
                    let group =
                        if layer.smooth { &layer.bound.linear } else { &layer.bound.nearest };
                    pass.set_bind_group(0, group, &[]);
                    pass.draw(0..4, 0..1);
                }
            }
        }
        core.queue.submit(Some(encoder.finish()));
    }
}

/// The windowed renderer: a [`Core`] plus a configured surface.
pub struct Gpu {
    core: Core,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    image: Option<(Uploaded, Bound)>,
    /// A tiled image's tiles, each its own texture, under [`TILE_BUDGET`].
    tiles: crate::tiles::TileCache<(Uploaded, Bound)>,
    /// A video frame's planes, held between frames so they are allocated once.
    yuv: Option<YuvFrame>,
    /// What a model found, as a grid of instance ids.
    mask: Option<MaskGrid>,
    /// How far away things are, when that is being shown.
    depth: Option<DepthMap>,
    /// Drawn over the image in order, each with its own rectangle.
    overlays: Vec<(Uploaded, Bound)>,
    /// Quarter turns clockwise the picture, and all drawn over it, is shown at.
    turn: u8,
}

impl Gpu {
    pub fn new(
        window: std::sync::Arc<winit::window::Window>,
        adapter_filter: Option<&str>,
        vsync: bool,
        backends: wgpu::Backends,
    ) -> Result<Self> {
        match Gpu::with_backends(window.clone(), adapter_filter, vsync, backends) {
            Ok(gpu) => Ok(gpu),
            Err(first) if backends != wgpu::Backends::all() => {
                timing::mark_note("gpu", "backend_fallback", first.to_string());
                Gpu::with_backends(window, adapter_filter, vsync, wgpu::Backends::all())
            }
            Err(e) => Err(e),
        }
    }

    fn with_backends(
        window: std::sync::Arc<winit::window::Window>,
        adapter_filter: Option<&str>,
        vsync: bool,
        backends: wgpu::Backends,
    ) -> Result<Self> {
        let instance = new_instance(backends);
        let surface = create_surface(&instance, window.clone())?;
        let core = Core::new(instance, Some(&surface), adapter_filter)?;
        Gpu::finish(core, surface, window, vsync)
    }

    /// Attach a [`Core`] built somewhere else -- in practice, on a background
    /// thread while the CPU was busy putting the first frame on screen.
    ///
    /// The adapter was chosen without a surface to be compatible with, which on
    /// any desktop driver is fine; if it turns out not to be, the caller still
    /// has [`Gpu::new`] to fall back to.
    pub fn from_core(
        core: Core,
        window: std::sync::Arc<winit::window::Window>,
        vsync: bool,
    ) -> Result<Self> {
        let surface = create_surface(&core.instance, window.clone())?;
        Gpu::finish(core, surface, window, vsync)
    }

    fn finish(
        core: Core,
        surface: wgpu::Surface<'static>,
        window: std::sync::Arc<winit::window::Window>,
        vsync: bool,
    ) -> Result<Self> {
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        let caps = surface.get_capabilities(&core.adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);

        // Mailbox shows the newest finished frame without blocking, so the
        // first frame reaches the screen a refresh sooner than Fifo allows.
        let present_mode = if vsync {
            wgpu::PresentMode::Fifo
        } else {
            [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Immediate]
                .into_iter()
                .find(|m| caps.present_modes.contains(m))
                .unwrap_or(wgpu::PresentMode::Fifo)
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        {
            let _s = timing::span("gpu", "configure");
            surface.configure(&core.device, &config);
        }

        let renderer = Renderer::new(&core.device, format);
        Ok(Gpu {
            core,
            surface,
            config,
            renderer,
            image: None,
            tiles: crate::tiles::TileCache::new(TILE_BUDGET),
            yuv: None,
            mask: None,
            depth: None,
            overlays: Vec::new(),
            turn: 0,
        })
    }

    #[allow(dead_code)] // Part of the Gpu surface; used by diagnostics.
    pub fn max_texture(&self) -> u32 {
        self.core.max_texture
    }

    #[allow(dead_code)] // Part of the Gpu surface; used by diagnostics.
    pub fn adapter_name(&self) -> &str {
        &self.core.adapter_name
    }

    pub fn size(&self) -> (f32, f32) {
        (self.config.width as f32, self.config.height as f32)
    }

    pub fn downscaled_by(&self) -> u32 {
        self.image.as_ref().map(|(u, _)| u.downscaled_by).unwrap_or(1)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.config.width && height == self.config.height) {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.core.device, &self.config);
    }

    pub fn upload(&mut self, rgba: &[u8], width: u32, height: u32) {
        let uploaded = self.core.upload_mipmapped(rgba, width, height);
        self.adopt(uploaded);
    }

    /// As [`Gpu::upload`], on a thread of its own: the copy, the mips and the
    /// wait for both happen there, and the window keeps drawing whatever it
    /// has meanwhile. A 183 MB upload costs about 41 ms, which on this thread
    /// would be several dropped frames. `done` gets the texture, which is
    /// shown once it comes back to [`Gpu::adopt`].
    pub fn upload_in_background(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        done: impl FnOnce(Uploaded) + Send + 'static,
    ) {
        let uploader = self.core.uploader();
        let spawned = std::thread::Builder::new().name("upload".into()).spawn(move || {
            done(uploader.upload(&rgba, width, height, true, true));
        });
        if let Err(e) = spawned {
            eprintln!("show: could not start an upload thread: {e}");
        }
    }

    /// Show a texture made by [`Gpu::upload_in_background`].
    pub fn adopt(&mut self, uploaded: Uploaded) {
        // An image replaces any video planes that were being drawn.
        self.yuv = None;
        let bound = self.renderer.bind(&self.core.device, &uploaded.view);
        self.image = Some((uploaded, bound));
    }

    /// Upload a planar YUV frame, reusing the plane textures between frames.
    ///
    /// Called once per frame during playback, so nothing here allocates unless
    /// the video's dimensions change — which for a given file they never do.
    pub fn upload_yuv(
        &mut self,
        planes: [&[u8]; 3],
        size: (u32, u32),
        chroma: (u32, u32),
    ) {
        let matches =
            self.yuv.as_ref().is_some_and(|f| f.size == size && f.chroma == chroma);
        if !matches {
            self.yuv = Some(self.renderer.make_yuv(&self.core, size, chroma));
            // Planes replace the image; whichever was set last is what draws.
            self.image = None;
        }
        let frame = self.yuv.as_ref().expect("just built");

        let write = |texture: &wgpu::Texture, data: &[u8], w: u32, h: u32| {
            let want = w as usize * h as usize;
            if data.len() < want {
                return;
            }
            self.core.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data[..want],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // One byte per sample, packed tight by the decoder.
                    bytes_per_row: Some(w),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        };
        write(&frame.y, planes[0], size.0, size.1);
        write(&frame.u, planes[1], chroma.0, chroma.1);
        write(&frame.v, planes[2], chroma.0, chroma.1);
    }

    /// Put a grid of instance ids on screen, in the given colours.
    ///
    /// `labels` is one byte per cell, row-major, zero meaning nothing there.
    /// The grid is allocated on first use and reused after, so a result
    /// arriving five times a second costs one texture write rather than one
    /// allocation.
    pub fn set_mask(
        &mut self,
        labels: &[u8],
        size: (u32, u32),
        rect: [f32; 4],
        palette: &[[f32; 4]],
        outline: f32,
        fill: f32,
    ) {
        if size.0 == 0 || size.1 == 0 || labels.len() < (size.0 * size.1) as usize {
            self.mask = None;
            return;
        }
        if !self.mask.as_ref().is_some_and(|m| m.size == size) {
            self.mask = Some(self.renderer.make_mask(&self.core, size));
        }
        let grid = self.mask.as_ref().expect("just built");

        self.core.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &grid.labels,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &labels[..(size.0 * size.1) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.0),
                rows_per_image: Some(size.1),
            },
            wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        );

        let mut colours = [[0.0f32; 4]; 16];
        for (slot, colour) in colours.iter_mut().zip(palette) {
            *slot = *colour;
        }
        self.core.queue.write_buffer(
            &grid.uniforms,
            0,
            bytemuck::bytes_of(&MaskUniforms {
                rect,
                style: [outline, fill, size.0 as f32, size.1 as f32],
                palette: colours,
                turn: [self.turn as f32, 0.0, 0.0, 0.0],
            }),
        );
    }

    pub fn clear_mask(&mut self) {
        self.mask = None;
    }

    /// Put a depth map over the picture: a byte per cell, 255 nearest,
    /// drawn in `rect` at `opacity`. Reallocated only when its size changes.
    pub fn set_depth(&mut self, cells: &[u8], size: (u32, u32), rect: [f32; 4], opacity: f32) {
        if size.0 == 0 || size.1 == 0 || cells.len() < (size.0 * size.1) as usize {
            self.depth = None;
            return;
        }
        if !self.depth.as_ref().is_some_and(|d| d.size == size) {
            self.depth = Some(self.renderer.make_depth(&self.core, size));
        }
        let map = self.depth.as_ref().expect("just built");
        // Rows of a single-byte texture need no padding for `write_texture`;
        // the alignment rule is for buffer copies.
        self.core.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &map.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &cells[..(size.0 * size.1) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.0),
                rows_per_image: Some(size.1),
            },
            wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        );
        self.core.queue.write_buffer(
            &map.uniforms,
            0,
            bytemuck::bytes_of(&DepthUniforms { rect, style: [opacity, self.turn as f32, 0.0, 0.0] }),
        );
    }

    /// Move the depth map with the picture, without new contents.
    pub fn place_depth(&mut self, rect: [f32; 4], opacity: f32) {
        if let Some(map) = self.depth.as_ref() {
            self.core.queue.write_buffer(
                &map.uniforms,
                0,
                bytemuck::bytes_of(&DepthUniforms { rect, style: [opacity, self.turn as f32, 0.0, 0.0] }),
            );
        }
    }

    /// How the picture is turned: the image, its tiles, a video frame, and
    /// the depth and outlines over them. Set before the rest of a frame's
    /// uniforms are written, which read it.
    pub fn set_turn(&mut self, turn: u8) {
        self.turn = turn % 4;
    }

    pub fn clear_depth(&mut self) {
        self.depth = None;
    }

    /// Set the colour transform for the frames being uploaded.
    pub fn set_yuv_colour(&mut self, rect: [f32; 4], rows: [[f32; 3]; 3], offsets: [f32; 2]) {
        let Some(frame) = self.yuv.as_ref() else { return };
        let pad = |r: [f32; 3]| [r[0], r[1], r[2], 0.0];
        self.core.queue.write_buffer(
            &frame.uniforms,
            0,
            bytemuck::bytes_of(&YuvUniforms {
                rect,
                row_r: pad(rows[0]),
                row_g: pad(rows[1]),
                row_b: pad(rows[2]),
                offsets: [offsets[0], offsets[1], 0.0, 0.0],
                turn: [self.turn as f32, 0.0, 0.0, 0.0],
            }),
        );
    }

    pub fn clear_yuv(&mut self) {
        self.yuv = None;
    }


    /// Replace every overlay texture. An empty slice removes them all.
    pub fn set_overlays(&mut self, overlays: &[(&[u8], u32, u32)]) {
        self.overlays = overlays
            .iter()
            .map(|(rgba, w, h)| {
                let uploaded = self.core.upload(rgba, *w, *h);
                let bound = self.renderer.bind(&self.core.device, &uploaded.view);
                (uploaded, bound)
            })
            .collect();
    }

    /// Put one tile on the card, ready to draw.
    pub fn upload_tile(&mut self, key: crate::tiles::TileKey, tile: &crate::tiled::Tile) {
        let uploaded = self.core.upload_unflushed(&tile.rgba, tile.width, tile.height);
        let bound = self.renderer.bind(&self.core.device, &uploaded.view);
        self.tiles.insert(key, (uploaded, bound), tile.rgba.len() as u64);
    }

    pub fn has_tile(&self, key: &crate::tiles::TileKey) -> bool {
        self.tiles.peek(key).is_some()
    }

    /// Mark a tile as used this frame, so eviction takes others first.
    pub fn touch_tile(&mut self, key: &crate::tiles::TileKey) {
        let _ = self.tiles.get(key);
    }

    /// Shed tiles over budget, never those in `keep`.
    pub fn evict_tiles(&mut self, keep: &std::collections::HashSet<crate::tiles::TileKey>) {
        self.tiles.evict(keep);
    }

    pub fn clear_tiles(&mut self) {
        self.tiles.clear();
    }

    /// Tiles held, and their bytes, for the debug readout.
    pub fn tile_usage(&self) -> (usize, u64) {
        (self.tiles.len(), self.tiles.bytes())
    }

    /// Draw the image, then its tiles over it, then the overlays.
    ///
    /// Each tile draw is a tile already on the card, where it goes, and which
    /// part of it: a coarser tile standing in for a missing finer one shows
    /// only the corner that covers the same ground. Tiles not on the card are
    /// skipped — the image beneath is their fallback.
    pub fn render(
        &mut self,
        rect: [f32; 4],
        smooth: bool,
        tile_draws: &[(crate::tiles::TileKey, [f32; 4], [f32; 4])],
        overlay_rects: &[[f32; 4]],
    ) -> Result<(), wgpu::SurfaceError> {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.core.device, &self.config);
                self.surface.get_current_texture()?
            }
            Err(e) => return Err(e),
        };

        let mut layers = Vec::with_capacity(1 + tile_draws.len() + self.overlays.len());
        if let Some((_, bound)) = &self.image {
            layers.push(Layer { bound, rect, uv: FULL_UV, smooth, turn: self.turn });
        }
        for (key, rect, uv) in tile_draws {
            if let Some((_, bound)) = self.tiles.peek(key) {
                layers.push(Layer { bound, rect: *rect, uv: *uv, smooth, turn: self.turn });
            }
        }
        let pictures = layers.len();
        // Video planes draw through their own pipeline, before the overlays.
        let yuv_bind = self.yuv.as_ref().map(|f| &f.bind);
        for ((_, bound), rect) in self.overlays.iter().zip(overlay_rects) {
            // Nearest keeps the bitmap font's edges square; filtering it would
            // only blur pixels that were authored at exactly this size.
            layers.push(Layer { bound, rect: *rect, uv: FULL_UV, smooth: false, turn: 0 });
        }

        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mask_bind = self.mask.as_ref().map(|m| &m.bind);
        let depth_bind = self.depth.as_ref().map(|d| &d.bind);
        // With a video frame beneath, there is no picture layer: the frame is.
        let pictures = if yuv_bind.is_some() { 0 } else { pictures };
        self.renderer.draw_with_yuv(&self.core, &view, yuv_bind, depth_bind, mask_bind, &layers, pictures);
        frame.present();
        Ok(())
    }
}

/// The pipeline that draws found things over the picture.
///
/// Alpha blended, unlike the video and image pipelines: it is drawn *over*
/// something rather than replacing it, and most of its texels are transparent.
fn build_mask_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mask"),
        source: wgpu::ShaderSource::Wgsl(include_str!("mask.wgsl").into()),
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mask"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                // Unsigned integers, not floats: these are identifiers, and a
                // filtered sample between two of them would mean a third thing.
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mask"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mask"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, layout)
}

/// The pipeline that colours the picture by distance. Alpha blended, as the
/// mask is; its texture is filtered, as the mask's is not.
fn build_depth_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("depth"),
        source: wgpu::ShaderSource::Wgsl(include_str!("depth.wgsl").into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("depth"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("depth"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("depth"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    (pipeline, layout)
}

fn build_yuv_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("yuv"),
        source: wgpu::ShaderSource::Wgsl(include_str!("yuv.wgsl").into()),
    });

    let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("yuv"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            texture_entry(1),
            texture_entry(2),
            texture_entry(3),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("yuv"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("yuv"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, layout)
}

fn create_surface(
    instance: &wgpu::Instance,
    window: std::sync::Arc<winit::window::Window>,
) -> Result<wgpu::Surface<'static>> {
    let _s = timing::span("gpu", "surface");
    instance.create_surface(window).context("creating surface")
}

fn pick_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'static>>,
    filter: Option<&str>,
) -> Result<wgpu::Adapter> {
    if let Some(filter) = filter {
        let all = instance.enumerate_adapters(wgpu::Backends::all());
        // Accept either an index or a case-insensitive substring of the name.
        let found = filter
            .parse::<usize>()
            .ok()
            .and_then(|i| all.get(i).cloned())
            .or_else(|| {
                let needle = filter.to_lowercase();
                all.iter().find(|a| a.get_info().name.to_lowercase().contains(&needle)).cloned()
            });
        if let Some(a) = found {
            return Ok(a);
        }
        anyhow::bail!(
            "no adapter matching {filter:?}; available:\n  {}",
            describe_adapters().join("\n  ")
        );
    }

    // LowPower asks for the integrated GPU, and for this workload that is
    // simply the faster device. Measured cold, first frame on screen: the
    // integrated Radeon takes 106 ms against the discrete RTX 4070 SUPER's
    // 217 ms, because NVIDIA's driver spends ~74 ms creating a device and
    // ~32 ms configuring a surface where RADV spends 1.4 ms and 3.3 ms.
    // Showing one texture on one quad never needs discrete throughput, so
    // there is nothing on the other side of the trade to weigh against it.
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: surface,
        force_fallback_adapter: false,
    }))
    .context("no suitable GPU adapter")
}

#[cfg(test)]
mod pipeline_tests {
    /// Every pipeline builds: a shader that fails validation is a panic at
    /// launch, and nothing else in the tests would catch it.
    ///
    /// Skipped, not failed, on a machine with no adapter at all.
    #[test]
    fn every_pipeline_builds() {
        let Ok(core) = super::Core::new_default(None, wgpu::Backends::all()) else {
            eprintln!("no adapter; pipelines not checked");
            return;
        };
        let _ = super::Renderer::new(&core.device, wgpu::TextureFormat::Bgra8UnormSrgb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mip_chain_ends_at_one_pixel() {
        assert_eq!(mip_levels(1, 1), 1);
        assert_eq!(mip_levels(2, 1), 2);
        assert_eq!(mip_levels(1024, 768), 11);
        assert_eq!(mip_levels(1025, 10), 11);
        assert_eq!(mip_levels(6000, 4000), 13);
        // The largest texture a card offers still fits.
        assert_eq!(mip_levels(16384, 16384), 15);
    }
}
