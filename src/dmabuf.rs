//! Hardware-decoded video frames taken straight into Vulkan.
//!
//! VA-API decodes into GPU memory, and reading every frame back to the CPU
//! only to upload it again cost about what decoding on the card saved:
//! measured at 24% CPU against software decoding's 21% on 4K HEVC. Here the
//! decoder's surface is exported as DMA-BUFs (one per plane), imported as
//! Vulkan images on the device that draws, and copied on the card into
//! textures of our own. The copy is what lets the decoder have its surface
//! back at once: the read-ahead queue holds three seconds of frames, far more
//! than the decoder's small pool of surfaces.
//!
//! It needs three device extensions, asked for when the device is created,
//! and the decoder and the display on the same GPU. Without either, nothing
//! here is used and frames are read back as before.

use std::ffi::CStr;
use std::os::fd::{AsRawFd, BorrowedFd, IntoRawFd};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use ash::vk;
use wgpu::hal::api::Vulkan;

const EXTENSIONS: [&CStr; 3] = [
    ash::khr::external_memory_fd::NAME,
    ash::ext::external_memory_dma_buf::NAME,
    ash::ext::image_drm_format_modifier::NAME,
];

/// DRM fourccs FFmpeg uses for the layers of an exported VA surface.
const DRM_R8: u32 = fourcc(b"R8  ");
const DRM_GR88: u32 = fourcc(b"GR88");
const DRM_R16: u32 = fourcc(b"R16 ");
const DRM_GR1616: u32 = fourcc(b"GR32");

/// libva's own export, called directly: FFmpeg's route to it needs FFmpeg
/// built with libdrm, which the vendored build is not. libva is already
/// linked, since FFmpeg's VA-API decoding needs it.
mod va {
    use std::ffi::c_void;

    pub const MEM_TYPE_DRM_PRIME_2: u32 = 0x4000_0000;
    pub const EXPORT_READ_ONLY: u32 = 0x0001;
    pub const EXPORT_SEPARATE_LAYERS: u32 = 0x0004;

    #[repr(C)]
    #[derive(Default)]
    pub struct Object {
        pub fd: i32,
        pub size: u32,
        pub modifier: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct Layer {
        pub drm_format: u32,
        pub num_planes: u32,
        pub object_index: [u32; 4],
        pub offset: [u32; 4],
        pub pitch: [u32; 4],
    }

    /// VADRMPRIMESurfaceDescriptor.
    #[repr(C)]
    #[derive(Default)]
    pub struct Descriptor {
        pub fourcc: u32,
        pub width: u32,
        pub height: u32,
        pub num_objects: u32,
        pub objects: [Object; 4],
        pub num_layers: u32,
        pub layers: [Layer; 4],
    }

    #[link(name = "va")]
    unsafe extern "C" {
        pub fn vaSyncSurface(display: *mut c_void, surface: u32) -> i32;
        pub fn vaExportSurfaceHandle(
            display: *mut c_void,
            surface: u32,
            mem_type: u32,
            flags: u32,
            descriptor: *mut c_void,
        ) -> i32;
    }
}

const fn fourcc(code: &[u8; 4]) -> u32 {
    code[0] as u32 | (code[1] as u32) << 8 | (code[2] as u32) << 16 | (code[3] as u32) << 24
}

/// The device frames are imported into, and which GPU that is.
#[derive(Clone)]
pub struct Interop {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Major and minor of the GPU's render node, to match the decoder's.
    render: (u64, u64),
    /// Whether 16-bit planes (10-bit video) can be held.
    deep: bool,
}

static INTEROP: Mutex<Option<Interop>> = Mutex::new(None);

/// Whether zero-copy is switched off: `SHOW_ZERO_COPY=0`.
fn disabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var("SHOW_ZERO_COPY").is_ok_and(|v| v == "0"))
}

/// Open a device on `adapter` with the import extensions enabled, when it is
/// a Vulkan adapter that has them. `None` means: create the device as usual.
pub fn request_device(
    adapter: &wgpu::Adapter,
    desc: &wgpu::DeviceDescriptor,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    if disabled() {
        return None;
    }
    // Safety: the hal adapter is only read here, and the device it opens is
    // handed straight back to wgpu, which owns it from then on.
    unsafe {
        let open = {
            let hal = adapter.as_hal::<Vulkan>()?;
            let instance = hal.shared_instance().raw_instance();
            let available = instance.enumerate_device_extension_properties(hal.raw_physical_device()).ok()?;
            let has = |name: &CStr| available.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
            if !EXTENSIONS.iter().all(|e| has(e)) || !has(ash::ext::physical_device_drm::NAME) {
                return None;
            }
            hal.open_with_callback(
                desc.required_features,
                &desc.memory_hints,
                Some(Box::new(|args| {
                    for e in EXTENSIONS {
                        if !args.extensions.contains(&e) {
                            args.extensions.push(e);
                        }
                    }
                })),
            )
            .ok()?
        };
        adapter.create_device_from_hal(open, desc).ok()
    }
}

/// Make `device` the one decoded frames are imported into. Called once the
/// device exists; a device opened without [`request_device`] is ignored.
pub fn install(device: &wgpu::Device, queue: &wgpu::Queue, opened_for_import: bool) {
    if !opened_for_import {
        return;
    }
    let Some(render) = render_node(device) else { return };
    let deep = device.features().contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);
    *INTEROP.lock().unwrap_or_else(|e| e.into_inner()) =
        Some(Interop { device: device.clone(), queue: queue.clone(), render, deep });
}

/// The render node of the GPU behind `device`, from VK_EXT_physical_device_drm.
fn render_node(device: &wgpu::Device) -> Option<(u64, u64)> {
    // Safety: a read-only query on the physical device.
    unsafe {
        let hal = device.as_hal::<Vulkan>()?;
        let instance = hal.shared_instance().raw_instance();
        let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
        instance.get_physical_device_properties2(hal.raw_physical_device(), &mut props);
        (drm.has_render == vk::TRUE).then_some((drm.render_major as u64, drm.render_minor as u64))
    }
}

/// The importer for frames decoded on `node`, if that is the GPU that draws.
pub fn for_node(node: &str) -> Option<Interop> {
    use std::os::unix::fs::MetadataExt;
    let interop = INTEROP.lock().unwrap_or_else(|e| e.into_inner()).clone()?;
    let rdev = std::fs::metadata(node).ok()?.rdev();
    let major = ((rdev >> 8) & 0xfff) | ((rdev >> 32) & !0xfff);
    let minor = (rdev & 0xff) | ((rdev >> 12) & !0xff);
    (interop.render == (major, minor)).then_some(interop)
}

/// A frame's planes on the card: luma, and chroma interleaved as two
/// channels, which is how the decoder lays them out.
pub struct CardPlanes {
    pub y: wgpu::Texture,
    pub uv: wgpu::Texture,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl CardPlanes {
    /// Bytes held on the card.
    pub fn bytes(&self) -> u64 {
        let size = |t: &wgpu::Texture| {
            let block = t.format().block_copy_size(None).unwrap_or(1) as u64;
            t.width() as u64 * t.height() as u64 * block
        };
        size(&self.y) + size(&self.uv)
    }

    /// Bring the planes back to memory as 8-bit 4:2:0, luma then Cb then Cr,
    /// each tightly packed: the layout every CPU consumer of a frame expects.
    /// Blocks until the card has copied them out.
    pub fn read_back(&self) -> Result<[Vec<u8>; 3]> {
        let y = read_texture(&self.device, &self.queue, &self.y)?;
        let uv = read_texture(&self.device, &self.queue, &self.uv)?;

        let deep = self.y.format() == wgpu::TextureFormat::R16Unorm;
        let (w, h) = (self.y.width() as usize, self.y.height() as usize);
        let (cw, ch) = (self.uv.width() as usize, self.uv.height() as usize);
        // 16-bit samples keep their value in the high byte.
        let sample = |row: &[u8], i: usize| if deep { row[i * 2 + 1] } else { row[i] };

        let mut luma = Vec::with_capacity(w * h);
        for row in y.rows() {
            luma.extend((0..w).map(|x| sample(row, x)));
        }
        let (mut cb, mut cr) = (Vec::with_capacity(cw * ch), Vec::with_capacity(cw * ch));
        for row in uv.rows() {
            for x in 0..cw {
                cb.push(sample(row, x * 2));
                cr.push(sample(row, x * 2 + 1));
            }
        }
        Ok([luma, cb, cr])
    }
}

/// A texture's contents, with the row padding the copy required.
struct Readback {
    data: Vec<u8>,
    padded_row: usize,
    height: usize,
}

impl Readback {
    fn rows(&self) -> impl Iterator<Item = &[u8]> {
        self.data.chunks(self.padded_row).take(self.height)
    }
}

fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Result<Readback> {
    let block = texture.format().block_copy_size(None).unwrap_or(1);
    let row = texture.width() * block;
    let padded_row = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame readback"),
        size: padded_row as u64 * texture.height() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame readback") });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(texture.height()),
            },
        },
        texture.size(),
    );
    let index = queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::WaitForSubmissionIndex(index));
    rx.recv().context("readback never finished")?.context("mapping the readback")?;
    let data = buffer.slice(..).get_mapped_range().to_vec();
    buffer.unmap();
    Ok(Readback { data, padded_row: padded_row as usize, height: texture.height() as usize })
}

/// What one layer of an exported surface becomes.
struct Layer {
    vk_format: vk::Format,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
}

impl Interop {
    /// Take a decoded VA-API frame onto the card: export it, import its
    /// layers, and copy them into textures of our own. The decoder's surface
    /// is free again when this returns.
    ///
    /// # Safety
    ///
    /// `frame` must be a valid VA-API frame (`AV_PIX_FMT_VAAPI`).
    pub unsafe fn take(&self, frame: *const ffmpeg_next::sys::AVFrame) -> Result<CardPlanes> {
        use ffmpeg_next::sys;

        // Export. The frame keeps its surface from the decoder until the copy
        // below has finished, because the caller holds it until this returns.
        let (display, surface, width, height) = unsafe {
            let frames = (*frame).hw_frames_ctx;
            if frames.is_null() {
                bail!("a hardware frame without a frames context");
            }
            let device = (*((*frames).data as *const sys::AVHWFramesContext)).device_ctx;
            let display = (*((*device).hwctx as *const sys::AVVAAPIDeviceContext)).display;
            ((display as *mut std::ffi::c_void), (*frame).data[3] as usize as u32, (*frame).width as u32, (*frame).height as u32)
        };
        let mut desc = va::Descriptor::default();
        unsafe {
            let rc = va::vaSyncSurface(display, surface);
            if rc != 0 {
                bail!("waiting for the decoder failed (VA status {rc})");
            }
            let rc = va::vaExportSurfaceHandle(
                display,
                surface,
                va::MEM_TYPE_DRM_PRIME_2,
                va::EXPORT_READ_ONLY | va::EXPORT_SEPARATE_LAYERS,
                &mut desc as *mut va::Descriptor as *mut std::ffi::c_void,
            );
            if rc != 0 {
                bail!("exporting the surface failed (VA status {rc})");
            }
        }
        // The export hands over one descriptor per object, ours to close.
        let _owned: Vec<std::os::fd::OwnedFd> = desc.objects[..(desc.num_objects as usize).min(4)]
            .iter()
            .map(|o| unsafe { std::os::fd::FromRawFd::from_raw_fd(o.fd) })
            .collect();
        if desc.num_layers != 2 {
            bail!("a surface with {} layers", desc.num_layers);
        }

        let chroma = (width.div_ceil(2), height.div_ceil(2));
        let layer = |fourcc: u32, (w, h): (u32, u32)| -> Result<Layer> {
            let (vk_format, format) = match fourcc {
                DRM_R8 => (vk::Format::R8_UNORM, wgpu::TextureFormat::R8Unorm),
                DRM_GR88 => (vk::Format::R8G8_UNORM, wgpu::TextureFormat::Rg8Unorm),
                DRM_R16 if self.deep => (vk::Format::R16_UNORM, wgpu::TextureFormat::R16Unorm),
                DRM_GR1616 if self.deep => (vk::Format::R16G16_UNORM, wgpu::TextureFormat::Rg16Unorm),
                other => bail!("layer format {:?}", other.to_le_bytes().map(|b| b as char)),
            };
            Ok(Layer { vk_format, format, width: w, height: h })
        };
        let luma = layer(desc.layers[0].drm_format, (width, height))?;
        let chroma = layer(desc.layers[1].drm_format, chroma)?;

        let imported_y = unsafe { self.import(&desc, 0, &luma)? };
        let imported_uv = unsafe { self.import(&desc, 1, &chroma)? };

        let own = |l: &Layer, label| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: l.width, height: l.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: l.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let y = own(&luma, "card luma");
        let uv = own(&chroma, "card chroma");

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("take frame") });
        encoder.copy_texture_to_texture(imported_y.as_image_copy(), y.as_image_copy(), y.size());
        encoder.copy_texture_to_texture(imported_uv.as_image_copy(), uv.as_image_copy(), uv.size());
        let index = self.queue.submit(Some(encoder.finish()));
        // The surface goes back to the decoder only once the copy is done.
        let _ = self.device.poll(wgpu::PollType::WaitForSubmissionIndex(index));
        drop((imported_y, imported_uv));

        Ok(CardPlanes { y, uv, device: self.device.clone(), queue: self.queue.clone() })
    }

    /// One layer of an exported surface as a texture that can be copied from.
    /// The texture owns the image and its memory, and frees both when wgpu is
    /// done with it.
    unsafe fn import(&self, desc: &va::Descriptor, index: usize, layer: &Layer) -> Result<wgpu::Texture> {
        let source = &desc.layers[index];
        if source.num_planes != 1 {
            bail!("a layer with {} planes", source.num_planes);
        }
        let object = desc.objects.get(source.object_index[0] as usize).context("no such object")?;

        let hal = unsafe { self.device.as_hal::<Vulkan>() }.context("not a Vulkan device")?;
        let device = hal.raw_device().clone();
        let instance = hal.shared_instance().raw_instance();
        let fd_api = ash::khr::external_memory_fd::Device::new(instance, &device);

        let layouts = [vk::SubresourceLayout {
            offset: source.offset[0] as u64,
            size: 0,
            row_pitch: source.pitch[0] as u64,
            array_pitch: 0,
            depth_pitch: 0,
        }];
        let mut modifier = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(object.modifier)
            .plane_layouts(&layouts);
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(layer.vk_format)
            .extent(vk::Extent3D { width: layer.width, height: layer.height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external)
            .push_next(&mut modifier);
        let image = unsafe { device.create_image(&info, None) }.context("creating the image")?;

        // Vulkan takes ownership of the descriptor it is given, so it gets a
        // duplicate: the export's own is closed when the frame is taken.
        let fd = unsafe { BorrowedFd::borrow_raw(object.fd) }.try_clone_to_owned()?;
        let memory = (|| -> Result<vk::DeviceMemory> {
            let mut fd_props = vk::MemoryFdPropertiesKHR::default();
            unsafe {
                fd_api.get_memory_fd_properties(
                    vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                    fd.as_raw_fd(),
                    &mut fd_props,
                )
            }
            .context("querying the buffer")?;
            let needs = unsafe { device.get_image_memory_requirements(image) };
            let types = needs.memory_type_bits & fd_props.memory_type_bits;
            if types == 0 {
                bail!("no memory type fits the buffer");
            }
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                .fd(fd.as_raw_fd());
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(types.trailing_zeros())
                .push_next(&mut import)
                .push_next(&mut dedicated);
            let memory = unsafe { device.allocate_memory(&alloc, None) }.context("importing the buffer")?;
            if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
                unsafe { device.free_memory(memory, None) };
                bail!("binding the buffer: {e}");
            }
            Ok(memory)
        })();
        let memory = match memory {
            Ok(memory) => {
                // The import succeeded, so the descriptor is Vulkan's now.
                let _ = fd.into_raw_fd();
                memory
            }
            Err(e) => {
                unsafe { device.destroy_image(image, None) };
                return Err(e);
            }
        };

        let size = wgpu::Extent3d { width: layer.width, height: layer.height, depth_or_array_layers: 1 };
        let hal_desc = wgpu::hal::TextureDescriptor {
            label: Some("imported plane"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: layer.format,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        let freeing = device.clone();
        let texture = unsafe {
            hal.texture_from_raw(
                image,
                &hal_desc,
                Some(Box::new(move || {
                    freeing.destroy_image(image, None);
                    freeing.free_memory(memory, None);
                })),
            )
        };
        drop(hal);
        Ok(unsafe {
            self.device.create_texture_from_hal::<Vulkan>(
                texture,
                &wgpu::TextureDescriptor {
                    label: Some("imported plane"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: layer.format,
                    usage: wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourccs_match_the_drm_headers() {
        // drm_fourcc.h: fourcc_code('R', '8', ' ', ' ') and friends.
        assert_eq!(DRM_R8, 0x2020_3852);
        assert_eq!(DRM_GR88, 0x3838_5247);
        assert_eq!(DRM_R16, 0x2036_3152);
        assert_eq!(DRM_GR1616, 0x3233_5247);
    }
}
