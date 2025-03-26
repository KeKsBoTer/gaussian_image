use std::{
    io::{Read, Seek},
    sync::Arc,
};

use eframe::{
    egui::{self, mutex::Mutex},
    egui_wgpu,
};
use half::f16;
use npyz::{Deserialize, TypeChar, npz::NpzArchive};
use num_traits::Float;
use wgpu::{VertexAttribute, include_wgsl, util::DeviceExt};

pub struct Gauss2D {
    pub mean: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
    pub offset: [f32; 2],
    pub cov: [f32; 4],
}

impl GaussianVertex {
    pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: [VertexAttribute; 4] = wgpu::vertex_attr_array![
            0 => Float32x2,
            1 => Float32x4,
            2 => Float32x2,
            3 => Float32x4,
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GaussianVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRS,
        }
    }
}

impl Gauss2D {
    fn vertices(&self, resolution: [u32; 2]) -> Option<[GaussianVertex; 6]> {
        let mu = self.mean;

        let scale_x = self.scale[0];
        let scale_y = self.scale[1];

        if scale_x.min(scale_y) <= 0.3 {
            return None;
        }

        let cov = build_covariance_matrix(self.rotation, [scale_x, scale_y]);

        // column major and transposed
        let rot: [[f32; 2]; 2] = [
            [
                f32::cos(self.rotation) * scale_x,
                f32::sin(self.rotation) * scale_y,
            ],
            [
                -f32::sin(self.rotation) * scale_x,
                f32::cos(self.rotation) * scale_y,
            ],
        ];

        let s = 2. * f32::ln(255.).sqrt();
        let offset = [
            transform2d([-s, -s], rot),
            transform2d([-s, s], rot),
            transform2d([s, -s], rot),
            transform2d([s, s], rot),
        ];
        // let offset_ndc = offset;
        let offset_ndc = [
            [
                2. * offset[0][0] / resolution[0] as f32,
                2. * offset[0][1] / resolution[1] as f32,
            ],
            [
                2. * offset[1][0] / resolution[0] as f32,
                2. * offset[1][1] / resolution[1] as f32,
            ],
            [
                2. * offset[2][0] / resolution[0] as f32,
                2. * offset[2][1] / resolution[1] as f32,
            ],
            [
                2. * offset[3][0] / resolution[0] as f32,
                2. * offset[3][1] / resolution[1] as f32,
            ],
        ];
        let pos = [
            vec_add(mu, offset_ndc[0]),
            vec_add(mu, offset_ndc[1]),
            vec_add(mu, offset_ndc[2]),
            vec_add(mu, offset_ndc[3]),
        ];

        let cov_inv = invert(cov);
        if cov_inv.is_none() {
            return None;
        }
        let cov_inv = cov_inv?;
        let cov_plain = [cov_inv[0][0], cov_inv[0][1], cov_inv[1][0], cov_inv[1][1]];
        Some([
            GaussianVertex {
                position: pos[0],
                color: self.color,
                offset: offset[0],
                cov: cov_plain,
            },
            GaussianVertex {
                position: pos[2],
                color: self.color,
                offset: offset[2],
                cov: cov_plain,
            },
            GaussianVertex {
                position: pos[1],
                color: self.color,
                offset: offset[1],
                cov: cov_plain,
            },
            GaussianVertex {
                position: pos[1],
                color: self.color,
                offset: offset[1],
                cov: cov_plain,
            },
            GaussianVertex {
                position: pos[2],
                color: self.color,
                offset: offset[2],
                cov: cov_plain,
            },
            GaussianVertex {
                position: pos[3],
                color: self.color,
                offset: offset[3],
                cov: cov_plain,
            },
        ])
    }
}

pub struct GaussianMixture {
    num_vertices: usize,
    buffer: wgpu::Buffer,
    pub resolution: [u32; 2],
}

impl GaussianMixture {
    pub fn new(
        device: &wgpu::Device,
        gaussian_image: &GaussianImage,
        resolution: [u32; 2],
    ) -> Self {
        let vertices: Vec<GaussianVertex> = gaussian_image
            .gaussians
            .iter()
            .filter_map(|g| g.vertices(gaussian_image.resolution))
            .flat_map(|v| v)
            .collect::<Vec<_>>();
        let num_vertices = vertices.len();
        let skipped = gaussian_image.gaussians.len() - num_vertices / 6;
        if skipped > 0 {
            log::warn!(
                "{} gaussians were skipped because they are invalid.",
                skipped
            );
        }

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian Mixture Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            buffer,
            num_vertices,
            resolution,
        }
    }

    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }
    pub fn len(&self) -> usize {
        self.num_vertices / 6
    }
}

#[derive(Clone)]
pub struct GaussianMixtureCallBack {
    pub gaussians: Arc<GaussianMixture>,
    pub settings: RasterizationSettings,
}

impl egui_wgpu::CallbackTrait for GaussianMixtureCallBack {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gaussian Rasterizer Command Encoder"),
        });
        let renderer: &mut GaussianRasterizer = resources.get_mut().unwrap();
        let resolution = renderer.resolution.lock().clone();


        renderer.prepare(
            device,
            queue,
            &mut encoder,
            self.gaussians.as_ref(),
            resolution,
            &self.settings,
        );
        vec![encoder.finish()]
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &GaussianRasterizer = resources.get().unwrap();
        let size = [
            info.viewport_in_pixels().width_px as u32,
            info.viewport_in_pixels().height_px as u32,
        ];
        resources.resolution.lock().copy_from_slice(&size);
        resources.render(render_pass);
    }
}

pub struct GaussianRasterizer {
    pipeline: wgpu::RenderPipeline,
    rasterization_bg: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    display_pipeline: wgpu::RenderPipeline,
    frame_buffer: FrameBuffer,
    resolution: Mutex<[u32; 2]>,
}

impl GaussianRasterizer {
    pub fn new(
        device: &wgpu::Device,
        render_format: wgpu::TextureFormat,
        resolution: [u32; 2],
    ) -> Self {
        let shader_module = device.create_shader_module(include_wgsl!("shaders/gaussian.wgsl"));

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gaussian Rasterizer Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian Rasterizer Buffer"),
            size: std::mem::size_of::<RasterizationSettings>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let rasterization_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gaussian Rasterizer Bind Group"),
            layout: &bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gaussian Rasterizer Pipeline Layout"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gaussian Rasterizer Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[GaussianVertex::vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                ],
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
        let display_shader_module =
            device.create_shader_module(include_wgsl!("shaders/display.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gaussian Display Pipeline Layout"),
            bind_group_layouts: &[&bg_layout, &FrameBuffer::bind_group_layout(device)],
            push_constant_ranges: &[],
        });

        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gaussian Display Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &display_shader_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &display_shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_format,
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
        let frame_buffer = FrameBuffer::new(device, resolution);
        Self {
            pipeline,
            rasterization_bg,
            display_pipeline,
            frame_buffer,
            uniform_buffer,
            resolution: Mutex::new(resolution),
        }
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussianMixture,
        resolution: [u32; 2],
        settings: &RasterizationSettings,
    ) {
        let tex_size = [
            self.frame_buffer.color.size().width,
            self.frame_buffer.color.size().height,
        ];
        if tex_size != resolution {
            self.frame_buffer.resize(device, resolution);
        }

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*settings]));

        {
            let mut render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Gaussian Rasterizer Render Pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self
                                .frame_buffer
                                .color
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self
                                .frame_buffer
                                .dx
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self
                                .frame_buffer
                                .dy
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self
                                .frame_buffer
                                .dxy
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();

            render_pass.set_vertex_buffer(0, gaussians.buffer.slice(..));
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.rasterization_bg, &[]);
            render_pass.draw(0..(gaussians.num_vertices()) as u32, 0..1);
        }
    }
    pub fn render(&self, render_pass: &mut wgpu::RenderPass<'static>) {
        render_pass.set_pipeline(&self.display_pipeline);
        render_pass.set_bind_group(0, &self.rasterization_bg, &[]);
        render_pass.set_bind_group(1, &self.frame_buffer.bind_group, &[]);
        render_pass.draw(0..4 as u32, 0..1);
    }
}

fn transform2d(v: [f32; 2], m: [[f32; 2]; 2]) -> [f32; 2] {
    [
        v[0] * m[0][0] + v[1] * m[0][1],
        v[0] * m[1][0] + v[1] * m[1][1],
    ]
}

fn vec_add(v1: [f32; 2], v2: [f32; 2]) -> [f32; 2] {
    [v1[0] + v2[0], v1[1] + v2[1]]
}

fn matmul(a: [[f32; 2]; 2], b: [[f32; 2]; 2]) -> [[f32; 2]; 2] {
    let mut res = [[0., 0.], [0., 0.]];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}
fn transpose(a: [[f32; 2]; 2]) -> [[f32; 2]; 2] {
    let mut res = [[0., 0.], [0., 0.]];
    for i in 0..2 {
        for j in 0..2 {
            res[i][j] = a[j][i];
        }
    }
    return res;
}

fn build_covariance_matrix(rotation: f32, scale: [f32; 2]) -> [[f32; 2]; 2] {
    let rot = [
        [f32::cos(rotation), -f32::sin(rotation)],
        [f32::sin(rotation), f32::cos(rotation)],
    ];
    let scale = [[scale[0], 0.], [0., scale[1]]];
    let l = matmul(rot, scale);
    let cov = matmul(l, transpose(l));
    [[cov[0][0], cov[0][1]], [cov[1][0], cov[1][1]]]
}

fn invert(mat: [[f32; 2]; 2]) -> Option<[[f32; 2]; 2]> {
    let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    if det == 0. {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [mat[1][1] * inv_det, -mat[0][1] * inv_det],
        [-mat[1][0] * inv_det, mat[0][0] * inv_det],
    ])
}

pub struct GaussianImage {
    pub gaussians: Vec<Gauss2D>,
    pub resolution: [u32; 2],
}
impl GaussianImage {
    pub fn from_npz<R: Read + Seek>(file: R) -> anyhow::Result<Self> {
        let mut reader = npyz::npz::NpzArchive::new(file)?;
        let dtype =reader.by_name("xyz")?.ok_or(anyhow::format_err!("missing field xyz"))?.dtype().clone();
        match dtype {
            npyz::DType::Plain(type_str) => {
                if type_str.type_char() != TypeChar::Float {
                    return Err(anyhow::anyhow!("data must be float"));
                }
                match type_str.num_bytes().ok_or(anyhow::anyhow!("cannot read num_bytes for datatype"))? {
                    2 => return Self::from_npz_dyn::<_, f16>(reader),
                    4 => return Self::from_npz_dyn::<_, f32>(reader),
                    _ => {
                        return Err(anyhow::anyhow!(
                            "unsupported float size: {}",
                            type_str.num_bytes().unwrap()
                        ))
                    }
                }
            }
            _ => {
                return Err(anyhow::anyhow!("data must be float"));
            }
        }
    }

    fn from_npz_dyn<R: Read + Seek, F: Float + Deserialize>(
        mut reader: NpzArchive<R>,
    ) -> anyhow::Result<Self> {
        let xyz_data = reader
            .by_name("xyz")?
            .ok_or(anyhow::anyhow!("xyz not found"))?;
        let xyz = xyz_data
            .into_vec::<F>()?
            .chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect::<Vec<_>>();

        let color_data = reader
            .by_name("color")?
            .ok_or(anyhow::anyhow!("color not found"))?;
        let color = color_data
            .into_vec::<F>()?
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect::<Vec<_>>();

        let scaling_data = reader
            .by_name("scaling")?
            .ok_or(anyhow::anyhow!("scaling not found"))?;
        let scaling = scaling_data
            .into_vec::<F>()?
            .chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect::<Vec<_>>();

        let rotation_data = reader
            .by_name("rotation")?
            .ok_or(anyhow::anyhow!("rotation not found"))?;
        let rotation = rotation_data.into_vec::<F>()?;

        let resolution_data = reader
            .by_name("resolution")?
            .ok_or(anyhow::anyhow!("resolution not found"))?;
        let resolution = resolution_data.into_vec::<u32>()?;
        let resolution = [resolution[0], resolution[1]];

        let gaussians = xyz
            .iter()
            .zip(color.iter())
            .zip(scaling.iter().zip(rotation.iter()))
            .map(|((mean, color), (scaling, rotation))| Gauss2D {
                mean: [mean[0].to_f32().unwrap(), mean[1].to_f32().unwrap()],
                color: [
                    color[0].to_f32().unwrap(),
                    color[1].to_f32().unwrap(),
                    color[2].to_f32().unwrap(),
                    1.0,
                ],
                rotation: rotation.to_f32().unwrap(),
                scale: [scaling[0].to_f32().unwrap(), scaling[1].to_f32().unwrap()],
            })
            .collect::<Vec<_>>();

        return Ok(Self {
            gaussians,
            resolution: [resolution[0] as u32, resolution[1] as u32],
        });
    }
}

#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum Channel {
    Color = 0,
    Dx = 1,
    Dy = 2,
    Dxy = 3,
}

impl ToString for Channel {
    fn to_string(&self) -> String {
        match self {
            Channel::Color => "Color".to_string(),
            Channel::Dx => "dx".to_string(),
            Channel::Dy => "dy".to_string(),
            Channel::Dxy => "dxy".to_string(),
        }
    }
}

unsafe impl bytemuck::Pod for Channel {}
unsafe impl bytemuck::Zeroable for Channel {}

#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum InterpolationMethod {
    Nearest = 0,
    Bilinear = 1,
    Bicubic = 2,
    Spline = 3,
}

impl ToString for InterpolationMethod {
    fn to_string(&self) -> String {
        match self {
            InterpolationMethod::Nearest => "Nearest".to_string(),
            InterpolationMethod::Bilinear => "Bilinear".to_string(),
            InterpolationMethod::Bicubic => "Bicubic".to_string(),
            InterpolationMethod::Spline => "Spline".to_string(),
        }
    }
}

unsafe impl bytemuck::Pod for InterpolationMethod {}
unsafe impl bytemuck::Zeroable for InterpolationMethod {}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RasterizationSettings {
    pub scaling: f32,
    pub channel: Channel,
    pub upscale_factor: f32,
    pub upscaling_method: InterpolationMethod,
    pub clamp_gradients: u32,
    pub clamp_image: u32,
    pub _padding: [u32; 2],
}

impl Default for RasterizationSettings {
    fn default() -> Self {
        Self {
            scaling: 1.0,
            channel: Channel::Color,
            upscale_factor: 1.,
            upscaling_method: InterpolationMethod::Spline,
            clamp_gradients: true as u32,
            clamp_image: true as u32,
            _padding: Default::default(),
        }
    }
}

pub struct FrameBuffer {
    pub color: wgpu::Texture,
    pub dx: wgpu::Texture,
    pub dy: wgpu::Texture,
    pub dxy: wgpu::Texture,
    pub bind_group: wgpu::BindGroup,
}

impl FrameBuffer {
    pub fn new(device: &wgpu::Device, resolution: [u32; 2]) -> Self {
        let create_texture = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: resolution[0],
                    height: resolution[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        let color = create_texture("FrameBuffer Color Texture");
        let dx = create_texture("FrameBuffer Dx Texture");
        let dy = create_texture("FrameBuffer Dy Texture");
        let dxy = create_texture("FrameBuffer Dxy Texture");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FrameBuffer Bind Group"),
            layout: &Self::bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &color.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &dx.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &dy.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &dxy.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        return Self {
            color,
            dx,
            dy,
            dxy,
            bind_group,
        };
    }

    pub fn resize(&mut self, device: &wgpu::Device, resolution: [u32; 2]) {
        let new = Self::new(device, resolution);
        self.color = new.color;
        self.dx = new.dx;
        self.dy = new.dy;
        self.dxy = new.dxy;
        self.bind_group = new.bind_group;
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FrameBuffer Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
    }
}
