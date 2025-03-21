use std::{
    io::{Read, Seek},
    sync::Arc,
};

use eframe::{egui, egui_wgpu};
use wgpu::{
    VertexAttribute,
    include_wgsl,
    util::DeviceExt,
};


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
    pub cov:[f32;4],
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
    fn vertices(&self) -> Option<[GaussianVertex; 6]> {
        let mu = self.mean;

        let scale_x = self.scale[0];
        let scale_y = self.scale[1];
        let rot: [[f32; 2]; 2] = [
            [f32::cos(self.rotation)*scale_x, f32::sin(self.rotation)*scale_y],
            [-f32::sin(self.rotation)*scale_x, f32::cos(self.rotation)*scale_y],
        ];


        let s = 2.*f32::ln(255.).sqrt();
        let offset = [
            transform2d([-s, -s], rot),
            transform2d([-s, s], rot),
            transform2d([s, -s], rot),
            transform2d([s, s], rot),
        ];
        let pos = [
            vec_add(mu, offset[0]),
            vec_add(mu, offset[1]),
            vec_add(mu, offset[2]),
            vec_add(mu, offset[3]),
        ];
        let cov = build_covariance_matrix(self.rotation,[scale_x,scale_y]);
        let cov_inv = invert(cov);
        if cov_inv.is_none(){
            return None;
        }
        let cov_inv = cov_inv.unwrap();
        let cov_plain = [cov_inv[0][0],cov_inv[0][1],cov_inv[1][0],cov_inv[1][1]];
        Some([
            GaussianVertex {
                position: pos[0],
                color: self.color,
                offset: offset[0],
                cov:cov_plain,
            },
            GaussianVertex {
                position: pos[2],
                color: self.color,
                offset: offset[2],
                cov:cov_plain,
            },
            GaussianVertex {
                position: pos[1],
                color: self.color,
                offset: offset[1],
                cov:cov_plain,
            },
            GaussianVertex {
                position: pos[1],
                color: self.color,
                offset: offset[1],
                cov:cov_plain,
            },
            GaussianVertex {
                position: pos[2],
                color: self.color,
                offset: offset[2],
                cov:cov_plain,
            },
            GaussianVertex {
                position: pos[3],
                color: self.color,
                offset: offset[3],
                cov:cov_plain,
            },
        ])
    }
}

pub struct GaussianMixture {
    gaussians: Vec<Gauss2D>,
    num_vertices: usize,
    buffer: wgpu::Buffer,
    pub resolution: [u32;2],
}

impl GaussianMixture {
    pub fn new(device: &wgpu::Device, gaussians: Vec<Gauss2D>,resolution:[u32;2]) -> Self {
        let vertices: Vec<GaussianVertex> = gaussians
            .iter()
            .filter_map(|g| g.vertices())
            .flat_map(|v| v)
            .collect::<Vec<_>>();
        let num_vertices = vertices.len();

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian Mixture Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self { gaussians, buffer, resolution,num_vertices}
    }

    pub fn from_npz<R: Read + Seek>(device: &wgpu::Device, file: R) -> anyhow::Result<Self> {
        let mut reader = npyz::npz::NpzArchive::new(file)?;
        let xyz_data = reader
            .by_name("xyz")?
            .ok_or(anyhow::anyhow!("xyz not found"))?;
        let xyz = xyz_data
            .into_vec::<f32>()?
            .chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect::<Vec<_>>();

        let color_data = reader
            .by_name("color")?
            .ok_or(anyhow::anyhow!("color not found"))?;
        let color = color_data
            .into_vec::<f32>()?
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect::<Vec<_>>();

        let scaling_data = reader
            .by_name("scaling")?
            .ok_or(anyhow::anyhow!("scaling not found"))?;
        let scaling = scaling_data
            .into_vec::<f32>()?
            .chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect::<Vec<_>>();

        let rotation_data = reader
            .by_name("rotation")?
            .ok_or(anyhow::anyhow!("rotation not found"))?;
        let rotation = rotation_data
            .into_vec::<f32>()?;


        let resolution_data = reader
            .by_name("resolution")?
            .ok_or(anyhow::anyhow!("resolution not found"))?;
        let resolution = resolution_data
            .into_vec::<u32>()?;
        let resolution = [resolution[0],resolution[1]];

        let gaussians = xyz
            .iter()
            .zip(color.iter())
            .zip(scaling.iter().zip(rotation.iter()))
            .map(|((mean, color), (scaling, rotation))| Gauss2D {
                mean: [(mean[0]+1.)/2.* resolution[0] as f32, (mean[1]+1.)/2.* resolution[1] as f32],
                color: [color[0], color[1], color[2], 1.0],
                rotation:*rotation,
                scale:[scaling[0],scaling[1]],
            })
            .collect::<Vec<_>>();


        return Ok(Self::new(device, gaussians,resolution));
    }

    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }
}

#[derive(Clone)]
pub struct GaussianMixtureCallBack {
    pub gaussians: Arc<GaussianMixture>,
}

impl egui_wgpu::CallbackTrait for GaussianMixtureCallBack {
    fn prepare(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gaussian Rasterizer Command Encoder"),
        });
        let renderer: &mut GaussianRasterizer = resources.get_mut().unwrap();
        
        renderer.prepare(device,&mut encoder, self.gaussians.as_ref(),screen_descriptor.size_in_pixels);
        vec![encoder.finish()]
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &GaussianRasterizer = resources.get().unwrap();
        resources.render(render_pass);
    }
}

pub struct GaussianRasterizer {
    pipeline: wgpu::RenderPipeline,
    display_pipeline: wgpu::RenderPipeline,
    display_texture: wgpu::Texture,
    display_bg: wgpu::BindGroup,
}

impl GaussianRasterizer {
    pub fn new(device: &wgpu::Device, render_format: wgpu::TextureFormat,resolution:[u32;2]) -> Self {
        let shader_module = device.create_shader_module(include_wgsl!("shaders/gaussian.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gaussian Rasterizer Pipeline Layout"),
            bind_group_layouts: &[],
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    // blend:None,
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
        let display_shader_module = device.create_shader_module(include_wgsl!("shaders/display.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gaussian Display Pipeline Layout"),
            bind_group_layouts: &[&Self::display_bind_group_layout(device)],
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
        let (display_texture,display_bg) = Self::create_display_texture(device, resolution);
        Self { pipeline,display_pipeline,display_texture,display_bg }
    }

    fn display_bind_group_layout(device: &wgpu::Device)->wgpu::BindGroupLayout{
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gaussian Display Bind Group Layout"),
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
            ]
        })
    }

    fn create_display_texture(device: &wgpu::Device, resolution: [u32; 2]) -> (wgpu::Texture ,wgpu::BindGroup) {
        let display_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gaussian Display Texture"),
            size: wgpu::Extent3d {
                width: resolution[0],
                height: resolution[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[]
        });
        let display_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gaussian Display Bind Group"),
            layout: &Self::display_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&display_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                },
            ]
        });
        (display_texture,display_bg)
    }

    pub fn resize(&mut self, device: &wgpu::Device, resolution: [u32; 2]) {
        let (display_texture,display_bg) = Self::create_display_texture(device, resolution);
        self.display_texture = display_texture;
        self.display_bg = display_bg;
    }

    pub fn prepare(&mut self,device: &wgpu::Device,encoder:&mut wgpu::CommandEncoder, gaussians: &GaussianMixture,resolution: [u32; 2]) {

        let tex_size = [self.display_texture.size().width, self.display_texture.size().height];
        if tex_size != resolution{
            self.resize(device, resolution);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Gaussian Rasterizer Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.display_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            }).forget_lifetime();

            render_pass.set_vertex_buffer(0, gaussians.buffer.slice(..));
            render_pass.set_pipeline(&self.pipeline);
            render_pass.draw(0..(gaussians.num_vertices()) as u32, 0..1);
        }
    }
    pub fn render(&self, render_pass: &mut wgpu::RenderPass<'static>) {
        render_pass.set_pipeline(&self.display_pipeline);
        render_pass.set_bind_group(0, &self.display_bg, &[]);
        render_pass.draw(0..4 as u32, 0..1);
    }
}

fn vec_length(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}
fn vec_normalize(v: [f32; 2]) -> [f32; 2] {
    let l = vec_length(v);
    [v[0] / l, v[1] / l]
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

fn matmul(a: [[f32;2];2], b: [[f32;2];2]) -> [[f32;2];2] {
    let mut res = [[0.,0.],[0.,0.]];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}
fn transpose(a: [[f32;2];2]) -> [[f32;2];2] {
    let mut res = [[0.,0.],[0.,0.]];
    for i in 0..2 {
        for j in 0..2 {
            res[i][j] = a[j][i];
        }
    }
    return res;
}

fn build_covariance_matrix(rotation: f32, scale: [f32; 2]) -> [[f32; 2];2] {
    let rot = [
        [f32::cos(rotation), -f32::sin(rotation)],
        [f32::sin(rotation), f32::cos(rotation)],
    ];
    let scale = [[scale[0], 0.], [0., scale[1]]];
    let l = matmul(rot, scale);
    let cov = matmul(l, transpose(l));
    [[cov[0][0], cov[0][1]],[ cov[1][0], cov[1][1]]]
}

fn invert(mat: [[f32; 2]; 2]) -> Option<[[f32; 2]; 2]> {
    let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    if det.abs() <1e-6{
        return None;
    }
    let inv_det = 1.0 / det;
    Some([[mat[1][1] * inv_det, -mat[0][1] * inv_det], [-mat[1][0] * inv_det, mat[0][0] * inv_det]])
}
