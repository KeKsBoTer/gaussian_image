use std::{
    io::{Read, Seek},
    sync::Arc,
};

use eframe::{
    egui::{self, ViewportBuilder},
    egui_wgpu,
};
use gaussian::{GaussianImage, GaussianMixture, GaussianMixtureCallBack, RasterizationSettings};
mod gaussian;

#[cfg(target_arch = "wasm32")]
pub mod web;

struct GaussianImageApp {
    gaussians: Arc<GaussianMixture>,
    settings: RasterizationSettings,
}

impl GaussianImageApp {
    fn new(cc: &eframe::CreationContext<'_>, gaussian_image: GaussianImage) -> Self {
        let wgpu_state = cc.wgpu_render_state.as_ref().unwrap();

        let device = &wgpu_state.device;

        let model = GaussianMixture::new(&device, &gaussian_image);

        let rasterizer =
            gaussian::GaussianRasterizer::new(&device, wgpu_state.target_format, [800, 600]);
        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(rasterizer);

        Self {
            gaussians: Arc::new(model),
            settings: RasterizationSettings::default(),
        }
    }
}

impl eframe::App for GaussianImageApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut self.settings.scaling, (0.1)..=(1.)).text("Gaussian Scaling"));

                egui::ComboBox::new("method select", "Upscaling Method")
                    .selected_text(self.settings.upscaling_method.to_string())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.settings.upscaling_method,
                            gaussian::InterpolationMethod::Nearest,
                            gaussian::InterpolationMethod::Nearest.to_string(),
                        );
                        ui.selectable_value(
                            &mut self.settings.upscaling_method,
                            gaussian::InterpolationMethod::Bilinear,
                            gaussian::InterpolationMethod::Bilinear.to_string(),
                        );
                        ui.selectable_value(
                            &mut self.settings.upscaling_method,
                            gaussian::InterpolationMethod::Bicubic,
                            gaussian::InterpolationMethod::Bicubic.to_string(),
                        );
                        ui.selectable_value(
                            &mut self.settings.upscaling_method,
                            gaussian::InterpolationMethod::Spline,
                            gaussian::InterpolationMethod::Spline.to_string(),
                        );
                    });

                ui.separator();
                ui.add(egui::Slider::new(
                    &mut self.settings.upscale_factor,
                    1..=(16),
                ));

                ui.separator();
                let mut clamp_image = self.settings.clamp_image != 0;
                ui.checkbox(&mut clamp_image, "Clamp Color")
                    .on_hover_text("Clamp color to [0,1] before interpolation");
                self.settings.clamp_image = clamp_image as u32;

                ui.separator();
                if self.settings.upscaling_method == gaussian::InterpolationMethod::Spline {
                    let mut clamp_gradients = self.settings.clamp_gradients != 0;
                    ui.checkbox(&mut clamp_gradients, "Clamp Gradients")
                        .on_hover_text("Clamp gradients to [-1,1] before interpolation");
                    self.settings.clamp_gradients = clamp_gradients as u32;

                    ui.separator();
                    egui::ComboBox::new("channel select", "Channel")
                        .selected_text(self.settings.channel.to_string())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.settings.channel,
                                gaussian::Channel::Color,
                                gaussian::Channel::Color.to_string(),
                            );
                            ui.selectable_value(
                                &mut self.settings.channel,
                                gaussian::Channel::Dx,
                                gaussian::Channel::Dx.to_string(),
                            );
                            ui.selectable_value(
                                &mut self.settings.channel,
                                gaussian::Channel::Dy,
                                gaussian::Channel::Dy.to_string(),
                            );
                            ui.selectable_value(
                                &mut self.settings.channel,
                                gaussian::Channel::Dxy,
                                gaussian::Channel::Dxy.to_string(),
                            );
                        });
                }
            });
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                self.render_gaussians(ui, self.settings);
            });
        });
    }
}

impl GaussianImageApp {
    fn render_gaussians(&mut self, ui: &mut egui::Ui, settings: RasterizationSettings) {
        let (rect, _response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            GaussianMixtureCallBack {
                gaussians: self.gaussians.clone(),
                settings: settings,
            },
        ));
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn start<R: Read + Seek>(file: R) {
    let gaussian_image = GaussianImage::from_npz(file).unwrap();
    let native_options = eframe::NativeOptions {
        viewport: ViewportBuilder::default().with_inner_size((
            gaussian_image.resolution[0] as f32,
            gaussian_image.resolution[1] as f32,
        )),
        ..Default::default()
    };
    eframe::run_native(
        &format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
        native_options,
        Box::new(|cc| Ok(Box::new(GaussianImageApp::new(cc, gaussian_image)))),
    )
    .unwrap();
}
