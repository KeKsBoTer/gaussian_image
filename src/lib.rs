#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Seek};
use std::sync::Arc;

use eframe::{egui, egui_wgpu};
use eframe::{
    egui::{Color32, Stroke},
    emath::Vec2,
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
    fn new(
        cc: &eframe::CreationContext<'_>,
        gaussian_image: GaussianImage,
    ) -> anyhow::Result<Self> {
        let wgpu_state = cc
            .wgpu_render_state
            .as_ref()
            .ok_or(anyhow::anyhow!("WGPU not available"))?;

        let device = &wgpu_state.device;

        let model = GaussianMixture::new(&device, &gaussian_image, gaussian_image.resolution);

        let rasterizer =
            gaussian::GaussianRasterizer::new(&device, wgpu_state.target_format, [800, 600]);
        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(rasterizer);

        Ok(Self {
            gaussians: Arc::new(model),
            settings: RasterizationSettings {
                scaling: 0.,
                ..Default::default()
            },
        })
    }
}

impl eframe::App for GaussianImageApp {

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.settings.scaling += (1. - self.settings.scaling) / 16.;
        if self.settings.scaling < 1. {
            ctx.request_repaint_after_secs(1. / 60.0);
        }
        egui::TopBottomPanel::top("top_panel")
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("ℹ")
                            .on_hover_cursor(egui::CursorIcon::Help)
                            .on_hover_ui(|ui| {
                                ui.heading("Gaussian Image Info");
                                egui::Grid::new("info grid").show(ui, |ui| {
                                    ui.label("Num Gaussians");
                                    ui.label(self.gaussians.len().to_string());
                                    ui.end_row();
                                    ui.label("Training Image Resolution");
                                    ui.label(format!(
                                        "{}x{}",
                                        self.gaussians.resolution[0], self.gaussians.resolution[1]
                                    ));
                                    ui.end_row();
                                });
                            });
                        ui.label("Gaussian Image");
                    });
                    // ui.separator();
                    ui.horizontal_wrapped(|ui| {
                        egui::ComboBox::new("method select", "Method")
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
                        ui.add(
                            egui::Slider::new(&mut self.settings.upscale_factor, (1.)..=(8.))
                                .text("Factor"),
                        );

                        #[cfg(debug_assertions)]
                        {
                            ui.separator();
                            let mut clamp_image = self.settings.clamp_image != 0;
                            ui.checkbox(&mut clamp_image, "Clamp Color")
                                .on_hover_text("Clamp color to [0,1] before interpolation");
                            self.settings.clamp_image = clamp_image as u32;
                        }
                        ui.separator();
                        if self.settings.upscaling_method == gaussian::InterpolationMethod::Spline {
                            #[cfg(debug_assertions)]
                            {
                                let mut clamp_gradients = self.settings.clamp_gradients != 0;
                                ui.checkbox(&mut clamp_gradients, "Clamp Gradients")
                                    .on_hover_text(
                                        "Clamp gradients to [-1,1] before interpolation",
                                    );
                                self.settings.clamp_gradients = clamp_gradients as u32;

                                ui.separator();
                            }
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
                                })
                                .response
                                .on_hover_text("Show channels used for spline interpolation");
                        }
                        ui.with_layout(
                            egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                            |ui| {
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.add(egui::Hyperlink::from_label_and_url(
                                            format!("{}", egui::special_emojis::GITHUB),
                                            "https://github.com/KeKsBoTer/gaussian_image",
                                        ).open_in_new_tab(true))
                                        .on_hover_text("Source Code");
                                        ui.separator();
                                    },
                                );
                            },
                        );
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.add(egui::Hyperlink::from_label_and_url(
                            format!("{}", egui::special_emojis::GITHUB),
                            "https://github.com/KeKsBoTer/gaussian_image",
                        ))
                        .on_hover_text("Source Code");
                    });
                });
            });

        if self.settings.upscaling_method != gaussian::InterpolationMethod::Spline {
            self.settings.channel = gaussian::Channel::Color;
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                egui::Frame::canvas(ui.style())
                    .fill(Color32::TRANSPARENT)
                    .stroke(Stroke::NONE)
                    .show(ui, |ui| {
                        self.render_gaussians(ui, self.settings);
                    });
            });
        });
    }
}

impl GaussianImageApp {
    fn render_gaussians(&mut self, ui: &mut egui::Ui, settings: RasterizationSettings) {
        let ratio = self.gaussians.resolution[0] as f32 / self.gaussians.resolution[1] as f32;
        let size = fit_rect(ui.available_size(), ratio);
        let (rect, _response) = ui.allocate_exact_size(size, egui::Sense::drag());

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
    use eframe::egui::ViewportBuilder;

    let gaussian_image = GaussianImage::from_npz(file).unwrap();
    let native_options = eframe::NativeOptions {
        viewport: ViewportBuilder::default().with_inner_size((
            gaussian_image.resolution[0] as f32,
            gaussian_image.resolution[1] as f32,
        )),
        window_builder: Some(Box::new(|b| {
            b.with_icon(Arc::new(
                eframe::icon_data::from_png_bytes(include_bytes!("../img/icon.png")).unwrap(),
            ))
        })),
        ..Default::default()
    };
    let ret = eframe::run_native(
        &format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
        native_options,
        Box::new(|cc| Ok(Box::new(GaussianImageApp::new(cc, gaussian_image).unwrap()))),
    );
    if let Err(e) = ret {
        log::error!("Error running viewer: {}", e);
    }
}

fn fit_rect(bounds: Vec2, aspect_ratio: f32) -> Vec2 {
    let ratio = bounds.x / bounds.y;
    if ratio > aspect_ratio {
        Vec2::new(bounds.y * aspect_ratio, bounds.y)
    } else {
        Vec2::new(bounds.x, bounds.x / aspect_ratio)
    }
}
