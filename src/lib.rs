use std::{
    io::{Read, Seek},
    sync::Arc,
};

use eframe::{
    egui::{self},
    egui_wgpu,
};
use gaussian::{Gauss2D, GaussianMixture, GaussianMixtureCallBack};
use rand::{Rng, SeedableRng};
mod gaussian;

struct MyEguiApp {
    gaussians: Arc<GaussianMixture>,
}

impl MyEguiApp {
    fn new<R: Read + Seek>(cc: &eframe::CreationContext<'_>, file: R) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.

        let wgpu_state = cc.wgpu_render_state.as_ref().unwrap();

        let device = &wgpu_state.device;

        let model = GaussianMixture::from_npz(&device, file).unwrap();

        let rasterizer =
            gaussian::GaussianRasterizer::new(&device, wgpu_state.target_format, [800, 600]);
        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(rasterizer);

        Self {
            gaussians: Arc::new(model),
        }
    }
}

impl eframe::App for MyEguiApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
                egui::Frame::canvas(ui.style()).show(ui, |ui| {
                self.render_gaussians(ui);
            });
        });
    }
}

impl MyEguiApp {
    fn render_gaussians(&mut self, ui: &mut egui::Ui) {
        let (rect, _response) = ui.allocate_exact_size(ui.available_size(),
            egui::Sense::drag(),
        );

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            GaussianMixtureCallBack {
                gaussians: self.gaussians.clone(),
            },
        ));
        
    }
}

pub async fn start<R: Read + Seek>(file: R) {
    let native_options = eframe::NativeOptions::default();

    eframe::run_native(
        &format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
        native_options,
        Box::new(|cc| Ok(Box::new(MyEguiApp::new(cc, file)))),
    )
    .unwrap();
}
