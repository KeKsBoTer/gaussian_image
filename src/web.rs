use std::io::Cursor;

use eframe::WebOptions;
use wasm_bindgen::prelude::*;

use crate::{GaussianImageApp, gaussian::GaussianImage};

/// Your handle to the web app from JavaScript.
#[derive(Clone)]
#[wasm_bindgen]
pub struct WebHandle {
    runner: eframe::WebRunner,
}

#[wasm_bindgen]
impl WebHandle {
    /// Installs a panic hook, then returns.
    #[allow(clippy::new_without_default)]
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Redirect [`log`] message to `console.log` and friends:
        eframe::WebLogger::init(log::LevelFilter::Debug).ok();

        Self {
            runner: eframe::WebRunner::new(),
        }
    }

    /// Call this once from JavaScript to start your app.
    #[wasm_bindgen]
    pub async fn start(
        &self,
        canvas: web_sys::HtmlCanvasElement
    ) -> Result<(), wasm_bindgen::JsValue> {

        let window = web_sys::window().ok_or(JsError::new("cannot access window"))?;
        let search_string = window.location().search()?;
        let url =web_sys::UrlSearchParams::new_with_str(&search_string)?.get("file").ok_or(JsError::new("file parameter not found"))?;
        let request = ehttp::Request::get(url);
        let resp = ehttp::fetch_async(request).await?;
        if resp.status != 200 {
            return Err(JsError::new(format!("failed to load '{}'. {} ({})",resp.url,resp.status_text, resp.status).as_str()).into());
        }
        let reader = Cursor::new(resp.bytes);

        let gaussian_image = GaussianImage::from_npz(reader).map_err(|e| e.to_string())?;

        self.runner
            .start(
                canvas,
                WebOptions::default(),
                Box::new(|cc| Ok(Box::new(GaussianImageApp::new(cc, gaussian_image).unwrap()))),
            )
            .await
    }

    // The following are optional:

    /// Shut down eframe and clean up resources.
    #[wasm_bindgen]
    pub fn destroy(&self) {
        self.runner.destroy();
    }

    /// The JavaScript can check whether or not your app has crashed:
    #[wasm_bindgen]
    pub fn has_panicked(&self) -> bool {
        self.runner.has_panicked()
    }

    #[wasm_bindgen]
    pub fn panic_message(&self) -> Option<String> {
        self.runner.panic_summary().map(|s| s.message())
    }

    #[wasm_bindgen]
    pub fn panic_callstack(&self) -> Option<String> {
        self.runner.panic_summary().map(|s| s.callstack())
    }
}
