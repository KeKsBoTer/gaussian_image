export RUSTFLAGS=--cfg=web_sys_unstable_apis 
cargo build \
    --no-default-features \
    --target wasm32-unknown-unknown \
    --profile web-release \
    --lib \
&& wasm-bindgen \
    --out-dir public \
    --web target/wasm32-unknown-unknown/web-release/gauss_img.wasm \
    --no-typescript     