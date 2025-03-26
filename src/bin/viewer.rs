use std::{fs::File, path::PathBuf};

use clap::{command, Parser};
use gauss_img::start;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,
}


#[pollster::main]
async fn main() {
    let opt = Opt::parse();
    env_logger::Builder::from_default_env().filter_module("gauss_img",log::LevelFilter::Warn).init();
    log::warn!("{}",env!("CARGO_PKG_NAME"));
    
    let file = File::open(opt.input).expect("Failed to open file");
    let buff_reader = std::io::BufReader::new(file);
    start(buff_reader).await;
}
