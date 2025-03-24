use std::{fs::File, path::PathBuf};

use clap::{command, Parser};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,
}


#[pollster::main]
async fn main() {
    let opt = Opt::parse();
    let file = File::open(opt.input).unwrap();
    let buff_reader = std::io::BufReader::new(file);
    start(buff_reader).await;
}
