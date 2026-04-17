use clap::Parser;
pub use third_harm_gen::{
    error::{Error, Result},
    run,
};

/// 3rd harmonic generation
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// Verbosity
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    run(args.verbose)
}
