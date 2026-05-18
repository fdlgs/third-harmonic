use clap::Parser;
use color_eyre::eyre::Result;
use strum::IntoEnumIterator;
pub use third_harmonic::{AlphaSquare, run};

/// Photon population evolution for 3rd harmonic generation
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// Verbosity
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
    /// Must be one of [10, 100, 1000, 2000]
    #[arg(short, long, default_value_t = 10)]
    alpha_square: u16,
}

fn main() -> Result<()> {
    let args = Args::parse();

    AlphaSquare::iter()
        .find(|&e_alpha_square| e_alpha_square as u16 == args.alpha_square)
        .map_or_else(
            || {
                println!(
                    "Oups!  -a, --alpha_square <ALPHA_SQUARE>: <ALPHA_SQUARE> must be one of {:?}",
                    AlphaSquare::iter()
                        .map(|e_alpha_square| e_alpha_square as u16)
                        .collect::<Vec<_>>()
                );
                Ok(())
            },
            |e_alpha_square| run(e_alpha_square, args.verbose),
        )
}
