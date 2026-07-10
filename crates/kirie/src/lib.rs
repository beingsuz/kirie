//! kirie (切り絵) — a Wallpaper Engine compatible wallpaper renderer.
//!
//! This crate is both a library and the `kirie` binary. The library exposes:
//!
//! * the existing `info` / `extract` subcommands (SPEC.md §I) via
//!   [`info`] / [`extract`], and
//! * the drop-in `linux-wallpaperengine` compatibility surface via
//!   [`compat`] — the full C++ flag parser (docs/compat-cli.md), per-screen
//!   wallpaper dispatch, the control socket (docs/compat-socket.md, via
//!   `kirie-ipc`), and offscreen `--screenshot` capture.
//!
//! [`run`] is the single entry point the binary calls: it picks the
//! `info`/`extract` subcommands when `argv[1]` names one, and otherwise
//! parses the full compat flag surface exactly like the C++ engine
//! (docs/compat-cli.md §1: `linux-wallpaperengine [options...] [background]`).

#![forbid(unsafe_code)]

pub mod compat;
pub mod detect;
pub mod extract;
pub mod info;

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

/// The `info` / `extract` subcommand surface (SPEC.md §I). Kept separate from
/// the compat flag surface: only `argv[1] ∈ {info, extract}` reaches clap.
#[derive(Parser)]
#[command(
    name = "kirie",
    version,
    about = "kirie (切り絵) — Wallpaper Engine compatible wallpaper renderer for Linux"
)]
struct Cli {
    /// Subcommand to run.
    #[command(subcommand)]
    command: Command,
}

/// The kirie-native subcommands (SPEC.md §I).
#[derive(Subcommand)]
enum Command {
    /// Summarize a workshop item directory, project.json, scene.pkg, or .tex
    Info {
        /// Workshop item directory, project.json, scene.pkg, or .tex file
        path: PathBuf,
    },
    /// Extract a scene.pkg's entries, or decode a .tex to PNG(s)
    Extract {
        /// scene.pkg or .tex file to extract
        path: PathBuf,
        /// Output directory (created if missing)
        #[arg(short = 'o', long = "output", default_value = ".")]
        output: PathBuf,
        /// For a pkg input: also decode every contained .tex entry to
        /// PNG(s) next to the extracted file (video textures are skipped
        /// with a warning)
        #[arg(long)]
        tex_to_png: bool,
    },
}

/// Run the kirie CLI from a full argv (`args[0]` is the program name).
///
/// Dispatch (docs/compat-cli.md §1):
///
/// * no arguments → print the version line (pre-subcommand behavior kept so
///   `kirie` alone stays a harmless probe);
/// * `argv[1] ∈ {info, extract}` → the kirie-native subcommands;
/// * anything else → the `linux-wallpaperengine` compat surface
///   ([`compat::run`]), which owns its own exit code.
#[must_use]
pub fn run(args: Vec<OsString>) -> ExitCode {
    match args.get(1).map(|s| s.to_string_lossy()) {
        None => {
            // Bare `kirie`: keep the version probe (a real engine would error
            // with "At least one background ID must be specified", but the
            // daemon never invokes kirie without arguments).
            println!(concat!("kirie ", env!("CARGO_PKG_VERSION"), "-dev"));
            ExitCode::SUCCESS
        }
        Some(sub) if sub == "info" || sub == "extract" => run_subcommand(args),
        _ => compat::run(&args),
    }
}

/// Run the clap-driven `info` / `extract` subcommands.
fn run_subcommand(args: Vec<OsString>) -> ExitCode {
    let cli = Cli::parse_from(args);
    let result = match cli.command {
        Command::Info { path } => info::run(&path),
        Command::Extract {
            path,
            output,
            tex_to_png,
        } => extract::run(&path, &output, tex_to_png),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {}", render_chain(&err));
            ExitCode::FAILURE
        }
    }
}

/// Render an error chain like `anyhow`'s `{:#}` but without repeating causes:
/// the kirie-formats errors already include their source in their `Display`
/// (self-contained messages), so blindly appending every `source()` prints
/// e.g. `invalid JSON: expected value…: expected value…`.
pub(crate) fn render_chain(err: &anyhow::Error) -> String {
    let mut message = String::new();
    for cause in err.chain() {
        let text = cause.to_string();
        if message.is_empty() {
            message = text;
        } else if !message.ends_with(&text) {
            message.push_str(": ");
            message.push_str(&text);
        }
    }
    message
}
