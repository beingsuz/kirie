//! The `linux-wallpaperengine` compatibility surface (docs/compat-cli.md).
//!
//! [`run`] is the compat entry point: it parses the full C++ flag surface
//! ([`args`]), handles `--help`/`--list-properties`/`--screenshot` exit modes,
//! prints the `Running with:` banner (doc §1.2), and otherwise dispatches to
//! per-screen wallpaper rendering with the control socket wired in ([`run`]
//! module, [`ipc_app`], [`screenshot`]).

pub mod args;
pub mod ipc_app;
pub mod list_props;
pub mod resolve;
pub mod run;
pub mod screenshot;
pub mod signals;

use std::ffi::OsString;
use std::process::ExitCode;
use std::sync::Once;

use args::ParseError;

/// Run the compat surface for a full argv (`argv[0]` is the program name).
///
/// Exit codes follow doc §5: `0` for `--help` and successful runs/clean stops,
/// `1` for any parse/startup fatal or abnormal termination.
pub fn run(argv: &[OsString]) -> ExitCode {
    init_tracing();
    let argv0 = argv
        .first()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "linux-wallpaperengine".to_owned());

    let parsed = match args::parse(argv) {
        Ok(a) => a,
        Err(e) => return fail(&argv0, &e),
    };

    // `--help`: print the synopsis and exit 0 before validation/banner
    // (doc §5, main.cpp:68-71).
    if parsed.help {
        print!("{}", args::HELP_TEXT);
        return ExitCode::SUCCESS;
    }

    let validated = match args::validate(parsed) {
        Ok(a) => a,
        Err(e) => return fail(&argv0, &e),
    };

    // The `Running with:` banner always prints on a successful parse
    // (doc §1.2, §4.8 step 4) — before running, and even for the list modes.
    // It must precede the screenshot-extension check (§4.8 step 6) so that a
    // bad `--screenshot` extension still emits the banner on stdout first,
    // reproducing the C++ post-parse validation order exactly.
    print_banner(&validated);

    // Screenshot extension validation (doc §3.6, §4.8 step 6 — after the
    // banner).
    if let Some(path) = &validated.screenshot
        && let Err(e) = args::validate_screenshot_ext(path.as_os_str())
    {
        return fail(&argv0, &e);
    }

    run::dispatch(validated)
}

/// Print a fatal parse error with the doc §4.7 doubling: the bare message,
/// then (for `sLog.exception` fatals) the message again with the
/// `. Use <argv0> --help for more information` suffix. Returns exit 1 (doc §5).
fn fail(argv0: &str, err: &ParseError) -> ExitCode {
    eprintln!("{}", err.message);
    if err.doubled {
        eprintln!("{}. Use {argv0} --help for more information", err.message);
    }
    ExitCode::FAILURE
}

/// Print the `Running with: <argv...> ` banner to stdout (doc §1.2): every
/// argv element space-separated with a trailing space, then a newline.
fn print_banner(args: &args::CompatArgs) {
    let mut line = String::from("Running with: ");
    for a in &args.argv {
        line.push_str(a);
        line.push(' ');
    }
    println!("{line}");
}

/// Initialize a stderr tracing subscriber once (best-effort). The C++ engine
/// logs to stderr; kirie routes kirie-video/platform diagnostics the same way.
fn init_tracing() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(std::io::stderr)
            .try_init();
    });
}
