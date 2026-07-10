//! Render an image wallpaper (plain file or `.tex`) on every output as a
//! background layer-shell surface for `--seconds N` (default 3), then exit
//! cleanly.
//!
//! ```sh
//! cargo run -p kirie-render --example image_wall -- \
//!     /path/to/preview.jpg --seconds 3 --scaling fill --clamp clamp
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use kirie_platform::Platform;
use kirie_render::{ClampMode, ImageContent, ImageOptions, ImageRenderer, ScalingMode};

struct Args {
    path: PathBuf,
    seconds: u64,
    options: ImageOptions,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut path: Option<PathBuf> = None;
    let mut seconds = 3u64;
    let mut options = ImageOptions::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seconds" => {
                let value = args.next().ok_or("--seconds requires a value")?;
                seconds = value
                    .parse::<u64>()
                    .map_err(|err| format!("invalid --seconds value {value:?}: {err}"))?;
            }
            "--scaling" => {
                let value = args.next().ok_or("--scaling requires a value")?;
                options.scaling = ScalingMode::from_cli(&value).map_err(|err| err.to_string())?;
            }
            "--clamp" => {
                let value = args.next().ok_or("--clamp requires a value")?;
                options.clamp = ClampMode::from_cli(&value).map_err(|err| err.to_string())?;
            }
            other if path.is_none() && !other.starts_with("--") => {
                path = Some(PathBuf::from(other));
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    Ok(Args {
        path: path.ok_or("usage: image_wall <image|.tex> [--seconds N] [--scaling M] [--clamp M]")?,
        seconds,
        options,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = parse_args()?;

    let content = Arc::new(ImageContent::from_path(&args.path)?);
    tracing::info!(
        path = %args.path.display(),
        pages = content.pages.len(),
        frames = content.frames.len(),
        content_width = content.content_width,
        content_height = content.content_height,
        animated = content.schedule().is_animated(),
        scaling = kirie_render::ScalingMode::as_cli_str(args.options.scaling),
        clamp = kirie_render::ClampMode::as_cli_str(args.options.clamp),
        "content loaded"
    );

    let options = args.options;
    let mut platform = Platform::connect(Box::new(move |target| {
        tracing::info!(output = %target.output_name, "creating image renderer");
        let renderer = ImageRenderer::new(target, &content, options).expect("image renderer construction");
        if let Some(next) = renderer.time_until_frame_change() {
            tracing::info!(?next, "animated content; first frame change");
        } else {
            tracing::info!("static content; no further frames needed after the first");
        }
        Box::new(renderer)
    }))?;

    platform.run(Some(Duration::from_secs(args.seconds)))?;

    tracing::info!(outputs = platform.output_count(), "clean exit");
    Ok(())
}
