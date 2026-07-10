//! Play a video wallpaper on every output as a background layer-shell
//! surface for `--seconds N` (default 4), then exit cleanly. Live-verify
//! harness for SPEC T10, patterned after kirie-platform's `layer_clear`.
//!
//! ```sh
//! cargo run -p kirie-video --example video_layer -- \
//!     --path ~/.steam/steam/steamapps/workshop/content/431960/3600453929/*.mp4 \
//!     --seconds 4 --scaling fill --volume 0
//! ```
//!
//! Defaults: the corpus video (docs/subsystems-misc.md header, item
//! 3600453929), 4 seconds, `fill`, volume 0 (audio pipeline still runs so
//! the audio-master clock is exercised — `--silent` semantics,
//! docs/subsystems-misc.md §2.3).

use std::path::PathBuf;
use std::time::Duration;

use kirie_platform::Platform;
use kirie_video::{ScalingMode, VideoOptions, VideoPlayer, VideoRenderer};

/// Corpus fallback (SPEC §C corpus; the one `"type":"video"` item).
const CORPUS_ITEM: &str = ".steam/steam/steamapps/workshop/content/431960/3600453929";

struct Args {
    seconds: u64,
    path: PathBuf,
    scaling: ScalingMode,
    volume: f64,
    mute: bool,
}

fn corpus_video() -> Option<PathBuf> {
    let dir = std::env::home_dir()?.join(CORPUS_ITEM);
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("mp4")))
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut seconds = 4u64;
    let mut path: Option<PathBuf> = None;
    let mut scaling = ScalingMode::Fill;
    let mut volume = 0.0f64;
    let mut mute = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seconds" => {
                let v = args.next().ok_or("--seconds requires a value")?;
                seconds = v.parse().map_err(|e| format!("invalid --seconds {v:?}: {e}"))?;
            }
            "--path" => path = Some(PathBuf::from(args.next().ok_or("--path requires a value")?)),
            "--scaling" => {
                let v = args.next().ok_or("--scaling requires a value")?;
                scaling = ScalingMode::from_cli(&v).ok_or(format!("unknown scaling mode {v:?}"))?;
            }
            "--volume" => {
                let v = args.next().ok_or("--volume requires a value")?;
                volume = v.parse().map_err(|e| format!("invalid --volume {v:?}: {e}"))?;
            }
            "--mute" => mute = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    let path = match path.or_else(corpus_video) {
        Some(p) => p,
        None => return Err("no --path given and corpus video not installed".into()),
    };
    Ok(Args {
        seconds,
        path,
        scaling,
        volume,
        mute,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = parse_args()?;
    tracing::info!(path = %args.path.display(), seconds = args.seconds, "starting video wallpaper");

    let mut platform = Platform::connect(Box::new(move |target| {
        let options = VideoOptions {
            volume: args.volume,
            mute: args.mute,
            // Volume 0 ≙ --silent playback; keep the audio pipeline (and
            // therefore the audio-master clock) alive regardless.
            silent: args.volume <= 0.0,
            scaling: args.scaling,
            ..VideoOptions::default()
        };
        match VideoPlayer::open(&args.path, options) {
            Ok((player, _control)) => {
                let info = player.info();
                tracing::info!(
                    output = %target.output_name,
                    width = info.width,
                    height = info.height,
                    fps = format!("{:.3}", info.frame_rate),
                    duration = format!("{:.2}s", info.duration),
                    audio = player.has_audio(),
                    "video player ready"
                );
                Box::new(VideoRenderer::new(target, player))
            }
            Err(err) => {
                tracing::error!(%err, "failed to open video; falling back to test pattern");
                Box::new(kirie_platform::TestPattern::new(target))
            }
        }
    }))?;

    platform.run(Some(Duration::from_secs(args.seconds)))?;
    tracing::info!(outputs = platform.output_count(), "clean exit");
    Ok(())
}
