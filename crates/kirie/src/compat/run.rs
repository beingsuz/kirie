//! Run-mode dispatch for the compat surface (docs/compat-cli.md §3.3): pick
//! the exit-early modes (`--list-properties*`, `--screenshot`) or run the
//! per-screen wallpapers on the wayland presentation layer with the control
//! socket wired in.
//!
//! Per-screen dispatch by resolved type (task scope): video → kirie-video,
//! image/gif/`.tex` file → kirie-render, scene → kirie-render scene renderer,
//! web → the kirie-web CEF backend **when built with `--features web-cef`**
//! (a [`WebRenderer`] blitting CEF's off-screen frames through the presentation
//! layer). An application-type item is launch-fatal with the reference's exact
//! refusal ("Application wallpapers are not supported on this platform",
//! WallpaperParser.cpp:22-24 — the C++ exception escapes the startup
//! `loadBackgrounds` and kills the whole launch). Any other background that
//! cannot run on this build — e.g. a web item in a binary without the CEF
//! backend — gets a clean per-screen message + nonzero exit *unless* another
//! screen can run (doc §3.1: unconfigured/unsupported screens do not sink the
//! whole launch when a sibling is renderable).
//!
//! # Web feature variants
//!
//! * **`web-cef`** — the Chromium Embedded Framework off-screen backend. It is
//!   a [`kirie_platform::Renderer`] (it uploads CEF's BGRA frames to a texture
//!   and blits), so it slots straight into the wgpu presentation layer here and
//!   into `--screenshot`.
//! * **`web-webview`** — the wry/webkit2gtk backend renders into its *own* GTK
//!   window, not a wgpu surface: webkit2gtk has no off-screen/pixel-readback
//!   path, so it can never composite through this presentation layer (upstream
//!   wry/webkit2gtk limitation — won't-fix; see `kirie-web/src/webview/mod.rs`
//!   for the API-level evidence). A `web-webview`-only binary therefore reports
//!   web wallpapers as unrunnable at dispatch time and points at `web-cef`.
//! * **Both enabled** — the CEF backend is preferred (it is the one that
//!   composites through this layer).
//! * **Neither** — the default build: web wallpapers report a clean message
//!   naming the two features to rebuild with.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kirie_audio::{AudioCapture, AudioConfig, AutoMute};
use kirie_platform::{CommandSender, Platform, RenderCommand, RenderTarget, Renderer, SurfaceSize};
use kirie_render::{ImageContent, ImageOptions, ImageRenderer};
use kirie_video::{VideoControl, VideoOptions, VideoPlayer, VideoRenderer};

use crate::compat::args::{ClampMode, CompatArgs, ScalingMode, WindowMode};
use crate::compat::ipc_app::{IpcApp, Register};
use crate::compat::playlist::{ActivePlaylist, PlaylistDefinition, Rng};
use crate::compat::resolve::{self, ClassifyError, Wallpaper};
use crate::compat::{list_props, screenshot, signals};

#[cfg(feature = "web-cef")]
use kirie_web::{WebBackend, WebRenderer, WebSize, cef::CefBackend};

/// Map the compat scaling enum to kirie-video's (doc §3.1 value table).
#[must_use]
pub fn to_video_scaling(mode: ScalingMode) -> kirie_video::ScalingMode {
    match mode {
        ScalingMode::Default => kirie_video::ScalingMode::Default,
        ScalingMode::Fit => kirie_video::ScalingMode::Fit,
        ScalingMode::Fill => kirie_video::ScalingMode::Fill,
        ScalingMode::Stretch => kirie_video::ScalingMode::Stretch,
    }
}

/// Map the compat scaling enum to kirie-render's (doc §3.1 value table).
#[must_use]
pub fn to_render_scaling(mode: ScalingMode) -> kirie_render::ScalingMode {
    match mode {
        ScalingMode::Default => kirie_render::ScalingMode::Default,
        ScalingMode::Fit => kirie_render::ScalingMode::Fit,
        ScalingMode::Fill => kirie_render::ScalingMode::Fill,
        ScalingMode::Stretch => kirie_render::ScalingMode::Stretch,
    }
}

/// Build the audio-capture config from the parsed CLI (doc §2).
///
/// `--no-audio-processing` disables the reactive capture entirely (permanent
/// silent spectrum, no threads — cpp `settings.audio.audioprocessing`).
/// `--audio-device` selects the PulseAudio *source* (`None` = default sink
/// monitor). `--silent` mutes the *wallpaper's own audio output* (video path),
/// not the system-audio reactive input, so it does not gate capture here.
#[must_use]
pub fn audio_config(args: &CompatArgs) -> AudioConfig {
    if args.no_audio_processing {
        AudioConfig::disabled()
    } else {
        AudioConfig::with_device(args.audio_device.clone())
    }
}

/// Map the compat clamp enum to kirie-render's (doc §3.1 value table).
#[must_use]
pub fn to_render_clamp(mode: ClampMode) -> kirie_render::ClampMode {
    match mode {
        ClampMode::Clamp => kirie_render::ClampMode::Clamp,
        ClampMode::Border => kirie_render::ClampMode::Border,
        ClampMode::Repeat => kirie_render::ClampMode::Repeat,
    }
}

/// Dispatch the validated args to the right run mode (doc §3.3, §3.6, §3.8).
pub fn dispatch(args: CompatArgs) -> ExitCode {
    // `-l`/`--list-properties-json` load, print, and exit (doc §3.8).
    if args.list_properties || args.list_properties_json {
        return match list_props::run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        };
    }

    // `--screenshot`: kirie captures one frame offscreen and exits (task
    // scope; the C++ engine keeps running, doc §3.6 — kirie's offscreen shot
    // exists to unlock the P4 SSIM gate).
    if let Some(path) = args.screenshot.clone() {
        return run_screenshot(&args, &path);
    }

    run_wallpapers(args)
}

/// Offscreen `--screenshot` capture of the default background (doc §3.6).
fn run_screenshot(args: &CompatArgs, path: &Path) -> ExitCode {
    let Some(bg) = args.default_background.clone() else {
        eprintln!("At least one background ID must be specified");
        return ExitCode::FAILURE;
    };
    let wallpaper = match resolve::classify(&bg) {
        Ok(w) => w,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };
    // Screenshot uses the window/global scaling+clamp (doc §3.1: screenshot is
    // a single composited background, no per-screen viewport here).
    // A short-lived capture for reactive scenes: it connects off-thread and
    // publishes lock-free, so the shot reflects whatever audio is playing by the
    // time the delay frames elapse (silent/zero if none). Disabled by
    // `--no-audio-processing` (doc §2).
    let audio = Arc::new(AudioCapture::start(audio_config(args)));
    match screenshot::capture(
        &wallpaper,
        args.window_scaling,
        args.window_clamp,
        args.screenshot_delay,
        path,
        Some(audio),
        &args.set_properties,
    ) {
        Ok(()) => {
            tracing::info!(path = %path.display(), "screenshot written");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("screenshot failed: {err:#}");
            ExitCode::FAILURE
        }
    }
}

/// What to build for one screen once its background is classified.
enum RunSpec {
    /// Video wallpaper (kirie-video).
    Video {
        /// Media file path.
        media: PathBuf,
        /// Output scaling mode for this screen.
        scaling: ScalingMode,
    },
    /// Image/gif/`.tex` wallpaper (kirie-render).
    Image {
        /// Content file path.
        file: PathBuf,
        /// Output scaling mode for this screen.
        scaling: ScalingMode,
        /// UV clamp mode for this screen.
        clamp: ClampMode,
    },
    /// Scene wallpaper (kirie-render scene renderer).
    Scene {
        /// Workshop item directory (`scene.pkg` + `project.json`).
        dir: PathBuf,
        /// Output scaling mode for this screen.
        scaling: ScalingMode,
        /// UV clamp mode for this screen.
        clamp: ClampMode,
    },
    /// Web wallpaper (kirie-web CEF off-screen backend). Only constructed in a
    /// `web-cef` build — the backend is a [`kirie_platform::Renderer`] that
    /// blits CEF's frames, so no scaling/clamp is needed (the page fills the
    /// surface).
    #[cfg(feature = "web-cef")]
    Web {
        /// `file://` (or `http(s)://`) URL of the wallpaper's entry page.
        url: String,
        /// The wallpaper directory (its `project.json` declares the typed user
        /// properties delivered to the page as `applyUserProperties`, doc §3.5).
        dir: PathBuf,
    },
    /// Not runnable (unsupported type or a load error) — this output renders
    /// black; a note was already emitted at startup.
    Skip,
}

/// One resolved screen target: its key, background path, and build spec.
struct Target {
    screen: String,
    bg: PathBuf,
    spec: RunSpec,
    runnable: bool,
    /// The background is an application-type item, which is launch-fatal in
    /// the reference (WallpaperParser.cpp:22-24 throws out of the startup
    /// `loadBackgrounds`, main catches → exit 1) — matched in
    /// [`run_wallpapers`].
    app_fatal: bool,
}

/// Run the per-screen wallpapers on the wayland presentation layer, with the
/// control socket (if any) wired to a dedicated applier thread.
fn run_wallpapers(args: CompatArgs) -> ExitCode {
    let window_mode = args.mode != WindowMode::DesktopBackground;
    let targets = build_targets(&args);
    if targets.is_empty() {
        eprintln!("At least one background ID must be specified");
        return ExitCode::FAILURE;
    }

    // Reference parity: an application-type background is launch-fatal even
    // when sibling screens could run — `WallpaperParser::parse` throws
    // "Application wallpapers are not supported on this platform"
    // (WallpaperParser.cpp:22-24) out of the startup `loadBackgrounds`
    // (WallpaperApplication.cpp:72/187), main catches it and exits 1. The
    // message appears twice on stderr (`sLog.exception` writes it, then main's
    // catch prints `e.what()` — the same string).
    if targets.iter().any(|t| t.app_fatal) {
        eprintln!("Application wallpapers are not supported on this platform");
        eprintln!("Application wallpapers are not supported on this platform");
        return ExitCode::FAILURE;
    }

    // If nothing is runnable, fail with the per-screen reasons (doc §3.3;
    // task scope: scene/web → clean message + nonzero exit).
    if !targets.iter().any(|t| t.runnable) {
        for t in &targets {
            eprintln!("{}: {} — {}", t.screen, t.bg.display(), unrunnable_note(&t.bg));
        }
        return ExitCode::FAILURE;
    }
    // Warn about the non-runnable siblings that will render black.
    for t in &targets {
        if !t.runnable {
            eprintln!(
                "{}: {} — {} (this output will stay black)",
                t.screen,
                t.bg.display(),
                unrunnable_note(&t.bg)
            );
        }
    }

    // Bind the control socket first so the daemon's readiness probe (a socket
    // file within ~5s of exec, doc §8.3) is satisfied before the slower
    // wayland/GPU bring-up.
    let seed: Vec<(String, Option<PathBuf>)> = targets
        .iter()
        .map(|t| (t.screen.clone(), Some(t.bg.clone())))
        .collect();
    let (socket, ipc_app) = setup_socket(&args, seed);
    let registrar = ipc_app.as_ref().map(IpcApp::registrar);

    // On SIGTERM/SIGINT unlink the control socket like the clean-exit path
    // (the daemon TERMs the engine before removing the socket, doc §1). No-op
    // when no `--control-socket` was given.
    signals::install_cleanup(args.control_socket.clone());

    // Per-screen build specs owned by the factory (must be `'static`).
    let specs: Vec<(String, RunSpec)> = targets.into_iter().map(|t| (t.screen, t.spec)).collect();
    let volume = args.volume;
    let silent = args.silent;
    // `--set-property` overrides (scene user properties), captured for the
    // factory closure — folded into the scene's property bag at build time so
    // color/combo/slider changes drive the render (docs/format-scene-json.md §3.2).
    let properties = args.set_properties.clone();

    // Playlists to rotate (reference `initializePlaylists`,
    // WallpaperApplication.cpp:265-327): the window-mode default playlist
    // drives the single `default` wallpaper; in desktop mode each screen with
    // a `--playlist` rotates independently. Each carries the currently shown
    // path so rotation starts from it (WallpaperApplication.cpp:290-298).
    let active_playlists: Vec<(String, PlaylistDefinition, Option<PathBuf>)> = if window_mode {
        args.window_playlist
            .clone()
            .map(|p| {
                let current = p.items.first().cloned();
                ("default".to_owned(), p, current)
            })
            .into_iter()
            .collect()
    } else {
        args.screens
            .iter()
            .filter_map(|s| {
                s.playlist.clone().map(|p| {
                    let current = s.background.clone().map(PathBuf::from);
                    (s.name.clone(), p, current)
                })
            })
            .collect()
    };
    let rotation_properties = args.set_properties.clone();
    let playlist_stop = Arc::new(AtomicBool::new(false));
    let mut playlist_handle: Option<std::thread::JoinHandle<()>> = None;

    // One shared system-audio capture for every output (mono monitor source;
    // the spectrum is scene-global, docs subsystems-misc.md §1.3). Started once,
    // its lock-free spectrum is read per-frame by each scene renderer (V4). Only
    // scenes consume audio uniforms, so image/video-only launches never open
    // PulseAudio (no needless capture threads — SPEC.md §V5/§V6 spirit).
    let audio = specs
        .iter()
        .any(|(_, spec)| matches!(spec, RunSpec::Scene { .. }))
        .then(|| {
            let cap = Arc::new(AudioCapture::start(audio_config(&args)));
            tracing::info!(status = ?cap.status(), device = ?cap.device(), "audio capture");
            cap
        });

    // Automute: mute the wallpaper's own audio while another application is
    // playing sound (docs subsystems-misc.md §1.2/§2.3). `--noautomute`
    // disables it. Only video wallpapers expose a mute control here (the sole
    // consumer wired), so the PulseAudio detector is only opened when a video
    // output exists — an image/scene-only launch never connects (no needless
    // work, SPEC §V6 spirit; scene sound muting is not yet plumbed).
    let has_video = specs
        .iter()
        .any(|(_, spec)| matches!(spec, RunSpec::Video { .. }));
    let automute = Arc::new(AutoMute::start(!args.noautomute && has_video));
    tracing::info!(enabled = automute.enabled(), "automute detector");

    // Video controls registered as outputs come up, so the automute applier can
    // toggle their mute flag live.
    let video_controls: Arc<Mutex<Vec<VideoControl>>> = Arc::new(Mutex::new(Vec::new()));
    let applier_stop = Arc::new(AtomicBool::new(false));
    let applier = spawn_automute_applier(&automute, &video_controls, &applier_stop);

    // Restrict wallpaper surfaces to the outputs the user actually configured
    // (`--screen-root <name>`), so unconfigured monitors are left untouched
    // instead of being blacked out by a stray surface (SPEC T30/V6). Window
    // mode registers its single wallpaper as `default` (not a real output
    // name) and is shown on every output, so its screen_roots stay empty.
    // Computed before the factory takes ownership of `specs`.
    let screen_roots: Vec<String> = if window_mode {
        Vec::new()
    } else {
        specs.iter().map(|(name, _)| name.clone()).collect()
    };

    // Params the IPC applier needs to build a replacement wallpaper off the
    // render thread for a live `bg`/`preload` (cloned before the factory takes
    // the originals). Scaling/clamp default to the first configured screen's.
    let (bg_scaling, bg_clamp) = args
        .screens
        .first()
        .map(|s| (s.scaling, s.clamp))
        .unwrap_or((args.window_scaling, args.window_clamp));
    let build_ctx = Arc::new(BuildContext {
        scaling: bg_scaling,
        clamp: bg_clamp,
        volume,
        silent,
        registrar: registrar.clone(),
        audio: audio.clone(),
        automute_controls: video_controls.clone(),
    });

    let factory_controls = video_controls.clone();
    let factory: kirie_platform::RendererFactory = Box::new(move |target: &RenderTarget<'_>| {
        build_renderer(
            target,
            &specs,
            window_mode,
            volume,
            silent,
            registrar.as_ref(),
            audio.clone(),
            &properties,
            &factory_controls,
        )
    });

    let present = kirie_platform::PresentOptions {
        screen_roots,
        ..Default::default()
    };

    let exit = match Platform::connect_with(kirie_platform::Backend::from_env(), present, factory) {
        Ok(mut platform) => {
            // Enable live `bg`/`preload` swaps: hand the applier the render
            // thread's command channel + the build params (Wayland only; X11
            // has no channel yet and falls back to relaunch).
            if let (Some(app), Some(cmd_tx)) = (ipc_app.as_ref(), platform.command_sender()) {
                if let Ok(mut slot) = app.swap_slot().lock() {
                    *slot = Some(SwapCtx {
                        cmd_tx,
                        build: build_ctx.clone(),
                    });
                }
            }
            // Playlist rotation rides the same live-swap channel as the socket
            // `bg` command (the reference drives it from the render loop via
            // `updatePlaylists`, WallpaperApplication.cpp:962).
            if !active_playlists.is_empty() {
                match platform.command_sender() {
                    Some(cmd_tx) => {
                        playlist_handle = spawn_playlist_rotator(
                            active_playlists,
                            window_mode,
                            cmd_tx,
                            build_ctx.clone(),
                            rotation_properties,
                            playlist_stop.clone(),
                        );
                    }
                    None => tracing::warn!(
                        "playlist rotation needs the live-swap command channel; disabled on this backend"
                    ),
                }
            }
            // A test/CI bound on the otherwise-infinite run; the daemon never
            // sets it, so live behavior is unchanged (runs until stopped).
            let duration = std::env::var("KIRIE_RUN_SECONDS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs);
            match platform.run(duration) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    // Losing the last layer surface is an abnormal exit the
                    // supervisor relaunches on (doc §5).
                    tracing::error!(%err, "presentation layer stopped");
                    ExitCode::FAILURE
                }
            }
        }
        Err(err) => {
            eprintln!("cannot start the wayland presentation layer: {err}");
            ExitCode::FAILURE
        }
    };

    // Stop the playlist rotator and join it.
    playlist_stop.store(true, Ordering::Relaxed);
    if let Some(h) = playlist_handle {
        let _ = h.join();
    }

    // Stop the automute applier and join it, then drop the detector (joins its
    // PulseAudio monitor thread) before returning.
    applier_stop.store(true, Ordering::Relaxed);
    if let Some(h) = applier {
        let _ = h.join();
    }
    drop(automute);

    // Drop the control socket (unlinks the socket file on the clean-exit path,
    // doc §1) and the IPC app last.
    drop(socket);
    drop(ipc_app);
    exit
}

/// Spawn the playlist rotation thread — the stand-in for the reference's
/// per-frame `updatePlaylists` call on the render loop
/// (WallpaperApplication.cpp:451-475/962): kirie's render thread belongs to the
/// platform, so a dedicated timer thread polls the same conditions (timer mode,
/// more than one item, delay elapsed) and drives swaps through the live-swap
/// channel — the exact path the control socket's `bg` command uses.
///
/// `playlists` is `(screen key, definition, currently shown path)`. In window
/// mode the single `default` wallpaper is swapped via the platform's `"*"`
/// output selector. Returns `None` when no playlist survives registration.
fn spawn_playlist_rotator(
    playlists: Vec<(String, PlaylistDefinition, Option<PathBuf>)>,
    window_mode: bool,
    cmd_tx: CommandSender,
    build: Arc<BuildContext>,
    properties: Vec<(String, String)>,
    stop: Arc<AtomicBool>,
) -> Option<std::thread::JoinHandle<()>> {
    let mut rng = Rng::seeded();
    let now = Instant::now();
    let mut active: Vec<(String, ActivePlaylist)> = playlists
        .into_iter()
        .filter_map(|(screen, def, current)| {
            ActivePlaylist::start(def, current.as_deref(), now, &mut rng)
                .map(|state| (screen, state))
        })
        .collect();
    if active.is_empty() {
        return None;
    }
    for (screen, state) in &active {
        tracing::info!(
            %screen,
            playlist = state.name(),
            items = state.item_count(),
            "playlist registered"
        );
    }
    std::thread::Builder::new()
        .name("kirie-playlist".into())
        .spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(250));
                let now = Instant::now();
                for (screen, state) in &mut active {
                    if !state.due(now) {
                        continue;
                    }
                    // Desktop mode targets the screen's own output; window mode
                    // swaps the platform's primary output (`"*"`).
                    let swap_screen = if window_mode { "*" } else { screen.as_str() };
                    let screen_key = screen.clone();
                    state.advance(
                        screen,
                        now,
                        &mut rng,
                        |path| playlist_preflight(&build, &screen_key, path, &properties),
                        |path| {
                            playlist_show(&cmd_tx, &build, &screen_key, swap_screen, path, &properties)
                        },
                    );
                }
            }
        })
        .ok()
}

/// Playlist candidate preflight (the reference `preflightWallpaper`,
/// WallpaperApplication.cpp:369-389, parses the item's `project.json` before
/// switching): can kirie build this item on this build? Web items pass only on
/// a `web-cef` build (they swap via the render-thread path).
fn playlist_preflight(
    build: &Arc<BuildContext>,
    screen: &str,
    path: &Path,
    properties: &[(String, String)],
) -> bool {
    if build
        .build_fn(screen.to_owned(), path, properties.to_vec())
        .is_some()
    {
        return true;
    }
    #[cfg(feature = "web-cef")]
    if build
        .build_local_fn(screen.to_owned(), path, properties.to_vec())
        .is_some()
    {
        return true;
    }
    false
}

/// Show a playlist item (the reference `setBackground` call in
/// `advancePlaylist`, WallpaperApplication.cpp:433-437): same dispatch as the
/// socket `bg` swap — off-thread build + [`RenderCommand::Swap`] for
/// video/image/scene, render-thread [`RenderCommand::SwapLocal`] for web on a
/// `web-cef` build. On success the IPC applier is told the new background so
/// socket `status` stays truthful (reference updates `screenBackgrounds`,
/// WallpaperApplication.cpp:1050).
fn playlist_show(
    cmd_tx: &CommandSender,
    build: &Arc<BuildContext>,
    screen: &str,
    swap_screen: &str,
    path: &Path,
    properties: &[(String, String)],
) -> bool {
    if let Some(build_fn) = build.build_fn(screen.to_owned(), path, properties.to_vec()) {
        let sent = cmd_tx
            .send(RenderCommand::Swap {
                screen: swap_screen.to_owned(),
                key: path.to_string_lossy().into_owned(),
                build: build_fn,
            })
            .is_ok();
        if sent {
            build.notify_background(screen, path);
        }
        return sent;
    }
    #[cfg(feature = "web-cef")]
    if let Some(build_local) = build.build_local_fn(screen.to_owned(), path, properties.to_vec()) {
        let sent = cmd_tx
            .send(RenderCommand::SwapLocal {
                screen: swap_screen.to_owned(),
                build_local,
            })
            .is_ok();
        if sent {
            build.notify_background(screen, path);
        }
        return sent;
    }
    false
}

/// Spawn the automute applier thread: while the detector reports another app is
/// playing, keep every registered video output muted; unmute when it stops. New
/// controls (later-registered outputs) are caught by the length check. Returns
/// `None` when automute is disabled (`--noautomute` or no video output).
fn spawn_automute_applier(
    automute: &Arc<AutoMute>,
    controls: &Arc<Mutex<Vec<VideoControl>>>,
    stop: &Arc<AtomicBool>,
) -> Option<std::thread::JoinHandle<()>> {
    if !automute.enabled() {
        return None;
    }
    let automute = automute.clone();
    let controls = controls.clone();
    let stop = stop.clone();
    Some(
        std::thread::Builder::new()
            .name("kirie-automute-apply".into())
            .spawn(move || {
                let mut last: Option<bool> = None;
                let mut applied_len = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let playing = automute.is_playing();
                    if let Ok(guard) = controls.lock() {
                        // Re-apply on a state change or when a new control was
                        // registered (so late outputs inherit the current mute).
                        if last != Some(playing) || guard.len() != applied_len {
                            for c in guard.iter() {
                                c.set_mute(playing);
                            }
                            last = Some(playing);
                            applied_len = guard.len();
                        }
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            })
            .expect("spawn automute applier"),
    )
}

/// Build the per-output list of screen targets (doc §3.1, §3.3).
fn build_targets(args: &CompatArgs) -> Vec<Target> {
    let default_bg = args.default_background.clone();
    if args.mode != WindowMode::DesktopBackground {
        // Window / preview mode: one wallpaper registered as `default`
        // (doc §3.3), rendered on every output. With a default playlist its
        // first item is what actually shows, even over an explicit `--bg`
        // (WallpaperApplication.cpp:187-195).
        let bg = args
            .window_playlist
            .as_ref()
            .and_then(|p| p.items.first())
            .map(|p| p.to_string_lossy().into_owned())
            .or(default_bg);
        let Some(bg) = bg else {
            return Vec::new();
        };
        return vec![make_target(
            "default".to_owned(),
            bg,
            args.window_scaling,
            args.window_clamp,
        )];
    }

    // Desktop-background mode: one target per declared screen; a screen with no
    // `--bg` inherits `default_background` at load time (doc §3.1).
    args.screens
        .iter()
        .map(|screen| {
            let bg = screen
                .background
                .clone()
                .or_else(|| default_bg.clone())
                .unwrap_or_default();
            make_target(screen.name.clone(), bg, screen.scaling, screen.clamp)
        })
        .collect()
}

/// Classify a screen's background and turn it into a build spec.
fn make_target(screen: String, bg: String, scaling: ScalingMode, clamp: ClampMode) -> Target {
    let bg_path = PathBuf::from(&bg);
    match resolve::classify(&bg) {
        Ok(Wallpaper::Video { media }) => Target {
            screen,
            bg: bg_path,
            spec: RunSpec::Video { media, scaling },
            runnable: true,
            app_fatal: false,
        },
        Ok(Wallpaper::Image { file }) => Target {
            screen,
            bg: bg_path,
            spec: RunSpec::Image { file, scaling, clamp },
            runnable: true,
            app_fatal: false,
        },
        Ok(Wallpaper::Scene { dir }) => Target {
            screen,
            bg: bg_path,
            spec: RunSpec::Scene { dir, scaling, clamp },
            runnable: true,
            app_fatal: false,
        },
        Ok(Wallpaper::Web { dir, file }) => make_web_target(screen, bg_path, &dir, &file),
        Ok(Wallpaper::Unsupported { kind }) => {
            tracing::warn!(%screen, kind, "wallpaper type not supported");
            Target {
                screen,
                bg: bg_path,
                spec: RunSpec::Skip,
                runnable: false,
                // Application items are refused for the whole launch, exactly
                // like the reference (WallpaperParser.cpp:22-24).
                app_fatal: kind == "application",
            }
        }
        Ok(Wallpaper::Asset) => {
            tracing::warn!(%screen, "background is a non-renderable asset (effect preset)");
            Target {
                screen,
                bg: bg_path,
                spec: RunSpec::Skip,
                runnable: false,
                app_fatal: false,
            }
        }
        Err(err) => {
            let reason = classify_reason(&err);
            tracing::warn!(%screen, %reason, "cannot load wallpaper");
            Target {
                screen,
                bg: bg_path,
                spec: RunSpec::Skip,
                runnable: false,
                app_fatal: false,
            }
        }
    }
}

/// Build a target for a web wallpaper (feature-gated on the CEF backend).
///
/// With `web-cef` the entry page becomes a runnable [`RunSpec::Web`]; without
/// it the screen is not runnable (see [`web_unrunnable_note`] for why).
#[cfg(feature = "web-cef")]
fn make_web_target(screen: String, bg: PathBuf, dir: &Path, file: &str) -> Target {
    let url = resolve::web_entry_url(dir, file);
    tracing::info!(%screen, url, "web wallpaper (CEF off-screen backend)");
    Target {
        screen,
        bg,
        spec: RunSpec::Web {
            url,
            dir: dir.to_path_buf(),
        },
        runnable: true,
        app_fatal: false,
    }
}

/// Build a target for a web wallpaper on a build without the CEF backend: not
/// runnable (this output stays black; a per-screen note is emitted).
#[cfg(not(feature = "web-cef"))]
fn make_web_target(screen: String, bg: PathBuf, _dir: &Path, _file: &str) -> Target {
    tracing::warn!(%screen, "web wallpaper not runnable in this build");
    Target {
        screen,
        bg,
        spec: RunSpec::Skip,
        runnable: false,
        app_fatal: false,
    }
}

/// A short reason string for a classification failure.
fn classify_reason(err: &ClassifyError) -> String {
    err.to_string()
}

/// The per-screen note for a web wallpaper that cannot run on this build,
/// naming the feature to rebuild with. Feature-aware:
///
/// * no web feature → name both `web-cef` and `web-webview`;
/// * `web-webview` only → explain the CEF backend is the one that composites
///   through this presentation layer and point at `web-cef`.
///
/// (A `web-cef` build never reaches here — web items are runnable there.)
fn web_unrunnable_note() -> String {
    #[cfg(all(feature = "web-webview", not(feature = "web-cef")))]
    {
        "web wallpapers cannot run on the webview backend: wry/webkit2gtk renders into \
         its own native window and has no off-screen path (upstream limitation, won't-fix); \
         rebuild with --features web-cef for the composited off-screen web backend"
            .to_owned()
    }
    #[cfg(not(any(feature = "web-cef", feature = "web-webview")))]
    {
        "web wallpapers are not supported by this build; rebuild with --features web-cef \
         (recommended) or --features web-webview"
            .to_owned()
    }
    // A `web-cef` build (with or without `web-webview`) runs web items, so this
    // function is never called there; give it a body so it still compiles.
    #[cfg(feature = "web-cef")]
    {
        "web wallpapers are supported on this build".to_owned()
    }
}

/// The per-screen notice explaining why a non-runnable background stays black
/// (web/application unimplemented, asset non-renderable, or a load error).
fn unrunnable_note(bg: &Path) -> String {
    match resolve::classify(&bg.to_string_lossy()) {
        // Web runnability depends on the compiled web feature; the precise
        // message lives here rather than in the feature-agnostic classifier.
        Ok(Wallpaper::Web { .. }) => web_unrunnable_note(),
        Ok(w) => w
            .unrunnable_reason()
            .unwrap_or_else(|| "not yet supported by kirie".to_owned()),
        Err(err) => classify_reason(&err),
    }
}

/// Build one output's renderer from the matching screen spec (or black).
#[allow(clippy::too_many_arguments)]
fn build_renderer(
    target: &RenderTarget<'_>,
    specs: &[(String, RunSpec)],
    window_mode: bool,
    volume: i64,
    silent: bool,
    registrar: Option<&crossbeam_channel::Sender<Register>>,
    audio: Option<Arc<AudioCapture>>,
    properties: &[(String, String)],
    automute_controls: &Arc<Mutex<Vec<VideoControl>>>,
) -> Box<dyn Renderer> {
    // Window mode: the single wallpaper is registered as `default` and shown
    // on every output (doc §3.3). Desktop mode: match by output name. Outputs
    // the user did not configure never reach here — `PresentOptions::screen_roots`
    // keeps kirie-platform from creating a surface for them (SPEC T30), matching
    // the C++ engine which only surfaces the requested screens. The `black`
    // fallback below is a defensive default only.
    let (screen_key, spec) = if window_mode {
        match specs.first() {
            Some((_, spec)) => ("default".to_owned(), spec),
            None => return black(target),
        }
    } else {
        match specs.iter().find(|(name, _)| name == target.output_name) {
            Some((name, spec)) => (name.clone(), spec),
            None => return black(target),
        }
    };

    // Web (CEF) is render-thread-bound (its backend is `!Send`), so it's built
    // here. Everything else is `Send` and goes through `build_for_spec`, which
    // the preload / async-swap paths also call from a worker thread.
    #[cfg(feature = "web-cef")]
    if let RunSpec::Web { url, dir } = spec {
        return build_web(target, url, dir, silent, properties);
    }
    build_for_spec(
        target,
        screen_key,
        spec,
        volume,
        silent,
        registrar,
        audio,
        properties,
        automute_controls,
    )
}

/// Build the CEF web renderer for `url` on the render thread (the backend is
/// `!Send`, so it must be created and live there). Degrades to `black` on a
/// backend-start failure rather than crashing the output. Shared by the
/// initial-launch path ([`build_renderer`]) and the live-swap path
/// ([`BuildContext::build_local_fn`]) so both build web identically.
#[cfg(feature = "web-cef")]
fn build_web(
    target: &RenderTarget<'_>,
    url: &str,
    dir: &Path,
    silent: bool,
    properties: &[(String, String)],
) -> Box<dyn Renderer> {
    // Initial off-screen size is arbitrary: `WebRenderer` resizes the CEF surface
    // to the real output on its first frame. `--silent` mutes the page's audio
    // (docs/subsystems-misc.md §3: host mute).
    let size = WebSize {
        width: 1920,
        height: 1080,
    };
    match <CefBackend as WebBackend>::new(url, size) {
        Ok(mut backend) => {
            if silent {
                backend.set_muted(true);
            }
            // Deliver the full typed property set once (project.json defaults
            // with `--set-property` overrides folded in) — the reference sends
            // this on the first frame and pages may block init on it
            // (CWeb.cpp `__wpApplyProps`; the CEF thread queues it until the
            // page's main frame exists).
            let props = web_props_json(dir, properties);
            if props != "{}" {
                backend.apply_properties(&props);
            }
            tracing::info!(output = %target.output_name, url, "web (CEF) wallpaper ready");
            Box::new(WebRenderer::new(target, Box::new(backend)))
        }
        Err(err) => {
            eprintln!("{}: failed to start web backend: {err}", target.output_name);
            black(target)
        }
    }
}

/// The `{name:{value:..}}` JSON batch for a web wallpaper's `applyUserProperties`
/// (doc §3.5), typed like the reference encoder (`CWeb.cpp`): bool bare, slider
/// bare number, color an `"r g b"` float string, `text` UI labels skipped,
/// everything else a JSON string. `--set-property` overrides replace the
/// project defaults by declared type.
#[cfg(feature = "web-cef")]
fn web_props_json(dir: &Path, overrides: &[(String, String)]) -> String {
    use kirie_formats::project::{Project, PropertyEntry, PropertyKind};
    let Ok(project) = Project::from_path(dir.join("project.json")) else {
        return "{}".to_owned();
    };
    let over: std::collections::HashMap<&str, &str> = overrides
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let mut out = String::from("{");
    let mut first = true;
    for (name, entry) in &project.general.properties {
        let PropertyEntry::Property(p) = entry else { continue };
        let raw = over.get(name.as_str()).copied();
        let value = match &p.kind {
            PropertyKind::Bool { value } => {
                let v = raw.map_or(*value, |r| matches!(r.trim(), "1" | "true" | "True" | "TRUE"));
                if v { "true".to_owned() } else { "false".to_owned() }
            }
            PropertyKind::Slider { value, .. } => {
                let v = raw.and_then(|r| r.trim().parse::<f64>().ok()).unwrap_or(f64::from(*value));
                format!("{v}")
            }
            PropertyKind::Color { value: [r, g, b] } => {
                // Override form is "r g b" floats; keep the reference's string form.
                let s = raw.map_or_else(|| format!("{r:.4} {g:.4} {b:.4}"), str::to_owned);
                format!("\"{}\"", esc(&s))
            }
            // `text` entries are UI labels, not values (the reference skips them).
            PropertyKind::Text => continue,
            PropertyKind::Combo { value, .. }
            | PropertyKind::TextInput { value }
            | PropertyKind::UserShortcut { value }
            | PropertyKind::File { value }
            | PropertyKind::Directory { value }
            | PropertyKind::SceneTexture { value } => {
                format!("\"{}\"", esc(raw.unwrap_or(value)))
            }
        };
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&format!("\"{}\":{{\"value\":{value}}}", esc(name)));
    }
    out.push('}');
    out
}

/// Build a `Send` (non-web) renderer for `spec` + `screen_key`. Returns
/// `Box<dyn Renderer + Send>` so it can be built off the render thread (preload /
/// async `bg` swap) as well as on it; the render thread stores it as a plain
/// `Box<dyn Renderer>`. Web backends are `!Send` and stay on the render thread in
/// [`build_renderer`].
#[allow(clippy::too_many_arguments)]
fn build_for_spec(
    target: &RenderTarget<'_>,
    screen_key: String,
    spec: &RunSpec,
    volume: i64,
    silent: bool,
    registrar: Option<&crossbeam_channel::Sender<Register>>,
    audio: Option<Arc<AudioCapture>>,
    properties: &[(String, String)],
    automute_controls: &Arc<Mutex<Vec<VideoControl>>>,
) -> Box<dyn Renderer + Send> {
    match spec {
        RunSpec::Video { media, scaling } => {
            let options = VideoOptions {
                volume: volume as f64 * 100.0 / 128.0,
                mute: false,
                silent,
                paused: false,
                scaling: to_video_scaling(*scaling),
                enable_audio: true,
            };
            match VideoPlayer::open(media, options) {
                Ok((player, control)) => {
                    // Register a clone with the automute applier so its mute
                    // flag tracks other-app playback (docs subsystems-misc.md
                    // §2.3). `VideoControl` is cheap to clone (channel senders).
                    if let Ok(mut guard) = automute_controls.lock() {
                        guard.push(control.clone());
                    }
                    if let Some(reg) = registrar {
                        let _ = reg.send(Register::Video {
                            screen: screen_key,
                            control,
                        });
                    }
                    let info = player.info();
                    tracing::info!(
                        output = %target.output_name,
                        width = info.width,
                        height = info.height,
                        audio = player.has_audio(),
                        "video wallpaper ready"
                    );
                    Box::new(VideoRenderer::new(target, player))
                }
                Err(err) => {
                    eprintln!("{}: failed to open video: {err}", target.output_name);
                    black(target)
                }
            }
        }
        RunSpec::Image { file, scaling, clamp } => match ImageContent::from_path(file) {
            Ok(content) => {
                let options = ImageOptions {
                    scaling: to_render_scaling(*scaling),
                    clamp: to_render_clamp(*clamp),
                };
                match ImageRenderer::new(target, &content, options) {
                    Ok(renderer) => {
                        tracing::info!(output = %target.output_name, "image wallpaper ready");
                        Box::new(renderer)
                    }
                    Err(err) => {
                        eprintln!("{}: failed to build image renderer: {err}", target.output_name);
                        black(target)
                    }
                }
            }
            Err(err) => {
                eprintln!("{}: failed to load image: {err}", target.output_name);
                black(target)
            }
        },
        RunSpec::Scene { dir, scaling, clamp } => {
            let options = kirie_render::SceneOptions {
                scaling: to_render_scaling(*scaling),
                clamp: to_render_clamp(*clamp),
            };
            match kirie_render::load_workshop_scene(
                target,
                dir,
                resolve::we_assets_dir().as_deref(),
                options,
                audio,
                properties,
            ) {
                Ok(renderer) => {
                    tracing::info!(output = %target.output_name, "scene wallpaper ready");
                    renderer
                }
                Err(err) => {
                    eprintln!("{}: failed to build scene renderer: {err}", target.output_name);
                    black(target)
                }
            }
        }
        // Web is routed to the render thread by `build_renderer` and never
        // reaches here; keep the arm total + defensive.
        #[cfg(feature = "web-cef")]
        RunSpec::Web { .. } => black(target),
        RunSpec::Skip => black(target),
    }
}

/// Send parameters the IPC applier needs to build a wallpaper renderer off the
/// render thread for a live `bg`/`preload` (everything [`build_for_spec`] needs
/// besides the per-command screen/spec/properties). Cheap clones of the launch
/// params, so the factory keeps its own copies.
pub(crate) struct BuildContext {
    scaling: ScalingMode,
    clamp: ClampMode,
    volume: i64,
    silent: bool,
    registrar: Option<crossbeam_channel::Sender<Register>>,
    audio: Option<Arc<AudioCapture>>,
    automute_controls: Arc<Mutex<Vec<VideoControl>>>,
}

impl BuildContext {
    /// Report an engine-driven background change (playlist rotation) to the
    /// IPC applier so socket `status` reflects the on-screen path — the
    /// reference's `setBackground` updates `screenBackgrounds` the same way
    /// (WallpaperApplication.cpp:1050). No-op without a control socket.
    pub(crate) fn notify_background(&self, screen: &str, path: &Path) {
        if let Some(reg) = &self.registrar {
            let _ = reg.send(Register::Background {
                screen: screen.to_owned(),
                bg: path.to_path_buf(),
            });
        }
    }

    /// Classify `path` and return an off-thread [`kirie_platform::BuildFn`] that
    /// builds it for `screen` with `properties`. `None` when the wallpaper isn't
    /// runnable, or is web (web is `!Send` / render-thread-only).
    pub(crate) fn build_fn(
        self: &Arc<Self>,
        screen: String,
        path: &Path,
        properties: Vec<(String, String)>,
    ) -> Option<kirie_platform::BuildFn> {
        let target = make_target(
            screen.clone(),
            path.to_string_lossy().into_owned(),
            self.scaling,
            self.clamp,
        );
        match &target.spec {
            RunSpec::Video { .. } | RunSpec::Image { .. } | RunSpec::Scene { .. } => {}
            // Skip (unsupported) and Web (render-thread-only) are not swappable.
            _ => return None,
        }
        let ctx = self.clone();
        let spec = target.spec;
        let build: kirie_platform::BuildFn = Box::new(move |device, queue, format, name| {
            let rt = RenderTarget {
                device,
                queue,
                format,
                output_name: name,
            };
            build_for_spec(
                &rt,
                screen,
                &spec,
                ctx.volume,
                ctx.silent,
                ctx.registrar.as_ref(),
                ctx.audio.clone(),
                &properties,
                &ctx.automute_controls,
            )
        });
        Some(build)
    }

    /// Classify `path` and, for a **web** item, return a render-thread
    /// [`kirie_platform::BuildLocalFn`] that builds it (CEF is `!Send`, so it
    /// can't use the off-thread [`build_fn`]). `None` for non-web / unsupported
    /// items (use `build_fn`). This is what lets the daemon's live `bg` swap
    /// bring in a web wallpaper without relaunching the engine — a brief hitch
    /// while CEF builds, then it swaps in. Only compiled with the CEF backend.
    #[cfg(feature = "web-cef")]
    pub(crate) fn build_local_fn(
        self: &Arc<Self>,
        screen: String,
        path: &Path,
        properties: Vec<(String, String)>,
    ) -> Option<kirie_platform::BuildLocalFn> {
        let target = make_target(
            screen,
            path.to_string_lossy().into_owned(),
            self.scaling,
            self.clamp,
        );
        let RunSpec::Web { url, dir } = target.spec else {
            return None;
        };
        let silent = self.silent;
        let build: kirie_platform::BuildLocalFn = Box::new(move |device, queue, format, name| {
            let rt = RenderTarget {
                device,
                queue,
                format,
                output_name: name,
            };
            build_web(&rt, &url, &dir, silent, &properties)
        });
        Some(build)
    }
}

/// The live-swap context handed to the IPC applier once the platform is up (its
/// command channel + the build params). Applier `bg`/`preload` use it to build
/// off-thread and swap on the render thread.
pub(crate) struct SwapCtx {
    pub cmd_tx: kirie_platform::CommandSender,
    pub build: Arc<BuildContext>,
}

/// Bind the control socket and spawn its applier thread, or `(None, None)` if
/// no `--control-socket` was given or the bind failed (doc §1: bind failure is
/// non-fatal — the engine runs without live control).
fn setup_socket(
    args: &CompatArgs,
    seed: Vec<(String, Option<PathBuf>)>,
) -> (Option<kirie_ipc::ControlSocket>, Option<IpcApp>) {
    let Some(path) = &args.control_socket else {
        return (None, None);
    };
    let (events_tx, events_rx) = crossbeam_channel::unbounded();
    match kirie_ipc::ControlSocket::bind(path.clone(), events_tx) {
        Ok(socket) => {
            let app = IpcApp::spawn(
                events_rx,
                seed,
                args.playback_speed as f32,
                args.volume as i32,
                false,
            );
            (Some(socket), Some(app))
        }
        Err(err) => {
            tracing::warn!(path = %path.display(), %err, "control socket unavailable; running without live control");
            (None, None)
        }
    }
}

/// A renderer that clears its surface to opaque black — the fallback for an
/// unconfigured or unsupported output.
struct BlackRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Build a [`BlackRenderer`] for `target`.
fn black(target: &RenderTarget<'_>) -> Box<dyn Renderer + Send> {
    Box::new(BlackRenderer {
        device: target.device.clone(),
        queue: target.queue.clone(),
    })
}

impl Renderer for BlackRenderer {
    fn render(&mut self, view: &wgpu::TextureView, _size: SurfaceSize, _dt: f32) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kirie-black-encoder"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("kirie-black-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit(Some(encoder.finish()));
    }
}
