//! The control-socket application thread (docs/compat-socket.md §4-§5).
//!
//! `kirie-ipc` owns the socket thread and delivers typed [`IpcEvent`]s over a
//! channel; this module owns the thread that *applies* them. Keeping a
//! dedicated applier thread (rather than touching the render thread) is the
//! SPEC V3/V4 shape: the applier solely owns the screen registry and the live
//! [`VideoControl`] handles, everything crosses threads by channel, and the
//! render thread is never blocked on a socket client.
//!
//! Live effects are applied where kirie has a handle: `speed`/`volume`/`mute`/
//! `scaling` reach video wallpapers through [`VideoControl`]; `status` is
//! answered from the owned registry; commands that need a render-thread
//! rebuild kirie cannot do live in P3 (`bg`, structural `property`, socket
//! `screenshot`) reply `error\n` honestly. `preload` and recognized `set`
//! keys reply `ok\n` per the protocol (doc §4.6, §4.8).

use std::collections::BTreeMap;
use std::path::PathBuf;

use crossbeam_channel::{Receiver, Sender, select};
use kirie_ipc::{Command, CommandOutcome, IpcEvent, ScalingMode as IpcScaling, ScreenStatus, StatusSnapshot};
use kirie_video::{ScalingMode as VideoScaling, VideoControl};

/// A live-control handle registered by the render factory as each wallpaper is
/// built, sent to the applier over the register channel.
pub enum Register {
    /// A video wallpaper became live on `screen`; its control handle drives
    /// speed/volume/mute/scaling.
    Video {
        /// The screen the wallpaper is registered under.
        screen: String,
        /// Its live-control handle.
        control: VideoControl,
    },
}

/// One registered background the socket reports and drives (doc §4.2, §4.7).
struct ScreenEntry {
    bg: Option<PathBuf>,
    control: Option<VideoControl>,
}

/// The applier's owned state (SPEC V3: sole owner, nothing shared).
struct AppState {
    /// Registered screens keyed lexicographically, matching the C++ `std::map`
    /// iteration order the `status` reply depends on (doc §4.2).
    screens: BTreeMap<String, ScreenEntry>,
    /// Global playback speed, reported by `status` and forwarded to videos
    /// (doc §4.3).
    speed: f32,
    /// Current volume 0-128 (doc §4.4). Forwarded to videos as 0-100.
    volume: i32,
    /// Mute gate (doc §4.5).
    muted: bool,
    /// Stored per-key property overrides (doc §4.9: recorded even when not
    /// live-applicable; "applies in P4/P5").
    properties: BTreeMap<String, String>,
}

/// Handle to the running applier thread.
///
/// The thread is detached: it exits on its own once the socket's event channel
/// closes (the [`kirie_ipc::ControlSocket`] was dropped). Dropping `IpcApp`
/// closes the register channel; the applier then serves the socket until it
/// too closes — so drop *ordering* between the socket and this handle is
/// irrelevant (no join, no deadlock).
pub struct IpcApp {
    register: Sender<Register>,
}

impl IpcApp {
    /// Spawn the applier thread, seeded with the parsed screen→background map
    /// (so `status` is correct from the first request, before the renderers
    /// attach their control handles) and the initial speed/volume/mute.
    ///
    /// `events` is the [`kirie_ipc::ControlSocket`] event receiver.
    pub fn spawn(
        events: Receiver<IpcEvent>,
        seed_screens: Vec<(String, Option<PathBuf>)>,
        speed: f32,
        volume: i32,
        muted: bool,
    ) -> Self {
        let (register_tx, register_rx) = crossbeam_channel::unbounded::<Register>();
        let mut state = AppState {
            screens: seed_screens
                .into_iter()
                .map(|(name, bg)| (name, ScreenEntry { bg, control: None }))
                .collect(),
            speed,
            volume,
            muted,
            properties: BTreeMap::new(),
        };
        std::thread::Builder::new()
            .name("kirie-ipc-app".into())
            .spawn(move || run(&mut state, &events, &register_rx))
            .expect("spawn ipc-app thread");
        Self {
            register: register_tx,
        }
    }

    /// A sender the render factory clones to register live controls.
    #[must_use]
    pub fn registrar(&self) -> Sender<Register> {
        self.register.clone()
    }
}

/// The applier loop: serve socket events and control registrations until both
/// channels close (SPEC V4: never blocks the render thread).
fn run(state: &mut AppState, events: &Receiver<IpcEvent>, register: &Receiver<Register>) {
    loop {
        select! {
            recv(events) -> msg => match msg {
                Ok(event) => handle_event(state, event),
                Err(_) => {
                    // Socket thread gone; drain any last registrations then stop.
                    while register.try_recv().is_ok() {}
                    return;
                }
            },
            recv(register) -> msg => match msg {
                Ok(reg) => handle_register(state, reg),
                Err(_) => {
                    // Factory side gone; keep serving the socket until it too
                    // closes, then stop.
                    while let Ok(event) = events.recv() {
                        handle_event(state, event);
                    }
                    return;
                }
            },
        }
    }
}

/// Attach a newly built wallpaper's live control to its screen entry.
fn handle_register(state: &mut AppState, reg: Register) {
    match reg {
        Register::Video { screen, control } => {
            // Apply the current global state to the freshly bound wallpaper
            // (doc §4.7: current volume/mute/speed re-applied to new loads).
            control.set_speed(f64::from(state.speed));
            control.set_volume(f64::from(state.volume) * 100.0 / 128.0);
            control.set_mute(state.muted);
            let entry = state.screens.entry(screen).or_insert(ScreenEntry {
                bg: None,
                control: None,
            });
            entry.control = Some(control);
        }
    }
}

/// Dispatch one socket event (doc §4 command table).
fn handle_event(state: &mut AppState, event: IpcEvent) {
    match event {
        IpcEvent::Status { reply } => {
            let snapshot = StatusSnapshot {
                speed: state.speed,
                screens: state
                    .screens
                    .iter()
                    .map(|(name, entry)| ScreenStatus {
                        screen: name.clone(),
                        bg: entry.bg.clone(),
                    })
                    .collect(),
            };
            let _ = reply.send(snapshot);
        }
        IpcEvent::GetProperties { screen, reply } => {
            // kirie extension (docs/compat-socket.md §11): report the selected
            // screen's property schema with the recorded overrides folded into
            // each `value`. The screen's background path is the workshop dir
            // that holds `project.json`; `None` ⇒ the first registered screen.
            let source = match &screen {
                Some(name) => state.screens.get(name).and_then(|e| e.bg.clone()),
                None => state.screens.values().find_map(|e| e.bg.clone()),
            };
            let body = match source {
                Some(dir) => super::list_props::properties_json_string(&dir, &state.properties),
                None => "[]".to_string(),
            };
            let _ = reply.send(body);
        }
        IpcEvent::Command { command, reply } => {
            let outcome = apply_command(state, command);
            let _ = reply.send(outcome);
        }
    }
}

/// Apply one command, returning the wire outcome (doc §4). For the always-ok
/// commands the server ignores the value, but a reply is still the completion
/// ack (kirie-ipc `IpcEvent` contract).
fn apply_command(state: &mut AppState, command: Command) -> CommandOutcome {
    match command {
        Command::Speed(s) => {
            state.speed = s;
            for entry in state.screens.values() {
                if let Some(c) = &entry.control {
                    c.set_speed(f64::from(s));
                }
            }
            CommandOutcome::Ok
        }
        Command::Volume(v) => {
            state.volume = v;
            let mapped = f64::from(v) * 100.0 / 128.0;
            for entry in state.screens.values() {
                if let Some(c) = &entry.control {
                    c.set_volume(mapped);
                }
            }
            CommandOutcome::Ok
        }
        Command::Mute(m) => {
            state.muted = m;
            for entry in state.screens.values() {
                if let Some(c) = &entry.control {
                    c.set_mute(m);
                }
            }
            CommandOutcome::Ok
        }
        // Recognized `set` keys always ack (doc §4.6); their live effect on a
        // running engine is partial/absent in P3, applied honestly where a
        // handle exists (none of these have one for video).
        Command::Set(_opt) => CommandOutcome::Ok,
        // Live wallpaper swap needs a render-thread rebuild kirie cannot do
        // inside the presentation loop yet (doc §4.7) → error, prior wallpaper
        // keeps running.
        Command::Bg { .. } => CommandOutcome::Error,
        // Warm-cache preload always acks, even on failure (doc §4.8).
        Command::Preload { .. } => CommandOutcome::Ok,
        Command::Property { screen, key, value } => {
            // doc §4.9: error if the screen has no registered background. The
            // override is recorded regardless (stored-before-validation), then
            // applied by later scene loads (P4/P5).
            state.properties.insert(key, value);
            if state.screens.get(&screen).is_some_and(|e| e.bg.is_some()) {
                CommandOutcome::Ok
            } else {
                CommandOutcome::Error
            }
        }
        Command::Scaling { screen, mode } => {
            // doc §4.10: mode already validated by the parser; error only if
            // the screen has no recorded background. Live effect via the video
            // control where present.
            match state.screens.get(&screen) {
                Some(entry) if entry.bg.is_some() => {
                    if let Some(c) = &entry.control {
                        c.set_scaling(map_scaling(mode));
                    }
                    CommandOutcome::Ok
                }
                _ => CommandOutcome::Error,
            }
        }
        Command::Clamp { screen, .. } => {
            // Same error semantics as scaling (doc §4.11). kirie-video has no
            // live clamp control, so this only validates the screen.
            match state.screens.get(&screen) {
                Some(entry) if entry.bg.is_some() => CommandOutcome::Ok,
                _ => CommandOutcome::Error,
            }
        }
        // Socket `screenshot` captures the *currently rendered* frame, which
        // requires render-thread readback kirie does not expose from the
        // applier in P3 (doc §4.12) → error (the daemon falls back to the
        // workshop preview image).
        Command::Screenshot { .. } => CommandOutcome::Error,
    }
}

/// Map the IPC scaling enum to kirie-video's (doc §4.10 mode table).
fn map_scaling(mode: IpcScaling) -> VideoScaling {
    match mode {
        IpcScaling::Stretch => VideoScaling::Stretch,
        IpcScaling::Fit => VideoScaling::Fit,
        IpcScaling::Fill => VideoScaling::Fill,
        IpcScaling::Default => VideoScaling::Default,
    }
}
