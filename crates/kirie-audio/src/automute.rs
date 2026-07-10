//! PulseAudio "another application is playing audio" detector — the input to
//! automute (docs/subsystems-misc.md §1.2 `PulseAudioPlayingDetector`).
//!
//! Mirrors the C++ detector's behaviour: own PulseAudio mainloop + context named
//! `"wallpaperengine"`, then periodically list all *sink inputs*; a sink input
//! whose `application.process.id` differs from our own pid and whose average
//! volume is not [`Volume::MUTED`] means "some other app is emitting sound"
//! (cpp `PulseAudioPlayingDetector.cpp:7-24`). While that holds, the wallpaper's
//! own audio output should be muted (the video path toggles `VideoControl`'s
//! mute; the C++ additionally silences scene sounds).
//!
//! `--noautomute` disables the detector entirely: [`AutoMute::disabled`] spawns
//! no thread and always reports not-playing (the base `AudioPlayingDetector`
//! never mutes, cpp table §1). Fullscreen-window detection (cpp step 2) lives in
//! the render/platform layer and is out of scope here — this monitor only
//! reports the sink-input signal.
//!
//! Threading follows the same shape as [`crate::AudioCapture`] (SPEC V3): all
//! PulseAudio state is `!Send` and lives inside a dedicated thread; the render
//! side reads a single lock-free [`AtomicBool`] snapshot (V4, never blocks). Any
//! failure — no PulseAudio server, connection error — is a graceful no-op (V9):
//! the flag stays `false` so the wallpaper simply never auto-mutes.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use libpulse_binding::callbacks::ListResult;
use libpulse_binding::context::{Context, FlagSet as ContextFlags, State as ContextState};
use libpulse_binding::mainloop::standard::{IterateResult, Mainloop};
use libpulse_binding::proplist::{Proplist, properties};
use libpulse_binding::volume::Volume;

use crate::AudioError;

/// Context name (cpp `PulseAudioPlayingDetector.cpp`, `"wallpaperengine"`).
const CONTEXT_NAME: &str = "wallpaperengine";

/// How often the detector re-lists sink inputs. The C++ engine polls once per
/// render frame (~16 ms); a coarser 200 ms cadence on a dedicated thread is
/// imperceptible for a mute toggle and keeps the introspection traffic tiny.
const POLL: Duration = Duration::from_millis(200);

/// Live handle to the automute detector. Owns the PulseAudio monitor thread
/// (joined on drop) and holds the lock-free "another app is playing" snapshot.
pub struct AutoMute {
    /// `true` ⇒ another application is emitting audio ⇒ mute the wallpaper.
    playing: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    enabled: bool,
    thread: Option<JoinHandle<()>>,
}

impl AutoMute {
    /// Start the detector. `enabled == false` (`--noautomute`) spawns no thread
    /// and always reports not-playing. Never fails: a missing PulseAudio server
    /// or any connection error leaves the flag `false` (V9). Returns
    /// immediately — the PulseAudio connection is established off-thread.
    #[must_use]
    pub fn start(enabled: bool) -> Self {
        let playing = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));

        if !enabled {
            return Self {
                playing,
                shutdown,
                enabled: false,
                thread: None,
            };
        }

        let thread = {
            let playing = playing.clone();
            let shutdown = shutdown.clone();
            std::thread::Builder::new()
                .name("kirie-automute".into())
                .spawn(move || {
                    if let Err(e) = run(&playing, &shutdown) {
                        // V9: without PulseAudio the detector is a no-op — the
                        // wallpaper is simply never auto-muted (matches the base
                        // `AudioPlayingDetector`, which never reports playing).
                        playing.store(false, Ordering::Relaxed);
                        // A shutdown-triggered exit is expected teardown, not a
                        // failure; only warn on a genuine PulseAudio error.
                        if !shutdown.load(Ordering::Relaxed) {
                            tracing::warn!(error = %e, "automute detector unavailable; wallpaper audio never auto-muted");
                        }
                    }
                })
                .expect("spawn automute thread")
        };

        Self {
            playing,
            shutdown,
            enabled: true,
            thread: Some(thread),
        }
    }

    /// A disabled detector (`--noautomute`): always reports not-playing.
    #[must_use]
    pub fn disabled() -> Self {
        Self::start(false)
    }

    /// Whether another application is currently emitting audio. Lock-free, never
    /// blocks the render thread (V4). Always `false` when disabled or when the
    /// PulseAudio connection failed.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    /// Whether the detector is active (`false` under `--noautomute`).
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled
    }
}

impl Drop for AutoMute {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

/// Drive one non-blocking `iterate` step, mapping quit/err to a typed error.
fn pump(mainloop: &mut Mainloop) -> Result<(), AudioError> {
    match mainloop.iterate(false) {
        IterateResult::Success(_) => Ok(()),
        IterateResult::Quit(_) | IterateResult::Err(_) => Err(AudioError::Mainloop),
    }
}

/// Iterate the mainloop until `check` returns `Some`, an error occurs, or the
/// shutdown flag trips. A short sleep between iterations avoids busy-spinning.
fn iterate_until<T>(
    mainloop: &mut Mainloop,
    shutdown: &AtomicBool,
    mut check: impl FnMut() -> Option<Result<T, AudioError>>,
) -> Result<T, AudioError> {
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return Err(AudioError::Mainloop);
        }
        pump(mainloop)?;
        if let Some(res) = check() {
            return res;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Sleep up to `dur`, waking early (within ~10 ms) if the shutdown flag trips so
/// a stopping detector joins promptly.
fn sleep_or_shutdown(shutdown: &AtomicBool, dur: Duration) {
    let step = Duration::from_millis(10);
    let mut slept = Duration::ZERO;
    while slept < dur {
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        std::thread::sleep(step);
        slept += step;
    }
}

/// Connect the context and poll the sink-input list until shutdown, publishing
/// the "another app is playing" flag. Errors leave the flag `false` (caller
/// logs).
fn run(playing: &AtomicBool, shutdown: &AtomicBool) -> Result<(), AudioError> {
    let mut mainloop = Mainloop::new().ok_or_else(|| AudioError::Connect("no mainloop".into()))?;

    let proplist = Proplist::new().ok_or_else(|| AudioError::Connect("no proplist".into()))?;
    let mut context = Context::new_with_proplist(&mainloop, CONTEXT_NAME, &proplist)
        .ok_or_else(|| AudioError::Connect("no context".into()))?;
    context
        .connect(None, ContextFlags::NOFLAGS, None)
        .map_err(|e| AudioError::Connect(format!("{e:?}")))?;

    // Block until PA_CONTEXT_READY (cpp construction blocks similarly).
    iterate_until(&mut mainloop, shutdown, || match context.get_state() {
        ContextState::Ready => Some(Ok(())),
        ContextState::Failed | ContextState::Terminated => {
            Some(Err(AudioError::Connect("context failed".into())))
        }
        _ => None,
    })?;

    // A sink input carrying our own pid is the wallpaper's own output, ignored.
    let own_pid = std::process::id();
    tracing::info!("automute detector running");

    let mut last: Option<bool> = None;
    while !shutdown.load(Ordering::Relaxed) {
        // Accumulate over one full sink-input list: `found` is set if any sink
        // input belongs to another process and is not muted.
        let found = Rc::new(Cell::new(false));
        let done = Rc::new(Cell::new(false));
        {
            let found = found.clone();
            let done = done.clone();
            let introspect = context.introspect();
            introspect.get_sink_input_info_list(move |res| match res {
                ListResult::Item(info) => {
                    // `application.process.id` differing from ours = another app
                    // (cpp:7-24). A sink input without the property is treated as
                    // "other" (conservative: prefer muting the wallpaper).
                    let pid = info
                        .proplist
                        .get_str(properties::APPLICATION_PROCESS_ID)
                        .and_then(|s| s.trim().parse::<u32>().ok());
                    let is_other = pid != Some(own_pid);
                    // Average volume != PA_VOLUME_MUTED ⇒ actually audible.
                    let audible = info.volume.avg() != Volume::MUTED;
                    if is_other && audible {
                        found.set(true);
                    }
                }
                ListResult::End | ListResult::Error => done.set(true),
            });
        }

        // Pump the mainloop until the list query completes (or shutdown trips).
        iterate_until(&mut mainloop, shutdown, || done.get().then_some(Ok(())))?;

        let now = found.get();
        if last != Some(now) {
            playing.store(now, Ordering::Relaxed);
            tracing::debug!(playing = now, "automute state changed");
            last = Some(now);
        }

        sleep_or_shutdown(shutdown, POLL);
    }

    Ok(())
}
