//! [`CefBackend`] — a [`WebBackend`] driven by Chromium Embedded Framework
//! off-screen rendering.
//!
//! CEF is single-threaded and process-global: every CEF object must live on
//! the one thread that called `cef_initialize`, and `cef_initialize` /
//! `cef_shutdown` may run at most once per process (docs/subsystems-misc.md
//! §3). So this backend owns a dedicated **CEF thread**: it initializes CEF,
//! creates the windowless browser, and pumps `cef_do_message_loop_work` in a
//! paced loop, servicing resize/pointer/mute/shutdown commands over a channel.
//! The render side only ever reads the lock-free frame slot the render handler
//! publishes into (SPEC §V4 — render never blocks on the browser).
//!
//! # Singleton
//!
//! Because CEF is a process singleton, only one [`CefBackend`] may be live at
//! a time; a second [`CefBackend::new`] returns [`WebError::AlreadyActive`].
//! (Multi-monitor web wallpapers would share one CEF context and one browser
//! per output — future work; the reference engine does exactly this with a
//! single `WebBrowserContext`.)

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use arc_swap::ArcSwapOption;
use cef::{
    BrowserSettings, CefString, ImplBrowser, ImplBrowserHost, ImplFrame, MouseButtonType, MouseEvent,
    Settings, WindowInfo, api_hash, args::Args, browser_host_create_browser_sync, do_message_loop_work,
    initialize, shutdown, sys::CEF_API_VERSION_LAST,
};

use crate::backend::{FrameBuffer, FrameSlot, PointerState, WebBackend, WebError, WebFrameRef, WebSize};

use super::client::{SharedSize, make_client};

/// Process-global "a CEF context is live" flag. CEF forbids a second
/// `cef_initialize`, so this gates construction rather than modelling app
/// state (SPEC §V1 permits an FFI-singleton guard for a C library that *is* a
/// singleton).
static CEF_ALIVE: AtomicBool = AtomicBool::new(false);

/// Target off-screen paint rate. The reference clamps to `max(60, fps)`
/// (docs/subsystems-misc.md §3.5); 60 is CEF's documented cap.
const FRAME_RATE: i32 = 60;

/// A command sent from the render side to the CEF thread.
enum Command {
    Resize(i32, i32),
    Pointer(PointerState),
    Mute(bool),
    Shutdown,
}

/// One-time configuration handed to the CEF thread at spawn.
struct ThreadConfig {
    url: String,
    muted: bool,
    runtime_dir: PathBuf,
    helper_path: Option<PathBuf>,
    slot: FrameSlot,
    size: Arc<SharedSize>,
    rx: Receiver<Command>,
}

/// A CEF-backed web wallpaper.
pub struct CefBackend {
    slot: FrameSlot,
    cmd_tx: Option<Sender<Command>>,
    thread: Option<JoinHandle<()>>,
    /// Cached handle on the latest published frame so [`Self::latest_frame`]
    /// can hand out a borrow tied to `&self`.
    cached: Option<Arc<FrameBuffer>>,
}

impl CefBackend {
    fn send(&self, cmd: Command) {
        if let Some(tx) = &self.cmd_tx {
            // A closed channel means the thread already exited; ignore.
            let _ = tx.send(cmd);
        }
    }
}

impl WebBackend for CefBackend {
    fn new(url: &str, size: WebSize) -> Result<Self, WebError> {
        let size = size.clamped();

        // Reserve the process-global CEF slot.
        if CEF_ALIVE
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(WebError::AlreadyActive);
        }

        let runtime_dir = resolve_runtime_dir()
            .ok_or_else(|| WebError::Init("could not locate the CEF runtime dir (icudtl.dat)".into()))?;
        let helper_path = resolve_helper_path(&runtime_dir);

        let slot: FrameSlot = Arc::new(ArcSwapOption::empty());
        let shared_size = SharedSize::new(size.width as i32, size.height as i32);
        let (tx, rx) = channel();

        let config = ThreadConfig {
            url: url.to_string(),
            muted: false,
            runtime_dir,
            helper_path,
            slot: slot.clone(),
            size: shared_size,
            rx,
        };

        // Result of CEF init/create is reported back so `new` can fail loudly.
        let (ready_tx, ready_rx) = channel::<Result<(), WebError>>();
        let thread = std::thread::Builder::new()
            .name("kirie-cef".into())
            .spawn(move || cef_thread_main(config, &ready_tx))
            .map_err(|e| {
                CEF_ALIVE.store(false, Ordering::SeqCst);
                WebError::Thread(e.to_string())
            })?;

        // Wait for the thread to report init success/failure.
        let outcome = ready_rx
            .recv()
            .unwrap_or_else(|_| Err(WebError::Init("CEF thread exited before init".into())));

        match outcome {
            Ok(()) => Ok(Self {
                slot,
                cmd_tx: Some(tx),
                thread: Some(thread),
                cached: None,
            }),
            Err(e) => {
                // Init failed; join the thread (which clears CEF_ALIVE via its
                // guard) and surface the error.
                let _ = thread.join();
                Err(e)
            }
        }
    }

    fn tick(&mut self, _dt: f32) {
        // Refresh the cached handle on the latest published frame. Cheap:
        // an atomic load + Arc clone, never a wait (SPEC §V4).
        if let Some(frame) = self.slot.load_full() {
            self.cached = Some(frame);
        }
    }

    fn latest_frame(&self) -> Option<WebFrameRef<'_>> {
        let frame = self.cached.as_ref()?;
        if !frame.is_consistent() {
            return None;
        }
        Some(WebFrameRef {
            data: &frame.data,
            width: frame.width,
            height: frame.height,
            format: frame.format,
        })
    }

    fn resize(&mut self, size: WebSize) {
        let size = size.clamped();
        self.send(Command::Resize(size.width as i32, size.height as i32));
    }

    fn send_pointer(&mut self, pointer: PointerState) {
        self.send(Command::Pointer(pointer));
    }

    fn set_muted(&mut self, muted: bool) {
        self.send(Command::Mute(muted));
    }

    fn shutdown(&mut self) {
        self.send(Command::Shutdown);
        self.cmd_tx = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for CefBackend {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The CEF thread: init → create browser → pump loop → shutdown.
fn cef_thread_main(config: ThreadConfig, ready: &Sender<Result<(), WebError>>) {
    // Release the process singleton whatever happens.
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            CEF_ALIVE.store(false, Ordering::SeqCst);
        }
    }
    let _guard = Guard;

    // Negotiate the CEF API version before constructing any CEF object. The
    // cef-rs wrappers stamp each struct's version from this global; skipping it
    // makes libcef reject the app with "invalid version -1".
    let _ = api_hash(CEF_API_VERSION_LAST, 0);

    let ThreadConfig {
        url,
        muted,
        runtime_dir,
        helper_path,
        slot,
        size,
        rx,
    } = config;

    // A disposable per-run Chrome profile dir; removed after `cef_shutdown`
    // so repeated runs (e.g. daemon restarts) do not leak profile trees under
    // the temp dir.
    let cache_dir = throwaway_cache_dir();

    // --- CefSettings (docs/subsystems-misc.md §3.2) -----------------------
    let mut settings = Settings {
        no_sandbox: 1,
        windowless_rendering_enabled: 1,
        // Running CEF on a spawned (non-main) thread: don't install signal
        // handlers, which assume the main thread.
        disable_signal_handlers: 1,
        command_line_args_disabled: 0,
        root_cache_path: CefString::from(cache_dir.to_string_lossy().as_ref()),
        resources_dir_path: CefString::from(runtime_dir.to_string_lossy().as_ref()),
        locales_dir_path: CefString::from(runtime_dir.join("locales").to_string_lossy().as_ref()),
        ..Default::default()
    };
    if let Some(helper) = &helper_path {
        settings.browser_subprocess_path = CefString::from(helper.to_string_lossy().as_ref());
    }

    let args = Args::new();
    let mut app = super::app::make_app();

    // `initialize` is the safe cef-rs wrapper over `cef_initialize`; it is
    // called exactly once per process on this thread (guarded by CEF_ALIVE),
    // the precondition the CEF C ABI requires.
    let init_ok = initialize(
        Some(args.as_main_args()),
        Some(&settings),
        Some(&mut app),
        std::ptr::null_mut(),
    );
    if init_ok != 1 {
        let _ = ready.send(Err(WebError::Init(
            "cef_initialize returned failure (missing libcef runtime files?)".into(),
        )));
        return;
    }

    // --- Create the windowless browser (docs §3.5) ------------------------
    let mut client = make_client(slot, size.clone());
    let window_info = WindowInfo::default().set_as_windowless(0);
    let browser_settings = BrowserSettings {
        windowless_frame_rate: FRAME_RATE,
        ..Default::default()
    };
    let url_str = CefString::from(url.as_str());

    let browser = browser_host_create_browser_sync(
        Some(&window_info),
        Some(&mut client),
        Some(&url_str),
        Some(&browser_settings),
        None,
        None,
    );

    let Some(browser) = browser else {
        let _ = ready.send(Err(WebError::BrowserCreation));
        shutdown();
        let _ = std::fs::remove_dir_all(&cache_dir);
        return;
    };

    // Apply the initial mute state.
    if muted && let Some(host) = browser.host() {
        host.set_audio_muted(1);
    }

    // Init succeeded — unblock `new`.
    let _ = ready.send(Ok(()));

    // --- Pump loop --------------------------------------------------------
    let frame_dt = Duration::from_secs_f64(1.0 / f64::from(FRAME_RATE));
    let mut pointer = PointerState::default();
    let mut last_left = false;
    let mut last_right = false;
    let audio_zero = [0.0f32; 128];

    'pump: loop {
        let frame_start = Instant::now();

        // Drain pending commands.
        loop {
            match rx.try_recv() {
                Ok(Command::Resize(w, h)) => {
                    size.set(w, h);
                    if let Some(host) = browser.host() {
                        host.was_resized();
                    }
                }
                Ok(Command::Pointer(p)) => pointer = p,
                Ok(Command::Mute(m)) => {
                    if let Some(host) = browser.host() {
                        host.set_audio_muted(i32::from(m));
                    }
                }
                Ok(Command::Shutdown) => break 'pump,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break 'pump,
            }
        }

        // Forward pointer + audio, mirroring CWeb::pushBridgeData (§3.5).
        if let Some(host) = browser.host() {
            let event = MouseEvent {
                x: pointer.x,
                y: pointer.y,
                modifiers: 0,
            };
            host.send_mouse_move_event(Some(&event), 0);
            if pointer.left != last_left {
                host.send_mouse_click_event(Some(&event), MouseButtonType::LEFT, i32::from(!pointer.left), 1);
                last_left = pointer.left;
            }
            if pointer.right != last_right {
                host.send_mouse_click_event(
                    Some(&event),
                    MouseButtonType::RIGHT,
                    i32::from(!pointer.right),
                    1,
                );
                last_right = pointer.right;
            }
        }
        if let Some(frame) = browser.main_frame() {
            let js = CefString::from(crate::shim::audio_call(&audio_zero).as_str());
            frame.execute_java_script(Some(&js), None, 0);
        }

        // Pump CEF once; OnPaint fires here on this thread.
        do_message_loop_work();

        // Pace to the target frame rate.
        if let Some(rem) = frame_dt.checked_sub(frame_start.elapsed()) {
            std::thread::sleep(rem);
        }
    }

    // --- Shutdown (docs §3.5): close, let the async close settle, shutdown.
    if let Some(host) = browser.host() {
        host.close_browser(1);
    }
    for _ in 0..10 {
        do_message_loop_work();
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(browser);
    shutdown();
    // CEF has fully torn down (all its threads joined); the throwaway profile
    // dir can be reclaimed.
    let _ = std::fs::remove_dir_all(&cache_dir);
}

/// Locate the directory holding the CEF runtime files (`libcef.so`,
/// `icudtl.dat`, `*.pak`, `locales/`). `cef-dll-sys`'s build script copies
/// these next to the built binary (docs §3.2: resources live beside the
/// executable). Honour `KIRIE_CEF_DIR` first for explicit control.
fn resolve_runtime_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("KIRIE_CEF_DIR") {
        let dir = PathBuf::from(dir);
        if dir.join("icudtl.dat").exists() {
            return Some(dir);
        }
    }
    let exe = std::env::current_exe().ok()?;
    let mut candidates = Vec::new();
    if let Some(dir) = exe.parent() {
        candidates.push(dir.to_path_buf());
        // Tests/examples run from `target/<profile>/deps`; the runtime files
        // are copied to `target/<profile>`.
        if let Some(parent) = dir.parent() {
            candidates.push(parent.to_path_buf());
        }
    }
    candidates.into_iter().find(|dir| dir.join("icudtl.dat").exists())
}

/// Locate the `kirie-cef-helper` subprocess binary. Honour `KIRIE_CEF_HELPER`,
/// else look beside the runtime files, else beside the current executable. If
/// none is found, CEF falls back to relaunching the current executable —
/// which is only correct if that binary also handles `--type=` first.
fn resolve_helper_path(runtime_dir: &Path) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("KIRIE_CEF_HELPER") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let named = runtime_dir.join("kirie-cef-helper");
    if named.exists() {
        return Some(named);
    }
    let exe = std::env::current_exe().ok()?;
    let beside = exe.parent()?.join("kirie-cef-helper");
    beside.exists().then_some(beside)
}

/// A throwaway per-run Chrome profile dir under the system temp dir
/// (docs §3.2: `root_cache_path` is a disposable uuid dir).
fn throwaway_cache_dir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("kirie-cef-{}-{nanos}", std::process::id()))
}
