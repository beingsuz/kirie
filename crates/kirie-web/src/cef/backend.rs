//! [`CefBackend`] — a [`WebBackend`] driven by Chromium Embedded Framework
//! off-screen rendering.
//!
//! CEF is single-threaded and process-global: every CEF object must live on
//! the one thread that called `cef_initialize`, and `cef_do_message_loop_work`
//! must be pumped on that same thread (docs/subsystems-misc.md §3). So one
//! dedicated **CEF thread** owns the whole CEF lifetime: it initializes CEF,
//! creates windowless browsers on demand, and pumps `cef_do_message_loop_work`
//! in a paced loop, servicing create/resize/pointer/mute/props/close commands
//! over a channel (property batches are queued per browser and delivered after
//! that browser's first published paint — see
//! [`super::registry::BrowserEntry::drain_props_if_painted`]). The render side only ever reads the lock-free frame slot each
//! browser's render handler publishes into (SPEC §V4 — render never blocks on
//! the browser).
//!
//! # One CEF context, many browsers
//!
//! `cef_initialize` may run at most once at a time per process, but an
//! initialized context hosts **any number** of browsers — the reference engine
//! runs one browser per output through a single `WebBrowserContext`. Each
//! [`CefBackend`] therefore owns one *browser* (its id, frame slot, and shared
//! size), not the CEF context:
//!
//! * the **first** [`CefBackend::new`] spawns the CEF thread (which reserves
//!   the process-global context via [`MANAGER`]) and creates its browser on it;
//! * every **subsequent** `new` sends a [`Command::Create`] to the existing
//!   thread and gets back its own [`BrowserId`] + frame slot — this is what
//!   lets a second web wallpaper run on another monitor;
//! * [`CefBackend::shutdown`] closes only *its* browser; the shared context
//!   stays up for the others. Full `cef_shutdown` runs when the **last**
//!   backend drops (the thread quits, and a later `new` starts fresh —
//!   sequential re-init, same as the previous singleton behaviour).

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use arc_swap::ArcSwapOption;
use cef::{
    Browser, BrowserSettings, CefString, ImplBrowser, ImplBrowserHost, ImplFrame, MouseButtonType,
    MouseEvent, Settings, WindowInfo, api_hash, args::Args, browser_host_create_browser_sync,
    do_message_loop_work, initialize, shutdown, sys::CEF_API_VERSION_LAST,
};

use crate::backend::{FrameBuffer, FrameSlot, PointerState, WebBackend, WebError, WebFrameRef, WebSize};

use super::client::{SharedSize, make_client};
use super::registry::{BrowserEntry, BrowserId, BrowserRegistry};

/// Target off-screen paint rate. The reference clamps to `max(60, fps)`
/// (docs/subsystems-misc.md §3.5); 60 is CEF's documented cap.
const FRAME_RATE: i32 = 60;

/// A command sent from a backend handle to the shared CEF thread.
enum Command {
    /// Create a windowless browser; reply with its id (or the failure).
    Create(CreateRequest),
    /// Resize one browser's off-screen surface.
    Resize(BrowserId, i32, i32),
    /// Latest pointer sample for one browser.
    Pointer(BrowserId, PointerState),
    /// (Un)mute one browser's audio.
    Mute(BrowserId, bool),
    /// Deliver a `__wpApplyProps` JSON batch to one browser's page (queued in
    /// its registry entry until that browser's first published paint, then
    /// executed in order).
    ApplyProps(BrowserId, String),
    /// Close one browser (the context stays up); ack when done.
    Close(BrowserId, Sender<()>),
    /// Tear the whole context down (sent when the last backend drops).
    Quit,
}

/// Everything the CEF thread needs to create one browser.
struct CreateRequest {
    url: String,
    muted: bool,
    slot: FrameSlot,
    size: Arc<SharedSize>,
    reply: Sender<Result<BrowserId, WebError>>,
}

/// One-time configuration handed to the CEF thread at spawn.
struct ThreadConfig {
    runtime_dir: PathBuf,
    helper_path: Option<PathBuf>,
    rx: Receiver<Command>,
}

/// The process-global handle on the running CEF thread.
///
/// CEF forbids concurrent double-init, so thread spawn / reuse / teardown is
/// serialized through this mutex (SPEC §V1 permits an FFI-singleton guard for
/// a C library that *is* a singleton). The CEF thread itself never touches
/// [`MANAGER`] — it talks only through channels — so holding the lock across a
/// create/join cannot deadlock.
struct Manager {
    tx: Sender<Command>,
    thread: Option<JoinHandle<()>>,
    /// Number of live [`CefBackend`] handles sharing the thread.
    live: usize,
}

/// See [`Manager`].
static MANAGER: Mutex<Option<Manager>> = Mutex::new(None);

/// Lock [`MANAGER`], recovering from a poisoned lock (a panicking backend
/// must not permanently wedge web wallpapers).
fn manager_lock() -> std::sync::MutexGuard<'static, Option<Manager>> {
    MANAGER.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A detached last-backend teardown in flight (see [`CefBackend::shutdown`]).
/// [`CefBackend::new`] joins it before re-initializing so `cef_shutdown` and
/// `cef_initialize` can never overlap.
static TEARDOWN: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);

fn teardown_lock() -> std::sync::MutexGuard<'static, Option<JoinHandle<()>>> {
    TEARDOWN.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Ask the CEF thread to quit and join it, clearing the global slot.
fn stop_thread(guard: &mut Option<Manager>) {
    if let Some(mut mgr) = guard.take() {
        let _ = mgr.tx.send(Command::Quit);
        if let Some(thread) = mgr.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A CEF-backed web wallpaper: one windowless browser on the shared CEF thread.
pub struct CefBackend {
    id: BrowserId,
    tx: Sender<Command>,
    slot: FrameSlot,
    /// Cached handle on the latest published frame so [`Self::latest_frame`]
    /// can hand out a borrow tied to `&self`.
    cached: Option<Arc<FrameBuffer>>,
    /// `true` once [`Self::shutdown`] ran (it must be idempotent, and `Drop`
    /// calls it again).
    closed: bool,
}

impl CefBackend {
    fn send(&self, cmd: Command) {
        // A closed channel means the thread already exited; ignore.
        let _ = self.tx.send(cmd);
    }
}

impl WebBackend for CefBackend {
    fn new(url: &str, size: WebSize) -> Result<Self, WebError> {
        let size = size.clamped();

        let slot: FrameSlot = Arc::new(ArcSwapOption::empty());
        let shared_size = SharedSize::new(size.width as i32, size.height as i32);
        let (reply_tx, reply_rx) = channel();
        let request = CreateRequest {
            url: url.to_string(),
            muted: false,
            slot: slot.clone(),
            size: shared_size,
            reply: reply_tx,
        };

        // A detached last-backend teardown may still be running cef_shutdown;
        // join it first so initialize can never overlap shutdown (the segfault
        // window of rapid web-to-none-to-web switches).
        if let Some(handle) = teardown_lock().take() {
            let _ = handle.join();
        }

        // Serialize thread spawn/reuse and the create round-trip: the lock is
        // held until the browser exists so a concurrent shutdown of the last
        // sibling cannot tear the thread down under us.
        let mut guard = manager_lock();

        let tx = match guard.as_ref() {
            Some(mgr) => mgr.tx.clone(),
            None => {
                // First backend (or first after a full teardown): spawn the
                // CEF thread. It initializes CEF and then services commands.
                let runtime_dir = resolve_runtime_dir().ok_or_else(|| {
                    WebError::Init("could not locate the CEF runtime dir (icudtl.dat)".into())
                })?;
                let helper_path = resolve_helper_path(&runtime_dir);
                let (tx, rx) = channel();
                let config = ThreadConfig {
                    runtime_dir,
                    helper_path,
                    rx,
                };
                let thread = std::thread::Builder::new()
                    .name("kirie-cef".into())
                    .spawn(move || cef_thread_main(config))
                    .map_err(|e| WebError::Thread(e.to_string()))?;
                *guard = Some(Manager {
                    tx: tx.clone(),
                    thread: Some(thread),
                    live: 0,
                });
                tx
            }
        };

        if tx.send(Command::Create(request)).is_err() {
            // The thread died unexpectedly; reap it so the next attempt can
            // start fresh.
            stop_thread(&mut guard);
            return Err(WebError::Init("the CEF thread is gone".into()));
        }

        let outcome = reply_rx
            .recv()
            .unwrap_or_else(|_| Err(WebError::Init("the CEF thread exited during browser creation".into())));

        match outcome {
            Ok(id) => {
                if let Some(mgr) = guard.as_mut() {
                    mgr.live += 1;
                }
                Ok(Self {
                    id,
                    tx,
                    slot,
                    cached: None,
                    closed: false,
                })
            }
            Err(e) => {
                // If no sibling backend is live, don't leave a browserless
                // thread pumping forever (init failures also land here — the
                // thread has already exited and the join reaps it).
                if guard.as_ref().is_some_and(|mgr| mgr.live == 0) {
                    stop_thread(&mut guard);
                }
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
        self.send(Command::Resize(self.id, size.width as i32, size.height as i32));
    }

    fn send_pointer(&mut self, pointer: PointerState) {
        self.send(Command::Pointer(self.id, pointer));
    }

    fn set_muted(&mut self, muted: bool) {
        self.send(Command::Mute(self.id, muted));
    }

    fn apply_properties(&mut self, json: &str) {
        self.send(Command::ApplyProps(self.id, json.to_owned()));
    }

    fn shutdown(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;

        // Close this browser fire-and-forget: `shutdown` runs on the RENDER
        // thread (a swapped-out renderer drops there), and blocking on a close
        // ack froze compositing for up to CLOSE_WAIT during rapid web↔web /
        // web↔scene switches. The pump honours queued closes in order, so the
        // browser still goes down; siblings keep the context.
        let (done_tx, _done_rx) = channel();
        let _ = self.tx.send(Command::Close(self.id, done_tx));

        // Last backend out tears the whole context down — on a DETACHED thread:
        // `Quit` + join can take a second of cef_shutdown, which must never run
        // on the render thread. `CefBackend::new` joins this handle before any
        // re-init, so shutdown/initialize can never overlap.
        let mut guard = manager_lock();
        if let Some(mgr) = guard.as_mut() {
            mgr.live = mgr.live.saturating_sub(1);
            if mgr.live == 0
                && let Some(mut mgr) = guard.take()
            {
                let handle = std::thread::spawn(move || {
                    let _ = mgr.tx.send(Command::Quit);
                    if let Some(thread) = mgr.thread.take() {
                        let _ = thread.join();
                    }
                    // The browser runtime just released its heaps — return
                    // them to the kernel and page the idle library out.
                    kirie_bake::trim_heap();
                    kirie_bake::pageout_cold_libs();
                });
                *teardown_lock() = Some(handle);
            }
        }
    }
}

impl Drop for CefBackend {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The CEF thread: init → service the browser registry → shutdown.
fn cef_thread_main(config: ThreadConfig) {
    // Negotiate the CEF API version before constructing any CEF object. The
    // cef-rs wrappers stamp each struct's version from this global; skipping it
    // makes libcef reject the app with "invalid version -1".
    let _ = api_hash(CEF_API_VERSION_LAST, 0);

    let ThreadConfig {
        runtime_dir,
        helper_path,
        rx,
    } = config;

    // A disposable per-run Chrome profile dir, shared by every browser this
    // thread hosts; removed after `cef_shutdown` so repeated runs (e.g. daemon
    // restarts) do not leak profile trees under the temp dir.
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

    // `initialize` is the safe cef-rs wrapper over `cef_initialize`; MANAGER
    // guarantees at most one CEF thread is live, the precondition the CEF C
    // ABI requires.
    let init_ok = initialize(
        Some(args.as_main_args()),
        Some(&settings),
        Some(&mut app),
        std::ptr::null_mut(),
    );
    if init_ok != 1 {
        // Fail whoever is already waiting with the precise reason; later
        // senders see the closed channel when this thread returns.
        fail_pending(&rx);
        return;
    }

    // --- Pump loop --------------------------------------------------------
    let mut registry: BrowserRegistry<Browser> = BrowserRegistry::new();
    let frame_dt = Duration::from_secs_f64(1.0 / f64::from(FRAME_RATE));
    let audio_zero = [0.0f32; 128];

    // Pump iterations to let CEF settle a browser teardown before the next
    // command (esp. a Create) is drained — interleaving a synchronous create
    // with an in-flight async close is the observed segfault window.
    let mut settle: u32 = 0;

    'pump: loop {
        let frame_start = Instant::now();

        // Drain pending commands.
        while settle == 0 {
            match rx.try_recv() {
                Ok(Command::Create(req)) => match create_browser(&req) {
                    Some(browser) => {
                        let id = registry.insert(browser, req.size.clone(), req.slot.clone());
                        let _ = req.reply.send(Ok(id));
                    }
                    None => {
                        let _ = req.reply.send(Err(WebError::BrowserCreation));
                    }
                },
                Ok(Command::Resize(id, w, h)) => {
                    if let Some(entry) = registry.get_mut(id) {
                        entry.size.set(w, h);
                        if let Some(host) = entry.browser.host() {
                            host.was_resized();
                        }
                    }
                }
                Ok(Command::Pointer(id, p)) => {
                    if let Some(entry) = registry.get_mut(id) {
                        entry.set_pointer(p);
                    }
                }
                Ok(Command::Mute(id, m)) => {
                    if let Some(entry) = registry.get_mut(id)
                        && let Some(host) = entry.browser.host()
                    {
                        host.set_audio_muted(i32::from(m));
                    }
                }
                Ok(Command::ApplyProps(id, json)) => {
                    if let Some(entry) = registry.get_mut(id) {
                        entry.push_props(json);
                    }
                }
                Ok(Command::Close(id, done)) => {
                    if let Some(entry) = registry.remove(id) {
                        close_browser(entry.browser);
                        // Let a few message-loop iterations run before the next
                        // command (esp. a Create) so the teardown settles.
                        settle = 4;
                    }
                    let _ = done.send(());
                }
                Ok(Command::Quit) => break 'pump,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break 'pump,
            }
        }

        // Forward pointer + audio to every live browser, mirroring
        // CWeb::pushBridgeData (§3.5).
        for (_, entry) in registry.iter_mut() {
            drive_browser(entry, &audio_zero);
        }

        // Pump CEF once; every browser's OnPaint fires here on this thread.
        do_message_loop_work();
        settle = settle.saturating_sub(1);

        // Pace to the target frame rate.
        if let Some(rem) = frame_dt.checked_sub(frame_start.elapsed()) {
            std::thread::sleep(rem);
        }
    }

    // --- Shutdown (docs §3.5): close the survivors, then pump until every
    // browser fired OnBeforeClose. `cef_shutdown` with a browser still alive
    // hangs Chromium's thread teardown — the runtime (30+ threads, zygote
    // subprocesses, V8 heaps: hundreds of MB) then outlives the web wallpaper.
    // Bounded at 5s; a fixed 10-iteration settle was not a guarantee.
    for (_, entry) in registry.drain() {
        close_browser(entry.browser);
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while super::client::LIVE_BROWSERS.load(std::sync::atomic::Ordering::SeqCst) > 0
        && Instant::now() < deadline
    {
        do_message_loop_work();
        std::thread::sleep(Duration::from_millis(5));
    }
    // A few extra iterations so post-close CEF tasks run before shutdown.
    for _ in 0..10 {
        do_message_loop_work();
        std::thread::sleep(Duration::from_millis(5));
    }
    tracing::info!(
        remaining = super::client::LIVE_BROWSERS.load(std::sync::atomic::Ordering::SeqCst),
        "cef context shutting down"
    );
    shutdown();
    tracing::info!("cef context shut down; browser runtime released");
    // CEF has fully torn down (all its threads joined); the throwaway profile
    // dir can be reclaimed.
    let _ = std::fs::remove_dir_all(&cache_dir);
}

/// Create one windowless browser for `req` (docs §3.5). Must run on the CEF
/// thread, after a successful `initialize`.
fn create_browser(req: &CreateRequest) -> Option<Browser> {
    let mut client = make_client(req.slot.clone(), req.size.clone());
    let window_info = WindowInfo::default().set_as_windowless(0);
    let browser_settings = BrowserSettings {
        windowless_frame_rate: FRAME_RATE,
        ..Default::default()
    };
    let url_str = CefString::from(req.url.as_str());

    let browser = browser_host_create_browser_sync(
        Some(&window_info),
        Some(&mut client),
        Some(&url_str),
        Some(&browser_settings),
        None,
        None,
    )?;

    // Apply the initial mute state.
    if req.muted && let Some(host) = browser.host() {
        host.set_audio_muted(1);
    }
    Some(browser)
}

/// Force-close one browser and drop our handle; the ongoing pump services the
/// asynchronous close while the context (and any sibling browsers) stay up.
fn close_browser(browser: Browser) {
    if let Some(host) = browser.host() {
        host.close_browser(1);
    }
    drop(browser);
}

/// Per-frame per-browser bridge data: pointer move, click edges, the audio
/// spectrum call (silent for now), and any deliverable property batches,
/// mirroring CWeb::pushBridgeData (§3.5).
fn drive_browser(entry: &mut BrowserEntry<Browser>, audio_zero: &[f32]) {
    let pointer = entry.pointer();
    let left_edge = entry.left_edge();
    let right_edge = entry.right_edge();
    if let Some(host) = entry.browser.host() {
        let event = MouseEvent {
            x: pointer.x,
            y: pointer.y,
            modifiers: 0,
        };
        host.send_mouse_move_event(Some(&event), 0);
        if let Some(down) = left_edge {
            host.send_mouse_click_event(Some(&event), MouseButtonType::LEFT, i32::from(!down), 1);
        }
        if let Some(down) = right_edge {
            host.send_mouse_click_event(Some(&event), MouseButtonType::RIGHT, i32::from(!down), 1);
        }
    }
    if let Some(frame) = entry.browser.main_frame() {
        let js = CefString::from(crate::shim::audio_call(audio_zero).as_str());
        frame.execute_java_script(Some(&js), None, 0);
        // Queued property batches are released only after *this* browser's
        // first published paint (its own frame slot is non-empty; see
        // `BrowserEntry::drain_props_if_painted`), then executed in order on
        // its main frame. Draining inside the `main_frame` arm keeps batches
        // queued when the frame does not exist yet.
        for json in entry.drain_props_if_painted() {
            let call = CefString::from(crate::shim::apply_user_properties_call(&json).as_str());
            frame.execute_java_script(Some(&call), None, 0);
        }
    }
}

/// After a failed `cef_initialize`: answer the commands already queued (a
/// waiting `new` gets the precise error instead of a bare disconnect).
fn fail_pending(rx: &Receiver<Command>) {
    while let Ok(cmd) = rx.try_recv() {
        match cmd {
            Command::Create(req) => {
                let _ = req.reply.send(Err(WebError::Init(
                    "cef_initialize returned failure (missing libcef runtime files?)".into(),
                )));
            }
            Command::Close(_, done) => {
                let _ = done.send(());
            }
            _ => {}
        }
    }
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
