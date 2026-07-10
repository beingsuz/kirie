//! [`ScriptEngine`] — the public, `Send` handle to a scene's script world.
//!
//! The world (QuickJS `Runtime` + `Context`) is `!Send` (SPEC.md §V3) so it lives
//! on a dedicated thread; this handle talks to it over a bounded
//! `crossbeam-channel` (commands in, typed results out). Dropping the handle
//! shuts the thread down.

use std::thread::JoinHandle;

use crossbeam_channel::{Sender, bounded};

use crate::error::ScriptError;
use crate::frame::{HostFrame, LogLine, TickOutput};
use crate::value::ScriptValue;
use crate::world::World;

/// SceneScript API level this engine targets — Wallpaper Engine's
/// `lib.sceneScript.d.ts` (docs/scripting-api.md §12). Reported so integrators
/// and bake keys can gate on the surface version.
pub const API_VERSION: &str = "2.8";

/// This translator/embedding's own version. Part of the bake cache key
/// (SPEC.md §V8): bumping it invalidates baked script results whose behavior
/// depends on the JS surface.
pub const TRANSLATOR_VERSION: u32 = 1;

/// Bounded command-queue depth (SPEC.md §V3: bounded back-pressure, never an
/// unbounded queue).
const QUEUE_DEPTH: usize = 64;

enum Command {
    Load {
        key: String,
        source: String,
        owner_id: Option<i64>,
        initial: ScriptValue,
        script_properties: serde_json::Value,
        reply: Sender<Result<(), ScriptError>>,
    },
    Tick {
        frame: Box<HostFrame>,
        overrides: Vec<(String, ScriptValue)>,
        reply: Sender<TickOutput>,
    },
    DispatchUserProperty {
        key: String,
        value: ScriptValue,
        reply: Sender<TickOutput>,
    },
    CreateLayerScript {
        source: String,
        script_properties: serde_json::Value,
        initial_text: String,
        reply: Sender<u32>,
    },
    TickLayer {
        handle: u32,
        time: f64,
        dt: f64,
        fps: f64,
        reply: Sender<Vec<LogLine>>,
    },
    LayerText {
        handle: u32,
        reply: Sender<String>,
    },
    DestroyLayer {
        handle: u32,
        reply: Sender<()>,
    },
    Eval {
        source: String,
        reply: Sender<Result<String, ScriptError>>,
    },
}

/// A handle to one scene's SceneScript world. `Send` and cheap to move; all work
/// happens on the owned script thread.
pub struct ScriptEngine {
    tx: Sender<Command>,
    thread: Option<JoinHandle<()>>,
}

impl ScriptEngine {
    /// Spawn a script world on its own thread. Fails only if the runtime or the
    /// embedded builtins fail to initialize (a bug, not user input).
    pub fn new() -> Result<Self, ScriptError> {
        let (tx, rx) = bounded::<Command>(QUEUE_DEPTH);
        let (ready_tx, ready_rx) = bounded::<Result<(), ScriptError>>(1);
        let thread = std::thread::Builder::new()
            .name("kirie-script".into())
            .spawn(move || {
                let mut world = match World::new() {
                    Ok(w) => {
                        let _ = ready_tx.send(Ok(()));
                        w
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                // Serve commands until the sender is dropped.
                while let Ok(cmd) = rx.recv() {
                    serve(&mut world, cmd);
                }
            })
            .map_err(|e| ScriptError::Internal(format!("spawn script thread: {e}")))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(ScriptEngine {
                tx,
                thread: Some(thread),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(ScriptError::ThreadGone),
        }
    }

    /// Load a property script (docs §4.2). `key` is the module key
    /// (`"<prop>_<objectId>"`); `owner_id` the scriptable layer id (for
    /// `thisLayer`); `initial` the property's initial value; `script_properties`
    /// the JSON `scriptproperties` bag.
    pub fn load_property_script(
        &self,
        key: impl Into<String>,
        source: impl Into<String>,
        owner_id: Option<i64>,
        initial: ScriptValue,
        script_properties: serde_json::Value,
    ) -> Result<(), ScriptError> {
        let (reply, rx) = bounded(1);
        self.send(Command::Load {
            key: key.into(),
            source: source.into(),
            owner_id,
            initial,
            script_properties,
            reply,
        })?;
        rx.recv().map_err(|_| ScriptError::ThreadGone)?
    }

    /// Run one frame (docs §3.2). `overrides` pushes user/integrator value
    /// changes for specific module keys before the tick.
    pub fn tick(
        &self,
        frame: HostFrame,
        overrides: Vec<(String, ScriptValue)>,
    ) -> Result<TickOutput, ScriptError> {
        let (reply, rx) = bounded(1);
        self.send(Command::Tick {
            frame: Box::new(frame),
            overrides,
            reply,
        })?;
        rx.recv().map_err(|_| ScriptError::ThreadGone)
    }

    /// Fire `applyUserProperties({key: value})` on every module (docs §5.3).
    pub fn dispatch_user_property(
        &self,
        key: impl Into<String>,
        value: ScriptValue,
    ) -> Result<TickOutput, ScriptError> {
        let (reply, rx) = bounded(1);
        self.send(Command::DispatchUserProperty {
            key: key.into(),
            value,
            reply,
        })?;
        rx.recv().map_err(|_| ScriptError::ThreadGone)
    }

    /// Create a text-layer script (docs §7); returns a positive handle, 0 on
    /// failure.
    pub fn create_layer_script(
        &self,
        source: impl Into<String>,
        script_properties: serde_json::Value,
        initial_text: impl Into<String>,
    ) -> Result<u32, ScriptError> {
        let (reply, rx) = bounded(1);
        self.send(Command::CreateLayerScript {
            source: source.into(),
            script_properties,
            initial_text: initial_text.into(),
            reply,
        })?;
        rx.recv().map_err(|_| ScriptError::ThreadGone)
    }

    /// Tick a text layer (docs §7.2); returns any console output produced.
    pub fn tick_layer(&self, handle: u32, time: f64, dt: f64, fps: f64) -> Result<Vec<LogLine>, ScriptError> {
        let (reply, rx) = bounded(1);
        self.send(Command::TickLayer {
            handle,
            time,
            dt,
            fps,
            reply,
        })?;
        rx.recv().map_err(|_| ScriptError::ThreadGone)
    }

    /// Read a text layer's current rendered text (docs §7.2).
    pub fn layer_text(&self, handle: u32) -> Result<String, ScriptError> {
        let (reply, rx) = bounded(1);
        self.send(Command::LayerText { handle, reply })?;
        rx.recv().map_err(|_| ScriptError::ThreadGone)
    }

    /// Destroy a text layer (docs §7.2), running its `destroy()` if present.
    pub fn destroy_layer(&self, handle: u32) -> Result<(), ScriptError> {
        let (reply, rx) = bounded(1);
        self.send(Command::DestroyLayer { handle, reply })?;
        rx.recv().map_err(|_| ScriptError::ThreadGone)
    }

    /// Evaluate an arbitrary global script, returning its value stringified
    /// (diagnostic / test helper).
    pub fn eval(&self, source: impl Into<String>) -> Result<String, ScriptError> {
        let (reply, rx) = bounded(1);
        self.send(Command::Eval {
            source: source.into(),
            reply,
        })?;
        rx.recv().map_err(|_| ScriptError::ThreadGone)?
    }

    fn send(&self, cmd: Command) -> Result<(), ScriptError> {
        self.tx.send(cmd).map_err(|_| ScriptError::ThreadGone)
    }
}

impl Drop for ScriptEngine {
    fn drop(&mut self) {
        // Dropping the sender ends the serve loop; join so the runtime is torn
        // down cleanly before we return.
        if let Some(thread) = self.thread.take() {
            drop(std::mem::replace(&mut self.tx, bounded(0).0));
            let _ = thread.join();
        }
    }
}

fn serve(world: &mut World, cmd: Command) {
    match cmd {
        Command::Load {
            key,
            source,
            owner_id,
            initial,
            script_properties,
            reply,
        } => {
            let r = world.load_property_script(&key, &source, owner_id, initial, &script_properties);
            let _ = reply.send(r);
        }
        Command::Tick {
            frame,
            overrides,
            reply,
        } => {
            let _ = reply.send(world.tick(&frame, &overrides));
        }
        Command::DispatchUserProperty { key, value, reply } => {
            let _ = reply.send(world.dispatch_user_property(&key, &value));
        }
        Command::CreateLayerScript {
            source,
            script_properties,
            initial_text,
            reply,
        } => {
            let _ = reply.send(world.create_layer_script(&source, &script_properties, &initial_text));
        }
        Command::TickLayer {
            handle,
            time,
            dt,
            fps,
            reply,
        } => {
            let _ = reply.send(world.tick_layer(handle, time, dt, fps));
        }
        Command::LayerText { handle, reply } => {
            let _ = reply.send(world.layer_text(handle));
        }
        Command::DestroyLayer { handle, reply } => {
            world.destroy_layer(handle);
            let _ = reply.send(());
        }
        Command::Eval { source, reply } => {
            let _ = reply.send(world.eval_to_string(&source));
        }
    }
}
