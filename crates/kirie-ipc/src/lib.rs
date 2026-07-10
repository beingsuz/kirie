//! kirie-ipc — drop-in `linux-wallpaperengine` control socket.
//!
//! Byte-exact reimplementation of the engine's `--control-socket` protocol as
//! specified in `docs/compat-socket.md` (reverse-engineered from the C++
//! `ControlSocket.{h,cpp}` and byte-verified against a live engine, doc §9):
//! one LF-terminated request line per connection, one response, close.
//!
//! Architecture (SPEC V3/V4):
//!
//! - [`ControlSocket::bind`] spawns one dedicated socket thread that owns the
//!   `UnixListener`. Nothing else touches the socket.
//! - Parsed requests are delivered to the app as typed [`Command`] values over
//!   a crossbeam channel ([`IpcEvent`]).
//! - Replies travel back over a per-request `bounded(1)` channel carrying
//!   either a [`CommandOutcome`] or an immutable [`StatusSnapshot`]. No state
//!   is shared between threads.
//! - Unlike the C++ engine, which services the socket inline on the render
//!   thread (doc §5), a stalled client here can only stall the socket thread,
//!   and only up to a bounded per-connection deadline — the render thread is
//!   never involved (SPEC V4).

#![forbid(unsafe_code)]

mod command;
mod error;
mod event;
mod server;
mod status;

pub use command::{ClampMode, Command, Request, ScalingMode, SetOption, parse_request};
pub use error::IpcError;
pub use event::{CommandOutcome, IpcEvent};
pub use server::ControlSocket;
pub use status::{ScreenStatus, StatusSnapshot};
