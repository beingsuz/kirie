//! Live demo of the compat control socket with a fake single-screen app.
//!
//! ```sh
//! cargo run -p kirie-ipc --example compat_server -- /tmp/kirie-ipc-demo.sock
//! printf 'status\n' | socat - UNIX-CONNECT:/tmp/kirie-ipc-demo.sock
//! ```
//!
//! The fake state mirrors the live capture in
//! `fixtures/socket-live-capture.txt` so responses can be diffed 1:1.

use std::path::PathBuf;

use kirie_ipc::{Command, CommandOutcome, ControlSocket, IpcEvent, ScreenStatus, StatusSnapshot};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/kirie-ipc-demo.sock".to_string());
    let (tx, rx) = crossbeam_channel::unbounded();
    let _server = ControlSocket::bind(PathBuf::from(&path), tx).expect("bind control socket");
    println!("listening on {path} (ctrl-c to quit)");
    let mut speed = 1.0f32;
    for event in rx {
        match event {
            IpcEvent::Status { reply } => {
                let _ = reply.send(StatusSnapshot {
                    speed,
                    screens: vec![ScreenStatus {
                        screen: "HDMI-A-1".into(),
                        bg: Some(PathBuf::from(
                            "/home/aiko/.local/share/Steam/steamapps/workshop/content/431960/3047596375",
                        )),
                    }],
                });
            }
            IpcEvent::GetProperties { screen, reply } => {
                // kirie extension (docs/compat-socket.md §11): a real app returns
                // the loaded scene's property schema; the demo has none.
                println!("getproperties: {screen:?}");
                let _ = reply.send("[]".to_string());
            }
            IpcEvent::Command { command, reply } => {
                println!("command: {command:?}");
                if let Command::Speed(v) = command {
                    speed = v;
                }
                let _ = reply.send(CommandOutcome::Ok);
            }
        }
    }
}
