//! Audio-capture CLI wiring (docs/compat-cli.md §2; subsystems-misc.md §1.3).
//!
//! Asserts the parsed compat flags map to the right [`kirie_audio::AudioConfig`]
//! the run/screenshot paths hand to `AudioCapture::start`. No device or GPU is
//! touched — this is pure flag → config mapping.

use std::ffi::OsString;

use kirie::compat::{args, run};

fn os(parts: &[&str]) -> Vec<OsString> {
    parts.iter().map(OsString::from).collect()
}

fn config_for(extra: &[&str]) -> kirie_audio::AudioConfig {
    let mut argv = vec!["linux-wallpaperengine"];
    argv.extend_from_slice(extra);
    argv.push("/tmp/wallpaper");
    let parsed = args::parse(&os(&argv)).expect("parse compat argv");
    run::audio_config(&parsed)
}

#[test]
fn default_run_enables_default_monitor_capture() {
    let cfg = config_for(&[]);
    assert!(cfg.enabled, "reactive capture is on by default");
    assert_eq!(cfg.device, None, "default sink monitor when no --audio-device");
}

#[test]
fn no_audio_processing_disables_capture() {
    let cfg = config_for(&["--no-audio-processing"]);
    assert!(!cfg.enabled, "--no-audio-processing ⇒ permanent silent spectrum");
}

#[test]
fn audio_device_selects_source() {
    let cfg = config_for(&["--audio-device", "alsa_output.pci.monitor"]);
    assert!(cfg.enabled);
    assert_eq!(cfg.device.as_deref(), Some("alsa_output.pci.monitor"));
}

#[test]
fn silent_does_not_disable_reactive_capture() {
    // --silent mutes the wallpaper's own audio output, not the system-audio
    // reactive input (docs §2); capture stays enabled.
    let cfg = config_for(&["--silent"]);
    assert!(cfg.enabled, "--silent must not gate reactive capture");
}
