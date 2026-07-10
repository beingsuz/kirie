//! MPRIS `Metadata` dict decoding.
//!
//! MPRIS exposes track metadata as a D-Bus `a{sv}` map — a `String → Variant`
//! dictionary. This module turns that (a [`HashMap<String, OwnedValue>`] as
//! produced by a `Properties.Get("Metadata")` deserialize) into a typed
//! [`TrackMetadata`], reading exactly the keys the C++ reference reads
//! (docs/subsystems-misc.md §5, `DBusMediaSource.cpp:87-184`):
//!
//! * `xesam:title`   — string
//! * `xesam:artist`  — array of strings, **first entry only** (a bare string is
//!   also accepted for players that violate the spec)
//! * `xesam:album`   — string
//! * `mpris:artUrl`  — string (an empty string resets the URL)
//! * `mpris:length`  — int64 microseconds → duration
//!
//! Every extractor is total and tolerant: an absent key, a wrong-typed value or
//! an empty array yields the field default rather than an error (SPEC §V9 — the
//! parser never panics on malformed input).

use std::collections::HashMap;

use zbus::zvariant::{OwnedValue, Value};

use super::state::TrackMetadata;

/// Interpret a [`Value`] as a UTF-8 string, unwrapping a boxed `Variant`
/// (`Value::Value`) one level. Returns `None` for non-string values.
fn as_str(value: &Value<'_>) -> Option<String> {
    match value {
        Value::Str(s) => Some(s.as_str().to_owned()),
        // Some senders wrap the payload in an extra variant.
        Value::Value(inner) => as_str(inner),
        _ => None,
    }
}

/// Interpret a [`Value`] as a signed 64-bit integer, accepting every D-Bus
/// integer width (players disagree on whether `mpris:length` is `x`/`t`/`i`…).
fn as_i64(value: &Value<'_>) -> Option<i64> {
    match value {
        Value::I64(v) => Some(*v),
        Value::U64(v) => i64::try_from(*v).ok(),
        Value::I32(v) => Some(i64::from(*v)),
        Value::U32(v) => Some(i64::from(*v)),
        Value::I16(v) => Some(i64::from(*v)),
        Value::U16(v) => Some(i64::from(*v)),
        Value::U8(v) => Some(i64::from(*v)),
        // A double microsecond length (rare, non-conforming) is truncated.
        Value::F64(v) if v.is_finite() => Some(*v as i64),
        Value::Value(inner) => as_i64(inner),
        _ => None,
    }
}

/// Interpret a [`Value`] as "the first string of a string array", per the C++
/// `xesam:artist` handling (first entry only, `DBusMediaSource.cpp:87-184`).
/// Falls back to treating a bare string value as a single-element list.
fn first_str_of_array(value: &Value<'_>) -> Option<String> {
    match value {
        Value::Array(arr) => arr.iter().find_map(as_str),
        Value::Value(inner) => first_str_of_array(inner),
        // Non-conforming players sometimes send a bare string.
        other => as_str(other),
    }
}

/// Decode an MPRIS `Metadata` dict into a [`TrackMetadata`].
///
/// Unknown keys are ignored; missing/ill-typed values leave the field at its
/// default. An empty `mpris:artUrl` string resets the art URL to `None`
/// (matching the C++ "emptied → url reset", docs/subsystems-misc.md §5).
#[must_use]
pub fn parse_metadata(dict: &HashMap<String, OwnedValue>) -> TrackMetadata {
    let mut meta = TrackMetadata::default();

    if let Some(v) = dict.get("xesam:title") {
        meta.title = as_str(v).unwrap_or_default();
    }
    if let Some(v) = dict.get("xesam:artist") {
        meta.artist = first_str_of_array(v).unwrap_or_default();
    }
    if let Some(v) = dict.get("xesam:album") {
        meta.album = as_str(v).unwrap_or_default();
    }
    if let Some(v) = dict.get("mpris:artUrl") {
        // Empty string → no art (reset), matching the reference.
        meta.art_url = as_str(v).filter(|s| !s.is_empty());
    }
    if let Some(v) = dict.get("mpris:length") {
        meta.length_us = as_i64(v);
    }

    meta
}

#[cfg(test)]
mod tests {
    use super::*;
    use zbus::zvariant::Array;

    /// Build an `a{sv}`-style dict from `(key, Value)` pairs.
    fn dict(entries: Vec<(&str, Value<'static>)>) -> HashMap<String, OwnedValue> {
        entries
            .into_iter()
            .map(|(k, v)| (k.to_owned(), OwnedValue::try_from(v).expect("owned value")))
            .collect()
    }

    fn str_array(items: &[&str]) -> Value<'static> {
        let owned: Vec<String> = items.iter().map(|s| (*s).to_owned()).collect();
        Value::Array(Array::from(owned))
    }

    #[test]
    fn parses_full_spotify_style_dict() {
        let d = dict(vec![
            ("xesam:title", Value::from("Blinding Lights")),
            ("xesam:artist", str_array(&["The Weeknd", "Ignored Second"])),
            ("xesam:album", Value::from("After Hours")),
            ("mpris:artUrl", Value::from("file:///tmp/cover.jpg")),
            ("mpris:length", Value::I64(200_040_000)),
            // An unrelated key the parser must ignore.
            ("mpris:trackid", Value::from("/track/1")),
        ]);
        let meta = parse_metadata(&d);
        assert_eq!(meta.title, "Blinding Lights");
        // Only the FIRST artist survives.
        assert_eq!(meta.artist, "The Weeknd");
        assert_eq!(meta.album, "After Hours");
        assert_eq!(meta.art_url.as_deref(), Some("file:///tmp/cover.jpg"));
        assert_eq!(meta.length_us, Some(200_040_000));
        assert!(!meta.is_empty());
    }

    #[test]
    fn empty_dict_is_empty_metadata() {
        let meta = parse_metadata(&HashMap::new());
        assert_eq!(meta, TrackMetadata::default());
        assert!(meta.is_empty());
    }

    #[test]
    fn empty_art_url_resets_to_none() {
        let d = dict(vec![("mpris:artUrl", Value::from(""))]);
        assert_eq!(parse_metadata(&d).art_url, None);
    }

    #[test]
    fn accepts_bare_string_artist() {
        // Non-conforming players send xesam:artist as a plain string.
        let d = dict(vec![("xesam:artist", Value::from("Solo Artist"))]);
        assert_eq!(parse_metadata(&d).artist, "Solo Artist");
    }

    #[test]
    fn empty_artist_array_yields_empty_string() {
        let d = dict(vec![("xesam:artist", str_array(&[]))]);
        assert_eq!(parse_metadata(&d).artist, "");
    }

    #[test]
    fn length_accepts_alternate_int_widths() {
        // u64 length (some players use `t`).
        let d = dict(vec![("mpris:length", Value::U64(123_456))]);
        assert_eq!(parse_metadata(&d).length_us, Some(123_456));
        // i32 length.
        let d = dict(vec![("mpris:length", Value::I32(42))]);
        assert_eq!(parse_metadata(&d).length_us, Some(42));
    }

    #[test]
    fn wrong_typed_values_fall_back_to_defaults() {
        // title as an int, length as a string → both ignored, no panic (V9).
        let d = dict(vec![
            ("xesam:title", Value::I64(7)),
            ("mpris:length", Value::from("not a number")),
        ]);
        let meta = parse_metadata(&d);
        assert_eq!(meta.title, "");
        assert_eq!(meta.length_us, None);
    }

    #[test]
    fn playback_status_mapping() {
        use super::super::state::PlaybackState;
        assert_eq!(PlaybackState::from_mpris("Playing"), PlaybackState::Playing);
        assert_eq!(PlaybackState::from_mpris("Paused"), PlaybackState::Paused);
        assert_eq!(PlaybackState::from_mpris("Stopped"), PlaybackState::Stopped);
        // Unknown → Stopped, no panic.
        assert_eq!(PlaybackState::from_mpris("Frobnicating"), PlaybackState::Stopped);
        assert_eq!(PlaybackState::Playing.as_i32(), 1);
        assert_eq!(PlaybackState::Paused.as_i32(), 2);
        assert_eq!(PlaybackState::Stopped.as_i32(), 0);
    }
}
