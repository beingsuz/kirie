//! `project.json` manifest model. Spec: docs/format-project-json.md
//!
//! Typed model of the Wallpaper Engine workshop-item manifest. Parsing follows
//! the C++ reference parser exactly (fatal cases, defaults, coercion — see the
//! per-item citations below); everything the renderer does not read is
//! preserved verbatim in `extra` maps so a model round-trips through serde
//! (docs/format-project-json.md §2.2: unknown keys tolerated).
//!
//! Round-trip contract: `Project::from_value(project.to_value()) == project`.
//! Byte-level file fidelity is *not* a goal — defaults the C++ would infer are
//! written out explicitly on re-serialization.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Number, Value};
use thiserror::Error;

/// Errors produced while loading or validating a `project.json` manifest.
///
/// The fatal cases mirror the C++ reference parser: `require(...)` failures
/// abort loading that background (docs/format-project-json.md §2.1).
#[derive(Debug, Error)]
pub enum ProjectError {
    /// The manifest file could not be read from disk.
    #[error("cannot read {path}: {source}")]
    Io {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The bytes are not valid JSON. docs/format-project-json.md §1: strict
    /// JSON only — no comments, no trailing commas, no BOM.
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The manifest root is not a JSON object.
    #[error("project.json root is not a JSON object")]
    NotAnObject,
    /// `title` key absent — fatal per docs/format-project-json.md §2.1
    /// (C++ `"Project title missing"`, ProjectParser.cpp:40).
    #[error("project title missing")]
    TitleMissing,
    /// `title` present but not a JSON string — string targets are not coerced,
    /// so the C++ throws a type error and aborts (docs/format-project-json.md
    /// §2.1 title row, §8).
    #[error("project title must be a string")]
    TitleNotString,
    /// `file` key absent — fatal per docs/format-project-json.md §2.1
    /// (C++ `"Project's main file missing"`, ProjectParser.cpp:23).
    #[error("project's main file missing")]
    FileMissing,
    /// `file` present but not a JSON string. The C++ tolerates this for type
    /// detection only (treats it as `""`) and then fails downstream, so it
    /// "effectively must be a string" (docs/format-project-json.md §2.1 file
    /// row); we reject it up front with a typed error.
    #[error("project's main file must be a string")]
    FileNotString,
    /// Neither the main file's extension/URL scheme nor the declared `type`
    /// string identifies the wallpaper type — fatal per
    /// docs/format-project-json.md §3.1 step 4 (C++ `"Cannot determine project
    /// type from file …"`, ProjectParser.cpp:99).
    #[error("cannot determine project type from file {file:?}")]
    TypeUndeterminable {
        /// The manifest's `file` value.
        file: String,
    },
    /// A declared user property failed to parse
    /// (docs/format-project-json.md §4).
    #[error("property {name:?}: {source}")]
    Property {
        /// Key of the offending property in `general.properties`.
        name: String,
        /// The underlying property error.
        source: PropertyError,
    },
}

/// Errors in a single `general.properties` entry
/// (docs/format-project-json.md §4.3 required fields).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PropertyError {
    /// A field the C++ `require(...)`s is absent — fatal
    /// (docs/format-project-json.md §4.3).
    #[error("required field {field:?} missing")]
    MissingField {
        /// Name of the missing field.
        field: &'static str,
    },
    /// A field has a JSON type the C++ would throw on
    /// (docs/format-project-json.md §4.3, §8: non-coercible target).
    #[error("field {field:?} has the wrong JSON type (expected {expected})")]
    WrongType {
        /// Name of the offending field.
        field: &'static str,
        /// What the parser expected there.
        expected: &'static str,
    },
    /// A combo's `options` value is not a JSON array — fatal
    /// (docs/format-project-json.md §4.3 combo).
    #[error("combo `options` must be an array")]
    OptionsNotArray,
    /// A color property's `value` string failed to parse
    /// (docs/format-project-json.md §5).
    #[error("invalid color value: {0}")]
    Color(#[from] ColorError),
}

/// Errors from the property-color string parser
/// (docs/format-project-json.md §5).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ColorError {
    /// `#`-hex digit string is not 3, 4, 6 or 8 digits long — fatal per
    /// docs/format-project-json.md §5 step 2 (ColorBuilder.cpp:39).
    #[error("unsupported hex color length {len} (expected 3, 4, 6 or 8 digits)")]
    HexLength {
        /// Number of hex digits after `#`.
        len: usize,
    },
    /// `#`-hex string contains non-hex characters — the C++ `std::stoi` throws
    /// and aborts (docs/format-project-json.md §5 step 2).
    #[error("invalid hex digits in color {digits:?}")]
    HexDigits {
        /// The offending digit string.
        digits: String,
    },
    /// Space-separated vector with a component count other than 3 or 4 —
    /// `std::invalid_argument` in the C++ (docs/format-project-json.md §5
    /// step 3, ColorBuilder.cpp:52–55).
    #[error("color has {count} components (expected 3 or 4)")]
    ComponentCount {
        /// Number of space-separated components found.
        count: usize,
    },
}

/// Resolved wallpaper type (docs/format-project-json.md §3, Project.h:15).
///
/// `Type_Unknown` is never produced by the parser (§3.1: resolution failure is
/// a fatal error instead), so it has no variant here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WallpaperType {
    /// Scene wallpaper — `file` is the scene descriptor JSON (§3.2).
    Scene,
    /// Web wallpaper — `file` is an HTML entry point or bare URL (§3.2).
    Web,
    /// Video wallpaper — `file` is a media file (§3.2).
    Video,
    /// Image wallpaper — rejected by the reference fork (§3.1/§3.2).
    Image,
    /// Application wallpaper — Windows-only, rejected by the reference fork
    /// (§3.1/§3.2).
    Application,
}

/// The raw declared `type` string, case-preserved.
///
/// docs/format-project-json.md §3.1: matching is case-insensitive (corpus has
/// `"Scene"`, `"Web"`); the declared string is only a fallback label for type
/// detection but the playlist preflight requires the key's presence (§2.1), so
/// unknown values are preserved rather than rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclaredType(pub String);

impl DeclaredType {
    /// Case-insensitive classification of the declared string per
    /// docs/format-project-json.md §3.1 step 3 (ProjectParser.cpp:83–97).
    /// Unknown strings (including `""`) yield `None`.
    pub fn classify(&self) -> Option<WallpaperType> {
        let lower = self.0.to_ascii_lowercase();
        match lower.as_str() {
            "scene" => Some(WallpaperType::Scene),
            "video" => Some(WallpaperType::Video),
            "web" => Some(WallpaperType::Web),
            "application" => Some(WallpaperType::Application),
            "image" => Some(WallpaperType::Image),
            _ => None,
        }
    }
}

/// `workshopid` is a string or a number in real files
/// (docs/format-project-json.md §2.1) — an untagged union.
///
/// Stored losslessly. Note: the C++ narrows numbers through
/// `std::to_string(get<int>())` (§2.1); we keep the full number instead and
/// normalize via [`WorkshopId::to_string`]. Other JSON types are not
/// representable: the C++ logs an error and substitutes an app-level synthetic
/// ID, so the parser maps them to "absent" (see [`Project::workshopid`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WorkshopId {
    /// Verbatim string form (all 17 corpus occurrences).
    Text(String),
    /// Numeric form — converted to a decimal string by WE (§2.1).
    Number(Number),
}

impl std::fmt::Display for WorkshopId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkshopId::Text(s) => f.write_str(s),
            WorkshopId::Number(n) => write!(f, "{n}"),
        }
    }
}

/// One option of a combo property (docs/format-project-json.md §4.3 combo).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ComboOption {
    /// Display text — required, string (§4.3).
    pub label: String,
    /// Option value — required; a JSON number is converted to its decimal
    /// integer string (§4.3, PropertyParser.cpp:66–67).
    pub value: String,
    /// Fields of the option object not read by the C++ — preserved verbatim.
    pub extra: Map<String, Value>,
}

/// Per-type payload of a user property (docs/format-project-json.md §4.3).
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyKind {
    /// `"bool"` — `value` coerced per §8, default `false` (§4.3,
    /// PropertyParser.cpp:112–121).
    Bool {
        /// Current/default state.
        value: bool,
    },
    /// `"slider"` — `value` required and *not* coerced (JSON number or bool
    /// only; a numeric string is a fatal type error, §4.3
    /// PropertyParser.cpp:123–137). `min`/`max`/`step` default `0.0` and go
    /// through §8 coercion (numeric strings accepted).
    Slider {
        /// Current/default value (required).
        value: f32,
        /// Lower bound, default `0.0` (§4.3).
        min: f32,
        /// Upper bound, default `0.0` (§4.3).
        max: f32,
        /// UI step, default `0.0` (§4.3).
        step: f32,
    },
    /// `"color"` — `value` required, must be a JSON string, parsed per §5 with
    /// float parsing forced (`"1 1 1"` is white, never 1/255-gray) and stored
    /// as `[r, g, b]` in 0..1 (§4.3, PropertyParser.cpp:101–110;
    /// Property.h:131–135).
    Color {
        /// Parsed RGB triple, each component nominally in 0..1.
        value: [f32; 3],
    },
    /// `"combo"` — `options` required; `value` defaults to the first option's
    /// value (empty options ⇒ `""`) (§4.3, PropertyParser.cpp:48–99).
    Combo {
        /// Declared options in file order; non-object entries were skipped
        /// (§4.3, PropertyParser.cpp:62–64).
        options: Vec<ComboOption>,
        /// Selected value.
        value: String,
    },
    /// `"text"` — static label; only `text` is read, `value` and even `order`
    /// are ignored (order stays 0) (§4.3, PropertyParser.cpp:139–144).
    Text,
    /// `"textinput"` — `value` required, any JSON; a JSON string stores its
    /// contents, anything else stores its compact JSON serialization
    /// (the doc's clean-implementation reading of the C++ `value.dump()`
    /// quirk, §4.3 PropertyParser.cpp:168–177).
    TextInput {
        /// Current/default text.
        value: String,
    },
    /// `"usershortcut"` — parsed exactly like `textinput` (§4.1,
    /// PropertyParser.cpp:35), kept as its own variant so the tag round-trips.
    UserShortcut {
        /// Current/default text.
        value: String,
    },
    /// `"file"` — `value` optional string, default `""` (§4.3,
    /// PropertyParser.cpp:157–166).
    File {
        /// User-chosen file path; empty until the user picks one.
        value: String,
    },
    /// `"directory"` — same parser as `"file"` (§4.1, PropertyParser.cpp:29).
    Directory {
        /// User-chosen directory path.
        value: String,
    },
    /// `"scenetexture"` — `value` required string; no corpus instances,
    /// shape from source only (§4.3, PropertyParser.cpp:146–155).
    SceneTexture {
        /// Texture path.
        value: String,
    },
}

impl PropertyKind {
    /// The §4.1 dispatch tag this kind was parsed from and serializes back to.
    pub fn type_tag(&self) -> &'static str {
        match self {
            PropertyKind::Bool { .. } => "bool",
            PropertyKind::Slider { .. } => "slider",
            PropertyKind::Color { .. } => "color",
            PropertyKind::Combo { .. } => "combo",
            PropertyKind::Text => "text",
            PropertyKind::TextInput { .. } => "textinput",
            PropertyKind::UserShortcut { .. } => "usershortcut",
            PropertyKind::File { .. } => "file",
            PropertyKind::Directory { .. } => "directory",
            PropertyKind::SceneTexture { .. } => "scenetexture",
        }
    }
}

/// A recognized user property (docs/format-project-json.md §4.2/§4.3).
#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    /// Human label, default `""`; `ui_*` values are WE localization keys
    /// passed through untranslated (§4.2).
    pub text: String,
    /// Display order, default `0`, §8-coerced; sort ascending, tie-break on
    /// the property key (§4.2). Always `0` for [`PropertyKind::Text`] — the
    /// C++ never reads `order` there (§4.3 text), so any raw `order` stays in
    /// [`Property::extra`].
    pub order: i64,
    /// Type-specific payload (§4.3).
    pub kind: PropertyKind,
    /// Fields the C++ does not read — preserved verbatim. Corpus examples:
    /// `index`, `condition` (§4.2, §6), slider `fraction`/`precision`, file
    /// `filter` (§4.3).
    pub extra: Map<String, Value>,
}

/// One entry of the `general.properties` map
/// (docs/format-project-json.md §4.1 dispatch).
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyEntry {
    /// A property with a recognized interactive type.
    Property(Property),
    /// `type` absent or `"group"` — separator/header row, ignored silently by
    /// the renderer (§4.1, PropertyParser.cpp:39–45). Preserved verbatim.
    Group(Map<String, Value>),
    /// Unrecognized `type` string (including `""`), a non-string `type`, or a
    /// non-object property body — the C++ logs an error and ignores the
    /// property (§4.1, PropertyParser.cpp:39–43). Preserved verbatim.
    Unrecognized(Value),
}

/// The `general` object (docs/format-project-json.md §2.1).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct General {
    /// `general.supportsaudioprocessing`, §8-coerced bool, default `false`
    /// (§2.1, ProjectParser.cpp:43).
    pub supportsaudioprocessing: bool,
    /// `general.properties` user-property map, default empty (§2.1, §4). The
    /// map key is the property's identity (what scene `"user"` bindings and
    /// the override mechanism address).
    pub properties: BTreeMap<String, PropertyEntry>,
    /// Other `general.*` keys, unread by the C++ — preserved verbatim.
    /// Corpus: `supportsvideo`, `supportsvideoflags` (§2.2).
    pub extra: Map<String, Value>,
}

/// A parsed `project.json` manifest (docs/format-project-json.md §2).
#[derive(Clone, Debug, PartialEq)]
pub struct Project {
    /// Display title — required, must be a JSON string (§2.1).
    pub title: String,
    /// Main entry file: scene JSON path / video file / HTML file / URL —
    /// required (§2.1, §3.2).
    pub file: String,
    /// The raw declared `type` string, if the key is present (§2.1: parser
    /// default `""`≡absent, but the playlist preflight requires the key).
    /// A non-string value degrades to `Some(DeclaredType(""))` per §8; a JSON
    /// `null` counts as absent (§8) and is preserved in [`Project::extra`].
    pub declared_type: Option<DeclaredType>,
    /// Wallpaper type resolved from the main file's extension/URL scheme with
    /// the declared string as fallback (§3.1). Computed at parse time; an
    /// undeterminable type is a parse error, exactly like the C++.
    pub resolved_type: WallpaperType,
    /// `workshopid`, string or number (§2.1). `None` when absent (7/24 corpus
    /// items) or of an unusable JSON type (the C++ then substitutes an
    /// app-level synthetic negative counter — that policy belongs to the
    /// application, not this parser; the raw value stays in
    /// [`Project::extra`]).
    pub workshopid: Option<WorkshopId>,
    /// The `general` object; all defaults when absent (§2.1).
    pub general: General,
    /// Every other top-level key — ignored by the C++ renderer, preserved
    /// verbatim (§2.2: `preview`, `description`, `tags`, `contentrating`,
    /// `ratingsex`, `ratingviolence`, `visibility`, `version`, `workshopurl`,
    /// `approved`, `oversized`, `category`, …).
    pub extra: Map<String, Value>,
}

impl Project {
    /// Read and parse `project.json` from `path`, running full validation
    /// (required keys, type resolution, property parsing).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ProjectError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| ProjectError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // docs/format-project-json.md §1: strict JSON (nlohmann): no comments,
        // no trailing commas; duplicate keys last-wins (serde_json matches).
        let value: Value = serde_json::from_slice(&bytes)?;
        Self::from_value(value)
    }

    /// Parse a manifest from an already-decoded JSON value
    /// (docs/format-project-json.md §2.1).
    pub fn from_value(value: Value) -> Result<Self, ProjectError> {
        let Value::Object(mut map) = value else {
            return Err(ProjectError::NotAnObject);
        };

        // §2.1: `title` required, strict string (no coercion — ProjectParser.cpp:40).
        let title = match map.remove("title") {
            None => return Err(ProjectError::TitleMissing),
            Some(Value::String(s)) => s,
            Some(_) => return Err(ProjectError::TitleNotString),
        };

        // §2.1: `file` required; non-string "effectively must be a string" → typed error.
        let file = match map.remove("file") {
            None => return Err(ProjectError::FileMissing),
            Some(Value::String(s)) => s,
            Some(_) => return Err(ProjectError::FileNotString),
        };

        // §2.1: `type` optional, default ""; §8: null ≡ absent, a non-string
        // value degrades to "" instead of aborting.
        let declared_type = match map.remove("type") {
            None => None,
            Some(Value::String(s)) => Some(DeclaredType(s)),
            Some(Value::Null) => {
                map.insert("type".to_owned(), Value::Null);
                None
            }
            Some(other) => {
                map.insert("type".to_owned(), other);
                Some(DeclaredType(String::new()))
            }
        };

        // §2.1: `workshopid` string kept verbatim, number kept losslessly;
        // any other type → the C++ logs an error and uses a synthetic ID
        // (app-level policy) — here it stays in `extra` and yields None.
        let workshopid = match map.remove("workshopid") {
            None => None,
            Some(Value::String(s)) => Some(WorkshopId::Text(s)),
            Some(Value::Number(n)) => Some(WorkshopId::Number(n)),
            Some(other) => {
                map.insert("workshopid".to_owned(), other);
                None
            }
        };

        // §2.1: `general` optional; absent ⇒ no properties, no audio flag.
        let general = match map.remove("general") {
            Some(Value::Object(g)) => parse_general(g)?,
            Some(other) => {
                // Non-object `general` — nothing the C++ reads survives;
                // preserve the raw value.
                map.insert("general".to_owned(), other);
                General::default()
            }
            None => General::default(),
        };

        // §3.1: resolve the wallpaper type; failure is fatal like the C++.
        let resolved_type = resolve_type(&file, declared_type.as_ref())?;

        Ok(Project {
            title,
            file,
            declared_type,
            resolved_type,
            workshopid,
            general,
            extra: map,
        })
    }

    /// Serialize the model back to a JSON value. Inverse of
    /// [`Project::from_value`] at the model level:
    /// `from_value(p.to_value()) == p`.
    pub fn to_value(&self) -> Value {
        let mut map = self.extra.clone();
        map.insert("title".to_owned(), Value::String(self.title.clone()));
        map.insert("file".to_owned(), Value::String(self.file.clone()));
        // A raw non-string `type` was preserved verbatim in `extra` (with the
        // §8-degraded `DeclaredType("")` beside it), so the preserved value
        // must win here — overwriting it with `""` would lose the raw value
        // and break the round-trip contract.
        if let Some(t) = &self.declared_type
            && !map.contains_key("type")
        {
            map.insert("type".to_owned(), Value::String(t.0.clone()));
        }
        if let Some(w) = &self.workshopid {
            let v = match w {
                WorkshopId::Text(s) => Value::String(s.clone()),
                WorkshopId::Number(n) => Value::Number(n.clone()),
            };
            map.insert("workshopid".to_owned(), v);
        }
        if self.general != General::default() {
            map.insert("general".to_owned(), general_to_value(&self.general));
        }
        Value::Object(map)
    }

    /// The `category` key (unread by the renderer, docs/format-project-json.md
    /// §2.2/§3.3); `"Asset"` marks published non-wallpaper items.
    pub fn category(&self) -> Option<&str> {
        self.extra.get("category").and_then(Value::as_str)
    }

    /// docs/format-project-json.md §3.3: a `category == "Asset"` item (e.g. an
    /// effect package) is not directly runnable even though extension-based
    /// type resolution classifies it as a Scene.
    pub fn is_asset(&self) -> bool {
        self.category() == Some("Asset")
    }

    /// Playlist preflight check (docs/format-project-json.md §2.1,
    /// WallpaperApplication.cpp:353–367): WE rejects items whose manifest
    /// lacks a `type` or `file` key — presence only, values unchecked
    /// (`json.contains(...)`, so even a JSON `null` or non-string `type`
    /// passes). `file` presence is guaranteed by construction here, so only
    /// `type` matters: either it parsed into [`Project::declared_type`]
    /// (string value) or its raw non-string value was preserved in
    /// [`Project::extra`].
    pub fn passes_preflight(&self) -> bool {
        self.declared_type.is_some() || self.extra.contains_key("type")
    }
}

impl Serialize for Project {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Project {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Project::from_value(value).map_err(D::Error::custom)
    }
}

/// Resolve the wallpaper type from the main file with the declared string as
/// fallback (docs/format-project-json.md §3.1, ProjectParser.cpp:53–100).
pub fn resolve_type(file: &str, declared: Option<&DeclaredType>) -> Result<WallpaperType, ProjectError> {
    // §3.1: `file` is lowercased first (ProjectParser.cpp:56–57).
    let lower = file.to_ascii_lowercase();

    // §3.1 step 1: URL scheme prefixes → Web (ProjectParser.cpp:59–61).
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("www.") {
        return Ok(WallpaperType::Web);
    }

    // §3.1 step 2: substring after the last `.`; no dot ⇒ empty extension.
    let ext = lower.rsplit_once('.').map_or("", |(_, e)| e);
    match ext {
        "json" | "pkg" => return Ok(WallpaperType::Scene),
        "html" | "htm" => return Ok(WallpaperType::Web),
        "mp4" | "webm" | "mkv" | "avi" | "mov" | "m4v" | "wmv" => return Ok(WallpaperType::Video),
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "webp" => return Ok(WallpaperType::Image),
        "exe" => return Ok(WallpaperType::Application),
        _ => {}
    }

    // §3.1 step 3: fall back to the declared type string, case-insensitive.
    if let Some(t) = declared.and_then(DeclaredType::classify) {
        return Ok(t);
    }

    // §3.1 step 4: fatal.
    Err(ProjectError::TypeUndeterminable {
        file: file.to_owned(),
    })
}

/// Parse a property-color string per docs/format-project-json.md §5 with float
/// parsing forced and alpha 1.0 (Property.h:131–135); returns `[r, g, b]`.
///
/// Deliberate clean-implementation deviations the doc prescribes: `#rrggbb`
/// gets an `ff` alpha byte appended (the C++ mis-parses it), and the hex int
/// parse is overflow-safe (the C++ `std::stoi` throws for values ≥
/// `0x80000000`).
pub fn parse_property_color(s: &str) -> Result<[f32; 3], ColorError> {
    // §5 step 1: commas are normalized to spaces first (ColorBuilder.cpp:17–21).
    let normalized = s.replace(',', " ");

    // §5 step 2: `#`-prefixed CSS hex (ColorBuilder.cpp:24–50).
    if let Some(digits) = normalized.strip_prefix('#') {
        let expanded: String = match digits.len() {
            // `#rgb`: each digit doubled, alpha byte appended (alpha 1.0 → "ff").
            3 => {
                let mut e: String = digits.chars().flat_map(|c| [c, c]).collect();
                e.push_str("ff");
                e
            }
            // `#rgba`: each digit doubled.
            4 => digits.chars().flat_map(|c| [c, c]).collect(),
            // `#rrggbb`: append "ff" (clean fix; see §5 step 2).
            6 => format!("{digits}ff"),
            8 => digits.to_owned(),
            len => return Err(ColorError::HexLength { len }),
        };
        let v = u32::from_str_radix(&expanded, 16).map_err(|_| ColorError::HexDigits {
            digits: digits.to_owned(),
        })?;
        // §5: bytes split RR GG BB AA from high to low, each /255.
        let byte = |shift: u32| ((v >> shift) & 0xff) as f32 / 255.0;
        return Ok([byte(24), byte(16), byte(8)]);
    }

    // §5 step 3: space-separated 3- or 4-component vector. Component count is
    // naive space counting (VectorBuilder.h:34–53): a trailing space adds a
    // phantom empty component, which strtof parses as 0.
    let parts: Vec<&str> = normalized.split(' ').collect();
    match parts.as_slice() {
        // forceFloat=true for manifest properties: components are floats
        // used as-is, never the int-0–255 scene path (§5, Property.h:131–135).
        [r, g, b] | [r, g, b, _] => Ok([strtof(r), strtof(g), strtof(b)]),
        other => Err(ColorError::ComponentCount { count: other.len() }),
    }
}

/// C `strtof` semantics (docs/format-project-json.md §5: `strtof` per
/// component, VectorBuilder.h:128): skip leading whitespace, convert the
/// longest valid float prefix, and yield `0.0` when no conversion is
/// possible. Out-of-range values saturate to ±inf exactly like C `strtof`
/// (which returns `HUGE_VALF` without failing).
fn strtof(s: &str) -> f32 {
    float_prefix(skip_c_whitespace(s)).parse().unwrap_or(0.0)
}

/// Skip the leading whitespace the C `strto*` family skips: the C-locale
/// `isspace` set (space, `\t`, `\n`, `\v`, `\f`, `\r`) — not Unicode
/// whitespace.
fn skip_c_whitespace(s: &str) -> &str {
    s.trim_start_matches([' ', '\t', '\n', '\x0b', '\x0c', '\r'])
}

/// Longest prefix of `s` that the C `strtof`/`strtod` grammar converts:
/// optional sign, then digits with an optional `.` and fraction, an optional
/// exponent (kept only if it has at least one digit), or an
/// `inf`/`infinity`/`nan` token (ASCII case-insensitive). Empty when no
/// prefix converts. Single O(n) scan — a try-every-length loop is
/// quadratic on adversarial input (SPEC.md §V9: fuzzed parsers must not
/// hang). C hex-float syntax (`0x1p3`) is not recognized, matching the
/// previous behavior; `strtof("0x…")` stops at the `x` either way here.
fn float_prefix(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut i = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    for token in ["infinity", "inf", "nan"] {
        if s.len() >= i + token.len() && s[i..i + token.len()].eq_ignore_ascii_case(token) {
            return &s[..i + token.len()];
        }
    }
    // Byte length of the longest prefix that is a *complete* number so far.
    let mut valid = 0;
    let int_start = i;
    while bytes.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    let has_int_digits = i > int_start;
    if has_int_digits {
        valid = i;
    }
    if bytes.get(i) == Some(&b'.') {
        let frac_start = i + 1;
        let mut j = frac_start;
        while bytes.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        // "1." and ".5" convert; a lone "." does not.
        if has_int_digits || j > frac_start {
            valid = j;
            i = j;
        }
    }
    if valid > 0 && matches!(bytes.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(bytes.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        let exp_digits_start = j;
        while bytes.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        // "1e", "1e+" back off to "1" like strtod.
        if j > exp_digits_start {
            valid = j;
        }
    }
    &s[..valid]
}

/// Longest prefix of `s` that base-10 `strtoll` converts: optional sign, then
/// decimal digits. Empty when no prefix converts.
fn int_prefix(s: &str) -> &str {
    let bytes = s.as_bytes();
    let sign = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let mut i = sign;
    while bytes.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    if i == sign { "" } else { &s[..i] }
}

/// Whether a [`float_prefix`] result is an infinity token (after the optional
/// sign) — a *valid* `strtod` conversion, as opposed to a finite literal that
/// overflowed the double range.
fn is_infinity_token(prefix: &str) -> bool {
    let rest = prefix.trim_start_matches(['+', '-']);
    rest.eq_ignore_ascii_case("inf") || rest.eq_ignore_ascii_case("infinity")
}

/// §8 coercion, bool target: bool as-is; number ≠ 0; string ∈
/// {"1","true","True","TRUE"}. `None` = not coercible (caller applies the
/// default; `optional()` swallows failures in the C++).
fn coerce_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => Some(n.as_f64().unwrap_or(f64::NAN) != 0.0),
        Value::String(s) => Some(matches!(s.as_str(), "1" | "true" | "True" | "TRUE")),
        _ => None,
    }
}

/// §8 coercion, numeric target: number as-is; bool → 0/1; string → `stod`
/// prefix parse with failure → 0. `None` = not coercible.
///
/// "Failure" includes out-of-range finite literals: `std::stod` throws
/// `out_of_range` where `strtod` would saturate, and the `coerce<T>` catch
/// maps every such throw to `T{}` = 0 (docs/format-project-json.md §8,
/// Data/JSON.h:41–79). A literal `inf`/`infinity` token is a valid
/// conversion and stays infinite.
fn coerce_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => {
            let prefix = float_prefix(skip_c_whitespace(s));
            let parsed = prefix.parse::<f64>().unwrap_or(0.0);
            Some(if parsed.is_infinite() && !is_infinity_token(prefix) {
                0.0 // overflowed finite literal, e.g. "1e999" → stod throws → 0
            } else {
                parsed
            })
        }
        _ => None,
    }
}

/// §8 coercion, integer target: number truncated; bool → 0/1; string →
/// `stoll` prefix parse with failure → 0. `None` = not coercible.
///
/// "Failure" includes i64 overflow: `std::stoll` throws `out_of_range` and
/// the `coerce<T>` catch maps it to 0 (docs/format-project-json.md §8,
/// Data/JSON.h:41–79) — Rust's checked parse errors on the same inputs.
fn coerce_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => Some(n.as_i64().unwrap_or_else(|| n.as_f64().map_or(0, |f| f as i64))),
        Value::Bool(b) => Some(i64::from(*b)),
        Value::String(s) => Some(int_prefix(skip_c_whitespace(s)).parse().unwrap_or(0)),
        _ => None,
    }
}

/// §4.3 combo: a JSON number option/selected value becomes its decimal integer
/// string (the C++ does `std::to_string(get<int>())`, PropertyParser.cpp:66–67,
/// 85–89 — truncating; we truncate at i64 width instead of i32).
fn number_to_int_string(n: &Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        (n.as_f64().unwrap_or(0.0) as i64).to_string()
    }
}

/// Parse the `general` object (docs/format-project-json.md §2.1).
fn parse_general(mut map: Map<String, Value>) -> Result<General, ProjectError> {
    // §2.1: `supportsaudioprocessing` §8-coerced bool, default false
    // (ProjectParser.cpp:43). Non-coercible values stay preserved in `extra`.
    let supportsaudioprocessing = match map.get("supportsaudioprocessing").and_then(coerce_bool) {
        Some(b) => {
            map.remove("supportsaudioprocessing");
            b
        }
        None => false,
    };

    // §2.1/§4: `properties` object, default {}. A non-object value is
    // preserved raw; the C++ reads nothing from it.
    let mut properties = BTreeMap::new();
    match map.remove("properties") {
        Some(Value::Object(props)) => {
            for (name, raw) in props {
                let entry = parse_property_entry(raw).map_err(|source| ProjectError::Property {
                    name: name.clone(),
                    source,
                })?;
                properties.insert(name, entry);
            }
        }
        Some(other) => {
            map.insert("properties".to_owned(), other);
        }
        None => {}
    }

    Ok(General {
        supportsaudioprocessing,
        properties,
        extra: map,
    })
}

/// Parse one `general.properties` entry per the §4.1 dispatch table.
fn parse_property_entry(value: Value) -> Result<PropertyEntry, PropertyError> {
    let Value::Object(mut raw) = value else {
        // Non-object body: nothing to dispatch on; the C++ error-logs and
        // ignores it (§4.1 catch-all). Preserved verbatim.
        return Ok(PropertyEntry::Unrecognized(value));
    };

    let tag = match raw.get("type") {
        // §4.1: absent type → ignored silently (group separator).
        None => return Ok(PropertyEntry::Group(raw)),
        Some(Value::String(s)) => s.clone(),
        // §4.1: a non-string type never matches any dispatch string → error
        // logged, property ignored.
        Some(_) => return Ok(PropertyEntry::Unrecognized(Value::Object(raw))),
    };

    // §4.3 per-type payloads. Each arm consumes exactly the fields the C++
    // reads; everything else lands in `extra`.
    let kind = match tag.as_str() {
        // §4.1: "group" → ignored silently.
        "group" => return Ok(PropertyEntry::Group(raw)),
        "bool" => PropertyKind::Bool {
            // §4.3 bool: `value` optional, §8-coerced, default false.
            value: raw
                .remove("value")
                .as_ref()
                .and_then(coerce_bool)
                .unwrap_or(false),
        },
        "slider" => {
            // §4.3 slider: `value` required and NOT coerced — a JSON number
            // (or bool) works, a numeric string throws (PropertyParser.cpp:135).
            let value = match raw.remove("value") {
                None => return Err(PropertyError::MissingField { field: "value" }),
                Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) as f32,
                Some(Value::Bool(b)) => {
                    if b {
                        1.0
                    } else {
                        0.0
                    }
                }
                Some(_) => {
                    return Err(PropertyError::WrongType {
                        field: "value",
                        expected: "number or bool",
                    });
                }
            };
            // §4.3 slider: min/max/step optional, default 0.0, §8-coerced
            // (numeric strings accepted, unlike `value`).
            let mut take = |field: &str| -> f32 {
                raw.remove(field).as_ref().and_then(coerce_f64).unwrap_or(0.0) as f32
            };
            let (min, max, step) = (take("min"), take("max"), take("step"));
            PropertyKind::Slider {
                value,
                min,
                max,
                step,
            }
        }
        "color" => {
            // §4.3 color: `value` required, must be a JSON string (raw JSON
            // bound to std::string, no coercion), parsed per §5.
            let value = match raw.remove("value") {
                None => return Err(PropertyError::MissingField { field: "value" }),
                Some(Value::String(s)) => parse_property_color(&s)?,
                Some(_) => {
                    return Err(PropertyError::WrongType {
                        field: "value",
                        expected: "string",
                    });
                }
            };
            PropertyKind::Color { value }
        }
        "combo" => {
            // §4.3 combo: `options` required, fatal if absent or not an array.
            let raw_options = match raw.remove("options") {
                None => return Err(PropertyError::MissingField { field: "options" }),
                Some(Value::Array(a)) => a,
                Some(_) => return Err(PropertyError::OptionsNotArray),
            };
            let mut options = Vec::new();
            for entry in raw_options {
                // §4.3: non-object entries are skipped (PropertyParser.cpp:62–64).
                let Value::Object(mut opt) = entry else { continue };
                let label = match opt.remove("label") {
                    None => {
                        return Err(PropertyError::MissingField {
                            field: "options[].label",
                        });
                    }
                    Some(Value::String(s)) => s,
                    Some(_) => {
                        return Err(PropertyError::WrongType {
                            field: "options[].label",
                            expected: "string",
                        });
                    }
                };
                let value = match opt.remove("value") {
                    None => {
                        return Err(PropertyError::MissingField {
                            field: "options[].value",
                        });
                    }
                    Some(Value::String(s)) => s,
                    Some(Value::Number(n)) => number_to_int_string(&n),
                    Some(_) => {
                        return Err(PropertyError::WrongType {
                            field: "options[].value",
                            expected: "string or number",
                        });
                    }
                };
                options.push(ComboOption {
                    label,
                    value,
                    extra: opt,
                });
            }
            // §4.3 combo: `value` optional (string or number); missing/other →
            // first option's value, or "" when the accepted-but-empty options
            // array has no entries.
            let value = match raw.remove("value") {
                Some(Value::String(s)) => s,
                Some(Value::Number(n)) => number_to_int_string(&n),
                _ => options.first().map(|o| o.value.clone()).unwrap_or_default(),
            };
            PropertyKind::Combo { options, value }
        }
        "text" => PropertyKind::Text,
        "scenetexture" => {
            // §4.3 scenetexture: `value` required string.
            let value = match raw.remove("value") {
                None => return Err(PropertyError::MissingField { field: "value" }),
                Some(Value::String(s)) => s,
                Some(_) => {
                    return Err(PropertyError::WrongType {
                        field: "value",
                        expected: "string",
                    });
                }
            };
            PropertyKind::SceneTexture { value }
        }
        // §4.3 file/directory: `value` optional string, default "" (optional()
        // swallows a non-string per §8).
        "file" => PropertyKind::File {
            value: take_optional_string(&mut raw),
        },
        "directory" => PropertyKind::Directory {
            value: take_optional_string(&mut raw),
        },
        "textinput" => PropertyKind::TextInput {
            value: take_textinput_value(&mut raw)?,
        },
        // §4.1: "usershortcut" is parsed as textinput (PropertyParser.cpp:35).
        "usershortcut" => PropertyKind::UserShortcut {
            value: take_textinput_value(&mut raw)?,
        },
        // §4.1: anything else (including "") → error logged, property ignored.
        _ => return Ok(PropertyEntry::Unrecognized(Value::Object(raw))),
    };

    raw.remove("type");

    // §4.2 common fields: `text` default "" (optional() swallows a non-string
    // per §8), `order` default 0 §8-coerced.
    let text = match raw.remove("text") {
        Some(Value::String(s)) => s,
        _ => String::new(),
    };
    // §4.3 text: `order` is never read for static labels — it stays 0 and any
    // raw value remains preserved in `extra`.
    let order = if matches!(kind, PropertyKind::Text) {
        0
    } else {
        raw.remove("order").as_ref().and_then(coerce_i64).unwrap_or(0)
    };

    Ok(PropertyEntry::Property(Property {
        text,
        order,
        kind,
        extra: raw,
    }))
}

/// §4.3 file/directory: `value` optional string, default `""`.
fn take_optional_string(raw: &mut Map<String, Value>) -> String {
    match raw.remove("value") {
        Some(Value::String(s)) => s,
        _ => String::new(),
    }
}

/// §4.3 textinput/usershortcut: `value` required, any JSON. A JSON string
/// stores its contents (the doc's clean-implementation reading); any other
/// JSON stores its compact serialization (the C++ `value.dump()`).
fn take_textinput_value(raw: &mut Map<String, Value>) -> Result<String, PropertyError> {
    match raw.remove("value") {
        None => Err(PropertyError::MissingField { field: "value" }),
        Some(Value::String(s)) => Ok(s),
        Some(other) => Ok(other.to_string()),
    }
}

/// Serialize a [`General`] back to a JSON object (inverse of `parse_general`).
fn general_to_value(general: &General) -> Value {
    let mut map = general.extra.clone();
    if general.supportsaudioprocessing {
        map.insert("supportsaudioprocessing".to_owned(), Value::Bool(true));
    }
    if !general.properties.is_empty() {
        let props: Map<String, Value> = general
            .properties
            .iter()
            .map(|(name, entry)| (name.clone(), property_entry_to_value(entry)))
            .collect();
        map.insert("properties".to_owned(), Value::Object(props));
    }
    Value::Object(map)
}

/// JSON `Number` from an `f32`, widened exactly. JSON cannot represent
/// non-finite numbers, but the model can hold them (a huge JSON number like
/// `1e300` casts to ±∞ exactly like the C++ `static_cast<float>`): ±∞ is
/// written as the largest-magnitude finite double, which the parse-side
/// f64→f32 cast maps back to the same infinity, keeping the round-trip
/// contract. NaN (reachable only via a `"nan"` string through §8 coercion)
/// collapses to 0 — NaN is outside the round-trip contract anyway, since it
/// is not even equal to itself.
fn f32_number(v: f32) -> Value {
    let widened = if v == f32::INFINITY {
        f64::MAX
    } else if v == f32::NEG_INFINITY {
        f64::MIN
    } else {
        f64::from(v) // NaN falls through: from_f64 rejects it → 0 below
    };
    Value::Number(Number::from_f64(widened).unwrap_or_else(|| Number::from(0)))
}

/// Serialize one property entry back to a JSON value (inverse of
/// `parse_property_entry`).
fn property_entry_to_value(entry: &PropertyEntry) -> Value {
    let property = match entry {
        PropertyEntry::Group(raw) => return Value::Object(raw.clone()),
        PropertyEntry::Unrecognized(v) => return v.clone(),
        PropertyEntry::Property(p) => p,
    };
    let mut map = property.extra.clone();
    map.insert(
        "type".to_owned(),
        Value::String(property.kind.type_tag().to_owned()),
    );
    map.insert("text".to_owned(), Value::String(property.text.clone()));
    if !matches!(property.kind, PropertyKind::Text) {
        map.insert("order".to_owned(), Value::Number(property.order.into()));
    }
    match &property.kind {
        PropertyKind::Bool { value } => {
            map.insert("value".to_owned(), Value::Bool(*value));
        }
        PropertyKind::Slider {
            value,
            min,
            max,
            step,
        } => {
            map.insert("value".to_owned(), f32_number(*value));
            map.insert("min".to_owned(), f32_number(*min));
            map.insert("max".to_owned(), f32_number(*max));
            map.insert("step".to_owned(), f32_number(*step));
        }
        PropertyKind::Color { value: [r, g, b] } => {
            // §5 canonical encoding: space-separated floats in 0..1. Rust's
            // shortest round-trip float display re-parses to the same f32.
            map.insert("value".to_owned(), Value::String(format!("{r} {g} {b}")));
        }
        PropertyKind::Combo { options, value } => {
            let opts: Vec<Value> = options
                .iter()
                .map(|o| {
                    let mut m = o.extra.clone();
                    m.insert("label".to_owned(), Value::String(o.label.clone()));
                    m.insert("value".to_owned(), Value::String(o.value.clone()));
                    Value::Object(m)
                })
                .collect();
            map.insert("options".to_owned(), Value::Array(opts));
            map.insert("value".to_owned(), Value::String(value.clone()));
        }
        PropertyKind::Text => {}
        PropertyKind::TextInput { value }
        | PropertyKind::UserShortcut { value }
        | PropertyKind::File { value }
        | PropertyKind::Directory { value }
        | PropertyKind::SceneTexture { value } => {
            map.insert("value".to_owned(), Value::String(value.clone()));
        }
    }
    Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default corpus location (SPEC.md §C); override with `KIRIE_CORPUS`.
    const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";

    fn corpus_dir() -> Option<PathBuf> {
        let dir = std::env::var("KIRIE_CORPUS").map_or_else(|_| PathBuf::from(CORPUS_DIR), PathBuf::from);
        if dir.is_dir() {
            Some(dir)
        } else {
            eprintln!("skipping corpus test: {} not present", dir.display());
            None
        }
    }

    /// All corpus manifests as (workshop id, raw bytes), sorted by id.
    /// The 24 items documented in docs/format-project-json.md §10 / corpus.md.
    /// The census tests below assert exact counts over this snapshot; the corpus
    /// is a live Steam dir, so newly subscribed items are excluded here (they are
    /// still parse-checked by kirie-formats/tests/corpus.rs gate1).
    const DOC_ITEM_IDS: &[&str] = &[
        "1388331347",
        "1627026721",
        "2082653325",
        "2085292947",
        "2155933185",
        "2395163768",
        "2968833989",
        "3047596375",
        "3118949804",
        "3293156956",
        "3347128360",
        "3421423611",
        "3428443753",
        "3445942378",
        "3551997868",
        "3576956643",
        "3585875739",
        "3587565260",
        "3600453929",
        "3609007632",
        "3611478368",
        "3631634316",
        "3679122549",
        "3738467344",
    ];

    fn corpus_manifests() -> Option<Vec<(String, Vec<u8>)>> {
        let dir = corpus_dir()?;
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("read corpus dir") {
            let entry = entry.expect("corpus dir entry");
            let manifest = entry.path().join("project.json");
            let id = entry.file_name().to_string_lossy().into_owned();
            if manifest.is_file() && DOC_ITEM_IDS.contains(&id.as_str()) {
                out.push((id, std::fs::read(&manifest).expect("read manifest")));
            }
        }
        out.sort();
        Some(out)
    }

    fn parse(json: &str) -> Project {
        Project::from_value(serde_json::from_str(json).expect("valid JSON")).expect("valid project")
    }

    fn parse_err(json: &str) -> ProjectError {
        Project::from_value(serde_json::from_str(json).expect("valid JSON"))
            .expect_err("expected a parse error")
    }

    fn property<'p>(p: &'p Project, name: &str) -> &'p Property {
        match p.general.properties.get(name) {
            Some(PropertyEntry::Property(prop)) => prop,
            other => panic!("property {name:?} is {other:?}"),
        }
    }

    // ---- §9 minimal valid examples --------------------------------------

    /// docs/format-project-json.md §9: absolute floor accepted by the parser.
    #[test]
    fn absolute_floor_manifest() {
        let p = parse(r#"{"title": "x", "file": "scene.json"}"#);
        assert_eq!(p.title, "x");
        assert_eq!(p.file, "scene.json");
        assert_eq!(p.resolved_type, WallpaperType::Scene);
        assert_eq!(p.declared_type, None);
        assert!(!p.passes_preflight()); // §2.1: playlist preflight demands a `type` key
        assert_eq!(p.workshopid, None);
        assert_eq!(p.general, General::default());
    }

    /// docs/format-project-json.md §9 scene example (trimmed item 2085292947).
    #[test]
    fn minimal_scene_manifest() {
        let p = parse(
            r#"{
            "contentrating": "Mature",
            "file": "scene.json",
            "general": {
                "properties": {
                    "schemecolor": {
                        "order": 0,
                        "text": "ui_browse_properties_scheme_color",
                        "type": "color",
                        "value": "0 0 0"
                    },
                    "style": {
                        "options": [
                            { "label": "X-Ray", "value": "1" },
                            { "label": "CG 1", "value": "2" }
                        ],
                        "order": 100,
                        "text": "Style",
                        "type": "combo"
                    },
                    "x_ray_radius": {
                        "fraction": true, "precision": 2,
                        "min": 0.1, "max": 1.2, "step": 0.1,
                        "order": 101,
                        "text": "X-Ray radius",
                        "type": "slider",
                        "value": 0.6
                    }
                }
            },
            "preview": "preview.jpg",
            "tags": [ "Anime" ],
            "title": "[R18] 松永 時雨 01 [X-Ray]",
            "type": "scene"
        }"#,
        );
        assert_eq!(p.resolved_type, WallpaperType::Scene);
        assert_eq!(p.declared_type, Some(DeclaredType("scene".to_owned())));
        assert!(p.passes_preflight());
        assert_eq!(p.title, "[R18] 松永 時雨 01 [X-Ray]");
        assert_eq!(p.workshopid, None);
        assert_eq!(
            p.extra.get("contentrating"),
            Some(&Value::String("Mature".to_owned()))
        );
        assert_eq!(p.general.properties.len(), 3);

        let scheme = property(&p, "schemecolor");
        assert_eq!(
            scheme.kind,
            PropertyKind::Color {
                value: [0.0, 0.0, 0.0]
            }
        );
        assert_eq!(scheme.text, "ui_browse_properties_scheme_color");
        assert_eq!(scheme.order, 0);

        // §9 note: the `style` combo ships without `value` → defaults to the
        // first option's value "1" (§4.3 combo).
        let style = property(&p, "style");
        let PropertyKind::Combo { options, value } = &style.kind else {
            panic!("style is {:?}", style.kind);
        };
        assert_eq!(value, "1");
        assert_eq!(options.len(), 2);
        assert_eq!(options[0].label, "X-Ray");
        assert_eq!(options[1].value, "2");

        let radius = property(&p, "x_ray_radius");
        assert_eq!(
            radius.kind,
            PropertyKind::Slider {
                value: 0.6,
                min: 0.1,
                max: 1.2,
                step: 0.1
            }
        );
        assert_eq!(radius.order, 101);
        // Unread slider fields preserved (§4.3): fraction, precision.
        assert_eq!(radius.extra.get("fraction"), Some(&Value::Bool(true)));
        assert_eq!(radius.extra.get("precision"), Some(&Value::Number(2.into())));
    }

    /// docs/format-project-json.md §9 video example (item 3600453929, complete).
    #[test]
    fn minimal_video_manifest() {
        let p = parse(
            r#"{
            "contentrating": "Everyone",
            "file": "冷冰冰的誓言.mp4",
            "general": {
                "properties": {
                    "schemecolor": {
                        "order": 0,
                        "text": "ui_browse_properties_scheme_color",
                        "type": "color",
                        "value": "0.00000 0.00000 0.00000"
                    }
                }
            },
            "preview": "preview.jpg",
            "ratingsex": "none",
            "ratingviolence": "none",
            "tags": [ "Anime" ],
            "title": "冷冰冰的誓言",
            "type": "video",
            "version": 0
        }"#,
        );
        assert_eq!(p.resolved_type, WallpaperType::Video);
        assert_eq!(p.file, "冷冰冰的誓言.mp4"); // §3.2: non-ASCII UTF-8 file names
        assert_eq!(p.title, "冷冰冰的誓言");
        assert_eq!(p.workshopid, None); // §2.1: absent in this item
        assert_eq!(p.extra.get("version"), Some(&Value::Number(0.into())));
        assert_eq!(
            property(&p, "schemecolor").kind,
            PropertyKind::Color {
                value: [0.0, 0.0, 0.0]
            }
        );
    }

    /// docs/format-project-json.md §9 web example (trimmed item 3679122549).
    #[test]
    fn minimal_web_manifest() {
        let p = parse(
            r#"{
            "contentrating": "Everyone",
            "file": "index.html",
            "general": {
                "properties": {
                    "schemecolor": {
                        "order": 0,
                        "text": "ui_browse_properties_scheme_color",
                        "type": "color",
                        "value": "0.7529411764705882 0.7529411764705882 0.7529411764705882"
                    },
                    "custom_user_bg": {
                        "filter": "*.jpg|*.png|*.jpeg|*.webp",
                        "index": 0, "order": 100,
                        "text": "Custom BG Image",
                        "type": "file"
                    },
                    "audio_sensitivity": {
                        "index": 5, "order": 105,
                        "min": 0.5, "max": 1.5, "step": 0.1,
                        "text": "Audio Sensitivity",
                        "type": "slider",
                        "value": 1
                    }
                }
            },
            "preview": "preview.jpg",
            "title": "[16:9] 超かぐや姫 All-in-One",
            "type": "web",
            "version": 2,
            "visibility": "public",
            "workshopid": "3679122549",
            "workshopurl": "steam://url/CommunityFilePage/3679122549"
        }"#,
        );
        assert_eq!(p.resolved_type, WallpaperType::Web);
        assert_eq!(p.workshopid, Some(WorkshopId::Text("3679122549".to_owned())));

        let grey: f32 = "0.7529411764705882".parse().expect("float");
        assert_eq!(
            property(&p, "schemecolor").kind,
            PropertyKind::Color { value: [grey; 3] }
        );

        // §4.3 file: `value` absent → "" default; `filter` preserved unread.
        let bg = property(&p, "custom_user_bg");
        assert_eq!(bg.kind, PropertyKind::File { value: String::new() });
        assert_eq!(
            bg.extra.get("filter"),
            Some(&Value::String("*.jpg|*.png|*.jpeg|*.webp".to_owned()))
        );

        // §4.3 slider: integer JSON `value` binds to float.
        assert_eq!(
            property(&p, "audio_sensitivity").kind,
            PropertyKind::Slider {
                value: 1.0,
                min: 0.5,
                max: 1.5,
                step: 0.1
            }
        );
    }

    /// docs/format-project-json.md §9/§3.3: workshop *asset* manifest — no
    /// `type`, extension-resolved as Scene, flagged by `category`.
    #[test]
    fn asset_manifest_is_scene_by_extension_but_flagged() {
        let p = parse(
            r#"{
            "category": "Asset",
            "contentrating": "Everyone",
            "file": "effects/gradient_generator/effect.json",
            "preview": "preview.jpg",
            "tags": [ "Background" ],
            "title": "Gradient generator",
            "visibility": "public",
            "workshopid": "3347128360"
        }"#,
        );
        assert_eq!(p.resolved_type, WallpaperType::Scene); // §3.3: .json → Scene, which is wrong for assets
        assert_eq!(p.declared_type, None);
        assert!(!p.passes_preflight());
        assert!(p.is_asset());
        assert_eq!(p.category(), Some("Asset"));
    }

    // ---- type discrimination (§3.1) --------------------------------------

    #[test]
    fn type_resolution_follows_spec_order() {
        // Step 1: URL schemes → Web, checked on the lowercased file.
        for f in [
            "http://example.com",
            "HTTPS://EXAMPLE.COM/x",
            "www.example.com",
            "WWW.X.COM",
        ] {
            assert_eq!(resolve_type(f, None).unwrap(), WallpaperType::Web, "{f}");
        }
        // Step 2: extension table, case-insensitive.
        assert_eq!(resolve_type("scene.json", None).unwrap(), WallpaperType::Scene);
        assert_eq!(resolve_type("SCENE.PKG", None).unwrap(), WallpaperType::Scene);
        assert_eq!(resolve_type("index.html", None).unwrap(), WallpaperType::Web);
        assert_eq!(resolve_type("a.htm", None).unwrap(), WallpaperType::Web);
        for f in ["a.mp4", "a.webm", "a.mkv", "a.avi", "a.mov", "a.m4v", "a.wmv"] {
            assert_eq!(resolve_type(f, None).unwrap(), WallpaperType::Video, "{f}");
        }
        for f in ["a.png", "a.jpg", "a.jpeg", "a.gif", "a.bmp", "a.webp"] {
            assert_eq!(resolve_type(f, None).unwrap(), WallpaperType::Image, "{f}");
        }
        assert_eq!(resolve_type("a.exe", None).unwrap(), WallpaperType::Application);
        // Extension wins over a contradicting declared type (§3.1: declared is
        // fallback only).
        let scene = DeclaredType("scene".to_owned());
        assert_eq!(resolve_type("a.mp4", Some(&scene)).unwrap(), WallpaperType::Video);
        // Step 3: unknown extension → declared type, case-insensitive
        // (corpus has "Scene"/"Web" — §3.1).
        for (t, want) in [
            ("Scene", WallpaperType::Scene),
            ("VIDEO", WallpaperType::Video),
            ("Web", WallpaperType::Web),
            ("application", WallpaperType::Application),
            ("Image", WallpaperType::Image),
        ] {
            let d = DeclaredType(t.to_owned());
            assert_eq!(resolve_type("main.dat", Some(&d)).unwrap(), want, "{t}");
        }
        // No dot ⇒ empty extension ⇒ declared fallback.
        let video = DeclaredType("video".to_owned());
        assert_eq!(resolve_type("movie", Some(&video)).unwrap(), WallpaperType::Video);
        // Step 4: nothing matches → fatal.
        let weird = DeclaredType("WeIrD".to_owned());
        assert!(matches!(
            resolve_type("main.dat", Some(&weird)),
            Err(ProjectError::TypeUndeterminable { .. })
        ));
        assert!(matches!(
            resolve_type("main.dat", None),
            Err(ProjectError::TypeUndeterminable { .. })
        ));
    }

    /// Unknown declared type strings are preserved verbatim (§2.1/§3.1) when
    /// the extension already determines the type.
    #[test]
    fn unknown_declared_type_is_preserved() {
        let p = parse(r#"{"title": "x", "file": "scene.json", "type": "WeIrD"}"#);
        assert_eq!(p.resolved_type, WallpaperType::Scene);
        assert_eq!(p.declared_type, Some(DeclaredType("WeIrD".to_owned())));
        assert_eq!(p.declared_type.as_ref().unwrap().classify(), None);
        assert!(p.passes_preflight());
        // ... and it round-trips.
        let p2 = Project::from_value(p.to_value()).unwrap();
        assert_eq!(p, p2);
        assert_eq!(p2.to_value()["type"], Value::String("WeIrD".to_owned()));
    }

    /// §8: a non-string `type` degrades to `""` for parsing but the raw value
    /// is preserved — and must survive re-serialization (`to_value` must not
    /// clobber it with the degraded `""`).
    #[test]
    fn nonstring_type_preserved_and_round_trips() {
        let p = parse(r#"{"title": "x", "file": "scene.json", "type": 5}"#);
        assert_eq!(p.declared_type, Some(DeclaredType(String::new())));
        assert_eq!(p.extra.get("type"), Some(&Value::Number(5.into())));
        // §2.1 preflight is key-presence only (`json.contains`,
        // WallpaperApplication.cpp:357–360): a non-string `type` passes.
        assert!(p.passes_preflight());
        let value = p.to_value();
        assert_eq!(value["type"], Value::Number(5.into()));
        let p2 = Project::from_value(value).unwrap();
        assert_eq!(p, p2);
    }

    /// §2.1/§8: a JSON `null` `type` is absent for the *parser* (default "")
    /// but present for the *preflight* (`json.contains("type")` is true,
    /// WallpaperApplication.cpp:357–360), and round-trips verbatim.
    #[test]
    fn null_type_passes_preflight_and_round_trips() {
        let p = parse(r#"{"title": "x", "file": "scene.json", "type": null}"#);
        assert_eq!(p.declared_type, None);
        assert_eq!(p.extra.get("type"), Some(&Value::Null));
        assert!(p.passes_preflight());
        let value = p.to_value();
        assert_eq!(value["type"], Value::Null);
        let p2 = Project::from_value(value).unwrap();
        assert_eq!(p, p2);
    }

    // ---- property-type matrix ---------------------------------------------

    #[test]
    fn property_type_matrix() {
        let p = parse(
            r#"{
            "title": "matrix", "file": "scene.json", "type": "scene",
            "general": { "properties": {
                "b1":  { "type": "bool", "value": true, "order": 1, "text": "b" },
                "b2":  { "type": "bool", "value": "1" },
                "b3":  { "type": "bool", "value": 0 },
                "b4":  { "type": "bool" },
                "s1":  { "type": "slider", "value": 2, "min": "2.5", "max": 10, "step": true },
                "s2":  { "type": "slider", "value": true },
                "c1":  { "type": "color", "value": "0.012 0.192 0.251" },
                "co1": { "type": "combo", "options": [
                            { "label": "A", "value": 5 },
                            "skipped-non-object",
                            { "label": "B", "value": "b", "note": "kept" }
                         ] },
                "co2": { "type": "combo", "options": [], "value": "z" },
                "t1":  { "type": "text", "text": "label", "order": 7, "value": "ignored" },
                "ti1": { "type": "textinput", "value": "hello" },
                "ti2": { "type": "textinput", "value": 12 },
                "us1": { "type": "usershortcut", "value": "ctrl+k" },
                "f1":  { "type": "file", "filter": "*.png" },
                "f2":  { "type": "file", "value": "pic.png" },
                "d1":  { "type": "directory" },
                "st1": { "type": "scenetexture", "value": "materials/tex.tex" },
                "g1":  { "type": "group", "text": "Header", "order": 3 },
                "g2":  { "text": "no type at all" },
                "u1":  { "type": "", "text": "_______ Section" },
                "u2":  { "type": 42 },
                "u3":  "not an object"
            } }
        }"#,
        );
        let props = &p.general.properties;
        assert_eq!(props.len(), 22);

        // §4.3 bool + §8 coercion.
        assert_eq!(property(&p, "b1").kind, PropertyKind::Bool { value: true });
        assert_eq!(property(&p, "b1").order, 1);
        assert_eq!(property(&p, "b1").text, "b");
        assert_eq!(property(&p, "b2").kind, PropertyKind::Bool { value: true }); // "1" → true
        assert_eq!(property(&p, "b3").kind, PropertyKind::Bool { value: false }); // 0 → false
        assert_eq!(property(&p, "b4").kind, PropertyKind::Bool { value: false }); // default

        // §4.3 slider: min coerced from numeric string, step from bool; value
        // from number/bool.
        assert_eq!(
            property(&p, "s1").kind,
            PropertyKind::Slider {
                value: 2.0,
                min: 2.5,
                max: 10.0,
                step: 1.0
            }
        );
        assert_eq!(
            property(&p, "s2").kind,
            PropertyKind::Slider {
                value: 1.0,
                min: 0.0,
                max: 0.0,
                step: 0.0
            }
        );

        // §5 color.
        assert_eq!(
            property(&p, "c1").kind,
            PropertyKind::Color {
                value: [0.012, 0.192, 0.251]
            }
        );

        // §4.3 combo: numeric option value → "5"; non-object option skipped;
        // missing `value` → first option; option extras preserved.
        let PropertyKind::Combo { options, value } = &property(&p, "co1").kind else {
            panic!("co1 not a combo");
        };
        assert_eq!(options.len(), 2);
        assert_eq!(options[0].value, "5");
        assert_eq!(value, "5");
        assert_eq!(
            options[1].extra.get("note"),
            Some(&Value::String("kept".to_owned()))
        );
        // Empty options accepted; explicit value kept (§4.3).
        assert_eq!(
            property(&p, "co2").kind,
            PropertyKind::Combo {
                options: vec![],
                value: "z".to_owned()
            }
        );

        // §4.3 text: only `text` read; `order`/`value` ignored (order stays 0)
        // but preserved in extra.
        let t1 = property(&p, "t1");
        assert_eq!(t1.kind, PropertyKind::Text);
        assert_eq!(t1.text, "label");
        assert_eq!(t1.order, 0);
        assert_eq!(t1.extra.get("order"), Some(&Value::Number(7.into())));
        assert_eq!(t1.extra.get("value"), Some(&Value::String("ignored".to_owned())));

        // §4.3 textinput: string contents kept; non-string stored as its dump.
        assert_eq!(
            property(&p, "ti1").kind,
            PropertyKind::TextInput {
                value: "hello".to_owned()
            }
        );
        assert_eq!(
            property(&p, "ti2").kind,
            PropertyKind::TextInput {
                value: "12".to_owned()
            }
        );
        // §4.1 usershortcut → textinput shape, own tag.
        assert_eq!(
            property(&p, "us1").kind,
            PropertyKind::UserShortcut {
                value: "ctrl+k".to_owned()
            }
        );

        // §4.3 file/directory: value default "".
        assert_eq!(
            property(&p, "f1").kind,
            PropertyKind::File { value: String::new() }
        );
        assert_eq!(
            property(&p, "f2").kind,
            PropertyKind::File {
                value: "pic.png".to_owned()
            }
        );
        assert_eq!(
            property(&p, "d1").kind,
            PropertyKind::Directory { value: String::new() }
        );

        // §4.3 scenetexture.
        assert_eq!(
            property(&p, "st1").kind,
            PropertyKind::SceneTexture {
                value: "materials/tex.tex".to_owned()
            }
        );

        // §4.1 ignore paths.
        assert!(matches!(props.get("g1"), Some(PropertyEntry::Group(_))));
        assert!(matches!(props.get("g2"), Some(PropertyEntry::Group(_))));
        assert!(matches!(props.get("u1"), Some(PropertyEntry::Unrecognized(_))));
        assert!(matches!(props.get("u2"), Some(PropertyEntry::Unrecognized(_))));
        assert!(matches!(
            props.get("u3"),
            Some(PropertyEntry::Unrecognized(Value::String(_)))
        ));

        // The whole matrix round-trips.
        let p2 = Project::from_value(p.to_value()).unwrap();
        assert_eq!(p, p2);
    }

    // ---- malformed manifests ----------------------------------------------

    #[test]
    fn malformed_top_level() {
        assert!(matches!(
            Project::from_value(serde_json::json!([])),
            Err(ProjectError::NotAnObject)
        ));
        // §2.1: title required, fatal (C++ "Project title missing").
        assert!(matches!(
            parse_err(r#"{"file": "scene.json"}"#),
            ProjectError::TitleMissing
        ));
        // Verifier note: non-string title is an error (§2.1: string targets
        // are not coerced).
        assert!(matches!(
            parse_err(r#"{"title": 42, "file": "scene.json"}"#),
            ProjectError::TitleNotString
        ));
        assert!(matches!(
            parse_err(r#"{"title": true, "file": "scene.json"}"#),
            ProjectError::TitleNotString
        ));
        // §2.1: file required, fatal (C++ "Project's main file missing").
        assert!(matches!(
            parse_err(r#"{"title": "x"}"#),
            ProjectError::FileMissing
        ));
        assert!(matches!(
            parse_err(r#"{"title": "x", "file": 5}"#),
            ProjectError::FileNotString
        ));
        // §3.1 step 4: undeterminable type is fatal.
        assert!(matches!(
            parse_err(r#"{"title": "x", "file": "main.dat"}"#),
            ProjectError::TypeUndeterminable { .. }
        ));
        // §1: strict JSON — trailing commas rejected.
        assert!(matches!(
            Project::from_path("/nonexistent/project.json"),
            Err(ProjectError::Io { .. })
        ));
        assert!(serde_json::from_str::<Value>(r#"{"title": "x",}"#).is_err());
    }

    #[test]
    fn malformed_properties() {
        fn prop_err(body: &str) -> PropertyError {
            let json =
                format!(r#"{{"title":"x","file":"scene.json","general":{{"properties":{{"p":{body}}}}}}}"#);
            match parse_err(&json) {
                ProjectError::Property { name, source } => {
                    assert_eq!(name, "p");
                    source
                }
                other => panic!("expected property error, got {other:?}"),
            }
        }

        // §4.3 color: value required, must be a JSON string.
        assert_eq!(
            prop_err(r#"{"type":"color"}"#),
            PropertyError::MissingField { field: "value" }
        );
        assert_eq!(
            prop_err(r#"{"type":"color","value":5}"#),
            PropertyError::WrongType {
                field: "value",
                expected: "string"
            }
        );
        // §5: bad color strings.
        assert_eq!(
            prop_err(r#"{"type":"color","value":"1 1"}"#),
            PropertyError::Color(ColorError::ComponentCount { count: 2 })
        );
        assert_eq!(
            prop_err(r#"{"type":"color","value":"1 1 1 1 1"}"#),
            PropertyError::Color(ColorError::ComponentCount { count: 5 })
        );
        assert_eq!(
            prop_err(r##"{"type":"color","value":"#12345"}"##),
            PropertyError::Color(ColorError::HexLength { len: 5 })
        );
        assert_eq!(
            prop_err(r##"{"type":"color","value":"#zzz"}"##),
            PropertyError::Color(ColorError::HexDigits {
                digits: "zzz".to_owned()
            })
        );

        // §4.3 slider: value required; numeric strings are NOT accepted there.
        assert_eq!(
            prop_err(r#"{"type":"slider"}"#),
            PropertyError::MissingField { field: "value" }
        );
        assert_eq!(
            prop_err(r#"{"type":"slider","value":"3"}"#),
            PropertyError::WrongType {
                field: "value",
                expected: "number or bool"
            }
        );

        // §4.3 combo: options required and must be an array; option label and
        // value required.
        assert_eq!(
            prop_err(r#"{"type":"combo"}"#),
            PropertyError::MissingField { field: "options" }
        );
        assert_eq!(
            prop_err(r#"{"type":"combo","options":"x"}"#),
            PropertyError::OptionsNotArray
        );
        assert_eq!(
            prop_err(r#"{"type":"combo","options":[{"value":"1"}]}"#),
            PropertyError::MissingField {
                field: "options[].label"
            }
        );
        assert_eq!(
            prop_err(r#"{"type":"combo","options":[{"label":"A"}]}"#),
            PropertyError::MissingField {
                field: "options[].value"
            }
        );
        assert_eq!(
            prop_err(r#"{"type":"combo","options":[{"label":"A","value":true}]}"#),
            PropertyError::WrongType {
                field: "options[].value",
                expected: "string or number"
            }
        );

        // §4.3 textinput / scenetexture: value required.
        assert_eq!(
            prop_err(r#"{"type":"textinput"}"#),
            PropertyError::MissingField { field: "value" }
        );
        assert_eq!(
            prop_err(r#"{"type":"scenetexture"}"#),
            PropertyError::MissingField { field: "value" }
        );
        assert_eq!(
            prop_err(r#"{"type":"scenetexture","value":9}"#),
            PropertyError::WrongType {
                field: "value",
                expected: "string"
            }
        );
    }

    // ---- color parser (§5) ------------------------------------------------

    #[test]
    fn color_parsing() {
        // Canonical corpus forms: floats in 0..1.
        assert_eq!(
            parse_property_color("0.012 0.192 0.251").unwrap(),
            [0.012, 0.192, 0.251]
        );
        // forceFloat: "1 1 1" is white, never 1/255-gray (§5).
        assert_eq!(parse_property_color("1 1 1").unwrap(), [1.0, 1.0, 1.0]);
        assert_eq!(parse_property_color("0 0 0").unwrap(), [0.0, 0.0, 0.0]);
        // Commas normalized to spaces (§5 step 1).
        assert_eq!(parse_property_color("1,0,0.5").unwrap(), [1.0, 0.0, 0.5]);
        // 4-component vector accepted; alpha dropped for the RGB triple.
        assert_eq!(parse_property_color("0.1 0.2 0.3 0.4").unwrap(), [0.1, 0.2, 0.3]);
        // Trailing space: naive component counting sees a phantom 4th
        // component, strtof turns it into 0 (§5 step 3 note).
        assert_eq!(parse_property_color("1 1 1 ").unwrap(), [1.0, 1.0, 1.0]);
        // strtof: no conversion possible → 0.0, not an error (§5).
        assert_eq!(parse_property_color("a b c").unwrap(), [0.0, 0.0, 0.0]);
        // strtof prefix parsing.
        assert_eq!(parse_property_color("1.5x 2 3").unwrap(), [1.5, 2.0, 3.0]);

        // Hex forms (§5 step 2). #rgb doubled + implicit alpha.
        assert_eq!(parse_property_color("#fff").unwrap(), [1.0, 1.0, 1.0]);
        // #rgba doubled.
        assert_eq!(parse_property_color("#f00f").unwrap(), [1.0, 0.0, 0.0]);
        // #rrggbb: clean fix appends the alpha byte instead of the C++ shift bug.
        assert_eq!(parse_property_color("#ff0000").unwrap(), [1.0, 0.0, 0.0]);
        let g: f32 = 0x80 as f32 / 255.0;
        assert_eq!(parse_property_color("#008000").unwrap(), [0.0, g, 0.0]);
        // #rrggbbaa, including values ≥ 0x80000000 (the C++ stoi would throw).
        assert_eq!(parse_property_color("#ffffffff").unwrap(), [1.0, 1.0, 1.0]);

        // Errors.
        assert_eq!(
            parse_property_color("1 1"),
            Err(ColorError::ComponentCount { count: 2 })
        );
        assert_eq!(
            parse_property_color(""),
            Err(ColorError::ComponentCount { count: 1 })
        );
        assert_eq!(
            parse_property_color("#12345"),
            Err(ColorError::HexLength { len: 5 })
        );
        assert_eq!(
            parse_property_color("#gg0000"),
            Err(ColorError::HexDigits {
                digits: "gg0000".to_owned()
            })
        );
    }

    // ---- §8 string→number coercion (stoll/stod semantics) ------------------

    /// §8: string coercion is `std::stoll`/`std::stod` inside the `coerce<T>`
    /// try/catch (Data/JSON.h:41–79) — longest-prefix conversion, and *any*
    /// throw (no conversion or out-of-range) yields 0.
    #[test]
    fn string_coercion_follows_stoll_stod_semantics() {
        let slider = |min: &str| -> f32 {
            let json = format!(
                r#"{{"title":"x","file":"scene.json","general":{{"properties":{{
                    "s":{{"type":"slider","value":1,"min":{min}}}}}}}}}"#
            );
            let p = parse(&json);
            match property(&p, "s").kind {
                PropertyKind::Slider { min, .. } => min,
                ref other => panic!("not a slider: {other:?}"),
            }
        };
        // Longest-prefix conversion, junk tail ignored.
        assert_eq!(slider(r#""2.5x""#), 2.5);
        assert_eq!(slider(r#"" \t-3.5e1junk""#), -35.0);
        // Incomplete exponents back off to the mantissa like strtod.
        assert_eq!(slider(r#""2e""#), 2.0);
        assert_eq!(slider(r#""2e+""#), 2.0);
        assert_eq!(slider(r#"".5""#), 0.5);
        assert_eq!(slider(r#""1.""#), 1.0);
        // No conversion possible → 0.
        assert_eq!(slider(r#""x1""#), 0.0);
        assert_eq!(slider(r#""""#), 0.0);
        // std::stod throws out_of_range for finite literals beyond the double
        // range; coerce<T> catches → 0 (JSON.h:41–79).
        assert_eq!(slider(r#""1e999""#), 0.0);
        assert_eq!(slider(r#""-1e999""#), 0.0);
        // A literal infinity is a *valid* strtod conversion and stays inf.
        assert_eq!(slider(r#""inf""#), f32::INFINITY);
        assert_eq!(slider(r#""-Infinity""#), f32::NEG_INFINITY);

        let order = |order: &str| -> i64 {
            let json = format!(
                r#"{{"title":"x","file":"scene.json","general":{{"properties":{{
                    "b":{{"type":"bool","value":true,"order":{order}}}}}}}}}"#
            );
            property(&parse(&json), "b").order
        };
        assert_eq!(order(r#""42abc""#), 42);
        assert_eq!(order(r#""+7""#), 7);
        assert_eq!(order(r#""-7.9""#), -7); // stoll stops at '.'
        assert_eq!(order(r#""abc""#), 0);
        // std::stoll throws out_of_range past i64 → coerce catches → 0,
        // never a shorter numeric prefix.
        assert_eq!(order(r#""99999999999999999999""#), 0);
        assert_eq!(order(r#""-99999999999999999999""#), 0);
    }

    /// A slider value beyond the f32 range casts to ±∞ like the C++
    /// `static_cast<float>` and must still round-trip (JSON cannot encode
    /// inf; it re-serializes as the largest finite double).
    #[test]
    fn slider_value_beyond_f32_range_round_trips() {
        let p = parse(
            r#"{"title":"x","file":"scene.json","general":{"properties":{
                "s":{"type":"slider","value":1e300,"min":-1e300}}}}"#,
        );
        assert_eq!(
            property(&p, "s").kind,
            PropertyKind::Slider {
                value: f32::INFINITY,
                min: f32::NEG_INFINITY,
                max: 0.0,
                step: 0.0
            }
        );
        let p2 = Project::from_value(p.to_value()).unwrap();
        assert_eq!(p, p2);
    }

    /// SPEC.md §V9: prefix parsing is a single O(n) scan — a pathological
    /// component must not hang the parser (the previous try-every-length
    /// loop was quadratic).
    #[test]
    fn pathological_numeric_strings_parse_in_linear_time() {
        // Long digit run with a junk tail: worst case for backtracking.
        let long = format!("1{}x{}", "9".repeat(100_000), "9".repeat(100_000));
        let started = std::time::Instant::now();
        assert_eq!(parse_property_color(&format!("{long} 0 0")).unwrap()[1], 0.0);
        assert_eq!(coerce_i64(&Value::String(long)), Some(0)); // overflow → 0
        assert_eq!(coerce_f64(&Value::String("e".repeat(100_000))), Some(0.0));
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "prefix parse took {:?}",
            started.elapsed()
        );
    }

    // ---- workshopid (§2.1) -------------------------------------------------

    #[test]
    fn workshopid_string_or_number() {
        let p = parse(r#"{"title":"x","file":"scene.json","workshopid":"3679122549"}"#);
        assert_eq!(p.workshopid, Some(WorkshopId::Text("3679122549".to_owned())));
        assert_eq!(p.workshopid.as_ref().unwrap().to_string(), "3679122549");

        let p = parse(r#"{"title":"x","file":"scene.json","workshopid":3679122549}"#);
        assert!(matches!(p.workshopid, Some(WorkshopId::Number(_))));
        assert_eq!(p.workshopid.as_ref().unwrap().to_string(), "3679122549");
        // Round-trips as a number, not a string.
        assert!(p.to_value()["workshopid"].is_number());

        // §2.1: other JSON types → error logged + synthetic app-level ID in
        // the C++; here: None, raw value preserved.
        let p = parse(r#"{"title":"x","file":"scene.json","workshopid":true}"#);
        assert_eq!(p.workshopid, None);
        assert_eq!(p.extra.get("workshopid"), Some(&Value::Bool(true)));
    }

    // ---- general (§2.1) -----------------------------------------------------

    #[test]
    fn general_flags_and_coercion() {
        let p = parse(
            r#"{"title":"x","file":"scene.json",
                "general":{"supportsaudioprocessing":true,"supportsvideo":true,"supportsvideoflags":1}}"#,
        );
        assert!(p.general.supportsaudioprocessing);
        // §2.2: unread general.* keys preserved.
        assert_eq!(p.general.extra.get("supportsvideo"), Some(&Value::Bool(true)));
        assert_eq!(
            p.general.extra.get("supportsvideoflags"),
            Some(&Value::Number(1.into()))
        );

        // §8: "1" coerces to true; 0 to false.
        let p = parse(r#"{"title":"x","file":"scene.json","general":{"supportsaudioprocessing":"1"}}"#);
        assert!(p.general.supportsaudioprocessing);
        let p = parse(r#"{"title":"x","file":"scene.json","general":{"supportsaudioprocessing":0}}"#);
        assert!(!p.general.supportsaudioprocessing);
    }

    // ---- serde round-trip ----------------------------------------------------

    #[test]
    fn serde_round_trip_preserves_model_and_unknown_keys() {
        let src = r#"{
            "title": "rt", "file": "scene.json", "type": "Scene",
            "workshopid": "123",
            "preview": "preview.gif",
            "description": "hello [b]world[/b]",
            "tags": ["Anime"],
            "contentrating": "Everyone",
            "version": 4,
            "approved": true,
            "custom_unknown_key": {"nested": [1, 2, 3]},
            "general": {
                "supportsaudioprocessing": true,
                "supportsvideo": true,
                "properties": {
                    "schemecolor": {"type": "color", "value": "1 0 0", "order": 0,
                                    "text": "ui_browse_properties_scheme_color"},
                    "speed": {"type": "slider", "value": 1.5, "min": 0.1, "max": 3,
                              "step": 0.1, "order": 100, "text": "Speed",
                              "condition": "toggle.value", "index": 0, "precision": 1},
                    "toggle": {"type": "bool", "value": true, "order": 101, "text": "T"},
                    "sep": {"type": "group", "text": "Header", "order": 99},
                    "banner": {"text": "<img src=x>"},
                    "weird": {"type": "", "text": "_______"}
                }
            }
        }"#;
        // Through the Deserialize impl (serde entry point).
        let p1: Project = serde_json::from_str(src).expect("deserialize");
        // Unknown top-level keys preserved (§2.2, no deny_unknown_fields).
        assert!(p1.extra.contains_key("custom_unknown_key"));
        assert_eq!(
            p1.extra.get("preview"),
            Some(&Value::String("preview.gif".to_owned()))
        );
        // Unread property fields preserved (§4.2 condition/index, §4.3 precision).
        let speed = property(&p1, "speed");
        assert_eq!(
            speed.extra.get("condition"),
            Some(&Value::String("toggle.value".to_owned()))
        );

        // Through the Serialize impl and back: model equality.
        let text = serde_json::to_string(&p1).expect("serialize");
        let p2: Project = serde_json::from_str(&text).expect("re-deserialize");
        assert_eq!(p1, p2);

        // And via the Value-level helpers.
        let p3 = Project::from_value(p1.to_value()).expect("from_value");
        assert_eq!(p1, p3);

        // Group/unrecognized rows survive verbatim.
        assert!(matches!(
            p2.general.properties.get("sep"),
            Some(PropertyEntry::Group(_))
        ));
        assert!(matches!(
            p2.general.properties.get("banner"),
            Some(PropertyEntry::Group(_))
        ));
        assert!(matches!(
            p2.general.properties.get("weird"),
            Some(PropertyEntry::Unrecognized(_))
        ));
        // Declared-type case preserved.
        assert_eq!(p2.declared_type, Some(DeclaredType("Scene".to_owned())));
    }

    // ---- corpus ---------------------------------------------------------------

    /// SPEC.md V11 / task: all 24 corpus manifests parse; assert the exact
    /// type split the real files give: declared `type` (lowercased) scene 19 /
    /// web 3 / video 1 / absent 1 (the Asset item), resolved (§3.1,
    /// extension-based) Scene 20 / Web 3 / Video 1
    /// (docs/format-project-json.md §10 census).
    #[test]
    fn corpus_all_manifests_parse_with_expected_type_split() {
        let Some(manifests) = corpus_manifests() else {
            return;
        };
        assert_eq!(
            manifests.len(),
            24,
            "corpus should have 24 project.json manifests"
        );

        let mut declared: BTreeMap<String, usize> = BTreeMap::new();
        let mut resolved: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut assets: Vec<String> = Vec::new();
        let mut with_workshopid = 0usize;

        for (id, bytes) in &manifests {
            let value: Value = serde_json::from_slice(bytes).unwrap_or_else(|e| panic!("{id}: {e}"));
            let p = Project::from_value(value).unwrap_or_else(|e| panic!("{id}: {e}"));
            assert!(!p.title.is_empty(), "{id}: empty title");
            assert!(!p.file.is_empty(), "{id}: empty file");

            let key = match &p.declared_type {
                Some(t) => t.0.to_ascii_lowercase(),
                None => "<absent>".to_owned(),
            };
            *declared.entry(key).or_default() += 1;

            let res = match p.resolved_type {
                WallpaperType::Scene => "scene",
                WallpaperType::Web => "web",
                WallpaperType::Video => "video",
                WallpaperType::Image => "image",
                WallpaperType::Application => "application",
            };
            *resolved.entry(res).or_default() += 1;

            if p.is_asset() {
                assets.push(id.clone());
                assert_eq!(p.declared_type, None, "{id}: asset has no type key (§3.3)");
                assert_eq!(
                    p.resolved_type,
                    WallpaperType::Scene,
                    "{id}: §3.3 misclassification"
                );
            }
            if let Some(w) = &p.workshopid {
                with_workshopid += 1;
                // §2.1: corpus workshopids are always decimal strings.
                assert!(matches!(w, WorkshopId::Text(_)), "{id}: non-string workshopid");
                assert_eq!(w.to_string(), *id, "{id}: workshopid mismatch");
            }
        }

        // Declared split (docs §10 census; the 5 mixed-case spellings fold in).
        let expected: BTreeMap<String, usize> = [
            ("scene".to_owned(), 19),
            ("web".to_owned(), 3),
            ("video".to_owned(), 1),
            ("<absent>".to_owned(), 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(declared, expected);

        // Resolved split (§3.1: the Asset item's effect.json resolves to Scene).
        let expected_resolved: BTreeMap<&'static str, usize> =
            [("scene", 20), ("web", 3), ("video", 1)].into_iter().collect();
        assert_eq!(resolved, expected_resolved);

        // Exactly the one known Asset item (§3.3).
        assert_eq!(assets, vec!["3347128360".to_owned()]);
        // §2.1: workshopid present in 17/24, absent in 7.
        assert_eq!(with_workshopid, 17);
    }

    /// Every declared property in the corpus parses to the variant its raw
    /// `type` tag dictates, with the exact §4.1 census counts (211 total).
    #[test]
    fn corpus_every_declared_property_parses_to_matching_variant() {
        let Some(manifests) = corpus_manifests() else {
            return;
        };

        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut defaulted_combos = 0usize;

        for (id, bytes) in &manifests {
            let raw: Value = serde_json::from_slice(bytes).unwrap_or_else(|e| panic!("{id}: {e}"));
            let p = Project::from_value(raw.clone()).unwrap_or_else(|e| panic!("{id}: {e}"));

            let raw_props = raw
                .get("general")
                .and_then(|g| g.get("properties"))
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            assert_eq!(
                p.general.properties.len(),
                raw_props.len(),
                "{id}: property count"
            );

            for (name, entry) in &p.general.properties {
                let raw_prop = raw_props
                    .get(name)
                    .unwrap_or_else(|| panic!("{id}:{name} missing"));
                let raw_tag = raw_prop.get("type").and_then(Value::as_str);

                // §4.1 dispatch: the parsed variant must match the raw tag.
                let variant: &'static str = match entry {
                    PropertyEntry::Property(prop) => {
                        assert_eq!(
                            Some(prop.kind.type_tag()),
                            raw_tag,
                            "{id}:{name}: variant/tag mismatch"
                        );
                        prop.kind.type_tag()
                    }
                    PropertyEntry::Group(_) => {
                        assert!(
                            raw_tag.is_none() || raw_tag == Some("group"),
                            "{id}:{name}: Group from tag {raw_tag:?}"
                        );
                        "group"
                    }
                    PropertyEntry::Unrecognized(_) => {
                        assert!(
                            raw_tag.is_some_and(|t| !matches!(
                                t,
                                "color"
                                    | "bool"
                                    | "slider"
                                    | "combo"
                                    | "text"
                                    | "scenetexture"
                                    | "file"
                                    | "directory"
                                    | "textinput"
                                    | "usershortcut"
                                    | "group"
                            )),
                            "{id}:{name}: Unrecognized from tag {raw_tag:?}"
                        );
                        "unrecognized"
                    }
                };
                *counts.entry(variant).or_default() += 1;

                // §4.3 combo: a missing `value` defaults to the first option.
                if let PropertyEntry::Property(Property {
                    kind: PropertyKind::Combo { options, value },
                    ..
                }) = entry
                    && raw_prop.get("value").is_none()
                {
                    defaulted_combos += 1;
                    let first = options.first().map(|o| o.value.as_str()).unwrap_or("");
                    assert_eq!(value, first, "{id}:{name}: combo default");
                }
            }
        }

        // Exact census (docs §4.1: 211 properties; the 12 "group" + 4
        // absent-type rows are Group, the 7 `type: ""` rows Unrecognized).
        let expected: BTreeMap<&'static str, usize> = [
            ("bool", 49),
            ("color", 37),
            ("combo", 11),
            ("file", 2),
            ("group", 16),
            ("slider", 66),
            ("text", 11),
            ("textinput", 12),
            ("unrecognized", 7),
        ]
        .into_iter()
        .collect();
        assert_eq!(counts, expected);
        assert_eq!(counts.values().sum::<usize>(), 211);
        // §4.3/§10: exactly the two known value-less combos (2082653325 x_ray,
        // 2085292947 style).
        assert_eq!(defaulted_combos, 2);
    }

    /// Every corpus manifest survives a full model round-trip through both the
    /// Value helpers and the serde impls (load-from-path exercised too).
    #[test]
    fn corpus_round_trip() {
        let Some(dir) = corpus_dir() else { return };
        let mut seen = 0usize;
        for entry in std::fs::read_dir(&dir).expect("read corpus dir") {
            let manifest = entry.expect("dir entry").path().join("project.json");
            if !manifest.is_file() {
                continue;
            }
            seen += 1;
            let p1 = Project::from_path(&manifest).unwrap_or_else(|e| panic!("{}: {e}", manifest.display()));
            let p2 =
                Project::from_value(p1.to_value()).unwrap_or_else(|e| panic!("{}: {e}", manifest.display()));
            assert_eq!(p1, p2, "{}: value round-trip", manifest.display());

            let text = serde_json::to_string(&p1).expect("serialize");
            let p3: Project =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", manifest.display()));
            assert_eq!(p1, p3, "{}: serde round-trip", manifest.display());
        }
        // Floor, not exact: every installed manifest (incl. newly subscribed
        // ones) round-trips; live corpus may hold more than the documented 24.
        assert!(seen >= 24, "corpus manifest count {seen} below floor 24");
    }
}
