//! Corpus-gated empirical test (SPEC.md §V11, task exit criterion): translate a
//! large sample of REAL workshop shaders and, for each success, create an actual
//! `wgpu::ShaderModule` on the live GPU. Reports pass/fail counts and asserts the
//! achievable fraction. Skipped (not failed) when the corpus or a GPU is absent.
//!
//! Corpus: `~/.steam/steam/steamapps/workshop/content/431960` (SPEC.md §C).
//! Includes resolve from the stock assets `shaders/` dir (docs/shader-pipeline.md
//! §1.1 mount order); workshop-local `.h` headers (if any) take precedence.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kirie_formats::pkg::OwnedPkg;
use kirie_shader::{FsIncludeResolver, MapIncludeResolver, ShaderInputs, Stage, TranslatePath, translate};

const CORPUS_DIR: &str = "/home/aiko/.steam/steam/steamapps/workshop/content/431960";
const STOCK_SHADERS: &str = "/home/aiko/.steam/steam/steamapps/common/wallpaper_engine/assets/shaders";

/// The fraction of corpus shaders that must translate to a GPU-loadable module.
/// Empirically ~0.93 is achievable; the remainder are the patched-glslang
/// leniencies (docs/shader-pipeline.md §7) and dynamic array varyings that stock
/// glslang / wgpu cannot express (see the printed failure list). We gate a hair
/// below the measured value to stay robust to driver/toolchain drift.
const REQUIRED_PASS_FRACTION: f64 = 0.90;

fn corpus_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("KIRIE_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(CORPUS_DIR));
    dir.is_dir().then_some(dir)
}

struct GpuHarness {
    device: wgpu::Device,
}

impl GpuHarness {
    fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        // Capture validation errors instead of aborting the run.
        device.on_uncaptured_error(std::sync::Arc::new(|e| eprintln!("wgpu uncaptured: {e}")));
        Some(Self { device })
    }

    /// Create a shader module from a naga IR module; returns whether the device
    /// accepted it (no validation error raised).
    fn accepts(&self, module: naga::Module) -> bool {
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _sm = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
        });
        pollster::block_on(guard.pop()).is_none()
    }
}

/// Build an include resolver for a pkg: pkg-local `.h` headers (keyed by their
/// name below `shaders/`) shadow the stock-assets fallback.
fn resolver_for(pkg: &OwnedPkg) -> MapIncludeResolver {
    let mut headers = BTreeMap::new();
    for entry in pkg.entries() {
        let Some(name) = entry.name_str() else { continue };
        if let Some(rel) = name.strip_prefix("shaders/")
            && rel.ends_with(".h")
            && let Ok(bytes) = pkg.read(&entry)
            && let Ok(text) = std::str::from_utf8(bytes)
        {
            headers.insert(rel.to_string(), text.to_string());
        }
    }
    MapIncludeResolver {
        headers,
        fallback: Some(FsIncludeResolver::new(vec![PathBuf::from(STOCK_SHADERS)])),
    }
}

fn scene_pkgs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.path().join("scene.pkg"))
        .filter(|p| p.is_file())
        .collect();
    v.sort();
    v
}

#[test]
fn corpus_translates_and_loads_on_gpu() {
    let Some(dir) = corpus_dir() else {
        eprintln!("skipping: corpus {CORPUS_DIR} absent (set KIRIE_CORPUS)");
        return;
    };
    if !Path::new(STOCK_SHADERS).is_dir() {
        eprintln!("skipping: stock assets {STOCK_SHADERS} absent");
        return;
    }
    let Some(gpu) = GpuHarness::new() else {
        eprintln!("skipping: no Vulkan GPU");
        return;
    };

    let mut total = 0usize;
    let mut translated = 0usize;
    let mut gpu_ok = 0usize;
    let mut via_naga = 0usize;
    let mut via_shaderc = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for pkg_path in scene_pkgs(&dir) {
        let Ok(pkg) = OwnedPkg::from_path(&pkg_path) else {
            continue;
        };
        let resolver = resolver_for(&pkg);
        let item = pkg_path
            .parent()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        // Collect shader entries first (borrow of pkg for reads is fine).
        let shaders: Vec<(String, Stage, Vec<u8>)> = pkg
            .entries()
            .filter_map(|e| {
                let name = e.name_str()?;
                let stage = if name.ends_with(".frag") {
                    Stage::Fragment
                } else if name.ends_with(".vert") {
                    Stage::Vertex
                } else {
                    return None;
                };
                let bytes = pkg.read(&e).ok()?.to_vec();
                Some((name.to_string(), stage, bytes))
            })
            .collect();

        for (name, stage, bytes) in shaders {
            let Ok(src) = String::from_utf8(bytes) else {
                continue;
            };
            total += 1;
            let label = format!("{item}/{name}");
            match translate(stage, &name, &src, &resolver, &ShaderInputs::default()) {
                Ok(ts) => {
                    translated += 1;
                    match ts.path {
                        TranslatePath::NagaGlsl => via_naga += 1,
                        TranslatePath::Shaderc => via_shaderc += 1,
                    }
                    if gpu.accepts(ts.module) {
                        gpu_ok += 1;
                    } else {
                        failures.push(format!("GPU-REJECT {label}"));
                    }
                }
                Err(e) => {
                    // One-line diagnostic.
                    let msg = e.to_string();
                    let line = msg.lines().find(|l| l.contains("error")).unwrap_or(&msg);
                    failures.push(format!("XLATE {label}: {}", line.trim()));
                }
            }
        }
    }

    eprintln!("\n=== kirie-shader corpus translation report ===");
    eprintln!("total shaders     : {total}");
    eprintln!(
        "translated        : {translated} ({:.1}%)",
        100.0 * translated as f64 / total as f64
    );
    eprintln!(
        "created on GPU     : {gpu_ok} ({:.1}%)",
        100.0 * gpu_ok as f64 / total as f64
    );
    eprintln!("  via naga glsl-in : {via_naga}");
    eprintln!("  via shaderc→spv  : {via_shaderc}");
    eprintln!("failures ({}):", failures.len());
    for f in &failures {
        eprintln!("  {f}");
    }

    assert!(total >= 150, "expected a large corpus, got {total}");
    let frac = gpu_ok as f64 / total as f64;
    assert!(
        frac >= REQUIRED_PASS_FRACTION,
        "GPU-loadable fraction {frac:.3} below required {REQUIRED_PASS_FRACTION:.3}"
    );
}
