# kirie (切り絵)

A fast, memory-safe wallpaper renderer for Linux, compatible with
[Wallpaper Engine](https://www.wallpaperengine.io/) content. Written in Rust on
wgpu/Vulkan, fully multithreaded, with a hash-keyed prebaked scene-bundle cache.

Drop-in compatible with the CLI and control socket of
[linux-wallpaperengine](https://github.com/Almamu/linux-wallpaperengine) — kirie
is a from-scratch Rust descendant, validated scene-by-scene against it.

Renders scene, video, image, and web wallpapers, with audio-reactive visualizers,
SceneScript (JS) scripting, 3D puppet/model layers, and the full effect pipeline.

## Build

Rust (stable) plus the system libraries the workspace links against.

```sh
# Arch
sudo pacman -S --needed rust ffmpeg alsa-lib libpulse shaderc glslang \
    wayland libxkbcommon libx11 mpv freetype2 dbus

# Debian/Ubuntu
sudo apt install -y build-essential clang cmake pkg-config \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev \
    libasound2-dev libpulse-dev libshaderc-dev glslang-dev \
    libwayland-dev libxkbcommon-dev libx11-dev libfreetype-dev libdbus-1-dev

# Default build (no web backend): lean and always-green.
cargo build --release -p kirie
```

The binary is `target/release/kirie`.

### Web wallpapers (optional)

Web (`"type": "web"`) wallpapers need an embedded browser, behind a cargo
feature. The default build enables neither, so it needs no browser libraries.

| Feature | Backend | System deps |
|---------|---------|-------------|
| `web-cef` | Chromium Embedded Framework (off-screen) | cmake, a C++ toolchain, clang; libcef downloaded on first build |
| `web-webview` | wry + system webkit2gtk-4.1 | `libwebkit2gtk-4.1-dev`, `libsoup-3.0-dev` |

```sh
cargo build --release -p kirie --features web-cef      # bundles CEF, composites via wgpu
cargo build --release -p kirie --features web-webview  # needs webkit2gtk-4.1
```

## Usage

kirie mirrors the `linux-wallpaperengine` CLI and control socket, so it drops
into the same launchers:

```sh
kirie --screen-root HDMI-A-1 --bg /path/to/workshop/item --scaling fill
kirie info  <item|scene.pkg|.tex>      # inspect
kirie extract <scene.pkg|.tex> -o DIR  # unpack
```

`scripts/install.sh` installs it as `linux-wallpaperengine` for daemons that look
for that binary name.

## Credits

- [Almamu/linux-wallpaperengine](https://github.com/Almamu/linux-wallpaperengine) —
  the C++ reference implementation this project derives from and is validated against.
- Wallpaper Engine is a product of Wallpaper Engine Team; this project is an
  independent, unaffiliated renderer for its content formats.

## License

AGPL-3.0-or-later. See `LICENSE`.

Copyleft: you may use kirie (including commercially) and modify it, but if you
distribute it **or run a modified version as a network service**, you must make
your modified source available under the same license.
