#!/usr/bin/env bash
#
# install.sh — build kirie and install it under the name `linux-wallpaperengine`
# so the Hypr wallpaper-daemon (and anything else expecting the upstream
# linux-wallpaperengine CLI) drives kirie as a drop-in replacement.
#
# kirie dispatches on argv[0]: invoked as `linux-wallpaperengine` it runs the
# compat flag surface (crates/kirie/src/lib.rs → compat::run), and the daemon's
# process bookkeeping (`pgrep -f`/`pkill -f -- "--screen-root <mon> "`) matches
# on the flags, not the binary name — so a symlink or rename is all it takes.
#
# Usage:
#   scripts/install.sh [--copy] [--dest PATH] [-- <extra cargo build args>]
#
#   --copy        Install a real copy instead of a symlink (default: symlink,
#                 so a later `cargo build --release` is picked up automatically).
#   --dest PATH   Install location (a directory or a full file path). Default:
#                 $KIRIE_DEST, else ~/.local/bin/linux-wallpaperengine.
#   --            Everything after is forwarded to `cargo build`, e.g.
#                 `-- --features web-cef` for the CEF web backend.
#
# The daemon (~/.config/hypr/wallpaper-daemon/wallpaperengine.sh) resolves the
# binary in this order:
#   1. $HOME/linux-wallpaperengine/build/output/linux-wallpaperengine
#   2. the first `linux-wallpaperengine` on $PATH
# The default dest (~/.local/bin) uses lookup #2. That deliberately does NOT
# touch an existing upstream C++ build at #1 (which may be the live wallpaper on
# screen). To make kirie the binary the daemon launches, either keep #1 absent
# or point --dest at it explicitly once you have retired the C++ build.

set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

mode="symlink"
dest="${KIRIE_DEST:-$HOME/.local/bin/linux-wallpaperengine}"
cargo_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --copy)    mode="copy"; shift ;;
        --dest)    dest="$2"; shift 2 ;;
        --dest=*)  dest="${1#--dest=}"; shift ;;
        --)        shift; cargo_args=("$@"); break ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)         echo "install.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

# If --dest is a directory (or ends in /), install the file inside it.
if [ -d "$dest" ] || [ "${dest%/}/" = "$dest" ]; then
    dest="${dest%/}/linux-wallpaperengine"
fi

echo "==> building kirie (release)"
( cd "$here" && cargo build --release -p kirie "${cargo_args[@]}" )

target_dir="${CARGO_TARGET_DIR:-$here/target}"
bin="$target_dir/release/kirie"
[ -x "$bin" ] || { echo "install.sh: built binary not found at $bin" >&2; exit 1; }

mkdir -p "$(dirname -- "$dest")"
rm -f -- "$dest"

if [ "$mode" = "copy" ]; then
    cp -f -- "$bin" "$dest"
    echo "==> installed copy: $dest"
else
    ln -s -- "$bin" "$dest"
    echo "==> symlinked: $dest -> $bin"
fi

# Sanity check: the installed name reports the compat banner (argv[0] routing).
if out="$("$dest" --help 2>/dev/null | head -n1)" && [ -n "$out" ]; then
    echo "==> ok: '$(basename -- "$dest")' responds ($out)"
fi

case ":$PATH:" in
    *":$(dirname -- "$dest"):"*) ;;
    *) echo "note: $(dirname -- "$dest") is not on \$PATH — add it, or use --dest to install where the daemon looks." ;;
esac

# Reminder for the CEF web backend: the helper + CEF runtime must sit beside the
# installed binary (see README "Web wallpapers").
for a in "${cargo_args[@]}"; do
    if [ "$a" = "web-cef" ] || [ "$a" = "--features=web-cef" ]; then
        echo "note: web-cef build — also install kirie-cef-helper and the CEF runtime next to $dest (see README)."
    fi
done
