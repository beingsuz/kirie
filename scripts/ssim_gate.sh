#!/usr/bin/env bash
#
# P4 corpus render GATE + SSIM quality sweep.
#
# For EVERY scene-type workshop item (a directory holding scene.pkg) under the
# corpus root this runs two phases:
#
#   PHASE 1 — RENDER GATE (fast, authoritative pass/fail)
#     Render a screenshot with the release `kirie` binary (offscreen wgpu) and
#     assert exit 0, the PNG exists, and it is non-black (>NONBLACK_MIN% lit
#     pixels). This ALL-SCENES-RENDER check is the P4 exit gate.
#
#   PHASE 2 — SSIM QUALITY (slow, never gates)
#     Render the C++ oracle screenshot in a FRESH process (never touching the
#     running daemon) and compute SSIM(kirie, oracle) as a tracked metric.
#     Items < SSIM_FLAG are flagged for the review phase.
#
# Env overrides:
#   KIRIE_BIN         path to the kirie binary (default: target/release/kirie)
#   CPP_BIN           path to the C++ oracle    (default: linux-wallpaperengine)
#   KIRIE_CORPUS      corpus root (default: the Steam workshop 431960 dir)
#   SCREENSHOT_DELAY  frame delay passed to both engines (default: 3)
#   OUT_DIR           where to write PNGs (default: a /tmp scratch dir)
#   NONBLACK_MIN      minimum non-black percentage to pass (default: 5)
#   SSIM_FLAG         SSIM below this is flagged for review (default: 0.5)
#   SKIP_SSIM         set to 1 to run only the render gate (no oracle)
#
# Exit status: 0 iff every scene item rendered non-black without crashing.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIRIE_BIN="${KIRIE_BIN:-$REPO/target/release/kirie}"
CPP_BIN="${CPP_BIN:-/home/aiko/linux-wallpaperengine/build/output/linux-wallpaperengine}"
CORPUS="${KIRIE_CORPUS:-/home/aiko/.steam/steam/steamapps/workshop/content/431960}"
DELAY="${SCREENSHOT_DELAY:-3}"
OUT_DIR="${OUT_DIR:-/tmp/claude-1000/kirie-p4-gate}"
NONBLACK_MIN="${NONBLACK_MIN:-5}"
SSIM_FLAG="${SSIM_FLAG:-0.5}"
SKIP_SSIM="${SKIP_SSIM:-0}"
METRICS="$REPO/scripts/img_metrics.py"

mkdir -p "$OUT_DIR"

if [[ ! -x "$KIRIE_BIN" ]]; then
    echo "FATAL: kirie binary not found at $KIRIE_BIN (build with: cargo build --release -p kirie)" >&2
    exit 2
fi
if [[ ! -d "$CORPUS" ]]; then
    echo "FATAL: corpus dir not found at $CORPUS" >&2
    exit 2
fi

mapfile -t ITEMS < <(find "$CORPUS" -mindepth 1 -maxdepth 1 -type d \
    -exec test -f '{}/scene.pkg' \; -print | sort)
if [[ "${#ITEMS[@]}" -eq 0 ]]; then
    echo "FATAL: no scene.pkg items under $CORPUS" >&2
    exit 2
fi

declare -A RENDER NB SSIM NOTE
fails=0

# ---- PHASE 1: render gate -------------------------------------------------
echo "== PHASE 1: render gate (${#ITEMS[@]} scene items) =="
for dir in "${ITEMS[@]}"; do
    id="$(basename "$dir")"
    kpng="$OUT_DIR/kirie-$id.png"
    rm -f "$kpng"
    rc=0
    timeout 180 "$KIRIE_BIN" --bg "$dir" --screenshot "$kpng" \
        --screenshot-delay "$DELAY" >/dev/null 2>&1 || rc=$?

    if [[ $rc -ne 0 ]]; then
        RENDER[$id]="CRASH($rc)"; NB[$id]="0.00"; NOTE[$id]="kirie exited nonzero"
        fails=$((fails + 1))
    elif [[ ! -f "$kpng" ]]; then
        RENDER[$id]="NOPNG"; NB[$id]="0.00"; NOTE[$id]="no screenshot written"
        fails=$((fails + 1))
    else
        nb="$(python3 "$METRICS" nonblack "$kpng" 2>/dev/null || echo 0)"
        NB[$id]="$nb"
        if awk "BEGIN{exit !($nb < $NONBLACK_MIN)}"; then
            RENDER[$id]="BLACK"; NOTE[$id]="only ${nb}% non-black (< ${NONBLACK_MIN}%)"
            fails=$((fails + 1))
        else
            RENDER[$id]="OK"; NOTE[$id]=""
        fi
    fi
    SSIM[$id]="NA"
    printf '  %-14s %-9s %s%%\n' "$id" "${RENDER[$id]}" "${NB[$id]}"
done

# ---- PHASE 2: SSIM vs C++ oracle (quality only) ---------------------------
flagged=()
if [[ "$SKIP_SSIM" != "1" && -x "$CPP_BIN" ]]; then
    echo
    echo "== PHASE 2: SSIM vs C++ oracle (fresh process) =="
    for dir in "${ITEMS[@]}"; do
        id="$(basename "$dir")"
        [[ "${RENDER[$id]}" == "OK" ]] || continue
        kpng="$OUT_DIR/kirie-$id.png"
        opng="$OUT_DIR/oracle-$id.png"
        rm -f "$opng"
        ( cd "$(dirname "$CPP_BIN")" && \
          timeout 120 "$CPP_BIN" --screenshot "$opng" \
            --screenshot-delay "$DELAY" "$dir" ) >/dev/null 2>&1
        if [[ -f "$opng" ]]; then
            s="$(python3 "$METRICS" ssim "$kpng" "$opng" 2>/dev/null || echo NA)"
            SSIM[$id]="$s"
            if [[ "$s" != "NA" ]] && awk "BEGIN{exit !($s < $SSIM_FLAG)}"; then
                flagged+=("$id"); NOTE[$id]="${NOTE[$id]:+${NOTE[$id]}; }low SSIM — review"
            fi
        else
            NOTE[$id]="${NOTE[$id]:+${NOTE[$id]}; }oracle screenshot failed"
        fi
        printf '  %-14s SSIM=%s\n' "$id" "${SSIM[$id]}"
    done
fi

# ---- FINAL TABLE ----------------------------------------------------------
echo
printf '%-14s %-10s %-10s %-9s %s\n' "ITEM" "RENDER" "NONBLACK%" "SSIM" "NOTE"
printf '%.0s-' {1..70}; echo
for dir in "${ITEMS[@]}"; do
    id="$(basename "$dir")"
    printf '%-14s %-10s %-10s %-9s %s\n' \
        "$id" "${RENDER[$id]}" "${NB[$id]}" "${SSIM[$id]}" "${NOTE[$id]}"
done

echo
echo "scene items: ${#ITEMS[@]} | render failures: $fails | flagged (SSIM < $SSIM_FLAG): ${#flagged[@]}"
[[ "${#flagged[@]}" -gt 0 ]] && printf '  flagged: %s\n' "${flagged[*]}"

if [[ $fails -eq 0 ]]; then
    echo "GATE: PASS (every scene item rendered non-black without crashing)"
    exit 0
fi
echo "GATE: FAIL ($fails item(s) did not render)"
exit 1
