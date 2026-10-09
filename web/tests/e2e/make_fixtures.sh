#!/usr/bin/env bash
# Regenerates the E2E media fixtures in var/e2e from the scenario media in
# var/examples (built by `petperception evaluate e5`, which composites public
# Open Images photos with ESC-50 audio; see docs/TEST_REPORT.md).
# Fixtures are derived media, not real household recordings, and are not
# committed (var/ is ignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EX="$ROOT/var/examples"
OUT="$ROOT/var/e2e"
if [[ ! -f "$EX/s01_food_zone_meow.mp4" || ! -f "$EX/s07_audio_only_meow.wav" ]]; then
  echo "scenario media missing; run: uv run petperception evaluate e5" >&2
  exit 1
fi
mkdir -p "$OUT"
cd "$OUT"
ff() { ffmpeg -nostdin -v error -y "$@"; }
# Fake camera / microphone streams for Chromium (--use-file-for-fake-*-capture).
ff -i "$EX/s01_food_zone_meow.mp4" -t 8 -pix_fmt yuv420p -r 15 cat_camera.y4m
ff -i "$EX/s07_audio_only_meow.wav" -ar 48000 -ac 1 meow_mic.wav
# ~6.6 MB clip (temporal noise defeats compression) so uploads need several chunks.
ff -i "$EX/s01_food_zone_meow.mp4" -vf "noise=alls=25:allf=t" -c:v libx264 -b:v 6M -maxrate 6M -bufsize 12M -preset veryfast -c:a aac big_cat_clip.mp4
# WebM copy for on-device tests: open-source Chromium (Playwright) has no H.264/AAC decoder.
ff -i big_cat_clip.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 -deadline realtime -cpu-used 8 -c:a libopus big_cat_clip.webm
# Truncated container: must be rejected with an actionable error.
head -c 6000 "$EX/s01_food_zone_meow.mp4" > broken.mp4
# 64 s clip used by the cancellation test (same recipe as E6).
[[ -f "$EX/e6_long_64s.mp4" ]] || ff -stream_loop 7 -i "$EX/s01_food_zone_meow.mp4" -c copy "$EX/e6_long_64s.mp4"
ls -la "$OUT"
