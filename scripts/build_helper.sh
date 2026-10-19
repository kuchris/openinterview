#!/usr/bin/env bash
# Build the native helpers into bin/. Requires Xcode or Command Line Tools.
#   bin/sysaudio          system audio capture (Core Audio process tap, macOS 14.2+)
#   bin/appleasr          Apple on-device streaming ASR (SpeechAnalyzer, macOS 26+)
#   bin/openinterview-ui  native app window (SwiftUI, macOS 14+)
#   bin/pdftext           PDF text via PDFKit (CV import)
#
# A binary is rebuilt only when one of its sources is newer (use --force to rebuild all).
# This matters: binaries are ad-hoc signed, and macOS asks for privacy permissions again
# whenever sysaudio's signature changes.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bin
FORCE=${1:-}

stale() {  # out, sources...
  local out=$1; shift
  [[ "$FORCE" == "--force" || ! -f "$out" ]] && return 0
  for src in "$@"; do [[ "$src" -nt "$out" ]] && return 0; done
  return 1
}

build() {  # name, extra swiftc flags, sources...
  local name=$1 flags=$2; shift 2
  local out="bin/$name"
  if ! stale "$out" "$@" "helpers/${name/openinterview-/}/Info.plist"; then
    echo "up to date: $out"
    return
  fi
  # shellcheck disable=SC2086
  swiftc -O -swift-version 5 $flags -o "$out" "$@" \
    -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist \
    -Xlinker "helpers/${name/openinterview-/}/Info.plist"
  codesign --force --sign - --identifier "local.openinterview.${name/openinterview-/}" "$out"
  echo "built $out"
}

build sysaudio "" helpers/common/common.swift helpers/sysaudio/main.swift
build openinterview-ui "-parse-as-library" helpers/ui/OpenInterviewApp.swift helpers/ui/Views.swift
build pdftext "" helpers/pdftext/main.swift
if [[ "$(sw_vers -productVersion | cut -d. -f1)" -ge 26 ]]; then
  build appleasr "" helpers/common/common.swift helpers/appleasr/main.swift \
    || echo "note: appleasr build failed (needs the macOS 26 SDK); Apple ASR unavailable"
else
  echo "note: macOS < 26; skipping appleasr (Apple SpeechAnalyzer needs macOS 26)"
fi
