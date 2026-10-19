#!/usr/bin/env bash
# Build the native helpers into bin/. Requires Xcode or Command Line Tools.
#   bin/sysaudio  system audio capture (Core Audio process tap, macOS 14.2+)
#   bin/appleasr  Apple on-device streaming ASR (SpeechAnalyzer, macOS 26+; needs Xcode 26 SDK)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bin

build() {  # name, min macOS
  local name=$1
  swiftc -O -swift-version 5 \
    -o "bin/$name" helpers/common/common.swift "helpers/$name/main.swift" \
    -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "helpers/$name/Info.plist"
  codesign --force --sign - --identifier "local.openinterview.$name" "bin/$name"
  echo "built bin/$name"
}

build sysaudio
if [[ "$(sw_vers -productVersion | cut -d. -f1)" -ge 26 ]]; then
  build appleasr || echo "note: appleasr build failed (needs the macOS 26 SDK); Apple ASR unavailable"
else
  echo "note: macOS < 26; skipping appleasr (Apple SpeechAnalyzer needs macOS 26)"
fi
