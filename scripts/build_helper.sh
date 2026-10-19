#!/usr/bin/env bash
# Build the system-audio helper into bin/sysaudio. Requires Xcode or Command Line Tools.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bin
swiftc -O -swift-version 5 \
  -o bin/sysaudio helpers/sysaudio/main.swift \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker helpers/sysaudio/Info.plist
codesign --force --sign - --identifier local.openinterview.sysaudio bin/sysaudio
echo "built bin/sysaudio"
