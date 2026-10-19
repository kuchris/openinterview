"""Stand-in for bin/sysaudio in tests: emits the helper protocol, then exits as told.

usage: fake_sysaudio.py <frames> <exit_code> [stall|idle]
  stall: stay alive but send nothing (no heartbeat) -> supervisor must restart
  idle:  keep heartbeating without frames (nobody playing audio) -> must NOT restart
"""
import json
import struct
import sys
import time

import numpy as np

frames, code = int(sys.argv[1]), int(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 else ""
print(json.dumps({"event": "format", "sample_rate": 48000}), file=sys.stderr, flush=True)
print(json.dumps({"event": "started"}), file=sys.stderr, flush=True)
block = (0.1 * np.ones(160)).astype("<f4").tobytes()
out = sys.stdout.buffer
for _ in range(frames):
    out.write(struct.pack("<IIQ", 0x31554153, 160, time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)))
    out.write(block)
    out.flush()
    time.sleep(0.01)
if mode == "stall":
    time.sleep(60)  # alive but silent: the supervisor must notice and restart
elif mode == "idle":
    for _ in range(600):
        print(json.dumps({"event": "heartbeat"}), file=sys.stderr, flush=True)
        time.sleep(0.1)
sys.exit(code)
