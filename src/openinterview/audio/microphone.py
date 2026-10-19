"""Input-device capture via sounddevice (PortAudio / Core Audio).

Used for the microphone and for the BlackHole system-audio fallback (a loopback *input*
device). The PortAudio callback only downmixes, optionally resamples (soxr streaming, C
code), and copies into the ring buffer. A watchdog thread reopens the stream if the device
stops delivering audio (unplugged headset, AirPods switching away, default device change).
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import sounddevice as sd
import soxr

from . import portaudio
from .buffer import AudioRingBuffer
from .source import SourceState, StatusCallback

log = logging.getLogger(__name__)

STALL_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 10.0


def list_input_devices() -> list[dict]:
    devices = []
    default_in = sd.default.device[0]
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({
                "index": idx,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "default_samplerate": dev["default_samplerate"],
                "is_default": idx == default_in,
            })
    return devices


def resolve_device(device: int | str | None) -> int | None:
    """Accept an index, a case-insensitive name substring, or None (system default)."""
    if device is None or isinstance(device, int):
        return device
    if isinstance(device, str) and device.isdigit():
        return int(device)
    matches = [d for d in list_input_devices() if device.lower() in d["name"].lower()]
    if not matches:
        raise ValueError(f"No input device matching {device!r}")
    return matches[0]["index"]


class MicrophoneSource:
    def __init__(
        self,
        buffer: AudioRingBuffer,
        device: int | str | None = None,
        block_ms: int = 32,
        on_status: StatusCallback | None = None,
        name: str = "mic",
    ) -> None:
        self.name = name
        self.buffer = buffer
        self.device_spec = device
        self.block_ms = block_ms
        self.on_status = on_status
        self.state = SourceState.STOPPED
        self.device_name = "?"
        self.capture_rate = buffer.sample_rate

        self._stream: sd.InputStream | None = None
        self._resampler: soxr.ResampleStream | None = None
        self._channels = 1
        self._last_callback = 0.0
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None

        # Stats (written from the callback; plain int/float assignment is atomic enough here).
        self.callbacks = 0
        self.overflows = 0
        self.reconnects = 0
        self.last_capture_latency = 0.0  # seconds from ADC to callback, per PortAudio

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._set_state(SourceState.STARTING, "")
        self._open()
        self._watchdog = threading.Thread(target=self._watch, name=f"{self.name}-watchdog",
                                          daemon=True)
        self._watchdog.start()

    def stop(self) -> None:
        self._stop.set()
        if self._watchdog:
            self._watchdog.join(timeout=2)
        self._close()
        self._set_state(SourceState.STOPPED, "")

    def stats(self) -> dict:
        return {
            "device": self.device_name,
            "capture_rate": self.capture_rate,
            "state": self.state.value,
            "callbacks": self.callbacks,
            "overflows": self.overflows,
            "reconnects": self.reconnects,
            "capture_latency_ms": round(self.last_capture_latency * 1000, 1),
        }

    # -- internals ---------------------------------------------------------------------

    def _set_state(self, state: SourceState, detail: str) -> None:
        self.state = state
        if self.on_status:
            self.on_status(self.name, state, detail)

    def _open(self) -> None:
        with portaudio.lock():
            self._open_locked()

    def _open_locked(self) -> None:
        device = resolve_device(self.device_spec)
        info = sd.query_devices(device if device is not None else sd.default.device[0])
        self.device_name = info["name"]
        target = self.buffer.sample_rate
        # Loopback devices (BlackHole) carry stereo program audio: mix both channels.
        channels = 2 if self.name != "mic" and info["max_input_channels"] >= 2 else 1
        try:
            sd.check_input_settings(device=device, samplerate=target, channels=channels,
                                    dtype="float32")
            rate, self._resampler = target, None
        except sd.PortAudioError:
            rate = int(info["default_samplerate"])
            self._resampler = soxr.ResampleStream(rate, target, 1, dtype="float32", quality="HQ")
        self.capture_rate = rate
        self._channels = channels
        self._stream = sd.InputStream(
            device=device,
            samplerate=rate,
            channels=self._channels,
            dtype="float32",
            blocksize=int(rate * self.block_ms / 1000),
            latency="low",
            callback=self._callback,
        )
        self._last_callback = time.monotonic()
        self._stream.start()
        self._set_state(SourceState.RUNNING, f"{self.device_name} @ {rate} Hz")
        log.info("%s opened: %s @ %d Hz (resample=%s)", self.name, self.device_name, rate,
                 self._resampler is not None)

    def _close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort(ignore_errors=True)
                stream.close(ignore_errors=True)
            except Exception:  # device may already be gone
                pass

    def _callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags) -> None:
        now = time.monotonic()
        self._last_callback = now
        self.callbacks += 1
        if status.input_overflow:
            self.overflows += 1
        try:
            self.last_capture_latency = max(0.0, time_info.currentTime - time_info.inputBufferAdcTime)
        except Exception:
            pass
        mono = indata[:, 0] if self._channels == 1 else indata.mean(axis=1, dtype=np.float32)
        if self._resampler is not None:
            mono = self._resampler.resample_chunk(mono)
        self.buffer.write(mono, now)

    def _watch(self) -> None:
        backoff = 1.0
        while not self._stop.wait(0.5):
            stream = self._stream
            healthy = (
                stream is not None
                and stream.active
                and time.monotonic() - self._last_callback < STALL_SECONDS
            )
            if healthy:
                backoff = 1.0
                continue
            self._set_state(SourceState.RECONNECTING, "input stalled or device lost")
            self._close()
            if self._stop.wait(backoff):
                return
            try:
                # PortAudio caches the device list; reinitialise (shared with other
                # PortAudio sources, rate-limited) to see devices that came back.
                portaudio.reset()
                self._open()
                self.reconnects += 1
                backoff = 1.0
            except Exception as exc:
                log.warning("%s reconnect failed: %s", self.name, exc)
                self._set_state(SourceState.RECONNECTING, f"retrying: {exc}")
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
