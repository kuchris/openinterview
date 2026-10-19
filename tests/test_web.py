"""Dashboard server: HTTP API + WebSocket, with a fake session (no audio, no model)."""

import functools
import json
import time

import numpy as np
from fastapi.testclient import TestClient

from openinterview.asr.transcription import PipelineConfig, StreamingTranscriber
from openinterview.asr.vad import EnergyVAD
from openinterview.audio.buffer import AudioRingBuffer
from openinterview.audio.file_source import FileSource
from openinterview.config import Config
from openinterview.session import Session
from openinterview.web.server import create_app
from test_pipeline import FakeEngine, silence, tone

SR = 16000


def fake_sources(cfg, source, on_status):
    audio = np.concatenate([silence(0.3), tone(1.2), silence(1.0)])
    return [FileSource(AudioRingBuffer(30, SR), audio, name="mic", speed=2.0, tail_silence_s=0)]


def fake_transcriber(cfg, sources, on_event):
    return StreamingTranscriber(lambda: FakeEngine(), sources, on_event,
                                vad_factory=lambda: EnergyVAD(0.01),
                                config=PipelineConfig(prompt_with_context=False))


def make_client():
    factory = functools.partial(Session, sources_factory=fake_sources,
                                transcriber_factory=fake_transcriber)
    return TestClient(create_app(Config(), session_factory=factory))


def recv_until(ws, pred, limit=500):
    for _ in range(limit):
        msg = json.loads(ws.receive_text())
        if pred(msg):
            return msg
    raise AssertionError("expected message not received")


def test_index_and_state():
    with make_client() as client:
        r = client.get("/")
        assert r.status_code == 200 and "OpenInterview" in r.text
        state = client.get("/api/state").json()
        assert "small.en" in state["options"]["models"]
        assert state["status"]["state"] == "idle"
        assert client.get("/static/app.js").status_code == 200


def test_start_stream_stop_export():
    with make_client() as client, client.websocket_connect("/ws") as ws:
        snap = json.loads(ws.receive_text())
        assert snap["type"] == "snapshot" and snap["lines"] == []
        assert client.post("/api/start", json={"source": "mic", "model": "base.en"}).status_code == 200
        recv_until(ws, lambda m: m["type"] == "status" and m["state"] == "running")
        final = recv_until(ws, lambda m: m["type"] == "transcript"
                           and m["event"]["kind"] == "final" and m["event"]["text"])
        assert final["event"]["speaker"] == "Me"
        recv_until(ws, lambda m: m["type"] == "metrics")
        client.post("/api/stop")
        recv_until(ws, lambda m: m["type"] == "status" and m["state"] == "idle")
        md = client.get("/api/transcript.md").text
        assert "**Me**" in md and final["event"]["text"] in md


def test_double_start_conflicts_and_bad_source_rejected():
    with make_client() as client:
        assert client.post("/api/start", json={"source": "nope"}).status_code == 400
        assert client.post("/api/start", json={}).status_code == 200
        assert client.post("/api/start", json={}).status_code == 409
        client.post("/api/stop")
        session = client.app.state.session
        assert session.wait_state({"idle"}, 15)


def test_late_client_gets_snapshot_of_existing_transcript():
    with make_client() as client:
        session = client.app.state.session
        client.post("/api/start", json={})
        assert session.wait_state({"running"}, 10)
        deadline = time.monotonic() + 10
        while not session.lines and time.monotonic() < deadline:
            time.sleep(0.05)
        with client.websocket_connect("/ws") as ws:
            snap = json.loads(ws.receive_text())
            assert snap["type"] == "snapshot" and snap["lines"]
        client.post("/api/stop")
        assert session.wait_state({"idle"}, 15)


def test_hub_coalesces_partials():
    from openinterview.web.server import Hub
    import asyncio

    hub = Hub()
    loop = asyncio.new_event_loop()
    hub.attach(loop)
    for i in range(50):
        hub._enqueue({"type": "transcript", "event": {"kind": "partial", "source": "mic", "i": i}})
    assert hub.queue.qsize() == 0 and hub._pending_partials["mic"]["event"]["i"] == 49
    hub._enqueue({"type": "transcript", "event": {"kind": "final", "source": "mic"}})
    assert "mic" not in hub._pending_partials and hub.queue.qsize() == 1
    loop.close()
