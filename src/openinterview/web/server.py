"""Local dashboard: FastAPI + WebSocket, bound to 127.0.0.1 only.

    openinterview serve [--port 8765]

HTTP
  GET  /                   dashboard (static/index.html)
  GET  /api/state          options (models, sources, devices) + session snapshot
  POST /api/start          {"source", "model", "mic_device", "save_transcript"}
  POST /api/stop
  GET  /api/transcript.md  export the in-memory transcript
  GET  /api/assistant      practice assistant state + cards
  POST /api/assistant      {"enabled"?: bool, "mode"?: "practice"|"immediate"}
  POST /api/assistant/ask  {"question": str}   ask a typed question
  POST /api/assistant/{id}/grade               feedback on your spoken answer
WebSocket /ws
  server -> client: the Session messages (status, source, transcript, metrics) plus a
  "snapshot" on connect. Partials are coalesced to at most one per source per 100 ms.
  client -> server: nothing required (pings are ignored).
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..assistant.practice import PracticeAssistant, make_provider
from ..config import Config
from ..session import Session

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
PARTIAL_FLUSH_S = 0.1


class Hub:
    """Fan-out from worker threads to all connected websockets."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue | None = None
        self._pending_partials: dict[str, dict] = {}

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.queue = asyncio.Queue(maxsize=10000)

    def sink(self, msg: dict) -> None:
        """Thread-safe entry point used as the Session sink."""
        if self.loop is None or self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(self._enqueue, msg)

    def _enqueue(self, msg: dict) -> None:
        if msg.get("type") == "transcript" and msg["event"]["kind"] == "partial":
            self._pending_partials[msg["event"]["source"]] = msg  # newest wins
            return
        if msg.get("type") == "transcript" and msg["event"]["kind"] == "final":
            self._pending_partials.pop(msg["event"]["source"], None)  # final supersedes
        try:
            self.queue.put_nowait(msg)
        except asyncio.QueueFull:  # a stuck client must not grow memory without bound
            log.warning("dashboard queue full; dropping message")

    async def run(self) -> None:
        while True:
            try:
                msg = await asyncio.wait_for(self.queue.get(), timeout=PARTIAL_FLUSH_S)
                await self.broadcast(msg)
            except asyncio.TimeoutError:
                pass
            if self._pending_partials:
                pending, self._pending_partials = self._pending_partials, {}
                for msg in pending.values():
                    await self.broadcast(msg)

    async def broadcast(self, msg: dict) -> None:
        if not self.clients:
            return
        data = json.dumps(msg)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


def create_app(cfg: Config, session_factory=None, provider_factory=make_provider) -> FastAPI:
    hub = Hub()
    session: Session = (session_factory or Session)(cfg, hub.sink)
    assistant = PracticeAssistant(cfg.assistant, hub.sink, provider_factory)
    session.listeners.append(assistant.on_message)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.attach(asyncio.get_running_loop())
        task = asyncio.create_task(hub.run())
        yield
        session.stop()
        session.wait_state({"idle", "error"}, timeout=25)
        assistant.close()
        task.cancel()

    app = FastAPI(title="OpenInterview", lifespan=lifespan)
    app.state.session = session
    app.state.hub = hub
    app.state.assistant = assistant

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    async def state():
        return {"options": _options(cfg), **session.snapshot()}

    @app.post("/api/start")
    async def start(body: dict = Body(default={})):
        if body.get("source") not in (None, "mic", "system", "both"):
            raise HTTPException(400, "source must be mic, system or both")
        try:
            session.start(source=body.get("source"), model=body.get("model"),
                          mic_device=body.get("mic_device"),
                          save_transcript=body.get("save_transcript"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return session.status()

    @app.post("/api/stop")
    async def stop():
        session.stop()
        return session.status()

    @app.get("/api/transcript.md", response_class=PlainTextResponse)
    async def transcript_md():
        return PlainTextResponse(session.export_markdown(), headers={
            "Content-Disposition": 'attachment; filename="transcript.md"'})

    @app.get("/api/assistant")
    async def assistant_get():
        return assistant.snapshot()

    @app.post("/api/assistant")
    async def assistant_set(body: dict = Body(default={})):
        try:
            if "mode" in body:
                assistant.set_mode(body["mode"])
            if "enabled" in body:
                # health check does network I/O: keep it off the event loop
                await asyncio.to_thread(assistant.set_enabled, bool(body["enabled"]))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return assistant.state()

    @app.post("/api/assistant/ask")
    async def assistant_ask(body: dict = Body(default={})):
        try:
            return assistant.ask(str(body.get("question", "")))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/assistant/{card_id}/grade")
    async def assistant_grade(card_id: int):
        try:
            assistant.grade(card_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "snapshot", **session.snapshot()}))
        await websocket.send_text(json.dumps({"type": "metrics", **session.metrics()}))
        await websocket.send_text(json.dumps({"type": "assistant_snapshot",
                                              **assistant.snapshot()}))
        hub.clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(websocket)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _options(cfg: Config) -> dict:
    from ..asr.factory import MODELS
    from ..asr.apple_stream import available as apple_available

    try:
        from ..audio.microphone import list_input_devices

        devices = [d for d in list_input_devices() if "blackhole" not in d["name"].lower()]
    except Exception:
        devices = []
    models = [m for m in MODELS if m != "apple" or apple_available()]
    return {
        "models": models,
        "sources": ["mic", "system", "both"],
        "devices": devices,
        "defaults": {"model": cfg.asr.model, "source": cfg.audio.source,
                     "mic_device": cfg.audio.mic_device,
                     "save_transcript": cfg.storage.save_transcripts},
    }


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    import socket

    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("the dashboard only binds to localhost")
    with socket.socket() as probe:
        if probe.connect_ex((host, port)) == 0:
            raise SystemExit(f"A dashboard is already running at http://{host}:{port} "
                             "(open it in the browser, or stop it first).")
    print(f"OpenInterview dashboard: http://{host}:{port}")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
