"use strict";

const $ = (id) => document.getElementById(id);
const SPEAKER = { mic: "Me", system: "Them" };
const state = { status: { state: "idle" }, sources: {}, lines: new Map(), partials: {} };
let ws = null;
let retryMs = 500;
let stickToBottom = true;

// -- formatting -------------------------------------------------------------------------

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
function ms(summary) {
  return summary && summary.n ? `${Math.round(summary.p50)} ms` : "–";
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// -- transcript -------------------------------------------------------------------------

function lineKey(ev) { return `${ev.source}:${ev.segment_id}`; }

function addLine(ev) {
  const li = el("li", "line");
  li.dataset.key = lineKey(ev);
  li.append(el("span", "t", clock(ev.start)), el("span", `who ${ev.source}`, SPEAKER[ev.source] || ev.source),
            el("span", "text", ev.text));
  $("lines").append(li);
  state.lines.set(li.dataset.key, li);
}

function removeLine(ev) {
  const li = state.lines.get(lineKey(ev));
  if (li) { li.remove(); state.lines.delete(lineKey(ev)); }
}

function renderPartials() {
  const box = $("partials");
  box.replaceChildren();
  for (const [source, ev] of Object.entries(state.partials)) {
    const row = el("div", "partial");
    const text = el("span", "text");
    text.append(el("span", "stable", ev.stable ? ev.stable + " " : ""), el("span", "unstable", ev.unstable || ""));
    row.append(el("span", "t", clock(ev.start)), el("span", `who ${source}`, SPEAKER[source] || source), text);
    box.append(row);
  }
  const t = $("transcript");
  if (stickToBottom) t.scrollTop = t.scrollHeight;
}

function afterTranscriptChange() {
  $("empty").hidden = state.lines.size > 0;
  $("count").textContent = state.lines.size ? `${state.lines.size} line${state.lines.size === 1 ? "" : "s"}` : "";
  const t = $("transcript");
  if (stickToBottom) t.scrollTop = t.scrollHeight;
}

function onTranscript(ev) {
  if (ev.kind === "partial") {
    state.partials[ev.source] = ev;
    renderPartials();
  } else if (ev.kind === "final") {
    delete state.partials[ev.source];
    renderPartials();
    if (ev.text) addLine(ev);
  } else if (ev.kind === "retract") {
    removeLine(ev);
  }
  afterTranscriptChange();
}

// -- status / metrics -------------------------------------------------------------------

function renderStatus() {
  const s = state.status;
  const running = s.state === "running";
  const busy = s.state === "loading" || s.state === "stopping";
  $("rec").className = `rec ${running ? "on" : "off"}`;
  $("rec-label").textContent = running ? "Recording" : s.state === "loading" ? "Starting…" : "Not recording";
  const chip = $("state");
  chip.textContent = s.state === "loading" ? `loading ${s.model}…` : s.state;
  chip.className = `chip ${running ? "ok" : s.state === "error" ? "bad" : ""}`;
  const btn = $("toggle");
  btn.textContent = running || s.state === "loading" ? "Stop" : s.state === "stopping" ? "Stopping…" : "Start";
  btn.classList.toggle("stop", running || s.state === "loading");
  btn.disabled = s.state === "stopping";
  for (const id of ["source", "model", "mic", "save"]) $(id).disabled = running || busy;
  $("error").hidden = !s.error;
  $("error").textContent = s.error || "";
}

function renderSources() {
  const box = $("sources");
  box.replaceChildren();
  for (const [name, info] of Object.entries(state.sources)) {
    const ok = info.state === "running";
    const bad = info.state === "error" || info.state === "reconnecting";
    const c = el("span", `chip ${ok ? "ok" : bad ? "warn" : ""}`,
                 `${SPEAKER[name] || name}: ${info.state}${info.backend ? " · " + info.backend : ""}`);
    if (info.detail) c.title = info.detail;
    box.append(c);
  }
}

function onMetrics(m) {
  const p = m.pipeline || {};
  $("m-final").textContent = p.final_latency_ms && p.final_latency_ms.n
    ? `${Math.round(p.final_latency_ms.p50)} / ${Math.round(p.final_latency_ms.p95)} ms` : "–";
  $("m-first").textContent = ms(p.first_partial_ms);
  $("m-asr").textContent = ms(p.inference_final_ms);
  $("m-rtf").textContent = p.rtf && p.rtf.n ? p.rtf.p50.toFixed(3) : "–";
  $("m-queue").textContent = p.pending_jobs !== undefined ? String(p.pending_jobs) : "–";
  const echo = (p.echo_gated || 0) + (p.echo_text_dropped || 0) + (p.echo_retracted || 0);
  $("m-drop").textContent = p.finals_dropped !== undefined ? `${p.finals_dropped} / ${echo}` : "–";
  const mem = m.memory || {};
  $("m-mem").textContent = mem.rss_peak_mb ? `${Math.round(mem.rss_peak_mb)} / ${Math.round(mem.mlx_peak_mb || 0)} MB` : "–";
  $("degraded").hidden = !m.degraded;
  for (const [name, st] of Object.entries(m.sources || {})) {
    state.sources[name] = { ...(state.sources[name] || {}), state: st.state, backend: st.backend };
  }
  renderSources();
}

// -- websocket --------------------------------------------------------------------------

function onMessage(msg) {
  switch (msg.type) {
    case "snapshot":
      state.status = msg.status;
      $("lines").replaceChildren();
      state.lines.clear();
      for (const ev of msg.lines) addLine(ev);
      state.partials = msg.partials || {};
      renderPartials();
      renderStatus();
      afterTranscriptChange();
      break;
    case "status":
      if (msg.state === "loading") {  // new session: clear the previous transcript view
        $("lines").replaceChildren(); state.lines.clear(); state.partials = {}; state.sources = {};
        renderPartials(); renderSources(); afterTranscriptChange();
      }
      state.status = msg;
      renderStatus();
      break;
    case "source":
      state.sources[msg.source] = { ...(state.sources[msg.source] || {}), state: msg.state, detail: msg.detail };
      renderSources();
      break;
    case "transcript":
      onTranscript(msg.event);
      break;
    case "metrics":
      onMetrics(msg);
      break;
    default:
      if (window.onAssistantMessage) window.onAssistantMessage(msg);
  }
}

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { $("conn").textContent = "connected"; retryMs = 500; };
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => {
    $("conn").textContent = "disconnected, retrying…";
    setTimeout(connect, retryMs);
    retryMs = Math.min(retryMs * 2, 5000);
  };
}

// -- controls ---------------------------------------------------------------------------

async function loadOptions() {
  const res = await fetch("/api/state");
  const data = await res.json();
  const o = data.options;
  $("model").replaceChildren(...o.models.map((m) => { const x = el("option", "", m); x.value = m; return x; }));
  $("model").value = o.defaults.model;
  const mics = [el("option", "", "System default")];
  mics[0].value = "";
  for (const d of o.devices) { const x = el("option", "", d.name); x.value = d.name; mics.push(x); }
  $("mic").replaceChildren(...mics);
  $("mic").value = o.defaults.mic_device || "";
  $("source").value = o.defaults.source;
  $("save").checked = !!o.defaults.save_transcript;
  syncMicVisibility();
}

function syncMicVisibility() { $("mic-wrap").hidden = $("source").value === "system"; }

async function post(path, body) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify(body || {}) });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    $("error").hidden = false;
    $("error").textContent = err.detail || "request failed";
  }
}

$("controls").addEventListener("submit", (e) => {
  e.preventDefault();
  const s = state.status.state;
  if (s === "running" || s === "loading") return post("/api/stop");
  return post("/api/start", { source: $("source").value, model: $("model").value,
                              mic_device: $("mic").value || null, save_transcript: $("save").checked });
});
$("source").addEventListener("change", syncMicVisibility);
$("transcript").addEventListener("scroll", () => {
  const t = $("transcript");
  stickToBottom = t.scrollHeight - t.scrollTop - t.clientHeight < 40;
});

loadOptions().catch((e) => { $("error").hidden = false; $("error").textContent = String(e); });
connect();
