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
    case "assistant_snapshot":
      onAssistantState(msg);
      assistant.cards.clear();
      for (const c of msg.cards) assistant.cards.set(c.id, c);
      renderCards();
      break;
    case "assistant_state":
      onAssistantState(msg);
      break;
    case "assistant_card":
      assistant.cards.set(msg.card.id, msg.card);
      renderCards();
      break;
    default:
      break;
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

// -- practice assistant ------------------------------------------------------------------

const assistant = { state: {}, cards: new Map(), revealed: new Set() };

function onAssistantState(st) {
  assistant.state = st;
  $("as-enabled").checked = !!st.enabled;
  $("as-mode").value = st.mode || "practice";
  const h = $("as-health");
  if (!st.enabled) h.textContent = st.health && st.health !== "not checked" && !st.healthy
    ? `Off: ${st.health}` : "Off. Answers come from your local OpenCode server; transcription works without it.";
  else h.textContent = st.healthy ? `On: ${st.health}. Listening to ${st.listen_to.map((s) => SPEAKER[s] || s).join(", ")}.`
    : `On, but the provider is unreachable: ${st.health}`;
  h.classList.toggle("error", !!st.enabled && !st.healthy);
}

function cardEl(c) {
  const li = el("li", "card");
  const meta = el("div", "meta");
  const chip = el("span", `chip ${c.status === "ready" ? "ok" : c.status === "error" ? "bad" : ""}`,
                  c.status === "thinking" ? "thinking…" : c.status);
  meta.append(chip, el("span", "muted", c.source === "typed" ? "typed" : `from ${SPEAKER[c.source] || c.source}`));
  li.append(meta, el("p", "q", c.question));
  if (c.error) li.append(el("p", "err", c.error));
  const practice = (assistant.state.mode || "practice") === "practice";
  const shown = !practice || assistant.revealed.has(c.id);
  if (practice && !shown && c.status !== "error") {
    li.append(el("p", "turn", "Your turn: answer out loud, then reveal the reference answer."));
  }
  if (c.spoken && c.spoken.length) li.append(el("p", "spoken", `You said: ${c.spoken.join(" ")}`));
  const actions = el("div", "actions");
  if (c.status === "ready" && c.answer) {
    if (shown) {
      li.append(el("p", "short", c.answer.short_answer));
      const ul = el("ul");
      for (const p of c.answer.key_points) ul.append(el("li", "", p));
      li.append(ul);
      if (c.answer.python_example) li.append(el("pre", "", c.answer.python_example));
      if (c.feedback) {
        if (c.feedback.covered.length) { const u = el("ul", "fb-ok"); c.feedback.covered.forEach((x) => u.append(el("li", "", x))); li.append(u); }
        if (c.feedback.missed.length) { const u = el("ul", "fb-miss"); c.feedback.missed.forEach((x) => u.append(el("li", "", x))); li.append(u); }
        if (c.feedback.tip) li.append(el("p", "spoken", `Tip: ${c.feedback.tip}`));
      } else {
        const b = el("button", "", c.feedback_status === "thinking" ? "Checking…" : "Check my answer");
        b.disabled = c.feedback_status === "thinking";
        b.onclick = () => post(`/api/assistant/${c.id}/grade`);
        actions.append(b);
      }
      if (c.answer.model) li.append(el("p", "hint", `${c.answer.model} · ${c.answer.seconds}s`));
    } else {
      const b = el("button", "", "Reveal answer");
      b.onclick = () => { assistant.revealed.add(c.id); renderCards(); };
      actions.append(b);
    }
  }
  if (actions.children.length) li.append(actions);
  return li;
}

function renderCards() {
  const cards = [...assistant.cards.values()].sort((a, b) => b.id - a.id);
  $("as-cards").replaceChildren(...cards.map(cardEl));
}

$("as-enabled").addEventListener("change", (e) => post("/api/assistant", { enabled: e.target.checked }));
$("as-mode").addEventListener("change", (e) => post("/api/assistant", { mode: e.target.value }).then(renderCards));
$("as-ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("as-question").value.trim();
  if (!q) return;
  post("/api/assistant/ask", { question: q });
  $("as-question").value = "";
});

loadOptions().catch((e) => { $("error").hidden = false; $("error").textContent = String(e); });
connect();
