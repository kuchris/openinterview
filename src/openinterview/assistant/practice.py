"""Practice assistant: question cards with reveal-after-you-answer and feedback.

Disabled by default, and fully independent of transcription: provider errors only mark
cards as failed, while ASR keeps running.

Flow (mode "practice"):
  1. A complete utterance from a listened-to speaker (default "Them") is detected as a
     interview question (technical or behavioural) -> a card is created and the answer is requested in the background.
  2. The UI shows "Your turn: answer out loud". Your mic utterances are attached to the
     card as your spoken answer.
  3. You click Reveal to see the short answer, 3-5 key points and an optional Python
     example; "Check my answer" asks the model which key points you covered or missed.
Mode "immediate" shows the answer as soon as it arrives (for note-taking in meetings).

Duplicate requests are avoided by the detector's similarity check, and by having at most one
request in flight with a small bounded backlog (oldest dropped).
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import OrderedDict, deque
from ..config import AssistantConfig
from ..queues import DropOldestQueue
from . import profile
from .detector import DetectorConfig, QuestionDetector
from .providers import Answer, AnswerProvider, OpenAICompatibleProvider, OpenCodeProvider

log = logging.getLogger(__name__)
SPEAKER = {"mic": "Me", "system": "Them"}


def make_provider(cfg: AssistantConfig) -> AnswerProvider:
    text = profile.load(profile.resolve(cfg.profile_path))
    if cfg.provider == "openai":
        import os

        if not cfg.model:
            raise ValueError("assistant.model is required for the openai provider")
        return OpenAICompatibleProvider(cfg.base_url, cfg.model,
                                        os.environ.get(cfg.api_key_env, ""), cfg.timeout_s,
                                        profile=text)
    return OpenCodeProvider(cfg.base_url, cfg.model, timeout=cfg.timeout_s, profile=text)


def import_profile(cfg: AssistantConfig, source: str, provider: AnswerProvider | None = None,
                   provider_factory=None) -> dict:
    """Import a CV into the profile: extract text, let the AI organise it (facts only, gaps
    marked [to add]), and fall back to the raw text if the AI is unavailable."""
    from pathlib import Path

    path = Path(source).expanduser()
    if not path.is_file():
        raise ValueError(f"file not found: {source}")
    dest = profile.resolve(cfg.profile_path)
    cv_text = profile.extract_text(path)
    if not cv_text:
        raise ValueError(f"no text found in {path.name} (is it a scanned image?)")
    profile.save_cv_text(cv_text, dest.parent)
    existing = profile.strip_raw_cv(profile.load(dest))  # old raw CV text is not "notes"
    temp = None
    try:
        if provider is None:
            provider = temp = (provider_factory or make_provider)(cfg)
        markdown, to_add = provider.build_profile(cv_text, existing)
        profile.write_generated(dest, markdown, to_add, path.name)
        return {"method": "ai", "to_add": to_add, "warning": ""}
    except Exception as exc:
        log.warning("AI profile build failed, importing raw text: %s", exc)
        profile.import_cv(path, dest)
        return {"method": "raw", "to_add": [],
                "warning": f"AI analysis unavailable ({exc}); imported the raw CV text instead."}
    finally:
        if temp is not None and hasattr(temp, "close"):
            temp.close()


class PracticeAssistant:
    def __init__(self, cfg: AssistantConfig, publish, provider_factory=make_provider) -> None:
        self.cfg = cfg
        self.publish = publish
        self.provider_factory = provider_factory
        self.provider: AnswerProvider | None = None
        self.enabled = False
        self.health = (False, "not checked")
        self.session_source = "mic"
        self.target: dict = {}
        self.detector = QuestionDetector(DetectorConfig(filter_small_talk=cfg.filter_small_talk))
        self.cards: OrderedDict[int, dict] = OrderedDict()
        self.context: deque[str] = deque(maxlen=12)
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self._queue: DropOldestQueue[int] = DropOldestQueue(3)
        self._worker: threading.Thread | None = None
        if cfg.enabled:
            self.set_enabled(True)

    # -- control -------------------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = enabled
            if enabled and self.provider is None:
                try:
                    self.provider = self.provider_factory(self.cfg)
                    if self.target and hasattr(self.provider, "set_context"):
                        self.provider.set_context(profile.load(self.profile_path), self.target)
                except Exception as exc:
                    self.health = (False, str(exc))
                    self.enabled = False
            if self.enabled and self.provider is not None:
                self.health = self.provider.health()
                if self._worker is None:
                    self._worker = threading.Thread(target=self._work, name="assistant",
                                                    daemon=True)
                    self._worker.start()
        self._publish_state()

    def new_session(self, target: dict) -> None:
        """A new recording session: clear its questions and transcript context, and give the
        model a fresh conversation that knows this interview's target."""
        with self._lock:
            self.cards.clear()
            self.context.clear()
            self.target = dict(target)
            self.detector = QuestionDetector(DetectorConfig(
                filter_small_talk=self.cfg.filter_small_talk))
            provider = self.provider
        if provider is not None and hasattr(provider, "set_context"):
            threading.Thread(target=provider.set_context,
                             args=(profile.load(self.profile_path), self.target),
                             daemon=True).start()
        self.publish({"type": "assistant_snapshot", **self.snapshot()})

    # -- personal profile ------------------------------------------------------------------

    @property
    def profile_path(self):
        return profile.resolve(self.cfg.profile_path)

    def profile_info(self) -> dict:
        return profile.info(self.profile_path)

    def reload_profile(self) -> dict:
        """Re-read the profile file and hand it to the provider (new context)."""
        text = profile.load(self.profile_path)
        with self._lock:
            if self.provider is not None and hasattr(self.provider, "set_profile"):
                self.provider.set_profile(text)
        self._publish_state()
        return self.profile_info()

    def import_cv(self, source: str) -> dict:
        result = import_profile(self.cfg, source, self.provider, self.provider_factory)
        info = self.reload_profile()
        return {**info, **result}

    def set_mode(self, mode: str) -> None:
        if mode not in ("practice", "immediate"):
            raise ValueError("mode must be practice or immediate")
        self.cfg.mode = mode
        self._publish_state()

    def ask(self, question: str, source: str = "typed") -> dict:
        """Ask a question directly (typed in the UI)."""
        question = question.strip()
        if not question:
            raise ValueError("empty question")
        self.detector.remember(question)
        return self._new_card(question, source)

    def grade(self, card_id: int) -> None:
        with self._lock:
            card = self.cards.get(card_id)
            if card is None or card["status"] != "ready":
                raise KeyError("no answered card with that id")
            card["feedback_status"] = "thinking"
        self._publish_card(card)
        threading.Thread(target=self._grade, args=(card_id,), daemon=True).start()

    def close(self) -> None:
        self._queue.close()
        if self.provider is not None and hasattr(self.provider, "close"):
            self.provider.close()

    # -- session events --------------------------------------------------------------------

    def listening_to(self) -> list[str]:
        """Whose utterances may be questions. "auto": the interviewer ("Them") when system
        audio is captured; your own mic when practising mic-only (reading questions aloud)."""
        if self.cfg.listen_to != ["auto"]:
            return self.cfg.listen_to
        return ["mic"] if self.session_source == "mic" else ["system"]

    def on_message(self, msg: dict) -> None:
        if msg.get("type") == "status":
            if msg.get("source") and msg["source"] != self.session_source:
                self.session_source = msg["source"]
                self._publish_state()
            if msg.get("state") == "loading":
                self.new_session(msg.get("target") or {})
            return
        if msg.get("type") != "transcript" or msg["event"]["kind"] != "utterance":
            return
        ev = msg["event"]
        text, source = ev["text"].strip(), ev["source"]
        if not text:
            return
        self.context.append(f"{SPEAKER.get(source, source)}: {text}")
        if self.enabled and source in self.listening_to() and self.detector.check(text):
            self._new_card(text, source)
        elif source == "mic":
            self._attach_spoken_answer(text)  # your answer to the latest question

        # -- internals ---------------------------------------------------------------------------

    def _new_card(self, question: str, source: str) -> dict:
        with self._lock:
            card = {"id": next(self._ids), "question": question, "source": source,
                    "created": time.time(), "status": "thinking", "answer": None,
                    "error": "", "spoken": [], "feedback": None, "feedback_status": ""}
            self.cards[card["id"]] = card
            while len(self.cards) > self.cfg.max_cards:
                self.cards.popitem(last=False)
            if not self.enabled or self.provider is None:
                card["status"] = "error"
                card["error"] = "assistant is disabled"
            else:
                dropped = self._queue.put(card["id"])
                if dropped is not None and dropped in self.cards:
                    self.cards[dropped].update(status="error", error="skipped: too many "
                                               "questions at once")
                    self._publish_card(self.cards[dropped])
        self._publish_card(card)
        return card

    def _attach_spoken_answer(self, text: str) -> None:
        with self._lock:
            if not self.cards:
                return
            card = next(reversed(self.cards.values()))
            if card["status"] in ("thinking", "ready") and not card["feedback"]:
                card["spoken"].append(text)
        self._publish_card(card)

    def _work(self) -> None:
        while True:
            card_id = self._queue.get(timeout=1.0)
            if card_id is None:
                if self._queue.closed:
                    return
                continue
            with self._lock:
                card = self.cards.get(card_id)
                if card is None:
                    continue
                question = card["question"]
                context = list(self.context)[:-1]  # exclude the question itself
            try:
                answer = self.provider.answer(question, context)
                with self._lock:
                    card.update(status="ready", answer=answer.to_dict())
            except Exception as exc:
                log.warning("assistant answer failed: %s", exc)
                with self._lock:
                    card.update(status="error", error=str(exc))
            self._publish_card(card)

    def _grade(self, card_id: int) -> None:
        with self._lock:
            card = self.cards[card_id]
            answer = Answer(**card["answer"])
            spoken = " ".join(card["spoken"])
        try:
            fb = self.provider.grade(card["question"], answer, spoken)
            with self._lock:
                card.update(feedback=fb.to_dict(), feedback_status="ready")
        except Exception as exc:
            with self._lock:
                card.update(feedback_status="error", error=f"feedback failed: {exc}")
        self._publish_card(card)

    def state(self) -> dict:
        return {"enabled": self.enabled, "mode": self.cfg.mode, "provider": self.cfg.provider,
                "healthy": self.health[0], "health": self.health[1],
                "listen_to": self.listening_to(), "profile": self.profile_info(),
                "target": self.target}

    def snapshot(self) -> dict:
        with self._lock:
            return {**self.state(), "cards": [dict(c) for c in self.cards.values()]}

    def _publish_state(self) -> None:
        self.publish({"type": "assistant_state", **self.state()})

    def _publish_card(self, card: dict) -> None:
        with self._lock:
            data = dict(card)
        self.publish({"type": "assistant_card", "card": data})
