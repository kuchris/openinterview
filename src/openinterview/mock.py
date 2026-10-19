"""A scripted mock interview for testing the whole system end to end.

An "interviewer" (macOS `say`) reads a fixed script through the speakers. The script mixes
real questions (technical and behavioural) with lines the assistant should ignore: small
talk, statements, and a repeated question. Each line says what is expected.

    openinterview mock                 print the script and what the detector decides
    openinterview mock --app           drive the running app: start a session, speak,
                                       wait for each answer, stop, print a report
    openinterview mock --url URL       same, against `openinterview serve`
    openinterview mock --play          just speak it (start the app yourself)
    openinterview mock --render f.wav  write it to a 16 kHz WAV (for transcribe-file)

The session uses the system source, so it works with headphones or speakers.
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .asr.filters import normalize_text
from .assistant.detector import QuestionDetector
from .languages import get as get_language
from .text import tokens

TARGET = {
    "company": "Acme Pay",
    "role": "Backend engineer",
    "job_description": "Build and run payment APIs in Python and Go. Own reliability of "
                       "high-volume transaction services: idempotency, retries, rate limits, "
                       "PostgreSQL, Kafka, on-call.",
}


@dataclass(frozen=True)
class Line:
    text: str
    expect: str  # "question" | "ignore"
    why: str = ""


SCRIPT = [
    Line("Hi, thanks for joining today. Can you hear me okay?", "ignore", "small talk"),
    Line("Great. I'm Sam, and I lead the payments platform team here.", "ignore", "statement"),
    Line("So, tell me about yourself.", "question", "behavioural"),
    Line("Why do you want to work at Acme Pay?", "question", "behavioural"),
    Line("What is the difference between a process and a thread?", "question", "technical"),
    Line("Okay, that makes sense. Let me take a note of that.", "ignore", "statement"),
    Line("Sorry, what is the difference between a process and a thread?", "ignore",
         "repeat of an earlier question"),
    Line("How would you design a rate limiter for a payments API?", "question", "technical"),
    Line("Describe a time when you had to debug a difficult production issue.", "question",
         "behavioural"),
    Line("Explain how you would make a payment request idempotent.", "question", "technical"),
    Line("What is your biggest weakness?", "question", "behavioural"),
    Line("Tell me about a conflict you had with a teammate, and how you resolved it.",
         "question", "behavioural"),
    Line("Do you have any questions for us?", "question", "closing"),
    Line("Thanks, that's all from me. We'll be in touch.", "ignore", "statement"),
]

Q, I = "question", "ignore"

# The same interview in other languages, line for line (same expectations as SCRIPT).
SCRIPTS = {
    "en": SCRIPT,
    "ja": [
        Line("本日はお時間いただきありがとうございます。音声は聞こえますか？", I, "small talk"),
        Line("私は決済プラットフォームチームのリーダーの佐藤です。", I, "statement"),
        Line("では、自己紹介をお願いします。", Q, "behavioural"),
        Line("なぜ弊社で働きたいと思ったのですか？", Q, "behavioural"),
        Line("プロセスとスレッドの違いは何ですか？", Q, "technical"),
        Line("なるほど、わかりました。メモしておきます。", I, "statement"),
        Line("すみません、プロセスとスレッドの違いは何ですか？", I, "repeat"),
        Line("決済APIのレートリミッターをどのように設計しますか？", Q, "technical"),
        Line("本番環境で難しい障害を調査した経験について教えてください。", Q, "behavioural"),
        Line("決済リクエストを冪等にするにはどうしますか？", Q, "technical"),
        Line("あなたの一番の弱みは何ですか？", Q, "behavioural"),
        Line("チームメンバーと意見が対立した時、どう解決しましたか？", Q, "behavioural"),
        Line("何か質問はありますか？", Q, "closing"),
        Line("本日はありがとうございました。また連絡します。", I, "statement"),
    ],
    "zh": [
        Line("你好，感谢你今天来参加面试。能听到我说话吗？", I, "small talk"),
        Line("我是支付平台团队的负责人小王。", I, "statement"),
        Line("那么，请先做一个自我介绍。", Q, "behavioural"),
        Line("你为什么想加入我们公司？", Q, "behavioural"),
        Line("进程和线程有什么区别？", Q, "technical"),
        Line("好的，明白了，我记一下。", I, "statement"),
        Line("不好意思，进程和线程有什么区别？", I, "repeat"),
        Line("你会如何为支付API设计一个限流器？", Q, "technical"),
        Line("说说你在生产环境中排查一个困难问题的经历。", Q, "behavioural"),
        Line("你会怎么让支付请求做到幂等？", Q, "technical"),
        Line("你最大的缺点是什么？", Q, "behavioural"),
        Line("你和同事发生过冲突吗？你是怎么解决的？", Q, "behavioural"),
        Line("你有什么问题想问我们吗？", Q, "closing"),
        Line("好的，今天就到这里，我们会再联系你。", I, "statement"),
    ],
    "zh-TW": [
        Line("你好，謝謝你今天來參加面試。聽得到我的聲音嗎？", I, "small talk"),
        Line("我是支付平台團隊的負責人小王。", I, "statement"),
        Line("那麼，請先簡單自我介紹一下。", Q, "behavioural"),
        Line("你為什麼想加入我們公司？", Q, "behavioural"),
        Line("程序和執行緒有什麼差別？", Q, "technical"),
        Line("好的，了解，我記一下。", I, "statement"),
        Line("不好意思，程序和執行緒有什麼差別？", I, "repeat"),
        Line("你會怎麼為支付API設計限流機制？", Q, "technical"),
        Line("說說你在正式環境排查困難問題的經驗。", Q, "behavioural"),
        Line("你會怎麼讓支付請求做到冪等？", Q, "technical"),
        Line("你最大的缺點是什麼？", Q, "behavioural"),
        Line("你跟同事有過衝突嗎？你是怎麼解決的？", Q, "behavioural"),
        Line("你有什麼問題想問我們嗎？", Q, "closing"),
        Line("好，今天就到這裡，我們會再跟你聯絡。", I, "statement"),
    ],
    "yue": [
        Line("你好，多謝你今日嚟面試。聽唔聽到我講嘢？", I, "small talk"),
        Line("我係支付平台團隊嘅主管阿明。", I, "statement"),
        Line("咁你先介紹下你自己啦。", Q, "behavioural"),
        Line("點解你想加入我哋公司？", Q, "behavioural"),
        Line("Process同thread有咩分別？", Q, "technical, mixed English"),
        Line("好，明白，我記低先。", I, "statement"),
        Line("唔好意思，process同thread有咩分別？", I, "repeat"),
        Line("你會點樣幫支付API設計一個rate limiter？", Q, "technical, mixed English"),
        Line("講下你喺production處理過最難搞嘅問題。", Q, "behavioural"),
        Line("你會點樣令支付request做到idempotent？", Q, "technical, mixed English"),
        Line("你最大嘅弱點係咩？", Q, "behavioural"),
        Line("你同同事有冇試過有衝突？你點樣解決？", Q, "behavioural"),
        Line("你有冇問題想問我哋？", Q, "closing"),
        Line("好，今日就到呢度，我哋會再聯絡你。", I, "statement"),
    ],
}

QUICK = [0, 2, 4, 5, 6, 10]  # small talk, intro, technical, statement, repeat, weakness


def script(quick: bool = False, language: str = "en") -> list[Line]:
    lines = SCRIPTS[language]
    return [lines[i] for i in QUICK] if quick else list(lines)


def dry_run(lines: list[Line]) -> int:
    """Run the question detector over the script text (no audio, no model)."""
    detector = QuestionDetector()
    wrong = 0
    for ln in lines:
        got = "question" if detector.check(ln.text) else "ignore"
        ok = got == ln.expect
        wrong += not ok
        print(f"{'ok ' if ok else 'BAD'}  {ln.expect:8}  {ln.text}  ({ln.why})")
    print(f"\n{len(lines) - wrong}/{len(lines)} lines detected as expected (text only).")
    return 1 if wrong else 0


# -- speech ------------------------------------------------------------------------------

PREFERRED_VOICES = ("Samantha", "Daniel", "Karen")


def default_voice() -> str | None:
    """An English voice: the system default may be any language (e.g. a Japanese Siri
    voice), which reads English with a heavy accent and is not a fair ASR test."""
    r = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    voices = {}
    for row in r.stdout.splitlines():
        m = re.match(r"(.+?)\s{2,}(\w\w_\w\w)\s", row)
        if m:
            voices[m.group(1).strip()] = m.group(2)
    for name in PREFERRED_VOICES:
        if name in voices:
            return name
    return next((n for n, lang in voices.items() if lang == "en_US"), None)


def speak(text: str, voice: str | None, rate: int) -> None:
    cmd = ["say", "-r", str(rate)] + (["-v", voice] if voice else []) + [text]
    subprocess.run(cmd, check=True)


def render(lines: list[Line], out: Path, voice: str | None, rate: int,
           gap_s: float = 2.0, sr: int = 16000) -> float:
    parts = [np.zeros(int(sr * 0.5), np.int16)]
    with tempfile.TemporaryDirectory() as tmp:
        for i, ln in enumerate(lines):
            wav = Path(tmp) / f"{i}.wav"
            cmd = ["say", "-r", str(rate), "-o", str(wav), "--file-format=WAVE",
                   f"--data-format=LEI16@{sr}"] + (["-v", voice] if voice else []) + [ln.text]
            subprocess.run(cmd, check=True)
            with wave.open(str(wav)) as w:
                parts.append(np.frombuffer(w.readframes(w.getnframes()), np.int16))
            gap = gap_s * (3 if ln.expect == "question" else 1)
            parts.append(np.zeros(int(sr * gap), np.int16))
    audio = np.concatenate(parts)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return audio.size / sr


# -- driving a running backend -----------------------------------------------------------

def app_url() -> str | None:
    """The private backend port of the running app window, if any."""
    r = subprocess.run(["pgrep", "-fl", "openinterview-ui"], capture_output=True, text=True)
    m = re.search(r"openinterview-ui --port (\d+)", r.stdout)
    return f"http://127.0.0.1:{m.group(1)}" if m else None


class Backend:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def call(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"null")

    def state(self) -> str:
        return self.call("GET", "/api/state")["status"]["state"]

    def wait_state(self, states: set[str], timeout: float) -> str:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            s = self.state()
            if s in states:
                return s
            time.sleep(0.3)
        return self.state()

    def cards(self) -> list[dict]:
        return self.call("GET", "/api/assistant")["cards"]


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=tokens(normalize_text(a)),
                                   b=tokens(normalize_text(b))).ratio()


def drive(url: str, lines: list[Line], voice: str | None, rate: int, answer_timeout: float,
          save: bool, model: str | None, language: str = "en") -> int:
    be = Backend(url)
    try:
        be.state()
    except OSError as exc:
        print(f"backend not reachable at {url}: {exc}", file=sys.stderr)
        return 2
    if be.state() not in ("idle", "error"):
        print("a session is already running: stop it first", file=sys.stderr)
        return 2
    was_enabled = be.call("GET", "/api/assistant")["enabled"]
    if not was_enabled:  # turned back off at the end
        st = be.call("POST", "/api/assistant", {"enabled": True})
        if not st["healthy"]:
            print(f"note: assistant unavailable ({st['health']}); only detection is tested",
                  file=sys.stderr)

    title = "Mock interview (test)" + ("" if language == "en" else f" [{language}]")
    body = {"source": "system", "title": title, "target": TARGET, "language": language,
            "save": save, "save_audio": save}
    if model:
        body["model"] = model
    be.call("POST", "/api/start", body)
    print("starting session (system audio)...", file=sys.stderr)
    if be.wait_state({"running", "error"}, 120) != "running":
        print("session did not start: " + be.call("GET", "/api/state")["status"]["error"],
              file=sys.stderr)
        return 2
    time.sleep(1.5)

    rows = []
    seen: set[int] = set()
    try:
        for ln in lines:
            print(f"  interviewer: {ln.text}", file=sys.stderr)
            spoke = time.monotonic()
            speak(ln.text, voice, rate)
            ended = time.monotonic()
            card, ready_s = None, None
            # Questions: wait for the card and its answer. Others: a short pause, then
            # check nothing was asked.
            wait = answer_timeout if ln.expect == "question" else 4.0
            end = ended + wait
            while time.monotonic() < end:
                new = [c for c in be.cards() if c["id"] not in seen]
                if new:
                    card = new[-1]
                    if card["status"] in ("ready", "error"):
                        ready_s = time.monotonic() - ended
                        break
                    if ln.expect != "question":
                        end = max(end, time.monotonic() + answer_timeout)
                time.sleep(0.25)
            for c in be.cards():
                seen.add(c["id"])
            rows.append((ln, card, ready_s, ended - spoke))
            time.sleep(1.0)
    finally:
        time.sleep(1.0)
        be.call("POST", "/api/stop")
        be.wait_state({"idle", "error"}, 60)
        if not was_enabled:
            be.call("POST", "/api/assistant", {"enabled": False})

    return report(rows)


def report(rows) -> int:
    print()
    print(f"{'':3} {'expected':8}  {'got':8}  {'answer':>6}  line / detected text")
    wrong = 0
    latencies = []
    for ln, card, ready_s, _ in rows:
        got = "question" if card else "ignore"
        ok = got == ln.expect
        wrong += not ok
        lat = ""
        if card and card["status"] == "ready" and ready_s is not None:
            lat = f"{ready_s:.1f}s"
            latencies.append(ready_s)
        elif card and card["status"] == "error":
            lat = "error"
        elif card:
            lat = "slow"
        print(f"{'ok ' if ok else 'BAD'} {ln.expect:8}  {got:8}  {lat:>6}  {ln.text}")
        if card:
            heard = card["question"]
            if similarity(heard, ln.text) < 0.8:
                print(f"{'':32}heard: {heard}")
            if card["status"] == "ready":
                print(f"{'':32}-> {card['answer']['short_answer']}")
            elif card["error"]:
                print(f"{'':32}!! {card['error']}")
    n = len(rows)
    print(f"\n{n - wrong}/{n} lines handled as expected.", end="")
    if latencies:
        latencies.sort()
        print(f"  Answer time after the question ended: median {latencies[len(latencies) // 2]:.1f}s,"
              f" max {latencies[-1]:.1f}s.", end="")
    print()
    return 1 if wrong else 0


def main(args) -> int:
    language = get_language(getattr(args, "language", None)).code
    lines = script(args.quick, language)
    if not args.voice and (args.render or args.app or args.url or args.play):
        args.voice = default_voice() if language == "en" else get_language(language).voice
    if args.render:
        secs = render(lines, Path(args.render), args.voice, args.rate)
        print(f"wrote {args.render} ({secs:.0f} s). Try: openinterview transcribe-file "
              f"{args.render} --utterances")
        return 0
    if args.app or args.url:
        url = args.url or app_url()
        if url is None:
            print("the app is not running: start it with `openinterview app`", file=sys.stderr)
            return 2
        return drive(url, lines, args.voice, args.rate, args.answer_timeout,
                     save=not args.no_save, model=args.model, language=language)
    if args.play:
        print("Start a session (system or both) in the app first. Speaking in 3 s...",
              file=sys.stderr)
        time.sleep(3)
        for ln in lines:
            print(f"  interviewer: {ln.text}", file=sys.stderr)
            speak(ln.text, args.voice, args.rate)
            time.sleep(args.gap * (3 if ln.expect == "question" else 1))
        return 0
    return dry_run(lines)
