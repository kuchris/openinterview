"""Decide which complete utterances are interview questions (technical or behavioural).

Input is *utterances*: finals already merged into complete sentences (asr/sentences.py).
Partials and fragments are never considered, so incomplete transcripts are not sent.

Heuristics (no model call):
  * at least ``min_words`` words
  * a question form: ends with "?", starts with an interrogative ("how would you ..."),
    or an interview prompt ("explain ...", "walk me through ...", "design ...")
  * optionally (``filter_small_talk``) an interview cue: a technical term, or a behavioural
    cue ("tell me about yourself", "why did you choose ...", "describe a time ...",
    strengths, experience, team, ...). Small talk such as "How are you doing today?" or
    "Can you hear me?" is ignored.
Duplicate suppression: a question too similar to a recent one (difflib ratio >= 0.85 on
normalised words) is not asked again.

Japanese, Mandarin and Cantonese use the same steps with their own cues (question marks
and particles 嗎/吗/呢/咩/か, interrogatives 為什麼/点解/なぜ, requests 請…/教えてください,
interview terms 進程/自我介紹/経験…). Words are characters there (text.tokens), and
mixed sentences ("Process同thread有咩分別？") also match the English cues.
"""

from __future__ import annotations

import difflib
import re
from collections import deque
from dataclasses import dataclass

from ..asr.filters import normalize_text
from ..text import has_cjk, tokens

INTERROGATIVE = re.compile(
    r"^(so,?\s+|and\s+|okay,?\s+|ok,?\s+|alright,?\s+|now,?\s+|next,?\s+)*"
    r"(what|why|how|when|where|which|who|whose|can you|could you|would you|will you|"
    r"do you|did you|have you|are you|is there|is it|are there|should|tell me|"
    r"explain|describe|walk me through|talk me through|compare|implement|write|design|"
    r"define|give me|show me|what's|how's|whats)\b"
)

TECH_TERMS = {
    "algorithm", "api", "array", "async", "asyncio", "big", "binary", "bug", "cache",
    "class", "complexity", "concurrency", "consistency", "container", "data", "database",
    "debug", "deploy", "design", "distributed", "docker", "dynamic", "error", "function",
    "gil", "graph", "hash", "heap", "http", "index", "inheritance", "interface", "kubernetes",
    "latency", "linked", "list", "lock", "memory", "microservice", "multiprocessing",
    "network", "object", "optimize", "performance", "pointer", "process", "python", "query",
    "queue", "recursion", "redis", "rest", "scale", "scaling", "schema", "security", "server",
    "service", "sorting", "sql", "stack", "string", "system", "test", "testing", "thread",
    "threads", "throughput", "transaction", "tree", "type", "variable", "code", "coding",
    "program", "programming", "software", "architecture", "framework", "library", "loop",
    "map", "dictionary", "tuple", "generator", "decorator", "closure", "exception", "mutex",
    "deadlock", "race", "load", "balancer", "sharding", "replication", "kafka", "grpc",
    "tcp", "udp", "dns", "url", "browser", "frontend", "backend", "javascript", "java",
    "golang", "rust", "c++", "react", "git", "ci", "cd", "pipeline", "cloud", "aws",
    "idempotent", "idempotency", "oauth", "authentication", "encryption", "o(n)", "runtime",
    "compile", "garbage", "collection", "virtual", "machine", "kernel", "os", "cpu",
    "production", "incident", "outage", "limiter", "request",
}


BEHAVIOURAL = re.compile(
    r"\b(tell (me|us) about|about yourself|introduce yourself|walk me through your|"
    r"why (did|do|would|should|are|were) (you|we)|why (this|our|us|here)|what made you|"
    r"what (is|are|was|were|would be) your|what's your|whats your|what have you|"
    r"what do you know about|where do you see yourself|describe (a|an|the) (time|situation|project)|"
    r"give (me|us) an example|a time when|how (do|did|would) you (handle|deal|manage|"
    r"prioriti[sz]e|work|approach|learn|stay|motivate|resolve|cope)|"
    r"do you have (any )?questions|how would your (colleagues|manager|team))\b"
)

BEHAVIOURAL_TERMS = {
    "strength", "strengths", "weakness", "weaknesses", "experience", "project", "projects",
    "team", "teammate", "conflict", "challenge", "challenging", "mistake", "failure",
    "achievement", "accomplishment", "goal", "goals", "career", "motivation", "motivates",
    "salary", "role", "position", "company", "background", "studied", "study", "major",
    "school", "university", "college", "degree", "manager", "leadership", "lead", "deadline",
    "pressure", "feedback", "hire", "job", "internship", "intern", "passion", "proud",
    "difficult", "colleague", "customer", "stakeholder", "decision", "skills", "hobbies",
}

SMALL_TALK = re.compile(
    r"\b(how are you|how's it going|how is it going|how have you been|can you hear me|"
    r"can you see (me|my screen)|is (that|this) (ok|okay|alright|fine)|does that make sense|"
    r"what are you doing|are you there|are you ready|shall we (start|begin)|"
    r"nice to meet you|how was your (day|weekend))\b"
)


# -- Japanese / Chinese / Cantonese ----------------------------------------------------------
# Recognition errors often hit exactly the cue word (幂等 -> 密等, べき等 -> べきなど), so a
# clear question of at least this many characters counts even without a cue (small talk
# is still filtered).
CJK_CLEAR_QUESTION_CHARS = 8
# Matched as substrings of the text (no word boundaries in these scripts).

CJK_QUESTION = re.compile(
    r"[？?]|(嗎|吗|呢|咩|未|冇|か|かな)[。！!]?$|"
    r"什麼|什么|甚麼|怎麼|怎么|怎樣|怎样|為什麼|为什么|為何|为何|如何|哪|幾|几|多少|"
    r"點解|点解|點樣|点样|邊個|边个|乜嘢|有冇|係咪|會唔會|有咩|咩嘢|做咩|係咩|"
    r"なぜ|なんで|どう|どの|どんな|どれ|いつ|誰|何|"
    r"請|请|說說|说说|講下|講吓|讲下|講講|讲讲|介紹下|介紹吓|介绍下|談談|谈谈|描述|解釋|解释|"
    r"比較|比较|分享下|分享吓|"
    r"教えて|お願いします|聞かせて|説明して|話して"
)
CJK_INTERROGATIVE = re.compile(
    r"什麼|什么|甚麼|怎麼|怎么|為什麼|为什么|如何|點解|点解|點樣|点样|有咩|"
    r"なぜ|どう|どのよう|どんな|何(を|が|で|か)"
)
CJK_TECH = re.compile(
    r"進程|进程|線程|线程|執行緒|行程|程序|程式|編程|编程|代碼|代码|算法|演算法|資料結構|数据结构|"
    r"數據庫|数据库|資料庫|系統|系统|架構|架构|設計|设计|限流|緩存|缓存|快取|併發|并发|鎖|锁|"
    r"分布式|分散式|伺服器|服务器|服務器|接口|介面|網絡|网络|網路|性能|效能|延遲|延迟|測試|测试|"
    r"冪等|幂等|事務|事务|交易|部署|容器|記憶體|内存|內存|支付|付款|請求|请求|"
    r"プロセス|スレッド|データベース|アルゴリズム|設計|システム|サーバー|キャッシュ|並行|並列|"
    r"排他|ロック|テスト|障害|本番|性能|遅延|冪等|べき等|決済|リクエスト|"
    r"分別|區別|区别|差別|差别|違い|比較|比较"  # "what's the difference between X and Y?"
)
CJK_BEHAVIOURAL = re.compile(
    r"自我介紹|自我介绍|介紹下你自己|介绍一下你自己|自己紹介|"
    r"公司|貴司|贵司|我哋|我們|我们|弊社|御社|当社|會社|会社|"
    r"經驗|经验|經歷|经历|経験|缺點|缺点|弱點|弱点|弱み|短所|優點|优点|長處|长处|強み|長所|"
    r"團隊|团队|同事|チーム|衝突|冲突|対立|意見|意见|挑戰|挑战|困難|困难|失敗|失败|"
    r"目標|目标|職業|职业|職位|职位|工作|加入|動機|动机|志望|理由|成就|成果|壓力|压力|"
    r"問題想問|问题想问|質問|有冇問題|管理|領導|领导|リーダー|"
    r"處理過|处理过|遇過|遇到過|遇到过|做過|做过|難搞|最難|最难"
)
CJK_SMALL_TALK = re.compile(
    r"聽得到|听得到|聽唔聽到|能聽到|能听到|聽到我|听到我|聞こえ|你好嗎|你好吗|最近點|最近好嗎|"
    r"準備好|准备好|可以開始|可以开始|見えますか|大丈夫ですか|画面"
)


@dataclass
class DetectorConfig:
    min_words: int = 4
    filter_small_talk: bool = True
    similarity: float = 0.85
    history: int = 30


class QuestionDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.cfg = config or DetectorConfig()
        self._recent: deque[list[str]] = deque(maxlen=self.cfg.history)

    @staticmethod
    def is_question_form(text: str) -> bool:
        t = text.strip().lower()
        if has_cjk(t) and CJK_QUESTION.search(t):
            return True
        return t.endswith("?") or bool(INTERROGATIVE.match(t)) or "?" in t

    @staticmethod
    def is_technical(words: list[str]) -> bool:
        return any(w in TECH_TERMS for w in words)

    @staticmethod
    def is_behavioural(norm: str, words: list[str]) -> bool:
        return bool(BEHAVIOURAL.search(norm)) or any(w in BEHAVIOURAL_TERMS for w in words)

    def is_interview_question(self, text: str) -> bool:
        norm = normalize_text(text)
        words = tokens(norm)
        if self.is_technical(words) or self.is_behavioural(norm, words):
            return True
        if not has_cjk(norm):
            return False
        if CJK_TECH.search(norm) or CJK_BEHAVIOURAL.search(norm):
            return True
        return (len(words) >= CJK_CLEAR_QUESTION_CHARS
                and bool(re.search(r"[？?]", text) or CJK_INTERROGATIVE.search(norm)))

    @staticmethod
    def is_small_talk(norm: str) -> bool:
        return bool(SMALL_TALK.search(norm) or (has_cjk(norm) and CJK_SMALL_TALK.search(norm)))

    def is_duplicate(self, words: list[str]) -> bool:
        for prev in self._recent:
            sm = difflib.SequenceMatcher(a=words, b=prev, autojunk=False)
            if sm.ratio() >= self.cfg.similarity:
                return True
            # A repeat with a short opener ("Sorry, ...", "不好意思，..."): the earlier
            # question is almost all inside the new one, and the lengths are close.
            shorter, longer = sorted((len(words), len(prev)))
            if shorter and shorter / longer >= 0.6:
                matched = sum(b.size for b in sm.get_matching_blocks())
                if matched / shorter >= 0.9:
                    return True
        return False

    def check(self, text: str) -> str | None:
        """Return a reason string if `text` should be asked, else None. Records it if so."""
        norm = normalize_text(text)
        words = tokens(norm)  # characters for Chinese / Japanese
        if len(words) < self.cfg.min_words or not self.is_question_form(text):
            return None
        if self.cfg.filter_small_talk:
            if self.is_small_talk(norm) and not (self.is_technical(words)
                                                 or (has_cjk(norm) and CJK_TECH.search(norm))):
                return None
            if not self.is_interview_question(text):
                return None
        if self.is_duplicate(words):
            return None
        self._recent.append(words)
        return "question"

    def remember(self, text: str) -> None:
        """Record a manually asked question so it is not detected again."""
        self._recent.append(tokens(normalize_text(text)))
