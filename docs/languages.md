# Languages

[← README](../README.md) · Japanese, Mandarin (Simplified / Traditional) and Cantonese interviews.

Pick **Interview language** in the New session sheet, or `--language` / `-l` on the
command line (`asr.language` in `config.yaml`):

| Code | Language | Default model | Apple locale | Whisper | Mock voice |
|---|---|---|---|---|---|
| `en` | English | (config) | en-US | en | Samantha |
| `ja` | Japanese | large-v3-turbo | ja-JP | ja | Kyoko |
| `zh` | Mandarin, Simplified | apple | zh-CN | zh | Tingting |
| `zh-TW` | Mandarin, Traditional | apple | zh-TW | zh | Meijia |
| `yue` | Cantonese (Hong Kong) | large-v3-turbo | zh-HK | yue | Sinji |

- **Speech recognition.** Both Apple's on-device recognizer (`apple`) and Whisper's
  multilingual models (`small`, `large-v3-turbo`) handle these languages. The `.en`
  models and Parakeet are English-only. Choosing a language switches to the model that
  did best in the mock interviews; you can pick the other one in ⚙︎ settings.
  - **Japanese → Whisper turbo.** Apple dropped whole questions after long pauses in a
    long session (even replayed offline, even on a fresh recognizer); Whisper, which
    transcribes each speech segment on its own, got all of them (~1.8 s final latency).
  - **Cantonese → Whisper turbo.** Apple (`zh-HK`) writes real colloquial Cantonese
    (嚟、嘢、我哋) but garbles English terms ("process" → "Cricess", "rate limiter" →
    "Rigely Mater"), which led to wrong answers. Whisper gets the English right
    ("Process和Thread有什么分别?") but writes Mandarin-style Simplified text. The LLM
    understands both; the technical terms matter more.
  - **Mandarin → Apple.** Both are accurate; Apple streams with lower latency and writes
    Traditional characters for `zh-TW`.
- **Apple's locale limit.** An app can reserve 5 Apple speech languages. When a new one
  is needed, another non-English reservation is released (its download stays on disk).
- **Text handling.** Chinese and Japanese have no spaces, so each character is a token
  (`openinterview/text.py`). Words in Latin script inside them stay whole
  ("我用Python寫API"). LocalAgreement, the repeat/duplicate filters, echo matching and
  sentence merging all use these tokens. 。？！ end sentences. Apple's results are split
  at pauses using word timings: it gives the silence to the next character (a "咁"
  lasting two seconds), and this is detected.
- **Clear questions count.** Recognition errors often hit the very cue word (幂等 → 密等,
  べき等 → べきなど), so a clear question (？ or a question word) of 8+ characters counts
  even without an interview term. Short small talk (你今日點過嚟？) is still ignored.
- **Question detection** has cues for each language: question marks and particles
  (嗎/吗/呢/咩/か), interrogatives (為什麼/点解/なぜ/有咩), requests (請…/講下/教えて
  ください), interview terms (進程/线程/自我介紹/経験/弱點/分別…), and small talk
  (聽唔聽到/聞こえますか). Mixed sentences ("Process同thread有咩分別？") also match the
  English cues. A repeat with a polite opener ("不好意思，…") counts as a duplicate.
- **Answers** come back in the interview language, written to be said aloud: polite
  です・ます Japanese, Mandarin in Simplified or Traditional characters, colloquial Hong
  Kong Cantonese (口語, not written Chinese). Technical terms stay in English where
  engineers say them that way. Your profile can stay in English.
- **Mock tests.** `openinterview -l yue mock --app` runs the same 14-line interview in
  Cantonese (also `ja`, `zh`, `zh-TW`).

Mixing Cantonese with English mid-sentence is the hard case (see above: Whisper keeps
the English, Apple keeps the Cantonese). The mock's voices are text-to-speech; real
speakers may do better or worse.
