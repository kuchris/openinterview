# Benchmarks

[← README](../README.md) · Latency, accuracy (WER) and memory of the ASR backends.

`bench/benchmark.py` replays a fixed stream through the **real live pipeline** at real-time
speed: 12 LibriSpeech dev-clean utterances (130 s, 303 words) plus 12 interview-style
technical sentences (52 s, 171 words, synthesized with `say -v Samantha`), separated by 1 s
gaps. Each model runs in a fresh process.

```bash
uv run python bench/benchmark.py prepare
uv run python bench/benchmark.py run --models tiny.en base.en small.en large-v3-turbo parakeet-v2 apple
```

Measured on an **Apple M2 (16 GB), macOS 26.6.2**. Raw output is in `bench/results/`.
Vocabulary prompt is on unless noted.

| Model | WER LibriSpeech | WER technical | First partial p50 | Final after speech end p50 / p95 | ASR per final p50 | RTF p50 | Peak RSS | MLX peak |
|---|---|---|---|---|---|---|---|---|
| tiny.en | 10.6% | 4.1% | 584 ms | 716 / 745 ms | 103 ms | 0.026 | 490 MB | 321 MB |
| base.en | 7.6% | 4.1% | 616 ms | 754 / 795 ms | 138 ms | 0.035 | 613 MB | 576 MB |
| **small.en** (default) | 4.6% | **1.8%** | 759 ms | 971 / 1241 ms | 324 ms | 0.081 | 1022 MB | 1172 MB |
| large-v3-turbo | 6.9% | 3.5% | 2390 ms | 2409 / 3263 ms | 1103 ms | 0.266 | 1904 MB | 2041 MB |
| parakeet-v2 | 3.0% | 6.4% | 634 ms | 833 / 1017 ms | 216 ms | 0.045 | 1425 MB | 1976 MB |
| apple | **1.3%** | 11.1% | n/a¹ | 1033 / 1304 ms | n/a | n/a | 254 MB + 124 MB helper² | — |
| base.en, no prompt | 8.6% | 3.5% | 615 ms | 753 / 814 ms | 135 ms | 0.034 | 594 MB | 526 MB |
| small.en, no prompt | 4.3% | 3.5% | 747 ms | 961 / 1214 ms | 313 ms | 0.077 | 985 MB | 1172 MB |

¹ Apple's volatile results carry no per-word timing, so text lag cannot be measured from
them. Observed by hand: volatile text trails speech by about 1 s.
² Plus the system `localspeechrecognition` service (about 33 MB RSS during the run).
Its model memory is not attributable to this process.

What the numbers say:

- **Why small.en is the default.** It is best on technical vocabulary (3 errors in 171
  words) and second-best on read speech, with finals about 1 s after you stop speaking. On
  an M2 it uses about 30% of the GPU for partials.
- **base.en** is the low-resource choice: about half the memory, finals 0.2 s sooner, about
  twice the errors.
- **large-v3-turbo** is too slow on an M2 for 1 s partial updates. Partials were superseded
  under load, and it was *less* accurate than small.en on this set.
- **Parakeet** is excellent on read speech but has no vocabulary prompt, so it misses jargon.
- **Apple SpeechTranscriber** is the most accurate on general speech, nearly free in process
  memory, and leaves the GPU idle. But it mangles jargon ("Kubernates", "redisk cash",
  "item potent", "Asencio"). `AnalysisContext.contextualStrings` had **no measurable
  effect** with SpeechTranscriber in my tests. With `--dictation`, `DictationTranscriber`
  was worse overall.
- **Vocabulary prompt.** The technical-set numbers are within one or two words of each
  other (1 word = 0.6%), so the prompt's effect is small and noisy at this sample size.
- **Final latency.** About 600 ms of the final latency is the deliberate end-of-speech
  silence wait (`pipeline.min_silence_s`). VAD costs about 0.7 ms per 32 ms frame while the
  GPU is busy (about 2% of one core). No run dropped a final.

Caveats: the sample is small, and the technical set is synthetic speech from one voice.
Treat differences of a point or two as noise. Re-run on your machine and voice before
relying on them.

Synthetic test audio: macOS's default `say` voice is badly out-of-distribution for small
Whisper models (tiny.en output was nonsense). `say -v Samantha` transcribes verbatim.
