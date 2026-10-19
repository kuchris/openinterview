"""Model name -> engine / VAD construction."""

from __future__ import annotations

from .engine import ASREngine
from .parakeet_engine import MODEL_REPOS as PARAKEET_MODELS
from .whisper_engine import MODEL_REPOS as WHISPER_MODELS

APPLE_MODELS = {"apple": "SpeechTranscriber (system, macOS 26+)"}
MODELS = {**WHISPER_MODELS, **PARAKEET_MODELS, **APPLE_MODELS}


def make_engine(model: str, language: str = "en") -> ASREngine:
    if model in PARAKEET_MODELS:
        from .parakeet_engine import ParakeetEngine

        return ParakeetEngine(model)
    from .whisper_engine import WhisperEngine

    return WhisperEngine(model, language)


def make_vad(kind: str = "silero"):
    from .vad import EnergyVAD, SileroVAD

    if kind == "energy":
        return EnergyVAD()
    return SileroVAD()


def build_transcriber(cfg, sources, on_event):
    """Wire a transcriber from the app Config (used by CLI, benchmark, web server)."""
    if cfg.asr.model == "apple":
        from .apple_stream import AppleStreamingTranscriber

        return AppleStreamingTranscriber(sources, on_event, locale="en-US",
                                         vocabulary=cfg.asr.vocabulary_prompt,
                                         merge_gap_s=cfg.pipeline.merge_gap_s,
                                         echo_suppression=cfg.pipeline.echo_suppression)
    from .filters import FilterConfig
    from .segmenter import SegmenterConfig
    from .transcription import PipelineConfig, StreamingTranscriber

    p = cfg.pipeline
    seg = SegmenterConfig(
        sample_rate=cfg.audio.sample_rate,
        threshold=p.vad_threshold,
        neg_threshold=max(0.05, p.vad_threshold - 0.15),
        min_silence_s=p.min_silence_s,
        max_segment_s=p.max_segment_s,
        partial_interval_s=p.partial_interval_s,
    )
    pipe_cfg = PipelineConfig(
        segmenter=seg,
        filters=FilterConfig(),
        max_finals=p.max_finals,
        degrade_at=p.degrade_at,
        vocabulary_prompt=cfg.asr.vocabulary_prompt,
        prompt_with_context=cfg.asr.prompt_with_context,
        merge_gap_s=p.merge_gap_s,
        echo_suppression=p.echo_suppression,
    )
    return StreamingTranscriber(
        engine_factory=lambda: make_engine(cfg.asr.model, cfg.asr.language),
        sources=sources,
        on_event=on_event,
        vad_factory=lambda: make_vad(p.vad),
        config=pipe_cfg,
    )
