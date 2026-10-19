"""Interview languages: speech recognition settings, answer style, test voice.

    en     English
    ja     Japanese
    zh     Mandarin, Simplified characters
    zh-TW  Mandarin, Traditional characters
    yue    Cantonese (Hong Kong, Traditional characters)

ASR: Apple's on-device recognizer (model "apple") and Whisper's multilingual models
("small", "large-v3-turbo") support them; the ".en" models and Parakeet are English-only,
so for other languages the model is switched (resolve_model) to the language's preferred
one. Measured on the mock interviews:
  * Japanese: Apple dropped whole questions after long pauses; Whisper turbo got all.
  * Cantonese: Apple writes colloquial Cantonese but garbles English terms ("process" ->
    "Cricess"); Whisper turbo gets the English right but writes Mandarin-style text.
    The English terms matter more for answering, so Whisper is preferred.
  * Mandarin: both accurate; Apple streams with lower latency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    apple_locale: str
    whisper: str
    voice: str          # macOS `say` voice for mock interviews
    answer_style: str   # how the coach writes answers to say out loud
    preferred: tuple = ("apple", "large-v3-turbo")  # speech models, best first


LANGUAGES = {
    "en": Language("en", "English", "en-US", "en", "Samantha", "plain spoken English"),
    "ja": Language("ja", "Japanese", "ja-JP", "ja", "Kyoko",
                   "natural spoken Japanese in polite interview register (です・ます), "
                   "keeping technical terms as Japanese engineers say them",
                   ("large-v3-turbo", "apple")),
    "zh": Language("zh", "Mandarin (Simplified)", "zh-CN", "zh", "Tingting",
                   "natural spoken Mandarin in Simplified Chinese characters; technical "
                   "terms may stay in English where engineers say them that way"),
    "zh-TW": Language("zh-TW", "Mandarin (Traditional)", "zh-TW", "zh", "Meijia",
                      "natural spoken Mandarin in Traditional Chinese characters (Taiwan "
                      "usage); technical terms may stay in English where engineers say "
                      "them that way"),
    "yue": Language("yue", "Cantonese", "zh-HK", "yue", "Sinji",
                    "natural spoken Hong Kong Cantonese written in Traditional characters "
                    "(口語, e.g. 我哋、係、唔係、嘅), not formal written Chinese; technical "
                    "terms stay in English as Hong Kong engineers say them",
                    ("large-v3-turbo", "apple")),
}

# No Chinese / Japanese: the .en Whisper models, and Parakeet (v3 is European languages).
ENGLISH_ONLY_MODELS = {"tiny.en", "base.en", "small.en", "parakeet-v2", "parakeet-v3"}


def get(code: str | None) -> Language:
    if not code:
        return LANGUAGES["en"]
    if code not in LANGUAGES:
        raise ValueError(f"unknown language {code!r}; choose from {', '.join(LANGUAGES)}")
    return LANGUAGES[code]


def supports(model: str, code: str) -> bool:
    if code == "en":
        return True
    if model in ENGLISH_ONLY_MODELS:
        return False
    if code == "yue" and model == "small":  # Cantonese needs Whisper large-v3 (or Apple)
        return False
    return True


def preferred_models(code: str, apple_available: bool = True) -> list[str]:
    return [m for m in get(code).preferred if m != "apple" or apple_available]


def resolve_model(model: str, code: str, apple_available: bool = True) -> str:
    """The model to use for this language: `model` if it can, else the preferred one."""
    if supports(model, code):
        return model
    return (preferred_models(code, apple_available) or ["large-v3-turbo"])[0]
