"""Transcript deduplication: LocalAgreement, repeat collapsing, hallucination and duplicate filters."""

from openinterview.asr.agreement import LocalAgreement
from openinterview.asr.engine import ASRResult, Segment
from openinterview.asr.filters import (DuplicateSuppressor, clean_result, collapse_repeats,
                                       normalize_text)
from openinterview.asr.sentences import SentenceAssembler


# -- LocalAgreement ---------------------------------------------------------------------

def test_first_hypothesis_commits_nothing():
    la = LocalAgreement()
    s = la.update("how would you")
    assert s.stable == "" and s.unstable == "how would you"


def test_commits_prefix_two_hypotheses_agree_on():
    la = LocalAgreement()
    la.update("how would you")
    s = la.update("how would you design a")
    assert s.stable == "how would you"
    assert s.unstable == "design a"


def test_revised_tail_is_not_committed():
    la = LocalAgreement()
    la.update("how would you design arrays")
    s = la.update("how would you design a rate limiter")
    assert s.stable == "how would you design"
    assert s.unstable == "a rate limiter"


def test_committed_words_are_never_repeated_or_retracted():
    la = LocalAgreement()
    hyps = [
        "explain the",
        "explain the difference",
        "Explain the difference between",
        "explain the difference between a process",
        "explain the difference between a process and a thread",
        "explain the difference between a process and a thread",
    ]
    stable_history = [la.update(h).stable for h in hyps]
    for prev, cur in zip(stable_history, stable_history[1:]):
        assert cur.startswith(prev)
    words = stable_history[-1].split()
    assert words.count("difference") == 1
    assert stable_history[-1].lower() == "explain the difference between a process and a thread"


def test_agreement_ignores_case_and_punctuation():
    la = LocalAgreement()
    la.update("So, the GIL")
    s = la.update("so the gil, means")
    # Committed words take the newer hypothesis's surface form (it has more context).
    assert normalize_text(s.stable) == "so the gil"
    assert s.unstable == "means"


def test_hypothesis_rewriting_committed_words_keeps_commit():
    la = LocalAgreement()
    la.update("the global interpreter")
    la.update("the global interpreter lock")
    s = la.update("a global interpreter lock means")
    assert s.stable == "the global interpreter"  # not retracted
    assert s.unstable == "lock means"


# -- repeat collapsing ------------------------------------------------------------------

def test_collapse_single_word_loop():
    assert collapse_repeats("The The The The The baby") == "The baby"


def test_collapse_phrase_loop():
    text = "see you in the next video. see you in the next video. see you in the next video."
    assert collapse_repeats(text) == "see you in the next video."


def test_legit_double_word_kept():
    assert collapse_repeats("it had had an effect") == "it had had an effect"
    assert collapse_repeats("that that is") == "that that is"


# -- hallucination filter ---------------------------------------------------------------

def seg(text, logprob=-0.2, no_speech=0.05, cr=1.2):
    return Segment(text, 0.0, 1.0, avg_logprob=logprob, no_speech_prob=no_speech,
                   compression_ratio=cr)


def test_blocklisted_low_confidence_dropped():
    r = ASRResult("Thank you for watching!", [seg("Thank you for watching!", logprob=-0.9)])
    assert clean_result(r) == ""


def test_blocklisted_on_mostly_nonspeech_dropped():
    r = ASRResult("Thank you.", [seg("Thank you.", logprob=-0.1)])
    assert clean_result(r, speech_ratio=0.3) == ""


def test_confident_spoken_thank_you_kept():
    r = ASRResult("Thank you.", [seg("Thank you.", logprob=-0.1)])
    assert clean_result(r, speech_ratio=0.9) == "Thank you."


def test_high_compression_segment_dropped():
    r = ASRResult("", [seg("ok so", cr=1.1), seg("la la la la la la la la", cr=3.5)])
    assert clean_result(r) == "ok so"


# -- duplicate finals -------------------------------------------------------------------

def test_duplicate_final_within_window_suppressed():
    d = DuplicateSuppressor(window_s=10)
    assert not d.is_duplicate("system", "Can you hear me?", 1.0)
    assert d.is_duplicate("system", "can you hear me", 3.0)
    assert not d.is_duplicate("mic", "can you hear me", 3.5)       # other source
    assert not d.is_duplicate("system", "can you hear me", 30.0)   # outside window


def test_short_repeated_acknowledgement_kept():
    d = DuplicateSuppressor()
    assert not d.is_duplicate("system", "Okay.", 1.0)
    assert not d.is_duplicate("system", "Okay.", 2.0)


# -- sentence assembly (incomplete sentences) -------------------------------------------

def test_incomplete_sentence_merged_with_next_final():
    a = SentenceAssembler(merge_gap_s=1.2)
    assert a.add_final("system", "So how would you", 0.0, 1.5, now=10.0) == []
    (u,) = a.add_final("system", "design a rate limiter?", 2.2, 4.0, now=11.0)
    assert u.text == "So how would you design a rate limiter?"
    assert (u.start, u.end) == (0.0, 4.0)


def test_incomplete_sentence_released_after_quiet_gap():
    a = SentenceAssembler(merge_gap_s=1.2)
    a.add_final("system", "tell me about yourself", 0.0, 2.0, now=10.0)
    assert a.tick(10.5, {"system": False}) == []
    assert a.tick(12.0, {"system": True}) == []   # speaker resumed: keep waiting
    (u,) = a.tick(12.0, {"system": False})
    assert u.text == "tell me about yourself"
