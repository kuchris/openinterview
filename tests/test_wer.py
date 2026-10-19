import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parents[1] / "bench" / "benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_wer_identical_after_normalisation():
    assert bench.wer(["MISTER QUILTER IS THE APOSTLE"], ["Mister Quilter is the apostle."]) == 0


def test_wer_counts_sub_ins_del():
    # ref has 4 words; each case below makes 2 edits -> 2/4
    assert bench.wer(["a b c d"], ["a x c d e"][:1]) == 0.5  # sub + ins = 2/4
    assert bench.wer(["a b c d"], ["a c"]) == 0.5           # two deletions


def test_hyphens_joined_and_abbreviations_folded():
    assert bench.normalize("trade-off, look-ups") == ["tradeoff", "lookups"]
    assert bench.normalize("Mr. Quilter") == bench.normalize("MISTER QUILTER")
