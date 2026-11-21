from hbc_harness.eval import exact_match, f1, mcnemar, normalize, paired_bootstrap, support_recall
from hbc_harness.metrics import summarize


def test_normalize_strips_articles_and_punctuation():
    assert normalize("The Arthur's Magazine.") == "arthurs magazine"


def test_exact_match_is_normalized():
    assert exact_match("The Gamma.", "gamma")


def test_f1_partial_overlap():
    assert 0 < f1("alpha beta", "alpha gamma") < 1


def test_support_recall():
    assert support_recall(["a", "b", "c"], ["a", "c"]) == 1.0
    assert support_recall(["a"], ["a", "b"]) == 0.5


def test_bootstrap_identical_samples():
    result = paired_bootstrap([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    assert result["mean_diff"] == 0.0
    assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0


def test_mcnemar_counts_discordant_pairs():
    result = mcnemar([True, False, True, False], [True, True, False, False])
    assert result["b01"] == 1 and result["b10"] == 1


def test_percentiles_are_monotonic():
    summary = summarize([100, 200, 300, 400, 500])
    assert summary["p50_ms"] <= summary["p95_ms"] <= summary["p99_ms"]
    assert summary["goodput"] == 1.0
