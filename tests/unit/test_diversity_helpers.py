"""Pure-Python tests for diversity helpers (no embedding model required)."""

from __future__ import annotations

from bridge_steering.stages.aggregate import (
    diversity_columns,
    ngram_diversity,
    semantic_diversity_mean_pairwise_cosine,
)
from bridge_steering.config import DiversityConfig


def test_ngram_diversity_unigrams():
    # 6 tokens, 5 distinct → 5/6
    assert ngram_diversity(["a b c", "a d e"], 1) == 5 / 6


def test_ngram_diversity_bigrams_with_overlap():
    # "a b c" → (a,b),(b,c); "a b d" → (a,b),(b,d)
    # 4 bigrams total, 3 distinct → 3/4
    assert ngram_diversity(["a b c", "a b d"], 2) == 3 / 4


def test_ngram_diversity_empty():
    assert ngram_diversity([], 2) == 0.0
    assert ngram_diversity(["", "  "], 2) == 0.0


def test_ngram_diversity_skips_short_uses():
    # n=3 on a 2-token line yields no trigrams → diversity computed over
    # the rest only.
    assert ngram_diversity(["one two", "alpha beta gamma"], 3) == 1.0


def test_semantic_diversity_single_use_is_zero():
    assert semantic_diversity_mean_pairwise_cosine(["just one"], DiversityConfig()) == 0.0


def test_semantic_diversity_empty_is_zero():
    assert semantic_diversity_mean_pairwise_cosine([], DiversityConfig()) == 0.0


def test_diversity_columns_off_is_empty():
    assert diversity_columns(None) == []


def test_diversity_columns_on_lists_all_metrics():
    cols = diversity_columns(DiversityConfig())
    # Only the within-item semantic-diversity is in the summary CSV by
    # default to match the paper-style "Sem. div." column. The n-gram
    # distinct-ratio variants are still computed by the helper but not
    # exposed in the summary schema.
    assert "aut_semantic_diversity_mean_pairwise_cosine" in cols
