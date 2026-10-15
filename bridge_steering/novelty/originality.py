"""
Originality metric: fraction of unseen n-grams in generated text.

Implements the originality component of the novelty metric from:
  Padmakumar et al., "Measuring LLM Novelty As The Frontier Of Original
  And High-Quality Output", ICLR 2026.

Originality for a single generation is defined as the proportion of n-grams
in the text that do **not** appear in the reference corpus, for n = 4, 5, 6.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .corpus_index import NgramCorpusIndex, tokenize_text

DEFAULT_N_VALUES: list[int] = [4, 5, 6]


def compute_originality(
    text: str,
    index: NgramCorpusIndex,
    n_values: Sequence[int] = DEFAULT_N_VALUES,
) -> dict[int, float]:
    """Compute originality scores for a single piece of text.

    Parameters
    ----------
    text : str
        The generated text to evaluate.
    index : NgramCorpusIndex
        Pre-built n-gram index of the reference corpus.
    n_values : sequence of int
        Which n-gram sizes to evaluate (default [4, 5, 6]).

    Returns
    -------
    dict mapping n -> unseen_fraction (float in [0, 1], or NaN if text
    is too short to form any n-grams of that size).
    """
    tokens = tokenize_text(text)
    return {n: index.unseen_fraction(tokens, n) for n in n_values}


def compute_originality_batch(
    texts: Sequence[str],
    index: NgramCorpusIndex,
    n_values: Sequence[int] = DEFAULT_N_VALUES,
) -> pd.DataFrame:
    """Compute originality scores for a batch of texts.

    Parameters
    ----------
    texts : sequence of str
        Generated texts to evaluate.
    index : NgramCorpusIndex
        Pre-built n-gram index of the reference corpus.
    n_values : sequence of int
        Which n-gram sizes to evaluate.

    Returns
    -------
    DataFrame with columns ``text_idx``, ``num_tokens``,
    ``originality_4``, ``originality_5``, ``originality_6`` (one column per n).
    """
    rows: list[dict] = []
    for i, text in enumerate(texts):
        tokens = tokenize_text(text)
        row: dict = {"text_idx": i, "num_tokens": len(tokens)}
        for n in n_values:
            row[f"originality_{n}"] = index.unseen_fraction(tokens, n)
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_originality(df: pd.DataFrame, n_values: Sequence[int] = DEFAULT_N_VALUES) -> dict:
    """Aggregate originality scores from a batch DataFrame.

    Returns a dict with mean, median, std, min, max for each n-value,
    plus total text count.
    """
    summary: dict = {"count": len(df)}
    for n in n_values:
        col = f"originality_{n}"
        if col not in df.columns:
            continue
        series = df[col].dropna()
        summary[f"originality_{n}_mean"] = round(float(series.mean()), 4)
        summary[f"originality_{n}_median"] = round(float(series.median()), 4)
        summary[f"originality_{n}_std"] = round(float(series.std()), 4)
        summary[f"originality_{n}_min"] = round(float(series.min()), 4)
        summary[f"originality_{n}_max"] = round(float(series.max()), 4)
    return summary
