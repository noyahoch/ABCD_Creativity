"""
N-gram originality and novelty evaluation.

Implements the originality metric from:
  Padmakumar et al., "Measuring LLM Novelty As The Frontier Of Original
  And High-Quality Output", ICLR 2026.

Originality = fraction of n-grams in generated text unseen in a reference
corpus, for n = 4, 5, 6.
"""

from .corpus_index import NgramCorpusIndex, tokenize_text
from .originality import (
    compute_originality,
    compute_originality_batch,
    summarise_originality,
)

__all__ = [
    "NgramCorpusIndex",
    "compute_originality",
    "compute_originality_batch",
    "summarise_originality",
    "tokenize_text",
]
