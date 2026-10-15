#!/usr/bin/env python3
"""
Build an n-gram corpus index from a HuggingFace dataset (default: FineWeb).

The index is used by the originality metric to measure the fraction of
n-grams in generated text that are unseen in a reference corpus.

Only needed for MacGyver originality (the macgyver_novelty stage). The paper
uses a 500M-token FineWeb sample with 4-, 5- and 6-grams (the defaults):

  python scripts/build_ngram_index.py --workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge_steering.novelty import NgramCorpusIndex


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build an n-gram corpus index from a HuggingFace dataset.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset identifier (default: HuggingFaceFW/fineweb)",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="sample-10BT",
        help="Dataset configuration / subset (default: sample-10BT)",
    )
    ap.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Column name containing document text (default: text)",
    )
    ap.add_argument(
        "--num-tokens",
        type=int,
        default=500_000_000,
        help="Number of word tokens to ingest (default: 500M, as in the paper)",
    )
    ap.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[4, 5, 6],
        help="N-gram sizes to index (default: 4 5 6)",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results/novelty/fineweb_500m.npz"),
        help="Output path for the .npz index file",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        metavar="N",
        help="Save a checkpoint every N tokens (e.g. 500000000 for 500M). "
             "If the run stops, use the .checkpoint file as a partial index.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from an existing checkpoint file. Requires --checkpoint-every "
             "to have been used in the original run. The script loads the checkpoint, "
             "skips already-processed documents, and continues building.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for hashing. 1 = legacy single-process. "
             "Setting >1 ignores --checkpoint-every / --resume.",
    )
    ap.add_argument(
        "--docs-per-chunk",
        type=int,
        default=256,
        help="Documents per worker task when --workers > 1 (default: 256). "
             "Larger = less IPC overhead, smaller = better load balancing.",
    )
    args = ap.parse_args()

    checkpoint_path = None
    if args.checkpoint_every > 0:
        # e.g. fineweb_10b.npz -> fineweb_10b.checkpoint.npz
        checkpoint_path = args.output.parent / (
            args.output.stem + ".checkpoint" + args.output.suffix
        )

    index = NgramCorpusIndex.build_from_hf(
        dataset_name=args.dataset,
        dataset_config=args.config,
        text_field=args.text_field,
        num_tokens=args.num_tokens,
        n_values=args.n_values,
        split=args.split,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        num_workers=args.workers,
        docs_per_chunk=args.docs_per_chunk,
    )
    index.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
