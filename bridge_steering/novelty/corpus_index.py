"""
N-gram corpus index for originality measurement.

Builds a memory-efficient index of n-grams from a reference corpus
(e.g. HuggingFaceFW/fineweb) and supports fast membership queries.

Based on the methodology from:
  Padmakumar et al., "Measuring LLM Novelty As The Frontier Of Original
  And High-Quality Output", ICLR 2026.

Storage format: each n-gram is hashed to uint64 via blake2b and stored in a
sorted numpy array.  Lookups use binary search (O(log N) per query).
Disk format: np.savez_compressed with one array per n-value plus metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_WORD_SPLIT_RE = re.compile(r"\S+")


def tokenize_text(text: str) -> list[str]:
    """Lowercase whitespace tokenisation (word-level).

    Consistent with Merrill et al. (2024) and the WIMBD text pipeline.
    """
    return _WORD_SPLIT_RE.findall(text.lower())


def _hash_ngram(ngram: Sequence[str]) -> int:
    """Deterministic 64-bit hash of a word-level n-gram."""
    blob = " ".join(ngram).encode("utf-8")
    digest = hashlib.blake2b(blob, digest_size=8).digest()
    return struct.unpack("<Q", digest)[0]


def _extract_ngram_hashes(tokens: list[str], n: int) -> list[int]:
    """Return the list of uint64 hashes for all n-grams in *tokens*."""
    if len(tokens) < n:
        return []
    hashes: list[int] = []
    for i in range(len(tokens) - n + 1):
        hashes.append(_hash_ngram(tokens[i : i + n]))
    return hashes


# ---------------------------------------------------------------------------
# Parallel worker: tokenize + hash a chunk of documents in a child process.
# Defined at module level so multiprocessing can pickle it. Returns the
# per-chunk *unique* hashes per n — main process unions all chunk arrays once
# at the end, which is dramatically cheaper than maintaining one giant set.
# ---------------------------------------------------------------------------


def _hash_chunk_worker(payload: tuple) -> tuple[int, int, dict]:
    """``payload`` = (docs, n_values).

    Returns ``(total_tokens, doc_count, {n: np.ndarray[uint64]})``.
    The returned arrays are unique *within the chunk* but not across chunks.
    """
    docs, n_values = payload
    sets: dict[int, set] = {int(n): set() for n in n_values}
    total_tokens = 0
    doc_count = 0
    for text in docs:
        toks = tokenize_text(text)
        if not toks:
            continue
        total_tokens += len(toks)
        doc_count += 1
        for n in sets:
            if len(toks) < n:
                continue
            add = sets[n].add
            for i in range(len(toks) - n + 1):
                add(_hash_ngram(toks[i : i + n]))
    return (
        total_tokens,
        doc_count,
        {n: np.fromiter(s, dtype=np.uint64, count=len(s)) for n, s in sets.items()},
    )


class NgramCorpusIndex:
    """Hash-based n-gram index backed by sorted numpy uint64 arrays.

    Parameters
    ----------
    arrays : dict mapping n -> sorted np.ndarray of uint64 hashes
    metadata : dict with build-time info (corpus name, token count, etc.)
    """

    def __init__(
        self,
        arrays: dict[int, np.ndarray],
        metadata: dict | None = None,
    ) -> None:
        self._arrays = arrays
        self.metadata = metadata or {}

    @property
    def n_values(self) -> list[int]:
        return sorted(self._arrays)

    def num_unique(self, n: int) -> int:
        return len(self._arrays[n])

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
        n_values: Sequence[int] = (4, 5, 6),
        max_tokens: int | None = None,
        progress: bool = True,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 0,
        resume_from: NgramCorpusIndex | None = None,
    ) -> NgramCorpusIndex:
        """Build an index from an iterable of plain-text documents.

        Parameters
        ----------
        texts : iterable of str
            Each element is one document / passage.
        n_values : sequence of int
            Which n-gram sizes to index.
        max_tokens : int or None
            Stop after ingesting this many word tokens.  *None* = consume all.
        progress : bool
            Print progress every 1M tokens.
        checkpoint_path : path or None
            If set with checkpoint_every > 0, save state here every checkpoint_every tokens.
        checkpoint_every : int
            Token interval for checkpoints (0 = disabled).
        resume_from : NgramCorpusIndex or None
            If set, resume building from this checkpoint index. The hash arrays
            are loaded back into sets, token/doc counts are restored, and the
            corresponding number of documents in *texts* are skipped.
        """
        path = Path(checkpoint_path) if checkpoint_path else None

        docs_to_skip = 0
        if resume_from is not None:
            meta = resume_from.metadata
            total_tokens = meta.get("total_tokens", 0)
            doc_count = meta.get("doc_count", 0)
            docs_to_skip = doc_count
            hash_sets = {
                n: set(resume_from._arrays[n].tolist())
                for n in n_values
                if n in resume_from._arrays
            }
            for n in n_values:
                if n not in hash_sets:
                    hash_sets[n] = set()
            print(
                f"Resuming from checkpoint: {total_tokens:,} tokens, "
                f"{doc_count:,} docs — skipping those docs in the stream …"
            )
        else:
            hash_sets = {n: set() for n in n_values}
            total_tokens = 0
            doc_count = 0

        last_checkpoint_at = total_tokens
        t0 = time.monotonic()

        text_iter = iter(texts)
        if docs_to_skip > 0:
            t_skip = time.monotonic()
            for i, _ in enumerate(text_iter):
                if i + 1 >= docs_to_skip:
                    break
                if progress and (i + 1) % 100_000 == 0:
                    print(f"  Skipping … {i + 1:,} / {docs_to_skip:,} docs")
            print(
                f"  Skipped {docs_to_skip:,} docs in {time.monotonic() - t_skip:.1f}s"
            )

        for text in text_iter:
            tokens = tokenize_text(text)
            if not tokens:
                continue
            total_tokens += len(tokens)
            doc_count += 1
            for n in n_values:
                for h in _extract_ngram_hashes(tokens, n):
                    hash_sets[n].add(h)
            if max_tokens is not None and total_tokens >= max_tokens:
                break

            if (
                path is not None
                and checkpoint_every > 0
                and total_tokens - last_checkpoint_at >= checkpoint_every
            ):
                last_checkpoint_at = total_tokens
                print(
                    f"  Checkpointing at {total_tokens:,} tokens "
                    f"(building arrays + writing to disk may take several minutes)…"
                )
                t_ckpt = time.monotonic()
                elapsed = time.monotonic() - t0
                arrays = {
                    n: np.sort(np.fromiter(s, dtype=np.uint64, count=len(s)))
                    for n, s in hash_sets.items()
                }
                meta = {
                    "total_tokens": total_tokens,
                    "doc_count": doc_count,
                    "n_values": sorted(n_values),
                    "build_time_s": round(elapsed, 1),
                    "checkpoint": True,
                }
                tmp = cls(arrays, meta)
                tmp.save(path, compressed=False)
                print(f"  Checkpoint done in {time.monotonic() - t_ckpt:.1f}s → {path}")

            if progress and total_tokens % 1_000_000 < len(tokens):
                elapsed = time.monotonic() - t0
                print(
                    f"  [{elapsed:6.1f}s] {total_tokens:>12,} tokens  "
                    f"({doc_count:,} docs)  "
                    + "  ".join(
                        f"{n}-grams: {len(hash_sets[n]):,}"
                        for n in sorted(n_values)
                    )
                )

        elapsed = time.monotonic() - t0
        print(
            f"Build complete: {total_tokens:,} tokens from {doc_count:,} docs "
            f"in {elapsed:.1f}s"
        )
        for n in sorted(n_values):
            print(f"  {n}-grams: {len(hash_sets[n]):,} unique")

        arrays = {
            n: np.sort(np.fromiter(s, dtype=np.uint64, count=len(s)))
            for n, s in hash_sets.items()
        }
        metadata = {
            "total_tokens": total_tokens,
            "doc_count": doc_count,
            "n_values": sorted(n_values),
            "build_time_s": round(elapsed, 1),
        }
        return cls(arrays, metadata)

    @classmethod
    def build_from_texts_parallel(
        cls,
        texts: Iterable[str],
        n_values: Sequence[int] = (4, 5, 6),
        max_tokens: int | None = None,
        num_workers: int = 8,
        docs_per_chunk: int = 256,
        progress_every_tokens: int = 1_000_000,
    ) -> NgramCorpusIndex:
        """Multi-process variant of :meth:`build_from_texts`.

        Main process streams documents from ``texts`` and dispatches chunks
        of ``docs_per_chunk`` to a ``multiprocessing.Pool`` of size
        ``num_workers``. Workers tokenize + hash + deduplicate locally; the
        main process collects per-chunk hash arrays and unifies them once at
        the end with a single ``np.unique(np.concatenate(...))`` per n —
        far cheaper than maintaining one global set across all 1B tokens.
        """
        import multiprocessing as mp

        n_values = tuple(int(n) for n in n_values)
        per_n_arrays: dict[int, list[np.ndarray]] = {n: [] for n in n_values}
        total_tokens = 0
        doc_count = 0
        last_print_tokens = 0
        t0 = time.monotonic()

        def _chunked_payloads():
            buf: list[str] = []
            for text in texts:
                buf.append(text)
                if len(buf) >= docs_per_chunk:
                    yield (buf, n_values)
                    buf = []
            if buf:
                yield (buf, n_values)

        ctx = mp.get_context("spawn")  # avoid fork-then-exec issues with HF datasets
        with ctx.Pool(num_workers) as pool:
            for chunk_tokens, chunk_docs, chunk_arrays in pool.imap_unordered(
                _hash_chunk_worker, _chunked_payloads(), chunksize=1
            ):
                total_tokens += chunk_tokens
                doc_count += chunk_docs
                for n, arr in chunk_arrays.items():
                    per_n_arrays[n].append(arr)
                if total_tokens - last_print_tokens >= progress_every_tokens:
                    last_print_tokens = total_tokens
                    elapsed = time.monotonic() - t0
                    rate = total_tokens / elapsed if elapsed else 0
                    print(
                        f"  [{elapsed:6.1f}s] {total_tokens:>14,} tokens  "
                        f"({doc_count:,} docs)  {rate / 1000:.0f}k tok/s  "
                        f"workers={num_workers}"
                    )
                if max_tokens is not None and total_tokens >= max_tokens:
                    pool.terminate()
                    break

        elapsed = time.monotonic() - t0
        print(
            f"Worker phase done: {total_tokens:,} tokens from {doc_count:,} docs "
            f"in {elapsed:.1f}s — merging chunk hash arrays …"
        )
        arrays: dict[int, np.ndarray] = {}
        for n in n_values:
            chunks = per_n_arrays[n]
            if not chunks:
                arrays[n] = np.empty(0, dtype=np.uint64)
                continue
            t_merge = time.monotonic()
            merged = np.unique(np.concatenate(chunks))
            print(
                f"  {n}-grams: {len(merged):,} unique  "
                f"(merge {time.monotonic() - t_merge:.1f}s)"
            )
            arrays[n] = merged

        return cls(
            arrays,
            {
                "total_tokens": total_tokens,
                "doc_count": doc_count,
                "n_values": list(n_values),
                "build_time_s": round(time.monotonic() - t0, 1),
                "num_workers": num_workers,
                "docs_per_chunk": docs_per_chunk,
            },
        )

    @classmethod
    def build_from_hf(
        cls,
        dataset_name: str = "HuggingFaceFW/fineweb",
        dataset_config: str = "sample-10BT",
        text_field: str = "text",
        num_tokens: int = 10_000_000,
        n_values: Sequence[int] = (4, 5, 6),
        split: str = "train",
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 0,
        resume: bool = False,
        num_workers: int = 1,
        docs_per_chunk: int = 256,
    ) -> NgramCorpusIndex:
        """Build an index by streaming from a HuggingFace dataset.

        Parameters
        ----------
        dataset_name : str
            HF dataset identifier.
        dataset_config : str
            Dataset configuration / subset name.
        text_field : str
            Column in the dataset that holds the document text.
        num_tokens : int
            Target number of word tokens to ingest.
        n_values : sequence of int
            Which n-gram sizes to index.
        split : str
            Dataset split to stream from.
        checkpoint_path : path or None
            If set with checkpoint_every > 0, save state here every checkpoint_every tokens.
        checkpoint_every : int
            Token interval for checkpoints (0 = disabled).
        resume : bool
            If True and a checkpoint file exists at *checkpoint_path*, load it
            and continue building from where it left off (skipping already-
            processed documents in the stream).
        """
        from datasets import load_dataset

        resume_from = None
        if resume and checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint for resume: {checkpoint_path}")
            resume_from = cls.load(checkpoint_path)
        elif resume:
            print("Resume requested but no checkpoint found — starting from scratch.")

        print(
            f"Streaming {dataset_name} (config={dataset_config}, "
            f"split={split}) — target {num_tokens:,} tokens …"
        )
        if checkpoint_every > 0 and checkpoint_path:
            print(f"Checkpointing every {checkpoint_every:,} tokens → {checkpoint_path}")
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True,
        )

        def _iter_texts():
            for row in ds:
                yield row[text_field]

        if num_workers > 1:
            if checkpoint_every > 0 or resume:
                print(
                    "WARNING: --workers > 1 ignores --checkpoint-every / --resume "
                    "(parallel path keeps no global state). Build will run to "
                    "completion in one go."
                )
            index = cls.build_from_texts_parallel(
                _iter_texts(),
                n_values=n_values,
                max_tokens=num_tokens,
                num_workers=num_workers,
                docs_per_chunk=docs_per_chunk,
            )
        else:
            index = cls.build_from_texts(
                _iter_texts(),
                n_values=n_values,
                max_tokens=num_tokens,
                checkpoint_path=checkpoint_path,
                checkpoint_every=checkpoint_every,
                resume_from=resume_from,
            )
        index.metadata.update(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            text_field=text_field,
        )
        return index

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path, compressed: bool = True) -> None:
        """Save the index as a ``.npz`` file.

        Parameters
        ----------
        path : path
            Output .npz path (metadata written to same stem with .meta.json).
        compressed : bool
            If True (default), use np.savez_compressed (smaller file, slower).
            If False, use np.savez (faster, larger). Use False for checkpoints
            to avoid blocking the build for a long time.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict = {}
        for n, arr in self._arrays.items():
            save_kwargs[f"ngram_{n}"] = arr
        if compressed:
            np.savez_compressed(path, **save_kwargs)
        else:
            np.savez(path, **save_kwargs)
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(self.metadata, indent=2))
        print(f"Saved index to {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        print(f"Saved metadata to {meta_path}")

    @classmethod
    def load(cls, path: str | Path) -> NgramCorpusIndex:
        """Load an index from a ``.npz`` file (+ optional ``.meta.json``)."""
        path = Path(path)
        data = np.load(path)
        arrays: dict[int, np.ndarray] = {}
        for key in data.files:
            if key.startswith("ngram_"):
                n = int(key.split("_", 1)[1])
                arrays[n] = data[key]
        meta_path = path.with_suffix(".meta.json")
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        print(
            f"Loaded index from {path}: "
            + ", ".join(f"{n}-grams={len(a):,}" for n, a in sorted(arrays.items()))
        )
        return cls(arrays, metadata)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def contains(self, ngram: Sequence[str], n: int) -> bool:
        """Check whether a single n-gram exists in the index."""
        arr = self._arrays[n]
        h = np.uint64(_hash_ngram(ngram))
        idx = np.searchsorted(arr, h)
        return idx < len(arr) and arr[idx] == h

    def contains_hashes(self, hashes: np.ndarray, n: int) -> np.ndarray:
        """Vectorised membership test for an array of uint64 hashes.

        Returns a boolean array of the same length.
        """
        arr = self._arrays[n]
        idxs = np.searchsorted(arr, hashes)
        idxs = np.clip(idxs, 0, len(arr) - 1)
        return arr[idxs] == hashes

    def unseen_fraction(self, tokens: list[str], n: int) -> float:
        """Fraction of n-grams in *tokens* that are absent from the index.

        Returns a float in [0, 1].  Returns NaN if the text has fewer
        than *n* tokens (i.e. no n-grams can be formed).
        """
        if len(tokens) < n:
            return float("nan")
        hashes = np.array(_extract_ngram_hashes(tokens, n), dtype=np.uint64)
        found = self.contains_hashes(hashes, n)
        return 1.0 - float(found.sum()) / len(hashes)
