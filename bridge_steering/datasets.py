"""ABCD (AUT) and MacGyver dataset loaders + deterministic eval/train split.

Dataset loading and the eval/train split. Behaviour for :func:`compute_eval_indices`
matches the legacy code exactly so that old steering checkpoints can
still be replayed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from .repro import sha256_file


# ---------------------------------------------------------------------------
# Deterministic split (shared by ABCD steering and ABCD eval)
# ---------------------------------------------------------------------------


def compute_eval_indices(n_items: int, eval_seed: int, eval_holdout: int) -> list[int]:
    """random.Random(eval_seed).sample(range(n_items), eval_holdout), sorted."""
    if eval_holdout < 0:
        raise ValueError(f"eval_holdout must be >= 0, got {eval_holdout}")
    if n_items < 0:
        raise ValueError(f"n_items must be >= 0, got {n_items}")
    rng = random.Random(eval_seed)
    n = min(eval_holdout, n_items)
    return sorted(rng.sample(range(n_items), n)) if n_items > 0 else []


def split_items_by_eval_indices(
    data: list[dict],
    eval_indices: Iterable[int],
) -> tuple[list[dict], list[dict], set[int]]:
    ev_set = set(int(i) for i in eval_indices)
    eval_items = [data[i] for i in sorted(ev_set)]
    train_items = [data[i] for i in range(len(data)) if i not in ev_set]
    return eval_items, train_items, ev_set


# ---------------------------------------------------------------------------
# ABCD (AUT)
# ---------------------------------------------------------------------------


@dataclass
class ABCDDataset:
    """Lightweight wrapper around the abcd_*.json file.

    Carries the file SHA so the steering checkpoint and downstream stages
    can detect data drift.
    """

    path: Path
    items: list[dict]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "ABCDDataset":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            raw: Any = json.load(f)
        # Accept both shapes: top-level list (v0) and {_provenance, items} (v1+).
        if isinstance(raw, dict) and isinstance(raw.get("items"), list):
            items = raw["items"]
        else:
            items = raw if isinstance(raw, list) else [raw]
        # Stamp deterministic source_id on each row so per-stage CSVs share
        # a stable join key even if the input ever lacks an `id` field.
        for i, item in enumerate(items):
            if "id" not in item or item["id"] in (None, ""):
                item["id"] = f"abcd_{i:04d}"
            item.setdefault("source_id", item["id"])
            item.setdefault("task_type", "aut")
        return cls(path=p, items=items, sha256=sha256_file(p))

    def split(self, eval_seed: int, eval_holdout: int) -> "ABCDSplit":
        idx = compute_eval_indices(len(self.items), eval_seed, eval_holdout)
        eval_items, train_items, eval_set = split_items_by_eval_indices(self.items, idx)
        return ABCDSplit(
            dataset=self,
            eval_indices=idx,
            train_items=train_items,
            eval_items=eval_items,
            eval_seed=eval_seed,
            eval_holdout=eval_holdout,
        )


@dataclass
class ABCDSplit:
    dataset: ABCDDataset
    eval_indices: list[int]
    train_items: list[dict]
    eval_items: list[dict]
    eval_seed: int
    eval_holdout: int

    @property
    def train_indices(self) -> list[int]:
        ev = set(self.eval_indices)
        return [i for i in range(len(self.dataset.items)) if i not in ev]


# ---------------------------------------------------------------------------
# MacGyver
# ---------------------------------------------------------------------------


class MacGyverSubset(Enum):
    ALL = "all"
    SOLVABLE = "solvable"
    SOLVABLE_UNCONVENTIONAL = "solvable_unconventional"
    SOLVABLE_CONVENTIONAL = "solvable_conventional"
    UNSOLVABLE = "unsolvable"


@dataclass
class MacGyverDataset:
    """MacGyver problem_solution_pair.xlsx loader. No train/eval split."""

    path: Path
    items: list[dict]
    subset: MacGyverSubset
    sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        subset: str | MacGyverSubset = "solvable_unconventional",
        max_items: Optional[int] = None,
    ) -> "MacGyverDataset":
        import pandas as pd  # local import — pandas is heavy

        p = Path(path)
        sub = MacGyverSubset(subset) if isinstance(subset, str) else subset
        df = pd.read_excel(p)
        df = _filter_macgyver(df, sub)

        items: list[dict] = []
        for _, row in df.iterrows():
            item_id = str(int(row["ID"]))
            items.append({
                "id": item_id,
                "source_id": item_id,
                "task_type": "macgyver",
                "domain": "macgyver",
                "problem_id": item_id,
                "problem": str(row["Problem"]),
                "problem_text": str(row["Problem"]),
                "solvable": row.get("Solvable?") == "Yes",
                "unconventional": row.get("Unconventional?") == "unconventional",
                "solution": str(row.get("Solution", "")),
                "label": str(row.get("Label", "")),
            })
        if max_items is not None:
            items = items[:max_items]
        return cls(path=p, items=items, subset=sub, sha256=sha256_file(p))


def _filter_macgyver(df: "Any", subset: MacGyverSubset) -> "Any":
    if subset == MacGyverSubset.ALL:
        return df
    if subset == MacGyverSubset.SOLVABLE:
        return df[df["Solvable?"] == "Yes"]
    if subset == MacGyverSubset.SOLVABLE_UNCONVENTIONAL:
        return df[(df["Solvable?"] == "Yes") & (df["Unconventional?"] == "unconventional")]
    if subset == MacGyverSubset.SOLVABLE_CONVENTIONAL:
        return df[(df["Solvable?"] == "Yes") & (df["Unconventional?"] == "conventional")]
    if subset == MacGyverSubset.UNSOLVABLE:
        return df[df["Solvable?"] == "No"]
    raise ValueError(f"Unknown MacGyver subset: {subset!r}")
