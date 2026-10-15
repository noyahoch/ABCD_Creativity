"""
Completion sources: control how B-condition completions are produced for each
contrastive pair.

This is one of two composable strategy axes in the steering framework:
  --b-source  (CompletionSource)  x  --method  (VectorExtractor)

Any source can be combined with any extractor.
"""

from __future__ import annotations

import argparse
import random
from abc import ABC, abstractmethod
from typing import Any

from bridge_steering.data_prep import TaskType


SOURCE_REGISTRY: dict[str, type[CompletionSource]] = {}


def register_source(cls: type[CompletionSource]) -> type[CompletionSource]:
    """Class decorator that adds a CompletionSource subclass to the registry."""
    SOURCE_REGISTRY[cls.name] = cls
    return cls


class CompletionSource(ABC):
    """
    Base class for B-completion sources.

    Subclasses control how many (and what) B-condition completion strings are
    produced for each contrastive pair.  The activation collection loop calls
    ``get_b_completions`` once per pair and collects activations for *every*
    returned completion.  With a single-completion source (``FixedCompletion``)
    this reproduces the original behaviour; with a multi-completion source the
    B-activation set grows proportionally.
    """

    name: str  # registry key, set on subclass

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        """Return one or more B-condition completion strings for *pair*.

        Args:
            pair: A contrastive-pair dict (output of ``build_contrastive_pair``).
                  Bridge runs attach non-empty ``"b_list"`` from JSON ``B_list``.
                  Also has ``"prompt_msgs"`` and ``"b_completion"`` (not used when
                  ``b_list`` is required by the source).
            model: The loaded HF CausalLM (available for generation sources).
            tokenizer: The loaded HF tokenizer.

        Returns:
            List of completion strings.  Length >= 1.
        """
        ...

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add source-specific CLI arguments.  Override in subclasses."""

    def setup(
        self,
        train_pairs: list[dict[str, Any]],
        task_type: TaskType,
    ) -> None:
        """Optional hook: run once after train split, before activation collection.

        Used e.g. to attach cross-object B completions that depend on the full
        train set. Default: no-op.
        """
        return


# --------------------------------------------------------------------------- #
# Concrete sources
# --------------------------------------------------------------------------- #


def _require_b_list(pair: dict[str, Any]) -> list[dict[str, str]]:
    """Return ``pair['b_list']`` only: non-empty list of mechanism dicts. No fallbacks."""
    b_list = pair.get("b_list")
    if not isinstance(b_list, list) or len(b_list) == 0:
        raise ValueError(
            "Pair must have non-empty 'b_list' (attach JSON B_list in run_bridge_steering). "
            "No fallback to top-level B or b_completion."
        )
    return b_list


@register_source
class FixedCompletion(CompletionSource):
    """Teacher-forced B from ``pair['b_list']`` (JSON ``B_list``); no other source.

    Behavior is controlled by ``fixed_b_list_mode`` (CLI ``--fixed-b-list-mode``):

    * ``all`` (default): one completion per list element — e.g. four bridges
      yield four activation passes per pair.
    * ``first``: only the first list entry — one B activation per pair.
    """

    name = "fixed"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--fixed-b-list-mode",
            type=str,
            choices=["all", "first"],
            default="all",
            help=(
                "For --b-source fixed: 'all' = one activation per B_list entry (default); "
                "'first' = only the first bridge (one B per pair)."
            ),
        )

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        b_list = _require_b_list(pair)
        mode = str(self.config.get("fixed_b_list_mode", "all"))
        style = str(self.config.get("b_format_style", "structured"))
        if mode == "first":
            b_list = b_list[:1]
        if style == "bullets_unlocks":
            return [f"- {b['unlocks']}" for b in b_list]
        if style == "bullets_unlocks_simple":
            return [f"- {b['unlocks_simple']}" for b in b_list]
        if style == "bullets_rules":
            return [f"- {b['rule']}" for b in b_list]
        if style == "bullets_rules_simple":
            return [f"- {b['rule_simple']}" for b in b_list]
        if style == "bullets_rules_hybrid":
            return [f"- {b['rule_hybrid']}" for b in b_list]
        if style == "bullets_type_rule":
            return [f"- {b['type']}: {b['rule']}" for b in b_list]
        if style == "bullets_rule_unlocks":
            def _strip(s: str) -> str:
                return s.rstrip(".!?,;: ").rstrip()
            return [f"- {_strip(b['rule'])}, unlocks {b['unlocks']}" for b in b_list]
        if style == "bullets_rule_unlocks_noword":
            # bullets_rule_unlocks minus the literal connector token "unlocks".
            # Same payloads, same order, same seam — only the WORD is gone.
            def _strip(s: str) -> str:
                return s.rstrip(".!?,;: ").rstrip()
            return [f"- {_strip(b['rule'])}, {b['unlocks']}" for b in b_list]
        if style == "bullets_rule_unlocks_simple":
            def _strip(s: str) -> str:
                return s.rstrip(".!?,;: ").rstrip()
            return [f"- {_strip(b['rule_simple'])}, unlocks {b['unlocks_simple']}" for b in b_list]
        if style == "bullets_rule_unlocks_hybrid":
            def _strip(s: str) -> str:
                return s.rstrip(".!?,;: ").rstrip()
            return [f"- {_strip(b['rule_hybrid'])}, unlocks {b['unlocks_hybrid']}" for b in b_list]
        return [format_single_b(b) for b in b_list]


# --------------------------------------------------------------------------- #
# Multi-B sources (for datasets with B_list)
# --------------------------------------------------------------------------- #


def format_single_b(
    b: dict[str, str],
    include_rule: bool = True,
    include_unlocks: bool = True,
) -> str:
    """Format one B-mechanism dict into a teacher-forced completion string.

    Mirrors ``format_B_completion_aut`` in ``data_prep.py`` but operates on
    a single B dict rather than an item with a ``"B"`` key.
    """
    lines = [f"Mechanism type: {b['type']}"]
    if include_rule:
        lines.append(f"Rule: {b['rule']}")
    if include_unlocks:
        lines.append(f"This unlocks: {b['unlocks']}")
    return "\n".join(lines) + "\n"


def format_multi_b_concat_block(
    b_list: list[dict[str, str]],
    *,
    include_rule: bool = True,
    include_unlocks: bool = True,
    style: str = "structured",
) -> str:
    """One teacher-forced string with every bridge.

    ``style``:
      * ``"structured"`` (default): ``Bridge N type: ... Bridge N rule: ... Bridge N unlocks: ...``
      * ``"bullets_unlocks"``: bullet list of each bridge's ``unlocks`` field
        (matches D's bullet register, removes structural confound).
      * ``"bullets_rules"``: bullet list of each bridge's ``rule`` field.
      * ``"bullets_rules_simple"``: bullet list of each bridge's ``rule_simple``
        field — the everyday-language paraphrase of ``rule``. Requires the
        dataset to carry ``rule_simple`` on every B_list entry (see
        ``data/abcd_aut_multiple_b_new_with_rule_simple.json``).
    """
    if style == "bullets_unlocks":
        return "\n".join(f"- {b['unlocks']}" for b in b_list)
    if style == "bullets_unlocks_simple":
        return "\n".join(f"- {b['unlocks_simple']}" for b in b_list)
    if style == "bullets_rules":
        return "\n".join(f"- {b['rule']}" for b in b_list)
    if style == "bullets_rules_simple":
        return "\n".join(f"- {b['rule_simple']}" for b in b_list)
    if style == "bullets_rules_hybrid":
        return "\n".join(f"- {b['rule_hybrid']}" for b in b_list)
    if style == "bullets_type_rule":
        return "\n".join(f"- {b['type']}: {b['rule']}" for b in b_list)
    if style == "bullets_rule_unlocks":
        # User-proposed hybrid: keep bullets_rules' quality boost, add the
        # unlocks payload to restore the novelty register. Strip trailing
        # punctuation from rule so the seam is `..., unlocks ...` not
        # `....., unlocks ...`.
        def _strip(s: str) -> str:
            return s.rstrip(".!?,;: ").rstrip()
        return "\n".join(f"- {_strip(b['rule'])}, unlocks {b['unlocks']}" for b in b_list)
    if style == "bullets_rule_unlocks_noword":
        # bullets_rule_unlocks with the literal connector token "unlocks"
        # DELETED and nothing else touched: same rule payload, same unlocks
        # payload, same order, same bullet, same comma seam. Isolates whether
        # the combined-content gain needs the affordance CONTENT or merely the
        # lexical cue "unlocks" — a strong affordance/creativity prime that
        # appears verbatim in every bridge of the shipped method.
        # NB: bullets_rules drops the unlocks CONTENT and answers a different
        # question; this drops only the WORD.
        def _strip(s: str) -> str:
            return s.rstrip(".!?,;: ").rstrip()
        return "\n".join(f"- {_strip(b['rule'])}, {b['unlocks']}" for b in b_list)
    if style == "bullets_rule_unlocks_simple":
        # Plain-English version of bullets_rule_unlocks: combines rule_simple
        # and unlocks_simple to control for engineering vocabulary across the
        # combined-content cell. Requires the dataset to carry both fields per
        # B_list entry (see data/abcd_aut_multiple_b_new_with_unlocks_simple.json).
        def _strip(s: str) -> str:
            return s.rstrip(".!?,;: ").rstrip()
        return "\n".join(f"- {_strip(b['rule_simple'])}, unlocks {b['unlocks_simple']}" for b in b_list)
    if style == "bullets_rule_unlocks_hybrid":
        # Middle-register version of bullets_rule_unlocks: combines rule_hybrid
        # and unlocks_hybrid to test the partially-softened vocabulary cell.
        # Requires the dataset to carry both fields per B_list entry (see
        # data/abcd_aut_multiple_b_new_with_unlocks_hybrid.json).
        def _strip(s: str) -> str:
            return s.rstrip(".!?,;: ").rstrip()
        return "\n".join(f"- {_strip(b['rule_hybrid'])}, unlocks {b['unlocks_hybrid']}" for b in b_list)
    # default structured behavior
    lines: list[str] = []
    for i, b in enumerate(b_list, 1):
        lines.append(f"Bridge {i} type: {b['type']}")
        if include_rule:
            lines.append(f"Bridge {i} rule: {b['rule']}")
        if include_unlocks:
            lines.append(f"Bridge {i} unlocks: {b['unlocks']}")
    return "\n".join(lines)


@register_source
class MultiBConcatenated(CompletionSource):
    """All B entries concatenated into a single multi-bridge completion.

    Returns exactly 1 completion per item where every bridge mechanism is
    laid out sequentially.  This lets the model "see" all mechanisms in one
    forward pass rather than separately.

    Requires ``pair['b_list']`` (non-empty); same as JSON ``B_list``.
    """

    name = "multi_b_concat"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--multi-b-omit-rule",
            action="store_true",
            help="Omit 'rule' lines from multi_b_concat B text (shorter, less justification-like).",
        )
        parser.add_argument(
            "--multi-b-omit-unlocks",
            action="store_true",
            help="Omit 'unlocks' lines from multi_b_concat B text.",
        )

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        b_list = _require_b_list(pair)
        include_rule = not bool(self.config.get("multi_b_omit_rule", False))
        include_unlocks = not bool(self.config.get("multi_b_omit_unlocks", False))
        style = str(self.config.get("b_format_style", "structured"))
        return [
            format_multi_b_concat_block(
                b_list,
                include_rule=include_rule,
                include_unlocks=include_unlocks,
                style=style,
            )
        ]


@register_source
class CCreativeBullets(CompletionSource):
    """CAA-style positive completion: gold creative uses (C field) as bullets.

    Returns the per-item ``C`` field — 8 gold creative uses — as a single
    bullet-list completion. Pair this with ``default_source="D_banal"`` and
    ``method="mean_diff"`` to produce ``μ_C − μ_D``: the symmetric,
    output-vs-output creativity direction (CAA-style on our data).

    This is the within-data CAA baseline for ``μ_B − μ_D`` (bridge steering).
    Same recipe shape (prompt fixed, multi-token teacher-forced completion,
    bullet register), so any difference between the two vectors is
    attributable to the *content* of the contrast.

    Requires ``pair['c_list']`` (non-empty list[str]); attached by
    :func:`bridge_steering.stages.steer_collect.build_pairs` when the source dataset
    has a per-item ``C`` field — see ``abcd_aut_v3_bridges_with_c.json``.
    """

    name = "c_creative_bullets"

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        c_list = pair.get("c_list")
        if not isinstance(c_list, list) or len(c_list) == 0:
            raise ValueError(
                "c_creative_bullets requires non-empty pair['c_list']. "
                "Use a dataset with a per-item C field (e.g. "
                "abcd_aut_v3_bridges_with_c.json)."
            )
        if not all(isinstance(s, str) and s.strip() for s in c_list):
            raise ValueError(
                "c_creative_bullets: every C entry must be a non-empty string."
            )
        return ["\n".join(f"- {use}" for use in c_list)]


@register_source
class CrossObjectTrainPermutedC(CompletionSource):
    """AUT ablation: C activations use creative-uses from *other* train items.

    Analog of ``cross_object_train_b`` for the CAA-style C-vs-D contrast.
    For each train pair (prompt still matches object *i*), teacher-forced C
    text is the full creative-uses bullet block from a donor object — same
    bullet layout as ``c_creative_bullets`` (8 bullets, one per ``C`` entry).
    Donors are distinct other train rows; never object *i*'s own list.

    Pair with ``default_source="D_banal"`` and ``method="mean_diff"`` to
    produce ``μ_C(other) − μ_D``. This is the C-side analogue of the
    ``cross_object_train_b`` permutation: if the C−D contrast's effect is
    object-grounded (the creative answers are *about this object*), the
    permuted version should perform worse than the matched
    ``c_creative_bullets``. If the C−D effect is mostly a "polished creative
    register" tone shift, the permuted version should retain most of it.

    * ``--cross-c-num-foreign 1`` (default): one foreign object's C block
      per row, assigned by a random **derangement**.
    * ``--cross-c-num-foreign K`` (K>1): concatenate K such blocks from K
      distinct other train objects (per-row sampling).
    """

    name = "cross_object_train_c"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--cross-c-perm-seed",
            type=int,
            default=12345,
            help=(
                "RNG seed for cross-object C permutation / foreign-item sampling "
                "(cross_object_train_c)"
            ),
        )
        parser.add_argument(
            "--cross-c-num-foreign",
            type=int,
            default=1,
            help=(
                "How many *other* train objects' C blocks to concatenate per row "
                "(1 = single permuted foreign C via derangement). AUT only."
            ),
        )

    def setup(
        self,
        train_pairs: list[dict[str, Any]],
        task_type: TaskType,
    ) -> None:
        if task_type != TaskType.AUT:
            raise ValueError(
                "cross_object_train_c is only defined for task_type=aut (AUT)."
            )
        n = len(train_pairs)
        if n < 2:
            raise ValueError(
                "cross_object_train_c needs at least 2 train pairs "
                "(need another object to borrow C from)."
            )

        c_pool: list[str] = []
        for p in train_pairs:
            c_list = p.get("c_list")
            if not isinstance(c_list, list) or len(c_list) == 0:
                raise ValueError(
                    f"cross_object_train_c: train pair {p.get('id', '?')!r} "
                    "is missing 'c_list'. Use a dataset with per-item 'C' field."
                )
            if not all(isinstance(s, str) and s.strip() for s in c_list):
                raise ValueError(
                    "cross_object_train_c: every C entry must be a non-empty string."
                )
            c_pool.append("\n".join(f"- {use}" for use in c_list))

        k = int(self.config.get("cross_c_num_foreign", 1))
        if k < 1:
            raise ValueError("--cross-c-num-foreign must be >= 1")
        seed = int(self.config.get("cross_c_perm_seed", 12345))
        rng = random.Random(seed)

        if k == 1:
            sigma = _random_derangement(n, rng)
            for i, pair in enumerate(train_pairs):
                pair["_cross_object_c_completion"] = c_pool[sigma[i]]
        else:
            k_eff = min(k, n - 1)
            for i, pair in enumerate(train_pairs):
                others = [j for j in range(n) if j != i]
                rng_i = random.Random(seed + 100_003 * i)
                pick = rng_i.sample(others, k_eff)
                chunks = [c_pool[j] for j in pick]
                pair["_cross_object_c_completion"] = "\n\n".join(chunks)

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        text = pair.get("_cross_object_c_completion")
        if not text:
            raise RuntimeError(
                "cross_object_train_c: setup() was not run or did not attach "
                "_cross_object_c_completion."
            )
        return [text]


def _random_derangement(n: int, rng: random.Random) -> list[int]:
    """Uniform random derangement of [0..n-1] (no fixed points)."""
    if n <= 1:
        raise ValueError("derangement requires at least 2 elements")
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


@register_source
class CrossObjectTrainPermutedB(CompletionSource):
    """AUT ablation: B activations use mechanisms from *other* train items only.

    For each train pair (prompt still matches object *i*), teacher-forced B text
    is the **full multi-bridge block** for a donor object — the same layout as
    ``multi_b_concat`` (all entries in that row's ``B_list``, not a single bridge).
    Donors are **distinct other** train rows; never object *i*'s own list.

    Optional ``--multi-b-omit-rule`` / ``--multi-b-omit-unlocks`` match
    ``multi_b_concat`` so the two b-sources stay comparable.

    * ``--cross-b-num-foreign 1`` (default): one foreign **object's** full B block
      per row, assigned by a random **derangement**.
    * ``--cross-b-num-foreign K`` (K>1): concatenate K such blocks from K distinct
      other train objects (per-row sampling from ``--cross-b-perm-seed``).
    * ``--cross-b-pick-mode bridges``: instead of full donor blocks, take ONE
      randomly chosen bridge from each of K distinct donor objects and format
      them as a single K-bullet block. With K=4 this is length-matched to the
      own multi-bridge block (~4 bullets) while every bullet comes from a
      different foreign object.
    """

    name = "cross_object_train_b"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--cross-b-perm-seed",
            type=int,
            default=12345,
            help=(
                "RNG seed for cross-object B permutation / foreign-item sampling "
                "(cross_object_train_b)"
            ),
        )
        parser.add_argument(
            "--cross-b-num-foreign",
            type=int,
            default=1,
            help=(
                "How many *other* train objects' B blocks to concatenate per row "
                "(1 = single permuted foreign B via derangement). AUT only."
            ),
        )
        parser.add_argument(
            "--cross-b-pick-mode",
            type=str,
            default="blocks",
            choices=("blocks", "bridges"),
            help=(
                "(cross_object_train_b) 'blocks' (default): each foreign donor "
                "contributes its FULL multi-bridge block. 'bridges': each of the "
                "K distinct donors contributes ONE bridge, yielding a single "
                "K-bullet block (length-matched at K=4; K=1 = single permuted "
                "bridge, the permuted counterpart of the single_bridge ablation)."
            ),
        )
        parser.add_argument(
            "--cross-b-bridge-pick",
            type=str,
            default="random",
            choices=("random", "first"),
            help=(
                "(cross_object_train_b, pick_mode=bridges) Which bridge to take "
                "from each donor: 'random' (per-row RNG) or 'first' (donor's "
                "B_list[0] — mirrors single_bridge's fixed_b_list_mode=first)."
            ),
        )
        parser.add_argument(
            "--multi-b-omit-rule",
            action="store_true",
            help=(
                "(cross_object_train_b) Omit rule lines from donor multi-bridge text "
                "(same as multi_b_concat)."
            ),
        )
        parser.add_argument(
            "--multi-b-omit-unlocks",
            action="store_true",
            help=(
                "(cross_object_train_b) Omit unlocks lines from donor multi-bridge text "
                "(same as multi_b_concat)."
            ),
        )

    def setup(
        self,
        train_pairs: list[dict[str, Any]],
        task_type: TaskType,
    ) -> None:
        if task_type != TaskType.AUT:
            raise ValueError(
                "cross_object_train_b is only defined for task_type=aut (AUT)."
            )
        n = len(train_pairs)
        if n < 2:
            raise ValueError(
                "cross_object_train_b needs at least 2 train pairs "
                "(need another object to borrow B from)."
            )

        include_rule = not bool(self.config.get("multi_b_omit_rule", False))
        include_unlocks = not bool(self.config.get("multi_b_omit_unlocks", False))
        style = str(self.config.get("b_format_style", "structured"))

        k = int(self.config.get("cross_b_num_foreign", 1))
        if k < 1:
            raise ValueError("--cross-b-num-foreign must be >= 1")
        seed = int(self.config.get("cross_b_perm_seed", 12345))
        rng = random.Random(seed)
        pick_mode = str(self.config.get("cross_b_pick_mode", "blocks"))
        if pick_mode not in ("blocks", "bridges"):
            raise ValueError(
                f"cross_b_pick_mode must be 'blocks' or 'bridges', got {pick_mode!r}"
            )

        if pick_mode == "bridges":
            bridge_pick = str(self.config.get("cross_b_bridge_pick", "random"))
            if bridge_pick not in ("random", "first"):
                raise ValueError(
                    f"cross_b_bridge_pick must be 'random' or 'first', got {bridge_pick!r}"
                )
            b_lists = [_require_b_list(p) for p in train_pairs]
            k_eff = min(k, n - 1)
            for i, pair in enumerate(train_pairs):
                rng_i = random.Random(seed + 100_003 * i)
                donors = rng_i.sample([j for j in range(n) if j != i], k_eff)
                picked = [b_lists[j][0] if bridge_pick == "first"
                          else rng_i.choice(b_lists[j]) for j in donors]
                pair["_cross_object_b_completion"] = format_multi_b_concat_block(
                    picked,
                    include_rule=include_rule,
                    include_unlocks=include_unlocks,
                    style=style,
                )
            return

        b_pool: list[str] = []
        for p in train_pairs:
            b_list = _require_b_list(p)
            b_pool.append(
                format_multi_b_concat_block(
                    b_list,
                    include_rule=include_rule,
                    include_unlocks=include_unlocks,
                    style=style,
                )
            )

        if k == 1:
            sigma = _random_derangement(n, rng)
            for i, pair in enumerate(train_pairs):
                pair["_cross_object_b_completion"] = b_pool[sigma[i]]
        else:
            k_eff = min(k, n - 1)
            for i, pair in enumerate(train_pairs):
                others = [j for j in range(n) if j != i]
                rng_i = random.Random(seed + 100_003 * i)
                pick = rng_i.sample(others, k_eff)
                chunks = [b_pool[j] for j in pick]
                pair["_cross_object_b_completion"] = "\n\n".join(chunks)

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        text = pair.get("_cross_object_b_completion")
        if not text:
            raise RuntimeError(
                "cross_object_train_b: setup() was not run or did not attach "
                "_cross_object_b_completion."
            )
        return [text]


@register_source
class CrossObjectTrainPermutedDB(CompletionSource):
    """Like ``cross_object_train_b`` but the D-side is **also** swapped to the
    same donor(s). For each train pair (prompt still matches object *i*), both
    teacher-forced B and D come from the chosen foreign donor row(s).

    Intent: when only B is swapped, the prompt-completion mismatch on the B
    side leaks into the mean-diff vector (the D side is clean, the B side is
    mismatched). Swapping D too makes the mismatch symmetric, so it cancels in
    ``mean(B) - mean(D)``. What remains is closer to the true
    creative-vs-banal direction.

    * ``--cross-b-num-foreign 1`` (default): one foreign object per row,
      assigned by a random **derangement**. Both D and B come from that one
      donor.
    * ``--cross-b-num-foreign K`` (K>1): concatenate K donors' B blocks and
      K donors' D completions (same donors for both sides, per-row sampling
      from ``--cross-b-perm-seed``).
    """

    name = "cross_object_train_db"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        # Reuse the same flag names as cross_object_train_b so configs stay
        # symmetric. Re-registering is safe; argparse dedupes by dest.
        CrossObjectTrainPermutedB.add_args(parser)

    def setup(
        self,
        train_pairs: list[dict[str, Any]],
        task_type: TaskType,
    ) -> None:
        if task_type != TaskType.AUT:
            raise ValueError(
                "cross_object_train_db is only defined for task_type=aut (AUT)."
            )
        n = len(train_pairs)
        if n < 2:
            raise ValueError(
                "cross_object_train_db needs at least 2 train pairs."
            )

        include_rule = not bool(self.config.get("multi_b_omit_rule", False))
        include_unlocks = not bool(self.config.get("multi_b_omit_unlocks", False))
        style = str(self.config.get("b_format_style", "structured"))

        b_pool: list[str] = []
        d_pool: list[str] = []
        for p in train_pairs:
            b_list = _require_b_list(p)
            b_pool.append(
                format_multi_b_concat_block(
                    b_list,
                    include_rule=include_rule,
                    include_unlocks=include_unlocks,
                    style=style,
                )
            )
            d_text = p.get("d_completion")
            if not isinstance(d_text, str) or not d_text:
                raise ValueError(
                    "cross_object_train_db: every train pair must carry a "
                    "non-empty 'd_completion' string."
                )
            d_pool.append(d_text)

        k = int(self.config.get("cross_b_num_foreign", 1))
        if k < 1:
            raise ValueError("--cross-b-num-foreign must be >= 1")
        seed = int(self.config.get("cross_b_perm_seed", 12345))
        rng = random.Random(seed)

        if k == 1:
            sigma = _random_derangement(n, rng)
            for i, pair in enumerate(train_pairs):
                j = sigma[i]
                pair["_cross_object_b_completion"] = b_pool[j]
                pair["d_completion"] = d_pool[j]
        else:
            k_eff = min(k, n - 1)
            for i, pair in enumerate(train_pairs):
                others = [j for j in range(n) if j != i]
                rng_i = random.Random(seed + 100_003 * i)
                pick = rng_i.sample(others, k_eff)
                pair["_cross_object_b_completion"] = "\n\n".join(b_pool[j] for j in pick)
                pair["d_completion"] = "\n".join(d_pool[j] for j in pick)

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        text = pair.get("_cross_object_b_completion")
        if not text:
            raise RuntimeError(
                "cross_object_train_db: setup() was not run or did not attach "
                "_cross_object_b_completion."
            )
        return [text]


@register_source
class CrossObjectTrainAllMismatch(CompletionSource):
    """All-mismatch ablation: B from donor *j*, D from donor *k*, i≠j≠k.

    ``cross_object_train_b`` swaps only B (asymmetric mismatch — leaks into
    mean(B)−mean(D)); ``cross_object_train_db`` swaps B and D to the SAME
    donor (symmetric mismatch — cancels). This source breaks the remaining
    confound: B and D come from two *different* foreign donors, so the
    prompt-completion mismatch is present on both sides but shares no donor
    identity. If steering survives this, neither own-object B content nor a
    shared-donor artifact explains the effect.

    Config keys (subset of ``cross_object_train_b``; K is fixed at 1):
      * ``cross_b_perm_seed``   — per-row donor RNG seed (default 12345)
      * ``cross_b_pick_mode``   — 'blocks' (donor j's full multi-bridge block)
                                  or 'bridges' (ONE bridge from donor j — the
                                  all-mismatch counterpart of single_bridge)
      * ``cross_b_bridge_pick`` — 'first' | 'random' (bridges mode only)
      * ``b_format_style``, ``multi_b_omit_rule``, ``multi_b_omit_unlocks``
    """

    name = "cross_object_train_allmis"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        CrossObjectTrainPermutedB.add_args(parser)

    def setup(
        self,
        train_pairs: list[dict[str, Any]],
        task_type: TaskType,
    ) -> None:
        if task_type != TaskType.AUT:
            raise ValueError(
                "cross_object_train_allmis is only defined for task_type=aut (AUT)."
            )
        n = len(train_pairs)
        if n < 3:
            raise ValueError(
                "cross_object_train_allmis needs at least 3 train pairs "
                "(i, B-donor j and D-donor k must be pairwise distinct)."
            )
        k_foreign = int(self.config.get("cross_b_num_foreign", 1))
        if k_foreign < 1:
            raise ValueError(
                "cross_object_train_allmis needs cross_b_num_foreign >= 1."
            )
        if k_foreign > 1 and n < 2 * k_foreign + 1:
            raise ValueError(
                f"cross_object_train_allmis with cross_b_num_foreign={k_foreign} "
                f"needs at least {2 * k_foreign + 1} train pairs (i plus "
                f"{k_foreign} B-donors plus {k_foreign} D-donors, all distinct); "
                f"got {n}."
            )

        include_rule = not bool(self.config.get("multi_b_omit_rule", False))
        include_unlocks = not bool(self.config.get("multi_b_omit_unlocks", False))
        style = str(self.config.get("b_format_style", "structured"))
        pick_mode = str(self.config.get("cross_b_pick_mode", "blocks"))
        if pick_mode not in ("blocks", "bridges"):
            raise ValueError(
                f"cross_b_pick_mode must be 'blocks' or 'bridges', got {pick_mode!r}"
            )
        bridge_pick = str(self.config.get("cross_b_bridge_pick", "random"))
        if bridge_pick not in ("random", "first"):
            raise ValueError(
                f"cross_b_bridge_pick must be 'random' or 'first', got {bridge_pick!r}"
            )
        seed = int(self.config.get("cross_b_perm_seed", 12345))

        b_lists = [_require_b_list(p) for p in train_pairs]
        d_pool: list[str] = []
        for p in train_pairs:
            d_text = p.get("d_completion")
            if not isinstance(d_text, str) or not d_text:
                raise ValueError(
                    "cross_object_train_allmis: every train pair must carry a "
                    "non-empty 'd_completion' string."
                )
            d_pool.append(d_text)

        for i, pair in enumerate(train_pairs):
            rng_i = random.Random(seed + 100_003 * i)
            if k_foreign == 1:
                # K=1 draw sequence kept byte-identical to the original
                # implementation so finished cells stay reproducible.
                b_donors = [rng_i.choice([x for x in range(n) if x != i])]
                d_donors = [rng_i.choice(
                    [x for x in range(n) if x != i and x != b_donors[0]])]
            else:
                # K>1 ("allmis mix"): 2K pairwise-distinct donors, none == i;
                # first K feed B, last K feed D — B/D donor sets are disjoint.
                pick2k = rng_i.sample([x for x in range(n) if x != i],
                                      2 * k_foreign)
                b_donors, d_donors = pick2k[:k_foreign], pick2k[k_foreign:]
            if pick_mode == "bridges":
                bridges = [
                    (b_lists[j][0] if bridge_pick == "first"
                     else rng_i.choice(b_lists[j]))
                    for j in b_donors
                ]
                pair["_cross_object_b_completion"] = format_multi_b_concat_block(
                    bridges,
                    include_rule=include_rule,
                    include_unlocks=include_unlocks,
                    style=style,
                )
            else:
                pair["_cross_object_b_completion"] = "\n\n".join(
                    format_multi_b_concat_block(
                        b_lists[j],
                        include_rule=include_rule,
                        include_unlocks=include_unlocks,
                        style=style,
                    )
                    for j in b_donors
                )
            pair["d_completion"] = "\n".join(d_pool[k] for k in d_donors)
            pair["_allmis_donors"] = (i, tuple(b_donors), tuple(d_donors))

    def get_b_completions(
        self, pair: dict[str, Any], model: Any, tokenizer: Any
    ) -> list[str]:
        text = pair.get("_cross_object_b_completion")
        if not text:
            raise RuntimeError(
                "cross_object_train_allmis: setup() was not run or did not "
                "attach _cross_object_b_completion."
            )
        return [text]
