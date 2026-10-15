"""Single source of truth for every prompt the pipeline sends to a model.

Importing rule:
    from bridge_steering.prompts.aut import AUT_USER_OBJECT_V1, JUDGE_NOVELTY_USABILITY_V3
    from bridge_steering.prompts.macgyver import MACGYVER_USER_DEFAULT_V1, ...

Never define a prompt string in a stage / evaluation script. If you need a
variation, add a new :class:`PromptSpec` here with a new ``id`` (e.g.
``..._v4``), move the old one to :mod:`bridge_steering.prompts._archive` with
``archived=True``, and add a line to ``prompt_lock.json``. The three changes
land in the same PR; reviewers see all of them.

Every artifact's ``_manifest.json`` records the ``id`` and the ``sha256``
of the prompt (covering the full chat structure: system + user template +
parser contract).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSpec:
    """An immutable, named, hashable, chat-aware prompt template.

    The hash covers the full chat structure (system + user_template +
    parser_contract), not just the user string — so moving a sentence from
    ``system`` to ``user`` still counts as drift.
    """

    id: str
    user_template: str
    system: str | None = None
    parser_contract: str | None = None
    archived: bool = False

    # ----- rendering ----------------------------------------------------

    def render(self, **kw) -> list[dict[str, str]]:
        """Return ``[{role, content}, ...]`` — the messages list, not a string."""
        msgs: list[dict[str, str]] = []
        if self.system is not None:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.user_template.format(**kw)})
        return msgs

    def render_user(self, **kw) -> str:
        """Convenience: only the rendered user content (for stages that need
        the bare user string, e.g. for token-span debugging)."""
        return self.user_template.format(**kw)

    # ----- hashing ------------------------------------------------------

    def sha256(self) -> str:
        """sha256 of system + user_template + parser_contract, NUL-separated."""
        h = hashlib.sha256()
        h.update(b"SYS:")
        h.update((self.system or "").encode("utf-8"))
        h.update(b"\x00USR:")
        h.update(self.user_template.encode("utf-8"))
        h.update(b"\x00PAR:")
        h.update((self.parser_contract or "").encode("utf-8"))
        return h.hexdigest()

    def lock_hash(self) -> str:
        return "sha256:" + self.sha256()

    # ----- manifest -----------------------------------------------------

    def manifest_entry(self) -> dict:
        prev = self.user_template[:160].replace("\n", "\\n")
        return {
            "id": self.id,
            "sha256": self.lock_hash(),
            "archived": self.archived,
            "text_preview": prev + ("…" if len(self.user_template) > 160 else ""),
        }


# ---------------------------------------------------------------------------
# Registry helpers (resolve a PromptSpec by id from anywhere in the package)
# ---------------------------------------------------------------------------


_LOCKFILE_PATH = Path(__file__).with_name("prompt_lock.json")


def all_specs() -> dict[str, PromptSpec]:
    """Return every PromptSpec exported from this package (live + archived).

    Imported lazily to avoid circular imports between sub-modules.
    """
    from . import aut, macgyver, _archive

    out: dict[str, PromptSpec] = {}
    for module in (aut, macgyver, _archive):
        for spec in module.SPECS:
            if spec.id in out:
                raise ValueError(
                    f"Duplicate PromptSpec id across prompts/*: {spec.id!r}"
                )
            out[spec.id] = spec
    return out


def resolve(prompt_id: str, allow_archived: bool = False) -> PromptSpec:
    """Look up a PromptSpec by id. Errors are loud and explicit."""
    registry = all_specs()
    if prompt_id not in registry:
        raise KeyError(
            f"Unknown prompt id: {prompt_id!r}. "
            f"Available ids: {sorted(registry)}"
        )
    spec = registry[prompt_id]
    if spec.archived and not allow_archived:
        raise ValueError(
            f"Prompt {prompt_id!r} is archived; set cfg.allow_archived_prompts: true "
            f"to use it (only for replaying old runs)."
        )
    return spec


def load_lockfile() -> dict[str, str]:
    """Return the committed ``{prompt_id: 'sha256:...'}`` map."""
    if not _LOCKFILE_PATH.is_file():
        return {}
    return json.loads(_LOCKFILE_PATH.read_text("utf-8"))


def build_lockfile_dict() -> dict[str, str]:
    """Compute the current lockfile content from all exported specs (for the
    test that regenerates the lockfile on demand)."""
    return {spec.id: spec.lock_hash() for spec in all_specs().values()}
