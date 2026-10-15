"""
SFT data preparation: prompt/target formatting and contrastive pair building (AUT only).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal
class TaskType(Enum):
    """Alternative Uses Task only (ABCD AUT JSON)."""

    AUT = "aut"


# ---------------------------------------------------------------------------
# User-prompt formatters
# ---------------------------------------------------------------------------


def parse_aut_object_from_a(a_text: str) -> str:
    """Extract the object phrase after ``uses for`` in an AUT ``A`` instruction.

    Shared by dataset ``A_OBJECT``, ``run_aut_inference``, and ``select_best_uses``.
    """
    m = re.search(
        r"uses for (?:a |an )?(?:piece of )?([\w\s\-]+?)(?:\s*\([^)]*\))?"
        r"(?:\s*,\s*focusing\b|\s+(?:in|focusing|that|as)\b|\.|$)",
        a_text,
        re.I,
    )
    return m.group(1).strip() if m else a_text


AutPromptSubject = Literal["full_a", "object"]


def aut_object_for_aut_prompt(item: dict) -> str:
    """Object phrase for object-style AUT prompts: ``A_OBJECT`` if set, else parse ``A``."""
    ao = item.get("A_OBJECT")
    if isinstance(ao, str) and ao.strip():
        return ao.strip()
    return parse_aut_object_from_a(str(item.get("A", "")))


def format_user_prompt_aut(
    item: dict,
    *,
    aut_prompt_subject: AutPromptSubject = "full_a",
) -> str:
    """Canonical AUT user message for bridge (``build_contrastive_pair``) and eval.

    * ``full_a`` — embed the full instruction string ``A`` (legacy bridge / ablations).
    * ``object`` — embed only the object (``A_OBJECT`` or parsed from ``A``). AUT inference
      uses ``object`` only (see ``run_aut_inference``).

    Kept in sync with ``scripts/run_aut_inference._format_user_prompt``.
    """
    if aut_prompt_subject == "object":
        subject = aut_object_for_aut_prompt(item)
    else:
        subject = str(item.get("A", ""))
    return (
        f"Write 8 unconventional uses for {subject}."
        '\nFormat: one use per line, starting with "-".'
    )


def format_user_prompt_aut_march2026_bridge(item: dict) -> str:
    """OPT-IN: March 2026 (commit ff7f7eb) bridge-activation prompt.

    Wraps ``A`` in "Alternative Uses Task\\n\\n…\\n\\nGive 8 unconventional uses.
    List them clearly." Used by the published bridge-steering 14-19 α=0.17 row
    (Llama novelty 6.38, sem.div 0.43). Not the default — opt in via the
    `--aut-prompt-style march_2026` flag in ``run_bridge_steering.py``.
    """
    task = str(item.get("A", ""))
    return (
        "Alternative Uses Task\n\n"
        f"{task}\n\n"
        "Give 8 unconventional uses. List them clearly."
    )


def format_user_prompt_baseline_aut(item: dict) -> str:
    """Baseline AUT prompt (same as standard; mechanism is not requested)."""
    return format_user_prompt_aut(item)


def format_user_prompt(item: dict) -> str:
    """Alias for ``format_user_prompt_aut``."""
    return format_user_prompt_aut(item)


# ---------------------------------------------------------------------------
# Teacher-forced completions for contrastive / steering pairs
# ---------------------------------------------------------------------------


def format_D_completion(item: dict) -> str:
    """Default-listing completion (teacher-forced, condition D_banal)."""
    return "\n".join(f"- {use}" for use in item["D_banal"])


# ---------------------------------------------------------------------------
# Contrastive pair builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = ""


def build_contrastive_pair(
    item: dict,
    d_key: str = "D_banal",
    *,
    aut_prompt_subject: AutPromptSubject = "full_a",
    aut_prompt_style: str = "current",
) -> dict:
    """Build a contrastive pair using B_list and D_banal only.

    ``aut_prompt_style`` selects which user-prompt template renders ``A``:

    * ``"current"`` (default, backward compatible): the live
      ``format_user_prompt_aut(item, aut_prompt_subject=...)`` template.
    * ``"march_2026"``: byte-exact restoration of the bridge prompt used by
      commit ff7f7eb to produce the published bridge-steering 14-19 α=0.17 row
      (renders ``Alternative Uses Task\\n\\n{A}\\n\\nGive 8 unconventional uses.
      List them clearly.``).
    """
    if d_key != "D_banal":
        raise ValueError(f"Only d_key='D_banal' is supported, got: {d_key}")
    try:
        b = item["B_list"][0]
    except Exception as e:
        raise ValueError(
            f"Invalid AUT item {item.get('id', '<no-id>')}: expected non-empty 'B_list'."
        ) from e
    if aut_prompt_style == "march_2026":
        user_prompt = format_user_prompt_aut_march2026_bridge(item)
    elif aut_prompt_style == "current":
        user_prompt = format_user_prompt_aut(item, aut_prompt_subject=aut_prompt_subject)
    else:
        raise ValueError(
            f"aut_prompt_style must be 'current' or 'march_2026', got {aut_prompt_style!r}"
        )
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return {
        "id": item.get("id", ""),
        "domain": item.get("domain", item.get("task", "")),
        "user_prompt": user_prompt,
        "prompt_msgs": prompt_msgs,
        "d_completion": format_D_completion(item),
        "b_completion": (
            f"Mechanism type: {b['type']}\n"
            f"Rule: {b['rule']}\n"
            f"This unlocks: {b['unlocks']}\n"
        ),
    }
