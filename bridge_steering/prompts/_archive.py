"""Deprecated prompt versions, kept so old runs can be replayed by ``id``.

Add a prompt here (with ``archived=True``) when you bump its version. Never
delete — the manifest of an old run will reference an ``id`` we must still
resolve.

To select an archived prompt for a new run you must set
``cfg.allow_archived_prompts: true``; otherwise :func:`prompts.resolve`
raises. This blocks accidental experiments on obsolete prompts while
keeping replay possible.
"""

from __future__ import annotations

from . import PromptSpec


# The 5-point integer rubric that the original implementation
# silently shadowed the 10-point JSON version with (bug #1).
AUT_JUDGE_NOVELTY_USABILITY_V0_5PT = PromptSpec(
    id="aut_judge_novelty_usability_v0_5pt",
    system=None,
    user_template=(
        "Rate the creativity and originality of the following alternative use "
        "for a {object} on a scale of 1 to 5:\n"
        "\n"
        "Alternative use: {use}\n"
        "\n"
        "Rating criteria:\n"
        "1 - Common/obvious use, not creative\n"
        "2 - Somewhat uncommon but still fairly obvious\n"
        "3 - Moderately creative and original\n"
        "4 - Very creative and original\n"
        "5 - Exceptionally creative, highly original and unconventional\n"
        "\n"
        "Respond with only the number (1-5):"
    ),
    parser_contract="integer 1-5",
    archived=True,
)


# The 1-10 JSON v2 used by the original implementation before
# being shadowed by the 5-point version at line 334. v3 in aut.py replaces it.
AUT_JUDGE_NOVELTY_USABILITY_V2 = PromptSpec(
    id="aut_judge_novelty_usability_v2",
    system=None,
    user_template=(
        "Output ONLY one valid JSON object. No extra text.\n"
        "\n"
        "Object: {object}\n"
        "Proposed use: {use}\n"
        "\n"
        "You are a judge for a dataset of intentionally unconventional uses.\n"
        "This dataset contains many intentionally unconventional uses, so do not default to 1.\n"
        "Choose the closest anchor below.\n"
        "\n"
        "NOVELTY (1-10): how uncommon/non-obvious the underlying idea is for "
        "this object+setting.\n"
        "- 1-2: truly default / first-thought / widely common\n"
        "- 3-4: common pattern with a small twist\n"
        "- 5-6: clearly non-default; plausible new angle\n"
        "- 7-8: rare and surprising; object-specific in a meaningful way\n"
        "- 9-10: extremely rare; would surprise most evaluators; still coherent\n"
        "\n"
        "USABILITY (1-10): likelihood it works as stated with realistic constraints.\n"
        "- 1-2: basically won't work as stated\n"
        "- 3-4: fragile / missing key details / major assumptions\n"
        "- 5-6: plausible but needs care or minor additions\n"
        "- 7-8: practical and straightforward\n"
        "- 9-10: very reliable/robust\n"
        "\n"
        "overall = round((novelty + usability) / 2)\n"
        "\n"
        "Output JSON exactly:\n"
        "{{\n"
        '  "analysis": {{\n'
        '    "key_object_property": "string (<= 8 words)",\n'
        '    "reasoning": "string (<= 20 words)"\n'
        "  }},\n"
        '  "scores": {{\n'
        '    "novelty": integer,\n'
        '    "usability": integer,\n'
        '    "overall": integer\n'
        "  }}\n"
        "}}"
    ),
    parser_contract="json: {scores.novelty, scores.usability, scores.overall}",
    archived=True,
)


# The v3 that drops "overall" from the JSON schema (it's computed in code as
# round((novelty + usability) / 2)). Superseded by v4 which fixes (a) the
# dangling "object+setting" reference — no setting field is ever rendered —
# and (b) the inverted prompt structure (format-hint at top, schema at
# bottom, data sandwiched between rubrics). v4 reorders to role → rubric →
# data → schema, all contiguous.
AUT_JUDGE_NOVELTY_USABILITY_V3 = PromptSpec(
    id="aut_judge_novelty_usability_v3",
    system=None,
    user_template=(
        "Output ONLY one valid JSON object. No extra text.\n"
        "\n"
        "Object: {object}\n"
        "Proposed use: {use}\n"
        "\n"
        "You are a judge for a dataset of intentionally unconventional uses.\n"
        "This dataset contains many intentionally unconventional uses, so do not default to 1.\n"
        "Choose the closest anchor below.\n"
        "\n"
        "NOVELTY (1-10): how uncommon/non-obvious the underlying idea is for "
        "this object+setting.\n"
        "- 1-2: truly default / first-thought / widely common\n"
        "- 3-4: common pattern with a small twist\n"
        "- 5-6: clearly non-default; plausible new angle\n"
        "- 7-8: rare and surprising; object-specific in a meaningful way\n"
        "- 9-10: extremely rare; would surprise most evaluators; still coherent\n"
        "\n"
        "USABILITY (1-10): likelihood it works as stated with realistic constraints.\n"
        "- 1-2: basically won't work as stated\n"
        "- 3-4: fragile / missing key details / major assumptions\n"
        "- 5-6: plausible but needs care or minor additions\n"
        "- 7-8: practical and straightforward\n"
        "- 9-10: very reliable/robust\n"
        "\n"
        "Output JSON exactly:\n"
        "{{\n"
        '  "analysis": {{\n'
        '    "key_object_property": "string (<= 8 words)",\n'
        '    "reasoning": "string (<= 20 words)"\n'
        "  }},\n"
        '  "scores": {{ "novelty": integer, "usability": integer }}\n'
        "}}"
    ),
    parser_contract="json: {scores.novelty: int 1-10, scores.usability: int 1-10}",
    archived=True,
)


# The v4 that kept an unused "analysis" block (key_object_property +
# reasoning). Nothing downstream read those fields — only ``scores.*``
# made it into the CSV — and on gpt-5 family they chewed up output-token
# budget for no gain. v5 drops the wrapper: schema is just ``{novelty,
# usability}``.
AUT_JUDGE_NOVELTY_USABILITY_V4 = PromptSpec(
    id="aut_judge_novelty_usability_v4",
    system=None,
    user_template=(
        "You are a judge for a dataset of intentionally unconventional uses "
        "for everyday objects. Many entries are deliberately uncommon — use "
        "the full 1-10 range and choose the closest anchor in each rubric "
        "below.\n"
        "\n"
        "NOVELTY (1-10): how uncommon/non-obvious the underlying idea is for "
        "this object.\n"
        "- 1-2: truly default / first-thought / widely common\n"
        "- 3-4: common pattern with a small twist\n"
        "- 5-6: clearly non-default; plausible new angle\n"
        "- 7-8: rare and surprising; object-specific in a meaningful way\n"
        "- 9-10: extremely rare; would surprise most evaluators; still coherent\n"
        "\n"
        "USABILITY (1-10): likelihood it works as stated with realistic constraints.\n"
        "- 1-2: basically won't work as stated\n"
        "- 3-4: fragile / missing key details / major assumptions\n"
        "- 5-6: plausible but needs care or minor additions\n"
        "- 7-8: practical and straightforward\n"
        "- 9-10: very reliable/robust\n"
        "\n"
        "Now judge this entry.\n"
        "Object: {object}\n"
        "Proposed use: {use}\n"
        "\n"
        "Respond with ONE valid JSON object and nothing else, matching this "
        "schema exactly:\n"
        "{{\n"
        '  "analysis": {{\n'
        '    "key_object_property": "<= 8 words",\n'
        '    "reasoning": "<= 20 words"\n'
        "  }},\n"
        '  "scores": {{ "novelty": integer 1-10, "usability": integer 1-10 }}\n'
        "}}"
    ),
    parser_contract="json: {scores.novelty: int 1-10, scores.usability: int 1-10}",
    archived=True,
)


SPECS: list[PromptSpec] = [
    AUT_JUDGE_NOVELTY_USABILITY_V0_5PT,
    AUT_JUDGE_NOVELTY_USABILITY_V2,
    AUT_JUDGE_NOVELTY_USABILITY_V3,
    AUT_JUDGE_NOVELTY_USABILITY_V4,
]
