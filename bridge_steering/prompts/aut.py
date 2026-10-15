"""AUT prompts: user prompts (steering AND inference), judge, OCSAI.

The bug to remember: in the old code the activation collector and the AUT
inference script formatted the user prompt differently, and the judge
prompt was defined in two places (the judge-prompt module
vs. line 334, plus a third copy in ``scripts/run_vllm_judge.py``).

Here every prompt has exactly one definition and one ``id``. Both
``stages/steer_collect.py`` and ``stages/aut_infer.py`` call
:func:`build_user_prompt` with the same ``subject`` argument resolved
from ``RunConfig.aut.prompts``.
"""

from __future__ import annotations

import re
from typing import Literal

from . import PromptSpec


# What part of the ABCD item's `A` field becomes the AUT prompt's subject:
#   - "object_only"     : just the object phrase (e.g. "brick")
#   - "full_instruction": the entire `A` instruction string
#                         (e.g. "Give 8 uses for a brick in a kitchen.")
PromptSubject = Literal["object_only", "full_instruction"]


# ---------------------------------------------------------------------------
# AUT user prompts (used by steering activation collection AND inference)
# ---------------------------------------------------------------------------


AUT_USER_OBJECT_ONLY_V1 = PromptSpec(
    id="aut_user_object_only_v1",
    system="",  # legacy: empty system message, matches bridge_steering.data_prep.SYSTEM_PROMPT
    user_template=(
        "Write 8 unconventional uses for {object}."
        '\nFormat: one use per line, starting with "-".'
    ),
    parser_contract="bullet_or_numbered_lines",
)

# Byte-exact port of the bridge-time activation-collection user prompt from
# commit ff7f7eb (March 2026) — the run that produced the published
# Llama bridge 14-19 α=0.17 row. Use this as `aut.prompts.steering.prompt_id`
# to reproduce that bridge's steering vector (||v||≈3.95).
#
# Pair with AUT_USER_A_PAPER_INFER on `inference.prompt_id` and set
# `aut.prompts.allow_decoupling: true` — bridge and inference templates are
# deliberately different in this setup, matching how the published run was
# generated.
AUT_USER_A_PAPER_BRIDGE = PromptSpec(
    id="aut_user_a_paper_bridge",
    system="",
    user_template=(
        "Alternative Uses Task\n"
        "\n"
        "{full_instruction}\n"
        "\n"
        "Give 8 unconventional uses. List them clearly."
    ),
    parser_contract="bullet_or_numbered_lines",
)

# Byte-exact port of the AUT-inference user prompt from commit ff7f7eb
# (March 2026) — the gen-time prompt used in the published Llama bridge
# 14-19 α=0.17 row. Pair with AUT_USER_A_PAPER_BRIDGE on the bridge side.
AUT_USER_A_PAPER_INFER = PromptSpec(
    id="aut_user_a_paper_infer",
    system="",
    user_template=(
        "{full_instruction}"
        "\nReturn exactly 8 unconventional uses."
        '\nFormat: one use per line, starting with "-".'
    ),
    parser_contract="bullet_or_numbered_lines",
)


# Style-matched prompting control for the judge-confound question (reviewer,
# 2026-07-19): can mechanism/affordance FRAMING alone — no steering — reproduce
# the bridge-steering gains? Same skeleton, format contract, and decoding as
# AUT_USER_A_PAPER_INFER; the only addition is the mechanism-framing
# instruction, worded to mirror the bridge register (rule -> unlocks) so the
# control is honestly style-matched. Baseline-family configs only (no steering).
AUT_USER_MECHANISM_STYLE_V1 = PromptSpec(
    id="aut_user_mechanism_style_v1",
    system="",
    user_template=(
        "{full_instruction}"
        "\nBefore answering, re-represent the object in terms of its underlying"
        " physical mechanisms and affordances — the functional roles it can play"
        " (for example: tension, friction, geometry, conductivity)."
        "\nReturn exactly 8 unconventional uses that exploit these mechanisms,"
        " phrased in functional, mechanism-oriented language."
        '\nFormat: one use per line, starting with "-".'
    ),
    parser_contract="bullet_or_numbered_lines",
)


# v2 of the mechanism-style control (2026-07-19 23:4x). v1 asked the model to
# "re-represent the object ... before answering" and the model complied OUT LOUD:
# the affordance list ("Adhesion", "Geometry", ...) landed in the reply and the
# shared parser scored those fragments as uses. Result: 13.1 parsed "uses" per
# reply vs 8.2 for every other cell, which contaminated BOTH metrics — OCSAI
# scores reply_text as a whole, and the per-use judge averaged over the
# fragments. v2 keeps the identical mechanism FRAMING but makes the output
# contract match every other cell (list only). Same skeleton/decoding as
# AUT_USER_A_PAPER_INFER.
AUT_USER_MECHANISM_STYLE_V2 = PromptSpec(
    id="aut_user_mechanism_style_v2",
    system="",
    user_template=(
        "{full_instruction}"
        "\nFirst, silently re-represent the object in terms of its underlying"
        " physical mechanisms and affordances — the functional roles it can play"
        " (for example: tension, friction, geometry, conductivity)."
        " Do NOT write this analysis out."
        "\nReturn exactly 8 unconventional uses that exploit these mechanisms,"
        " phrased in functional, mechanism-oriented language."
        '\nFormat: one use per line, starting with "-". Output nothing else.'
    ),
    parser_contract="bullet_or_numbered_lines",
)


# AUT_USER_A_BARE: emits A as-is + format hint, with no wrapping/header/preamble.
# Useful when the dataset's A field already contains a complete instruction
# (e.g. "Give 8 unconventional uses for a brick in a kitchen.") so any wrapping
# would just add redundant prefixes. Pin via `aut.prompts.{steering,inference}
# .prompt_id` (same id on both roles to avoid the decoupling guard).
AUT_USER_A_BARE = PromptSpec(
    id="aut_user_a_bare",
    system="",
    user_template=(
        "{full_instruction}"
        '\nFormat: one use per line, starting with "-".'
    ),
    parser_contract="bullet_or_numbered_lines",
)

# AUT_USER_OBJECT_ONLY_V4_PUB: V3_PUB + an explicit "8 uses" count, matching
# the project's standard AUT count. V3_PUB without a count produced ~20 uses
# per reply (the 20-word constraint dominated over the implied "8" in
# "list"), inflating per-method item counts and making comparisons across
# cells uneven. V4_PUB pins the count at 8 to bring r2_pub back in line with
# r2 / r3 cells (also ~8 uses per reply).
AUT_USER_OBJECT_ONLY_V4_PUB = PromptSpec(
    id="aut_user_object_only_v4_pub",
    system="",
    user_template=(
        "Create a list of 8 creative alternative uses for {object}. "
        "Each use should be no more than 20 words long. No adjectives."
        '\nFormat: one use per line, starting with "-".'
    ),
    parser_contract="bullet_or_numbered_lines",
)
# === EXPERIMENTAL [r2_extension] END =======================================


# ---------------------------------------------------------------------------
# Olson-style prompt-induced creativity contrast
#
# Olson et al. (2024), "Steering Large Language Models to Evaluate and
# Amplify Creativity" (arXiv:2412.06060) construct their steering vector
# from activations gathered while the model is *prompted* differently —
# original creative writing prompts vs uncreative rewrites. Their paper
# uses Fan et al. (2018) writing prompts, not AUT, and does not publish a
# uniform "be creative" template.
#
# These two prompts adapt the recipe to AUT via instruction prefixes,
# the standard community translation when applying Olson-style contrast
# to a non-writing-prompts task. The user prompt is otherwise byte-equal
# to the inference prompt — only the leading instruction differs.
#
# These are NOT used at inference. They are used ONLY by
# `stages/olson_collect.py` to build the contrast pairs from which the
# steering vector is constructed.
# ---------------------------------------------------------------------------


AUT_OLSON_CREATIVE_V1 = PromptSpec(
    id="aut_olson_creative_v1",
    system="",
    user_template=(
        "Be very creative and unconventional.\n"
        "Write 8 unconventional uses for {object}."
        '\nFormat: one use per line, starting with "-".'
    ),
    parser_contract="bullet_or_numbered_lines",
)

AUT_OLSON_BORING_V1 = PromptSpec(
    id="aut_olson_boring_v1",
    system="",
    user_template=(
        "Be conventional and ordinary.\n"
        "Write 8 uses for {object}."
        '\nFormat: one use per line, starting with "-".'
    ),
    parser_contract="bullet_or_numbered_lines",
)


# v2_paper: Olson contrast prompts derived from AUT_USER_A_PAPER_BRIDGE so the
# collection-side prompt distribution matches the bridge collection (and shares
# the same {full_instruction} → paper_infer body shift bridge pays at inference).
# Body is byte-identical to paper_bridge except:
#   - the creative side adds "Be very creative and unconventional." prefix;
#     body keeps "unconventional uses" (matches bridge collection word-for-word).
#   - the boring side adds "Be conventional and ordinary." prefix; body changes
#     "unconventional uses" → "standard, common uses" so the prompt is coherent
#     (avoids the "be conventional / give unconventional uses" contradiction).
#
# Use these instead of v1 when matching the paper bridge+infer track, so the
# only delta between Olson collection and bridge collection is the instruction
# prefix + the single body word.
AUT_OLSON_CREATIVE_V2_PAPER = PromptSpec(
    id="aut_olson_creative_v2_paper",
    system="",
    user_template=(
        "Be very creative and unconventional.\n"
        "\n"
        "Alternative Uses Task\n"
        "\n"
        "{full_instruction}\n"
        "\n"
        "Give 8 unconventional uses. List them clearly."
    ),
    parser_contract="bullet_or_numbered_lines",
)

AUT_OLSON_BORING_V2_PAPER = PromptSpec(
    id="aut_olson_boring_v2_paper",
    system="",
    user_template=(
        "Be conventional and ordinary.\n"
        "\n"
        "Alternative Uses Task\n"
        "\n"
        "{full_instruction}\n"
        "\n"
        "Give 8 standard, common uses. List them clearly."
    ),
    parser_contract="bullet_or_numbered_lines",
)

AUT_OLSON_CREATIVE_V3_PAPER = PromptSpec(
    id="aut_olson_creative_v3_paper",
    system="",
    user_template=(
        "Alternative Uses Task\n"
        "\n"
        "{full_instruction}\n"
        "\n"
        "Give 8 unconventional uses. Be very creative and unconventional. List them clearly."
    ),
    parser_contract="bullet_or_numbered_lines",
)

AUT_OLSON_BORING_V3_PAPER = PromptSpec(
    id="aut_olson_boring_v3_paper",
    system="",
    user_template=(
        "Alternative Uses Task\n"
        "\n"
        "{full_instruction}\n"
        "\n"
        "Give 8 standard, common uses. Be conventional and ordinary. List them clearly."
    ),
    parser_contract="bullet_or_numbered_lines",
)



# ---------------------------------------------------------------------------
# Brainstorm, then Select — Summers-Stay, Lukin & Voss (AAAI-23)
# Source: openreview.net/pdf?id=8HwKaJ1wvl
#
# AUT_BRAINSTORM_V1   ← Table 1 (Stevenson et al. 2022 prompt, adopted as
#                       Phase-1 input by Summers-Stay et al.)
# AUT_BS_ADVANTAGES_V1, AUT_BS_DRAWBACKS_V1, AUT_BS_UTIL_EVAL_V1
#                     ← Table 2 (utility chain)
# AUT_BS_ORIG_EVAL_V1 ← Table 4 (originality probe)
#
# All four Phase-2 prompts end with "A:" — the next-token log-prob of
# " Yes" vs " No" is the paper's confidence signal.
# ---------------------------------------------------------------------------


AUT_BRAINSTORM_V1 = PromptSpec(
    id="aut_brainstorm_v1",
    system="",
    user_template=(
        "What are some creative uses for {object}? The goal is to come up with "
        "creative ideas, which are ideas that strike people as clever, unusual, "
        "interesting, uncommon, humorous, innovative, or different.\n"
        "List 10 creative uses for {object}."
    ),
    parser_contract="bullet_or_numbered_lines",
)


AUT_BS_ADVANTAGES_V1 = PromptSpec(
    id="aut_bs_advantages_v1",
    system="",
    user_template=(
        "Q: Name one or more advantages to using a {object} for the following "
        "purpose: {purpose}? A:"
    ),
    parser_contract="free_text_single_completion",
)


AUT_BS_DRAWBACKS_V1 = PromptSpec(
    id="aut_bs_drawbacks_v1",
    system="",
    user_template=(
        "Q: Name one or more drawbacks to using a {object} for the following "
        "purpose: {purpose}? A:"
    ),
    parser_contract="free_text_single_completion",
)


AUT_BS_UTIL_EVAL_V1 = PromptSpec(
    id="aut_bs_util_eval_v1",
    system="",
    user_template=(
        "Advantages: {advantages}\n"
        "Drawbacks: {drawbacks}\n"
        "Q: Based on these advantages and drawbacks, do you think using a "
        "{object} for the purpose {purpose} is a good idea? Answer Yes or No. A:"
    ),
    parser_contract="logprob_probe: yes_vs_no",
)


AUT_BS_ORIG_EVAL_V1 = PromptSpec(
    id="aut_bs_orig_eval_v1",
    system="",
    user_template=(
        "Q: If someone suggested using a {object} for the following purpose: "
        "{purpose}, would you be surprised and think it was a novel idea? "
        "Answer Yes or No. A:"
    ),
    parser_contract="logprob_probe: yes_vs_no",
)


# ---------------------------------------------------------------------------
# AUT judge prompt (LLM-as-a-judge: novelty + usability, 1-10 JSON)
# ---------------------------------------------------------------------------


# Byte-exact port of the original pipeline's judge prompt
# (`scripts/run_vllm_judge.py::build_judge_prompt`, prompt_style=
# `creativity_and_usability`). This is the prompt that produced the
# Llama bridge-steering 14-19 α=0.17 novelty=6.38 / usability=6.04 row
# in the published table — porting it verbatim into bridge_steering lets the
# new pipeline replicate those numbers.
#
# Differences vs v5: leads with data (`Object: …\nProposed use: …`)
# then the "do not default to 1" anchor, then rubrics that reference
# "object+setting" (not just "object"), and emits a nested schema
# with an analysis block (`key_object_property`, `reasoning`) before
# the integer `scores.{novelty, usability}`. parse_judge_response
# already grabs `scores.*` if present (see stages/aut_judge.py).
JUDGE_CREATIVITY_USABILITY_V1 = PromptSpec(
    id="aut_judge_creativity_usability_v1",
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
        "NOVELTY (1–10): how uncommon/non-obvious the underlying idea is for this object+setting.\n"
        "- 1–2: truly default / first-thought / widely common\n"
        "- 3–4: common pattern with a small twist\n"
        "- 5–6: clearly non-default; plausible new angle\n"
        "- 7–8: rare and surprising; object-specific in a meaningful way\n"
        "- 9–10: extremely rare; would surprise most evaluators; still coherent\n"
        "\n"
        "USABILITY (1–10): likelihood it works as stated with realistic constraints/resources.\n"
        "- 1–2: basically won't work as stated\n"
        "- 3–4: fragile / missing key details / major assumptions\n"
        "- 5–6: plausible but needs care or minor additions\n"
        "- 7–8: practical and straightforward\n"
        "- 9–10: very reliable/robust\n"
        "\n"
        "Output JSON exactly:\n"
        "{{\n"
        "  \"analysis\": {{\n"
        "    \"key_object_property\": \"string (<= 8 words)\",\n"
        "    \"reasoning\": \"string (<= 20 words)\"\n"
        "  }},\n"
        "  \"scores\": {{\n"
        "    \"novelty\": integer,\n"
        "    \"usability\": integer\n"
        "  }}\n"
        "}}"
    ),
    parser_contract="json: {scores: {novelty: int 1-10, usability: int 1-10}}",
)


JUDGE_NOVELTY_USABILITY_V5 = PromptSpec(
    id="aut_judge_novelty_usability_v5",
    system=None,
    user_template=(
        # Same order as V4 (role → rubrics → data → schema) but drops the
        # analysis block (key_object_property / reasoning). Two reasons:
        # (a) the analysis was never read — only scores.* end up in the
        # CSV, and (b) on gpt-5 family the analysis chews up output-token
        # budget for no downstream gain.
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
        '{{ "novelty": integer 1-10, "usability": integer 1-10 }}'
    ),
    parser_contract="json: {novelty: int 1-10, usability: int 1-10}",
)


# ---------------------------------------------------------------------------
# OCSAI scoring (HTTP request — schema is the "prompt")
# ---------------------------------------------------------------------------


OCSAI_REQUEST_V1 = PromptSpec(
    id="ocsai_openscoring_v1",
    system=None,
    user_template=(
        # The actual request body is built in stages/aut_ocsai.py; this is
        # the version-tagged schema we record in the manifest.
        "POST {url}\n"
        "body: {{model: {model}, prompt: 'object,use', input: <object,use_text>}}"
    ),
    parser_contract="ocsai_response: {originality: float}",
)


# ---------------------------------------------------------------------------
# Public registries + helpers
# ---------------------------------------------------------------------------


# Denial prompting (Lu et al. 2024 — "Benchmarking Language Model Creativity")
# adapted to AUT. Round 0 uses the standard AUT_USER_A_PAPER_INFER prompt;
# rounds 1..T inject an "Avoid the following approaches" block listing one
# previously-extracted approach per round, sampled by the reviewer pass.
#
# {constraints_block} is rendered by the denial_aut_infer stage as:
#     Avoid the following approaches:
#     - {approach_1}
#     - {approach_2}
#     - ...
# (empty string would leave a dangling header, so the round-0 path is gated
# in the stage code, not by passing an empty placeholder here.)
AUT_USER_A_PAPER_INFER_DENIAL = PromptSpec(
    id="aut_user_a_paper_infer_denial_v1",
    system="",
    user_template=(
        "{full_instruction}"
        "\n\n{constraints_block}"
        "\n\nReturn exactly 8 unconventional uses."
        '\nFormat: one use per line, starting with "-".'
    ),
    parser_contract="bullet_or_numbered_lines",
)


# Reviewer prompt (offline pass): given a model's AUT reply, extract ALL the
# core idea categories present as short freeform phrases. Output is parsed
# as a bullet list and de-duplicated against the running constraints set
# (denial_aut_infer.parse_approaches).
AUT_REVIEWER_APPROACHES_V1 = PromptSpec(
    id="aut_reviewer_approaches_v1",
    system="",
    user_template=(
        "The following is a list of alternative uses for an object:\n"
        "\n"
        "{aut_response}\n"
        "\n"
        "Detect ALL the core ideas or use categories present in these uses.\n"
        "Use short phrases (max 8 words), one per line, starting with \"-\".\n"
        "\n"
        "Your output should look like this:\n"
        "- short phrase 1\n"
        "- short phrase 2\n"
        "- short phrase 3\n"
        "- ..."
    ),
    parser_contract="bullet_or_numbered_lines",
)


SPECS: list[PromptSpec] = [
    JUDGE_CREATIVITY_USABILITY_V1,
    AUT_USER_OBJECT_ONLY_V1,
    AUT_USER_A_PAPER_BRIDGE,
    AUT_USER_A_PAPER_INFER,
    AUT_USER_MECHANISM_STYLE_V1,
    AUT_USER_MECHANISM_STYLE_V2,
    AUT_USER_A_PAPER_INFER_DENIAL,
    AUT_REVIEWER_APPROACHES_V1,
    # === EXPERIMENTAL [r2_extension] START =================================
    AUT_USER_A_BARE,
    AUT_USER_OBJECT_ONLY_V4_PUB,
    # === EXPERIMENTAL [r2_extension] END ===================================
    # Olson-style prompt-induced contrast (steering-only, never inference)
    AUT_OLSON_CREATIVE_V1,
    AUT_OLSON_BORING_V1,
    AUT_OLSON_CREATIVE_V2_PAPER,
    AUT_OLSON_BORING_V2_PAPER,
    AUT_OLSON_CREATIVE_V3_PAPER,
    AUT_OLSON_BORING_V3_PAPER,
    # Brainstorm-then-Select (Summers-Stay 2022)
    AUT_BRAINSTORM_V1,
    AUT_BS_ADVANTAGES_V1,
    AUT_BS_DRAWBACKS_V1,
    AUT_BS_UTIL_EVAL_V1,
    AUT_BS_ORIG_EVAL_V1,
    JUDGE_NOVELTY_USABILITY_V5,
    OCSAI_REQUEST_V1,
]


# Phase-2 prompt registry — gives ``aut_brainstorm_select`` a single source
# of truth for the four chain prompts (and lets a test verify the set).
BRAINSTORM_SELECT_REGISTRY: dict[str, PromptSpec] = {
    "brainstorm":   AUT_BRAINSTORM_V1,
    "advantages":   AUT_BS_ADVANTAGES_V1,
    "drawbacks":    AUT_BS_DRAWBACKS_V1,
    "utility_eval": AUT_BS_UTIL_EVAL_V1,
    "originality_eval": AUT_BS_ORIG_EVAL_V1,
}


JUDGE_AUT_REGISTRY: dict[str, PromptSpec] = {
    JUDGE_CREATIVITY_USABILITY_V1.id: JUDGE_CREATIVITY_USABILITY_V1,
    JUDGE_NOVELTY_USABILITY_V5.id: JUDGE_NOVELTY_USABILITY_V5,
}


# ---------------------------------------------------------------------------
# Subject resolution for the AUT user prompt
# ---------------------------------------------------------------------------


def _parse_object_from_a(a_text: str) -> str:
    """Port of ``bridge_steering.data_prep.parse_aut_object_from_a``."""
    m = re.search(
        r"uses for (?:a |an )?(?:piece of )?([\w\s\-]+?)(?:\s*\([^)]*\))?"
        r"(?:\s*,\s*focusing\b|\s+(?:in|focusing|that|as)\b|\.|$)",
        a_text,
        re.I,
    )
    return m.group(1).strip() if m else a_text


def aut_object(item: dict) -> tuple[str, Literal["A_OBJECT", "parsed_from_A"]]:
    """Object phrase + source label, for stable per-row provenance."""
    ao = item.get("A_OBJECT")
    if isinstance(ao, str) and ao.strip():
        return ao.strip(), "A_OBJECT"
    return _parse_object_from_a(str(item.get("A", ""))), "parsed_from_A"


def spec_for_subject(subject: PromptSubject) -> PromptSpec:
    if subject == "object_only":
        return AUT_USER_OBJECT_ONLY_V1
    if subject == "full_instruction":
        # Default to the bare A + format template. Callers that want the paper
        # bridge / inference templates pin them explicitly via
        # ``aut.prompts.{steering,inference}.prompt_id`` (which overrides this
        # subject-based default in ``build_pairs`` / ``aut_infer``).
        return AUT_USER_A_BARE
    raise ValueError(f"Unknown AUT prompt subject: {subject!r}")


def build_user_prompt(
    item: dict,
    subject: PromptSubject,
) -> tuple[list[dict[str, str]], PromptSpec, str]:
    """Build the AUT user prompt for a dataset item.

    Returns ``(messages, spec_used, rendered_user_string)``.

    Both ``stages.steer_collect`` and ``stages.aut_infer`` call this. The
    caller writes ``spec_used.manifest_entry()`` into the per-stage manifest
    and ``sha256(rendered_user_string)`` into the per-row CSV.
    """
    spec = spec_for_subject(subject)
    if subject == "object_only":
        obj, _ = aut_object(item)
        msgs = spec.render(object=obj)
    else:
        msgs = spec.render(full_instruction=str(item.get("A", "")))
    user_str = msgs[-1]["content"]
    return msgs, spec, user_str


def build_judge_prompt(
    prompt_id: str,
    object_name: str,
    use_text: str,
    allow_archived: bool = False,
) -> tuple[list[dict[str, str]], PromptSpec, str]:
    """Build the LLM-judge prompt for a single (object, use) pair."""
    from . import resolve

    spec = resolve(prompt_id, allow_archived=allow_archived)
    if spec.id not in JUDGE_AUT_REGISTRY and not spec.archived:
        raise ValueError(
            f"{prompt_id!r} is not an AUT judge prompt "
            f"(known: {sorted(JUDGE_AUT_REGISTRY)})"
        )
    msgs = spec.render(object=object_name, use=use_text or "")
    return msgs, spec, msgs[-1]["content"]
