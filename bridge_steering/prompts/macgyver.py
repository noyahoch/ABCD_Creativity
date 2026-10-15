"""MacGyver prompts: user prompts (inference) and LLM-as-a-judge."""

from __future__ import annotations

from typing import Literal

from . import PromptSpec


MacgyverPromptStyle = Literal["default", "novelty_instruct", "mechanism_style"]


# ---------------------------------------------------------------------------
# User prompts for inference
# ---------------------------------------------------------------------------


_DEFAULT_SYSTEM = (
    "You are a creative problem solver. Given a scenario and a set of "
    "available tools, propose a practical step-by-step solution using only "
    "the provided items."
)

_DEFAULT_SUFFIX = (
    "\n\nGive a feasible solution very concisely. "
    "Use Step 1, Step 2, etc., and mention the tools used in each step. "
    "Use as few steps as possible and keep the answer under 150 words."
)


MACGYVER_USER_DEFAULT_V1 = PromptSpec(
    id="macgyver_user_default_v1",
    system=_DEFAULT_SYSTEM,
    user_template="{problem}" + _DEFAULT_SUFFIX,
    parser_contract="freeform_steps",
)


# Style-matched prompting control (judge-confound reviewer question, 2026-07-19):
# the MacGyver analog of aut_user_mechanism_style_v1. Identical system message
# and format suffix to the default prompt — the ONLY delta is the mechanism/
# affordance-framing instruction, worded to mirror the bridge register. If
# framing alone reproduced the steering gains, they would be promptable style.
MACGYVER_USER_MECHANISM_STYLE_V1 = PromptSpec(
    id="macgyver_user_mechanism_style_v1",
    system=_DEFAULT_SYSTEM,
    user_template=(
        "{problem}"
        "\n\nBefore answering, re-represent each available object in terms of"
        " its underlying physical mechanisms and affordances — the functional"
        " roles it can play (for example: tension, friction, geometry,"
        " conductivity). Construct your solution by exploiting these mechanisms,"
        " and phrase each step in functional, mechanism-oriented language."
        + _DEFAULT_SUFFIX
    ),
    parser_contract="freeform_steps",
)


# v2 (2026-07-19 23:4x, user): same flaw as the AUT v1 control — "Before
# answering, re-represent each available object..." made the model WRITE the
# affordance analysis into the solution text, which inflates mac 4-gram
# originality (more varied prose) and dilutes the judge's quality read of the
# actual steps. v2 keeps identical framing but requires it to stay silent, so
# the output contract matches every other MacGyver cell (steps only).
MACGYVER_USER_MECHANISM_STYLE_V2 = PromptSpec(
    id="macgyver_user_mechanism_style_v2",
    system=_DEFAULT_SYSTEM,
    user_template=(
        "{problem}"
        "\n\nBefore answering, silently re-represent each available object in"
        " terms of its underlying physical mechanisms and affordances — the"
        " functional roles it can play (for example: tension, friction,"
        " geometry, conductivity). Do NOT write this analysis out."
        " Construct your solution by exploiting these mechanisms, and phrase"
        " each step in functional, mechanism-oriented language."
        " Output only the numbered steps."
        + _DEFAULT_SUFFIX
    ),
    parser_contract="freeform_steps",
)


MACGYVER_USER_NOVELTY_INSTRUCT_V1 = PromptSpec(
    id="macgyver_user_novelty_instruct_v1",
    system=None,  # paper E.2.3 uses user-only message
    user_template=(
        "MacGyver are real-world problems deliberately designed "
        "to trigger innovative usage of objects and necessitate "
        "out-of-the-box thinking.\n"
        "Here are some tips for answering MacGyver questions:\n"
        "1. Understand the Problem Context Thoroughly\n"
        "* Carefully read the problem description, including the tools and "
        "constraints provided.\n"
        "* Identify the objective and key limitations, focusing on how they "
        "constrain traditional solutions.\n"
        "2. Leverage Divergent Thinking:\n"
        "* Enumerate potential unconventional uses for each tool provided, "
        "exploring creative possibilities beyond typical applications.\n"
        "* Consider combining tools in innovative ways to enhance "
        "functionality or bypass constraints.\n"
        "3. Apply Convergent Thinking:\n"
        "* Refine the solution to ensure it directly addresses the problem "
        "with minimal steps.\n"
        "* Validate that the approach adheres to physical, logical, and "
        "contextual constraints described in the task.\n"
        "4. Avoid Physically or Contextually Infeasible Proposals:\n"
        "* Cross-check the proposed actions against basic physical laws.\n"
        "* Ensure that all tools suggested in the solution are explicitly "
        "available and aligned with stated constraints.\n"
        "5. Demonstrate High-Quality Creativity:\n"
        "* Propose solutions that are novel and insightful.\n"
        "* Structure responses to emphasize clarity and logical progression.\n"
        "Here is the MacGyver prompt I want you to answer:\n"
        "{problem}\n"
        "Instruction:\n"
        "- First, think about how to answer in a way that demonstrates "
        "high quality and creativity while avoiding over-reliance on "
        "n-grams from pretraining data by using the tips provided above.\n"
        "- Return your response, ensuring it is enclosed with asterisks."
    ),
    parser_contract="freeform_steps_in_asterisks",
)


# ---------------------------------------------------------------------------
# Judge prompt (additive 5-point rubric)
# ---------------------------------------------------------------------------


_ADDITIVE_5PT_USER_TEMPLATE = (
    "Review the solution generated in response to a MacGyver-style problem "
    "and score it using the additive 5-point scoring system described "
    "below. Points are accumulated based on the satisfaction of each "
    "criterion:\n"
    "\n"
    "Problem statement: {user_prompt}\n"
    "\n"
    "Model's solution: {model_response}\n"
    "\n"
    "- Add 1 point if the solution attempts to address the problem using "
    "only the given resources, without introducing external tools or "
    "elements not mentioned.\n"
    "- Add another point if the solution demonstrates a reasonable "
    "understanding of the properties and limitations of the available "
    "resources, and applies them correctly.\n"
    "- Award a third point if the solution adheres to the physical "
    "constraints of the problem (e.g., size, weight, strength) and does "
    "not propose an obviously unfeasible approach.\n"
    "- Grant a fourth point if the solution is practical and likely to "
    "solve the problem effectively within the constraints of the "
    "scenario.\n"
    "- Bestow a fifth point for a solution that is complete, logically "
    "structured, and provides a clear explanation of how it solves the "
    "problem.\n"
    "\n"
    "After examining the problem, available resources, and the proposed "
    "solution:\n"
    "- First explain your total score, up to 100 words.\n"
    "- Then conclude with the score using the format:\n"
    "Score: <total points>"
)


MACGYVER_JUDGE_ADDITIVE_5PT_V1 = PromptSpec(
    id="macgyver_judge_additive_5pt_v1",
    system=(
        "You are an expert evaluator for creative problem-solving tasks. "
        "Score solutions using the additive rubric provided."
    ),
    user_template=_ADDITIVE_5PT_USER_TEMPLATE,
    parser_contract="freeform; final line matches `Score: <int 0-5>`",
)


# Same rubric as V1 but with system=None — used to test whether the
# "expert evaluator" framing affects scores (Qwen's chat template otherwise
# inserts its default "You are Qwen, ... a helpful assistant." system msg).
MACGYVER_JUDGE_ADDITIVE_5PT_NOSYS_V1 = PromptSpec(
    id="macgyver_judge_additive_5pt_nosys_v1",
    system=None,
    user_template=_ADDITIVE_5PT_USER_TEMPLATE,
    parser_contract="freeform; final line matches `Score: <int 0-5>`",
)


# ---------------------------------------------------------------------------
# Public registries + helpers
# ---------------------------------------------------------------------------


# Denial prompting (Lu et al. 2024) adapted to MacGyver. Round 0 uses
# MACGYVER_USER_DEFAULT_V1; rounds 1..T inject a "{constraints_block}"
# enumerating short freeform "approaches" extracted by the reviewer pass
# from the previous round's solution. The system message is unchanged.
MACGYVER_USER_DEFAULT_DENIAL_V1 = PromptSpec(
    id="macgyver_user_default_denial_v1",
    system=_DEFAULT_SYSTEM,
    user_template=(
        "{problem}"
        "\n\n{constraints_block}"
        + _DEFAULT_SUFFIX
    ),
    parser_contract="freeform_steps",
)


# Reviewer prompt (offline pass): given a MacGyver solution, extract ALL
# the core ideas or approaches as short freeform phrases. Output is parsed
# as a bullet list by denial_macgyver_infer.parse_approaches and
# de-duplicated against the running constraints set.
MACGYVER_REVIEWER_APPROACHES_V1 = PromptSpec(
    id="macgyver_reviewer_approaches_v1",
    system=None,
    user_template=(
        "The following is a step-by-step solution to a real-world problem:\n"
        "\n"
        "{macgyver_response}\n"
        "\n"
        "Detect ALL the core ideas or approaches used in this solution.\n"
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
    MACGYVER_USER_DEFAULT_V1,
    MACGYVER_USER_MECHANISM_STYLE_V1,
    MACGYVER_USER_MECHANISM_STYLE_V2,
    MACGYVER_USER_NOVELTY_INSTRUCT_V1,
    MACGYVER_USER_DEFAULT_DENIAL_V1,
    MACGYVER_REVIEWER_APPROACHES_V1,
    MACGYVER_JUDGE_ADDITIVE_5PT_V1,
    MACGYVER_JUDGE_ADDITIVE_5PT_NOSYS_V1,
]


JUDGE_MACGYVER_REGISTRY: dict[str, PromptSpec] = {
    MACGYVER_JUDGE_ADDITIVE_5PT_V1.id:       MACGYVER_JUDGE_ADDITIVE_5PT_V1,
    MACGYVER_JUDGE_ADDITIVE_5PT_NOSYS_V1.id: MACGYVER_JUDGE_ADDITIVE_5PT_NOSYS_V1,
}


def spec_for_style(style: MacgyverPromptStyle) -> PromptSpec:
    if style == "default":
        return MACGYVER_USER_DEFAULT_V1
    if style == "novelty_instruct":
        return MACGYVER_USER_NOVELTY_INSTRUCT_V1
    if style == "mechanism_style":
        return MACGYVER_USER_MECHANISM_STYLE_V1
    raise ValueError(f"Unknown MacGyver prompt style: {style!r}")


def build_user_prompt(
    item: dict,
    style: MacgyverPromptStyle,
) -> tuple[list[dict[str, str]], PromptSpec, str]:
    """Build the MacGyver user prompt. Returns (messages, spec, rendered_user)."""
    spec = spec_for_style(style)
    msgs = spec.render(problem=item["problem"])
    return msgs, spec, msgs[-1]["content"]


def build_judge_prompt(
    prompt_id: str,
    user_prompt: str,
    model_response: str,
    allow_archived: bool = False,
    response_char_cap: int = 2000,
) -> tuple[list[dict[str, str]], PromptSpec, str]:
    """Build the MacGyver judge prompt for a single (problem, solution) pair.

    ``response_char_cap`` matches the original implementation
    behaviour (truncate model_response to 2000 chars).
    """
    from . import resolve

    spec = resolve(prompt_id, allow_archived=allow_archived)
    if spec.id not in JUDGE_MACGYVER_REGISTRY and not spec.archived:
        raise ValueError(
            f"{prompt_id!r} is not a MacGyver judge prompt "
            f"(known: {sorted(JUDGE_MACGYVER_REGISTRY)})"
        )
    msgs = spec.render(
        user_prompt=(user_prompt or "").strip(),
        model_response=(model_response or "").strip()[:response_char_cap],
    )
    return msgs, spec, msgs[-1]["content"]
