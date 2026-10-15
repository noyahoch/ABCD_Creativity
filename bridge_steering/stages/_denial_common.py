"""Shared helpers for the denial_aut_infer / denial_macgyver_infer stages.

Both stages follow the same Lu et al. 2024 ("Benchmarking Language Model
Creativity", arXiv:2407.09007) iterative-constraint loop:

  for t in 0..T:
      if t == 0:
          sol_t[i] = standard_prompt_inference(i)
      else:
          approaches_t[i] = reviewer(sol_{t-1}[i])             # vLLM/API
          new = sample(approaches_t[i] - constraints[i])       # per-item RNG
          constraints[i] += new
          sol_t[i] = inference(i, augmented_prompt(constraints[i]))

The reviewer prompts are freeform short-phrase extractors (no fixed
vocabulary, unlike Lu et al.'s 33-item programming-techniques whitelist).
Per-round outputs are written to the *infer-stage* directory as
``denial_round{t}_results.csv`` so existing ocsai/judge/novelty discovery
picks them up without code changes.
"""

from __future__ import annotations

import asyncio
import random
import re
from typing import Iterable

from ..config import DenialConfig
from ._judge_client import completion_kwargs, judge_model_name, make_judge_client


# ---------------------------------------------------------------------------
# Reviewer-output parsing
# ---------------------------------------------------------------------------


_BULLET_RE = re.compile(r"^[\-\*•]\s*")
_NUM_RE = re.compile(r"^\d+[\.\)]\s*")


def parse_approaches(raw: str) -> list[str]:
    """Extract a deduplicated list of short approach phrases from the reviewer's reply.

    Drops empty lines, bullet markers, and numeric prefixes. Lowercases and
    strips trailing punctuation so naive set-membership against the running
    constraints list catches near-duplicates.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        line = _BULLET_RE.sub("", line.strip())
        line = _NUM_RE.sub("", line).strip().rstrip(".,;:")
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def format_constraints_block(constraints: list[str]) -> str:
    """Build the literal text injected into the denial-augmented user prompt.

    Returns:
        ``"Avoid the following approaches:\\n- a_1\\n- a_2\\n..."``

    Empty input returns an empty string (caller is responsible for routing
    round 0 to the unaugmented prompt; we don't paper over it here).
    """
    if not constraints:
        return ""
    body = "\n".join(f"- {c}" for c in constraints)
    return f"Avoid the following approaches:\n{body}"


# ---------------------------------------------------------------------------
# Per-item RNG: stable across runs without coupling between items
# ---------------------------------------------------------------------------


def per_item_rng(seed: int, item_id, round_idx: int) -> random.Random:
    """Reproducible per-item sampler. Mixes the run-level seed, the item id
    (cast to str), and the round index so each (item, round) gets its own
    deterministic but independent stream."""
    key = f"{seed}|{item_id}|{round_idx}"
    return random.Random(key)


# ---------------------------------------------------------------------------
# Async reviewer batch
# ---------------------------------------------------------------------------


async def _review_one(
    client, model: str, dcfg: DenialConfig, prompt_spec, response_text: str,
    response_var: str,
) -> list[str]:
    """Call the reviewer once on a single solution; return parsed approaches.

    ``response_var`` is the template placeholder name in the reviewer
    PromptSpec (e.g. ``"aut_response"`` or ``"macgyver_response"``).
    Caller resolves the spec by id and passes it in.
    """
    msgs = prompt_spec.render(**{response_var: (response_text or "").strip()})
    last_raw = ""
    last_exc: Exception | None = None
    for _ in range(dcfg.reviewer_max_retries):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    **completion_kwargs(
                        dcfg.reviewer_backend,
                        dcfg.reviewer_max_tokens,
                        dcfg.reviewer_temperature,
                        dcfg.reviewer_seed,
                    ),
                ),
                timeout=dcfg.reviewer_timeout_s,
            )
            last_raw = (resp.choices[0].message.content or "").strip()
            last_exc = None
            approaches = parse_approaches(last_raw)
            if approaches:
                return approaches
        except Exception as exc:
            last_exc = exc
            continue
    # Out of retries (or only empty parses): return whatever the last raw
    # gave us, even if empty — the caller (review_batch) is responsible for
    # raising loudly when ALL items came back empty, which signals a
    # systemic reviewer failure (e.g. vLLM down) rather than an isolated
    # parse miss. ``last_exc`` is stashed on the returned list via the
    # caller, since plain ``list[str]`` can't carry sideband info itself.
    out = parse_approaches(last_raw)
    if not out and last_exc is not None:
        # Tag the list so review_batch can include the exception in its
        # error message. Using a subclass keeps the regular [] case clean.
        out = _TaggedList(out, last_exc=last_exc)  # type: ignore[assignment]
    return out


class _TaggedList(list):
    """List subclass that lets _review_one stash the last exception it saw
    when every retry failed. review_batch unpacks ``last_exc`` to surface
    the underlying cause (connection-refused, 4xx, etc.) instead of just
    "0 approaches extracted." Plain ``[]`` (empty parse from a successful
    HTTP response) carries no ``last_exc`` and is not flagged as systemic."""
    last_exc: Exception | None = None

    def __init__(self, iterable=(), *, last_exc: Exception | None = None):
        super().__init__(iterable)
        self.last_exc = last_exc


class ReviewerSystemicFailure(RuntimeError):
    """Every reviewer call in a round failed (HTTP / timeout / exception),
    not just an isolated parse miss. Almost always means the vLLM reviewer
    isn't running or isn't reachable at ``reviewer_backend.base_url``."""


async def review_batch(
    dcfg: DenialConfig,
    prompt_spec,
    response_var: str,
    responses: list[str],
) -> list[list[str]]:
    """Call the reviewer on a batch of solutions in parallel.

    Returns a list aligned with ``responses`` where each element is the
    deduplicated approach list for that solution. Empty inner lists mean
    the reviewer produced no parseable output for that item; that's tolerable
    per-item.

    Raises ``ReviewerSystemicFailure`` if EVERY item failed with an exception
    (vs. returning a parsed-but-empty list). This catches the "vLLM is down"
    failure mode — silently returning all-empty constraint lists across all
    rounds would otherwise produce a "success" manifest with denial doing
    nothing.
    """
    client = make_judge_client(dcfg.reviewer_backend)
    model = judge_model_name(dcfg.reviewer_backend)
    sema = asyncio.Semaphore(dcfg.reviewer_max_concurrent)

    async def _one(r: str) -> list[str]:
        async with sema:
            return await _review_one(client, model, dcfg, prompt_spec, r, response_var)

    results: list[list[str]] = await asyncio.gather(*(_one(r) for r in responses))

    # Systemic failure: every item came back via the exception path (tagged
    # with last_exc by _review_one). A plain ``[]`` from a successful HTTP
    # response is fine and won't trigger this.
    if results and all(
        isinstance(r, _TaggedList) and r.last_exc is not None for r in results
    ):
        sample_exc = results[0].last_exc  # type: ignore[union-attr]
        raise ReviewerSystemicFailure(
            f"All {len(results)} reviewer calls failed. Likely the vLLM "
            f"reviewer at {dcfg.reviewer_backend.base_url!r} is not running. "
            f"Example error: {sample_exc!r}"
        )
    return results


def select_next_constraint(
    candidates: list[str], existing: list[str], rng: random.Random,
) -> str | None:
    """Pick one freeform approach not already in ``existing``.

    Lowercased+stripped set-membership (so "Use Lever" and "use lever"
    collide). Returns ``None`` if no novel candidate is available.
    """
    existing_set = {c.lower().strip() for c in existing}
    pool = [c for c in candidates if c.lower().strip() not in existing_set]
    if not pool:
        return None
    return rng.choice(pool)


__all__ = [
    "parse_approaches",
    "format_constraints_block",
    "per_item_rng",
    "review_batch",
    "ReviewerSystemicFailure",
    "select_next_constraint",
]
