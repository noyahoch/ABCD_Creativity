"""
Client for Open Creativity Scoring (OCSAI) LLM endpoint at openscoring.du.edu/llm.

See https://openscoring.du.edu/swagger.json for query parameters and optional X-API-KEY.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

DEFAULT_OCSAI_LLM_URL = "https://openscoring.du.edu/llm"


def _parse_openscoring_json(text: str) -> Any:
    """OpenScoring may emit bare NaN/Infinity; stdlib json rejects those without parse_constant."""

    def parse_constant(s: str) -> float:
        if s == "NaN":
            return float("nan")
        if s == "Infinity":
            return float("inf")
        if s == "-Infinity":
            return float("-inf")
        raise ValueError(f"Unexpected JSON constant: {s!r}")

    return json.loads(text, parse_constant=parse_constant)


def format_ocsai_row(prompt: str, response: str) -> str:
    """
    Build one CSV-style row: prompt,response per OpenScoring input format.
    Escapes fields with double quotes when they contain comma, quote, or newline.
    """

    def esc(s: str) -> str:
        if any(c in s for c in ',"\n\r'):
            return '"' + s.replace('"', '""') + '"'
        return s

    return f"{esc(prompt)},{esc(response)}"


def score_ocsai_llm(
    inputs: list[str],
    url: str = DEFAULT_OCSAI_LLM_URL,
    model: str | None = None,
    timeout: float = 30.0,
    api_key: str | None = None,
    extra_params: dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> Any:
    """
    POST JSON {"input": [...]} to the OpenScoring /llm endpoint and return parsed JSON.

    api_key: if None, uses OPENSCORING_API_KEY, OCSAI_API_KEY, or legacy OSCAI_API_KEY.
    extra_params: merged into query string (e.g. task, prompt, question, language).
    """
    key = api_key
    if key is None:
        key = (
            os.environ.get("OPENSCORING_API_KEY")
            or os.environ.get("OCSAI_API_KEY")
            or os.environ.get("OSCAI_API_KEY")
        )

    params: dict[str, Any] = {}
    if model is not None:
        params["model"] = model
    if extra_params:
        params.update(extra_params)

    headers: dict[str, str] = {}
    if key:
        headers["X-API-KEY"] = key

    sess = session or requests.Session()
    # OpenScoring changed its /llm contract (observed 2026-07): the endpoint
    # no longer reads a JSON body or query params — it validates a
    # form-encoded body with `model` and `input` as one newline-joined blob
    # of CSV rows (quoted fields with embedded newlines parse correctly).
    r = sess.post(
        url,
        data={**params, "input": "\n".join(inputs)},
        headers=headers or None,
        timeout=timeout,
    )
    r.raise_for_status()
    return _parse_openscoring_json(r.text)
