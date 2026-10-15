"""Unit tests for the pure-Python helpers of stages/_denial_common.

Covers:
* ``parse_approaches`` — bullet/numbered prefix stripping, dedup, blank-line
  handling.
* ``format_constraints_block`` — empty-input contract, bullet formatting.
* ``per_item_rng`` — determinism across runs, independence across items/rounds.
* ``select_next_constraint`` — case-insensitive dedup, ``None`` when exhausted.
* Prompt registry — denial inference + reviewer specs are present, distinct,
  and hashed in the lockfile.
* Config wiring — RunConfig accepts a DenialConfig with default prompt ids.

No model is loaded; the per-round generate loop in ``denial_aut_infer.run`` /
``denial_macgyver_infer.run`` requires a real LLM and is left to integration
tests.
"""

from __future__ import annotations

import random

import pytest

from bridge_steering.config import (
    AUTConfig, DecodingConfig, DenialConfig, RunConfig, SteeringConfig,
    VllmJudgeBackend,
)
from bridge_steering.prompts import all_specs, load_lockfile
from bridge_steering.stages._denial_common import (
    ReviewerSystemicFailure,
    _TaggedList,
    format_constraints_block,
    parse_approaches,
    per_item_rng,
    review_batch,
    select_next_constraint,
)


# --------------------------------------------------------------------------- #
# parse_approaches
# --------------------------------------------------------------------------- #


def test_parse_approaches_strips_bullets_and_numbers():
    raw = "- use as a lever\n* hammer substitute\n• stir paint\n1. press flowers\n2) tie hair"
    out = parse_approaches(raw)
    assert out == [
        "use as a lever",
        "hammer substitute",
        "stir paint",
        "press flowers",
        "tie hair",
    ]


def test_parse_approaches_dedupes_case_insensitively():
    raw = "- Use Lever\n- use lever\n- USE LEVER\n- something else"
    out = parse_approaches(raw)
    assert out == ["Use Lever", "something else"]


def test_parse_approaches_drops_blank_lines_and_trailing_punct():
    raw = "\n- foo.\n\n- bar;\n  \n- baz,"
    assert parse_approaches(raw) == ["foo", "bar", "baz"]


def test_parse_approaches_empty_input_returns_empty_list():
    assert parse_approaches("") == []
    assert parse_approaches(None) == []  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# format_constraints_block
# --------------------------------------------------------------------------- #


def test_format_constraints_block_empty_returns_empty_string():
    # Contract: round-0 gating is the stage code's job; we must not
    # emit a dangling "Avoid the following approaches:" header here.
    assert format_constraints_block([]) == ""


def test_format_constraints_block_renders_bulleted_list():
    out = format_constraints_block(["use as a lever", "hammer substitute"])
    assert out == "Avoid the following approaches:\n- use as a lever\n- hammer substitute"


# --------------------------------------------------------------------------- #
# per_item_rng
# --------------------------------------------------------------------------- #


def test_per_item_rng_is_deterministic_across_calls():
    a = per_item_rng(42, "item_7", 3).random()
    b = per_item_rng(42, "item_7", 3).random()
    assert a == b


def test_per_item_rng_differs_across_items_rounds_seeds():
    base = per_item_rng(42, "item_7", 3).random()
    assert per_item_rng(42, "item_8", 3).random() != base
    assert per_item_rng(42, "item_7", 4).random() != base
    assert per_item_rng(43, "item_7", 3).random() != base


def test_per_item_rng_accepts_int_and_str_item_ids():
    # Cast-to-str is what the stages do; both forms must work.
    int_choice = per_item_rng(42, 7, 1).choice(["a", "b", "c"])
    str_choice = per_item_rng(42, "7", 1).choice(["a", "b", "c"])
    assert int_choice == str_choice


# --------------------------------------------------------------------------- #
# select_next_constraint
# --------------------------------------------------------------------------- #


def test_select_next_constraint_skips_existing_case_insensitively():
    rng = random.Random(0)
    pick = select_next_constraint(["Use Lever", "use lever", "novel idea"],
                                  existing=["use lever"], rng=rng)
    assert pick == "novel idea"


def test_select_next_constraint_returns_none_when_all_seen():
    rng = random.Random(0)
    pick = select_next_constraint(["a", "b"], existing=["a", "B"], rng=rng)
    assert pick is None


def test_select_next_constraint_is_reproducible_with_seeded_rng():
    cands = ["one", "two", "three", "four"]
    rng1 = random.Random("k|abc|1")
    rng2 = random.Random("k|abc|1")
    assert select_next_constraint(cands, [], rng1) == select_next_constraint(cands, [], rng2)


# --------------------------------------------------------------------------- #
# Prompt registry sanity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pid", [
    "aut_user_a_paper_infer_denial_v1",
    "aut_reviewer_approaches_v1",
    "macgyver_user_default_denial_v1",
    "macgyver_reviewer_approaches_v1",
])
def test_denial_prompts_registered_and_active(pid):
    specs = all_specs()
    assert pid in specs, f"missing prompt {pid} in registry"
    assert not specs[pid].archived, f"prompt {pid} is archived"


def test_denial_prompts_in_lockfile():
    lock = load_lockfile()
    for pid in ("aut_user_a_paper_infer_denial_v1",
                "aut_reviewer_approaches_v1",
                "macgyver_user_default_denial_v1",
                "macgyver_reviewer_approaches_v1"):
        assert pid in lock, f"prompt {pid} missing from prompt_lock.json"


def test_denial_renders_include_constraints_block():
    from bridge_steering.prompts import resolve
    aut = resolve("aut_user_a_paper_infer_denial_v1")
    msgs = aut.render(
        full_instruction="Give 8 uses for a brick.",
        constraints_block=format_constraints_block(["use as a lever"]),
    )
    user = msgs[-1]["content"]
    assert "Avoid the following approaches:" in user
    assert "- use as a lever" in user

    mac = resolve("macgyver_user_default_denial_v1")
    msgs = mac.render(
        problem="You need to open a stuck jar with...",
        constraints_block=format_constraints_block(["use as a lever"]),
    )
    user = msgs[-1]["content"]
    assert "Avoid the following approaches:" in user
    assert "- use as a lever" in user


# --------------------------------------------------------------------------- #
# Config wiring
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# review_batch: systemic-failure detection (the bug we just hit in production)
# --------------------------------------------------------------------------- #


def test_review_batch_raises_when_all_calls_raised():
    """If every per-item call raised (e.g. vLLM down → connection refused),
    review_batch must raise ReviewerSystemicFailure instead of silently
    returning a list of empty constraint lists."""
    import asyncio
    from unittest.mock import patch
    from bridge_steering.stages import _denial_common as dc

    cfg = DenialConfig()  # default backend is vLLM at localhost:8001

    # Stand in for _review_one: every call returns a _TaggedList carrying a
    # connection-refused exception, mimicking the failure mode from the
    # broken first run (vLLM not yet started by the pipeline).
    async def fake_review_one(client, model, dcfg, spec, text, var):
        return _TaggedList([], last_exc=ConnectionRefusedError("port 8000"))

    class _FakeSpec:
        def render(self, **kw): return [{"role": "user", "content": ""}]

    with patch.object(dc, "_review_one", side_effect=fake_review_one), \
         patch.object(dc, "make_judge_client", return_value=None), \
         patch.object(dc, "judge_model_name", return_value="fake"):
        with pytest.raises(ReviewerSystemicFailure) as ei:
            asyncio.run(review_batch(cfg, _FakeSpec(), "x", ["r1", "r2", "r3"]))
    msg = str(ei.value)
    assert "All 3 reviewer calls failed" in msg
    assert "ConnectionRefusedError" in msg or "port 8000" in msg


def test_review_batch_tolerates_some_empty_parses():
    """A plain empty parse from a SUCCESSFUL HTTP response (no exception)
    is fine — it just means that one item had no extractable approaches."""
    import asyncio
    from unittest.mock import patch
    from bridge_steering.stages import _denial_common as dc

    cfg = DenialConfig()
    results_to_return = [["a", "b"], [], ["c"]]  # second item: clean empty

    async def fake_review_one(client, model, dcfg, spec, text, var):
        # Return a plain list (not _TaggedList) — signals "HTTP OK, parse OK,
        # just nothing to extract", which review_batch must accept.
        return results_to_return.pop(0)

    class _FakeSpec:
        def render(self, **kw): return [{"role": "user", "content": ""}]

    with patch.object(dc, "_review_one", side_effect=fake_review_one), \
         patch.object(dc, "make_judge_client", return_value=None), \
         patch.object(dc, "judge_model_name", return_value="fake"):
        out = asyncio.run(review_batch(cfg, _FakeSpec(), "x", ["r1", "r2", "r3"]))
    assert out == [["a", "b"], [], ["c"]]


# --------------------------------------------------------------------------- #
# Pipeline wiring: denial stages count as vLLM stages
# --------------------------------------------------------------------------- #


def test_denial_stages_skipped_when_cfg_denial_is_none():
    """DEFAULT_ORDER includes denial_aut_infer + denial_macgyver_infer, so
    any full-pipeline run on a config without ``denial:`` would crash inside
    the stage at ``cfg.denial is None`` — destroying all prior HF/judge work.
    _filter_to_enabled must skip both denial stages when cfg.denial is None.

    Regression test for the bug that crashed `D_generic_s42` at stage 6 of
    DEFAULT_ORDER after burning through steer_collect → steer_extract →
    aut_infer → macgyver_infer (~25-30 min of inference)."""
    from bridge_steering.config import DecodingConfig
    from bridge_steering.pipeline import _filter_to_enabled

    cfg_no_denial = RunConfig(
        run_id="t", output_root="/tmp/t", model_name="m", seed=0,
        aut=AUTConfig(abcd_data="data/aut_bridgeset.json"),
        steering=SteeringConfig(),
        decoding_aut=DecodingConfig(),
        denial=None,
    )
    full = ["steer_collect", "steer_extract", "aut_infer",
            "denial_aut_infer", "denial_macgyver_infer", "aggregate"]
    filtered = _filter_to_enabled(cfg_no_denial, full)
    assert "denial_aut_infer" not in filtered
    assert "denial_macgyver_infer" not in filtered


def test_denial_macgyver_skipped_when_macgyver_data_missing():
    """denial_macgyver_infer requires macgyver_data even if cfg.denial is set.
    Mirrors the macgyver_infer gate."""
    from bridge_steering.config import DecodingConfig
    from bridge_steering.pipeline import _filter_to_enabled

    cfg = RunConfig(
        run_id="t", output_root="/tmp/t", model_name="m", seed=0,
        aut=AUTConfig(abcd_data="data/aut_bridgeset.json"),
        steering=SteeringConfig(),
        decoding_aut=DecodingConfig(),
        denial=DenialConfig(),
        macgyver_data=None,
    )
    filtered = _filter_to_enabled(cfg, ["denial_aut_infer", "denial_macgyver_infer"])
    assert filtered == ["denial_aut_infer"]


def test_denial_stages_kept_when_fully_configured():
    """Sanity: with cfg.denial set + cfg.macgyver_data set, both denial
    stages survive the filter."""
    from bridge_steering.config import DecodingConfig
    from bridge_steering.pipeline import _filter_to_enabled

    cfg = RunConfig(
        run_id="t", output_root="/tmp/t", model_name="m", seed=0,
        aut=AUTConfig(abcd_data="data/aut_bridgeset.json"),
        steering=SteeringConfig(),
        decoding_aut=DecodingConfig(),
        denial=DenialConfig(),
        macgyver_data="data/macgyver/problem_solution_pair.xlsx",
    )
    filtered = _filter_to_enabled(cfg, ["denial_aut_infer", "denial_macgyver_infer"])
    assert filtered == ["denial_aut_infer", "denial_macgyver_infer"]


def test_denial_stages_are_not_pipeline_vllm_stages():
    """Since 2026-07-19 the denial stages manage their own EPHEMERAL judge
    (generator unloaded around each reviewer round), so the pipeline-level
    vLLM lifecycle must NOT include them — a pipeline-lifetime server would
    co-reside with the HF generator, which a 24B model cannot afford (KV
    starvation on a 45 GiB card). History: originally they were excluded by
    accident (the silent all-empty-constraints bug), then included, now
    excluded again BY DESIGN — the reviewer's server comes from managed_vllm()
    called inside the stage itself."""
    from bridge_steering.config import DecodingConfig, JudgeConfig
    from bridge_steering.pipeline import _vllm_stages_in

    cfg = RunConfig(
        run_id="t", output_root="/tmp/t", model_name="m", seed=0,
        aut=AUTConfig(abcd_data="data/aut_bridgeset.json"),
        steering=SteeringConfig(),
        decoding_aut=DecodingConfig(),
        denial=DenialConfig(),
        aut_judge=JudgeConfig(prompt_id="aut_judge_creativity_usability_v1",
                              backend=VllmJudgeBackend()),
    )
    stages = ["aut_infer", "denial_aut_infer", "aut_ocsai", "aut_judge",
              "aggregate"]
    idxs = _vllm_stages_in(cfg, stages)
    assert [stages[i] for i in idxs] == ["aut_judge"], \
        "denial stages must NOT be pipeline vLLM stages (they run their own " \
        "ephemeral judge with the generator unloaded)"


def test_runconfig_accepts_denial_block():
    cfg = RunConfig(
        run_id="t", output_root="/tmp/t", model_name="m", seed=0,
        aut=AUTConfig(abcd_data="data/aut_bridgeset.json"),
        steering=SteeringConfig(),
        decoding_aut=DecodingConfig(),
        denial=DenialConfig(rounds=3),
    )
    assert cfg.denial is not None
    assert cfg.denial.rounds == 3
    assert cfg.denial.aut_denial_prompt_id == "aut_user_a_paper_infer_denial_v1"
    assert cfg.denial.macgyver_denial_prompt_id == "macgyver_user_default_denial_v1"
    assert cfg.denial.reviewer_aut_prompt_id == "aut_reviewer_approaches_v1"
    assert cfg.denial.reviewer_macgyver_prompt_id == "macgyver_reviewer_approaches_v1"
    assert isinstance(cfg.denial.reviewer_backend, VllmJudgeBackend)
