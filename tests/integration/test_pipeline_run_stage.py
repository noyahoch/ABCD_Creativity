"""Integration tests for ``pipeline.run_stage`` and the strict/with-deps
prerequisite handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge_steering.pipeline import PipelineError, run_pipeline, run_stage, stage_is_done
from bridge_steering.repro import StageResult, build_run_context


# Insert a no-op fake stage so we can exercise run_stage without a model.
def _register_fake_stage(name: str, raises: bool = False, output: Path | None = None):
    from bridge_steering.stages import STAGES, DEFAULT_ORDER

    def fake_run(ctx):
        if raises:
            raise RuntimeError("boom")
        out = output or (ctx.stage_dir(name) / "fake.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("hello", encoding="utf-8")
        from bridge_steering.repro import hashed_output
        return StageResult(name=name, outputs=[hashed_output(out)])

    STAGES[name] = fake_run
    if name not in DEFAULT_ORDER:
        DEFAULT_ORDER.append(name)
    return name


def test_run_stage_writes_success_manifest(base_run_cfg, tmp_path):
    name = _register_fake_stage("fake_ok_unique_1")
    ctx = build_run_context(base_run_cfg)
    res = run_stage(name, ctx)
    assert res.name == name
    m = json.loads((ctx.stage_dir(name) / "_manifest.json").read_text())
    assert m["status"] == "success"
    assert m["stage"] == name
    assert m["outputs"]


def test_run_stage_writes_failed_manifest_on_exception(base_run_cfg):
    name = _register_fake_stage("fake_raises_unique_1", raises=True)
    ctx = build_run_context(base_run_cfg)
    with pytest.raises(RuntimeError, match="boom"):
        run_stage(name, ctx)
    m = json.loads((ctx.stage_dir(name) / "_manifest.json").read_text())
    assert m["status"] == "failed"
    assert "boom" in m["error"]


def test_resume_skips_done_stage(base_run_cfg):
    name = _register_fake_stage("fake_resume_unique_1")
    ctx = build_run_context(base_run_cfg)
    run_stage(name, ctx)
    assert stage_is_done(ctx, name)
    # A pipeline call with --only and --resume should treat it as done.
    results = run_pipeline(ctx, only=[name], resume=True)
    assert results == []


def test_resume_detects_tampered_output(base_run_cfg):
    name = _register_fake_stage("fake_tamper_unique_1")
    ctx = build_run_context(base_run_cfg)
    run_stage(name, ctx)
    # Tamper with the output file: sha mismatch should make stage_is_done == False
    out = ctx.stage_dir(name) / "fake.txt"
    out.write_text("CHANGED", encoding="utf-8")
    assert not stage_is_done(ctx, name)


def test_only_strict_fails_without_prereqs(base_run_cfg):
    """--only aut_judge with no aut_infer outputs ⇒ PipelineError, no work runs."""
    base_run_cfg.aut_judge = None   # config gate would skip; force enable
    from bridge_steering.config import JudgeConfig, VllmJudgeBackend
    base_run_cfg.aut_judge = JudgeConfig(
        backend=VllmJudgeBackend(),
        prompt_id="aut_judge_novelty_usability_v3",
    )
    ctx = build_run_context(base_run_cfg)
    with pytest.raises(PipelineError, match="missing prerequisites"):
        run_pipeline(ctx, only=["aut_judge"], with_deps=False)


def test_dry_run_prints_plan_without_executing(base_run_cfg, capsys):
    ctx = build_run_context(base_run_cfg)
    out = run_pipeline(ctx, only=["steer_collect"], dry_run=True)
    captured = capsys.readouterr().out
    assert "dry-run" in captured
    assert "steer_collect" in captured
    assert out == []
