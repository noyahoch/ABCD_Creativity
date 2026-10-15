"""Unit tests for repro primitives."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from bridge_steering.repro import (
    StageResult,
    build_run_context,
    config_hash,
    git_sha,
    hashed_input,
    hashed_output,
    read_manifest,
    sha256_file,
    sha256_str,
    write_manifest,
)


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha256_file(p) == expected


def test_sha256_str_matches_hashlib():
    assert sha256_str("abc") == hashlib.sha256(b"abc").hexdigest()


def test_hashed_input_schema(tmp_path):
    p = tmp_path / "y.json"
    p.write_text("{}")
    info = hashed_input(p)
    assert set(info) == {"path", "sha256", "bytes"}
    assert info["sha256"].startswith("sha256:")
    assert info["bytes"] == 2


def test_hashed_output_optional_counts(tmp_path):
    p = tmp_path / "y.csv"
    p.write_text("a,b\n1,2\n")
    info = hashed_output(p, rows=1)
    assert info["rows"] == 1
    info2 = hashed_output(p, lines=2)
    assert info2["lines"] == 2


def test_config_hash_is_deterministic(base_run_cfg):
    h1 = config_hash(base_run_cfg)
    h2 = config_hash(base_run_cfg)
    assert h1 == h2 and h1.startswith("sha256:")


def test_config_hash_changes_when_seed_changes(base_run_cfg):
    h1 = config_hash(base_run_cfg)
    base_run_cfg.seed = 9999
    assert config_hash(base_run_cfg) != h1


def test_build_run_context_writes_frozen_config(base_run_cfg):
    ctx = build_run_context(base_run_cfg)
    assert ctx.run_dir.is_dir()
    frozen = ctx.run_dir / "_config_resolved.yaml"
    assert frozen.is_file()
    text = frozen.read_text()
    assert "run_id: test_run" in text


def test_write_manifest_records_required_fields(base_run_cfg):
    ctx = build_run_context(base_run_cfg)
    sd = ctx.stage_dir("steer")
    res = StageResult(
        name="steer",
        inputs={"abcd_data": {"path": "x", "sha256": "sha256:00"}},
        outputs=[{"path": "out.pt", "sha256": "sha256:01", "bytes": 1}],
        notes={"stage_config": {"k": "v"}, "model": {"name": "m"}, "decoding": {"t": 0}},
    )
    p = write_manifest(ctx, sd, res, status="success", command="cli steer")
    assert p.is_file()
    m = read_manifest(sd)
    assert m["stage"] == "steer"
    assert m["status"] == "success"
    assert m["config_hash"].startswith("sha256:")
    assert m["stage_config"] == {"k": "v"}
    assert m["outputs"][0]["path"] == "out.pt"
    # AUT prompts block is always present, even for non-AUT stages.
    assert "aut_prompts" in m
    assert m["aut_prompts"]["decoupled"] is False
    assert m["seeds"]["global"] == base_run_cfg.seed


def test_write_manifest_records_failure(base_run_cfg):
    ctx = build_run_context(base_run_cfg)
    sd = ctx.stage_dir("steer")
    res = StageResult(name="steer")
    write_manifest(ctx, sd, res, status="failed", error="boom")
    m = read_manifest(sd)
    assert m["status"] == "failed"
    assert m["error"] == "boom"


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not installed",
)
def test_git_sha_nonempty_in_repo():
    sha = git_sha(Path(__file__).resolve().parents[2])
    assert sha  # may be empty outside a git checkout, but in this repo it isn't
