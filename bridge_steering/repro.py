"""Reproducibility primitives: git info, file hashing, RunContext, manifests.

Everything a stage might want to record about *how* it ran lives here, so
no stage has to reinvent it. The `run_stage` wrapper in
``bridge_steering.pipeline`` is the only caller; stages just fill a
:class:`StageResult` and the wrapper handles the rest.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import RunConfig, aut_prompt_pair, resolved_yaml, to_dict


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_str(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hashed_input(path: str | Path) -> dict:
    p = Path(path)
    return {"path": str(p), "sha256": "sha256:" + sha256_file(p), "bytes": p.stat().st_size}


def hashed_output(path: str | Path, rows: Optional[int] = None, lines: Optional[int] = None) -> dict:
    p = Path(path)
    entry: dict[str, Any] = {
        "path": str(p),
        "sha256": "sha256:" + sha256_file(p),
        "bytes": p.stat().st_size,
    }
    if rows is not None:
        entry["rows"] = rows
    if lines is not None:
        entry["lines"] = lines
    return entry


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ("git",) + args,
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_sha(cwd: Path | None = None) -> str:
    return _git("rev-parse", "--short", "HEAD", cwd=cwd)


def git_branch(cwd: Path | None = None) -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)


def git_dirty(cwd: Path | None = None) -> bool:
    return bool(_git("status", "--porcelain", cwd=cwd))


def git_info(cwd: Path | None = None) -> dict:
    return {
        "sha": git_sha(cwd),
        "branch": git_branch(cwd),
        "dirty": git_dirty(cwd),
    }


# ---------------------------------------------------------------------------
# Run context (computed once, passed through every stage)
# ---------------------------------------------------------------------------


def config_hash(cfg: RunConfig) -> str:
    """Stable sha256 of the resolved config dict (sorted keys)."""
    payload = json.dumps(to_dict(cfg), sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + sha256_bytes(payload)


@dataclass(frozen=True)
class RunContext:
    cfg: RunConfig
    run_dir: Path
    git_sha: str
    git_dirty: bool
    git_branch: str
    config_hash: str
    started_at: str

    def stage_dir(self, name: str) -> Path:
        return self.run_dir / name


def build_run_context(cfg: RunConfig, repo_root: Path | None = None) -> RunContext:
    """Build a :class:`RunContext` and create the run dir if missing."""
    g = git_info(repo_root)
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(
        cfg=cfg,
        run_dir=run_dir,
        git_sha=g["sha"],
        git_dirty=g["dirty"],
        git_branch=g["branch"],
        config_hash=config_hash(cfg),
        started_at=_utc_now(),
    )
    # Frozen config snapshot — every stage reads this, not the source YAML.
    (run_dir / "_config_resolved.yaml").write_text(resolved_yaml(cfg), encoding="utf-8")
    return ctx


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# StageResult + manifest
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Filled in by a stage's ``run(ctx)``. The wrapper writes the manifest."""

    name: str
    inputs: dict[str, dict] = field(default_factory=dict)
    outputs: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)
    extra_prompts: dict[str, Any] = field(default_factory=dict)  # judge_prompt etc.


def _aut_prompts_manifest(cfg: RunConfig) -> dict[str, Any]:
    """The ``aut_prompts`` block written into every stage manifest."""
    s, i, decoupled = aut_prompt_pair(cfg)
    return {
        "steering_collection": s.manifest_entry(),
        "inference": i.manifest_entry(),
        "decoupled": decoupled,
        "decoupling_reason": cfg.aut.prompts.decoupling_reason,
    }


def write_manifest(
    ctx: RunContext,
    stage_dir: Path,
    result: StageResult,
    status: str = "success",
    error: Optional[str] = None,
    command: Optional[str] = None,
) -> Path:
    """Write ``<stage_dir>/_manifest.json``. Returns the path."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": result.name,
        "status": status,
        "error": error,
        "started_utc": ctx.started_at,
        "ended_utc": _utc_now(),
        "elapsed_s": round(result.elapsed_s, 3),
        "git": {
            "sha": ctx.git_sha,
            "dirty": ctx.git_dirty,
            "branch": ctx.git_branch,
        },
        "run_id": ctx.cfg.run_id,
        "config_hash": ctx.config_hash,
        "stage_config": result.notes.get("stage_config", {}),
        "inputs": result.inputs,
        "outputs": result.outputs,
        "model": result.notes.get("model"),
        "decoding": result.notes.get("decoding"),
        "seeds": result.notes.get("seeds", {"global": ctx.cfg.seed}),
        "aut_prompts": _aut_prompts_manifest(ctx.cfg),
        "command": command or " ".join(sys.argv),
        "platform": {
            "python": platform.python_version(),
            "os": platform.platform(),
        },
        "notes": {k: v for k, v in result.notes.items()
                  if k not in ("stage_config", "model", "decoding", "seeds")},
    }
    manifest.update(result.extra_prompts)  # e.g. {"judge_prompt": {...}}

    out_path = stage_dir / "_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out_path


def read_manifest(stage_dir: Path) -> dict:
    return json.loads((stage_dir / "_manifest.json").read_text("utf-8"))
