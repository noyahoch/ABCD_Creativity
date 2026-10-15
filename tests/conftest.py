"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def abcd_tiny_path() -> Path:
    return FIXTURES / "abcd_tiny.json"


@pytest.fixture
def abcd_tiny_items() -> list[dict]:
    """3 AUT items, each with valid B_list and D_banal."""
    import json
    return json.loads((FIXTURES / "abcd_tiny.json").read_text())


@pytest.fixture
def base_run_cfg(tmp_path, abcd_tiny_path):
    """Minimal, valid RunConfig — coupled by default. Override fields as needed."""
    from bridge_steering.config import (
        AUTConfig, DecodingConfig, RunConfig, SteeringConfig,
    )

    return RunConfig(
        run_id="test_run",
        output_root=tmp_path / "runs",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        seed=42,
        aut=AUTConfig(abcd_data=abcd_tiny_path),
        steering=SteeringConfig(),
        decoding_aut=DecodingConfig(),
    )
