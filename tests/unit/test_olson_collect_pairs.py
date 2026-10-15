"""Unit tests for the pure-Python helpers of stages/olson_collect.

Covers:
* Prompt registry — Olson creative + boring specs are present, distinct, and
  hashed in the lockfile.
* ``build_olson_pairs`` — two rendered prompts per item, both contain the
  object, both come from the right spec, instruction prefixes differ.
* Config wiring — RunConfig accepts an OlsonConfig and steer_extract picks
  the Olson collector when cfg.olson is set.

No model is loaded; ``collect_olson_activations`` and ``run`` require a
real LLM and are covered by integration tests (run separately).
"""

from __future__ import annotations

import pytest

from bridge_steering.config import (
    AUTConfig, DecodingConfig, OlsonConfig, RunConfig, SteeringConfig,
)
from bridge_steering.prompts import all_specs, load_lockfile, resolve
from bridge_steering.prompts.aut import AUT_OLSON_BORING_V1, AUT_OLSON_CREATIVE_V1
from bridge_steering.stages.olson_collect import build_olson_pairs


# --------------------------------------------------------------------------- #
# Prompt registry sanity
# --------------------------------------------------------------------------- #


def test_olson_prompts_in_registry():
    specs = all_specs()
    assert "aut_olson_creative_v1" in specs
    assert "aut_olson_boring_v1" in specs
    assert not specs["aut_olson_creative_v1"].archived
    assert not specs["aut_olson_boring_v1"].archived


def test_olson_prompts_in_lockfile():
    lock = load_lockfile()
    assert lock["aut_olson_creative_v1"].startswith("sha256:")
    assert lock["aut_olson_boring_v1"].startswith("sha256:")
    # They must be distinct hashes — otherwise the templates collide.
    assert lock["aut_olson_creative_v1"] != lock["aut_olson_boring_v1"]


def test_olson_prompts_resolvable_via_id():
    s = resolve("aut_olson_creative_v1")
    assert s.id == "aut_olson_creative_v1"
    s = resolve("aut_olson_boring_v1")
    assert s.id == "aut_olson_boring_v1"


def test_olson_creative_has_creativity_instruction():
    """The creative prompt must contain a 'creative' / 'unconventional' cue."""
    txt = AUT_OLSON_CREATIVE_V1.user_template.lower()
    assert "creative" in txt or "unconventional" in txt


def test_olson_boring_has_boring_instruction():
    """The boring prompt must contain a 'conventional' / 'ordinary' cue."""
    txt = AUT_OLSON_BORING_V1.user_template.lower()
    assert "conventional" in txt or "ordinary" in txt


def test_olson_prompts_share_object_placeholder():
    """Both must take ``{object}`` so they accept the same per-item value."""
    assert "{object}" in AUT_OLSON_CREATIVE_V1.user_template
    assert "{object}" in AUT_OLSON_BORING_V1.user_template


# --------------------------------------------------------------------------- #
# build_olson_pairs
# --------------------------------------------------------------------------- #


def test_build_olson_pairs_renders_both_sides(abcd_tiny_items):
    pairs = build_olson_pairs(
        abcd_tiny_items, AUT_OLSON_CREATIVE_V1, AUT_OLSON_BORING_V1,
    )
    assert len(pairs) == len(abcd_tiny_items)
    p0 = pairs[0]
    # Both sides rendered as chat-message lists
    assert isinstance(p0["creative_msgs"], list) and p0["creative_msgs"]
    assert isinstance(p0["boring_msgs"], list) and p0["boring_msgs"]
    # Both contain the object (resolved from A_OBJECT / parsed-from-A)
    assert "brick" in p0["creative_user"]
    assert "brick" in p0["boring_user"]
    # Provenance fields propagated
    assert p0["id"] == "AUT_TEST_001"
    assert p0["_creative_spec"].id == AUT_OLSON_CREATIVE_V1.id
    assert p0["_boring_spec"].id == AUT_OLSON_BORING_V1.id


def test_build_olson_pairs_instruction_differs(abcd_tiny_items):
    """The two prompts for the same item must differ on the instruction
    prefix — that *is* the contrast."""
    pairs = build_olson_pairs(
        abcd_tiny_items, AUT_OLSON_CREATIVE_V1, AUT_OLSON_BORING_V1,
    )
    p0 = pairs[0]
    assert p0["creative_user"] != p0["boring_user"]
    # Sanity: the actual instruction words appear on each side.
    assert "creative" in p0["creative_user"].lower() or \
           "unconventional" in p0["creative_user"].lower()
    assert "conventional" in p0["boring_user"].lower() or \
           "ordinary" in p0["boring_user"].lower()


def test_build_olson_pairs_object_resolves_from_A_when_missing_A_OBJECT(
    abcd_tiny_items,
):
    """One fixture item (AUT_TEST_002) has no A_OBJECT — must still render
    with the parsed object from A (paperclip)."""
    item_002 = [it for it in abcd_tiny_items if it["id"] == "AUT_TEST_002"][0]
    pairs = build_olson_pairs(
        [item_002], AUT_OLSON_CREATIVE_V1, AUT_OLSON_BORING_V1,
    )
    p = pairs[0]
    # Either rendered object is "paperclip" (parsed) or contains paperclip.
    assert "paperclip" in p["creative_user"].lower()
    assert "paperclip" in p["boring_user"].lower()


def test_build_olson_pairs_boring_side_swaps_unconventional_via_full_instruction(
    abcd_tiny_items,
):
    """On the BORING side, the dataset's A field — which contains the literal
    word 'unconventional' — must be rewritten before {full_instruction} is
    substituted into the boring template. Tested with v2_paper which uses
    {full_instruction}; v1 (which only uses {object}) is unaffected."""
    from bridge_steering.prompts.aut import (
        AUT_OLSON_BORING_V2_PAPER, AUT_OLSON_CREATIVE_V2_PAPER,
    )

    pairs = build_olson_pairs(
        abcd_tiny_items, AUT_OLSON_CREATIVE_V2_PAPER, AUT_OLSON_BORING_V2_PAPER,
    )
    p0 = pairs[0]
    # The dataset's A for AUT_TEST_001 contains "unconventional" — that word
    # should appear unchanged on the creative side, and be rewritten on the
    # boring side.
    assert "unconventional" in p0["creative_user"]
    assert "unconventional" not in p0["boring_user"]
    # The boring side replaces "unconventional" with "conventional" inside
    # the embedded instruction.
    assert "conventional" in p0["boring_user"]


def test_build_olson_pairs_boring_side_swaps_unusual_via_full_instruction():
    """Items whose A field uses 'unusual' (not 'unconventional') must also
    have that marker rewritten on the boring side."""
    from bridge_steering.prompts.aut import (
        AUT_OLSON_BORING_V2_PAPER, AUT_OLSON_CREATIVE_V2_PAPER,
    )

    item = {
        "id": "AUT_TEST_UNUSUAL_001",
        "domain": "AUT",
        "A": "Give 8 unusual uses for a teacup focusing on bounded volume.",
        "A_OBJECT": "teacup",
        "D_banal": ["drink tea"],
        "B_list": [{"type": "x", "rule": "x", "unlocks": "x"}],
    }
    pairs = build_olson_pairs(
        [item], AUT_OLSON_CREATIVE_V2_PAPER, AUT_OLSON_BORING_V2_PAPER,
    )
    p = pairs[0]
    assert "unusual" in p["creative_user"]
    assert "unusual" not in p["boring_user"]
    assert "ordinary" in p["boring_user"]


def test_build_olson_pairs_v1_unaffected_by_substitution(abcd_tiny_items):
    """v1 templates only use {object} — the boring-side A substitution
    happens but has no visible effect on the rendered prompt."""
    pairs = build_olson_pairs(
        abcd_tiny_items, AUT_OLSON_CREATIVE_V1, AUT_OLSON_BORING_V1,
    )
    p0 = pairs[0]
    # v1 boring prompt: "Be conventional...Write 8 uses for {object}..."
    # — neither "unconventional" nor "conventional" should appear since
    # v1's body doesn't include the A field.
    assert "unconventional" not in p0["boring_user"]
    assert "conventional" in p0["boring_user"]  # from the "Be conventional" prefix


# --------------------------------------------------------------------------- #
# Config + pipeline wiring
# --------------------------------------------------------------------------- #


def test_olson_config_optional_on_run_config(tmp_path, abcd_tiny_path):
    """RunConfig accepts cfg.olson=None (default) and cfg.olson=OlsonConfig()."""
    cfg = RunConfig(
        run_id="t",
        output_root=tmp_path / "runs",
        model_name="m",
        seed=1,
        aut=AUTConfig(abcd_data=abcd_tiny_path),
        steering=SteeringConfig(),
        decoding_aut=DecodingConfig(),
    )
    assert cfg.olson is None  # default

    cfg2 = RunConfig(
        run_id="t",
        output_root=tmp_path / "runs",
        model_name="m",
        seed=1,
        aut=AUTConfig(abcd_data=abcd_tiny_path),
        steering=SteeringConfig(),
        decoding_aut=DecodingConfig(),
        olson=OlsonConfig(),
    )
    assert cfg2.olson is not None
    assert cfg2.olson.creative_prompt_id == "aut_olson_creative_v1"
    assert cfg2.olson.boring_prompt_id == "aut_olson_boring_v1"


def test_steer_extract_picks_olson_collector_when_cfg_olson_set(
    tmp_path, abcd_tiny_path,
):
    """When cfg.olson is set and olson_collect output is missing, steer_extract
    raises pointing at the *olson_collect* path (not steer_collect)."""
    from bridge_steering.repro import build_run_context
    from bridge_steering.stages.steer_extract import run

    cfg = RunConfig(
        run_id="olson_pick",
        output_root=tmp_path / "runs",
        model_name="m",
        seed=1,
        aut=AUTConfig(abcd_data=abcd_tiny_path),
        steering=SteeringConfig(),
        decoding_aut=DecodingConfig(),
        olson=OlsonConfig(),
    )
    ctx = build_run_context(cfg)
    with pytest.raises(FileNotFoundError, match="olson_collect"):
        run(ctx)


def test_steer_extract_picks_steer_collect_when_olson_unset(
    tmp_path, abcd_tiny_path,
):
    """Default path: cfg.olson=None ⇒ steer_extract reads steer_collect."""
    from bridge_steering.repro import build_run_context
    from bridge_steering.stages.steer_extract import run

    cfg = RunConfig(
        run_id="bridge_pick",
        output_root=tmp_path / "runs",
        model_name="m",
        seed=1,
        aut=AUTConfig(abcd_data=abcd_tiny_path),
        steering=SteeringConfig(),
        decoding_aut=DecodingConfig(),
    )
    ctx = build_run_context(cfg)
    with pytest.raises(FileNotFoundError, match="steer_collect"):
        run(ctx)


def test_olson_collect_registered_in_stages():
    """The new stage is discoverable via STAGES dict."""
    from bridge_steering.stages import STAGES, PREREQUISITES

    assert "olson_collect" in STAGES
    assert callable(STAGES["olson_collect"])
    assert "olson_collect" in PREREQUISITES
    # No prerequisites — it's a leaf collector like steer_collect.
    assert PREREQUISITES["olson_collect"] == ()


# --------------------------------------------------------------------------- #
# Integration: steer_extract actually consumes an Olson activations.pt
#
# This proves the schema we save matches what steer_extract / MeanDiffExtractor
# expects — the integration point I couldn't otherwise verify without a GPU.
# --------------------------------------------------------------------------- #


def _write_olson_acts_pt(ctx, layers, acts_B, acts_D, abcd_path):
    """Write a hand-crafted activations.pt in the Olson schema."""
    import torch

    olson_dir = ctx.stage_dir("olson_collect")
    olson_dir.mkdir(parents=True, exist_ok=True)
    out = olson_dir / "activations.pt"
    torch.save({
        "acts_B": acts_B, "acts_D": acts_D, "probe_layers": layers,
        "steer_target": "residual", "steer_scale": "fixed",
        "window": 48, "collect_span": "assistant_full",
        "collector": "olson",
        "bridge_source": "olson_creative",
        "default_source": "olson_boring",
        "source_kwargs": {
            "creative_prompt_id": "aut_olson_creative_v1",
            "boring_prompt_id": "aut_olson_boring_v1",
        },
        "model_name": "fake-model",
        "abcd_data_path": str(abcd_path),
        "abcd_sha256": "deadbeef",
        "eval_indices": [0], "train_indices": [1, 2],
        "eval_seed": 7, "eval_holdout": 1,
        "aut_prompt_id": "aut_olson_creative_v1",
        "aut_prompt_sha256": "sha256:abc",
        "boring_prompt_id": "aut_olson_boring_v1",
        "boring_prompt_sha256": "sha256:def",
    }, out)
    return out


def test_steer_extract_reads_olson_activations_end_to_end(
    tmp_path, abcd_tiny_path,
):
    """Write a hand-crafted Olson activations.pt, run steer_extract, verify
    the resulting steering_vectors.pt has the expected μ_creative − μ_boring."""
    import torch

    from bridge_steering.repro import build_run_context
    from bridge_steering.stages.steer_extract import run

    cfg = RunConfig(
        run_id="olson_e2e",
        output_root=tmp_path / "runs",
        model_name="fake-model",
        seed=1,
        aut=AUTConfig(abcd_data=abcd_tiny_path),
        steering=SteeringConfig(method="mean_diff"),
        decoding_aut=DecodingConfig(),
        olson=OlsonConfig(),
    )
    ctx = build_run_context(cfg)

    layers = [12, 13]
    # creative side centroid (1,1,1,1) at L12, (2,0,0,0) at L13
    # boring side centroid (1,0,0,0) at L12, (0,0,0,0) at L13
    # ⇒ v_olson = (0,1,1,1) at L12 ; (2,0,0,0) at L13
    acts_B = {12: [torch.tensor([1.0, 1.0, 1.0, 1.0]) for _ in range(3)],
              13: [torch.tensor([2.0, 0.0, 0.0, 0.0]) for _ in range(3)]}
    acts_D = {12: [torch.tensor([1.0, 0.0, 0.0, 0.0]) for _ in range(3)],
              13: [torch.tensor([0.0, 0.0, 0.0, 0.0]) for _ in range(3)]}
    _write_olson_acts_pt(ctx, layers, acts_B, acts_D, cfg.aut.abcd_data)

    result = run(ctx)
    assert result.name == "steer_extract"
    pt_path = ctx.stage_dir("steer_extract") / "steering_vectors.pt"
    state = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Same arithmetic as bridge — the recipe doesn't care which collector
    # produced the activations.
    assert torch.allclose(state["v_bridge"][12], torch.tensor([0.0, 1.0, 1.0, 1.0]))
    assert torch.allclose(state["v_bridge"][13], torch.tensor([2.0, 0.0, 0.0, 0.0]))
    # Provenance from the Olson side is preserved in the extract output.
    assert state["aut_prompt_id"] == "aut_olson_creative_v1"
