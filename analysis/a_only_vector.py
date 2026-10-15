"""Build the A-only reference steering vector for Llama-3.1-8B-Instruct.

Definition: for every construction object (the 110 training items of
the paper's split, same bridge prompt and the same ``bullets_rule_unlocks`` B completion as
the deployed B-D vector), run ONE teacher-forced pass over A+B and take, per layer 14-19,

    v_L = mean_i [ mean_{t in B tokens} h_L(t) - mean_{t in A tokens} h_L(t) ]

where "B tokens" are the assistant completion tokens (exactly the span the paper averages
for mu_B) and "A tokens" are the tokens of the user message content (the anchor instruction
rendered by the ``aut_user_a_paper_bridge`` template), located in the same forward pass.

Two variants are saved in the steer_extract checkpoint schema so the unmodified pipeline
stages (SteeredMethod) can consume them:
  * a_only_raw.pt      : v_L as defined above
  * a_only_renorm.pt   : v_L rescaled per layer to ||v_{B-D,L}|| of the deployed alpha=0.17 run

Sanity check: the B-side centroid collected here must equal the paper's stored centroids_B.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from .registry import REPO, RESULTS_ROOT

REF_CFG = REPO / "configs" / "ablations" / "llama" / "bridge_a017_s42.yaml"
REF_RUN = RESULTS_ROOT / "ablations" / "llama" / "bridge_a017_s42"   # run REF_CFG first
REF_VEC = REF_RUN / "steer_extract" / "steering_vectors.pt"
VEC_DIR = RESULTS_ROOT / "ablations" / "llama" / "a_only_vectors"


def _find_subsequence(hay: list[int], needle: list[int]) -> int:
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return -1


def anchor_positions(full_ids: list[int], assistant_start: int, user_str: str, tokenizer) -> tuple[list[int], str]:
    """Token positions of the user-message content inside the full sequence."""
    needle = tokenizer(user_str, add_special_tokens=False)["input_ids"]
    i = _find_subsequence(full_ids[:assistant_start], needle)
    if i >= 0:
        return list(range(i, i + len(needle))), "user_message_content"
    # tolerate a boundary merge at either end
    for a, b in ((1, 0), (0, 1), (1, 1)):
        sub = needle[a:len(needle) - b]
        i = _find_subsequence(full_ids[:assistant_start], sub)
        if i >= 0:
            return list(range(i, i + len(sub))), f"user_message_content_trimmed_{a}_{b}"
    return list(range(0, assistant_start)), "whole_prompt_fallback"


def main() -> int:
    from bridge_steering.config import load_run_config
    from bridge_steering.datasets import ABCDDataset
    from bridge_steering.stages.steer_collect import _tokenize_full_sequence, build_pairs
    from bridge_steering.models import HFLoader
    from bridge_steering.steering import ResidualStreamCache
    from bridge_steering.steering.completion_sources import SOURCE_REGISTRY
    from bridge_steering.data_prep import TaskType

    t0 = time.time()
    cfg = load_run_config(REF_CFG)
    dataset = ABCDDataset.load(cfg.aut.abcd_data)
    split = dataset.split(cfg.aut.eval_seed, cfg.aut.eval_holdout)
    train_pairs = build_pairs(split.train_items, cfg)
    assert len(train_pairs) == 110, len(train_pairs)
    source = SOURCE_REGISTRY[cfg.steering.bridge_source]({**cfg.steering.source_kwargs, **cfg.steering.method_kwargs})
    source.setup(train_pairs, TaskType.AUT)
    layers = list(cfg.steering.probe_layers)

    ref = torch.load(REF_VEC, map_location="cpu", weights_only=False)
    assert ref["probe_layers"] == layers and ref["train_indices"] == split.train_indices

    model, tokenizer = HFLoader(model_name_or_path=cfg.model_name).load()
    model.eval()
    device = next(model.parameters()).device
    cache = ResidualStreamCache(model, layers, steer_target=cfg.steering.steer_target).register()
    acts_B = {L: [] for L in layers}
    acts_A = {L: [] for L in layers}
    anchor_kinds, n_anchor_tokens, n_b_tokens = [], [], []
    try:
        with torch.no_grad():
            for p in train_pairs:
                b_texts = source.get_b_completions(p, model, tokenizer)
                assert len(b_texts) == 1
                tok = _tokenize_full_sequence(p["prompt_msgs"], b_texts[0], tokenizer, cfg.steering.collect_span)
                full = tok["full_ids"]
                start, n_comp = tok["assistant_start"], tok["n_completion_tokens"]
                a_pos, kind = anchor_positions(full[0].tolist(), start, p["user_prompt"], tokenizer)
                anchor_kinds.append(kind); n_anchor_tokens.append(len(a_pos)); n_b_tokens.append(n_comp)
                model(full.to(device), use_cache=False)
                b_pos = list(range(start, start + n_comp))
                for L in layers:
                    acts_B[L].append(cache.get_positions(L, b_pos).mean(dim=0).cpu())
                    acts_A[L].append(cache.get_positions(L, a_pos).mean(dim=0).cpu())
                cache.clear()
    finally:
        cache.remove()

    dtype = ref["v_bridge"][layers[0]].dtype
    v_raw, v_renorm, stats = {}, {}, []
    for L in layers:
        mu_B = torch.stack(acts_B[L]).float().mean(0)
        mu_A = torch.stack(acts_A[L]).float().mean(0)
        v = mu_B - mu_A
        v_bd = ref["v_bridge"][L].float()
        ref_mu_B = ref["centroids_B"][L].float()
        scale = v_bd.norm() / v.norm()
        v_raw[L] = v.to(dtype)
        v_renorm[L] = (v * scale).to(dtype)
        stats.append(dict(layer=L, norm_v_a_only=round(v.norm().item(), 4), norm_v_bd=round(v_bd.norm().item(), 4),
                          renorm_scale=round(scale.item(), 4),
                          cos_a_only_vs_bd=round(torch.nn.functional.cosine_similarity(v, v_bd, dim=0).item(), 4),
                          norm_mu_B=round(mu_B.norm().item(), 3), norm_mu_A=round(mu_A.norm().item(), 3),
                          max_abs_diff_mu_B_vs_paper=float((mu_B - ref_mu_B).abs().max().item())))
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    common = {k: ref.get(k) for k in ("probe_layers", "window", "collect_span", "steer_target", "steer_scale",
                                      "model_name", "abcd_data_path", "abcd_sha256", "eval_indices", "train_indices",
                                      "eval_seed", "eval_holdout", "aut_prompt_id", "aut_prompt_sha256")}
    for name, vb in (("a_only_raw", v_raw), ("a_only_renorm", v_renorm)):
        torch.save({**common, "steer_layer": int(ref["steer_layer"]), "v_steer": vb[int(ref["steer_layer"])].clone(),
                    "v_bridge": vb, "method": name, "bridge_source": ref.get("bridge_source"),
                    "default_source": "A_anchor_tokens_same_pass",
                    "definition": __doc__, "anchor_kinds": anchor_kinds,
                    "reference_vectors": str(REF_VEC)}, VEC_DIR / f"{name}.pt")
    with open(VEC_DIR / "stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
    meta = dict(elapsed_s=round(time.time() - t0, 1), n_pairs=len(train_pairs),
                anchor_kinds={k: anchor_kinds.count(k) for k in set(anchor_kinds)},
                anchor_tokens_mean=sum(n_anchor_tokens) / len(n_anchor_tokens),
                b_tokens_mean=sum(n_b_tokens) / len(n_b_tokens),
                files={n: hashlib.sha256((VEC_DIR / f"{n}.pt").read_bytes()).hexdigest() for n in ("a_only_raw", "a_only_renorm")},
                reference=str(REF_VEC), stats=stats)
    (VEC_DIR / "_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
