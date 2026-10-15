#!/usr/bin/env python3
"""Fit a low-rank Gaussian over steering directions from collected activations.

Per probe layer L, the per-sample difference vectors d_i = B_i - mu_D form an
empirical distribution whose mean is exactly the mean_diff vector v_bridge[L].
We PCA those diffs and store {mu, components, scales} so a server can sample
on-manifold steering vectors v = mu + sum_j eps_j * scale_j * u_j, eps~N(0,tau^2).

Usage:
  python scripts/fit_steering_distribution.py \
      --activations <run>/steer_collect/activations.pt \
      --out <run>/steer_extract/steering_distribution.pt [--rank 16]
"""
import argparse
import hashlib
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    cache = torch.load(args.activations, map_location="cpu", weights_only=False)
    acts_B, acts_D = cache["acts_B"], cache["acts_D"]

    dist = {}
    for L in cache["probe_layers"]:
        B = torch.stack([a.float() for a in acts_B[L]])          # (n, d)
        D = torch.stack([a.float() for a in acts_D[L]])
        mu_D = D.mean(0)
        diffs = B - mu_D                                          # (n, d)
        mu = diffs.mean(0)                                        # == v_bridge[L] (mean_diff, beta=1)
        centered = diffs - mu
        # SVD of centered diffs: rows span the empirical variation
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        k = min(args.rank, S.shape[0])
        scales = S[:k] / (centered.shape[0] - 1) ** 0.5           # component std devs
        dist[L] = {"mu": mu, "components": Vh[:k], "scales": scales}
        var_kept = float((S[:k] ** 2).sum() / (S ** 2).sum())
        print(f"layer {L}: n={diffs.shape[0]} rank={k} "
              f"var_explained={var_kept:.2%} ||mu||={mu.norm():.3f} "
              f"top_scales={[round(float(s),3) for s in scales[:4]]}")

    out = {
        "distribution": dist,
        "rank": args.rank,
        "steer_target": cache.get("steer_target"),
        "steer_scale": cache.get("steer_scale"),
        "model_name": cache.get("model_name"),
        "source_activations": str(Path(args.activations).resolve()),
        "source_sha256": hashlib.sha256(Path(args.activations).read_bytes()).hexdigest(),
    }
    torch.save(out, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
