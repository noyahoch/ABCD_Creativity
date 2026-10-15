#!/usr/bin/env python3
"""OpenAI-compatible /v1/chat/completions server for a steered HF model.

Bridges the activation-steering inference path (bridge_steering.steering.steered_generate,
the same code aut_infer uses) to external OpenAI-protocol benchmarks such as
EQ-Bench creative-writing-bench, which can only talk to a chat/completions URL.

Requests are served one generation at a time (a lock guards the model — the
steering hooks are stateful per forward pass), so keep the client's thread
count modest (~4).

Usage (LiveIdeaBench, Llama deployed configuration, stochastic variant tau=1):
  python scripts/fit_steering_distribution.py \
      --activations results/main/llama/bridge_steering_s42/steer_collect/activations.pt \
      --out results/main/llama/bridge_steering_s42/steer_extract/steering_distribution.pt
  python scripts/serve_steered_openai.py \
      --vectors results/main/llama/bridge_steering_s42/steer_extract/steering_vectors.pt \
      --distribution results/main/llama/bridge_steering_s42/steer_extract/steering_distribution.pt \
      --model meta-llama/Llama-3.1-8B-Instruct --layers 14 15 16 17 18 19 \
      --alpha 0.20 --tau 1.0 --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bridge_steering.models.hf_loader import HFLoader  # noqa: E402
from bridge_steering.steering import steered_generate, steered_generate_batch  # noqa: E402

STATE: dict = {}
GEN_LOCK = threading.Lock()


def sample_pairs(tau=None):
    """Per-request steering vectors.

    Fixed mode (default): the checkpoint's v_bridge vectors, unchanged.
    Distribution mode (--distribution): one shared eps ~ N(0, tau^2 I_k) draw
    across layers (layer diffs are correlated within a contrast pair), applied
    to each layer's PCA basis, then renormalized to ||mu|| so the effective
    dose alpha*||v|| matches the fixed-vector setup.

    tau may be overridden per request (screening sweeps); tau=0 reproduces the
    fixed vector exactly.
    """
    if STATE.get("dist") is None:
        return STATE["pairs"]
    tau = STATE["tau"] if tau is None else tau
    if tau == 0:
        return STATE["pairs"]
    eps = torch.randn(STATE["dist_rank"]) * tau
    pairs = []
    for L, _ in STATE["pairs"]:
        d = STATE["dist"][L]
        k = min(STATE["dist_rank"], d["scales"].shape[0])
        v = d["mu"] + (eps[:k] * d["scales"][:k]) @ d["components"][:k]
        v = v * (d["mu"].norm() / v.norm())
        pairs.append((L, v))
    return pairs


def load_everything(args) -> None:
    ckpt = torch.load(args.vectors, map_location="cpu", weights_only=False)
    v_bridge = ckpt.get("v_bridge", {})
    missing = [L for L in args.layers if L not in v_bridge]
    if missing:
        raise SystemExit(
            f"Layers {missing} not in checkpoint v_bridge. "
            f"Available: {sorted(v_bridge.keys())}"
        )
    STATE["pairs"] = [(L, v_bridge[L]) for L in args.layers]

    STATE["dist"] = None
    if args.distribution:
        dckpt = torch.load(args.distribution, map_location="cpu", weights_only=False)
        dist = dckpt["distribution"]
        dmissing = [L for L in args.layers if L not in dist]
        if dmissing:
            raise SystemExit(f"Layers {dmissing} not in distribution file")
        STATE["dist"] = {L: {k2: v.float() for k2, v in dist[L].items()} for L in args.layers}
        STATE["dist_rank"] = min(args.rank, dckpt["rank"])
        STATE["tau"] = args.tau
        print(f"distribution sampling ON: rank={STATE['dist_rank']} tau={args.tau}")
    STATE["steer_target"] = ckpt.get("steer_target") or "residual"
    STATE["steer_scale"] = ckpt.get("steer_scale") or "fixed"
    STATE["alpha"] = args.alpha
    STATE["mode"] = args.steer_mode
    STATE["steer_start_offset"] = args.steer_start_offset
    STATE["k_assist"] = args.k_assist
    STATE["label"] = args.label
    STATE["max_new_tokens_cap"] = args.max_new_tokens_cap
    STATE["force_temperature"] = args.force_temperature
    STATE["force_min_p"] = args.force_min_p

    ckpt_model = ckpt.get("model_name")
    if ckpt_model and ckpt_model != args.model:
        print(
            f"WARNING: checkpoint was extracted on {ckpt_model!r} "
            f"but serving {args.model!r}",
            file=sys.stderr,
        )

    loader = HFLoader(args.model)
    loader.load()  # load once up front so the first request isn't slow
    STATE["loader"] = loader
    print(
        f"ready: {args.model} | layers={args.layers} alpha={args.alpha} "
        f"mode={STATE['mode']} target={STATE['steer_target']} "
        f"scale={STATE['steer_scale']}"
    )


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # health check / model listing
        self._json(200, {"object": "list", "data": [{"id": STATE["label"], "object": "model"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "invalid JSON"}})
            return

        # BATCHING. `batch_messages` is a list of message-lists generated in ONE
        # forward pass via steered_generate_batch (left-padded). `n` repeats a
        # single `messages` n times, which is how a client asks for n independent
        # samples of the same prompt in one call.
        #
        # Why this matters: batch size 1 leaves the GPU badly underutilized --
        # measured 23.6s/idea for Mistral-24B (pipeline-split over 2 cards) and
        # 4.0s/idea for Llama-8B. Batching is the single largest throughput win
        # available, since GEN_LOCK serializes concurrent requests anyway, so
        # extra client workers cannot help.
        #
        # Sampling is unaffected: each sequence in a batch is drawn
        # independently, so batched and sequential generation are draws from the
        # same distribution.
        batch_messages = payload.get("batch_messages")
        messages = payload.get("messages", [])
        n_samples = int(payload.get("n", 1) or 1)
        if not batch_messages and not messages:
            self._json(400, {"error": {"message": "messages or batch_messages required"}})
            return
        if not batch_messages:
            batch_messages = [messages] * n_samples
        # --force-temperature / --force-min-p override the request, so decoding
        # baselines (e.g. creative decoding T=1.5, min_p=0.1) can be served to
        # clients that don't send sampling params (LiveIdeaBench).
        temperature = STATE["force_temperature"] if STATE["force_temperature"] is not None \
            else float(payload.get("temperature", 0.7))
        max_tokens = min(int(payload.get("max_tokens", 2048)), STATE["max_new_tokens_cap"])
        gen_kwargs = {}
        min_p = STATE["force_min_p"] if STATE["force_min_p"] is not None else payload.get("min_p")
        if min_p is not None:
            gen_kwargs["min_p"] = float(min_p)
        if payload.get("top_p") is not None:
            gen_kwargs["top_p"] = float(payload["top_p"])

        t0 = time.time()
        with GEN_LOCK:
            try:
                # Optional per-request seed. With the SAME seed, two shims that
                # steer the opening identically (e.g. full-span steering vs
                # first_k_assistant_tokens with k>=that span) emit an IDENTICAL
                # prefix and diverge only where their steering differs — giving
                # matched pairs instead of two independent samples. Without this,
                # any single side-by-side is dominated by resampling noise
                # (measured test-retest reliability of an idea is r~0.23).
                req_seed = payload.get("seed")
                if req_seed is not None:
                    torch.manual_seed(int(req_seed))
                req_tau = payload.get("tau")
                texts = steered_generate_batch(
                    STATE["loader"],
                    batch_messages,
                    layer_vector_pairs=sample_pairs(req_tau),
                    alpha=STATE["alpha"],
                    mode=STATE["mode"],
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    steer_start_offset=STATE["steer_start_offset"],
                    k_assist=STATE["k_assist"],
                    steer_target=STATE["steer_target"],
                    steer_scale=STATE["steer_scale"],
                    **gen_kwargs,
                )
            except Exception as e:  # surface generation errors to the client
                self._json(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
                return
        dt = time.time() - t0
        n_out = len(texts)
        print(f"generated {n_out} seq, {sum(len(t) for t in texts)} chars in {dt:.1f}s "
              f"({dt / max(1, n_out):.1f}s/seq, T={temperature}, max={max_tokens})")

        self._json(200, {
            "id": "steered-cmpl",
            "object": "chat.completion",
            "model": STATE["label"],
            "choices": [{
                "index": i,
                "message": {"role": "assistant", "content": t},
                "finish_reason": "stop",
            } for i, t in enumerate(texts)],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", required=True, help="Path to steering_vectors.pt")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 15, 16, 17, 18, 19])
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--steer-mode", default="all_new_tokens")
    parser.add_argument("--steer-start-offset", type=int, default=0)
    parser.add_argument("--k-assist", type=int, default=16,
                        help="With --steer-mode first_k_assistant_tokens: steer only "
                             "the first k generated tokens, then release. Lets the model "
                             "commit to a creative framing under steering and elaborate "
                             "in its own register.")
    parser.add_argument("--distribution", default=None,
                        help="Optional steering_distribution.pt (from fit_steering_distribution.py); "
                             "enables per-request on-manifold vector sampling")
    parser.add_argument("--tau", type=float, default=1.0, help="Sampling scale for distribution mode")
    parser.add_argument("--rank", type=int, default=16, help="Max PCA components used in distribution mode")
    parser.add_argument("--force-temperature", type=float, default=None,
                        help="Override request temperature (for decoding baselines)")
    parser.add_argument("--force-min-p", type=float, default=None,
                        help="Override request min_p (for decoding baselines)")
    parser.add_argument("--label", default="steered-model", help="Model id reported in responses")
    parser.add_argument("--max-new-tokens-cap", type=int, default=12000)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_everything(args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"serving on http://0.0.0.0:{args.port}/v1/chat/completions")
    server.serve_forever()


if __name__ == "__main__":
    main()
