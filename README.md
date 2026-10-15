# Bridging Creativity in LLMs: Mechanism-Driven Creative Leaps

**Noya Hochwald, Shahar Katz, Lior Wolf** · Tel Aviv University

[Paper](#) · [Dataset](data/aut_bridgeset.json) · [Citation](#citation)

**Bridge Steering** makes language models more creative without telling them how.
For each example we teacher-force the same prompt *A* twice, once with a default
continuation *D* and once with a *bridge* *B* (a reframing of the problem), and take
the difference of their mean activations. Adding that direction to the residual
stream at inference raises originality on the Alternative Uses Task, and transfers
zero-shot to MacGyver problem solving and LiveIdeaBench scientific ideation.

## Setup

```bash
git clone <this repo> && cd <repo>
uv sync                    # or: pip install -e .
uv sync --extra vllm       # Linux + CUDA: adds the local LLM judge (needed for full runs)
```

Python 3.10–3.12. A run needs one GPU with ~24 GB for the 7–8B models, and ~48 GB for
Mistral-Small-24B and for the Qwen2.5-32B-AWQ judge (the pipeline starts and stops the
judge for you). Environment variables:

| Variable | Needed for |
|---|---|
| `HF_TOKEN` | gated models (Llama, Mistral) |
| `OPENSCORING_API_KEY` | OCSAI originality scores on AUT ([openscoring.du.edu](https://openscoring.du.edu)) |

**One-off prerequisite for MacGyver originality.** Originality is the fraction of
4-grams unseen in a 500M-token FineWeb sample. The index is not shipped; build it once:

```bash
python scripts/build_ngram_index.py --workers 8     # writes results/novelty/fineweb_500m.npz
```

## Quick start

Run bridge steering on Llama-3.1-8B-Instruct (vector extraction → AUT and MacGyver
generation → scoring → summary):

```bash
python -m bridge_steering pipeline --config configs/main/llama/bridge_steering_s42.yaml
```

Outputs go to `results/main/llama/bridge_steering_s42/`; the summary is
`aggregate/cross_stage_summary.csv`. Useful flags: `--dry-run` (show the plan),
`--only steer_collect,steer_extract,aut_infer` (run some stages), `--resume`.

## Reproducing the paper

Every config produces one row of one table for one seed (42, 101, 999):

| Paper | Configs |
|---|---|
| Tables 2, 3, 8, 9 — main results (Llama, Mistral, Falcon) | `configs/main/{llama,mistral,falcon}/` |
| Tables 5, 12, 13, 14 — vector composition, bridge wording, construction | `configs/ablations/llama/` |
| A-only reference | `configs/ablations/llama/a_only_*` (run `python -m analysis.a_only_vector` first) |
| Table 15 — steering strength α | `configs/alpha_sweep/llama/` (α = 0.17 and 0.20 are `ablations/llama/bridge_a017_*` and `main/llama/bridge_steering_*`) |
| Table 16 — sample efficiency | `configs/sample_efficiency/llama/` (N = 110 is `ablations/llama/bridge_a017_s42`) |
| Table 11 — second judge (Gemma-3-27B-IT) | `configs/judge_robustness/mistral/` (`gemma_*` re-judge each run with Gemma; the others are the Qwen-judged runs not in the main tables) |

Each config starts with a comment naming its table and method. Method names in `main/`:
`baseline` (T=0.7), `baseline_high_temp` (T=1.5), `creative_decoding_1/2` (min-p),
`brainstorm`, `crpo` (Llama only), `denial_aut` / `denial_macgyver`, `novelty_prompting`
(MacGyver), `olson` (Olson et al., 2024), `bridge_steering` (ours).

**Confidence intervals and per-item scores.** `analysis/scores_long.parquet` holds the
item-level scores behind Tables 2–5 and the A-only ablation, so the bootstrap CIs
(Table `ci_all`) need no GPU. The resulting tables are committed as
`results/analysis/ci_results.csv` and `ci_results_a_only.csv`; to regenerate them:

```bash
python -m analysis.bootstrap_ci          # 10,000 paired item-level resamples → results/analysis/
python -m analysis.reproduction_check    # re-aggregates the scores and checks every published cell
```

To rebuild the score table from your own runs: `python -m analysis.build_scores_long`
(the LiveIdeaBench rows also need a LiveIdeaBench checkout next to this repo).

**LiveIdeaBench (Tables 4, 10, 21)** is run with the benchmark's own code against the
steered model served as an OpenAI-compatible endpoint, using the stochastic variant (τ = 1).
See the usage in `scripts/serve_steered_openai.py`.

**Human evaluation (Tables 22, 23):** the rating app and analysis are in
`bridge_steering/human_eval/` (`uv sync --extra human-eval`).

Not included: Table 17 (layer choice) and the qualitative examples at α ≥ 1.0 come from
an earlier single-layer pipeline that this code does not reproduce.

## Data

`data/aut_bridgeset.json` is **AUT-BridgeSet**: 150 everyday objects, each with an AUT
prompt (`A`), default uses (`D`, `D_banal`), four bridges, one per family (`B_list`), and
reference creative uses (`C`, used only for the C−D ablation). Objects are split 110
construction / 40 held-out by `random.Random(7)`. Two ablation variants are in
`data/variants/`. `data/bridge_audit_flags.csv` tags the six intermediate-construct
bridges judged borderline in the audit (`python -m analysis.bridge_audit` reruns the LLM screen). MacGyver problems are
from Tian et al. (2024).

## Repository layout

```
bridge_steering/   the method and evaluation pipeline (python -m bridge_steering)
  steering/        activation caching, steering vectors, hooks
  stages/          pipeline stages: steer_collect, steer_extract, aut_infer, aut_judge, ...
  prompts/         every prompt, versioned and hash-locked
  human_eval/      human-study app and analysis
configs/           one YAML per paper table row and seed
data/              AUT-BridgeSet and MacGyver
analysis/          per-item scores, bootstrap CIs, table reproduction, A-only ablation
scripts/           n-gram index, steered OpenAI-compatible server (LiveIdeaBench)
tests/             CPU-only tests: uv run pytest
```

## Citation

```bibtex
@inproceedings{hochwald2026bridging,
  title     = {Bridging Creativity in {LLM}s: Mechanism-Driven Creative Leaps},
  author    = {Hochwald, Noya and Katz, Shahar and Wolf, Lior},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```

## License

Code: [MIT](LICENSE). AUT-BridgeSet: CC BY 4.0.
