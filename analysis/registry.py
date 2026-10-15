"""Maps every published table row (Tables 2-5, plus the Falcon rows of Tables 8-10)
to the run directories / LiveIdeaBench arm labels that produced it, and records the
published numbers as fixtures.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO / "results"                       # where `python -m bridge_steering pipeline` writes
LIVEIDEA_ROOT = REPO.parent / "liveideabench"         # separate LiveIdeaBench checkout (Table 4)
SCORES = REPO / "analysis" / "scores_long.parquet"    # item-level scores behind Tables 2-5
OUT_DIR = RESULTS_ROOT / "analysis"

SEEDS = (42, 101, 999)


# ---------------------------------------------------------------------------
# AUT / MacGyver runs (bridge_steering pipeline).  pattern % seed -> run id.
# ---------------------------------------------------------------------------
# (table, benchmark, model) -> {method: pattern}
PIPELINE_RUNS: dict[tuple[str, str, str], dict[str, str]] = {
    # ---- Table 2 (AUT) and Table 3 (MacGyver): same run dirs serve both -----
    ("2", "aut", "llama"): {
        "baseline": "main/llama/baseline_s%s",
        "olson":    "main/llama/olson_s%s",
        "bridge":   "main/llama/bridge_steering_s%s",
    },
    ("2", "aut", "mistral"): {
        "baseline": "main/mistral/baseline_s%s",
        "olson":    "main/mistral/olson_s%s",
        "bridge":   "main/mistral/bridge_steering_s%s",
    },
    ("2", "aut", "falcon"): {
        "baseline": "main/falcon/baseline_s%s",
        "olson":    "main/falcon/olson_s%s",
        "bridge":   "main/falcon/bridge_steering_s%s",
    },
    # ---- Table 5 (ablations, Llama, alpha=0.17) -----------------------------
    ("5", "aut", "llama"): {
        "baseline":   "main/llama/baseline_s%s",
        "bd":         "ablations/llama/bridge_a017_s%s",
        "b_only":     "ablations/llama/bridge_only_s%s",
        "d_only":     "ablations/llama/default_only_s%s",
        "bd_generic": "ablations/llama/generic_default_s%s",
        "cd":         "ablations/llama/endpoint_contrast_s%s",
    },
}
# Table 3 uses exactly the Table 2 run dirs (MacGyver stages of the same runs);
# Table 5 MacGyver uses the Table 5 run dirs.
for (t, b, m), runs in list(PIPELINE_RUNS.items()):
    if t == "2":
        PIPELINE_RUNS[("3", "macgyver", m)] = dict(runs)
    if t == "5":
        PIPELINE_RUNS[("5", "macgyver", m)] = dict(runs)

def run_dir(model: str, pattern: str, seed: int) -> Path:
    return RESULTS_ROOT / (pattern % seed)


# ---------------------------------------------------------------------------
# LiveIdeaBench arms (Table 4 / Table 10).  Labels are ``idea_model`` values in
# liveideabench/data/ideabench.db; the three entries are the three replicate
# arms in the order used by liveideabench/table_chosen_o3_allseeds.py.
# ---------------------------------------------------------------------------
LIVEIDEA_ARMS: dict[tuple[str, str], list[str]] = {
    ("llama", "baseline"): ["llama31-8b-shim-baseline-a0", "llama31-8b-a0-r2", "llama31-8b-a0-r3"],
    ("llama", "creative"): ["llama31-8b-creative-decode-t10-minp005",
                            "llama31-8b-creative-decode-t10-minp005-r2",
                            "llama31-8b-creative-decode-t10-minp005-r3"],
    ("llama", "bridge"):   ["llama31-8b-tau10-a020", "llama31-8b-tau10-a020-s101", "llama31-8b-tau10-a020-s999"],
    ("falcon", "baseline"): ["falcon7b-a0-r1", "falcon7b-a0-r2", "falcon7b-a0-r3"],
    ("falcon", "creative"): ["falcon7b-creative-t10-minp005-r1", "falcon7b-creative-t10-minp005-r2",
                             "falcon7b-creative-t10-minp005-r3"],
    ("falcon", "bridge"):   ["falcon7b-l10_19-a012-brule", "falcon7b-l10_19-a012-brule-s101",
                             "falcon7b-l10_19-a012-brule-s999"],
    ("mistral", "baseline"): ["mistral24b-a0lb-r1", "mistral24b-a0lb-r2", "mistral24b-a0lb-r3"],
    ("mistral", "creative"): ["mistral24b-creative-t10-minp005-r1", "mistral24b-creative-t10-minp005-r2",
                              "mistral24b-creative-t10-minp005-r3"],
    ("mistral", "bridge"):   ["mistral24b-a020-brule-s42", "mistral24b-a020-brule-s101", "mistral24b-a020-brule-s999"],
}
LIVEIDEA_SEED_LABELS = (1, 2, 3)      # replicate index
LIVEIDEA_CRITIC = "openai/o3-mini"
LIVEIDEA_FLUENCY_FILE = LIVEIDEA_ROOT / "data" / "idea_hash_o3.json"
LIVEIDEA_DB = Path(os.environ.get("LIVEIDEA_DB", LIVEIDEA_ROOT / "data" / "ideabench.db"))
# (override with a local copy: the DB stalls readers on network filesystems)
LIVEIDEA_PARQUET = LIVEIDEA_ROOT / "data" / "data.parquet"
LIVEIDEA_KEYWORDS = LIVEIDEA_ROOT / "keywords_full_1180.txt"

# ---------------------------------------------------------------------------
# Metric specifications.
#   base metrics live in scores_long; derived metrics are plug-in functionals
#   of per-seed aggregates (exactly how the pipeline / table scripts compute them).
#   weighting: "obs" = per seed, sum(value*n_obs)/sum(n_obs) over items
#                      (the pipeline's mean over all judged uses);
#              "item" = per seed, plain mean over items with a value
#                      (OCSAI per reply, MacGyver per problem, LiveIdea per keyword);
#              "p30"  = per seed, 30th percentile over items (LiveIdea flexibility).
#   The across-seed step is always the unweighted mean of the 3 per-seed values.
# ---------------------------------------------------------------------------
BASE_METRICS: dict[str, dict[str, str]] = {
    "aut": {"ocsai": "item", "novelty": "obs", "usability": "obs", "semdiv": "item"},
    "macgyver": {"orig4": "item", "quality": "item"},
    "liveidea": {"originality": "item", "feasibility": "item", "clarity": "item",
                 "fluency": "item", "flex_kwmean": "p30"},
}
# derived metric -> (base metrics it needs, name of functional)
DERIVED_METRICS: dict[str, dict[str, tuple[tuple[str, ...], str]]] = {
    "aut": {"hm": (("novelty", "usability"), "hm2")},
    "macgyver": {"hm": (("orig4", "quality"), "hm_mac")},
    "liveidea": {"hm": (("originality", "feasibility", "clarity"), "hm3"),
                 "flex": (("flex_kwmean",), "identity")},
}
# Column order of the paper tables (used for LaTeX snippets and the checks)
TABLE_METRICS: dict[str, list[str]] = {
    "aut": ["ocsai", "novelty", "usability", "hm", "semdiv"],
    "macgyver": ["orig4", "quality", "hm"],
    "liveidea": ["originality", "feasibility", "clarity", "hm", "fluency", "flex"],
}
METRIC_LABEL = {
    "ocsai": "OCSAI", "novelty": "Novelty", "usability": "Usability", "hm": "HM", "semdiv": "Sem. div.",
    "orig4": "Originality", "quality": "Quality",
    "originality": "Orig.", "feasibility": "Feas.", "clarity": "Clar.", "fluency": "Fluency", "flex": "Flex. (p30)",
}
METHOD_LABEL = {
    "baseline": "Baseline", "olson": "Olson et al. 2024", "bridge": "Bridge steering",
    "creative": "Creative decoding", "bd": "B$-$D (full)", "b_only": "$+B$ only", "d_only": "$-D$ only",
    "bd_generic": "B$-$D$_{\\mathrm{generic}}$", "cd": "C$-$D",
}
MODEL_LABEL = {"llama": "Llama", "mistral": "Mistral", "falcon": "Falcon"}

# ---------------------------------------------------------------------------
# Published numbers (fixtures).  Tables 2-5 (main text) and 8-10 (Appendix D,
# Falcon rows and the same Llama/Mistral means with stds).  Values as printed.
# ---------------------------------------------------------------------------
PUBLISHED: dict[tuple[str, str, str, str], dict[str, float]] = {}


def _add(table, bench, model, method, metrics, values):
    PUBLISHED[(table, bench, model, method)] = dict(zip(metrics, values))


_AUT = TABLE_METRICS["aut"]
_add("2", "aut", "llama",   "baseline", _AUT, [3.00, 5.68, 6.36, 6.00, 0.50])
_add("2", "aut", "llama",   "olson",    _AUT, [3.59, 6.51, 5.96, 6.22, 0.51])
_add("2", "aut", "llama",   "bridge",   _AUT, [3.63, 6.52, 5.98, 6.24, 0.47])
_add("2", "aut", "mistral", "baseline", _AUT, [2.97, 5.48, 6.46, 5.93, 0.60])
_add("2", "aut", "mistral", "olson",    _AUT, [3.38, 6.02, 6.14, 6.08, 0.61])
_add("2", "aut", "mistral", "bridge",   _AUT, [3.76, 6.38, 6.08, 6.23, 0.58])
_add("2", "aut", "falcon",  "baseline", _AUT, [3.14, 5.84, 6.30, 6.06, 0.57])   # Table 8
_add("2", "aut", "falcon",  "olson",    _AUT, [3.57, 6.39, 5.99, 6.18, 0.57])
_add("2", "aut", "falcon",  "bridge",   _AUT, [3.67, 6.46, 6.03, 6.24, 0.59])

_MG = TABLE_METRICS["macgyver"]
_add("3", "macgyver", "llama",   "baseline", _MG, [0.78, 4.18, 0.81])
_add("3", "macgyver", "llama",   "olson",    _MG, [0.78, 4.23, 0.81])
_add("3", "macgyver", "llama",   "bridge",   _MG, [0.85, 4.38, 0.86])
_add("3", "macgyver", "mistral", "baseline", _MG, [0.79, 4.32, 0.82])
_add("3", "macgyver", "mistral", "olson",    _MG, [0.79, 4.27, 0.82])
_add("3", "macgyver", "mistral", "bridge",   _MG, [0.91, 4.69, 0.92])
_add("3", "macgyver", "falcon",  "baseline", _MG, [0.83, 4.38, 0.85])            # Table 9
_add("3", "macgyver", "falcon",  "olson",    _MG, [0.84, 4.51, 0.87])
_add("3", "macgyver", "falcon",  "bridge",   _MG, [0.92, 4.46, 0.90])
# Table 3 / 9 "Delta HM" column (computed from unrounded scores vs baseline)
PUBLISHED_DELTA_HM = {
    ("llama", "olson"): 0.01, ("llama", "bridge"): 0.06,
    ("mistral", "olson"): -0.01, ("mistral", "bridge"): 0.10,
    ("falcon", "olson"): 0.01, ("falcon", "bridge"): 0.05,
}

_LI = TABLE_METRICS["liveidea"]
_add("4", "liveidea", "llama",   "baseline", _LI, [7.53, 5.58, 7.69, 6.79, 7.09, 6.63])
_add("4", "liveidea", "llama",   "creative", _LI, [7.58, 5.53, 7.63, 6.76, 7.39, 6.71])
_add("4", "liveidea", "llama",   "bridge",   _LI, [7.80, 5.73, 7.62, 6.91, 6.98, 6.75])
_add("4", "liveidea", "mistral", "baseline", _LI, [7.15, 6.38, 8.24, 7.18, 6.46, 6.67])
_add("4", "liveidea", "mistral", "creative", _LI, [7.16, 6.38, 8.22, 7.18, 6.60, 6.68])
_add("4", "liveidea", "mistral", "bridge",   _LI, [7.59, 6.55, 8.20, 7.38, 6.65, 6.88])
_add("4", "liveidea", "falcon",  "baseline", _LI, [6.91, 6.39, 8.14, 7.07, 6.10, 6.50])   # Table 10
_add("4", "liveidea", "falcon",  "creative", _LI, [6.96, 6.38, 8.13, 7.09, 6.21, 6.54])
_add("4", "liveidea", "falcon",  "bridge",   _LI, [7.22, 6.39, 8.05, 7.16, 5.85, 6.50])

_add("5", "aut", "llama", "baseline",   _AUT, [3.00, 5.68, 6.36, 6.00, 0.50])
_add("5", "aut", "llama", "bd",         _AUT, [3.49, 6.34, 6.03, 6.18, 0.50])
_add("5", "aut", "llama", "b_only",     _AUT, [3.03, 5.69, 6.18, 5.93, 0.47])
_add("5", "aut", "llama", "d_only",     _AUT, [3.19, 5.95, 6.20, 6.07, 0.51])
_add("5", "aut", "llama", "bd_generic", _AUT, [3.38, 6.20, 5.93, 6.06, 0.55])
_add("5", "aut", "llama", "cd",         _AUT, [3.20, 5.97, 6.22, 6.09, 0.50])
_add("5", "macgyver", "llama", "baseline",   _MG, [0.78, 4.18, 0.81])
_add("5", "macgyver", "llama", "bd",         _MG, [0.83, 4.40, 0.86])
_add("5", "macgyver", "llama", "b_only",     _MG, [0.78, 3.82, 0.77])
_add("5", "macgyver", "llama", "d_only",     _MG, [0.79, 4.20, 0.81])
_add("5", "macgyver", "llama", "bd_generic", _MG, [0.84, 4.08, 0.83])
_add("5", "macgyver", "llama", "cd",         _MG, [0.81, 4.39, 0.84])

# Comparisons with bootstrap CIs (method_a - method_b)
COMPARISONS: dict[str, list[tuple[str, str]]] = {
    "2": [("bridge", "baseline"), ("bridge", "olson")],
    "3": [("bridge", "baseline"), ("bridge", "olson")],
    "4": [("bridge", "baseline"), ("bridge", "creative")],
    "5": [("bd", "baseline"), ("bd", "b_only"), ("bd", "d_only"), ("bd", "bd_generic"), ("bd", "cd")],
}
TABLE_BENCHMARKS = {"2": ["aut"], "3": ["macgyver"], "4": ["liveidea"], "5": ["aut", "macgyver"]}
TABLE_MODELS = {"2": ["llama", "mistral", "falcon"], "3": ["llama", "mistral", "falcon"],
                "4": ["llama", "mistral", "falcon"], "5": ["llama"]}
