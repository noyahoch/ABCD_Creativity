"""Aggregate human + LLM judge results for AUT, MacGyver, and Pairwise.

Input: TSV/CSV files exported from the Google Sheets tabs.
Output: per-method tables with the exact columns reported in the paper.

Usage:
    python bridge_steering/human_eval/analyze_results.py \\
        --aut       /tmp/aut_ratings.tsv \\
        --macgyver  /tmp/macgyver_ratings.tsv \\
        --pairwise  /tmp/cw_design_ratings.tsv

Any of the three flags can be omitted — the script processes only the
tasks you pass. The TSV/CSV must include the column header row (the
script auto-detects '\\t' vs ',' from the first non-empty line).

Tables:
    AUT       per method: judge_nov, human_nov, judge_use, human_use,
                          judge_hm, human_hm (all on 1-10; human ×2)
    MacGyver  per method: judge_quality, human_quality, ngram,
                          judge_hm = HM(ngram, judge_q/5),
                          human_hm = HM(ngram, human_q/5)
    Pairwise  per method: clearly wins, slightly wins, total
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO     = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
REB_DIR  = REPO / "results"


# ---------- helpers ----------

def _read_rows(path: Path) -> list[dict]:
    """Read a CSV or TSV file with header row; autodetect delimiter."""
    text = path.read_text()
    # Sniff delimiter from the first line
    first = next((line for line in text.splitlines() if line.strip()), "")
    delim = "\t" if first.count("\t") >= first.count(",") else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delim))


def _dedup_latest(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Keep the row with the latest `ts` per composite key."""
    seen: dict[tuple, dict] = {}
    for r in rows:
        k = tuple(r.get(c, "") for c in keys)
        try:
            ts = float(r.get("ts", 0) or 0)
        except (TypeError, ValueError):
            ts = 0.0
        if k not in seen or float(seen[k].get("ts", 0) or 0) < ts:
            seen[k] = r
    return list(seen.values())


def _hm(a: float, b: float) -> float:
    if a + b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


def _to_int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _to_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows])
              for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}s}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*[str(c) for c in r]))


# ---------- correlation helpers (no scipy dep) ----------

def _rank(xs: list[float]) -> list[float]:
    """Average ranks (handles ties)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    num = sum(dx[i] * dy[i] for i in range(n))
    denom = (sum(d * d for d in dx) * sum(d * d for d in dy)) ** 0.5
    if denom == 0:
        return None
    return num / denom


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_rank(xs), _rank(ys))


def _avg_pairwise_rater_rho(
    rater_item_scores: dict[str, dict[str, float]],
    min_overlap: int = 3,
) -> tuple[float | None, int, int]:
    """Mean pairwise Pearson r across all rater pairs with >= min_overlap items.

    Returns (mean_r, n_pairs_used, n_pairs_total).
    """
    raters = sorted(rater_item_scores)
    rs: list[float] = []
    n_pairs_total = 0
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            n_pairs_total += 1
            shared = sorted(set(rater_item_scores[raters[i]]) &
                            set(rater_item_scores[raters[j]]))
            if len(shared) < min_overlap:
                continue
            xs = [rater_item_scores[raters[i]][k] for k in shared]
            ys = [rater_item_scores[raters[j]][k] for k in shared]
            r = _pearson(xs, ys)
            if r is not None:
                rs.append(r)
    if not rs:
        return None, 0, n_pairs_total
    return sum(rs) / len(rs), len(rs), n_pairs_total


def _fmt(x: float | None, w: int = 5, p: int = 2) -> str:
    return f"{x:.{p}f}" if x is not None else "n/a".rjust(w)


# ---------- judge data loaders ----------

AUT_RUN = {
    "baseline": ("main/mistral/baseline_s42",
                 "aut_judge/baseline_judge_scores.csv"),
    "baseline_t15": ("main/mistral/baseline_high_temp_s42",
                     "aut_judge/baseline_judge_scores.csv"),
    "ours_l18_23_a017": (
        "judge_robustness/mistral/structured_bridges_a017_s42",
        "aut_judge/steered_L18_L19_L20_L21_L22_L23_a0.17_judge_scores.csv",
    ),
}
MG_JUDGE_RUN = {
    "baseline": ("main/mistral/baseline_s42",
                 "macgyver_judge/baseline_judge_scores.csv"),
    "creative_minp005_t10": ("main/mistral/creative_decoding_1_s42",
                             "macgyver_judge/baseline_judge_scores.csv"),
    "ours_l18_23_a017": (
        "judge_robustness/mistral/structured_bridges_a017_s42",
        "macgyver_judge/steered_L18_L19_L20_L21_L22_L23_a0.17_judge_scores.csv",
    ),
}
MG_NGRAM_RUN = {
    "baseline": ("main/mistral/baseline_s42",
                 "macgyver_novelty/baseline_novelty.csv"),
    "creative_minp005_t10": ("main/mistral/creative_decoding_1_s42",
                             "macgyver_novelty/baseline_novelty.csv"),
    "ours_l18_23_a017": (
        "judge_robustness/mistral/structured_bridges_a017_s42",
        "macgyver_novelty/steered_L18_L19_L20_L21_L22_L23_a0.17_novelty.csv",
    ),
}


def _load_aut_judge() -> dict[tuple[str, str], tuple[int, int]]:
    """Return {(method, use_text): (judge_novelty 1-10, judge_usability 1-10)}.
    use_text matches the parsed_response item string."""
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for method, (run, suffix) in AUT_RUN.items():
        path = REB_DIR / run / suffix
        if not path.exists():
            print(f"  ! missing {path}")
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                ut = str(r.get("use_text", "")).strip()
                n = _to_int(r.get("novelty"))
                u = _to_int(r.get("usability"))
                if ut and n and u:
                    out[(method, ut)] = (n, u)
    return out


def _load_mg_judge() -> dict[tuple[str, str], float]:
    """Return {(method, source_item_id): judge_quality}."""
    out: dict[tuple[str, str], float] = {}
    for method, (run, suffix) in MG_JUDGE_RUN.items():
        path = REB_DIR / run / suffix
        if not path.exists():
            print(f"  ! missing {path}")
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                iid = str(r.get("item_id", "")).strip()
                q = _to_float(r.get("quality_score"))
                if iid and q is not None:
                    out[(method, iid)] = q
    return out


def _load_mg_ngram() -> dict[tuple[str, str], float]:
    """Return {(method, source_item_id): originality_ngram_4}."""
    out: dict[tuple[str, str], float] = {}
    for method, (run, suffix) in MG_NGRAM_RUN.items():
        path = REB_DIR / run / suffix
        if not path.exists():
            print(f"  ! missing {path}")
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                iid = str(r.get("item_id", "")).strip()
                ng = _to_float(r.get("originality_ngram_4"))
                if iid and ng is not None:
                    out[(method, iid)] = ng
    return out


# ---------- AUT ----------

def analyze_aut(path: Path, scale_max: int = 5) -> None:
    # Old `aut_ratings` sheet uses 1-5 (multiply by 2 to align with judge's 1-10).
    # New `aut_ratings_v2` sheet uses 1-10 directly (scale_max=10, no scaling).
    factor = 10 / scale_max
    print("\n" + "=" * 72)
    print(f"AUT  ←  {path}    (raw scale 1-{scale_max}; "
          f"{'no scaling' if factor == 1 else f'×{factor:g} to align 1-10'})")
    print("=" * 72)
    items = json.loads((DATA_DIR / "aut_items.json").read_text())
    method_of = {it["item_id"]: it["method"] for it in items}
    use_text_of = {it["item_id"]: it["content"] for it in items}
    judge_lookup = _load_aut_judge()

    rows = _read_rows(path)
    deduped = _dedup_latest(rows, ("rater_id", "item_id"))
    raters = Counter(r["rater_id"] for r in deduped)

    print(f"  {len(rows)} raw rows · {len(deduped)} unique (rater, item) · "
          f"{len(raters)} raters")
    for rid, n in raters.most_common():
        print(f"    {rid:25s}  {n} / 45")

    # Per-method aggregates: average across (rater, item) data points.
    per = defaultdict(lambda: {
        "h_nov": [], "h_use": [], "h_hm": [],
        "j_nov": [], "j_use": [], "j_hm": [],
    })
    for r in deduped:
        m = method_of.get(r["item_id"], "?")
        hn, hu = _to_int(r.get("novelty")), _to_int(r.get("usability"))
        if hn is None or hu is None or hn == 0 or hu == 0:
            continue
        # Scale to 1-10 to match judge (×2 for 1-5 inputs, ×1 for 1-10 inputs).
        hn_x, hu_x = hn * factor, hu * factor
        per[m]["h_nov"].append(hn_x)
        per[m]["h_use"].append(hu_x)
        per[m]["h_hm"].append(_hm(hn_x, hu_x))

        # Pair human rating with the judge's rating of the SAME (method, use).
        ut = use_text_of.get(r["item_id"])
        if ut and (m, ut) in judge_lookup:
            jn, ju = judge_lookup[(m, ut)]
            per[m]["j_nov"].append(jn)
            per[m]["j_use"].append(ju)
            per[m]["j_hm"].append(_hm(jn, ju))

    headers = ["method", "judge_nov", "human_nov", "judge_use", "human_use",
               "judge_hm", "human_hm", "n"]
    table = []
    for m in sorted(per):
        s = per[m]
        def mean(xs): return sum(xs) / len(xs) if xs else 0.0
        table.append([
            m,
            f"{mean(s['j_nov']):.2f}",
            f"{mean(s['h_nov']):.2f}",
            f"{mean(s['j_use']):.2f}",
            f"{mean(s['h_use']):.2f}",
            f"{mean(s['j_hm']):.2f}",
            f"{mean(s['h_hm']):.2f}",
            str(len(s["h_nov"])),
        ])
    _print_table(headers, table)

    # ------ Judge vs human comparison ------
    # Per-item: mean+std across raters, paired with judge's score for that item.
    # Both human and judge are on 1-10 here (factor scales 1-5 inputs).
    item_human_nov: dict[str, list[float]] = defaultdict(list)
    item_human_use: dict[str, list[float]] = defaultdict(list)
    rater_nov: dict[str, dict[str, float]] = defaultdict(dict)
    rater_use: dict[str, dict[str, float]] = defaultdict(dict)
    for r in deduped:
        iid = r["item_id"]
        hn, hu = _to_int(r.get("novelty")), _to_int(r.get("usability"))
        if hn is None or hu is None or hn == 0 or hu == 0:
            continue
        item_human_nov[iid].append(hn * factor)
        item_human_use[iid].append(hu * factor)
        rid = r.get("rater_id", "")
        rater_nov[rid][iid] = float(hn * factor)
        rater_use[rid][iid] = float(hu * factor)

    # Build paired (judge, mean_human) lists, overall and per method
    pairs_all = {"nov": [], "use": []}
    pairs_per = defaultdict(lambda: {"nov": [], "use": []})
    stds = {"nov": [], "use": []}
    for iid, mh_nov in item_human_nov.items():
        if not mh_nov:
            continue
        m = method_of.get(iid, "?")
        ut = use_text_of.get(iid)
        if not ut or (m, ut) not in judge_lookup:
            continue
        jn, ju = judge_lookup[(m, ut)]
        mean_nov = sum(mh_nov) / len(mh_nov)
        mean_use = sum(item_human_use[iid]) / len(item_human_use[iid])
        pairs_all["nov"].append((jn, mean_nov))
        pairs_all["use"].append((ju, mean_use))
        pairs_per[m]["nov"].append((jn, mean_nov))
        pairs_per[m]["use"].append((ju, mean_use))
        if len(mh_nov) > 1:
            n = len(mh_nov)
            stds["nov"].append((sum((v - mean_nov) ** 2 for v in mh_nov) / n) ** 0.5)
            stds["use"].append((sum((v - mean_use) ** 2 for v in item_human_use[iid]) / n) ** 0.5)

    def stats(pairs):
        if len(pairs) < 3:
            return None, None, len(pairs)
        js = [p[0] for p in pairs]
        hs = [p[1] for p in pairs]
        rho = _spearman(js, hs)
        mae = sum(abs(js[i] - hs[i]) for i in range(len(pairs))) / len(pairs)
        return rho, mae, len(pairs)

    print("\n  Judge ↔ human (per-item mean):")
    print(f"  {'scope':<22s}  {'rho_nov':>8s}  {'mae_nov':>8s}  "
          f"{'rho_use':>8s}  {'mae_use':>8s}  {'n':>3s}")
    for scope, p in [("ALL", pairs_all)] + [(m, pairs_per[m]) for m in sorted(pairs_per)]:
        rn, mn, nn = stats(p["nov"])
        ru, mu, _  = stats(p["use"])
        print(f"  {scope:<22s}  {_fmt(rn):>8s}  {_fmt(mn):>8s}  "
              f"{_fmt(ru):>8s}  {_fmt(mu):>8s}  {nn:>3d}")

    # Inter-rater ceiling (the noise floor judge has to beat)
    rn_r, np_n, np_total = _avg_pairwise_rater_rho(rater_nov)
    ru_r, np_u, _ = _avg_pairwise_rater_rho(rater_use)
    print(f"\n  Inter-rater Pearson r  (avg across rater pairs with >=3 shared items):")
    print(f"    novelty   : {_fmt(rn_r)}  ({np_n}/{np_total} rater pairs)")
    print(f"    usability : {_fmt(ru_r)}  ({np_u}/{np_total} rater pairs)")
    if stds["nov"]:
        avg_sn = sum(stds["nov"]) / len(stds["nov"])
        avg_su = sum(stds["use"]) / len(stds["use"])
        print(f"  Avg per-item human std (1-10 scale): "
              f"nov={avg_sn:.2f}, use={avg_su:.2f}  "
              f"(across {len(stds['nov'])} items with >=2 raters)")


# ---------- MacGyver ----------

def analyze_mg(path: Path) -> None:
    print("\n" + "=" * 72)
    print(f"MacGyver  ←  {path}")
    print("=" * 72)
    items = json.loads((DATA_DIR / "macgyver_items.json").read_text())
    method_of = {it["item_id"]: it["method"] for it in items}
    source_of = {it["item_id"]: str(it["source_item_id"]) for it in items}
    judge_lookup = _load_mg_judge()
    ngram_lookup = _load_mg_ngram()

    rows = _read_rows(path)
    deduped = _dedup_latest(rows, ("rater_id", "item_id"))
    raters = Counter(r["rater_id"] for r in deduped)
    print(f"  {len(rows)} raw rows · {len(deduped)} unique (rater, item) · "
          f"{len(raters)} raters")
    for rid, n in raters.most_common():
        print(f"    {rid:25s}  {n} / 30")

    per = defaultdict(lambda: {
        "human_q": [], "judge_q": [], "ngram": [],
        "human_hm": [], "judge_hm": [],
    })
    for r in deduped:
        m = method_of.get(r["item_id"], "?")
        hq = _to_int(r.get("overall"))
        if hq is None or hq == 0:
            continue
        src = source_of.get(r["item_id"])
        jq = judge_lookup.get((m, src))
        ng = ngram_lookup.get((m, src))
        if jq is None or ng is None:
            continue
        # HM defined on 0-1: divide quality by 5, keep ngram as-is.
        h_hm = _hm(hq / 5.0, ng)
        j_hm = _hm(jq / 5.0, ng)
        per[m]["human_q"].append(hq)
        per[m]["judge_q"].append(jq)
        per[m]["ngram"].append(ng)
        per[m]["human_hm"].append(h_hm)
        per[m]["judge_hm"].append(j_hm)

    headers = ["method", "judge_quality", "human_quality", "ngram_4",
               "judge_hm", "human_hm", "n"]
    table = []
    for m in sorted(per):
        s = per[m]
        def mean(xs): return sum(xs) / len(xs) if xs else 0.0
        table.append([
            m,
            f"{mean(s['judge_q']):.2f}",
            f"{mean(s['human_q']):.2f}",
            f"{mean(s['ngram']):.3f}",
            f"{mean(s['judge_hm']):.3f}",
            f"{mean(s['human_hm']):.3f}",
            str(len(s["human_q"])),
        ])
    _print_table(headers, table)

    # ------ Judge vs human comparison (per item) ------
    item_human_q: dict[str, list[float]] = defaultdict(list)
    rater_q: dict[str, dict[str, float]] = defaultdict(dict)
    for r in deduped:
        iid = r["item_id"]
        hq = _to_int(r.get("overall"))
        if hq is None or hq == 0:
            continue
        item_human_q[iid].append(float(hq))
        rater_q[r.get("rater_id", "")][iid] = float(hq)

    pairs_all = []
    pairs_per = defaultdict(list)
    stds = []
    for iid, hs in item_human_q.items():
        if not hs:
            continue
        m = method_of.get(iid, "?")
        src = source_of.get(iid)
        jq = judge_lookup.get((m, src))
        if jq is None:
            continue
        mean_h = sum(hs) / len(hs)
        # Compare on common scale: scale both to 0-1 (human ÷5, judge ÷5).
        pairs_all.append((jq / 5.0, mean_h / 5.0))
        pairs_per[m].append((jq / 5.0, mean_h / 5.0))
        if len(hs) > 1:
            n = len(hs)
            stds.append((sum((v - mean_h) ** 2 for v in hs) / n) ** 0.5)

    def stats(pairs):
        if len(pairs) < 3:
            return None, None, len(pairs)
        js = [p[0] for p in pairs]
        hs = [p[1] for p in pairs]
        rho = _spearman(js, hs)
        mae = sum(abs(js[i] - hs[i]) for i in range(len(pairs))) / len(pairs)
        return rho, mae, len(pairs)

    print("\n  Judge ↔ human quality (per-item mean, both on 0-1):")
    print(f"  {'scope':<26s}  {'rho_q':>6s}  {'mae_q':>6s}  {'n':>3s}")
    for scope, p in [("ALL", pairs_all)] + [(m, pairs_per[m]) for m in sorted(pairs_per)]:
        rho, mae, n = stats(p)
        print(f"  {scope:<26s}  {_fmt(rho):>6s}  {_fmt(mae, p=3):>6s}  {n:>3d}")

    rq_r, np_n, np_total = _avg_pairwise_rater_rho(rater_q)
    print(f"\n  Inter-rater Pearson r (avg across rater pairs, quality):")
    print(f"    {_fmt(rq_r)}  ({np_n}/{np_total} rater pairs with >=3 shared items)")
    if stds:
        avg_s = sum(stds) / len(stds)
        print(f"  Avg per-item human std (1-5 scale): {avg_s:.2f}  "
              f"(across {len(stds)} items with >=2 raters)")


# ---------- Pairwise ----------

def analyze_pairwise(path: Path) -> None:
    print("\n" + "=" * 72)
    print(f"Pairwise  ←  {path}")
    print("=" * 72)
    trials = json.loads((DATA_DIR / "pairwise_trials.json").read_text())["trials"]
    trial_map = {
        t["trial_id"]: {
            "a_method": t["a_internal_method"],
            "b_method": t["b_internal_method"],
            "pair_id":  t["pair_id"],
        }
        for t in trials
    }

    raw = _read_rows(path)
    rows = _dedup_latest(raw, ("rater_id", "trial_id"))
    raters = Counter(r["rater_id"] for r in rows)
    print(f"  {len(raw)} raw rows · {len(rows)} unique (rater, trial) · "
          f"{len(raters)} raters")
    for rid, n in raters.most_common():
        print(f"    {rid:25s}  {n} / 12")

    wins = Counter()                 # method -> total wins
    strengths = Counter()            # (method, 'clearly'|'slightly') -> count
    pair_results = defaultdict(lambda: {"ours_clearly": 0, "ours_slightly": 0,
                                        "other_clearly": 0, "other_slightly": 0})
    total = 0
    for r in rows:
        tid = str(r.get("trial_id", "")).strip()
        info = trial_map.get(tid, {})
        choice = str(r.get("rater_choice", ""))
        side = "a" if "A is" in choice else ("b" if "B is" in choice else None)
        if not side:
            continue
        winner = info.get(f"{side}_method", "?")
        strength = "clearly" if "clearly" in choice else "slightly"
        wins[winner] += 1
        strengths[(winner, strength)] += 1
        total += 1
        # Track per-pair ours stats
        ours_side = str(r.get("ours_position", "")).strip().lower()
        if ours_side in {"a", "b"}:
            picked_ours = (side == ours_side)
            key = ("ours" if picked_ours else "other") + "_" + strength
            pair_results[info.get("pair_id", "?")][key] += 1

    headers = ["method", "clearly", "slightly", "total", "%_of_total"]
    table = []
    for m, _ in wins.most_common():
        cl = strengths.get((m, "clearly"), 0)
        sl = strengths.get((m, "slightly"), 0)
        tot = cl + sl
        pct = (tot / total * 100) if total else 0
        table.append([m, str(cl), str(sl), str(tot), f"{pct:.0f}%"])
    _print_table(headers, table)
    print(f"  total decided trials: {total}")

    # Per-pair ours win rate
    if pair_results:
        print("\n  Per-pair ours stats:")
        for pid, st in pair_results.items():
            ours_total = st["ours_clearly"] + st["ours_slightly"]
            other_total = st["other_clearly"] + st["other_slightly"]
            tot = ours_total + other_total
            print(
                f"    {pid:<32s}  "
                f"ours: {ours_total}/{tot} "
                f"(clearly={st['ours_clearly']}, slightly={st['ours_slightly']})  ·  "
                f"other: {other_total}/{tot} "
                f"(clearly={st['other_clearly']}, slightly={st['other_slightly']})"
            )


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aut", type=Path)
    ap.add_argument("--aut-scale", type=int, default=10,
                    help="Raw AUT rating scale max (10 = new v2 sheet, 5 = old sheet)")
    ap.add_argument("--macgyver", type=Path)
    ap.add_argument("--pairwise", type=Path)
    args = ap.parse_args()

    if not any([args.aut, args.macgyver, args.pairwise]):
        ap.error("pass at least one of --aut / --macgyver / --pairwise")

    if args.aut:
        analyze_aut(args.aut, scale_max=args.aut_scale)
    if args.macgyver:
        analyze_mg(args.macgyver)
    if args.pairwise:
        analyze_pairwise(args.pairwise)


if __name__ == "__main__":
    main()
