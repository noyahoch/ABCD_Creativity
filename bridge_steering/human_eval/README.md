# Human rating study — LLM judge validation

Streamlit app for collecting 1–5 human ratings on 30 AUT items + 30 MacGyver
items per rater (10 per method × 3 methods). Designed for ~10 raters per task
to validate the Qwen-32B-AWQ judge against humans.

The UI is **bilingual (English + Hebrew, displayed in parallel)** — every
label and every LLM-generated example is shown in both languages side-by-side.

## Setup

```bash
pip install -r bridge_steering/human_eval/requirements.txt
```

## 1. Build the sample (once)

```bash
python bridge_steering/human_eval/build_sample.py
```

Produces `data/aut_items.json` and `data/macgyver_items.json` — 30 items each,
10 from each of 3 methods, with the LLM judge scores attached for later
admin-only reveal. The same 30-item pool is shown to every rater; per-rater
order is shuffled by the app.

## 1b. Translate sample content to Hebrew (once)

```bash
python bridge_steering/human_eval/translate_sample.py
```

Adds `object_he`, `content_he`, `problem_text_he` fields to the JSON files in
place. Idempotent — re-running only translates fields that are missing
(useful after editing/rebuilding the sample). Uses Google Translate via
`deep-translator`; needs internet access but no API key.

## 2. Run the app

```bash
streamlit run bridge_steering/human_eval/app.py
```

Raters open `http://<host>:8501`, type a rater ID (e.g. `R01`), pick a task,
and rate 30 items. Ratings are saved to `responses/ratings.db` (SQLite, WAL)
on each submit — closing the tab and returning resumes where they left off.

### Admin reveal

Open `http://<host>:8501/?admin=1` and tick the **Show method & judge scores**
checkbox in the sidebar. An expander appears under each item showing the
method label (e.g. `baseline_t15`, `ours_l18_23_a017`) and the LLM judge's
scores. Regular raters (no `?admin=1` in URL) never see this control.

## 3. Export ratings

```bash
sqlite3 -header -csv bridge_steering/human_eval/responses/ratings.db \
  "SELECT * FROM ratings" > ratings_export.csv
```

Columns: `rater_id, task, item_id, novelty, usability, overall, ts`.
For AUT, only `novelty` + `usability` are filled; for MacGyver only `overall`.

## Method legend

| Task | Method key | What it is |
| --- | --- | --- |
| AUT | `baseline` | Mistral-24B, paper prompt, T=0.7 |
| AUT | `baseline_t15` | same but T=1.5 |
| AUT | `ours_l18_23_a017` | steering, layers 18–23, α=0.17 |
| MacGyver | `baseline` | Mistral-24B, default prompt, T=0.7 |
| MacGyver | `creative_minp005_t10` | min-p 0.05, T=1.0 |
| MacGyver | `ours_l18_23_a017` | steering, layers 18–23, α=0.17 |
