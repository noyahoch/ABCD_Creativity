# Deploy to Streamlit Community Cloud (free)

A standalone deploy that doesn't depend on this workstation. Once live you'll
have a public HTTPS URL like `https://your-app.streamlit.app/` that you can
share with the 10 raters.

> **Persistence**: every rating is mirrored to a Google Sheet (durable
> across Cloud redeploys) if you wire that in (see §5 below). Local SQLite
> is just a per-session cache. Without Sheets configured, SQLite is the
> only store and gets wiped on redeploys — set up Sheets if the study
> runs more than a day.

## 1. Create a small standalone GitHub repo

Don't push the whole CreativeLM repo (it's huge and private). Make a fresh
repo with just the human-eval kit.

```bash
mkdir -p ~/human-eval-deploy && cd ~/human-eval-deploy
git init

# Copy just what the deployed app needs
cp <repo>/bridge_steering/human_eval/app.py            ./
cp <repo>/bridge_steering/human_eval/requirements.txt  ./
cp <repo>/bridge_steering/human_eval/.gitignore        ./

mkdir data
cp <repo>/bridge_steering/human_eval/data/aut_items.json       data/
cp <repo>/bridge_steering/human_eval/data/macgyver_items.json  data/
cp <repo>/bridge_steering/human_eval/data/method_mapping.json  data/
```

A minimal commit:

```bash
git add app.py requirements.txt .gitignore data
git status                              # sanity check — no responses/
git commit -m "human-eval app: initial deploy"
```

Create a **public or private** GitHub repo (Community Cloud can deploy
either; private repos need you to grant Streamlit access during signup):

```bash
gh repo create human-eval --public --source=. --push
# or use the GitHub web UI then `git remote add origin … && git push -u origin main`
```

## 2. Sign in to Streamlit Community Cloud

1. Go to <https://share.streamlit.io/>.
2. Sign in with GitHub.
3. First time: authorize Streamlit to read your repos.

## 3. Deploy

1. Click **"Create app"** → **"Deploy a public app from GitHub"**.
2. Select your `human-eval` repo.
3. Branch: `main`.
4. **Main file path**: `app.py`.
5. Click **Deploy**.

First deploy takes 2–3 minutes (installs streamlit + pandas + deep-translator).
You'll get a URL like `https://human-eval-<hash>.streamlit.app/`.

> If deploy hangs on "Building wheel for ...", that's pip building pandas.
> It's slow on the free tier but finishes within ~3 minutes.

## 4. Test before sharing

1. Open the URL. Type a test rater ID like `test1`.
2. Walk through 1–2 items in each task.
3. Click the sidebar **"Download my ratings (CSV)"** — confirm you get a
   valid CSV with your test rows.
4. (Optional) Open the same URL in incognito with rater_id `test2` —
   confirm the per-rater shuffle order differs from `test1`.
5. Reset the DB before going live by either:
   - Adding a no-op commit and triggering a redeploy on Community Cloud
     (clears `/mount/src/<repo>/responses/`), or
   - Loading the app, going through "Back to start", entering rater id
     `test1` again, and rating to overwrite (each rating is `INSERT OR
     REPLACE` so re-running is harmless).

## 5. Share the URL with raters

Email/Slack the URL plus a short note:

> Hi! Please open <URL> in your browser, type your name as the rater ID
> (so I know which ratings are yours), and rate 45 AUT items and 30
> MacGyver items. Should take ~30 minutes total. When you finish, click
> the sidebar "Download my ratings (CSV)" button and send me the file
> just in case. Thanks!

The `?admin=1` URL is for you — don't share that.

## 6. Collect results

Two ways:

**A. From each rater's CSV (recommended)**

When raters send you their `ratings_<id>.csv`, drop them all in a folder and:

```bash
python - <<'PY'
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in glob.glob("*.csv")], ignore_index=True)
df.to_csv("all_raters.csv", index=False)
print(df.groupby(["rater_id","task"]).size())
PY
```

**B. From the live DB**

While the app is up, click the sidebar "Download my ratings (CSV)" button
while logged in as a rater you trust to act as admin (the export grabs
THAT rater's rows, not everyone's). For an all-raters dump, add a
`?admin=1` flow if needed — easier to just use option A.

## Redeploying after fixing a bug

1. Edit code locally → push to GitHub.
2. Community Cloud auto-deploys within ~30 seconds.
3. **If Google Sheets is wired in (see §5)**: ratings survive the
   redeploy; the next time each rater opens the app, the app pulls their
   rows back from Sheets into SQLite, so they don't see already-rated
   items again.
4. **Without Sheets**: the redeploy wipes SQLite. Make sure all raters
   have their CSVs first.

---

## 5. Durable storage via Google Sheets

Add this if your study runs >1 day or you want to watch ratings come in
live in a spreadsheet. Setup takes ~15 minutes.

### 5.1 Create the destination sheet

1. Go to <https://sheets.new>. Name it e.g. `human_eval_ratings`.
2. You can leave the default tab as-is (`Sheet1`) — the app will **auto-create
   two task-specific tabs** on first write: `aut_ratings` and
   `macgyver_ratings`. Each gets its own header row.
3. Optional: rename `Sheet1` to `aut_ratings` and add a second tab
   `macgyver_ratings` yourself. Same result; just saves the app one API call.

### 5.2 Create a Google Cloud service account

1. Go to <https://console.cloud.google.com/projectcreate>, create a
   project (or reuse one). Anything works.
2. Enable the Sheets API:
   <https://console.cloud.google.com/apis/library/sheets.googleapis.com>.
3. Go to **APIs & Services → Credentials → Create Credentials → Service
   account**. Pick any name (`human-eval-bot`). Click through; no roles
   needed.
4. Open the new service account → **Keys → Add Key → JSON**. A `.json`
   file downloads. Treat this like a password.
5. Copy the `client_email` from the JSON (looks like
   `human-eval-bot@<project>.iam.gserviceaccount.com`).
6. Back on the Sheet you created in §5.1 → **Share** → paste that email,
   **Editor** access. Uncheck "Notify people".

### 5.3 Wire the secret into Streamlit Cloud

1. On <https://share.streamlit.io/>, open your deployed app → **⋮ menu
   → Settings → Secrets**.
2. Paste a TOML block. The exact format:

   ```toml
   [gsheets]
   spreadsheet_url = "https://docs.google.com/spreadsheets/d/.../edit"
   # Optional — these are the defaults if omitted:
   # worksheet_aut       = "aut_ratings"
   # worksheet_macgyver  = "macgyver_ratings"

   [gsheets.service_account]
   type = "service_account"
   project_id = "your-project"
   private_key_id = "abc123..."
   private_key = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANB...\n-----END PRIVATE KEY-----\n"
   client_email = "human-eval-bot@your-project.iam.gserviceaccount.com"
   client_id = "1234567890"
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
   universe_domain = "googleapis.com"
   ```

   Copy each field from the JSON you downloaded. **Important**: for
   `private_key`, the JSON has the newline-encoded form (`\n` literal) —
   keep it exactly as it appears in the JSON (don't expand the `\n`).

3. Click **Save**. The app restarts automatically.

### 5.4 Test the integration

1. Open your live URL, type `test`, rate one **AUT** item, click Submit & next.
   → an `aut_ratings` tab should auto-appear in your sheet with one row.
2. Then rate one **MacGyver** item → a `macgyver_ratings` tab should appear
   with one row.
3. If nothing happens, check **Manage app → Logs** in the Cloud
   dashboard for the Sheets init error.

### 5.5 Local development with Sheets

For local testing without deploying, put the same TOML in
`bridge_steering/human_eval/.streamlit/secrets.toml` (gitignored). Run normally
with `streamlit run`.

### 5.6 Reading the data

- **Live view**: just open the Google Sheet.
- **CSV export**: Sheet → File → Download → CSV.
- **Programmatic**: `gspread` or pandas's `pd.read_csv` on the published
  CSV URL.

> **Privacy note**: rater names are stored in plaintext. Keep the Sheet
> private (only share with collaborators). If publishing the data, hash
> the `rater_id` column first.

## Common gotchas

- **Empty data/ folder on cloud** → you forgot to commit `data/*.json`.
  Check `git ls-files data/`. The repo's `.gitignore` is set up so
  `data/` IS committed and `responses/` is NOT.
- **App boots but shows "no items"** → `data/aut_items.json` is
  empty/malformed. Rebuild locally with `build_sample.py` and re-commit.
- **"Module 'deep_translator' not found"** during cold start → it's
  needed only for the (optional) local `translate_sample.py`. Move it out
  of `requirements.txt` if you want a faster cloud boot.
