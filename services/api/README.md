# Skin Journey AI API

FastAPI service that accepts a selfie and returns:
- severity score (0–10 + mild/moderate/severe)
- region scores (forehead/cheeks/jawline)
- a heatmap overlay PNG (base64)
- rule-based OTC vs dermatologist recommendation (via `/recommend`)

## Run locally

The import path is `app.main:app`, so Python’s working directory / path must include the **`services/api`** folder (the parent of the `app/` package). If you run uvicorn from the repo root or from your home directory, you will get **`ModuleNotFoundError: No module named 'app'`**.

**Option A — from this directory (`services/api`):**

```bash
cd services/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Option B — from repo root (script `cd`s into `services/api` for you):**

```bash
bash services/api/run_dev.sh
```

Health check: `GET http://localhost:8000/health`

Eczema guidance (rule-based, driven by analyze JSON `eczema_bucket`): `GET http://localhost:8000/recommend-eczema?eczema_bucket=none` (also `mild_eczema`, `severe_eczema`).

## If you use Anaconda (common gotcha)

If `uvicorn` resolves to your global/base environment, you may see `ModuleNotFoundError: fastapi`.
Prefer:
- `python -m uvicorn ...` (always uses the active interpreter), or
- `./run_dev.sh`

