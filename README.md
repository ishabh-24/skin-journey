# Skin Journey AI

Phone-based AI system that tracks acne + inflammation over time, estimates severity, generates region heatmaps, and suggests OTC routines vs when to see a dermatologist.

## Repo layout

- `apps/mobile`: Expo React Native app (capture → analysis → recommendations → timeline)
- `services/api`: FastAPI service (image analysis + rule-based recommendations)

## Run (local dev)

### API

```bash
cd services/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Mobile (Expo)

```bash
cd apps/mobile
npm install
npm run ios
```

In the app, set the API base URL to your machine:
- iOS simulator: `http://localhost:8000`
- Physical device: `http://<your-lan-ip>:8000`

