# Skin Journey AI

Skin Journey AI is a phone-first acne and inflammation tracking system. It captures a selfie, analyzes lesion distribution, estimates severity, and recommends whether a routine change is reasonable or whether a dermatologist visit is the safer next step.

The project is split into a mobile client, an API, and a small ML workspace that can be used to reproduce the segmentation and severity models.

## What The System Does

- Tracks skin over time with a timeline-style history
- Estimates acne severity from the face image and returns a `mild`, `moderate`, or `severe` bucket plus a `0-10` score.
- Detects eczema / atopic-dermatitis-like patterns 
- Surfaces rule-based recommendation output for OTC vs dermatology escalation.

## How The Pipeline Works

1. The API loads a U-Net lesion segmentation model from `services/api/models/unet_acne.pt`.
2. The U-Net predicts a lesion probability mask on the selfie.
3. The API converts that mask into region scores and several segmentation-derived severity features.
4. A severity head in `services/api/models/severity_head.pt` converts the same feature set into a mild / moderate / severe distribution
5. OpenAI Vision is then used as a confirmation for acne severity and a separate eczema pattern assessment 
6. If OpenAI is missing or fails, the API keeps the local segmentation-based result.

## ML Stack

The `ml` workspace contains the training and evaluation scripts used to build the local models.

### U-Net Segmentation

The core segmentation model is a small U-Net trained on Acne04-v2 image/mask pairs. The training data can come from two mask-generation paths:

- Rasterized circle masks: a coarse filled-disk baseline.
- SAM-refined masks: the recommended path, where the expert `{cx, cy, radius}` annotations are used as prompts to Segment Anything or MobileSAM.

The SAM path produces pixel-level masks that hug the actual lesion boundaries instead of filling the entire annotated circle. That improves the quality of the training targets and gives a more useful probability map for downstream scoring.

### Severity Head

The severity head is a lightweight MLP trained on a 12-feature vector derived from:

- U-Net probability mass and thresholded area measurements
- lesion count / connected components
- redness statistics
- texture / edge statistics
- interaction terms between redness and lesion intensity

The head is trained as a 3-class classifier for mild / moderate / severe. When the U-Net weights change, the severity head should be retrained too because the features are derived from U-Net predictions.

### OpenAI Integration

The API also integrates OpenAI Vision for two tasks:

- acne severity classification
- eczema / atopic-pattern classification

In the live API flow, OpenAI is used as a confirmation when it is available. The local segmentation path is still important because it powers the region scores and the fallback behavior.

## Repo Layout

- `apps/mobile`: Expo React Native app for capture, analysis, recommendations, and timeline views.
- `services/api`: FastAPI service that loads the local ML models, calls OpenAI Vision when configured, and returns analysis + recommendation payloads.
- `ml`: Training, evaluation, and dataset-preparation scripts for the Acne04 / Acne04-v2 pipeline.
- `services/api/models`: Runtime model weights expected by the API.

## Local Development

### API

```bash
cd services/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Mobile App

```bash
cd apps/mobile
npm install
npm run ios
```

Set the API base URL in the app to your machine:

- iOS simulator: `http://localhost:8000`
- Physical device: `http://<your-lan-ip>:8000`

## ML Training Workflow

The ML workspace is designed so you can regenerate the mask data, train the U-Net, and then retrain the severity head.

### 1) Download dataset

```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.datasets.acne04v2_download --out-dir data/acne04v2
```

### 2) Generate masks

Coarse baseline:

```bash
python -m ml.datasets.rasterize_masks --data-dir data/acne04v2 --out-dir data/acne04v2_masks --size 512
```

Recommended SAM-based path:

```bash
python -m ml.datasets.acne04v2_sam_masks \
	--data-dir data/acne04v2 \
	--out-dir data/acne04v2_sam_masks \
	--sam-variant mobile_sam \
	--sam-ckpt data/sam_ckpts/mobile_sam.pt \
	--size 512 --max-side 1024 \
	--write-overlays --overlay-every 50
```

The SAM generator supports MobileSAM and the standard Segment Anything checkpoints.

### 3) Train the U-Net

```bash
python train_unet.py --images data/acne04v2/images --masks data/acne04v2_sam_masks/masks --out-dir outputs/run_sam --epochs 20
```

### 4) Deploy the U-Net weights

```bash
cp outputs/run_sam/best.pt ../services/api/models/unet_acne.pt
```

### 5) Retrain the severity head

```bash
python train_severity_head.py --use-unet \
	--data-dir data/acne04v2 \
	--balanced-batches \
	--out ../services/api/models/severity_head.pt
```

`--balanced-batches` is important because it oversamples rare severity classes and helps prevent the head from collapsing toward the majority class.

## Evaluation And Debugging

- `ml/eval.py` evaluates a segmentation checkpoint and writes sample overlays plus Dice / IoU metrics.
- `ml/eval_openai_classifier.py` runs the OpenAI Vision classifier on sample Acne04 overlays or raw images.
- `ml/data.py` contains the dataset pairing logic and the filename resolution rules used when masks and images do not share identical stems.

## Notes

- The API expects model weights at `services/api/models/unet_acne.pt` and `services/api/models/severity_head.pt`.
- OpenAI Vision is optional; the local segmentation path still works without it.
