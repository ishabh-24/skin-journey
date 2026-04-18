# ML (training workspace)

This folder contains scripts to train a **U-Net lesion segmentation** model using Acne04-v2 circle annotations rasterized into pixel masks.

## Quickstart

Create an environment (recommended: separate from `services/api/.venv`):

```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download dataset (annotations + (optionally) images):

```bash
python -m ml.datasets.acne04v2_download --out-dir data/acne04v2
```

If you see **401/404** errors downloading images, it usually means:
- the dataset is **gated** on Hugging Face (needs a token), and/or
- the Acne04-v2 release you’re using provides **annotations only** (images must come from the original ACNE04 dataset).

To use a Hugging Face token:

```bash
export HUGGINGFACE_TOKEN="..."
python -m ml.datasets.acne04v2_download --out-dir data/acne04v2 --max-images 200
```

Rasterize circle annotations → binary masks:

```bash
python -m ml.datasets.rasterize_masks --data-dir data/acne04v2 --out-dir data/acne04v2_masks --size 512
```

Train U-Net:

```bash
python train_unet.py --images data/acne04v2/images --masks data/acne04v2_masks/masks --out-dir outputs/run1 --epochs 20
```

Export weights to API (so `POST /analyze` uses them):

```bash
cp outputs/run1/best.pt ../services/api/models/unet_acne.pt
```

## Notes
- ACNE04/Acne04-v2 may be restricted to **academic use**. Verify licensing before broader use.
