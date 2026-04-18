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

### Option A: rasterize circle annotations → filled-disk masks (coarse)

```bash
python -m ml.datasets.rasterize_masks --data-dir data/acne04v2 --out-dir data/acne04v2_masks --size 512
```

### Option B (recommended): SAM-refined pixel masks from the same circles

This uses the expert `{cx, cy, radius}` annotations as **prompts** into Segment
Anything (point + tight box), so the output masks hug the actual lesion pixels
instead of being filled disks. U-Net trained on these is noticeably more
sensitive.

1. Install MobileSAM (small + fast) **or** segment-anything (ViT-B/L/H):
   ```bash
   pip install git+https://github.com/ChaoningZhang/MobileSAM.git
   # or
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```
2. Grab a checkpoint:
   ```bash
   python -m ml.datasets.download_sam_ckpt --variant mobile_sam \
       --out data/sam_ckpts/mobile_sam.pt
   ```
   If the download is blocked, the script prints the direct URLs — download
   manually and put the file at `data/sam_ckpts/mobile_sam.pt`.
3. Generate masks:
   ```bash
   python -m ml.datasets.acne04v2_sam_masks \
       --data-dir data/acne04v2 \
       --out-dir  data/acne04v2_sam_masks \
       --sam-variant mobile_sam \
       --sam-ckpt data/sam_ckpts/mobile_sam.pt \
       --size 512 --max-side 1024 \
       --write-overlays --overlay-every 50
   ```
   Start with `--limit 20` for a smoke test, check `data/acne04v2_sam_masks/overlays/`,
   then rerun without `--limit`.

### Train U-Net

```bash
# on coarse disk masks
python train_unet.py --images data/acne04v2/images --masks data/acne04v2_masks/masks --out-dir outputs/run1 --epochs 20

# on SAM-refined masks (recommended)
python train_unet.py --images data/acne04v2/images --masks data/acne04v2_sam_masks/masks --out-dir outputs/run_sam --epochs 20
```

### Deploy weights to the API

```bash
cp outputs/run_sam/best.pt ../services/api/models/unet_acne.pt
```

After swapping the U-Net weights, **retrain the severity head** too — its input
features come from U-Net predictions:

```bash
python train_severity_head.py --use-unet \
    --data-dir data/acne04v2 \
    --out ../services/api/models/severity_head.pt
```

## Notes
- ACNE04/Acne04-v2 may be restricted to **academic use**. Verify licensing before broader use.
