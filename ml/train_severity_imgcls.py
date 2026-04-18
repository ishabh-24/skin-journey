from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def _label_from_filename(name: str) -> int:
    base = Path(name).name.lower()
    j = base.find("levle")
    if j == -1 or j + 5 >= len(base) or not base[j + 5].isdigit():
        raise ValueError(f"cannot parse label from filename: {name}")
    lvl = int(base[j + 5])
    if lvl < 0 or lvl > 3:
        raise ValueError(f"level out of range 0..3: {lvl} ({name})")
    return lvl


def _list_images(images_dir: Path) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    for p in sorted(images_dir.rglob("*.jpg")):
        try:
            y = _label_from_filename(p.name)
        except Exception:
            continue
        items.append((p, y))
    if not items:
        raise RuntimeError(f"no labeled images found under {images_dir}")
    return items


def _split(items: List[Tuple[Path, int]], seed: int = 1337) -> Tuple[list, list]:
    rng = random.Random(seed)
    it = items[:]
    rng.shuffle(it)
    n_val = max(1, int(0.2 * len(it)))
    return it[n_val:], it[:n_val]


def _load_tensor(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224))
    x = np.asarray(img).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return x


@dataclass(frozen=True)
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    class_counts: list[float]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    images_dir = Path(args.images)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    items = _list_images(images_dir)
    train_it, val_it = _split(items, seed=args.seed)

    # Preload tensors (keeps training simple/fast on CPU)
    def preload(it):
        X = np.zeros((len(it), 224, 224, 3), dtype=np.float32)
        y = np.zeros((len(it),), dtype=np.int64)
        for i, (p, yy) in enumerate(it):
            X[i] = _load_tensor(p)
            y[i] = yy
            if (i + 1) % 200 == 0 or (i + 1) == len(it):
                print(f"loaded {i+1}/{len(it)}", flush=True)
        return X, y

    Xtr, ytr = preload(train_it)
    Xva, yva = preload(val_it)

    class_counts = np.bincount(ytr, minlength=4).astype(np.float32)
    w = (class_counts.sum() / np.clip(class_counts, 1.0, None))
    w = w / w.mean()

    import torch
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(self, num_classes: int = 4) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):  # type: ignore[no-untyped-def]
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    device = torch.device("cpu")
    model = SmallCNN(num_classes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))

    def batches(X, y, bs):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        for s in range(0, len(X), bs):
            j = idx[s : s + bs]
            yield X[j], y[j]

    best_acc = -1.0
    history: list[EpochLog] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in batches(Xtr, ytr, int(args.batch_size)):
            x = torch.as_tensor(xb, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            yt = torch.as_tensor(yb, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, yt)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            yva_t = torch.as_tensor(yva, dtype=torch.long, device=device)
            logits = model(xva_t)
            val_loss = float(loss_fn(logits, yva_t).item())
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            val_acc = float((pred == yva).mean())

        log = EpochLog(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)) if train_losses else float("nan"),
            val_loss=val_loss,
            val_acc=val_acc,
            class_counts=[float(x) for x in class_counts.tolist()],
        )
        history.append(log)
        print(json.dumps(asdict(log)), flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model": model.state_dict(),
                "meta": {"label_mapping": {"0": "level0", "1": "level1", "2": "level2", "3": "level3"}},
            }
            torch.save(ckpt, str(out_path))
            with out_path.with_suffix(".history.json").open("w") as f:
                json.dump([asdict(h) for h in history], f, indent=2)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

