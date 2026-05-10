"""
train.py — train ServeDetector and RallyClassifier from extracted poses.

Run on Kaggle (GPU) after uploading poses.pkl as a dataset.

Usage:
    python train.py [--poses path/to/poses.pkl] [--out path/to/weights/]

Outputs:
    serve_detector.pt    — binary serve classifier
    rally_classifier.pt  — 4-class rally classifier
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from collections import Counter
import sys

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.models.shot_classifier import _TennisCNN, LANDMARKS_PER_PLAYER, RALLY_CLASSES

# ── Config ────────────────────────────────────────────────────────────────────

SERVE_EPOCHS  = 40
RALLY_EPOCHS  = 60
BATCH_SIZE    = 32
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.15
SEED          = 42

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ───────────────────────────────────────────────────────────────────

class PoseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Time jitter: randomly shift sequence by ±2 frames
            shift = np.random.randint(-2, 3)
            if shift > 0:
                x = torch.cat([torch.zeros(shift, x.shape[1]), x[:-shift]], dim=0)
            elif shift < 0:
                x = torch.cat([x[-shift:], torch.zeros(-shift, x.shape[1])], dim=0)
            # Gaussian noise
            x = x + torch.randn_like(x) * 0.01
        return x, self.y[idx]


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    counts  = Counter(y.tolist())
    weights = np.array([1.0 / counts[yi] for yi in y.tolist()], dtype=np.float32)
    return WeightedRandomSampler(weights, len(weights))


def train_split(X, y, val_split=VAL_SPLIT):
    np.random.seed(SEED)
    n   = len(X)
    idx = np.random.permutation(n)
    val_n = max(1, int(n * val_split))
    return (
        X[idx[val_n:]], y[idx[val_n:]],
        X[idx[:val_n]], y[idx[:val_n]],
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    X_train, y_train,
    X_val,   y_val,
    epochs: int,
    label: str,
    class_names: list,
) -> nn.Module:
    sampler    = make_weighted_sampler(y_train)
    train_ds   = PoseDataset(X_train, y_train, augment=True)
    val_ds     = PoseDataset(X_val,   y_val,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Class-weighted loss
    class_counts = Counter(y_train.tolist())
    total        = sum(class_counts.values())
    weights      = torch.tensor(
        [total / (len(class_counts) * class_counts.get(i, 1)) for i in range(len(class_names))],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        # — Train
        model.train()
        train_loss = 0.0
        correct    = 0
        total_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
            correct    += (out.argmax(1) == yb).sum().item()
            total_seen += len(xb)
        scheduler.step()
        train_acc  = correct / total_seen
        train_loss /= total_seen

        # — Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out      = model(xb)
                val_loss += criterion(out, yb).item() * len(xb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total   += len(xb)
        val_acc  = val_correct / val_total
        val_loss /= val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label}] epoch {epoch:3d}/{epochs}  "
                  f"train loss={train_loss:.4f} acc={train_acc:.3f}  "
                  f"val acc={val_acc:.3f}  best={best_val_acc:.3f}")

    print(f"  [{label}] best val acc = {best_val_acc:.3f}")
    model.load_state_dict(best_state)
    return model


# ── Per-class accuracy ────────────────────────────────────────────────────────

def per_class_accuracy(model, X, y, class_names):
    model.eval()
    ds     = PoseDataset(X, y)
    loader = DataLoader(ds, batch_size=64)
    preds  = []
    truths = []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())
            truths.extend(yb.tolist())
    print("  Per-class accuracy:")
    for i, name in enumerate(class_names):
        mask = [t == i for t in truths]
        if not any(mask):
            continue
        correct = sum(p == i for p, t in zip(preds, truths) if t == i)
        print(f"    {name:12s}: {correct}/{sum(mask)} = {correct/sum(mask):.2%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", default=str(Path(__file__).parent / "poses.pkl"))
    parser.add_argument("--out",   default=str(ROOT / "ml" / "models" / "weights"))
    args = parser.parse_args()

    print(f"Loading poses from {args.poses}")
    with open(args.poses, 'rb') as f:
        data = pickle.load(f)

    X       = data['X']         # (N, 16, 132)
    y_serve = data['y_serve']   # (N,) binary
    y_rally = data['y_rally']   # (N,) 4-class, -1 for non-rally shots
    labels  = data['labels']
    classes = data.get('classes', RALLY_CLASSES)

    print(f"Dataset: {len(X)} sequences, shape {X.shape}")
    print(f"Label distribution: {Counter(labels)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ── 1. Serve Detector ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training ServeDetector  (device={device})")
    print(f"  Positives (serve):     {y_serve.sum()}")
    print(f"  Negatives (non-serve): {(y_serve == 0).sum()}")

    X_tr, y_tr, X_va, y_va = train_split(X, y_serve)
    serve_model = _TennisCNN(input_size=LANDMARKS_PER_PLAYER, num_classes=2).to(device)
    serve_model = train_model(serve_model, X_tr, y_tr, X_va, y_va, SERVE_EPOCHS,
                               label="ServeDetector", class_names=["NonServe", "Serve"])
    per_class_accuracy(serve_model, X_va, y_va, ["NonServe", "Serve"])

    # Tune threshold on validation set
    serve_model.eval()
    val_ds     = PoseDataset(X_va, y_va)
    val_loader = DataLoader(val_ds, batch_size=64)
    all_probs, all_labels_val = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            probs = torch.softmax(serve_model(xb.to(device)), dim=1)[:, 1].cpu().tolist()
            all_probs.extend(probs)
            all_labels_val.extend(yb.tolist())

    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.3, 0.75, 0.05):
        preds = [1 if p >= thr else 0 for p in all_probs]
        tp = sum(p == 1 and t == 1 for p, t in zip(preds, all_labels_val))
        fp = sum(p == 1 and t == 0 for p, t in zip(preds, all_labels_val))
        fn = sum(p == 0 and t == 1 for p, t in zip(preds, all_labels_val))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"  Best serve threshold: {best_thr:.2f}  (F1={best_f1:.3f})")

    torch.save({
        'model_state_dict': serve_model.state_dict(),
        'input_size':       LANDMARKS_PER_PLAYER,
        'threshold':        float(best_thr),
        'classes':          ["NonServe", "Serve"],
    }, out_dir / "serve_detector.pt")
    print(f"  Saved → {out_dir}/serve_detector.pt")

    # ── 2. Rally Classifier ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training RallyClassifier  (device={device})")

    # Only use samples with a valid rally label
    mask    = y_rally >= 0
    X_r     = X[mask]
    y_r     = y_rally[mask]
    labels_r = [labels[i] for i in range(len(labels)) if mask[i]]
    print(f"  Rally samples: {len(X_r)}")
    print(f"  Class distribution: {Counter(labels_r)}")

    if len(X_r) < 20:
        print("  ⚠ Not enough rally samples — skipping rally classifier training")
        return

    X_tr, y_tr, X_va, y_va = train_split(X_r, y_r)
    rally_model = _TennisCNN(input_size=LANDMARKS_PER_PLAYER, num_classes=len(classes)).to(device)
    rally_model = train_model(rally_model, X_tr, y_tr, X_va, y_va, RALLY_EPOCHS,
                               label="RallyClassifier", class_names=classes)
    per_class_accuracy(rally_model, X_va, y_va, classes)

    torch.save({
        'model_state_dict': rally_model.state_dict(),
        'input_size':       LANDMARKS_PER_PLAYER,
        'classes':          classes,
    }, out_dir / "rally_classifier.pt")
    print(f"  Saved → {out_dir}/rally_classifier.pt")

    print(f"\n✅ Done. Both models saved to {out_dir}")


if __name__ == "__main__":
    main()
