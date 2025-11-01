from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

from train.dataloader import ClipDataset
from train.models.r3d18_baseline import R3D18
from src.schemas import LABELS


def _labels_for_split(index_json: str, split) -> list[str]:
    """
    Return the list of string labels for whichever split we're training on,
    without reaching into Dataset/Subset internals that annoy Pylance.
    """
    data = json.loads(Path(index_json).read_text())
    if isinstance(split, Subset):
        idxs = list(split.indices)
        data = [data[i] for i in idxs]
    return [row["label"] for row in data]


def train(
    index_json: str,
    out_dir: str = "out_ckpt",
    epochs: int = 10,
    bs: int = 4,
    lr: float = 1e-4,
    device: str = "cpu",
):
    ds = ClipDataset(index_json)

    # If dataset is tiny, train on all samples (no validation split)
    if len(ds) < 6:
        train_ds, val_ds = ds, None
    else:
        n = len(ds)
        n_train = int(0.9 * n)
        n_val = n - n_train
        train_ds, val_ds = random_split(ds, [n_train, n_val])

    # macOS note: num_workers=0 avoids spawn issues in small setups
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0) if val_ds else None

    # Head-only fine-tune by default (fast & stable for micro-datasets)
    model = R3D18(head_only=True).to(device)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # ---- Class-weighted loss (robust to Dataset/Subset types) ----
    try:
        split_labels = _labels_for_split(index_json, train_ds)
        freq = Counter(split_labels)
        weights = torch.tensor(
            [1.0 / max(freq.get(lbl, 1), 1) for lbl in LABELS], dtype=torch.float, device=device
        )
        # Normalize to keep average weight ~1
        weights = weights / weights.mean()
        crit = nn.CrossEntropyLoss(weight=weights)
        print(f"[INFO] Using class-weighted loss with frequencies: {dict(freq)}")
        print(f"[INFO] Weights (LABELS order): {weights.detach().cpu().numpy().tolist()}")
    except Exception as e:
        print(f"[WARN] Could not compute class weights ({e}); using unweighted loss.")
        crit = nn.CrossEntropyLoss()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        for x, y in tqdm(tl, desc=f"epoch {ep} train"):
            x = x.to(device)  # (B,C,T,H,W)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.detach().item()

        # ---- validate (optional) ----
        if vl is not None:
            model.eval()
            corr = 0
            tot = 0
            with torch.no_grad():
                for x, y in tqdm(vl, desc=f"epoch {ep} val"):
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    corr += int((pred == y).sum())
                    tot += int(y.numel())
            acc = corr / max(1, tot)
            print(f"epoch {ep}: train_loss={tr_loss / max(1, len(tl)):.4f}  val_acc={acc:.3f}")
        else:
            print(f"epoch {ep}: train_loss={tr_loss / max(1, len(tl)):.4f}  (no validation)")

        # save checkpoint
        ckpt_path = out / f"r3d18_ep{ep:02d}.pt"
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("index_json")
    p.add_argument("--out_dir", default="out_ckpt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    train(args.index_json, args.out_dir, args.epochs, args.bs, args.lr, args.device)
