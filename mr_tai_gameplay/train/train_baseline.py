from __future__ import annotations
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from dataloader import ClipDataset
from models.r3d18_baseline import R3D18
from src.schemas import LABELS


def train(
    index_json: str,
    out_dir: str = "out_ckpt",
    epochs: int = 10,
    bs: int = 4,
    lr: float = 1e-4,
    device: str = "cpu",
):
    ds = ClipDataset(index_json)
    n = len(ds)
    n_train = int(0.9 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)

    model = R3D18().to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
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
            tr_loss += float(loss)

        # ---- validate ----
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
