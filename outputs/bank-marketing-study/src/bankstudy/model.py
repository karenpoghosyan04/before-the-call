"""Explicit PyTorch training, validation-AP early stopping, and portable checkpoints."""

import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        for width in hidden:
            layers.extend([nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(-1)  # one logit per row, including batch size 1


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def resolve_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    raise ValueError(f"Unsupported or unavailable device: {requested}")


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MLP(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict(model, x, device="cpu"):
    model.eval()
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    return torch.sigmoid(model(tensor)).cpu().numpy()


def train_mlp(x_train, y_train, x_val, y_val, config, weight, seed, path, device="cpu"):
    start = time.perf_counter()
    seed_everything(seed)
    architecture = {"input_dim": x_train.shape[1], "hidden": config["hidden"], "dropout": config["dropout"]}
    model = MLP(**architecture).to(device)
    xt, yt, xv, yv = [torch.as_tensor(a, dtype=torch.float32, device=device)
                      for a in (x_train, y_train, x_val, y_val)]
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum()) if weight == "balanced" else 1.0
    objective = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    unweighted = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best, stale, history, best_epoch = -np.inf, 0, [], 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(xt), generator=generator).to(device)
        loss_sum = 0.0
        for batch in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(xt[batch]), yt[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val_logits = model(xv)
            val_ap = float(average_precision_score(y_val, torch.sigmoid(val_logits).cpu().numpy()))
            history.append({"epoch": epoch, "train_objective": loss_sum / len(xt),
                            "train_bce": float(unweighted(model(xt), yt).item()),
                            "validation_bce": float(unweighted(val_logits, yv).item()),
                            "validation_ap": val_ap})
        if val_ap > best + config["min_delta"]:
            best, stale, best_epoch = val_ap, 0, epoch
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "architecture": architecture, "seed": seed, "weight": weight,
                        "pos_weight": pos_weight, "epoch": epoch, "validation_ap": val_ap,
                        "training_config": config}, path)
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model, _ = load_checkpoint(path, device)
    return model, {"seed": seed, "weight": weight, "pos_weight": pos_weight, "best_epoch": best_epoch,
                   "epochs_run": len(history), "history": history, "seconds": time.perf_counter() - start}
