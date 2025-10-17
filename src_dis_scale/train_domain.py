# ----------------------------
# Train a single domain from scratch
# ----------------------------
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import models as models


def _as_logits(out):
    """Return logits tensor from model output (handle tuple/list)."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out
# ----------------------------
# Train one epoch
# ----------------------------
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device).long()
        optimizer.zero_grad()
        logits = _as_logits(model(X))
        loss = criterion(logits, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total) if total > 0 else 0.0
    return avg_loss, acc


def train_domain(
    key,
    train_loader,
    test_loader,
    device,
    arch="LSTM",
    hidden_dim=10,
    output_dim=2,
    num_layers=1,
    fc_hidden_dim=10,
    lr=1e-3,
    epochs=10,
    patience=3,
    checkpoints_dir="checkpoints"
):
    """
    Trains a fresh model for one domain key and returns:
      - best_state (dict with model + meta)
      - history (loss/acc per epoch)
      - best_val_acc (float)
    Also saves the checkpoint to {checkpoints_dir}/{key}_LSTM.pth
    """
    os.makedirs(checkpoints_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    # Infer input dimension from a sample batch
    X_sample, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]

    # Build model
    if arch.upper() == "LSTM":
        model = models.LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            fc_hidden_dim=fc_hidden_dim
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state = None
    bad_epochs = 0
    patience = max(1, patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        # quick accuracy-only eval (fast)
        with torch.no_grad():
            model.eval()
            correct, total, val_loss_accum = 0, 0, 0.0
            for Xv, yv in test_loader:
                Xv, yv = Xv.to(device), yv.to(device).long()
                logits = _as_logits(model(Xv))
                loss_v = criterion(logits, yv)
                val_loss_accum += loss_v.item() * Xv.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yv).sum().item()
                total += yv.size(0)
            va_loss = val_loss_accum / max(1, total)
            va_acc = correct / max(1, total) if total > 0 else 0.0

        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"[{key}] Epoch {epoch:02d}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")

        # Early stopping on val acc
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "meta": {
                    "domain": key,
                    "epoch": epoch,
                    "val_acc": float(best_val_acc),
                    "input_dim": int(input_dim),
                    "arch": arch,
                    "hidden_dim": int(hidden_dim),
                    "output_dim": int(output_dim),
                    "num_layers": int(num_layers),
                    "fc_hidden_dim": int(fc_hidden_dim),
                    "lr": float(lr),
                    "epochs": int(epochs),
                }
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{key}] Early stopping at epoch {epoch}.")
                break

    # Save best checkpoint for this domain
    ckpt_path = os.path.join(checkpoints_dir, f"{key}_{arch.upper()}.pth")
    torch.save(best_state if best_state is not None else {
        "model": model.state_dict(),
        "meta": {"domain": key, "epoch": epochs, "val_acc": float(best_val_acc), "input_dim": int(input_dim)}
    }, ckpt_path)
    print(f"[{key}] Saved checkpoint: {ckpt_path}")

    # Load best weights back into model for final detailed evaluation
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    return model, best_state, history, best_val_acc, ckpt_path

