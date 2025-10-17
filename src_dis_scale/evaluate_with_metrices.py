# ----------------------------
# Metrics
# ----------------------------
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import torch
import torch.nn.functional as F
from train_domain import _as_logits

@torch.no_grad()
def evaluate_with_metrics(model, loader, device, criterion):
    """
    Evaluate a classifier on a DataLoader and compute:
      - avg loss
      - accuracy
      - F1 (binary)
      - ROC-AUC (binary, safe if one class present)
      - Confusion matrix [[tn, fp], [fn, tp]]
      - Support counts
    """
    model.eval()
    all_logits = []
    all_labels = []
    running_loss, total = 0.0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device).long()
        logits = _as_logits(model(X))
        loss = criterion(logits, y)

        running_loss += loss.item() * X.size(0)
        total += y.size(0)

        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    avg_loss = running_loss / max(1, total)

    logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, 2)
    y_true = torch.cat(all_labels, dim=0).numpy() if all_labels else np.array([])

    if logits.numel() == 0 or y_true.size == 0:
        return {
            "val_loss": float(avg_loss),
            "accuracy": None,
            "f1": None,
            "roc_auc": None,
            "confusion_matrix": None,
            "support": {"negatives": 0, "positives": 0, "total": 0},
        }

    y_prob_pos = F.softmax(logits, dim=1)[:, 1].numpy()
    y_pred = np.argmax(logits.numpy(), axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="binary")

    try:
        roc = roc_auc_score(y_true, y_prob_pos)
    except ValueError:
        roc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "val_loss": float(avg_loss),
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": (None if roc is None else float(roc)),
        "confusion_matrix": cm.tolist(),
        "support": {
            "negatives": int((y_true == 0).sum()),
            "positives": int((y_true == 1).sum()),
            "total": int(len(y_true)),
        }
    }
