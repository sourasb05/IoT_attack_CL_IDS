import os
import logging
import warnings
from copy import deepcopy
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from termcolor import colored
from utils import save_results_as_json
import evaluation as evaluate

warnings.filterwarnings("ignore")


# ===================================
# Elastic Weight Consolidation Class
# ===================================
class EWC:
    def __init__(self, model, dataloader, device, lambda_=1150, fisher_n_samples=None):
        self.model = model
        self.device = device
        self.lambda_ = lambda_

        # only track parameters that require grad
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        # snapshot of parameters after finishing this task
        self.prev_params = {n: p.clone().detach() for n, p in self.params.items()}
        # compute Fisher information
        self.fisher = self._compute_fisher(dataloader, fisher_n_samples)

    def _compute_fisher(self, dataloader, n_samples=None):
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.params.items()}
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        count = 0
        for X_batch, y_batch in dataloader:
            if n_samples is not None and count >= n_samples:
                break
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.model.zero_grad()
            outputs, _ = self.model(X_batch)
            loss = criterion(outputs, y_batch.long())
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1

        # average
        for n in fisher:
            fisher[n] /= count if count > 0 else 1
        return fisher

    def penalty(self):
        reg = 0.0
        for n, p in self.params.items():
            delta = p - self.prev_params[n]
            reg += (self.fisher[n] * delta ** 2).sum()
        return self.lambda_ * reg


# ===================================
# Similarity and Zero-shot Utilities
# ===================================
def compute_domain_similarity(model, loader_a, loader_b, device):
    """
    Compute cosine similarity between the mean penultimate‐layer features of two domains.
    Your model must return (logits, features) when called.
    """
    model.eval()
    feats_a, feats_b = [], []

    def extract(loader, feats_list):
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(device)
                logits, features = model(X)      # features: tensor of shape [batch, feat_dim]
                feats_list.append(features.cpu())
        return torch.cat(feats_list, dim=0)

    feats_a = extract(loader_a, [])
    feats_b = extract(loader_b, [])
    mu_a, mu_b = feats_a.mean(0), feats_b.mean(0)
    sim = F.cosine_similarity(mu_a.unsqueeze(0), mu_b.unsqueeze(0)).item()
    return sim


def zero_shot_performance(model, loader, device, test_domain, train_domain):
    """
    Run the current model on `loader` without any training and return F1 score.
    """
    model.eval()
    all_y, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs, _ = model(X)
            _, pred = outputs.max(1)
            prob = F.softmax(outputs, dim=1)[:, 1]

            all_y.extend(    y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())

    metrics = evaluate.evaluate_metrics(
        np.array(all_y), np.array(all_pred), np.array(all_prob),
        test_domain, train_domain
    )
    return metrics["f1"]



# ===================================
# Main Training Function with Similarity‐Based Decision
# ===================================
def train_domain_incremental_model(
    scenario,
    device,
    train_domain_loader,
    test_domain_loader,
    full_domain_loader,   # unused, kept for compatibility
    model,
    num_epochs=20,
    learning_rate=0.01,
    patience=3,
    forgetting_threshold=0.01,
    tau=0.5,               # similarity threshold
    delta=0.6              # generalization (zero‐shot) threshold
):
    """
    Implements Algorithm: Continual Domain‐Incremental Learning with Similarity‐Based Decision
    """
    exp_no = 0 # Placeholder for experiment number, if needed
    # ensure device
    model.to(device)

    # snapshot of initial weights to re‐initialize for scratch training
    initial_state = deepcopy(model.state_dict())

    criterion = nn.CrossEntropyLoss()

    
    ewc_list = []  # Track EWC penalties from past tasks
    # track history for each test domain
    performance_matrix  = {} # test_domain: [] for test_domain in test_domain_loader.keys()}

    # track which domains remain
    remaining_train_domains = set(train_domain_loader.keys())
    used_train_domains = set()
    train_domain_order = []

    print(f"remaining_train_domains: {remaining_train_domains}")
    prev_loader = None
    prev_domain = None

    logging.info(f"Will train on {len(remaining_train_domains)} domains in scenario '{scenario}'")
    remaining = 0
    # while remaining < 2:
    while remaining_train_domains:
        

        # ─── 1) SELECT NEXT DOMAIN ───────────────────────────────

        train_domain = remaining_train_domains.pop()
        used_train_domains.add(train_domain)
        train_domain_order.append(train_domain)
        
        logging.info(colored(f"\n=== Domain: {train_domain} ({len(used_train_domains)}/{len(train_domain_loader)}) ===", "cyan"))

        # ─── 2) SIMILARITY CHECK ──────────────────────────────────
        is_continual = True
        if prev_loader is not None:
            S_t = compute_domain_similarity(model, prev_loader, train_domain_loader[train_domain][exp_no], device)
            logging.info(f"Sim({prev_domain}→{train_domain_loader[train_domain][exp_no]}) = {S_t:.4f}")

            if S_t >= tau:
                # dissimilar: zero‐shot first
                G_t = zero_shot_performance(model, test_domain_loader[train_domain][exp_no], device, train_domain, prev_domain)
                logging.info(colored(f"Zero‐shot F1 on {train_domain}: {G_t:.4f}", "magenta"))

                if G_t >= delta:
                    # good zero‐shot: no re‐training needed so skip domain
                    logging.info(colored(f"Skipping {train_domain} (Good zero-shot)", "green"))
                    prev_loader, prev_domain = train_domain_loader[train_domain][exp_no], train_domain
                    performance_matrix.setdefault(train_domain, []).append(mets["f1"])

                    continue
                    
                else:
                    logging.info(colored(f"Generalization is poor so Re-training is needed on {train_domain}", "red"))
                    is_continual = True
                    
            else:
                logging.info(colored(f"Continual training on {train_domain}", "blue"))
        else:
            # very first domain: scratch
            logging.info(colored(f"Initial training (scratch) on {train_domain}", "yellow"))
            model.load_state_dict(initial_state)
            ewc_list.clear()
            is_continual = False

        def total_ewc_penalty():
            """Calculate total EWC penalty from all past tasks."""
            return sum(ewc.penalty() for ewc in ewc_list) if ewc_list else 0.0

        # ─── 3) TRAINING LOOP ────────────────────────────────────
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_f1 = -float("inf")
        epochs_no_improve = 0
        best_state = None

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(train_domain_loader[train_domain][exp_no]):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs, _ = model(X_batch)
                
                loss = criterion(outputs, y_batch.long()) + total_ewc_penalty()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= (i + 1)
            # validate on own domain
            all_y_true, all_pred, all_prob = [], [], []
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in test_domain_loader[train_domain][0]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs, _ = model(X_batch)
                    _, pred = outputs.max(1)
                    all_y_true.extend(y_batch.cpu().numpy())
                    all_pred.extend(pred.cpu().numpy())
                    all_prob.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())

            mets = evaluate.evaluate_metrics(
                np.array(all_y_true), np.array(all_pred), np.array(all_prob),
                train_domain, train_domain)
            curr_f1 = mets["f1"]
            logging.info(f"Epoch {epoch+1}/{num_epochs} | F1={curr_f1:.4f}")

            if curr_f1 > best_f1:
                best_f1 = curr_f1
                best_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

        # restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # save a checkpoint
        save_dir = f"models_selective_CL/{train_domain}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

        # Create EWC object for this domain and append to list
        ewc_data_loader = train_domain_loader[train_domain][exp_no]
        ewc_instance = EWC(model, ewc_data_loader, device, lambda_=900, fisher_n_samples=None)
        ewc_list.append(ewc_instance)

                # ─── 4) EVALUATE ON SEEN TEST DOMAINS ───────────────
        seen_domains = train_domain_order[:train_domain_order.index(train_domain) + 1]
        logging.info(f"seem_domains: {seen_domains}")
        for td in seen_domains:
            all_y, all_pred, all_prob = [], [], []
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in test_domain_loader[td][0]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs, _ = model(X_batch)
                    _, pred = outputs.max(1)
                    prob = F.softmax(outputs, dim=1)[:, 1]

                    all_y.extend(y_batch.cpu().numpy())
                    all_pred.extend(pred.cpu().numpy())
                    all_prob.extend(prob.cpu().numpy())

            mets = evaluate.evaluate_metrics(
                np.array(all_y), np.array(all_pred), np.array(all_prob),
                td, train_domain
            )
            performance_matrix.setdefault(td, []).append(mets["f1"])
            logging.info(f"Performance matrix: {performance_matrix}")


        prev_loader, prev_domain = train_domain_loader[train_domain][exp_no], train_domain

        remaining += 1

        # ─── 6) BWT ───────────────────────────────────
    
    BWT_dict = {}
    for k, vals in performance_matrix.items():
        if len(vals) <= 1:
            # single‐value: leave it as is
            BWT_dict[k] = vals[0] if vals else None
        else:
            last = vals[-1]
            # compute differences to the last element
            diffs = [last - x for x in vals[:-1]]
            # average them
            BWT_dict[k] = sum(diffs) / len(diffs)
    BWT = np.mean(list(BWT_dict.values()))
    logging.info(f"Final BWT: {BWT:.4f}")

    # save results
    results = {
        "order": train_domain_order,
        "performance_matrix": performance_matrix,
        "BWT": BWT
    }
    save_results_as_json(results, filename="experiment_results_EWC_similarity.json")
    logging.info("Training complete, results saved.")
