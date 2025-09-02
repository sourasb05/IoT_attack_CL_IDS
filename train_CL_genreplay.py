import os
import time
import random
import logging
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd  # only if you actually use it elsewhere
from tqdm import tqdm, trange
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

# Metrics (used in your eval utilities; keep if needed here)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, roc_curve, auc
)

# Your modules
from utils import save_results_as_json, _sync
import evaluation as evaluate
import evaluate_model
import result_utils as result_utils

# Silence warnings if you like
warnings.filterwarnings("ignore")

# ----------------------------
# LSTM-VAE for Generative Replay
# ----------------------------
class LSTMVAE(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, latent_dim=32, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # Encoder
        self.enc_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_init  = nn.Linear(latent_dim, hidden_dim)
        self.dec_lstm  = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.out_fc    = nn.Linear(hidden_dim, feature_dim)

    def encode(self, x):
        _, (h, _) = self.enc_lstm(x)  # h: (num_layers=1, B, H)
        h = h[-1]                      # (B, H)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: (B, latent_dim)
        h0 = self.dec_init(z).unsqueeze(0)      # (1, B, H)
        c0 = torch.zeros_like(h0)               # (1, B, H)
        # Start-of-seq zeros (teacher-forcing-free)
        seq0 = torch.zeros(z.size(0), self.seq_len, self.feature_dim, device=z.device)
        out, _ = self.dec_lstm(seq0, (h0, c0))  # (B, T, H)
        return self.out_fc(out)                 # (B, T, F)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    # MSE over sequence + KL
    recon_mse = F.mse_loss(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_mse + kl

def train_vae_on_dataset(vae, device, dataset, num_epochs=5, lr=1e-3, batch_size=64, log_prefix="VAE"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    for epoch in range(num_epochs):
        total = 0.0
        for xb, _ in loader:
            xb = xb.to(device)
            recon, mu, logvar = vae(xb)
            loss = vae_loss(recon, xb, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(1, len(loader))
        logging.info(f"{log_prefix} Epoch {epoch+1}/{num_epochs} | Loss: {avg:.4f}")

@torch.no_grad()
def generate_replay_samples(vae, n_samples, device):
    vae.eval()
    z = torch.randn(n_samples, vae.fc_mu.out_features, device=device)
    syn_seq = vae.decode(z)  # (N, T, F)
    return syn_seq

@torch.no_grad()
def teacher_pseudolabel(teacher_model, xb, device, architecture=None, domain_id=None):
    teacher_model.eval()
    xb = xb.to(device)
    if architecture == "LSTM_Attention_adapter":
        logits, _ = teacher_model(xb, domain_id=domain_id)
    else:
        logits, _ = teacher_model(xb)
    probs = F.softmax(logits, dim=1)
    y_hat = probs.argmax(dim=1)
    return y_hat.cpu()


def tdim_gr_random(args, run_wandb, train_domain_loader, test_domain_loader, device,
                   model, exp_no, num_epochs=500, learning_rate=0.01, patience=3,
                   # GR-specific knobs:
                   vae_hidden=64, vae_latent=32, window_size=10, num_features=14,
                   vae_epochs=5, vae_lr=1e-3, replay_samples_per_epoch=0,
                   replay_ratio=0.5,  # fraction of a combined batch that is replay (approx)
                   use_teacher_labels=True):
    """
    Domain-Incremental training with Generative Replay.

    - Trains an LSTM-VAE on (current real + previously generated) sequences.
    - Each epoch, generate synthetic sequences and mix with current real data.
    - If a previous best model exists, use it as a frozen 'teacher' to pseudo-label the synthetic sequences.

    Args expected:
        - train_domain_loader[domain]: DataLoader of (X,y) for current domain
        - test_domain_loader[domain]: DataLoader of (X,y) for eval
        - model(X) -> (logits, hidden)
    """
    exp_no = exp_no
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    performance_stability  = {d: [] for d in test_domain_loader.keys()}
    performance_plasticity = {d: [] for d in test_domain_loader.keys()}
    domain_training_cost   = {d: [] for d in test_domain_loader.keys()}

    seen_domain = set()
    train_domain_order = list(train_domain_loader.keys())
    domain_to_id = {name: i for i, name in enumerate(train_domain_loader.keys())}

    # ---- W&B config ----
    run_wandb.config.update({
        "batch_size": args.batch_size,
        "Loss Function": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "train_domains": train_domain_order,
        "algorithm": "GenerativeReplay",
        "replay_ratio": replay_ratio,
        "replay_samples_per_epoch": replay_samples_per_epoch,
        "vae_epochs": vae_epochs,
        "vae_lr": vae_lr,
        "vae_hidden": vae_hidden,
        "vae_latent": vae_latent,
        "window_size": window_size,
        "num_features": num_features,
        "use_teacher_labels": use_teacher_labels,
    })
    run_wandb.watch(model, criterion=criterion, log="all", log_freq=50)

    previous_domain   = None
    best_model_state  = None
    teacher_model     = None
    replay_vae        = None               # LSTM-VAE instance
    vae_memory_cache  = None               # Tensor of sequences to keep training VAE on (real + syn)

    for idx, train_domain in enumerate(tqdm(list(train_domain_loader.keys()),
                                            desc="Train Domains", total=len(train_domain_loader))):
        domain_id = domain_to_id[train_domain]
        domain_epoch = 0
        wandb.define_metric(f"{train_domain}/epoch")
        wandb.define_metric(f"{train_domain}/*", step_metric=f"{train_domain}/epoch")

        # Build teacher from previous best
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            teacher_model = deepcopy(model).to(device)
            for p in teacher_model.parameters():
                p.requires_grad = False
            teacher_model.eval()
        else:
            teacher_model = None

        # Pre-train eval (plasticity) on current domain
        if idx != 0:
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
            else:
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)
            current_f1 = metrics["f1"]
            performance_plasticity[train_domain].append(current_f1)
            logging.info(f"[Pre] {train_domain} F1 (from prev best): {current_f1:.4f}")
            run_wandb.log({f"{train_domain}/pretrain_f1": float(current_f1)})

        logging.info(f"====== Training on Domain: {train_domain} (Generative Replay) ======")
        best_f1 = -float("inf")
        epochs_no_improve = 0

        _sync(device)
        t0 = time.perf_counter()

        # Prepare real loader
        real_loader = train_domain_loader[train_domain]  # expected DataLoader of (X,y)

        # Initialize VAE the first time we see data
        if replay_vae is None:
            replay_vae = LSTMVAE(
                feature_dim=num_features, hidden_dim=vae_hidden,
                latent_dim=vae_latent, seq_len=window_size
            ).to(device)

        # ---------- Epoch loop ----------
        for epoch in trange(num_epochs, desc="training Epochs"):
            model.train()
            domain_epoch += 1
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            i = -1

            # 1) (Re)train VAE briefly each epoch on cached memory + current real
            with torch.no_grad():
                # collect current real X
                current_real_X = []
                for Xb, _ in real_loader:
                    current_real_X.append(Xb)
                if len(current_real_X) > 0:
                    current_real_X = torch.cat(current_real_X, dim=0)
                else:
                    current_real_X = torch.empty(0, window_size, num_features)

                if vae_memory_cache is None:
                    vae_memory_cache = current_real_X
                else:
                    if current_real_X.numel() > 0:
                        vae_memory_cache = torch.cat([vae_memory_cache, current_real_X], dim=0)

            vae_dataset = TensorDataset(vae_memory_cache.cpu(), torch.zeros(len(vae_memory_cache), dtype=torch.long))
            train_vae_on_dataset(replay_vae, device, vae_dataset,
                                 num_epochs=vae_epochs, lr=vae_lr, batch_size=max(32, args.batch_size//2),
                                 log_prefix=f"VAE[{train_domain}]")

            # 2) Build mixed loader: real + synthetic
            # Generate synthetic sequences
            syn_X = None
            syn_y = None
            if replay_samples_per_epoch > 0:
                syn_X = generate_replay_samples(replay_vae, replay_samples_per_epoch, device=device)
                # Pseudo-label synthetic with teacher if available, else fallback to zeros
                if use_teacher_labels and (teacher_model is not None):
                    syn_y = teacher_pseudolabel(teacher_model, syn_X, device,
                                                architecture=args.architecture,
                                                domain_id=domain_id if args.architecture=="LSTM_Attention_adapter" else None)
                else:
                    syn_y = torch.zeros(syn_X.size(0), dtype=torch.long)  # fallback

                dataset_syn  = TensorDataset(syn_X.cpu(), syn_y.cpu())
                dataset_real = real_loader.dataset
                combined = ConcatDataset([dataset_real, dataset_syn])

                # Approximate replay ratio by oversampling synthetic or adjusting batch size composition
                loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True, drop_last=False)
            else:
                loader = real_loader

            # 3) Train classifier on mixed batches
            for i, (X_batch, y_batch) in enumerate(loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                if args.architecture == "LSTM_Attention_adapter":
                    outputs, _ = model(X_batch, domain_id=domain_id)
                else:
                    outputs, _ = model(X_batch)

                loss = criterion(outputs, y_batch.long())
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= (i + 1) if i >= 0 else 1
            _sync(device)
            epoch_time = time.perf_counter() - epoch_start
            logging.info(f"[{train_domain}] | Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")    

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/train_loss": float(epoch_loss),
                f"{train_domain}/epoch_time_s": float(epoch_time),      
            })

            # 4) Validation on same domain
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_domain_loader[train_domain]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    if args.architecture == "LSTM_Attention_adapter":
                        logits, _ = model(X_batch, domain_id=domain_id)
                    else:
                        logits, _ = model(X_batch)
                    loss = criterion(logits, y_batch.long())
                    probs = F.softmax(logits, dim=1)
                    pred  = probs.argmax(dim=1)

                    all_y_true.extend(y_batch.cpu().numpy())
                    all_y_pred.extend(pred.cpu().numpy())
                    # class-1 prob for ROC
                    all_y_prob.extend(probs[:, 1].detach().cpu().numpy())
                    test_loss += loss.item()

            test_loss /= max(1, len(test_domain_loader[train_domain]))
            metrics = evaluate.evaluate_metrics(
                np.array(all_y_true), np.array(all_y_pred),
                np.array(all_y_prob), train_domain, train_domain
            )
            current_f1 = metrics["f1"]
            current_auc_roc = metrics["roc_auc"]

            logging.info(f"[{train_domain}] | Epoch: {epoch+1}/{num_epochs} | "
                         f"Test Loss: {test_loss:.4f} | F1: {current_f1:.4f} | AUC-ROC: {current_auc_roc:.4f}")
            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/val_loss": float(test_loss),
                f"{train_domain}/val_f1": float(current_f1),
                f"{train_domain}/val_ROC_AUC": float(current_auc_roc)
            })

            # Early stopping
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
                logging.info(f"New best F1 for {train_domain}: {best_f1:.4f}")
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement. Count: {epochs_no_improve}")
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping for {train_domain} at epoch {epoch+1}")
                break

        # ----- end epochs -----
        _sync(device)
        domain_training_time = time.perf_counter() - t0
        logging.info(f"Training time for {train_domain}: {domain_training_time:.2f} s")
        domain_training_cost[train_domain].append(domain_training_time)

        # Restore best and save checkpoint
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model_save_path = f"models/exp_no_{exp_no}_{args.architecture}_{args.algorithm}_{args.scenario}/best_model_after_{train_domain}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_state, model_save_path)
            logging.info(f"Best model for {train_domain} saved to {model_save_path}")
            previous_domain = train_domain
        else:
            logging.info(f"No improvement for {train_domain}. Model not saved.")

        # Evaluate best on current domain
        model.eval()
        if args.architecture == "LSTM_Attention_adapter":
            best_metrices = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
        else:
            best_metrices = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)
        cur_f1 = best_metrices["f1"]
        performance_plasticity[train_domain].append(cur_f1)
        performance_stability[train_domain].append(cur_f1)
        logging.info(f" F1 (best) : {train_domain}: {cur_f1:.4f}")

        # Stability: evaluate on all seen (previous) domains
        logging.info(f"====== Evaluating on all previous domains after training on {train_domain} ======")
        seen_domain.add(train_domain)
        for test_domain in tqdm(seen_domain, desc="Stability test"):
            if test_domain == train_domain:
                continue
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=domain_to_id[test_domain])
            else:
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=None)
            f1_seen = metrics["f1"]
            performance_stability[test_domain].append(f1_seen)
            logging.info(f"performance_stability | {test_domain}: {performance_stability[test_domain]}")

        print(f"====== Finished Training on Domain: {train_domain} ======")

    # ----- Final BWT / FWT -----
    logging.info(f"====== Final Metrics after training on all domains ======")
    bwt_values, bwt_dict, bwt_values_dict = result_utils.compute_BWT(performance_stability, train_domain_order)
    fwt_values, fwt_dict = result_utils.compute_FWT(performance_plasticity, train_domain_order)
    logging.info(f"\n BWT: {bwt_values}")
    logging.info(f"\n BWT per-domain list: {bwt_dict}")
    logging.info(f"\n FWT: {fwt_values}")
    logging.info(f"\n FWT per-domain: {fwt_dict}")

    results_to_save = {
        "exp_no": exp_no,
        "performance_stability": performance_stability,
        "performance_m": performance_plasticity,
        "BWT_values": bwt_values,
        "BWT_dict": bwt_dict,
        "FWT_values": fwt_values,
        "FWT_dict": fwt_dict,
        "train_domain_order": train_domain_order,
        "domain_training_cost": domain_training_cost,
    }
    save_results_as_json(results_to_save, filename=f"{exp_no}_experiment_results_{args.architecture}_{args.algorithm}_{args.scenario}.json")
    logging.info("Final training complete. Results saved.")

    run_wandb.summary["BWT/list"] =  bwt_values
    run_wandb.summary["FWT/list"] =  fwt_values
