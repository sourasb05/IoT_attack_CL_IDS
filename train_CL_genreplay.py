import os
import time
import sys
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

from utils import save_results_as_json, _sync
import evaluate_model
import evaluation as evaluate
import result_utils as result_utils

class LSTMVAE(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, latent_dim=32, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_init  = nn.Linear(latent_dim, hidden_dim)
        self.dec_lstm  = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.out_fc    = nn.Linear(hidden_dim, feature_dim)

    def encode(self, x):
        _, (h, _) = self.enc_lstm(x)     # h: (1, B, H)
        h = h[-1]                        # (B, H)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_len=None):
        # z: (B, latent_dim)
        T = self.seq_len if target_len is None else target_len
        h0 = self.dec_init(z).unsqueeze(0)           # (1,B,H)
        c0 = torch.zeros_like(h0)                    # (1,B,H)
        # Start-of-seq zeros (no teacher forcing)
        seq0 = torch.zeros(z.size(0), self.seq_len, self.feature_dim, device=z.device)
        out, _ = self.dec_lstm(seq0, (h0, c0))       # (B,T,H)
        return self.out_fc(out)                      # (B,T,F)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, target_len=x.size(1))
        return recon, mu, logvar



def vae_loss_beta(recon, x, mu, logvar, beta=1.0):
    recon_mse = F.mse_loss(recon, x, reduction="mean")   # mean is stabler for long seqs
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_mse + beta * kl

def train_vae_on_dataset(vae, device, dataset, num_epochs=5, lr=1e-3, batch_size=64,
                         beta_start=0.0, beta_end=1.0, log_prefix="VAE"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    for epoch in range(num_epochs):
        beta = beta_start + (beta_end - beta_start) * (epoch / max(1, num_epochs-1))
        total = 0.0
        for xb, _ in loader:
            xb = xb.to(device)
            recon, mu, logvar = vae(xb)
            loss = vae_loss_beta(recon, xb, mu, logvar, beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(1, len(loader))
        logging.info(f"{log_prefix} Epoch {epoch+1}/{num_epochs} | β={beta:.3f} | Loss: {avg:.4f}")



@torch.no_grad()
def teacher_predict_soft(teacher_model, xb, device, architecture=None, domain_id=None, T=2.0):
    teacher_model.eval()
    xb = xb.to(device)
    if architecture == "LSTM_Attention_adapter":
        logits, _ = teacher_model(xb, domain_id=domain_id)
    else:
        logits, _ = teacher_model(xb)
    return F.softmax(logits / T, dim=1)  # soft targets


def distillation_loss(student_logits, teacher_probs, T=2.0):
    # KL(student || teacher) with temperature scaling; use teacher as target
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    loss_kl = F.kl_div(log_p_s, teacher_probs, reduction="batchmean") * (T * T)
    return loss_kl



def tdim_gr_random(
    args, run_wandb, train_domain_loader, test_domain_loader, train_domain_order, device,
    model, exp_no, num_epochs=10, learning_rate=0.01, patience=3,
    vae_hidden=64, vae_latent=32, window_size=1, num_features=140,
    vae_epochs=5, vae_lr=1e-3,
    replay_samples_per_epoch=0,   # if 0 -> computed from r & real count
    replay_ratio=0.5,             # r: weight on *real* loss; (1-r) on replay loss
    use_teacher_labels=True, T=2.0
):
    """
    Domain-incremental training with Generative Replay (faithful to DGR):
      - No storage of old real data.
      - New VAE per domain trained on current real + replay from previous VAE.
      - Solver trained with CE on real and KL distillation on replay (soft targets).
      - Loss mixed by 'replay_ratio' r (r on real CE, 1-r on replay KL).
    """
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    performance_stability  = {d: [] for d in test_domain_loader.keys()}
    performance_plasticity = {d: [] for d in test_domain_loader.keys()}
    domain_training_cost   = {d: [] for d in test_domain_loader.keys()}

    # train_domain_order = list(train_domain_loader.keys())
    domain_to_id = {name: i for i, name in enumerate(train_domain_order)}

    run_wandb.config.update({
        "batch_size": args.batch_size,
        "Loss Function": "CE(real) + KL(replay)",
        "optimizer": "AdamW",
        "train_domains": train_domain_order,
        "algorithm": "GR",
        "replay_ratio_r": replay_ratio,
        "replay_samples_per_epoch": replay_samples_per_epoch,
        "vae_epochs": vae_epochs,
        "vae_lr": vae_lr,
        "vae_hidden": vae_hidden,
        "vae_latent": vae_latent,
        "window_size": window_size,
        "num_features": num_features,
        "distill_T": T,
        "use_teacher_labels": use_teacher_labels,
    })
    run_wandb.watch(model, criterion=criterion_ce, log="all", log_freq=50)

    best_model_state = None
    prev_solver = None    # frozen teacher (previous best solver)
    replay_vae  = None    # current VAE
    prev_vae    = None    # frozen previous VAE (for replay)

    seen_domain = set()

    for idx, train_domain in enumerate(tqdm(train_domain_order, desc="Train Domains", total=len(train_domain_order))):
        domain_id = domain_to_id[train_domain]
        if args.use_wandb:
            wandb.define_metric(f"{train_domain}/epoch")
            wandb.define_metric(f"{train_domain}/*", step_metric=f"{train_domain}/epoch")

        # Build teacher from previous best
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            prev_solver = deepcopy(model).to(device)
            for p in prev_solver.parameters():
                p.requires_grad = False
            prev_solver.eval()
        else:
            prev_solver = None

        # Pre-train eval (plasticity) on current domain (how well previous knowledge transfers)
        if idx != 0:
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
            else:
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)
            run_wandb.log({f"{train_domain}/pretrain_f1": float(metrics["f1"])})
            performance_plasticity[train_domain].append(metrics["f1"])
        logging.info(f"====== Training on Domain: {train_domain} (Generative Replay) ======")
        best_f1 = -float("inf")
        epochs_no_improve = 0

        _sync(device)
        t0 = time.perf_counter()

        # Prepare real loader
        real_loader = train_domain_loader[train_domain]  # DataLoader of (X,y)
        # Initialize a fresh VAE for this domain (student generator)
        replay_vae = LSTMVAE(
            feature_dim=num_features, hidden_dim=vae_hidden,
            latent_dim=vae_latent, seq_len=window_size
        ).to(device)

        # ===== Train VAE on current real + replay from prev VAE (no old real cache!) =====
        # Collect current real sequences once (optional sampling to limit memory)
        real_X_list = []
        for Xb, _ in real_loader:
            real_X_list.append(Xb)

        if len(real_X_list) > 0:
            X_current = torch.cat(real_X_list, dim=0)
            print(f"[DEBUG] X_current shape: {tuple(X_current.shape)}")
        else:
            X_current = torch.empty(0, window_size, num_features)
        X_current = X_current.to(device)
        
        # Generate replay inputs from previous VAE if available
        if prev_vae is not None and X_current.size(0) > 0:
            n_replay_for_gen = X_current.size(0)   # symmetric
            z_gen = torch.randn(n_replay_for_gen, prev_vae.latent_dim, device=device)
            X_replay_for_gen = prev_vae.decode(z_gen).detach()
            X_gen_train = torch.cat([X_current, X_replay_for_gen], dim=0)
        else:
            X_gen_train = X_current

        vae_dataset = TensorDataset(X_gen_train.detach().cpu(), torch.zeros(len(X_gen_train)))
        train_vae_on_dataset(
            replay_vae, device, vae_dataset,
            num_epochs=vae_epochs, lr=vae_lr, batch_size=max(32, args.batch_size//2),
            beta_start=0.0, beta_end=1.0, log_prefix=f"VAE[{train_domain}]"
        )

        # Freeze the trained VAE for this domain for solver replay generation
        replay_vae.eval()
        for p in replay_vae.parameters():
            p.requires_grad = False

        # ===== Epoch loop for solver =====
        domain_epoch = 0
        for epoch in trange(num_epochs, desc="training Epochs"):
            model.train()
            domain_epoch += 1
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            i = -1

            # Decide #synthetic samples for this epoch
            num_real = X_current.size(0)
            if replay_samples_per_epoch > 0:
                n_syn = replay_samples_per_epoch
            else:
                # choose n_syn so that expected loss mixing approximates r
                # r: real weight, (1-r): replay weight
                # Make counts proportional to weights (optional heuristic)
                n_syn = int(num_real * (1 - replay_ratio) / max(replay_ratio, 1e-6))
                n_syn = max(n_syn, args.batch_size)  # ensure some replay

            # Generate synthetic sequences (solver training)
            if n_syn > 0:
                z = torch.randn(n_syn, replay_vae.latent_dim, device=device)
                syn_X = replay_vae.decode(z).detach()
                if use_teacher_labels and (prev_solver is not None):
                    syn_soft = teacher_predict_soft(
                        prev_solver, syn_X, device,
                        architecture=args.architecture,
                        domain_id=domain_id if args.architecture=="LSTM_Attention_adapter" else None,
                        T=T
                    )  # probs
                else:
                    # fallback: uniform soft targets
                    num_classes = 2 if hasattr(args, "num_classes") is False else args.num_classes
                    syn_soft = torch.full((syn_X.size(0), num_classes), 1.0/num_classes, device=device)
            else:
                syn_X, syn_soft = None, None

            # Iterate over real batches; for each, also take a replay slice
            syn_ptr = 0
            syn_idx = torch.tensor([], device=device)  # Initialize as an empty tensor
            
            if syn_X is not None:
                assert syn_soft is not None, "syn_soft must be provided when syn_X is present"
                syn_idx = torch.randperm(syn_X.size(0), device=device)
                syn_ptr = 0


            for i, (Xr, yr) in enumerate(real_loader):
                Xr = Xr.to(device); yr = yr.to(device)

                # Optional: pick a replay chunk ~ matching batch size
                if syn_X is not None and syn_soft is not None:
                    take = min(args.batch_size, syn_X.size(0) - syn_ptr)
                    if take <= 0:
                        # reshuffle
                        syn_idx = torch.randperm(syn_X.size(0), device=device)
                        syn_ptr = 0
                        take = min(args.batch_size, syn_X.size(0))
                    idx_take = syn_idx[syn_ptr:syn_ptr+take]
                    Xs = syn_X[idx_take]
                    Ps = syn_soft[idx_take]
                    syn_ptr += take
                else:
                    Xs, Ps = None, None

                optimizer.zero_grad()

                # Real pass (CE)
                if args.architecture == "LSTM_Attention_adapter":
                    logits_real, _ = model(Xr, domain_id=domain_id)
                else:
                    logits_real, _ = model(Xr)
                loss_real = criterion_ce(logits_real, yr.long())

                # Replay pass (KL distillation)
                if Xs is not None:
                    if args.architecture == "LSTM_Attention_adapter":
                        logits_syn, _ = model(Xs, domain_id=domain_id)
                    else:
                        logits_syn, _ = model(Xs)
                    loss_replay = distillation_loss(logits_syn, Ps, T=T)
                else:
                    loss_replay = torch.tensor(0.0, device=device)

                # Mix by ratio r
                loss = replay_ratio * loss_real + (1.0 - replay_ratio) * loss_replay
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= (i + 1) if i >= 0 else 1
            _sync(device)
            epoch_time = time.perf_counter() - epoch_start
            logging.info(f"[{train_domain}] | Epoch [{epoch+1}/{num_epochs}] "
                         f"| Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/train_loss": float(epoch_loss),
                f"{train_domain}/epoch_time_s": float(epoch_time),
            })

            # Validation on this domain
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for Xb, yb in test_domain_loader[train_domain]:
                    Xb, yb = Xb.to(device), yb.to(device)
                    if args.architecture == "LSTM_Attention_adapter":
                        logits, _ = model(Xb, domain_id=domain_id)
                    else:
                        logits, _ = model(Xb)
                    loss_b = criterion_ce(logits, yb.long())
                    probs = F.softmax(logits, dim=1)
                    pred  = probs.argmax(dim=1)

                    all_y_true.extend(yb.cpu().numpy())
                    all_y_pred.extend(pred.cpu().numpy())
                    all_y_prob.extend(probs[:, 1].detach().cpu().numpy())
                    test_loss += loss_b.item()

            test_loss /= max(1, len(test_domain_loader[train_domain]))
            metrics = evaluate.evaluate_metrics(
                np.array(all_y_true), np.array(all_y_pred),
                np.array(all_y_prob), train_domain, train_domain
            )
            cur_f1 = metrics["f1"]; cur_auc_roc = metrics["roc_auc"]

            logging.info(f"[{train_domain}] | Epoch: {epoch+1}/{num_epochs} | "
                         f"Val Loss: {test_loss:.4f} | F1: {cur_f1:.4f} | AUC-ROC: {cur_auc_roc:.4f}")
            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/val_loss": float(test_loss),
                f"{train_domain}/val_f1": float(cur_f1),
                f"{train_domain}/val_ROC_AUC": float(cur_auc_roc),
            })

            # Early stopping on F1
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
                logging.info(f"New best F1 for {train_domain}: {best_f1:.4f}")
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement. Count: {epochs_no_improve}")
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping for {train_domain} at epoch {epoch+1}")
                break

        # ===== end epochs =====
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
        else:
            logging.info(f"No improvement for {train_domain}. Model not saved.")

        # Evaluate best on current domain
        model.eval()
        if args.architecture == "LSTM_Attention_adapter":
            best_metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
        else:
            best_metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)
        cur_f1 = best_metrics["f1"]
        performance_plasticity[train_domain].append(cur_f1)
        performance_stability[train_domain].append(cur_f1)
        logging.info(f" F1 : Current Domain : {train_domain}: {cur_f1:.4f}")
        logging.info(f"performance_plasticity: {performance_plasticity}")
        logging.info(f"metrics: {best_metrics}")
        
        # Stability on previously seen domains
        seen_domain.add(train_domain)
        logging.info(f"====== Evaluating on all previous domains after training on {train_domain} ======")
        for td in tqdm(seen_domain, desc="Stability test"):
            if td == train_domain:
                continue
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                m = evaluate_model.eval_model(args, model, test_domain_loader, td, device, domain_id=domain_to_id[td])
            else:
                m = evaluate_model.eval_model(args, model, test_domain_loader, td, device, domain_id=None)
            performance_stability[td].append(m["f1"])
            logging.info(f"performance_stability | {td}: {performance_stability[td]}")

        # Move current VAE to prev_vae for next domain
        prev_vae = deepcopy(replay_vae).to(device).eval()
        for p in prev_vae.parameters(): p.requires_grad = False

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
