import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import logging
import warnings
warnings.filterwarnings("ignore")
from utils import save_results_as_json, _sync
import evaluation as evaluate
import evaluate_model
import result_utils
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange
import time
import wandb



# ===================================
# Synaptic Intelligence Class
# ===================================

class SynapticIntelligence:
    """
    Implements SI (Zenke, Poole, Ganguli, ICML'17).
    - Online credit ω_k: accumulate sum_t g_k(t) * Δθ_k(t) during the task
    - Consolidation at task end: Ω_k += (-ω_k) / ((Δ_k_task)^2 + ξ)
    - Penalty for next tasks:  c * Σ_k Ω_k (θ_k - θ*_k)^2
    """
    def __init__(self, model, device, c=0.1, xi=1e-3):
        self.model = model
        self.device = device
        self.c = c      # trade-off old vs new (paper sets ~0.1 on permuted MNIST)
        self.xi = xi    # damping to avoid div-by-zero (paper uses 1e-3; 0.1 on permuted MNIST)

        # Track only trainable params
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}

        # Reference weights (θ* at last task end), and accumulators
        self.theta_star = {n: p.detach().clone() for n, p in self.params.items()}
        self.Omega     = {n: torch.zeros_like(p, device=device) for n, p in self.params.items()}  # cumulative importance
        self.omega     = {n: torch.zeros_like(p, device=device) for n, p in self.params.items()}  # online credit within the task

        # Per-step snapshot for Δθ
        self.p_old     = {n: torch.zeros_like(p, device=device) for n, p in self.params.items()}

    @torch.no_grad()
    def snapshot_before_step(self):
        "Call right BEFORE optimizer.step(): stores θ_old for Δθ."
        for n, p in self.params.items():
            self.p_old[n].copy_(p.detach())

    @torch.no_grad()
    def accumulate_from(self, grads_by_name):
        """
        Call right AFTER optimizer.step(): uses captured grads of task loss (not including SI penalty)
        and actual Δθ to accumulate ω_k += g_k * Δθ_k
        """
        for n, p in self.params.items():
            g = grads_by_name.get(n, None)
            if g is None:
                continue
            delta = (p.detach() - self.p_old[n])
            self.omega[n] += g.detach() * delta

    def penalty(self):
        "c * Σ Ω_k (θ - θ*)^2"
        reg = 0.0
        for n, p in self.params.items():
            reg = reg + (self.Omega[n] * (p - self.theta_star[n]).pow(2)).sum()
        return self.c * reg

    @torch.no_grad()
    def consolidate_task_end(self):
        """
        At the end of the current task:
        Ω_k += (-ω_k) / ((Δ_k_task)^2 + ξ),  θ*_k ← θ_k,  ω_k ← 0
        where Δ_k_task = θ_k(end) - θ*_k(previous)
        """
        for n, p in self.params.items():
            delta_task = (p.detach() - self.theta_star[n])
            self.Omega[n] += (-self.omega[n]) / (delta_task.pow(2) + self.xi)
            self.theta_star[n].copy_(p.detach())
            self.omega[n].zero_()


# ===================================
# Main Training Function with SI
# ===================================
# ===================================
# Main Training Function with SI
# ===================================
def tdim_si_random(args, run_wandb, train_domain_loader, test_domain_loader, train_domain_order, device,
                   model, exp_no, num_epochs=500, learning_rate=0.01, patience=3,
                   si_c=0.1, si_xi=1e-3):

    exp_no = exp_no
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- SI object (one for the whole run; consolidates across tasks) ---
    si = SynapticIntelligence(model, device, c=si_c, xi=si_xi)

    performance_stability = {test_domain: [] for test_domain in test_domain_loader.keys()}
    performance_plasticity = {test_domain: [] for test_domain in test_domain_loader.keys()}
    domain_training_cost = {test_domain: [] for test_domain in test_domain_loader.keys()}

    seen_domain = set()
    print(f"Training on {len(train_domain_order)} domains: {train_domain_order}")
    domain_to_id = {name: i for i, name in enumerate(train_domain_order)}

    # ---- W&B Enrich config for this run -----
    run_wandb.config.update({
        "batch_size": args.batch_size,
        "Loss Function": "CrossEntropyLoss + SI",
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "train_domains": train_domain_order,
        "si_c": si_c,
        "si_xi": si_xi
    })
    run_wandb.watch(model, criterion=criterion, log="all", log_freq=50)

    previous_domain = None
    best_model_state = None

    for idx, train_domain in enumerate(tqdm(list(train_domain_order),
                                            desc="Train Domains", total=len(train_domain_order))):

        domain_id = domain_to_id[train_domain]
        domain_epoch = 0
        if args.use_wandb:
            wandb.define_metric(f"{train_domain}/epoch")
            wandb.define_metric(f"{train_domain}/*", step_metric=f"{train_domain}/epoch")

        # ===== Pre-train evaluation on current domain (for FWT/plasticity) =====
        if idx != 0:
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                m = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
            else:
                m = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)
            current_f1 = m["f1"]
            performance_plasticity[train_domain].append(current_f1)
            run_wandb.log({f"{train_domain}/pretrain_f1": float(current_f1)})

        # ===== Training on this domain (with SI) =====
        best_f1 = -float("inf")
        epochs_no_improve = 0

        _sync(device)
        t0 = time.perf_counter()

        for epoch in trange(num_epochs, desc="training Epochs"):
            model.train()
            domain_epoch += 1
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            i = -1

            for i, (X_batch, y_batch) in enumerate(train_domain_loader[train_domain]):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # ---- Forward ----
                if args.architecture == "LSTM_Attention_adapter":
                    outputs, _ = model(X_batch, domain_id=domain_id)
                else:
                    outputs, _ = model(X_batch)

                # Base task loss (no SI here)
                task_loss = criterion(outputs, y_batch.long())

                # ---- Get per-parameter grads of *task loss* (not including SI penalty) ----
                # We’ll map them by name so SI can use them after the actual optimizer.step().
                grads_list = torch.autograd.grad(
                    task_loss, [p for _, p in si.params.items()],
                    retain_graph=True, allow_unused=True
                )
                grads_by_name = {n: g for (n, _), g in zip(si.params.items(), grads_list)}

                # ---- Compose total loss (task + SI penalty from *previous* tasks) ----
                total_loss = task_loss + si.penalty()

                # ---- Backprop & update ----
                optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)

                # SI needs Δθ = θ_new - θ_old → snapshot BEFORE optimizer.step()
                si.snapshot_before_step()
                optimizer.step()

                # SI online credit: ω_k += g_k_task * Δθ_k (online accumulation)
                si.accumulate_from(grads_by_name)

                epoch_loss += float(total_loss.item())

            epoch_loss /= (i + 1) if i >= 0 else 1
            _sync(device)
            epoch_time = time.perf_counter() - epoch_start
            logging.info(f"[{train_domain}] | Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/train_loss": float(epoch_loss),
                f"{train_domain}/epoch_time_s": float(epoch_time),
            })

            # ---- Eval on this domain ----
            model.eval()
            test_loss = 0.0
            all_y_true, all_y_pred, all_y_prob = [], [], []
            with torch.no_grad():
                for Xb, yb in test_domain_loader[train_domain]:
                    Xb, yb = Xb.to(device), yb.to(device)
                    if args.architecture == "LSTM_Attention_adapter":
                        out, _ = model(Xb, domain_id=domain_id)
                    else:
                        out, _ = model(Xb)
                    loss = criterion(out, yb.long())
                    _, pred = torch.max(out.data, 1)
                    all_y_true.extend(yb.cpu().numpy())
                    all_y_pred.extend(pred.cpu().numpy())
                    all_y_prob.extend(torch.nn.functional.softmax(out, dim=1)[:, 1].cpu().numpy())
                    test_loss += loss.item()

            test_loss /= max(1, len(test_domain_loader[train_domain]))
            metrics = evaluate.evaluate_metrics(np.array(all_y_true), np.array(all_y_pred),
                                                np.array(all_y_prob), train_domain, train_domain)
            current_f1 = metrics["f1"]
            current_auc_roc = metrics["roc_auc"]

            logging.info(f"[{train_domain}] | Epoch: {epoch+1}/{num_epochs} |  Test Loss: {test_loss:.4f} | F1: {current_f1:.4f} | AUC-ROC: {current_auc_roc:.4f}")
            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/val_loss": float(test_loss),
                f"{train_domain}/val_f1": float(current_f1),
                f"{train_domain}/val_ROC_AUC": float(current_auc_roc)
            })

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
                logging.info(f"New best F1 for {train_domain}: {best_f1:.4f}")
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement. Count: {epochs_no_improve}")

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered for {train_domain} at epoch {epoch+1}")
                break

        _sync(device)
        domain_training_time = time.perf_counter() - t0
        logging.info(f"Training time for {train_domain}: {domain_training_time:.2f} seconds")
        domain_training_cost[train_domain].append(domain_training_time)

        # ---- Restore best and save ----
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model_save_path = f"models/exp_no_{exp_no}_{args.architecture}_{args.algorithm}_{args.scenario}/best_model_after_{train_domain}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_state, model_save_path)
            logging.info(f"Best model for {train_domain} saved to {model_save_path}")
            previous_domain = train_domain
        else:
            logging.info(f"No improvement for {train_domain}. Model not saved.")

        # ====== SI CONSOLIDATION at TASK END ======
        # This turns the online ω into cumulative Ω (and updates θ*)
        si.consolidate_task_end()

        # Evaluate best model on current domain (post-training)
        model.eval()
        if args.architecture == "LSTM_Attention_adapter":
            best_metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
        else:
            best_metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)
        current_f1 = best_metrics["f1"]
        performance_plasticity[train_domain].append(current_f1)
        performance_stability[train_domain].append(current_f1)
        logging.info(f" F1 : Current Domain {train_domain}: {current_f1:.4f}")

        # Generalization to all *previous* domains
        seen_domain.add(train_domain)
        for test_domain in tqdm(seen_domain, desc="Stability test"):
            if test_domain == train_domain:
                continue
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                m = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=domain_to_id[test_domain])
            else:
                m = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=None)
            performance_stability[test_domain].append(m["f1"])

        print(f"====== Finished Training on Domain: {train_domain} ======")

    # ===== Final Metrics: BWT / FWT =====
    logging.info(f"====== Final Metrics after training on all domains ======")
    bwt_values, bwt_dict, bwt_values_dict = result_utils.compute_BWT(performance_stability, train_domain_order)
    fwt_values, fwt_dict = result_utils.compute_FWT(performance_plasticity, train_domain_order)
    logging.info(f"\n BWT: {bwt_values}")
    logging.info(f"\n BWT of all previous domain corresponding to the training domain: {bwt_values_dict}")
    logging.info(f"\n BWT per domain: {bwt_dict}")
    logging.info(f"FWT: {fwt_values}")
    logging.info(f"FWT per domain: {fwt_dict}")

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
    save_results_as_json(results_to_save, filename=f"{exp_no}_experiment_results_{args.architecture}_SI_{args.scenario}.json")
    logging.info("Final training complete. Results saved.")

    
