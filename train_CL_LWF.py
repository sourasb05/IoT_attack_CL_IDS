
import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from copy import deepcopy
import logging
import warnings
warnings.filterwarnings("ignore")
from utils import save_results_as_json, _sync
import evaluate_model
import evaluation as evaluate
import result_utils as result_utils
import wandb
from tqdm import tqdm, trange

# ===================================
# LwF Utilities
# ===================================
def kd_loss(student_logits, teacher_logits, T=2.0):
    """
    Knowledge distillation loss (KL Divergence between soft targets).
    Returns a scalar tensor.
    """
    import torch.nn.functional as F
    # Ensure both are logits
    s_log_prob = F.log_softmax(student_logits / T, dim=1)
    t_prob     = F.softmax(teacher_logits / T, dim=1)
    # KLDiv expects log-prob from student, prob from teacher
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T ** 2)

def make_frozen_teacher(model):
    """Return a frozen deepcopy of the model for distillation."""
    teacher = deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher

# ===================================
# Main Training Function with LwF
# ===================================
def tdim_lwf_random(args, run_wandb, train_domain_loader, test_domain_loader, device,
                    model, exp_no, num_epochs=500, learning_rate=0.01, patience=3,
                    alpha=0.5, T=2.0):
    """
    Continual/domain-incremental training using Learning without Forgetting (LwF).
    - For each domain, create a frozen teacher = copy(model_before_training_domain).
    - Train student on current domain with CE(new labels) + alpha * KD(student, teacher).
    - On the first domain (no prior knowledge), KD term is skipped.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    performance_stability  = {test_domain: [] for test_domain in test_domain_loader.keys()}
    performance_plasticity = {test_domain: [] for test_domain in test_domain_loader.keys()}
    domain_training_cost   = {test_domain: [] for test_domain in test_domain_loader.keys()}

    seen_domain    = set()
    train_order    = list(train_domain_loader.keys())
    domain_to_id   = {name: i for i, name in enumerate(train_domain_loader.keys())}

    # ---- W&B Enrich config for this run -----
    run_wandb.config.update({
        "batch_size": args.batch_size,
        "Loss Function": "CrossEntropyLoss + LwF",
        "optimizer": "AdamW",
        "alpha_lwf": float(alpha),
        "temperature": float(T),
        "weight_decay": 0.0,
        "train_domains": train_order
    })
    run_wandb.watch(model, criterion=criterion, log="all", log_freq=50)

    previous_domain   = None
    best_model_state  = None

    for idx, train_domain in enumerate(tqdm(train_order, desc="Train Domains", total=len(train_order))):
        domain_id = domain_to_id[train_domain]
        domain_epoch = 0
        wandb.define_metric(f"{train_domain}/epoch")
        wandb.define_metric(f"{train_domain}/*", step_metric=f"{train_domain}/epoch")

        # Pre-train eval on the upcoming domain (plasticity snapshot)
        logging.info(f"====== Evaluate current domain {train_domain} on model built in previous domain : {previous_domain} ======")
        if idx != 0:
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            else:
                logging.warning("best_model_state is uninitialized. Skipping model loading.")
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
            else:
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)
            current_f1 = metrics["f1"]
            performance_plasticity[train_domain].append(current_f1)
            run_wandb.log({f"{train_domain}/pretrain_f1": float(current_f1)})
            logging.info(f" F1 : Previous domain : {previous_domain} : Current Domain {train_domain}: {current_f1:.4f}")

        logging.info(f"====== Training on Domain: {train_domain} (LwF) ======")

        # Make frozen teacher BEFORE learning the new domain
        if best_model_state is not None:
            # Teacher is the best model after previous domain
            temp_model = deepcopy(model)
            temp_model.load_state_dict(best_model_state)
            teacher = make_frozen_teacher(temp_model)
        else:
            teacher = None  # No teacher for the first domain

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

                optimizer.zero_grad()
                # Student forward
                if args.architecture == "LSTM_Attention_adapter":
                    student_logits, _ = model(X_batch, domain_id=domain_id)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits, _ = teacher(X_batch, domain_id=domain_id)
                    else:
                        teacher_logits = None
                else:
                    student_logits, _ = model(X_batch)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits, _ = teacher(X_batch)
                    else:
                        teacher_logits = None

                # CE loss on current domain labels
                ce = criterion(student_logits, y_batch.long())

                # KD loss against the frozen teacher, if present
                if teacher_logits is not None:
                    kd = kd_loss(student_logits, teacher_logits, T=T)
                    loss = ce + alpha * kd
                else:
                    loss = ce

                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss.item())

            epoch_loss /= (i + 1) if i >= 0 else 1
            _sync(device)
            epoch_time = time.perf_counter() - epoch_start
            logging.info(f"[{train_domain}] | Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/train_loss": float(epoch_loss),
                f"{train_domain}/epoch_time_s": float(epoch_time),
            })

            # Evaluate on same domain's test set
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_domain_loader[train_domain]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    if args.architecture == "LSTM_Attention_adapter":
                        outputs, _ = model(X_batch, domain_id=domain_id)
                    else:
                        outputs, _ = model(X_batch)
                    loss_val = criterion(outputs, y_batch.long())
                    _, predicted = torch.max(outputs.data, 1)
                    all_y_true.extend(y_batch.cpu().numpy())
                    all_y_pred.extend(predicted.cpu().numpy())
                    all_y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    test_loss += float(loss_val.item())
            test_loss /= max(1, len(test_domain_loader[train_domain]))

            metrics = evaluate.evaluate_metrics(
                np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob), train_domain, train_domain
            )
            current_f1 = float(metrics["f1"])
            current_auc_roc = float(metrics["roc_auc"])

            logging.info(f"[{train_domain}] | Epoch: {epoch+1}/{num_epochs} | Test Loss: {test_loss:.4f} | F1: {current_f1:.4f} | AUC-ROC: {current_auc_roc:.4f}")

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/val_loss": float(test_loss),
                f"{train_domain}/val_f1": float(current_f1),
                f"{train_domain}/val_ROC_AUC": float(current_auc_roc),
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

        # Restore best state and save model checkpoint for current domain
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model_save_path = f"models/exp_no_{exp_no}_{args.architecture}_{args.algorithm}_{args.scenario}/best_model_after_{train_domain}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_state, model_save_path)
            logging.info(f"Best model for {train_domain} saved to {model_save_path}")
            previous_domain = train_domain
        else:
            logging.info(f"No improvement for {train_domain}. Model not saved.")

        # Evaluate on the best model on the currently trained domain
        model.eval()
        if args.architecture == "LSTM_Attention_adapter":
            best_metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
        else:
            best_metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)

        current_f1 = float(best_metrics["f1"])
        performance_plasticity[train_domain].append(current_f1)
        performance_stability[train_domain].append(current_f1)
        logging.info(f" F1 : Current Domain {train_domain}: {current_f1:.4f}")
        logging.info(f"performance_plasticity: {performance_plasticity}")
        logging.info(f"metrics: {best_metrics}")

        # Generalization to all previous domains (stability)
        logging.info(f"====== Evaluating on all previous domains after training on {train_domain} ======")
        seen_domain.add(train_domain)
        for test_domain in tqdm(seen_domain, desc="Stability test"):
            if test_domain == train_domain:
                continue
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                metrics_prev = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=domain_to_id[test_domain])
            else:
                metrics_prev = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=None)
            current_f1_prev = float(metrics_prev["f1"])
            performance_stability[test_domain].append(current_f1_prev)
            logging.info(f"performance_stability | {test_domain}: {performance_stability[test_domain]}")

        print(f"====== Finished Training on Domain: {train_domain} ======")

    # Final Metrics: BWT / FWT
    logging.info(f"====== Final Metrics after training on all domains ======")
    bwt_values, bwt_dict, bwt_values_dict = result_utils.compute_BWT(performance_stability, train_order)
    fwt_values, fwt_dict = result_utils.compute_FWT(performance_plasticity, train_order)
    logging.info(f"\\n BWT: {bwt_values}")
    logging.info(f"\\n BWT of all previous domain corresponding to the training domain: {bwt_values_dict}")
    logging.info(f"\\n BWT per domain: {bwt_dict}")
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
        "train_domain_order": train_order,
        "domain_training_cost": domain_training_cost,
    }
    save_results_as_json(results_to_save, filename=f"{exp_no}_experiment_results_{args.architecture}_{args.algorithm}_{args.scenario}.json")
    logging.info("Final training complete. Results saved.")

    run_wandb.summary["BWT/list"] =  bwt_values
    run_wandb.summary["FWT/list"] =  fwt_values

    tbl1 = wandb.Table(columns=["domain", "FWT"])
    for dom in train_order:
        tbl1.add_data(dom,  fwt_dict.get(dom, None))
    run_wandb.log({"fwt_metrics": tbl1})

    tbl2 = wandb.Table(columns=["domain", "BWT"])
    for dom in train_order:
        tbl2.add_data(dom,  bwt_values_dict.get(dom, None))
    run_wandb.log({"bwt_metrics": tbl2})
