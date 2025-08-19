import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import logging
import warnings
warnings.filterwarnings("ignore")
from utils import save_results_as_json
import evaluation as evaluate
import sys
from termcolor import colored

# ===================================
# Elastic Weight Consolidation Class
# ===================================
class EWC:
    def __init__(self, model, dataloader, device, lambda_=1150, fisher_n_samples=None):
        self.model = model
        self.device = device
        self.lambda_ = lambda_

        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.prev_params = {n: p.clone().detach() for n, p in self.params.items()}
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
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch.long())
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1

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
# Main Training Function with EWC
# ===================================
def train_domain_incremental_model(scenario, device,train_domain_loader, test_domain_loader, full_domain_loader,
                                   model, num_epochs=20, learning_rate=0.01, patience=3,
                                   forgetting_threshold=0.01):
    exp_no = 0
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ewc_list = []  # Track EWC penalties from past tasks

    performance_history = {test_domain: [] for test_domain in test_domain_loader.keys()}
    performance_matrix = {test_domain: [] for test_domain in test_domain_loader.keys()}

    # train_domain_order = list(train_domain_loader.keys())
    """ train_domain_order = [
    'worstparent_var15_dec', 'worstparent_var20_oo', 'worstparent_var20_base',
    'localrepair_var20_base', 'blackhole_var15_oo', 'localrepair_var20_dec',
    'worstparent_var20_dec', 'worstparent_var10_dec', 'localrepair_var20_oo',
    'localrepair_var15_dec', 'blackhole_var20_oo', 'worstparent_var10_base',
    'localrepair_var10_dec', 'worstparent_var10_oo', 'worstparent_var5_oo',
    'blackhole_var10_oo', 'localrepair_var15_oo', 'blackhole_var5_oo',
    'worstparent_var5_dec', 'blackhole_var10_dec', 'localrepair_var10_base',
    'localrepair_var5_oo', 'blackhole_var10_base', 'blackhole_var15_dec',
    'blackhole_var20_dec', 'worstparent_var5_base', 'disflooding_var5_base',
    'localrepair_var5_dec', 'worstparent_var15_base', 'worstparent_var15_oo',
    'disflooding_var10_dec', 'localrepair_var15_base', 'disflooding_var5_dec',
    'localrepair_var10_oo', 'disflooding_var15_base', 'disflooding_var5_oo',
    'blackhole_var5_dec', 'disflooding_var20_oo', 'disflooding_var10_base',
    'blackhole_var15_base', 'disflooding_var20_base', 'disflooding_var15_dec',
    'disflooding_var20_dec', 'disflooding_var10_oo', 'blackhole_var20_base',
    'blackhole_var5_base', 'disflooding_var15_oo', 'localrepair_var5_base'
    ]"""
    forget_counter_dictionary = {}


    remaining_train_domains = set(train_domain_loader.keys())
    used_train_domains = set()
    train_domain_order = []

    logging.info(f"Training on {len(remaining_train_domains)} domains.")

    while remaining_train_domains:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if not used_train_domains:
            train_domain = remaining_train_domains.pop()
        else:
            logging.info(f"performance_matrix: {performance_matrix}")
            available_domains = {d: performance_matrix[d][-1] if performance_matrix[d] else 0.0
                                 for d in remaining_train_domains}
            if not available_domains:
                break
            
            if scenario == "Generalization_worst":
                train_domain = min(available_domains.items(), key=lambda x: x[1])[0]
            elif scenario == "Generalization_best":
                train_domain = max(available_domains.items(), key=lambda x: x[1])[0]
            remaining_train_domains.remove(train_domain)

        used_train_domains.add(train_domain)
        train_domain_order.append(train_domain)
        logging.info(f"====== Training on Domain: {train_domain} : {len(used_train_domains)}/{len(train_domain_loader)} ======")

        best_f1 = -float("inf")
        epochs_no_improve = 0
        best_model_state = None

        model.train()

        # Compute EWC regularization from all past tasks
        def total_ewc_penalty():
            return sum([ewc.penalty() for ewc in ewc_list]) if ewc_list else 0.0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(train_domain_loader[train_domain][exp_no]):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)

                loss = criterion(outputs, y_batch.long()) + total_ewc_penalty()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= (i + 1)

            # Validation on the same domain
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in test_domain_loader[train_domain][0]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    all_y_true.extend(y_batch.cpu().numpy())
                    all_y_pred.extend(predicted.cpu().numpy())
                    all_y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())

            metrics = evaluate.evaluate_metrics(np.array(all_y_true), np.array(all_y_pred),
                                                np.array(all_y_prob), train_domain, train_domain)
            current_f1 = metrics["f1"]
            logging.info(f"Early Stopping | Domain: {train_domain} | Epoch: {epoch+1} | F1: {current_f1:.4f}")

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping for {train_domain} at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model_save_path = f"models/exp_no_{exp_no}/best_model_after_{train_domain}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_state, model_save_path)
            logging.info(f"Best model for {train_domain} saved to {model_save_path}")
        else:
            logging.info(f"No best model found for {train_domain}")

        # Create EWC object for this domain and append to list
        ewc_data_loader = train_domain_loader[train_domain][exp_no]
        ewc_instance = EWC(model, ewc_data_loader, device, lambda_=900, fisher_n_samples=None)
        ewc_list.append(ewc_instance)

        # Evaluation on all test domains
        overall_metrics = {}
        for test_domain in test_domain_loader.keys():
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in test_domain_loader[test_domain][0]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    all_y_true.extend(y_batch.cpu().numpy())
                    all_y_pred.extend(predicted.cpu().numpy())
                    all_y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())

            metrics = evaluate.evaluate_metrics(np.array(all_y_true), np.array(all_y_pred),
                                                np.array(all_y_prob), test_domain, train_domain)
            overall_metrics[test_domain] = metrics
            performance_history[test_domain].append(metrics["f1"])
            performance_matrix[test_domain].append(metrics["f1"])

        # === Generalization Gap after this domain ===
        seen_domains = set(train_domain_order[:train_domain_order.index(train_domain)+1])
        unseen_domains = set(test_domain_loader.keys()) - seen_domains

        print(f"\n--- Generalization Gap Report after training on {train_domain} ---\n")

        seen_accs = [performance_matrix[d][-1] for d in seen_domains if performance_matrix[d]]
        unseen_accs = [performance_matrix[d][-1] for d in unseen_domains if performance_matrix[d]]
        logging.info(f"Seen Domains: {seen_domains}")
        logging.info(f"Unseen Domains: {unseen_domains}")
        logging.info(f"Performance Matrix: {performance_matrix}")
        logging.info(f"Seen Accuracies: {seen_accs}")
        logging.info(f"Unseen Accuracies: {unseen_accs}")
       
        if seen_accs and unseen_accs:
            seen_avg = np.mean(seen_accs)
            unseen_avg = np.mean(unseen_accs)
            gen_gap = seen_avg - unseen_avg
            print(f"[{train_domain}] Seen Avg: {seen_avg:.4f} | Unseen Avg: {unseen_avg:.4f} | Gen Gap: {gen_gap:.4f}")
            logging.info(f"[{train_domain}] Seen Avg: {seen_avg:.4f} | Unseen Avg: {unseen_avg:.4f} | Gen Gap: {gen_gap:.4f}")
        else:
            print(f"[{train_domain}] Insufficient data to compute generalization gap.")


        if len(train_domain_order) > 1:
            t = train_domain_order.index(train_domain)  # current task index (0-based)
            if t > 0:
                bwt_scores = []
                for i in range(t):  # previous tasks 0 to t-1
                    domain_i = train_domain_order[i]
                    final_perf = performance_matrix[domain_i][t]   # after training on task t
                    original_perf = performance_matrix[domain_i][i]  # after training on task i
                    bwt_scores.append(final_perf - original_perf)

                bwt_so_far = np.mean(bwt_scores)
                print(f"BWT after training on {train_domain}: {bwt_so_far:.4f}")
                logging.info(f"BWT after training on {train_domain}: {bwt_so_far:.4f}")
        
        

        # Generalization analysis
        print(f"\n--- Improvement in F1 Score after training on {train_domain} ---\n")
        forget_counter = 0
        not_forget_counter = 0
        for test_domain, history in performance_history.items():
            current_F1 = history[-1]
            best_so_far = max(history)
            forgetting = best_so_far - current_F1
            if forgetting > forgetting_threshold:
                forget_counter += 1
                logging.info(f"{test_domain}: Generalization Gap={forgetting:.4f} --> Significant Drop in F1 Score Detected!")
            else:
                not_forget_counter += 1
                logging.info(f"{test_domain}: Generalization Gap={forgetting:.4f} --> No Significant Drop in F1 Score Detected.")
        forget_ratio = forget_counter / (forget_counter + not_forget_counter) * 100
        logging.info(f"F1 Score dropped in {forget_counter}/{forget_counter + not_forget_counter} domains ({forget_ratio:.2f}%)")
        forget_counter_dictionary[train_domain] = forget_ratio

    # Final Metrics: BWT / FWT
    T = len(train_domain_order)
    test_domains = list(performance_matrix.keys())
    perf_array = np.zeros((T, T))
    for i, test_domain in enumerate(test_domains):
        perf_array[:, i] = performance_matrix[test_domain]
    logging.info(f"Perdoemance matrix: {perf_array}")

    bwt_values = [perf_array[-1, i] - perf_array[i, i] for i in range(T - 1)]
    BWT = np.mean(bwt_values)

    fwt_values = [perf_array[i - 1, i] for i in range(1, T)]
    FWT = np.mean(fwt_values)

    
    
    logging.info(f"BWT: {BWT:.4f}")
    logging.info(f"FWT: {FWT:.4f}")
    logging.info(f"BWT per task: {bwt_values}")

   
    results_to_save = {
        "performance_history": performance_history,
        "performance_matrix": performance_matrix,
        "forget_counter_dictionary": forget_counter_dictionary,
        "BWT": BWT,
        "FWT": FWT,
        "train_domain_order": train_domain_order,
    }

    save_results_as_json(results_to_save, filename=f"{exp_no}_experiment_results_EWC.json")
    logging.info("Final training complete. Results saved.")
