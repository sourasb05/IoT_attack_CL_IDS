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

# ===================================
# Synaptic Intelligence Class
# ===================================
class SynapticIntelligence:
    def __init__(self, model, device, damping=0.1, epsilon=1.4):
        self.model = model
        self.device = device
        self.damping = damping
        self.epsilon = epsilon

        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.prev_params = {n: p.clone().detach() for n, p in self.params.items()}
        self.w = {n: torch.zeros_like(p, device=device) for n, p in self.params.items()}
        self.omega = {n: torch.zeros_like(p, device=device) for n, p in self.params.items()}

    def begin_task(self):
        for n in self.params:
            self.w[n].zero_()

    def update_w(self):
        for n, p in self.params.items():
            if p.grad is not None:
                delta_theta = p.detach() - self.prev_params[n]
                self.w[n] -= p.grad.detach() * delta_theta

    def end_task(self):
        for n, p in self.params.items():
            delta = p.detach() - self.prev_params[n]
            self.omega[n] += self.w[n] / (delta ** 2 + self.epsilon)
            self.prev_params[n] = p.clone().detach()

    def penalty(self):
        reg = 0.0
        for n, p in self.params.items():
            delta = p - self.prev_params[n]
            reg += (self.omega[n] * delta ** 2).sum()
        return self.damping * reg

# ===================================
# Main Training Function with SI
# ===================================
def train_domain_incremental_model(train_domain_loader, test_domain_loader, full_domain_loader,
                                   model, num_epochs=20, learning_rate=0.01, patience=3,
                                   forgetting_threshold=0.01,
                                   damping=7.0, epsilon=1e-4):
    exp_no = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    si = SynapticIntelligence(model, device, damping=damping, epsilon=epsilon)

    performance_history = {test_domain: [] for test_domain in test_domain_loader.keys()}
    performance_matrix = {test_domain: [] for test_domain in test_domain_loader.keys()}

    train_domain_order = list(train_domain_loader.keys())
    forget_counter_dictionary = {}

    for train_domain in train_domain_order:
        logging.info(f"====== Training on Domain: {train_domain} ======")
        best_f1 = -float("inf")
        epochs_no_improve = 0
        best_model_state = None

        model.train()
        si.begin_task()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(train_domain_loader[train_domain][exp_no]):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)

                loss = criterion(outputs, y_batch.long()) + si.penalty()
                loss.backward()
                si.update_w()
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

        si.end_task()

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model_save_path = f"models/exp_no_{exp_no}/best_model_after_{train_domain}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_state, model_save_path)
            logging.info(f"Best model for {train_domain} saved to {model_save_path}")
        else:
            logging.info(f"No best model found for {train_domain}")

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
                logging.info(f"{test_domain}: Forgetting={forgetting:.4f} --> Significant Drop in F1 Score Detected!")
            else:
                not_forget_counter += 1
                logging.info(f"{test_domain}: Forgetting={forgetting:.4f} --> No Significant Drop in F1 Score Detected.")
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

    save_results_as_json(results_to_save, filename=f"{exp_no}_experiment_results_SI.json")
    logging.info("Final training complete. Results saved.")
