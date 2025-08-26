import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from copy import deepcopy
import logging
from utils import save_results_as_json
import evaluation as evaluate
import sys

def train_domain_incremental_model_gen(scenario, device, train_domain_loader, test_domain_loader, full_domain_loader, 
                                   model, exp_no, num_epochs=20, learning_rate=0.01, patience=3, 
                                   forgetting_threshold=0.01, gpu=0):
    
   
    criterion = nn.CrossEntropyLoss()
    
    performance_history = {test_domain: [] for test_domain in test_domain_loader.keys()}
    performance_matrix = {test_domain: [] for test_domain in test_domain_loader.keys()}
    forget_counter_dictionary = {}

    remaining_train_domains = set(train_domain_loader.keys())
    used_train_domains = set()
    train_domain_order = []

    logging.info(f"Training on {len(remaining_train_domains)} domains.")

    while remaining_train_domains:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if not used_train_domains:
            current_train_domain = remaining_train_domains.pop()
        else:
            logging.info(f"performance_matrix: {performance_matrix}")
            available_domains = {d: performance_matrix[d][-1] if performance_matrix[d] else 0.0
                                 for d in remaining_train_domains}
            if not available_domains:
                break
            
            if scenario == "Generalization_worst":
                current_train_domain = min(available_domains.items(), key=lambda x: x[1])[0]
            elif scenario == "Generalization_best":
                current_train_domain = max(available_domains.items(), key=lambda x: x[1])[0]
            remaining_train_domains.remove(current_train_domain)

        used_train_domains.add(current_train_domain)
        train_domain_order.append(current_train_domain)
        logging.info(f"====== Training on Domain: {current_train_domain} : {len(used_train_domains)}/{len(train_domain_loader)} ======")

        best_f1 = -float("inf")
        epochs_no_improve = 0
        best_model_state = None

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(train_domain_loader[current_train_domain][exp_no]):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.long())
                loss.backward()
                # Log gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                logging.info(f"Gradient norm: {total_norm:.4f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= (i + 1)

            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in test_domain_loader[current_train_domain][0]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    all_y_true.extend(y_batch.cpu().numpy())
                    all_y_pred.extend(predicted.cpu().numpy())
                    all_y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())

            metrics = evaluate.evaluate_metrics(np.array(all_y_true), np.array(all_y_pred), 
                                                np.array(all_y_prob), current_train_domain, current_train_domain)
            current_f1 = metrics["f1"]

            logging.info(f"Early Stopping Check | Domain: {current_train_domain} | Epoch: {epoch+1} | F1: {current_f1:.4f}")

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
                logging.info(f"New best F1 for {current_train_domain}: {best_f1:.4f}")
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement. Count: {epochs_no_improve}")

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered for {current_train_domain} at epoch {epoch+1}")
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model_save_path = f"models/exp_no_{exp_no}/best_model_after_{current_train_domain}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_state, model_save_path)
            logging.info(f"Best model for {current_train_domain} saved to {model_save_path}")
        else:
            logging.info(f"No improvement for {current_train_domain}. Model not saved.")

        overall_metrics = {}
        logging.info(f"--- Overall Evaluations after training on {current_train_domain} ---")
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
                                                np.array(all_y_prob), test_domain, current_train_domain)
            overall_metrics[test_domain] = metrics
            performance_history[test_domain].append(metrics["f1"])
            performance_matrix[test_domain].append(metrics["f1"])

        logging.info(overall_metrics)

        print(f"\n--- Improvement in F1 Score after training on {current_train_domain} ---\n")
        forget_counter = 0
        not_forget_counter = 0
        for test_domain, history in performance_history.items():
            current_F1 = history[-1]
            best_so_far = max(history)
            forgetting = best_so_far - current_F1
            if forgetting > forgetting_threshold:
                forget_counter += 1
                logging.info(f"{test_domain}: Generalization Gap={-forgetting:.4f} --> Significant Drop in F1 Score Detected!")
            else:
                not_forget_counter += 1
                logging.info(f"{test_domain}: Generalization Gap={forgetting:.4f} --> No Significant Drop in F1 Score Detected.")
        forget_ratio = forget_counter / (forget_counter + not_forget_counter) * 100
        logging.info(f"F1 Score dropped in {forget_counter}/{forget_counter + not_forget_counter} domains ({forget_ratio:.2f}%)")
        forget_counter_dictionary[current_train_domain] = forget_ratio

    logging.info(f"====== Final Metrics after training on all domains ======")
    T = len(train_domain_order)
    test_domains = list(performance_matrix.keys())
    perf_array = np.zeros((T, T))
    for i, test_domain in enumerate(test_domains):
        perf_array[:, i] = performance_matrix[test_domain]
    logging.info(f"Performance matrix: \n{perf_array}")

    bwt_values = [perf_array[-1, i] - perf_array[i, i] for i in range(T - 1)]
    BWT = np.mean(bwt_values)

    fwt_values = [perf_array[i - 1, i] for i in range(1, T)]
    FWT = np.mean(fwt_values)

    bwt_dict = {
    train_domain_order[i]: bwt_values[i]
    for i in range(len(bwt_values))
    }
    
    logging.info(f"BWT: {BWT:.4f}")
    logging.info(f"FWT: {FWT:.4f}")
    logging.info(f"BWT per task (domain-wise): {bwt_dict}")


    results_to_save = {
        "performance_history": performance_history,
        "performance_matrix": performance_matrix,
        "forget_counter_dictionary": forget_counter_dictionary,
        "BWT": BWT,
        "FWT": FWT,
        "train_domain_order": train_domain_order,
    }

    save_results_as_json(results_to_save, filename=f"{exp_no}_experiment_results_wcl_gen_last.json")
    logging.info("Final training complete. Results saved.")
