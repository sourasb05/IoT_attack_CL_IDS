import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
import logging
import sys
import warnings
warnings.filterwarnings("ignore")
from utils import save_results_as_json
import evaluate_model
# import result_utils
import evaluation as evaluate
import result_utils as result_utils


# ===========================
# Step 4: Train with Early Stopping, Logging, Model Saving,
#         Catastrophic Forgetting Detection, and Graph Plotting
# ===========================
def train_domain_incremental_model(args, train_domain_loader, test_domain_loader, device,
                                   model, exp_no, num_epochs=500, learning_rate=0.01, patience=3, 
                                   forgetting_threshold=0.01):
    """
    For each training domain, trains the model on that domain (with early stopping using F1).
    After training on a domain, it evaluates on all test domains, logs the metrics, updates
    performance history, computes catastrophic forgetting, and plots graphs for precision,
    recall, and confusion matrices.
    """
    exp_no=0

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Maintain performance history per test domain (accuracy over training rounds)
    performance_stability = {test_domain: [] for test_domain in test_domain_loader.keys()}
    
    performance_plasticity = {test_domain: [] for test_domain in test_domain_loader.keys()}

    seen_domain = set()
    train_domain_order = list(train_domain_loader.keys())
    # Iterate over each training domain
    forget_counter_dictionary = {}

    
    previous_domain = None
    for idx, train_domain in enumerate(train_domain_loader.keys()):
        logging.info(f"====== Evaluate Current domain {train_domain} on model built in previous domain : {previous_domain} ======")
        # Evaluate on same domain's test set
        if idx != 0:
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.load_state_dict(best_model_state)
            model.eval()
            current_f1 = evaluate_model.eval_model(model, test_domain_loader, train_domain, device)
            performance_plasticity[train_domain].append(current_f1)
            logging.info(f" F1 : Previous domain : {previous_domain} : Current Domain {train_domain}: {current_f1:.4f}")

        logging.info(f"====== Training on Domain: {train_domain} ======")

        best_f1 = -float("inf")
        epochs_no_improve = 0
        best_model_state = None

        # Train for the current domain with early stopping (evaluation on same domain's test set)
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(train_domain_loader[train_domain][0]):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
                optimizer.zero_grad()
                outputs, _ = model(X_batch)
                loss = criterion(outputs, y_batch.long())
                # Clip gradients here
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= (i+1)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}")    
        
            
            # Evaluate on same domain's test set

            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            with torch.no_grad():
                # for id_test_domain in range(len(test_domain_loader[train_domain])):
                for X_batch, y_batch in test_domain_loader[train_domain][0]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs, _ = model(X_batch)
                    loss = criterion(outputs, y_batch.long())
                    _, predicted = torch.max(outputs.data, 1)
                    all_y_true.extend(y_batch.cpu().numpy())
                    all_y_pred.extend(predicted.cpu().numpy())
                    all_y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())

              
                metrics = evaluate.evaluate_metrics(np.array(all_y_true), np.array(all_y_pred), 
                                np.array(all_y_prob), train_domain, train_domain)
                current_f1 = metrics["f1"]
            
            print(f" Epoch: {epoch+1} | Test Loss: {loss.item()} F1: {current_f1:.4f}")
        
            logging.info(f"Early Stopping Check | Domain: {train_domain} | Epoch: {epoch+1} | F1: {current_f1:.4f}")
            
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
                # print(f"Early stopping triggered for {train_domain} at epoch {epoch+1}")
                break

        # Restore best state and save model checkpoint for current domain
        if best_model_state is not None:

            model.load_state_dict(best_model_state)
            model_save_path = f"models/exp_no_{exp_no}_{args.architecture}_{args.algorithm}_{args.scenario}/best_model_after_{train_domain}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_state, model_save_path)
            logging.info(f"Best model for {train_domain} saved to {model_save_path}")
            previous_domain = train_domain
        # print(f"Best model for {train_domain} saved to {model_save_path}")
        else:
            logging.info(f"No improvement for {train_domain}. Model not saved.")
        # print(f"No improvement for {train_domain}. Model not saved.")

        # Evaluate on the best model on the currently trained domain 
        model.load_state_dict(best_model_state)
        model.eval()
        current_f1 = evaluate_model.eval_model(model, test_domain_loader, train_domain, device)
        performance_plasticity[train_domain].append(current_f1)
        performance_stability[train_domain].append(current_f1)
        logging.info(f" F1 : Current Domain {train_domain}: {current_f1:.4f}")
        logging.info(f"performance_plasticity: {performance_plasticity}")

        # Generalization to all previous domains:
        logging.info(f"====== Evaluating on all previous domains after training on {train_domain} ======")
        seen_domain.add(train_domain)
        for test_domain in seen_domain:
            print(test_domain)
            if test_domain == train_domain:
                continue
            model.eval()
            current_f1 = evaluate_model.eval_model(model, test_domain_loader, test_domain, device)
            performance_stability[test_domain].append(current_f1)
            
            logging.info(f"performance_stability{test_domain}: {performance_stability[test_domain]}")

        print(f"====== Finished Training on Domain: {train_domain} ======")

    # Final Metrics: BWT / FWT
    logging.info(f"====== Final Metrics after training on all domains ======")
    bwt_values, bwt_dict = result_utils.compute_bwt(performance_stability)
    fwt_values, fwt_dict = result_utils.compute_fwt(performance_plasticity)
    logging.info(f"BWT: {bwt_values:.4f}")
    logging.info(f"BWT per domain: {bwt_dict}")

    logging.info(f"FWT: {fwt_values:.4f}")
    logging.info(f"FWT per domain: {fwt_dict}")



    results_to_save = {
        "performance_stability": performance_stability,
        "performance_m": performance_plasticity,
        "BWT_values": bwt_values,
        "BWT_dict": bwt_dict,
        "FWT_values": fwt_values,
        "FWT_dict": fwt_dict,
        "train_domain_order": train_domain_order,
    }

    save_results_as_json(results_to_save, filename=f"{exp_no}_experiment_results_{args.architecture}_{args.algorithm}_{args.scenario}.json")
    logging.info("Final training complete. Results saved.")


    
    ####
        
        # Evaluate on every test domain after training on current domain and collect metrics
    
    """overall_metrics = {}
        logging.info(f"--- Overall Evaluations after training on {train_domain} ---")
        # print(f"\n--- Overall Evaluations after training on {train_domain} ---\n")
        for test_domain in test_domain_loader.keys():
            #print(test_domain)
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in test_domain_loader[test_domain][0]:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs, _ = model(X_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    all_y_true.extend(y_batch.cpu().numpy())
                    all_y_pred.extend(predicted.cpu().numpy())
                    all_y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())

              
            metrics = evaluate.evaluate_metrics(np.array(all_y_true), np.array(all_y_pred),
                                    np.array(all_y_prob), test_domain, train_domain)
            overall_metrics[test_domain] = metrics
            
            # Update performance history (accuracy for forgetting detection)
            performance_history[test_domain].append(metrics["f1"])
            performance_matrix[test_domain].append(metrics["f1"])
        logging.info(overall_metrics)"""
            
        
        
        # After evaluations, create bar charts comparing precision and recall for all test domains.
        # test_domains = list(overall_metrics.keys())
        # precisions = [overall_metrics[d]["precision"] for d in test_domains]
        # recalls = [overall_metrics[d]["recall"] for d in test_domains]
        
        
        # Generalization analysis
        
    """print(f"\n--- Improvement in F1 Score after training on {train_domain} ---\n")
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
        forget_counter_dictionary[train_domain] = forget_ratio"""
    

    """"# Final Metrics: BWT / FWT
    logging.info(f"====== Final Metrics after training on all domains ======")
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

   
    

"""