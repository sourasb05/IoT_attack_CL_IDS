import torch
import numpy as np
import logging
import evaluation as evaluate

def eval_model(model,test_domain_loader, train_domain, device):
    all_y_true, all_y_pred, all_y_prob = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_domain_loader[train_domain][0]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs, _ = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())
            all_y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        
    metrics = evaluate.evaluate_metrics(np.array(all_y_true), np.array(all_y_pred),
                        np.array(all_y_prob), train_domain, train_domain)
    
    return metrics["f1"]