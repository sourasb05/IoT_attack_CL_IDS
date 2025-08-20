import os
import random
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import logging
import argparse

import sys


def save_results_as_json(results, filename, save_folder="results"):
    os.makedirs(save_folder, exist_ok=True)
    filepath = os.path.join(save_folder, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {filepath}")


# Function to create sliding windows
def create_sliding_windows(X, y, window_size, step_size):
    sequences, labels = [], []
    for i in range(0, len(X) - window_size, step_size):
        sequences.append(X[i:i + window_size])
        labels.append(y[i + window_size - 1])  # Label is the last time step in the window
    return np.array(sequences), np.array(labels)


def load_data(domain_path,domain_key, domain_dataset, window_size=10, step_size=3, batch_size=128, train=True):
    # Load your data (assuming it's already loaded as `data`)
    print(domain_dataset)
    domain_loader_list = []
    
    domain_data_path = domain_path + "/" + domain_key + "/" + domain_dataset
    data = pd.read_csv(domain_data_path)  # Replace with the correct path to your file
    X_all = data.iloc[:, 1:-1].values  # All columns except 'Unnamed: 0' and 'label'
    y_all = data['label'].values  # Labels are in the 'label' column

    # Apply sliding window to the entire dataset
    X_all_windows, y_all_windows = create_sliding_windows(X_all, y_all, window_size, step_size)

    # Convert to PyTorch tensors for training
    X_domain_tensor = torch.tensor(X_all_windows, dtype=torch.float32)
    y_domain_tensor = torch.tensor(y_all_windows, dtype=torch.float32)
    # Create DataLoader for training and testing
    domain_dataset = TensorDataset(X_domain_tensor, y_domain_tensor)
    if train:
        domain_loader = DataLoader(domain_dataset, batch_size=batch_size, shuffle=False)
    else:
        # For testing, we might not want to shuffle
        domain_loader = DataLoader(domain_dataset, batch_size=len(domain_dataset), shuffle=False)
    domain_loader_list.append(domain_loader)
    
    return domain_loader_list

def create_domains(domains_path):

        # Iterate over each item in the source folder.
    domains = {}
    for domain  in os.listdir(domains_path):
        #if domain in [ "blackhole_var5_base", "blackhole_var5_oo", "blackhole_var5_dec",
        #              "blackhole_var10_base", "blackhole_var10_oo", "blackhole_var10_dec",
        #              "blackhole_var15_base", "blackhole_var15_oo", "blackhole_var15_dec",
        #             "blackhole_var20_base", "blackhole_var20_oo", "blackhole_var20_dec",
        #]:
        #    continue
        # print(f"domain : {domain}")
        domain_path = os.path.join(domains_path, domain)
        # print(f"domain_path : {domain_path}")
        # Check if the item is a directory (subfolder).
        if os.path.isdir(domain_path):
            # print(f"Processing subfolder: {domain}")
            # List and sort all files in this subfolder.
            files = sorted(os.listdir(domain_path))
            # Select at most the first 10 files.
            selected_files = files
            
            """file_contents = []
            # Read the contents of each selected file.
            for file_name in selected_files:
                file_path = os.path.join(domain_path, file_name)
                # Ensure that the path refers to a file.
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        file_contents.append(content)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
            """
           # Use the subfolder name as the dictionary key and its list of file contents as the value.
            domains[domain] = selected_files
    logging.info(f"Domains found: {domains.keys()}")
    logging.info(f"Number of domains found: {len(domains)}")

    # for key, value_list in domains.items():
    #     print(f"key : {key} value : {value_list} num_element : {len(value_list)}")
    return domains



def compute_mmd(X1, X2, gamma=None):
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    Kxx = np.exp(-cdist(X1, X1, 'sqeuclidean') * gamma)
    Kyy = np.exp(-cdist(X2, X2, 'sqeuclidean') * gamma)
    Kxy = np.exp(-cdist(X1, X2, 'sqeuclidean') * gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

def cluster_domains(base_path, distance_threshold=2.0):
    # 1) find one CSV per domain‐folder
    domain_paths = {}
    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        csvs = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csvs:
            continue
        domain_paths[folder_name] = os.path.join(folder_path, csvs[0])

    # 2) load & standardize features
    scaler = StandardScaler()
    domain_features = {}
    for domain, path in domain_paths.items():
        df = pd.read_csv(path)
        X = df.drop(columns=['label'], errors='ignore').values
        X_scaled = scaler.fit_transform(X)
        domain_features[domain] = X_scaled

    domain_list = list(domain_features.keys())
    n = len(domain_list)
    mmd_matrix = np.zeros((n, n))
    for i in range(n):
        Xi = domain_features[domain_list[i]]
        for j in range(i, n):
            Xj = domain_features[domain_list[j]]
            m = compute_mmd(Xi, Xj)
            mmd_matrix[i, j] = m
            mmd_matrix[j, i] = m

    # 3) hierarchical clustering on the condensed form of mmd_matrix
    condensed = squareform(mmd_matrix)
    Z = linkage(condensed, method='ward')
    cluster_assignments = fcluster(Z, t=distance_threshold, criterion='distance')

    clusters = defaultdict(list)
    cluster_map = {}
    for idx, cid in enumerate(cluster_assignments):
        dom = domain_list[idx]
        cluster_map[dom] = cid
        clusters[cid].append(dom)

    return dict(clusters), cluster_map

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Training script with W&B logging")

    # W&B related
    parser.add_argument("--project", type=str, default="attack_CL",
                        help="W&B project name")
    parser.add_argument("--entity", type=str, default="sourasb05",
                        help="W&B entity (your username or team name)")
    parser.add_argument("--run_name", type=str, default="experiment-1",
                        help="Name of this run in W&B")
    
    # Model parameters
    parser.add_argument("--learning_rate", type=float, default=0.05,
                        help="Learning rate for the optimizer")
    parser.add_argument("--architecture", type=str, default="LSTM_Attention", 
                        help="Model architecture to use (e.g., LSTM, BiLSTM, LSTM_Attention, BiLSTM_Attention, LSTM_Attention_adapter)")
    
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs to train the model")
    parser.add_argument("--algorithm", type=str, default="WCL",
                        help="Algorithm to use for continual learning (e.g., EWC, EWC_ZS, genreplay, SI, WCL)")
    parser.add_argument("--scenario", type=str, default="random",
                        help="Scenario for training (e.g., random, b2w, w2b, clustered, toggle)")
    parser.add_argument("--exp_no", type=int, default=1,
                        help="Experiment number for logging")
    parser.add_argument("--window_size", type=int, default=10,
                        help="Size of the sliding window for time series data")
    parser.add_argument("--step_size", type=int, default=3,
                        help="Step size for sliding window")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--input_size", type=int, default=13,
                        help="Input size for the model (number of features)")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for the model")
    parser.add_argument("--output_size", type=int, default=2,
                        help="Output size for the model (number of classes)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers in the model")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout rate for the model")
    parser.add_argument("--bidirectional", action='store_true',
                        help="Use bidirectional LSTM if set")
    parser.add_argument("--patience", type=int, default=3,
                        help="Patience for early stopping")
    parser.add_argument("--forgetting_threshold", type=float, default=0.01,
                        help="Threshold for detecting catastrophic forgetting")
    
    return parser.parse_args()