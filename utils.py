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
import re


def safe_minmax_normalize(df, global_min, global_max, label_col="label"):
    feat_cols = [c for c in df.columns if c != label_col]
    denom = (global_max - global_min).replace(0, 1)  # avoid div/0
    out = df.copy()
    out[feat_cols] = (out[feat_cols] - global_min) / denom
    out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

def seq_maker(df, sequence_length=10, label_col="label"):
    df_feat = df.drop(columns=[label_col])
    labels = df[label_col].astype(int).values

    attack_idxs = np.where(labels == 1)[0]
    if len(attack_idxs) == 0:
        start_attack = len(labels) + sequence_length   # all zeros
    else:
        start_attack = max(0, attack_idxs[0] - sequence_length)

    sequences = []
    for i in range(len(df_feat) - sequence_length):
        sequences.append(df_feat.iloc[i:i+sequence_length].values.flatten())

    if not sequences:
        return pd.DataFrame(columns=[*range(df_feat.shape[1]*sequence_length), "label"])

    seq_df = pd.DataFrame(sequences)
    zeros = [0] * min(start_attack, len(seq_df))
    ones  = [1] * (len(seq_df) - len(zeros))
    seq_df["label"] = zeros + ones
    return seq_df

def extract_index(path):
    # pull the number between ..._ and _60_sec.csv
    m = re.search(r"_(\d+)_60_sec\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9  # push unknowns to end
DROP_COLS = ["Unnamed: 0"]
def load_csv(path):
    
    df = pd.read_csv(path)
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    assert "label" in df.columns, f"'label' column missing in {os.path.basename(path)}"
    return df


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


def load_data(domain_path, key, domain_dataset, window_size=10, step_size=3, batch_size=128):

    files = sorted(domain_dataset, key=extract_index)[:20]  # ensure exactly 20, ordered

    DROP_COLS = ["Unnamed: 0"]
    
    random.seed(42)
    random.shuffle(files)
    train_files_wo_path = files[:16]
    test_files_wo_path  = files[16:20]


    train_files = [domain_path + "/" + key  + "/" + f for f in train_files_wo_path]
    test_files = [domain_path + "/" + key + "/" + f for f in test_files_wo_path]


    

    # print("Train files:", train_files) 
    # print("Test  files:", test_files)

    # Load your data (assuming it's already loaded as `data`)
    # -----------------------
    # Load all, compute train-only global min/max (excluding 'label')
    # -----------------------
    train_dfs = [load_csv(p) for p in train_files]
    test_dfs  = [load_csv(p) for p in test_files]

    feat_cols = [c for c in train_dfs[0].columns if c != "label"]
    train_feat_mins = [df[feat_cols].min(axis=0) for df in train_dfs]
    train_feat_maxs = [df[feat_cols].max(axis=0) for df in train_dfs]
    global_min = pd.concat(train_feat_mins, axis=1).min(axis=1)
    global_max = pd.concat(train_feat_maxs, axis=1).max(axis=1)

    # -----------------------
    # Normalize using train stats
    # -----------------------
    norm_train = [safe_minmax_normalize(df, global_min, global_max, "label") for df in train_dfs]
    norm_test  = [safe_minmax_normalize(df, global_min, global_max, "label") for df in test_dfs]

    # -----------------------
    # Sequence-ify and concat
    # -----------------------
    seq_train_parts = [seq_maker(df, window_size, "label") for df in norm_train]
    seq_test_parts  = [seq_maker(df, window_size, "label") for df in norm_test]

    seq_train_parts = [df for df in seq_train_parts if not df.empty]
    seq_test_parts  = [df for df in seq_test_parts if not df.empty]

    seq_train = pd.concat(seq_train_parts, ignore_index=True)
    seq_test  = pd.concat(seq_test_parts,  ignore_index=True)

    """print("After seq_maker, train size:", len(seq_train), "test size:", len(seq_test))
    print("Class distribution in train:", seq_train['label'].value_counts().to_dict())
    print("Class distribution in test :", seq_test['label'].value_counts().to_dict())
    print("Feature columns:", seq_train.columns[:-1].tolist())
    print("Label column  :", seq_train.columns[-1])
    print("Any NaNs in train features?", seq_train.iloc[:, :-1].isna().any().any())
    print("Any NaNs in test features ?", seq_test.iloc[:,  :-1].isna().any().any())
    print("Any NaNs in train labels  ?", seq_train['label'].isna().any())
    print("Any NaNs in test labels   ?", seq_test['label'].isna().any())
    print("First few rows of train data:\n", seq_train.head())
    print("First few rows of test data:\n",  seq_test.head())
    print("Train feature stats:\n", seq_train.iloc[:, :-1].describe())
    """
    # -----------------------
    # Tensors & Dataloaders
    # -----------------------
    X_train = torch.tensor(seq_train.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(seq_train.iloc[:,  -1].values.astype(int), dtype=torch.long)
    X_test  = torch.tensor(seq_test.iloc[:,  :-1].values, dtype=torch.float32)
    y_test  = torch.tensor(seq_test.iloc[:,   -1].values.astype(int), dtype=torch.long)

    X_train = torch.nan_to_num(X_train, nan=0.0)
    X_test  = torch.nan_to_num(X_test,  nan=0.0)

    feature_dim = X_train.shape[1]  # (#features * SEQUENCE_LENGTH)
    X_train = X_train.view(-1, 1, feature_dim)
    X_test  = X_test.view(-1, 1, feature_dim)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size, shuffle=False, num_workers=4)


    # print("X_train:", X_train.shape, "y_train:", y_train.shape)
    # print("X_test :", X_test.shape,  "y_test :", y_test.shape)

    return train_loader, test_loader

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
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer")
    parser.add_argument("--architecture", type=str, default="LSTM_Attention_adapter", 
                        help="Model architecture to use (e.g., LSTM, BiLSTM, LSTM_Attention, BiLSTM_Attention, LSTM_Attention_adapter)")
    
    parser.add_argument("--epochs", type=int, default=3, 
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
    parser.add_argument("--input_size", type=int, default=140,
                        help="Input size for the model (number of features)")
    parser.add_argument("--hidden_size", type=int, default=10,
                        help="Hidden size for the model")
    parser.add_argument("--output_size", type=int, default=2,
                        help="Output size for the model (number of classes)")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of layers in the model")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout rate for the model")
    parser.add_argument("--bidirectional", action='store_true',
                        help="Use bidirectional LSTM if set")
    parser.add_argument("--patience", type=int, default=2,
                        help="Patience for early stopping")
    parser.add_argument("--forgetting_threshold", type=float, default=0.01,
                        help="Threshold for detecting catastrophic forgetting")
    
    return parser.parse_args()