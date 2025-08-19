import os
import torch
import numpy as np
from copy import deepcopy
import logging
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, balanced_accuracy_score, roc_curve, auc)
from utils import save_results_as_json
import evaluation as evaluate

# ----------------------------
# 1. LSTM-VAE for Generative Replay
# ----------------------------
class LSTMVAE(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, latent_dim=32, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.enc_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc_mu    = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar= nn.Linear(hidden_dim, latent_dim)
        self.dec_init = nn.Linear(latent_dim, hidden_dim)
        self.dec_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.out_fc   = nn.Linear(hidden_dim, feature_dim)

    def encode(self, x):
        _, (h, _) = self.enc_lstm(x)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h0 = self.dec_init(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        seq0 = torch.zeros(z.size(0), self.seq_len, self.feature_dim, device=z.device)
        out, _ = self.dec_lstm(seq0, (h0, c0))
        return self.out_fc(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ----------------------------
# 2. VAE Loss & Helpers
# ----------------------------
def vae_loss(recon, x, mu, logvar):
    recon_mse = nn.functional.mse_loss(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_mse + kl


def train_vae_on_dataset(vae, device, dataset, num_epochs=5, lr=1e-3):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss = vae_loss(recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        logging.info(f"VAE Epoch {epoch+1}/{num_epochs} Loss: {avg:.4f}")


def generate_replay_samples(vae, n_samples, device, window_size, num_features):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, vae.fc_mu.out_features, device=device)
        syn_seq = vae.decode(z)
        return syn_seq

# ----------------------------
# 3. Domain-Incremental Training with Generative Replay
# ----------------------------
def train_domain_incremental_model(scenario, device, train_loaders, test_loaders, full_loaders,
                                   model, exp_no, num_epochs=20, learning_rate=0.001,
                                   patience=3, forgetting_threshold=0.01,
                                   num_replay_samples=1000, window_size=10, num_features=20):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    performance_history = {d: [] for d in test_loaders.keys()}
    performance_matrix = {d: [] for d in test_loaders.keys()}
    replay_vae = None

    remaining = set(train_loaders.keys())
    used = []

    while remaining:
        # select next domain
        if not used:
            dom = remaining.pop()
        else:
            scores = {d: performance_matrix[d][-1] if performance_matrix[d] else 0.0 for d in remaining}
            if scenario == "Generalization_worst":
                dom = min(scores, key=scores.get)
            elif scenario == "Generalization_best":
                dom = max(scores, key=scores.get)
            else:
                dom = remaining.pop()
            remaining.remove(dom)
        used.append(dom)
        logging.info(f"Training on domain {dom} ({len(used)}/{len(train_loaders)})")

        # Build combined loader: real + replay
        real_loader = train_loaders[dom][exp_no]
        if replay_vae is not None:
            syn = generate_replay_samples(replay_vae, num_replay_samples, device, window_size, num_features)
            y_syn = torch.zeros(num_replay_samples, dtype=torch.long, device=device)
            dataset_syn = TensorDataset(syn.cpu(), y_syn.cpu())
            combined = ConcatDataset([real_loader.dataset, dataset_syn])
            loader = DataLoader(combined, batch_size=32, shuffle=True)
        else:
            loader = real_loader

        # Train classifier with early stopping
        best_f1 = -np.inf
        epochs_no = 0
        best_state = None
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for ep in range(num_epochs):
            model.train()
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # Eval same-domain
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for xb, yb in test_loaders[dom][exp_no]:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    pred = torch.argmax(logits, 1)
                    ys.extend(yb.cpu().numpy()); ps.extend(pred.cpu().numpy())
            f1 = f1_score(ys, ps, average='binary')
            if f1 > best_f1:
                best_f1 = f1; best_state = deepcopy(model.state_dict()); epochs_no = 0
            else:
                epochs_no += 1
            if epochs_no >= patience:
                break
        # Restore best state
        if best_state is not None:
            model.load_state_dict(best_state)
            path = f"best_model_{dom}.pt"
            torch.save(best_state, path)
            logging.info(f"Saved best model for {dom} to {path}")

        # Update replay VAE with real+past
        real_X = []
        for xb, _ in real_loader:
            real_X.append(xb)
        real_X = torch.cat(real_X, dim=0)
        if replay_vae is None:
            replay_vae = LSTMVAE(num_features, hidden_dim=64, latent_dim=32, seq_len=window_size).to(device)
            combined_flat = real_X
        else:
            syn = generate_replay_samples(replay_vae, num_replay_samples, window_size, num_features)
            combined_flat = torch.cat([real_X, syn], dim=0)
        vae_dataset = TensorDataset(combined_flat.cpu(), torch.zeros(combined_flat.size(0), dtype=torch.long))
        train_vae_on_dataset(replay_vae, device, vae_dataset, num_epochs=5, lr=1e-3)

        # Evaluate on all test domains
        for td in test_loaders.keys():
            ys, ps, probs = [], [], []
            with torch.no_grad():
                for xb, yb in test_loaders[td][exp_no]:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    pred = torch.argmax(logits, 1)
                    prob = torch.softmax(logits, dim=1)[:,1]
                    ys.extend(yb.cpu().numpy()); ps.extend(pred.cpu().numpy()); probs.extend(prob.cpu().numpy())
            metrics = evaluate.evaluate_metrics(np.array(ys), np.array(ps), np.array(probs), td, dom)
            performance_history[td].append(metrics['f1'])
            performance_matrix[td].append(metrics['f1'])

    # Save results
    results = {'performance_history': performance_history, 'performance_matrix': performance_matrix}
    save_results_as_json(results, filename="generative_replay_results.json")
    logging.info("Training complete. Results saved.")
