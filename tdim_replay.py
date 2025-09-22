import torch
import os
import time
import random
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from copy import deepcopy
import logging
from tqdm import tqdm, trange
from collections import defaultdict, deque
from utils import save_results_as_json
    
# ===========================
# Simple Exemplar Replay Buffer
# ===========================
class ReplayBuffer:
    """
    Exemplar replay buffer with per-domain quotas + reservoir sampling.
    Stores raw tensors on CPU to save GPU memory.
    """
    def __init__(self, total_capacity=5000, per_domain_cap=300, seed=42):
        self.total_capacity = total_capacity
        self.per_domain_cap = per_domain_cap
        self.buffers = defaultdict(lambda: {"x": deque(), "y": deque(), "n_seen": 0})
        self.domains = set()
        random.seed(seed)

    def __len__(self):
        return sum(len(self.buffers[d]["x"]) for d in self.buffers)

    def _reservoir_add(self, dq_x, dq_y, n_seen, x_item, y_item, cap):
        """
        Reservoir sampling: keep at most 'cap' items.
        n_seen counts total seen so far for that domain.
        """
        if len(dq_x) < cap:
            dq_x.append(x_item)
            dq_y.append(y_item)
        else:
            j = random.randint(0, n_seen - 1)
            if j < cap:
                dq_x[j] = x_item
                dq_y[j] = y_item

    def add_batch(self, domain_name, X_batch, y_batch):
        """
        Add a batch (on CPU) to domain buffer using reservoir sampling.
        X_batch: [B, T, F] or [B, ...]
        y_batch: [B]
        """
        self.domains.add(domain_name)
        buf = self.buffers[domain_name]
        for xb, yb in zip(X_batch, y_batch):
            buf["n_seen"] += 1
            self._reservoir_add(
                buf["x"], buf["y"], buf["n_seen"],
                xb.detach().cpu(), yb.detach().cpu(),
                self.per_domain_cap
            )

        # enforce global capacity (optional: naive trim oldest domain first)
        while len(self) > self.total_capacity:
            # drop one item from the largest domain buffer
            biggest = max(self.domains, key=lambda d: len(self.buffers[d]["x"]))
            if self.buffers[biggest]["x"]:
                self.buffers[biggest]["x"].popleft()
                self.buffers[biggest]["y"].popleft()
            else:
                break

    def sample(self, batch_size, device, domains_subset=None):
        """
        Uniform across stored items. If domains_subset provided, sample only from them.
        Returns tensors on 'device'. If buffer is empty, returns None.
        """
        candidates = []
        use_domains = domains_subset if domains_subset else self.domains
        for d in use_domains:
            bx, by = self.buffers[d]["x"], self.buffers[d]["y"]
            for i in range(len(bx)):
                candidates.append((d, i))
        if not candidates:
            return None

        idxs = random.sample(candidates, k=min(batch_size, len(candidates)))

        xs, ys, dnames = [], [], []
        for d, i in idxs:
            xs.append(self.buffers[d]["x"][i])
            ys.append(self.buffers[d]["y"][i])
            dnames.append(d)

        X = torch.stack(xs, dim=0).to(device)
        y = torch.stack(ys, dim=0).to(device)
        return {"X": X, "y": y, "domains": dnames}

# ===========================
# Replay-based training loop (W2B policy preserved)
# ===========================
def tdim_replay(
    args, run_wandb, train_domain_loader, test_domain_loader, device,
    model, exp_no, num_epochs=500, learning_rate=0.01, patience=3,
    # Replay knobs:
    replay_total_capacity=4000,   # total memory across all domains
    replay_per_domain_cap=250,    # per-domain cap
    replay_batch_size=128,        # size of replay mini-batch per iteration
    replay_ratio=0.5,             # 0.0..1.0, fraction of each step drawn from replay (rest from current domain)
    replay_seen_only=True         # if True, only sample from already seen domains
):
    """
    Exemplar replay variant of your W2B loop.
    - After each batch on the current domain, we optionally mix in a replay batch.
    - Stores examples from each domain via reservoir sampling (bounded memory).
    """
    import evaluate_model, evaluation as evaluate, result_utils as result_utils
    from utils import save_results_as_json, _sync

    exp_no = exp_no
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    performance_stability = {test_domain: [] for test_domain in test_domain_loader.keys()}
    performance_plasticity = {test_domain: [] for test_domain in test_domain_loader.keys()}
    domain_training_cost = {test_domain: [] for test_domain in test_domain_loader.keys()}

    train_domain_order = list(train_domain_loader.keys())
    seen_domain = set()
    unseen_domain = set(train_domain_order)
    domain_to_id = {name: i for i, name in enumerate(train_domain_loader.keys())}

    # W&B metadata
    run_wandb.config.update({
        "batch_size": args.batch_size,
        "Loss Function": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "train_domains": train_domain_order,
        "replay_total_capacity": replay_total_capacity,
        "replay_per_domain_cap": replay_per_domain_cap,
        "replay_batch_size": replay_batch_size,
        "replay_ratio": replay_ratio,
        "replay_seen_only": replay_seen_only,
        "method": "Exemplar Replay"
    })
    run_wandb.watch(model, criterion=criterion, log="all", log_freq=50)

    # --- Initialize replay memory ---
    memory = ReplayBuffer(total_capacity=replay_total_capacity,
                          per_domain_cap=replay_per_domain_cap,
                          seed=getattr(args, "seed", 42))

    previous_domain = None
    best_model_state = None
    idx = 0
    train_domain = None

    for idx in tqdm(range(len(train_domain_loader.keys())), desc="Iterating domains"):

        if train_domain is None:
            train_domain = random.choice(list(train_domain_loader.keys()))
        seen_domain.add(train_domain)
        unseen_domain.remove(train_domain)
        domain_id = domain_to_id[train_domain]
        domain_epoch = 0

        logging.info(f"====== Evaluate current domain {train_domain} on model from previous domain {previous_domain} ======")

        # Pre-train evaluation on the *incoming* domain (plasticity)
        if idx != 0:
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            else:
                logging.warning("best_model_state uninitialized. Skipping model loading.")
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
            else:
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)
            current_f1 = metrics["f1"]
            performance_plasticity[train_domain].append(current_f1)
            run_wandb.log({f"{train_domain}/pretrain_f1": float(current_f1)})

        logging.info(f"====== Training on Domain: {train_domain} (with replay) ======")

        best_f1 = -float("inf")
        epochs_no_improve = 0

        _sync(device)
        t0 = time.perf_counter()

        # --------- Training loop with replay ----------
        for epoch in trange(num_epochs):
            model.train()
            domain_epoch += 1
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            i = -1

            # Iterate over current-domain minibatches
            for i, (X_cur, y_cur) in enumerate(train_domain_loader[train_domain]):
                X_cur, y_cur = X_cur.to(device), y_cur.to(device)

                # Build mixed batch (current + replay)
                cur_bs = X_cur.size(0)
                n_replay = int(replay_ratio * cur_bs)

                # Sample replay (only from seen domains unless toggled)
                replay_domains = list(seen_domain) if replay_seen_only else None
                replay_pack = memory.sample(n_replay, device, domains_subset=replay_domains)

                if replay_pack is not None and n_replay > 0:
                    X_rep, y_rep = replay_pack["X"], replay_pack["y"]
                    # if shapes mismatch due to seq len, you might need to pad/trim; assume consistent here
                    X_mix = torch.cat([X_cur, X_rep], dim=0)
                    y_mix = torch.cat([y_cur, y_rep], dim=0)
                else:
                    X_mix, y_mix = X_cur, y_cur

                optimizer.zero_grad()
                if args.architecture == "LSTM_Attention_adapter":
                    outputs, _ = model(X_mix, domain_id=domain_id)
                else:
                    outputs, _ = model(X_mix)

                loss = criterion(outputs, y_mix.long())
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

                # === Update replay memory with CURRENT real examples (after the step) ===
                # Store a light slice (you can store all; reservoir handles quota)
                with torch.no_grad():
                    memory.add_batch(train_domain, X_cur.detach().cpu(), y_cur.detach().cpu())

            epoch_loss /= (i + 1) if i >= 0 else 1
            _sync(device)
            epoch_time = time.perf_counter() - epoch_start
            logging.info(f"[{train_domain}] | Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s")

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/train_loss": float(epoch_loss),
                f"{train_domain}/epoch_time_s": float(epoch_time),
                f"{train_domain}/replay_buffer_size": int(len(memory))
            })

            # ---------- Eval on same-domain test set ----------
            all_y_true, all_y_pred, all_y_prob = [], [], []
            model.eval()
            test_loss = 0.0
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

            metrics = evaluate.evaluate_metrics(
                np.array(all_y_true), np.array(all_y_pred),
                np.array(all_y_prob), train_domain, train_domain
            )
            current_f1 = metrics["f1"]
            current_auc_roc = metrics["roc_auc"]

            run_wandb.log({
                f"{train_domain}/epoch": domain_epoch,
                f"{train_domain}/val_loss": float(test_loss),
                f"{train_domain}/val_f1": float(current_f1),
                f"{train_domain}/val_auc": float(current_auc_roc)
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
                logging.info(f"Early stopping for {train_domain} at epoch {epoch+1}")
                break

        _sync(device)
        domain_training_time = time.perf_counter() - t0
        logging.info(f"Training time for {train_domain}: {domain_training_time:.2f} s")
        domain_training_cost[train_domain].append(domain_training_time)

        # Restore and save best
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model_save_path = f"models/exp_no_{exp_no}_{args.architecture}_{args.algorithm}_{args.scenario}/best_model_after_{train_domain}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(best_model_state, model_save_path)
            previous_domain = train_domain
        else:
            logging.info(f"No improvement for {train_domain}. Model not saved.")

        # Record plasticity/stability on the trained domain (best checkpoint)
        model.eval()
        if args.architecture == "LSTM_Attention_adapter":
            best_metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=domain_id)
        else:
            best_metrics = evaluate_model.eval_model(args, model, test_domain_loader, train_domain, device, domain_id=None)
        current_f1 = best_metrics["f1"]
        performance_plasticity[train_domain].append(current_f1)
        performance_stability[train_domain].append(current_f1)

        # --- Evaluate all previously seen domains (stability/BWT) ---
        logging.info(f"====== Evaluating seen domains after training on {train_domain} ======")
        for test_domain in tqdm(seen_domain, desc="Seen domains eval"):
            if test_domain == train_domain:
                continue
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=domain_to_id[test_domain])
            else:
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=None)
            performance_stability[test_domain].append(metrics["f1"])

        # --- Pick next domain: W2B policy (worst current F1 among unseen) ---
        logging.info(f"====== Evaluating unseen domains after {train_domain} to pick next ======")
        train_w2b = {domain: 0.0 for domain in unseen_domain}
        for test_domain in tqdm(unseen_domain, desc="Unseen domains eval"):
            model.eval()
            if args.architecture == "LSTM_Attention_adapter":
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=domain_to_id[test_domain])
            else:
                metrics = evaluate_model.eval_model(args, model, test_domain_loader, test_domain, device, domain_id=None)
            train_w2b[test_domain] = metrics["f1"]

        if train_w2b:
            train_domain = min(train_w2b, key=train_w2b.get)
            print(f"next train_domain : {train_domain}")
        else:
            continue

        print(f"====== Finished Training on Domain: {previous_domain} ======")

    # -------- Final metrics: BWT / FWT --------
    logging.info("====== Final Metrics after all domains ======")
    bwt_values, bwt_dict, bwt_values_dict = result_utils.compute_BWT(performance_stability, train_domain_order)
    fwt_values, fwt_dict = result_utils.compute_FWT(performance_plasticity, train_domain_order)

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
        "replay_config": {
            "total_capacity": replay_total_capacity,
            "per_domain_cap": replay_per_domain_cap,
            "replay_batch_size": replay_batch_size,
            "replay_ratio": replay_ratio,
            "replay_seen_only": replay_seen_only
        }
    }

    save_results_as_json(results_to_save, filename=f"{exp_no}_experiment_results_{args.architecture}_{args.algorithm}_REPLAY_{args.scenario}.json")
    logging.info("Final training complete. Results saved.")

    run_wandb.summary["BWT/list"] = bwt_values
    run_wandb.summary["FWT/list"] = fwt_values

    