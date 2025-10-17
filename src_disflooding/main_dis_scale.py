import torch
import torch.nn as nn
import os
import utils as utils
import models as models
from evaluate_with_metrices import evaluate_with_metrics
from train_domain import train_domain

def main():
    args = utils.parse_args()
        # ----------------------------
        # 0. Device Setup (MPS/CUDA/CPU)
        # ----------------------------
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    cwd = os.getcwd()
    print(cwd)
    pwd = os.path.abspath(os.path.join(cwd, ".."))
    print(f"Parent working directory: {pwd}")



    domains_path = pwd + '/data/cleaned_dis_scale'

    domains = utils.create_domains(domains_path)

    train_domains_loader = {}
    test_domains_loader = {}

    for key, files in domains.items():
        if key in ["disflooding_var25_base", "disflooding_var30_base", "disflooding_var35_base"]:

            train_domains_loader[key], test_domains_loader[key] = utils.load_data(domains_path, key, files, window_size=args.window_size, step_size=args.step_size, batch_size=args.batch_size)
    

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Aggregate results across domains
    all_domain_results = {}

    # Train from scratch per domain
    for key in sorted(train_domains_loader.keys()):
        if key in ["disflooding_var25_base", "disflooding_var30_base", "disflooding_var35_base"]:
            print(f"\n=== Training fresh model for domain: {key} ===")
            train_loader = train_domains_loader[key]
            test_loader  = test_domains_loader[key]

            model, best_state, history, best_val_acc, ckpt_path = train_domain(
                key=key,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                arch=args.architecture,
                hidden_dim=args.hidden_size,
                output_dim=args.output_size,
                num_layers=args.num_layers,
                fc_hidden_dim=10,
                lr=args.learning_rate,
                epochs=args.epochs,
                patience=args.patience,
                checkpoints_dir="checkpoints"
            )

            # Detailed metrics (AUC, F1, Acc, CM)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            detailed = evaluate_with_metrics(model, test_loader, device, criterion)

            # Merge results
            domain_result = {
                "domain": key,
                "checkpoint": ckpt_path,
                "best_val_acc": float(best_val_acc),
                "epochs_trained": len(history["val_acc"]),
                **detailed
            }

            # Save per-domain metrics
            utils.save_results_as_json(domain_result, filename=f"{key}_metrics.json", save_folder="results")

            # Add to aggregate
            all_domain_results[key] = domain_result

            # Optional: also save CM as a CSV for easy viewing
            cm = detailed.get("confusion_matrix")
            if cm is not None:
                import csv
                cm_csv = os.path.join("results", f"{key}_confusion_matrix.csv")
                with open(cm_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["", "Pred 0", "Pred 1"])
                    tn, fp = cm[0]
                    fn, tp = cm[1]
                    writer.writerow(["True 0", tn, fp])
                    writer.writerow(["True 1", fn, tp])
                print(f"[{key}] Saved confusion matrix CSV: {cm_csv}")

        # Save aggregated results
        utils.save_results_as_json(all_domain_results, filename="all_domains_metrics.json", save_folder="results")
        print(f"\n Saved aggregated results for {len(all_domain_results)} domains to results/all_domains_metrics.json")
        
if __name__ == "__main__":

    main()