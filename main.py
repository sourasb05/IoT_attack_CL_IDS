import utils as utils
import models as models
import train_CL as train_WCL
import train_WCL_gen as train_WCL_gen
import train_CL_SI as train_si
import train_CL_EWC as train_ewc
import train_CL_EWC_ZS as train_ewc_zs
import train_CL_genreplay as train_genreplay
from utils import cluster_domains  
import numpy as np
from tqdm import trange
import torch
import os
import random
import wandb
import sys
import logging
import datetime

def main():
    args = utils.parse_args()
    #gpu = args.gpu
    #device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    #print("device :",device)
    # ----------------------------
    # Wandb Setup
    # ----------------------------
    # Start a new wandb run to track this script.
    run_wandb = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity=args.entity,
    # Set the wandb project where this run will be logged.
    project=args.project,
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": args.learning_rate,
        "architecture": args.architecture,
        "dataset": "vinnova_attack_dataset",
        "epochs": args.epochs,
        "algorithm": args.algorithm,
        "scenario": args.scenario,
        "exp_no": args.exp_no,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
        "output_size": args.output_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional
    },
)
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
    
    current_directory = os.getcwd()
    
    print(current_directory)

    algorithm = args.algorithm  # "SI/EWC/WCL/Generative_Replay/" 
    scenario = args.scenario    # (randoem/b2w/w2b/clustered/toggle)
    architecture = args.architecture # LSTM/BiLSTM/LSTM_Attention/BiLSTM_Attention/LSTM_Attention_adapter
    exp_no = args.exp_no
    
    # === Setup Logging ===
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(current_directory, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{architecture}_{algorithm}_{scenario}_log_{timestamp}.log")

    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Also print logs to the console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(f"Initialized logging for: {architecture} | {algorithm} | {scenario}  | Experiment No: {args.exp_no}")
    
    logging.info(f"Logs will be saved to: {log_filename}")
    
    domains_path = current_directory + '/data/cross_val_scenario'

    domains = utils.create_domains(domains_path)

    train_domains_loader = {}
    test_domains_loader = {}
    full_domains_loader = {}

    for key, files in domains.items():

        # Construct the expected file names for this experiment
        
        train_file = f"run_{exp_no:02d}_train.csv"
        val_file   = f"run_{exp_no:02d}_val.csv"

        # print(f"train_file: {train_file} | val_file: {val_file}")
        
        # print(f"Processing domain: {key} with {len(files)} datasets : {files}")
        # Check they exist inside this domain’s file list
        # if train_file in files and val_file in files:
        #    print(f"Experiment {exp_no} | Domain: {key} | Train file: {train_file} | Val file:   {val_file}")

        train_domains_loader[key] = utils.load_data(domains_path, key, train_file, window_size=args.window_size, step_size=args.step_size, batch_size=args.batch_size, train=True)
        test_domains_loader[key] = utils.load_data(domains_path, key, val_file, window_size=args.window_size, step_size=args.step_size, train=False)
        print(train_domains_loader[key])
        print(test_domains_loader[key])
        sys.exit(0)
        # print(f"Train loader for domain {key} has {len(train_domains_loader[key])} batches")
        # print(f"Test loader for domain {key} has {len(test_domains_loader[key])} batches")
        
        # logging.info(f"Exp {exp_no} | Domain: {key} | Train size: {len(train_domains_loader[key])} | Test size: {len(test_domains_loader[key])}")
    
    # input_size = 13
    # hidden_size = 128
    # output_size = 2
    # num_layers = 2
    # dropout = 0.5
    # bidirectional = args.bidirectional
    # hidden_size = 128
    # output_size = 2
    # model = models.LSTMModel(input_size, hidden_size, output_size) #.to(device)
    # model = models.LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=2, dropout=0.5, bidirectional=True)
    if architecture == "LSTM":
        model = models.LSTMModel(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional).to(device)
    elif architecture == "BiLSTM":
        model = models.BiLSTMModel(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
    elif architecture == "LSTM_Attention":
        model = models.LSTMModelWithAttention(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional).to(device)
    elif architecture == "BiLSTM_Attention":
        model = models.LSTMModelWithAttention(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional).to(device)
    elif architecture == "LSTM_Attention_adapter":
        model = models.LSTMWithAdapterClassifier(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, num_domains=len(train_domains_loader), num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    logging.info(f" Experiment No: {exp_no} | Algorithm: {algorithm} | Scenario: {scenario}")
    logging.info(f"Model architecture: {architecture} | Input size: {args.input_size} | Hidden size: {args.hidden_size} | Output size: {args.output_size} | Num layers: {args.num_layers} | Dropout: {args.dropout} | Bidirectional: {args.bidirectional}")
    
    if algorithm == "WCL":  
        if scenario == "random":

            train_WCL.train_domain_incremental_model(args, run_wandb, train_domains_loader, test_domains_loader, device,
                                        model, exp_no, num_epochs=args.epochs, learning_rate=args.learning_rate, patience=args.patience)
                                        
        else:
            train_WCL_gen.train_domain_incremental_model_gen(scenario, device, train_domains_loader, test_domains_loader, full_domains_loader, 
                                        model, exp_no, num_epochs=50, learning_rate=0.001, patience=10,forgetting_threshold=0.01)
    
    elif algorithm == "SI":
        train_si.train_domain_incremental_model(train_domains_loader,test_domains_loader,full_domains_loader,model,num_epochs=args.epochs,
            learning_rate=args.learning_rate, device=device, exp_no=exp_no, patience=args.patience, forgetting_threshold=args.forgetting_threshold)
    
    elif algorithm == "EWC":
        # Train using EWC
        if scenario == "Generalization_worst" or scenario == "Generalization_best":
            train_ewc.train_domain_incremental_model(scenario,device,train_domains_loader,test_domains_loader,full_domains_loader,model,
                    num_epochs=args.epochs, learning_rate=args.learning_rate, patience=args.patience, forgetting_threshold=args.forgetting_threshold)
        elif scenario == "zero_shot":
            train_ewc_zs.train_domain_incremental_model(
                    scenario,device,train_domains_loader,test_domains_loader, full_domains_loader, model,
                    num_epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    patience=args.patience,
                    forgetting_threshold=args.forgetting_threshold,
                    tau=0.3,
                    delta=0.8
                )
    
    elif algorithm == "Generative_Replay":
        if scenario == "Generalization_worst" or scenario == "Generalization_best":
            train_genreplay.train_domain_incremental_model(
            scenario,
            device,
            exp_no,
            train_domains_loader,
            test_domains_loader,
            full_domains_loader,
            model,
            num_epochs=50,
            learning_rate=0.001,
            patience=10,
            forgetting_threshold=0.01
        )
        
            
if __name__ == "__main__":
    main()
