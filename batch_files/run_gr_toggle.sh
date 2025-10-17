#!/usr/bin/env bash
set -euo pipefail

PROJECT="attack_CL"
ENTITY="sourasb05"
ARCH="LSTM"
ALGO="GR"
SCENARIO="toggle"
LR=0.001
EPOCHS=100
WINDOW=10              # for main LSTM model
VAE_WINDOW=10          # 👈 separate for VAE sequences (can differ if needed)
STEP=3
BATCH=256
INPUT=140
HIDDEN=10
OUTPUT=2
LAYERS=1
DROPOUT=0.05
PATIENCE=2
FORGET=0.01
ALPHA=1.0
T=4.0
ENC_LR=0.5
WARM_UP=10
DECAY=0.0001

# GR / VAE settings
GR_REPLAY_RATIO=0.40
GR_SYN_PER_EPOCH=2000
USE_TEACHER=1
DISTILL_T=4.0
NUM_CLASSES=2

VAE_HIDDEN=64
VAE_LATENT=32
VAE_EPOCHS=30
VAE_LR=0.001
VAE_BS=128
VAE_BETA_START=0.0
VAE_BETA_END=1.0

SEED=42

TS="$(date +%Y%m%d-%H%M%S)"

for i in {1..3}; do
  PAD=$(printf "%02d" "$i")
  RUN_NAME="exp-${PAD}-${TS}"

  echo "=== Running experiment $i ($RUN_NAME) ==="

  ARGS=(
    --project "$PROJECT"
    --entity "$ENTITY"
    --run_name "$RUN_NAME"
    --learning_rate "$LR"
    --architecture "$ARCH"
    --epochs "$EPOCHS"
    --algorithm "$ALGO"
    --scenario "$SCENARIO"
    --exp_no "$i"
    --window_size "$WINDOW"         # 👈 LSTM window size stays here
    --vae_window_size "$VAE_WINDOW" # 👈 separate arg for GR's VAE
    --step_size "$STEP"
    --batch_size "$BATCH"
    --input_size "$INPUT"
    --hidden_size "$HIDDEN"
    --output_size "$OUTPUT"
    --num_layers "$LAYERS"
    --dropout "$DROPOUT"
    --patience "$PATIENCE"
    --forgetting_threshold "$FORGET"
    --alpha "$ALPHA"
    --temperature "$T"
    --enc_lr_scale "$ENC_LR"
    --warmup_epochs "$WARM_UP"
    --weight_decay "$DECAY"

    # GR-specific
    --gr_replay_ratio "$GR_REPLAY_RATIO"
    --replay_samples_per_epoch "$GR_SYN_PER_EPOCH"
    --distill_T "$DISTILL_T"
    --num_classes "$NUM_CLASSES"
    --num_features "$INPUT"
    --seed "$SEED"
  )

  if [[ "$USE_TEACHER" -eq 1 ]]; then
    ARGS+=(--use_teacher_labels)
  fi

  # VAE-specific
  ARGS+=(
    --vae_hidden "$VAE_HIDDEN"
    --vae_latent "$VAE_LATENT"
    --vae_epochs "$VAE_EPOCHS"
    --vae_lr "$VAE_LR"
    --vae_batch_size "$VAE_BS"
    --vae_beta_start "$VAE_BETA_START"
    --vae_beta_end "$VAE_BETA_END"
  )

  printf 'python main.py'; printf ' %q' "${ARGS[@]}"; printf '\n'
  python main.py "${ARGS[@]}"
done
