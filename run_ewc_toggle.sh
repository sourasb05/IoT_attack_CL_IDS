#!/usr/bin/env bash
set -euo pipefail

declare -a EXTRA_FLAGS=()   # default empty; e.g. EXTRA_FLAGS=(--bidirectional)


PROJECT="attack_CL"
ENTITY="sourasb05"
ARCH="LSTM"
ALGO="EWC"
SCENARIO="toggle"
LR=0.001
EPOCHS=100
WINDOW=10
STEP=3
BATCH=256
INPUT=140
HIDDEN=10
OUTPUT=2
LAYERS=1
DROPOUT=0.05
PATIENCE=100
FORGET=0.01
ALPHA=1.0
T=4.0
ENC_LR=0.5
WARM_UP=10
DECAY=0.0001

# Optional flags (leave empty or add --bidirectional)
EXTRA_FLAGS=() # (--bidirectional)   # e.g., EXTRA_FLAGS=(--bidirectional)

TS="$(date +%Y%m%d-%H%M%S)"

for i in {1..10}; do
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
    --window_size "$WINDOW"
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
    --use_wandb 
  )
  # append optional flags safely
  # ARGS+=("${EXTRA_FLAGS[@]:-}")

  # show exactly what will be run (great for debugging)
  printf 'python main.py'; printf ' %q' "${ARGS[@]}"; printf '\n'

  python main.py "${ARGS[@]}"
done
