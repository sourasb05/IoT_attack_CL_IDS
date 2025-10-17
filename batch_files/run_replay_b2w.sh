#!/usr/bin/env bash
set -euo pipefail

declare -a EXTRA_FLAGS=()   # default empty; e.g. EXTRA_FLAGS=(--bidirectional)


PROJECT="attack_CL"
ENTITY="sourasb05"
ARCH="LSTM"
ALGO="Replay"
SCENARIO="b2w"
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
PATIENCE=50
FORGET=0.01
ALPHA=1.0
T=4.0
ENC_LR=0.5
WARM_UP=10
DECAY=0.0001

# Replay-specific knobs
MEMORY_SIZE=4000
PER_DOMAIN_CAP=300
REPLAY_RATIO=0.5
REPLAY_BS=128


# Optional flags (leave empty or add --bidirectional)
EXTRA_FLAGS=() # (--bidirectional)   # e.g., EXTRA_FLAGS=(--bidirectional)

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

    --memory_size "$MEMORY_SIZE" 
    --per_domain_cap "$PER_DOMAIN_CAP" 
    --replay_ratio "$REPLAY_RATIO" 
    --replay_batch_size "$REPLAY_BS"
  )
  # append optional flags safely
  # ARGS+=("${EXTRA_FLAGS[@]:-}")

  # show exactly what will be run (great for debugging)
  printf 'python main.py'; printf ' %q' "${ARGS[@]}"; printf '\n'

  python main.py "${ARGS[@]}"
done
