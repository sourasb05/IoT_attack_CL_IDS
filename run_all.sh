#!/usr/bin/env bash
set -euo pipefail

declare -a EXTRA_FLAGS=()   # default empty; e.g. EXTRA_FLAGS=(--bidirectional)


PROJECT="attack_CL"
ENTITY="sourasb05"
ARCH="LSTM_Attention"
ALGO="WCL"
SCENARIO="random"
LR=0.05
EPOCHS=500
WINDOW=3
STEP=2
BATCH=128
INPUT=13
HIDDEN=128
OUTPUT=2
LAYERS=2
DROPOUT=0.5
PATIENCE=450
FORGET=0.01


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
  )
  # append optional flags safely


  # show exactly what will be run (great for debugging)
  printf 'python main.py'; printf ' %q' "${ARGS[@]}"; printf '\n'

  python main.py "${ARGS[@]}"
done
