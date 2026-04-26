#!/bin/bash
#
# Hyperparameter tuning grid for all three GNN architectures.
# Results go to results/tuning/<model>/
# Killed runs are logged to results/tuning/killed.log

SAMPLE=0.5
RESULTS_DIR="results/tuning"
KILLED_LOG="results/tuning/killed.log"

MODELS=("hgt" "hmpnn" "hetero_gat")
HIDDEN_DIMS=(32 64 128)
NUM_LAYERS=(2 3)
LRS=(1e-3 5e-4)

mkdir -p "$RESULTS_DIR"
echo "Tuning started: $(date)" > "$KILLED_LOG"
echo "=========================================" >> "$KILLED_LOG"

total=0
killed=0

for model in "${MODELS[@]}"; do
  for hd in "${HIDDEN_DIMS[@]}"; do
    for nl in "${NUM_LAYERS[@]}"; do
      for lr in "${LRS[@]}"; do
        total=$((total + 1))
        desc="$model | hidden=$hd layers=$nl lr=$lr"

        echo ""
        echo "========================================="
        echo "[$total/36] $desc"
        echo "========================================="

        python3 run.py --mode het --model "$model" \
          --sample "$SAMPLE" \
          --hidden-dim "$hd" \
          --num-layers "$nl" \
          --lr "$lr" \
          --patience 15 \
          --epochs 200 \
          --results-dir "$RESULTS_DIR"

        exit_code=$?
        if [ $exit_code -ne 0 ]; then
          killed=$((killed + 1))
          echo "KILLED: $desc (exit code $exit_code)" | tee -a "$KILLED_LOG"
        fi

      done
    done
  done
done

echo ""
echo "========================================="
echo "TUNING COMPLETE"
echo "  Total runs:  $total"
echo "  Killed:      $killed"
echo "  Results in:  $RESULTS_DIR/"
echo "  Kill log:    $KILLED_LOG"
echo "========================================="
