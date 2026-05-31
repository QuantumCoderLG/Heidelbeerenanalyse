#!/usr/bin/env bash

set -euo pipefail

LOG_DIR="outputs/experiment_logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/a3_runs_${STAMP}.txt"
CONCURRENCY="${CONCURRENCY:-2}"  # default parallel runs
export PYTHONUNBUFFERED=1  # ensure Python logs line-buffered

# Ensure A3 manual split exists (generate if necessary)
if [ ! -f "data/instance_crops/splits/a3/train.txt" ] || \
   [ ! -f "data/instance_crops/splits/a3/val.txt" ]   || \
   [ ! -f "data/instance_crops/splits/a3/test.txt" ]; then
  echo "A3 manual split not found; generating with defaults..."
  python tools/generate_a3_manual_split.py || true
fi

COMMANDS=(
  # === Extra verification runs (A/B/C) ===
  # A) Best-so-far recipe (EffNet-B1 + mask + redness,darkness) with different seed
  "python -m src.training.train_backbone_a --config configs/backbone_a3.yaml --task a3 --override \
   seed=2021 \
   model.name=efficientnet_b1 \
   data.mask_usage=mask_channel data.color_norm=null data.color_features=[\"redness\",\"darkness\"] \
   threshold.grid_mode=quantile threshold.grid_size=7201 threshold.min_recall_positive=0.70 threshold.min_value=0.6 threshold.max_value=0.999 \
   threshold.fallback_threshold=0.65 threshold.fallback_recall_floor=0.70 \
   training.min_recall=0.85 \
   calibration.enabled=false"

  # B) Combine optimizer/reg tweaks with color_features (from COMMAND[05] + redness,darkness)
  "python -m src.training.train_backbone_a --config configs/backbone_a3.yaml --task a3 --override \
   model.name=efficientnet_b1 \
   data.mask_usage=mask_channel data.color_norm=null data.color_features=[\"redness\",\"darkness\"] \
   optimizer.lr=0.00025 optimizer.weight_decay=0.02 model.dropout=0.5 \
   scheduler.name=cosine scheduler.t_max=80 training.max_epochs=80 training.patience=20 \
   threshold.grid_mode=quantile threshold.grid_size=7201 threshold.min_recall_positive=0.70 threshold.min_value=0.6 threshold.max_value=0.999 \
   threshold.fallback_threshold=0.65 threshold.fallback_recall_floor=0.70 \
   training.min_recall=0.85 \
   calibration.enabled=false"

  # C) B + moderated pos_weight to reduce FN cost (like COMMAND[04])
  "python -m src.training.train_backbone_a --config configs/backbone_a3.yaml --task a3 --override \
   model.name=efficientnet_b1 \
   data.mask_usage=mask_channel data.color_norm=null data.color_features=[\"redness\",\"darkness\"] \
   optimizer.lr=0.00025 optimizer.weight_decay=0.02 model.dropout=0.5 \
   scheduler.name=cosine scheduler.t_max=80 training.max_epochs=80 training.patience=20 \
   loss.pos_weight=auto loss.pos_weight_power=0.5 loss.pos_weight_max=2.0 \
   threshold.grid_mode=quantile threshold.grid_size=7201 threshold.min_recall_positive=0.70 threshold.min_value=0.6 threshold.max_value=0.999 \
   threshold.fallback_threshold=0.65 threshold.fallback_recall_floor=0.70 \
   training.min_recall=0.85 \
   calibration.enabled=false"
)

{
  echo "A3 experiment sweep started at ${STAMP}"
  echo "Master log: ${LOG_FILE}"
  echo "Concurrency: ${CONCURRENCY}"
  echo "----------------------------------------"
} > "${LOG_FILE}"

# Run commands with limited concurrency, all output into one master log
declare -a PIDS=()
declare -a IDX=()
declare -a CMDS=()
declare -a CMD_LOGS=()

for i in "${!COMMANDS[@]}"; do
  cmd="${COMMANDS[$i]}"
  idx_num=$((i + 1))
  idx_str=$(printf "%02d" "${idx_num}")
  # Ensure each run writes to a distinct fold directory by appending --fold-id
  fold_id=${i}
  if [[ "${cmd}" == *"--fold-id"* ]]; then
    cmd_with_fold="${cmd}"
  else
    cmd_with_fold="${cmd} --fold-id ${fold_id}"
  fi
  # Per-command log file to keep output grouped
  cmd_log="${LOG_DIR}/a3_runs_${STAMP}_cmd_${idx_str}.log"
  CMD_LOGS+=("${cmd_log}")

  # Launch in background; prefix each line with [idx] and write to per-command log
  (
    set -o pipefail
    {
      echo ">>> COMMAND[${idx_str}]: ${cmd_with_fold}"
      echo "----------------------------------------"
      # Use stdbuf to ensure line-buffered output from child processes
      stdbuf -oL -eL bash -lc "${cmd_with_fold}" 2>&1 | awk -v p="[${idx_str}] " '{ print p $0 }'
    } > "${cmd_log}"
  ) &
  pid=$!

  PIDS+=("${pid}")
  IDX+=("${idx_str}")
  CMDS+=("${cmd_with_fold}")

  # Throttle to CONCURRENCY without reaping (preserve ability to wait later in order)
  while [ "$(jobs -rp | wc -l)" -ge "${CONCURRENCY}" ]; do
    sleep 0.2
  done
done

# After launching all, append each command log to the master log in numeric order
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  idx_str="${IDX[$i]}"
  cmd_log="${CMD_LOGS[$i]}"
  rc=0
  if ! wait "${pid}"; then
    rc=$?
  fi

  # Append grouped output for this command, then the footer
  cat "${cmd_log}" >> "${LOG_FILE}"
  {
    echo "<<< COMMAND[${idx_str}] finished with exit code ${rc}"
    echo "----------------------------------------"
  } >> "${LOG_FILE}"
done

echo "Done. Master log: ${LOG_FILE}"
