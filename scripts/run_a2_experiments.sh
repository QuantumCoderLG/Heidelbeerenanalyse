#!/usr/bin/env bash

set -euo pipefail

LOG_DIR="outputs/experiment_logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/a2_runs_${STAMP}.txt"

COMMANDS=(
  "python -m src.training.train_backbone_a --config configs/backbone_a.yaml --task a2 --override model.name=efficientnet_b0 data.mask_usage=rgb_only data.use_weighted_sampler=false training.min_recall=0.70 threshold.min_recall_positive=0.80 threshold.fallback_threshold=0.08 threshold.fallback_recall_floor=0.70 calibration.fraction=0.5 threshold.grid_size=1201"
)

{
  echo "A2 experiment sweep started at ${STAMP}"
  echo "Log file: ${LOG_FILE}"
  echo "----------------------------------------"
} >> "${LOG_FILE}"

for cmd in "${COMMANDS[@]}"; do
  {
    echo ">>> COMMAND: ${cmd}"
    echo "----------------------------------------"
  } >> "${LOG_FILE}"

  {
    eval "${cmd}"
  } >> "${LOG_FILE}" 2>&1 || {
    {
      echo "COMMAND FAILED: ${cmd}"
      echo "----------------------------------------"
    } >> "${LOG_FILE}"
  }

  {
    echo "----------------------------------------"
    echo
  } >> "${LOG_FILE}"
done

echo "Done. Results saved to ${LOG_FILE}"
