#!/usr/bin/env bash

set -euo pipefail

LOG_DIR="outputs/experiment_logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/a3_reproduce_best_${STAMP}.txt"
export PYTHONUNBUFFERED=1

# Ensure A3 manual split exists (generate if necessary)
if [ ! -f "data/instance_crops/splits/a3/train.txt" ] || \
   [ ! -f "data/instance_crops/splits/a3/val.txt" ]   || \
   [ ! -f "data/instance_crops/splits/a3/test.txt" ]; then
  echo "A3 manual split not found; generating with defaults..." | tee -a "${LOG_FILE}"
  python tools/generate_a3_manual_split.py || true
fi

CMD="python -m src.training.train_backbone_a --config configs/backbone_a3.yaml --task a3 --override \
  seed=1337 \
  model.name=efficientnet_b1 \
  data.mask_usage=mask_channel data.color_norm=null data.color_features=[\"redness\",\"darkness\"] \
  threshold.grid_mode=quantile threshold.grid_size=7201 threshold.min_recall_positive=0.70 threshold.min_value=0.6 threshold.max_value=0.999 \
  threshold.fallback_threshold=0.65 threshold.fallback_recall_floor=0.70 \
  training.min_recall=0.85 \
  calibration.enabled=false \
  --fold-id 0"

{
  echo "Reproducing best A3 run at ${STAMP}" 
  echo "Log: ${LOG_FILE}"
  echo "----------------------------------------"
  echo ">>> COMMAND: ${CMD}"
  echo "----------------------------------------"
} | tee -a "${LOG_FILE}"

set -o pipefail
stdbuf -oL -eL bash -lc "${CMD}" 2>&1 | tee -a "${LOG_FILE}"

echo "Done. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"

