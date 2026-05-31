#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/github_upload/berries2.0"

mkdir -p "${DEST}"
rsync -a --delete \
  --exclude '/.git/' \
  --exclude '/.agents/' \
  --exclude '/.codex' \
  --exclude '/.claude/' \
  --exclude '/build/' \
  --exclude '/outputs/' \
  --exclude '/data/' \
  --exclude '/Kanditaten/' \
  --exclude '/zenodo_upload/' \
  --exclude '/github_upload/' \
  --exclude '__pycache__/' \
  --exclude '*.pt' \
  --exclude '*.onnx' \
  --exclude '*.safetensors' \
  "${ROOT}/" "${DEST}/"

mkdir -p "${DEST}/data"
cp "${ROOT}/data/.gitignore" "${DEST}/data/.gitignore"

echo "Synchronized GitHub staging repository under ${DEST}"
