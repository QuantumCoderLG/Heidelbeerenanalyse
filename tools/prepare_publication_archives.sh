#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-v1.0.0}"
OUT="${ROOT}/zenodo_upload/${VERSION}"
APP_DIR="${OUT}/Heidelbeeren-Bewertung-App"
RESULTS_ARCHIVE="${OUT}/blueberry-research-results-${VERSION}.zip"
RESULTS_BACKUP=""

if [[ ! -d "${ROOT}/outputs" && -f "${RESULTS_ARCHIVE}" ]]; then
  RESULTS_BACKUP="$(mktemp)"
  cp "${RESULTS_ARCHIVE}" "${RESULTS_BACKUP}"
fi
rm -rf "${OUT}"
mkdir -p "${OUT}"
if [[ -n "${RESULTS_BACKUP}" ]]; then
  mv "${RESULTS_BACKUP}" "${RESULTS_ARCHIVE}"
fi
cd "${ROOT}"

echo "Creating source-image archive..."
zip -0 -q -r "${OUT}/blueberry-source-images-${VERSION}.zip" \
  data/raw \
  data/all_images/Ampel \
  data/all_images/Heidelbeeren2 \
  data/BBoxes_annotation_data \
  docs/DATASET_README.md

echo "Creating curated-crop archive..."
zip -0 -q -r "${OUT}/blueberry-curated-crops-${VERSION}.zip" \
  data/instance_crops/images \
  data/instance_crops/masks \
  data/instance_crops/instance_masks \
  data/instance_crops/metadata \
  data/instance_crops/splits \
  data/instance_crops/rejections \
  docs/DATASET_README.md

echo "Creating model archive..."
zip -0 -q -r "${OUT}/blueberry-models-${VERSION}.zip" \
  inference_assets \
  Kanditaten \
  -x 'Kanditaten/converted/fold_1_best.onnx'

if [[ -d outputs ]]; then
  echo "Creating compact research-results archive..."
  mapfile -t result_files < <(
    find outputs -type f \
      \( -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' \) \
      -print | sort
  )
  if ((${#result_files[@]})); then
    zip -9 -q "${RESULTS_ARCHIVE}" "${result_files[@]}"
  fi
elif [[ ! -f "${RESULTS_ARCHIVE}" ]]; then
  echo "Warning: outputs/ missing; no research-results archive created." >&2
fi

echo "Creating Windows release asset..."
python -m tools.prepare_windows_app --dest "${APP_DIR}" --clean-dest
(
  cd "${OUT}"
  zip -0 -q -r "Heidelbeeren-Bewertung-App-${VERSION}.zip" \
    "Heidelbeeren-Bewertung-App"
  rm -rf "Heidelbeeren-Bewertung-App"
  cp "${ROOT}/docs/ZENODO_UPLOAD.md" UPLOAD_INSTRUCTIONS.md
  cp "${ROOT}/docs/DATASET_README.md" DATASET_README.md
  sha256sum ./*.zip > SHA256SUMS.txt
  sha256sum \
    blueberry-source-images-"${VERSION}".zip \
    blueberry-curated-crops-"${VERSION}".zip \
    > SHA256SUMS-dataset.txt
  sha256sum \
    blueberry-models-"${VERSION}".zip \
    blueberry-research-results-"${VERSION}".zip \
    Heidelbeeren-Bewertung-App-"${VERSION}".zip \
    > SHA256SUMS-software.txt
)

rm -rf "${ROOT}/build/thresholds"
rmdir "${ROOT}/build" 2>/dev/null || true

echo "Created publication archives under ${OUT}"
