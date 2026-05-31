#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPLOAD_DIR="${ROOT}/zenodo_upload/v1.0.0"
ZENODO_API="${ZENODO_API:-https://zenodo.org/api}"
TOKEN_FILE="${ZENODO_TOKEN_FILE:-${HOME}/.config/zenodo/token}"
DATASET_DOI="10.5281/zenodo.20479053"
SOFTWARE_DOI="10.5281/zenodo.20479124"
ACTION="${1:-status}"

if [[ -n "${ZENODO_TOKEN:-}" ]]; then
  TOKEN="${ZENODO_TOKEN}"
elif [[ -r "${TOKEN_FILE}" ]]; then
  TOKEN="$(<"${TOKEN_FILE}")"
else
  cat >&2 <<EOF
Zenodo token missing.

Create token with scopes deposit:write and deposit:actions:
  https://zenodo.org/account/settings/applications/tokens/new/

Store token without posting it in chat:
  mkdir -p ~/.config/zenodo
  chmod 700 ~/.config/zenodo
  printf '%s' 'PASTE_TOKEN_HERE' > ~/.config/zenodo/token
  chmod 600 ~/.config/zenodo/token
EOF
  exit 2
fi

AUTH_HEADER="Authorization: Bearer ${TOKEN}"

api_json() {
  curl --fail --silent --show-error \
    -H "${AUTH_HEADER}" \
    -H "Content-Type: application/json" \
    "$@"
}

list_depositions() {
  api_json "${ZENODO_API}/deposit/depositions?size=100&all_versions=true"
}

find_deposition_id() {
  local doi="$1"
  list_depositions | jq -er --arg doi "${doi}" '
    .[]
    | select(.metadata.prereserve_doi.doi == $doi)
    | .id
  ' | head -n 1
}

get_deposition() {
  local id="$1"
  api_json "${ZENODO_API}/deposit/depositions/${id}"
}

print_status() {
  local label="$1"
  local doi="$2"
  local id
  id="$(find_deposition_id "${doi}")"
  get_deposition "${id}" | jq -r --arg label "${label}" '
    "-- \($label) --",
    "id=\(.id)",
    "doi=\(.metadata.prereserve_doi.doi)",
    "state=\(.state)",
    "submitted=\(.submitted)",
    "title=\(.metadata.title // "")",
    "bucket=\(.links.bucket)",
    "files:",
    (.files[]? | "  \(.filename // .key)  \(.filesize // .size // "")")
  '
}

dataset_metadata() {
  jq -n '{
    metadata: {
      upload_type: "dataset",
      publication_date: "2026-05-31",
      title: "Blueberry Quality Dataset for Instance Segmentation and Multi-Stage Classification",
      creators: [{name: "Garbe, Lando Maximilian"}],
      description: "<p>Research dataset for blueberry instance segmentation and multi-stage quality classification. Includes source images, XML annotations, curated single-instance crops, masks, metadata tables and manual train/validation/test splits. Original EXIF metadata is intentionally retained.</p>",
      access_right: "open",
      license: "cc-by-4.0",
      version: "1.0.0",
      language: "eng",
      keywords: ["blueberry", "computer vision", "instance segmentation", "classification", "food quality"]
    }
  }'
}

software_metadata() {
  jq -n '{
    metadata: {
      upload_type: "software",
      publication_date: "2026-05-31",
      title: "Heidelbeerenanalyse: Trained Models and Windows Inference Application",
      creators: [{name: "Garbe, Lando Maximilian"}],
      description: "<p>Trained ONNX and PyTorch models, compact research results and Windows inference application for blueberry instance segmentation and multi-stage quality classification.</p>",
      access_right: "open",
      license: "mit-license",
      version: "1.0.0",
      language: "deu",
      keywords: ["blueberry", "computer vision", "ONNX", "instance segmentation", "classification"],
      related_identifiers: [{
        identifier: "https://github.com/QuantumCoderLG/Heidelbeerenanalyse",
        relation: "isSupplementTo",
        resource_type: "software"
      }]
    }
  }'
}

update_metadata() {
  local id="$1"
  local metadata_json="$2"
  api_json -X PUT --data "${metadata_json}" \
    "${ZENODO_API}/deposit/depositions/${id}" >/dev/null
}

prune_unexpected_files() {
  local id="$1"
  shift
  local allowed_file_ids="|$(printf '%s|' "$@")"
  local file_id filename
  while IFS=$'\t' read -r file_id filename; do
    [[ -n "${file_id}" ]] || continue
    if [[ "${allowed_file_ids}" != *"|${filename}|"* ]]; then
      echo "Deleting unexpected draft file ${filename}..."
      api_json -X DELETE \
        "${ZENODO_API}/deposit/depositions/${id}/files/${file_id}" >/dev/null
    fi
  done < <(
    get_deposition "${id}" |
      jq -r '.files[]? | [.id, (.filename // .key)] | @tsv'
  )
}

upload_file() {
  local id="$1"
  shift
  local bucket="$1"
  shift
  local file="$1"
  local path="${UPLOAD_DIR}/${file}"
  if [[ ! -f "${path}" ]]; then
    echo "Missing upload file: ${path}" >&2
    exit 3
  fi
  if get_deposition "${id}" |
    jq -e --arg filename "${file}" '.files[]? | select((.filename // .key) == $filename)' \
      >/dev/null; then
    echo "Keeping existing draft file ${file}."
    return
  fi
  echo "Uploading ${file}..."
  curl --fail --show-error --progress-bar \
    --retry 3 --retry-delay 5 \
    --upload-file "${path}" \
    -H "${AUTH_HEADER}" \
    "${bucket}/${file}" >/dev/null
}

prepare_dataset() {
  local id bucket
  id="$(find_deposition_id "${DATASET_DOI}")"
  bucket="$(get_deposition "${id}" | jq -er '.links.bucket')"
  update_metadata "${id}" "$(dataset_metadata)"
  prune_unexpected_files "${id}" \
    "blueberry-source-images-v1.0.0.zip" \
    "blueberry-curated-crops-v1.0.0.zip" \
    "SHA256SUMS-dataset.txt" \
    "DATASET_README.md"
  upload_file "${id}" "${bucket}" "blueberry-source-images-v1.0.0.zip"
  upload_file "${id}" "${bucket}" "blueberry-curated-crops-v1.0.0.zip"
  upload_file "${id}" "${bucket}" "SHA256SUMS-dataset.txt"
  upload_file "${id}" "${bucket}" "DATASET_README.md"
}

prepare_software() {
  local id bucket
  id="$(find_deposition_id "${SOFTWARE_DOI}")"
  bucket="$(get_deposition "${id}" | jq -er '.links.bucket')"
  update_metadata "${id}" "$(software_metadata)"
  prune_unexpected_files "${id}" \
    "blueberry-models-v1.0.0.zip" \
    "blueberry-research-results-v1.0.0.zip" \
    "Heidelbeeren-Bewertung-App-v1.0.0.zip" \
    "SHA256SUMS-software.txt" \
    "UPLOAD_INSTRUCTIONS.md"
  upload_file "${id}" "${bucket}" "blueberry-models-v1.0.0.zip"
  upload_file "${id}" "${bucket}" "blueberry-research-results-v1.0.0.zip"
  upload_file "${id}" "${bucket}" "Heidelbeeren-Bewertung-App-v1.0.0.zip"
  upload_file "${id}" "${bucket}" "SHA256SUMS-software.txt"
  upload_file "${id}" "${bucket}" "UPLOAD_INSTRUCTIONS.md"
}

configure_drafts() {
  local dataset_id software_id
  dataset_id="$(find_deposition_id "${DATASET_DOI}")"
  software_id="$(find_deposition_id "${SOFTWARE_DOI}")"
  update_metadata "${dataset_id}" "$(dataset_metadata)"
  prune_unexpected_files "${dataset_id}" \
    "blueberry-source-images-v1.0.0.zip" \
    "blueberry-curated-crops-v1.0.0.zip" \
    "SHA256SUMS-dataset.txt" \
    "DATASET_README.md"
  update_metadata "${software_id}" "$(software_metadata)"
  prune_unexpected_files "${software_id}" \
    "blueberry-models-v1.0.0.zip" \
    "blueberry-research-results-v1.0.0.zip" \
    "Heidelbeeren-Bewertung-App-v1.0.0.zip" \
    "SHA256SUMS-software.txt" \
    "UPLOAD_INSTRUCTIONS.md"
  echo "Configured both Zenodo drafts."
}

publish_one() {
  local doi="$1"
  local id
  id="$(find_deposition_id "${doi}")"
  api_json -X POST "${ZENODO_API}/deposit/depositions/${id}/actions/publish" >/dev/null
}

case "${ACTION}" in
  status)
    print_status "dataset" "${DATASET_DOI}"
    echo
    print_status "software" "${SOFTWARE_DOI}"
    ;;
  configure)
    configure_drafts
    ;;
  prepare)
    prepare_dataset
    prepare_software
    echo
    echo "Draft upload complete. Review with:"
    echo "  bash tools/zenodo_drafts.sh status"
    echo
    echo "Publish only after review:"
    echo "  bash tools/zenodo_drafts.sh publish"
    ;;
  publish)
    cat >&2 <<EOF
WARNING: This publishes both Zenodo drafts. Published records become public.
Files cannot be freely replaced afterward. Type PUBLISH to continue:
EOF
    read -r confirmation
    [[ "${confirmation}" == "PUBLISH" ]] || {
      echo "Cancelled." >&2
      exit 4
    }
    publish_one "${DATASET_DOI}"
    publish_one "${SOFTWARE_DOI}"
    echo "Published both Zenodo records."
    ;;
  *)
    echo "Usage: bash tools/zenodo_drafts.sh [status|configure|prepare|publish]" >&2
    exit 2
    ;;
esac
