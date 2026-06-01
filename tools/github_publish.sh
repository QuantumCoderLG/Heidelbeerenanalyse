#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="${ROOT}/github_upload/berries2.0"
UPLOAD_DIR="${ROOT}/zenodo_upload/v1.0.0"
REPO="QuantumCoderLG/Heidelbeerenanalyse"
GITHUB_API="${GITHUB_API:-https://api.github.com}"
TOKEN_FILE="${GITHUB_TOKEN_FILE:-${HOME}/.config/github/token}"
ACTION="${1:-status}"

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  TOKEN="${GITHUB_TOKEN}"
elif [[ -r "${TOKEN_FILE}" ]]; then
  TOKEN="$(<"${TOKEN_FILE}")"
else
  cat >&2 <<EOF
GitHub API token missing.

Create a fine-grained personal access token for ${REPO}:
  https://github.com/settings/personal-access-tokens/new

Required repository permissions:
  Administration: Read and write  # repository visibility
  Contents: Read and write        # release creation and assets

Store token without posting it in chat:
  mkdir -p ~/.config/github
  chmod 700 ~/.config/github
  printf '%s' 'PASTE_TOKEN_HERE' > ~/.config/github/token
  chmod 600 ~/.config/github/token
EOF
  exit 2
fi

AUTH_HEADER="Authorization: Bearer ${TOKEN}"
API_HEADER="X-GitHub-Api-Version: 2022-11-28"
ACCEPT_HEADER="Accept: application/vnd.github+json"

api_json() {
  curl --fail --silent --show-error \
    -H "${AUTH_HEADER}" \
    -H "${ACCEPT_HEADER}" \
    -H "${API_HEADER}" \
    "$@"
}

repo_status() {
  api_json "${GITHUB_API}/repos/${REPO}" |
    jq -r '"repo=\(.full_name)\nvisibility=\(.visibility)\ndefault_branch=\(.default_branch)\nurl=\(.html_url)"'
}

make_public() {
  cat >&2 <<EOF
WARNING: This changes ${REPO} visibility to public.
Repository source code becomes publicly readable. Type PUBLIC to continue:
EOF
  read -r confirmation
  [[ "${confirmation}" == "PUBLIC" ]] || {
    echo "Cancelled." >&2
    exit 4
  }
  api_json -X PATCH \
    -H "Content-Type: application/json" \
    --data '{"visibility":"public"}' \
    "${GITHUB_API}/repos/${REPO}" >/dev/null
  repo_status
}

decorate_repo() {
  api_json -X PATCH \
    -H "Content-Type: application/json" \
    --data '{
      "description": "Blueberry instance segmentation and multi-stage quality classification with trained models and Windows inference app.",
      "homepage": "https://doi.org/10.5281/zenodo.20479124"
    }' \
    "${GITHUB_API}/repos/${REPO}" >/dev/null
  api_json -X PUT \
    -H "Content-Type: application/json" \
    --data '{
      "names": [
        "blueberry",
        "computer-vision",
        "instance-segmentation",
        "classification",
        "onnx",
        "food-quality"
      ]
    }' \
    "${GITHUB_API}/repos/${REPO}/topics" >/dev/null
  echo "Updated GitHub repository description, homepage and topics."
}

ensure_release_tag() {
  local tag="$1"
  if ! git -C "${STAGING}" rev-parse --verify "refs/tags/${tag}" >/dev/null 2>&1; then
    git -C "${STAGING}" tag -a "${tag}" -m "Heidelbeerenanalyse ${tag}"
  fi
  git -C "${STAGING}" push origin "${tag}"
}

release_notes() {
  local tag="$1"
  if [[ "${tag}" == "v1.0.0" ]]; then
    cat <<'EOF'
Initial public research release.

Includes source code, documentation and Windows inference application.
Large research datasets and model artifacts are published on Zenodo.

Dataset DOI: https://doi.org/10.5281/zenodo.20479053
Software/Model DOI: https://doi.org/10.5281/zenodo.20479124
EOF
  else
    cat <<EOF
Source-code archival release ${tag}.

Includes finalized Zenodo metadata, DOI badges and console publication helpers.
Windows application and trained model bundle remain available from release v1.0.0 and Zenodo.

Dataset DOI: https://doi.org/10.5281/zenodo.20479053
Software/Model DOI: https://doi.org/10.5281/zenodo.20479124
EOF
  fi
}

upload_asset() {
  local upload_url="$1"
  local file="$2"
  local path="${UPLOAD_DIR}/${file}"
  [[ -f "${path}" ]] || {
    echo "Missing release asset: ${path}" >&2
    exit 3
  }
  echo "Uploading GitHub release asset ${file}..."
  curl --fail --silent --show-error \
    -H "${AUTH_HEADER}" \
    -H "${API_HEADER}" \
    -H "Content-Type: application/octet-stream" \
    --data-binary "@${path}" \
    "${upload_url}?name=${file}" >/dev/null
}

create_release() {
  local tag="${1:-v1.0.0}"
  cat >&2 <<EOF
WARNING: This creates public GitHub release ${tag}.
Type RELEASE to continue:
EOF
  read -r confirmation
  [[ "${confirmation}" == "RELEASE" ]] || {
    echo "Cancelled." >&2
    exit 4
  }
  ensure_release_tag "${tag}"
  if api_json "${GITHUB_API}/repos/${REPO}/releases/tags/${tag}" >/dev/null 2>&1; then
    echo "Release ${tag} already exists; refusing duplicate." >&2
    exit 5
  fi
  local response upload_url
  response="$(
    jq -n \
      --arg tag "${tag}" \
      --arg title "Heidelbeerenanalyse ${tag}" \
      --arg body "$(release_notes "${tag}")" \
      '{tag_name:$tag,name:$title,body:$body,draft:false,prerelease:false}' |
      api_json -X POST \
        -H "Content-Type: application/json" \
        --data-binary @- \
        "${GITHUB_API}/repos/${REPO}/releases"
  )"
  upload_url="$(jq -er '.upload_url | sub("\\{\\?name,label\\}$"; "")' <<<"${response}")"
  if [[ "${tag}" == "v1.0.0" ]]; then
    upload_asset "${upload_url}" "Heidelbeeren-Bewertung-App-v1.0.0.zip"
    upload_asset "${upload_url}" "SHA256SUMS-software.txt"
  fi
  jq -r '"release=\(.html_url)"' <<<"${response}"
}

case "${ACTION}" in
  status)
    repo_status
    ;;
  public)
    make_public
    ;;
  decorate)
    decorate_repo
    ;;
  release)
    create_release "${2:-v1.0.0}"
    ;;
  *)
    echo "Usage: bash tools/github_publish.sh [status|public|decorate|release [TAG]]" >&2
    exit 2
    ;;
esac
