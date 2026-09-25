#!/usr/bin/env bash
# Recipient-side: fetch the pre-computed embeddings bundle from Google Drive.
#
# The bundle holds ONLY the dense embeddings — everything a figure or table
# reads is already committed to this repository, so the bundle is needed only to
# re-run retrieval and the AllSides query-pairing similarity analysis. It is
# ~10 GB and lives on Google Drive; it is fetched with
# rclone (https://rclone.org), which verifies checksums as it copies.
#
# Prereqs: rclone configured with a remote that can read the bundle folder.
# The default reads the shared folder ID through a configured gdrive: remote.
# Override the entire source with BUNDLE_REMOTE.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUNDLE_REMOTE="${BUNDLE_REMOTE:-}"
BUNDLE_FOLDER_ID="1xQmMrxQALkfm4g31L4Ofdwu-QkkqeTHg"
BUNDLE_DRIVE_REMOTE="${BUNDLE_DRIVE_REMOTE:-gdrive:}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "error: rclone not found — install it from https://rclone.org and" >&2
  echo "       configure a remote that can read the bundle folder (see docs/data.md)." >&2
  exit 1
fi

copy_args=(--transfers 8 --checkers 16 --progress)
if [[ -z "$BUNDLE_REMOTE" ]]; then
  # Resolve the shared folder by ID, independent of the reader's My Drive layout.
  BUNDLE_REMOTE="$BUNDLE_DRIVE_REMOTE"
  copy_args+=(--drive-root-folder-id "$BUNDLE_FOLDER_ID")
fi

echo "[fetch] copying embeddings from ${BUNDLE_REMOTE} -> ${REPO_ROOT}"
echo "[fetch] (~10 GB; rclone verifies checksums as it copies)"
rclone copy "${BUNDLE_REMOTE}/" "${REPO_ROOT}/" \
  "${copy_args[@]}"

echo "[fetch] done — embeddings landed under <exp>/embeddings/"
