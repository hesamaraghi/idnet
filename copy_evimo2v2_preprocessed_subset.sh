#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-}"
DEST_ROOT="${1:-}"
JOBS="${JOBS:-3}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${SOURCE_ROOT}" || -z "${DEST_ROOT}" ]]; then
  cat >&2 <<'USAGE'
Usage:
  SOURCE_ROOT=user@host:/path/to/EVIMO2v2/preprocessed ./copy_evimo2v2_preprocessed_subset.sh /local/path/to/EVIMO2v2/preprocessed

Optional environment variables:
  SOURCE_ROOT=user@host:/path/to/EVIMO2v2/preprocessed
  JOBS=3
  DRY_RUN=1

Example dry run:
  SOURCE_ROOT=user@host:/path/to/EVIMO2v2/preprocessed DRY_RUN=1 ./copy_evimo2v2_preprocessed_subset.sh "$HOME/datasets/EVIMO2v2/preprocessed"

Example copy:
  SOURCE_ROOT=user@host:/path/to/EVIMO2v2/preprocessed JOBS=3 ./copy_evimo2v2_preprocessed_subset.sh "$HOME/datasets/EVIMO2v2/preprocessed"
USAGE
  exit 2
fi

mkdir -p "${DEST_ROOT}"
DEST_ROOT="$(cd "${DEST_ROOT}" && pwd -P)"

if command -v findmnt >/dev/null 2>&1; then
  DEST_SOURCE="$(findmnt -T "${DEST_ROOT}" -n -o SOURCE || true)"
  SOURCE_HOST="${SOURCE_ROOT%%:*}"
  if [[ "${SOURCE_ROOT}" == *:* && "${DEST_SOURCE}" == "${SOURCE_HOST}:"* ]]; then
    echo "Refusing to copy into ${DEST_ROOT}: it is on the source host mount (${DEST_SOURCE})." >&2
    echo "Choose a real local destination, for example: $HOME/datasets/EVIMO2v2/preprocessed" >&2
    exit 2
  fi
fi

PATHS=(
  # Full eval split.
  "eval/scene13_dyn_test_00_000000"
  "eval/scene13_dyn_test_05_000000"
  "eval/scene14_dyn_test_03_000000"
  "eval/scene14_dyn_test_04_000000"
  "eval/scene14_dyn_test_05_000000"
  "eval/scene15_dyn_test_01_000000"
  "eval/scene15_dyn_test_02_000000"
  "eval/scene15_dyn_test_05_000000"

  # Required 4-sequence train subset.
  "train/scene10_dyn_train_02_000000"
  "train/scene14_dyn_test_01_000000"
  "train/scene15_dyn_test_03_000000"
  "train/scene9_dyn_train_06_000000"

  # Extra train sequences, still keeping the full copy below 400 GB decimal.
  "train/scene10_dyn_train_00_000000"
  "train/scene12_dyn_test_01_000000"
  "train/scene15_dyn_test_04_000000"
  "train/scene6_dyn_train_03_000000"
  "train/scene9_dyn_train_01_000000"
)

copy_one() {
  local rel_path="$1"
  local src="${SOURCE_ROOT}/${rel_path}/"
  local dst="${DEST_ROOT}/${rel_path}/"
  local rsync_opts=(
    -a
    --human-readable
    --info=progress2
    --partial
    --append-verify
    --protect-args
    -e "ssh -T -x -o Compression=no"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    rsync_opts+=(--dry-run)
  fi

  mkdir -p "${dst}"
  echo "==> ${rel_path}"
  rsync "${rsync_opts[@]}" "${src}" "${dst}"
}

running_jobs=0
for rel_path in "${PATHS[@]}"; do
  copy_one "${rel_path}" &
  running_jobs=$((running_jobs + 1))
  if (( running_jobs >= JOBS )); then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
done
wait

echo "Done. Destination: ${DEST_ROOT}"
