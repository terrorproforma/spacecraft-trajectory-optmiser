#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
LOCK_FILE="${REPOSITORY_ROOT}/third_party/pdhcg.lock.json"
DESTINATION="${1:-${REPOSITORY_ROOT}/_upstream/pdhcg}"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "PDHCG lock file not found: ${LOCK_FILE}" >&2
  exit 2
fi

readarray -t LOCK_VALUES < <(
  python3 - "${LOCK_FILE}" <<'PY'
import json
import pathlib
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(lock["repository"])
print(lock["commit"])
print(lock.get("tree", ""))
PY
)

UPSTREAM_URL="${LOCK_VALUES[0]}"
UPSTREAM_COMMIT="${LOCK_VALUES[1]}"
EXPECTED_TREE="${LOCK_VALUES[2]}"

if [[ -d "${DESTINATION}/.git" ]]; then
  ACTUAL_REMOTE="$(git -C "${DESTINATION}" remote get-url origin)"
  if [[ "${ACTUAL_REMOTE%.git}" != "${UPSTREAM_URL%.git}" ]]; then
    echo "Existing checkout has unexpected origin: ${ACTUAL_REMOTE}" >&2
    exit 3
  fi
else
  rm -rf "${DESTINATION}"
  mkdir -p "$(dirname "${DESTINATION}")"
  git clone --filter=blob:none --no-checkout "${UPSTREAM_URL}" "${DESTINATION}"
fi

git -C "${DESTINATION}" fetch --no-tags origin "${UPSTREAM_COMMIT}"
git -C "${DESTINATION}" checkout --detach --force "${UPSTREAM_COMMIT}"
git -C "${DESTINATION}" clean -ffdqx

ACTUAL_COMMIT="$(git -C "${DESTINATION}" rev-parse HEAD)"
ACTUAL_TREE="$(git -C "${DESTINATION}" rev-parse 'HEAD^{tree}')"
if [[ "${ACTUAL_COMMIT}" != "${UPSTREAM_COMMIT}" ]]; then
  echo "Pinned PDHCG commit mismatch: ${ACTUAL_COMMIT}" >&2
  exit 4
fi
if [[ -n "${EXPECTED_TREE}" && "${ACTUAL_TREE}" != "${EXPECTED_TREE}" ]]; then
  echo "Pinned PDHCG tree mismatch: ${ACTUAL_TREE}" >&2
  exit 5
fi
if [[ -n "$(git -C "${DESTINATION}" status --porcelain=v1)" ]]; then
  echo "Pinned PDHCG checkout is unexpectedly dirty" >&2
  exit 6
fi

printf 'PDHCG_URL=%s\n' "${UPSTREAM_URL}"
printf 'PDHCG_COMMIT=%s\n' "${ACTUAL_COMMIT}"
printf 'PDHCG_TREE=%s\n' "${ACTUAL_TREE}"
printf 'PDHCG_CHECKOUT=%s\n' "$(realpath "${DESTINATION}")"
