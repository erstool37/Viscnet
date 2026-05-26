#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SUMMARY_DIR="outputs/rebuild_reproduction/no_rpm_policy_report"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
LOG_DIR="outputs/rebuild_reproduction/logs"
ACCEPTED_MARKER="${MARKER_DIR}/no_rpm_policy.accepted"
RETRY_MARKER="${MARKER_DIR}/no_rpm_policy.retry_required"
LOG_PATH="${LOG_DIR}/no_rpm_policy_acceptance.log"

mkdir -p "${SUMMARY_DIR}" "${MARKER_DIR}" "${LOG_DIR}"
rm -f "${ACCEPTED_MARKER}" "${RETRY_MARKER}"

while tmux has-session -t no_rpm_transfer_policy_queue 2>/dev/null; do
  sleep 60
done

while tmux has-session -t no_rpm_realonly_followup_queue 2>/dev/null; do
  sleep 60
done

set +e
python3 scripts/summarize_no_rpm_policy_results.py \
  --output-dir "${SUMMARY_DIR}" \
  --require-accepted \
  > "${LOG_PATH}" 2>&1
STATUS=$?
set -e

if [ "${STATUS}" -eq 0 ]; then
  cp "${SUMMARY_DIR}/summary.json" "${ACCEPTED_MARKER}"
else
  cp "${SUMMARY_DIR}/summary.json" "${RETRY_MARKER}"
fi

exit "${STATUS}"
