#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SUMMARY_DIR="outputs/rebuild_reproduction/no_rpm_policy_report"
MARKER_DIR="outputs/rebuild_reproduction/session_markers"
MARKER_PATH="${MARKER_DIR}/no_rpm_policy_report.done"
LOG_DIR="outputs/rebuild_reproduction/logs"
LOG_PATH="${LOG_DIR}/no_rpm_policy_report.log"

mkdir -p "${SUMMARY_DIR}" "${MARKER_DIR}" "${LOG_DIR}"
rm -f "${MARKER_PATH}"

while tmux has-session -t no_rpm_transfer_policy_queue 2>/dev/null; do
  sleep 60
done

while tmux has-session -t no_rpm_realonly_followup_queue 2>/dev/null; do
  sleep 60
done

python3 scripts/summarize_no_rpm_policy_results.py \
  --output-dir "${SUMMARY_DIR}" \
  > "${LOG_PATH}" 2>&1

cp "${SUMMARY_DIR}/summary.json" "${MARKER_PATH}"
