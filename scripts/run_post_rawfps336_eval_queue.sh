#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p outputs/rebuild_reproduction/logs outputs/rebuild_reproduction/session_markers

echo "=== $(date -Iseconds) :: realonly phase-offset window41 inference ==="
if [ ! -f outputs/rebuild_reproduction/repro_realonly_993_window10_stride5_phase5_ep50/window41_test_inference/window21_test_inference_metrics.json ]; then
  PYTHONPATH=src torchrun \
    --nproc_per_node="${NPROC_PER_NODE:-4}" \
    --master_port="${WINDOW41_MASTER_PORT:-29910}" \
    scripts/window21_test_inference.py \
    --config outputs/rebuild_reproduction/all_pending_training_queue/configs/repro_realonly_993_window10_stride5_phase5_ep50.yaml
else
  echo "=== $(date -Iseconds) :: realonly phase-offset window41 inference already exists ==="
fi

echo "=== $(date -Iseconds) :: corrected goal4/goal6 phase-offset real-test validation ==="
bash scripts/run_mod_frame_test_queue.sh

echo "=== $(date -Iseconds) :: full pattern leave-one-out training/evaluation ==="
bash scripts/run_pattern_lopo_queue.sh

finished_at="$(date -Iseconds)"
cat > outputs/rebuild_reproduction/session_markers/post_rawfps336_eval_queue.done <<EOF
exit_status=0
finished_at=${finished_at}
goal4_goal6_corrected=outputs/rebuild_reproduction/mod_frame_test_validation/results.json
pattern_lopo=outputs/rebuild_reproduction/pattern_lopo_queue/summary.json
EOF

if [ -n "${POST_RAWFPS336_EVAL_DONE_SIGNAL:-}" ]; then
  tmux wait-for -S "${POST_RAWFPS336_EVAL_DONE_SIGNAL}" || true
fi
