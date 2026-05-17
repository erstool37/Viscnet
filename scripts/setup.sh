#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

install_codex() {
  local version="${CODEX_VERSION:-latest}"

  if ! command -v npm >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update
      apt-get install -y nodejs npm
    else
      echo "npm is not installed and apt-get is unavailable; install Node.js/npm first." >&2
      return 1
    fi
  fi

  npm install -g "@openai/codex@${version}"

  if ! command -v codex >/dev/null 2>&1; then
    echo "codex was installed but is not on PATH." >&2
    echo "Check npm global bin path and PATH." >&2
    return 1
  fi

  codex --version
}

if [ "${1:-}" = "--codex-only" ]; then
  install_codex
  exit 0
fi

pip install -r requirements.txt

# Codex is installed into the pod/container filesystem, not the durable PVC.
# Re-run this setup after a pod recreation if `codex` is missing.
if ! command -v codex >/dev/null 2>&1; then
  install_codex
else
  codex --version
fi

# Use a pinned Codex version when needed:
# CODEX_VERSION=0.130.0 bash scripts/setup.sh --codex-only

# Optional manual setup notes:
# apt update
# apt install -y tmux
# tmux new-session -d -s codex
# tmux attach -t codex
# huggingface-cli login
