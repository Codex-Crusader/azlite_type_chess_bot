#!/usr/bin/env bash
# quick runner for self-play
set -euo pipefail
EPISODES=${1:-5}
SIMS=${2:-80}
PID=${3:-demo}

python azlite_portfolio_clean.py selfplay --episodes "$EPISODES" --sims "$SIMS" --pid "$PID"
