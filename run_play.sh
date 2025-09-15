#!/usr/bin/env bash
set -euo pipefail
SIMS=${1:-200}
python azlite_portfolio_clean.py play --sims "$SIMS"
