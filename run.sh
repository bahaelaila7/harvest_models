#!/usr/bin/env bash
# Reproduce the harvest-readiness models end to end.
#   1. create an isolated venv and install pinned deps
#   2. build the two labeled datasets from /workspace/FIADB.duckdb
#   3. train + validate both NNs, writing models/metrics/log to outputs/
# Idempotent: re-running rebuilds datasets and retrains from scratch.
set -euo pipefail
cd "$(dirname "$0")"

FIADB="${FIADB:-/workspace/FIADB.duckdb}"
if [[ ! -f "$FIADB" ]]; then
  echo "ERROR: FIA database not found at $FIADB (set FIADB=/path/to/FIADB.duckdb)" >&2
  exit 1
fi

PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then
  echo ">> creating venv + installing deps"
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

mkdir -p outputs
echo ">> building datasets"
$PY build_dataset.py 2>&1 | tee outputs/build.log
echo ">> training + validating"
$PY harvest_nn.py 2>&1 | tee outputs/train.log
echo ">> done. See outputs/summary.json"
