#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 orchestrator.py --config pipeline.yml run "$@"
