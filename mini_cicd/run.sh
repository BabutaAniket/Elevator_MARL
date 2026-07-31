#!/usr/bin/env bash
# Convenience launcher for Linux/macOS: run the pipeline once against the
# current tip of the configured branch.
set -e
cd "$(dirname "$0")"
python3 orchestrator.py --config pipeline.yml run "$@"
