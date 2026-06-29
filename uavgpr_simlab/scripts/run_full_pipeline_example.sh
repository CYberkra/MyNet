#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m uavgpr_simlab.cli pipeline --config configs/pipeline_automation_template.yaml
