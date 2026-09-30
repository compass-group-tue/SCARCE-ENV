#!/usr/bin/env bash
# Example: one 5-worker ProgramBench auction run driven by the pi harness, GLM-5.3 negotiator.
# Requires OPENROUTER_API_KEY in the environment (see README).
set -euo pipefail
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"

python run_experiment.py \
  --model z-ai/glm-5.3 --openrouter \
  --harness pi --harness-model openrouter/z-ai/glm-5.3 \
  --worker-count 5 \
  --tasks-per-worker 1 \
  --fixed-starting-bid \
  --communication-min-k 3 \
  --negotiation-reasoning-summary detailed \
  --container-cpus 1 --container-memory 3g --container-timeout 4h
