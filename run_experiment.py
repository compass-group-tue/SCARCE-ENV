#!/usr/bin/env python3
"""Entry point for the ProgramBench time-slot-auction experiment (pi-harness variant).

This is a thin wrapper that forwards every argument to the real runner,
``run_programbench_auction_subrun_harness_agents.py`` (which owns the full CLI).
It exists only so the repo has a stable, obvious entry point.

Example (see configs/example_run.sh and the README for the full flag reference):

    export OPENROUTER_API_KEY=...        # negotiator provider key
    python run_experiment.py \\
        --model z-ai/glm-5.3 --openrouter \\
        --harness pi --harness-model openrouter/z-ai/glm-5.3 \\
        --worker-count 5 --tasks-per-worker 1 --fixed-starting-bid

Output lands in ``outputs/programbench_auction_subrun_harness_agents/runs/<timestamp>/``.
Watch it live with ``python serve_monitor.py`` then open the printed URL.
"""
import os
import runpy
import sys

REAL_RUNNER = "run_programbench_auction_subrun_harness_agents.py"

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.argv[0] = REAL_RUNNER
    runpy.run_path(REAL_RUNNER, run_name="__main__")
