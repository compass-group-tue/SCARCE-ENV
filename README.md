<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo-light.svg" alt="SCARCE logo" width="130">
  </picture>
</p>

<h1 align="center">SCARCE</h1>

<p align="center"><b>S</b>hared <b>C</b>ompute <b>AR</b>ena for <b>C</b>oding ag<b>E</b>nts</p>

testbed for studying **strategic behavior among competing LLM agents** that must
share a single execution resource. Five autonomous coding agents each attempt a
[ProgramBench](https://github.com/facebookresearch/ProgramBench) reconstruction task under its own
hard deadline, and must **bid and bargain** for compute time through a sequential time-slot
auction. Their per-round declarations, private messages, and ballots are recorded for analysis.

This is the clean, runnable version of the *ProgramBench-bidding / `pi`-harness* configuration,
plus a single-page **live monitor** for watching runs as they happen.

---

## How it works

Each **worker** is two cooperating agents:

- a **solver** — a minimal agentic coding CLI (the `pi` harness) that works on the ProgramBench
  task inside an isolated Docker container, and
- a **negotiator** — an LLM (routed via LiteLLM) that bids and bargains for execution windows on
  the worker's behalf.

The solver can only touch its container while it holds an **execution window**, obtained through
exactly three MCP tools:

| tool | when | effect |
|---|---|---|
| `request_execution_slot <seconds>` | the only action available with no active window | enters a negotiation round to bid for time |
| `run_command` | only while holding a window | runs one shell command (draws down the granted slot) |
| `save_persistent_memory` | always (free) | writes a note carried into the next subrun |

**Negotiation** is a six-phase time-slot auction among the current candidates:
`declaration → reasoning → recipient_selection → direct_message → response → ballot`. A resolved
round becomes a full **schedule** (position 1 runs immediately, position 2 the moment 1's window
ends, …), and every worker that secures a position pays its own stated price. Bids are public;
each worker's **remaining budget and deadline are private**. There is no hard-coded clearing
rule — agents are told to weigh price as the primary signal and the order emerges from their
ballots.

Tasks are organized into **subruns** (one task per worker per subrun) with a synchronizing
barrier; budget and deadline reset each subrun. After every subrun an LLM writes an after-action
report that is fed into the next subrun's context.

---

## Quick setup

One command builds the whole environment — a Python venv with the **complete** stack (including
`programbench` and `mini-swe-agent`), the `pi` solver CLI, a Docker check, and your API key:

```bash
bash setup_env.sh          # creates .venv, installs everything, captures your key
```

Then run and watch:

```bash
source .venv/bin/activate
source .secrets.env                  # your OpenRouter key (setup_env.sh created this)
bash configs/example_run.sh          # one quick 5-worker, single-subrun run

# in a second terminal:
source .venv/bin/activate
python serve_monitor.py              # -> http://localhost:8000/dashboard/monitor.html
```

The script is **idempotent** (safe to re-run). It needs [`uv`](https://docs.astral.sh/uv/) (or any
Python 3.10+) and — for the two genuinely external pieces — `npm` (for `pi`) and a running Docker.
Prefer to wire things up by hand? The full breakdown is below.

---

## Prerequisites

`setup_env.sh` automates all of this; the list is here as reference (and for manual installs). You
need:

1. **Python 3.10+** and the packages in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. **Rootless Docker**, running. Each task executes in its own container, and evaluation runs in a
   fresh container. Confirm `docker ps` works without `sudo`.
3. **The `pi` harness** — the agentic-coding CLI that drives the solver — on your `PATH`:
   ```bash
   npm install -g @earendil-works/pi-coding-agent@0.74.2   # tested version (setup_env.sh installs this)
   ```
   (The harness is pluggable; `pi` is the one used for the results this repo reproduces.)
4. **The ProgramBench task/eval library** (`programbench`) — loads the task set
   (`programbench.utils.load_data`) and scores submissions (`programbench.eval.eval`) — **and
   `mini-swe-agent`**, which provides the ProgramBench benchmark config
   (`config/benchmarks/programbench.yaml`) and the Docker environment class the solver runs in.
   Both are on PyPI and already listed in `requirements.txt`, so `pip install -r requirements.txt`
   in step 1 covers them. The ProgramBench task images are pulled on first use.
5. **A negotiator API key.** The negotiator is routed by LiteLLM. For the default OpenRouter path:
   ```bash
   export OPENROUTER_API_KEY="KEY"
   ```
   (Use `--azure` with `azure_litellm_gpt54.yaml` for an Azure route instead.)

---

## Run

```bash
export OPENROUTER_API_KEY="KEY"
bash configs/example_run.sh          # one quick 5-worker, single-subrun run
```

or directly:

```bash
python run_experiment.py \
  --model z-ai/glm-5.3 --openrouter \
  --harness pi --harness-model openrouter/z-ai/glm-5.3 \
  --worker-count 5 --tasks-per-worker 4 --fixed-starting-bid \
  --communication-min-k 3 --negotiation-reasoning-summary detailed \
  --container-cpus 1 --container-memory 3g --container-timeout 4h
```

### Key flags

| flag | meaning |
|---|---|
| `--model` (required) | the **negotiator** model, LiteLLM-routed (e.g. `z-ai/glm-5.3`) |
| `--openrouter` / `--azure` | how to route the negotiator |
| `--harness pi` | the solver CLI (`pi`; `claude-code`/`opencode` also supported) |
| `--harness-model` | the model the solver CLI itself uses |
| `--worker-count` | number of competing agents (default 5) |
| `--tasks-per-worker` | subruns per run (use `1` for a quick run, `4` for the full setting) |
| `--fixed-starting-bid` | fixed-bid setting (drop it for revisable bids) |
| `--container-cpus` / `--container-memory` / `--container-timeout` | per-task Docker limits |
| `--output-root` | where runs are written (default `outputs/programbench_auction_subrun_harness_agents`) |

---

## Live monitor

While a run is going (or after), watch it in the browser:

```bash
python serve_monitor.py            # serves this directory on :8000
# open http://localhost:8000/dashboard/monitor.html
```

The monitor polls `outputs/programbench_auction_subrun_harness_agents/` every ~4 s and shows the
**Auction Subrun (Harness)** view: per-worker status, live command counts, the negotiation
communication topology, granted windows, and subrun outcomes.

---

## Output layout

```
outputs/programbench_auction_subrun_harness_agents/runs/<timestamp>/
├── monitor_state.json                 # live dashboard state
├── run_manifest.json                  # index the dashboard reads
├── negotiation_history_full.jsonl     # every negotiation round (declarations, messages, ballots)
├── subrun_reports/<worker>.jsonl      # per-worker after-action reports (resolved, deadline_hit, partial_score)
└── programbench_worker_XX/
    ├── agent_run/.../*.traj.json       # solver trajectory
    └── evaluation/.../*.eval.json      # test results
```

---

## Prompts (edit these to change behavior)

- `queue_negotiation_time_auction_prompts.yaml` — the six-phase negotiator prompts.
- `programbench_auction_report_prompts.yaml` — the after-action report prompt.

Nothing in the Python needs to change to edit prompt wording; pass a different file with
`--queue-negotiation-prompts` / `--subrun-report-prompt-path`.

---

## Repository map

| file | role |
|---|---|
| `run_experiment.py` | entry point (thin wrapper) |
| `run_programbench_auction_subrun_harness_agents.py` | the real orchestrator |
| `run_programbench_auction_subrun_repo_agents.py` | auction scheduler board, time-slot lease env, solver/negotiator prompts |
| `programbench_auction_mcp_server.py` | per-worker MCP server (the three solver tools) |
| `queue_negotiation.py` | the six-phase `QueueNegotiator` |
| `docker_task_scheduler.py` | the shared-resource scheduler + monitor-state writer |
| `agent_harness_adapters.py` | drives the `pi` / Claude Code / OpenCode solver CLIs |
| `run_programbench_subrun_repo_agents.py`, `run_swebench_*` | shared subrun/harness machinery reused by the auction runner |
| `dashboard/monitor.html` | single-tab live monitor |
| `serve_monitor.py` | tiny static server for the monitor 

---


```bash
export ANTHROPIC_API_KEY=...        # or OPENROUTER_API_KEY + --model openrouter/anthropic/claude-opus-4-8
python run_judge.py \
  --history outputs/programbench_auction_subrun_harness_agents/runs/<ts>/negotiation_history_full.jsonl \
  --run-label A1 --setting fixed \
  --run-desc "Group-A ablation (fixed bid), 5 workers, 4 tasks/worker, 1000-pt budget" \
  --out judge_A1.md
```

- `--setting revisable` adds the bid-revision task (for revisable-bid arms).
- `--render-only` builds and prints the exact prompt without any API call (no key/litellm needed).
- The judge appends a machine-readable `json` tally block; `run_judge.py` parses it to
  `<out>.tally.json`. Use `--as-dispatched` to match the original markdown-only output.

---

## Notes

- Reproduces the *fixed-bid, ProgramBench, `pi`-harness* configuration. Other ablation settings
  (revisable bids, unequal/random budgets, random deadlines, no-memory) are toggled with the
  corresponding flags; see `--help`.
- `outputs/` is git-ignored. Do not commit API keys.

