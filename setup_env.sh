#!/usr/bin/env bash
# =============================================================================
#  One-shot environment setup for the ProgramBench time-slot-auction testbed.
#
#  Creates a Python virtual-env (.venv) with the COMPLETE dependency stack
#  (litellm, datasets, mcp, ... + programbench + mini-swe-agent), installs the
#  `pi` solver CLI, checks Docker, and stores your negotiator API key.
#
#  Usage:
#       bash setup_env.sh
#
#  Idempotent: safe to re-run. Nothing is installed system-wide except `pi`
#  (a global npm package) and, if you don't already have one, a Python
#  interpreter fetched by `uv`.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
VENV="$HERE/.venv"
PY_VERSION="3.12"                                  # >=3.10 required; 3.12 is known-good
PI_NPM_PKG="@earendil-works/pi-coding-agent"       # the `pi` solver harness (npm global)
PI_NPM_VERSION="0.74.2"                             # tested pi version — pinned for reproducibility

# --- pretty output --------------------------------------------------------
b(){ printf '\033[1m%s\033[0m\n' "$*"; }
step(){ printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }
ok(){   printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn(){ printf '  \033[33m!\033[0m %s\n' "$*"; }
err(){  printf '  \033[31m✗\033[0m %s\n' "$*"; }

b "ProgramBench time-slot-auction — environment setup"
b "repo: $HERE"

# --- 1. Python virtual environment ---------------------------------------
step "1/5  Python virtual environment (.venv)"
if command -v uv >/dev/null 2>&1; then
    ok "uv found ($(uv --version)) — using it (fast; can fetch Python $PY_VERSION if missing)"
    uv venv --python "$PY_VERSION" "$VENV" >/dev/null
    PYBIN="$VENV/bin/python"
    pipinstall(){ uv pip install --python "$PYBIN" "$@"; }
else
    CAND=""
    for c in python3.12 python3.11 python3.13 python3.10 python3; do
        command -v "$c" >/dev/null 2>&1 || continue
        if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)'; then
            CAND="$c"; break
        fi
    done
    if [ -z "$CAND" ]; then
        err "No Python >=3.10 found, and 'uv' is not installed."
        err "Fix: install uv  ->  curl -LsSf https://astral.sh/uv/install.sh | sh"
        err "  or install Python 3.12  ->  (macOS) brew install python@3.12"
        exit 1
    fi
    ok "using $CAND ($($CAND --version 2>&1))"
    "$CAND" -m venv "$VENV"
    PYBIN="$VENV/bin/python"
    "$PYBIN" -m pip install --quiet --upgrade pip setuptools wheel
    pipinstall(){ "$PYBIN" -m pip install "$@"; }
fi
ok "venv ready: $VENV"

# --- 2. Python dependencies ----------------------------------------------
step "2/5  Python dependencies (this can take a minute)"
pipinstall -r "$HERE/requirements.txt"
ok "packages installed — verifying imports"
"$PYBIN" - <<'PYEOF'
import importlib, sys
mods = [
    "litellm", "yaml", "datasets", "mcp", "uvicorn", "requests", "psutil",
    "programbench", "programbench.utils.load_data", "programbench.eval.eval",
    "minisweagent", "minisweagent.run.benchmarks.programbench",
]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")
if bad:
    print("  IMPORT FAILURES:")
    for line in bad:
        print("   -", line)
    sys.exit(1)
import programbench, minisweagent
print(f"  ok: programbench {getattr(programbench,'__version__','?')} · "
      f"mini-swe-agent {getattr(minisweagent,'__version__','?')} · python "
      f"{sys.version.split()[0]}")
PYEOF
ok "all imports resolve"

# --- 3. pi solver harness (npm) ------------------------------------------
step "3/5  pi solver harness"
if command -v pi >/dev/null 2>&1; then
    PIV="$(pi --version 2>/dev/null | head -1 || true)"
    ok "pi already on PATH${PIV:+ ($PIV)} — leaving it; tested against $PI_NPM_VERSION"
    warn "to match the tested build:  npm install -g $PI_NPM_PKG@$PI_NPM_VERSION"
elif command -v npm >/dev/null 2>&1; then
    ok "installing $PI_NPM_PKG@$PI_NPM_VERSION globally (npm)"
    npm install -g "$PI_NPM_PKG@$PI_NPM_VERSION"
    if command -v pi >/dev/null 2>&1; then
        PIV="$(pi --version 2>/dev/null | head -1 || true)"
        ok "pi installed${PIV:+ ($PIV)}"
    else
        warn "npm install ran but 'pi' is not on PATH."
        warn "add your npm global bin to PATH:  export PATH=\"\$(npm config get prefix)/bin:\$PATH\""
    fi
else
    warn "npm/Node.js not found — install Node, then:  npm install -g $PI_NPM_PKG@$PI_NPM_VERSION"
fi

# --- 4. Docker ------------------------------------------------------------
step "4/5  Docker (each task runs in an isolated container)"
if ! command -v docker >/dev/null 2>&1; then
    warn "docker not installed — install Docker Desktop (macOS/Win) or rootless Docker (Linux)"
elif docker info >/dev/null 2>&1; then
    ok "docker daemon is up"
else
    warn "docker installed but the daemon is not running"
    if [ "$(uname)" = "Darwin" ]; then
        warn "attempting to start Docker Desktop…"
        open -a Docker >/dev/null 2>&1 || true
        warn "give it ~30s, then confirm with:  docker info"
    else
        warn "start the daemon (e.g. 'systemctl --user start docker' for rootless) and confirm:  docker info"
    fi
fi

# --- 5. Negotiator API key -----------------------------------------------
step "5/5  Negotiator API key"
SECRETS="$HERE/.secrets.env"
if [ -f "$SECRETS" ] && grep -Eq 'OPENROUTER_API_KEY=.{6,}' "$SECRETS" 2>/dev/null; then
    ok ".secrets.env already holds a key"
elif [ -n "${OPENROUTER_API_KEY:-}" ]; then
    ( umask 177; printf 'export OPENROUTER_API_KEY=%s\n' "$OPENROUTER_API_KEY" > "$SECRETS" )
    ok "captured OPENROUTER_API_KEY from your shell into .secrets.env (chmod 600)"
elif [ -t 0 ]; then
    warn "no key found. Paste your OpenRouter key (sk-or-…) and press Enter, or just Enter to skip:"
    printf '  key> '
    read -r -s KEY || true; echo
    if [ -n "${KEY:-}" ]; then
        ( umask 177; printf 'export OPENROUTER_API_KEY=%s\n' "$KEY" > "$SECRETS" )
        ok "saved to .secrets.env (chmod 600, git-ignored)"
    else
        warn "skipped — see .secrets.env.example (copy it to .secrets.env and paste your key)"
    fi
else
    warn "non-interactive shell — set the key before running:"
    warn "  cp .secrets.env.example .secrets.env  &&  edit it   (or: export OPENROUTER_API_KEY=…)"
fi

# --- done -----------------------------------------------------------------
b ""
b "════════════════════════════════════════════════════════════════"
b " Setup complete. To run a quick 5-worker, single-subrun auction:"
b "════════════════════════════════════════════════════════════════"
cat <<EOF

  source .venv/bin/activate      # the env this script just built
  source .secrets.env            # load your negotiator key
  bash configs/example_run.sh    # launch the run

  # in a second terminal, watch it live:
  source .venv/bin/activate
  python serve_monitor.py        # -> http://localhost:8000/dashboard/monitor.html

Output lands in outputs/programbench_auction_subrun_harness_agents/runs/<timestamp>/.
See README.md for all flags and the LLM-as-judge analysis step.
EOF
