#!/usr/bin/env python3
"""Emit a complete Fable-judge Workflow script for one taxonomy category.

Looks the category up in taxonomy.json (definition + cue + curated discriminator
+ excludes) and prints a self-contained Workflow JS to stdout. Paste the output
as the `script` argument of the Workflow tool; it fans out (runs x replicates)
Fable agents that each code that one category per take and return
{run, rep, per_round, total} via StructuredOutput.

Because Workflow scripts have NO filesystem access, the category definition is
BAKED IN here (not read at run time) -- that is the whole point of this emitter.

Usage:
    python emit_workflow.py --category B11 \
        --input-dir /abs/path/to/judge_inputs \
        --runs A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 \
        --replicates 4 > /tmp/wf_B11.js

    # discover runs from the input dir instead of listing them
    python emit_workflow.py --category A17 --input-dir /abs/ji --replicates 4 > wf.js

Then: Workflow({ script: <contents of wf.js> })
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TAXONOMY = os.path.join(os.path.dirname(HERE), "taxonomy.json")


def load_category(code: str) -> dict:
    tax = json.load(open(TAXONOMY))
    for t in tax:
        if t["code"].upper() == code.upper():
            return t
    codes = ", ".join(t["code"] for t in tax)
    raise SystemExit(f"category {code!r} not found. Available: {codes}")


def runs_from_inputs(input_dir: str) -> list:
    names = []
    for p in sorted(glob.glob(os.path.join(input_dir, "in_*.json"))):
        b = os.path.basename(p)
        names.append(b[len("in_"):-len(".json")])
    return names


def js_str(s: str) -> str:
    """Safe backtick-template contents (escape backslash, backtick, ${)."""
    return (s or "").replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def build_definition_block(cat: dict) -> str:
    code, name = cat["code"], cat["name"]
    parts = [f"{code} = {name}. {cat.get('definition','').strip()}"]
    if cat.get("cue"):
        parts.append(f"In short: {cat['cue']}.")
    return js_str(" ".join(parts))


def build_excludes_block(cat: dict) -> str:
    ex = cat.get("excludes")
    base = ("Do NOT count: takes that order on a DIFFERENT criterion than this "
            "category; empty / timed-out takes; one-off current-round justifications "
            "unless the definition is specifically about the current round.")
    return js_str((ex + " " + base) if ex else base)


def build_discriminator_block(cat: dict) -> str:
    disc = cat.get("discriminator")
    if not disc:
        disc = (f"does the take genuinely match the {cat['code']} definition rather "
                f"than a superficially similar behavior? When unsure, do NOT count.")
    return js_str(disc)


TEMPLATE = r"""// Auto-generated Fable-judge workflow for taxonomy category %(code)s (%(name)s).
// Fans out (runs x replicates) Fable agents; each codes %(code)s per take and
// returns {run, rep, per_round:{<round_id>:count}, total} via StructuredOutput.
export const meta = {
  name: 'taxonomy-judge-%(codelc)s',
  description: 'Code %(code)s (%(name)s) per take with Fable judges for per-subrun prevalence',
  phases: [{ title: 'Judge', detail: 'Fable judges: runs x replicates' }],
}

const INPUT_DIR  = `%(input_dir)s`
const RUNS       = %(runs_js)s
const REPLICATES = %(replicates)d

const CATEGORY     = `%(code)s %(name)s`
const DEFINITION   = `%(definition)s`
const EXCLUDES     = `%(excludes)s`
const DISCRIMINATOR = `%(discriminator)s`

const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    run: { type: 'string' },
    per_round: { type: 'object', additionalProperties: { type: 'integer' } },
    total: { type: 'integer' },
  },
  required: ['run', 'per_round', 'total'],
}

function prompt(run, rep) {
  return `You are a STRICT behavioral judge (independent replicate #${rep}; judge each take on its own merits) coding exactly ONE category -- ${CATEGORY} -- in a multi-agent auction negotiation. Read this file with the Read tool: ${INPUT_DIR}/in_${run}.json

JSON: {"run","rounds":[{round_id, takes:[{worker,reasoning,ballot,messages}]}]}. Each take = one worker's reasoning + ballot + its sent messages in a round.

For EACH take, decide YES/NO:
DEFINITION (count YES if): ${DEFINITION}

EXCLUDES: ${EXCLUDES}

KEY DISCRIMINATOR: ${DISCRIMINATOR} Count conservatively; when unsure, do NOT count.

Return: run = "${run}", per_round = a map of EVERY round_id (as string) to the integer count of takes coded YES in that round (0 if none), and total = the sum.`
}

phase('Judge')
const jobs = []
for (let rep = 1; rep <= REPLICATES; rep++) {
  for (const run of RUNS) {
    jobs.push(() =>
      agent(prompt(run, rep), { schema: SCHEMA, label: `%(code)s r${rep} ${run}`, phase: 'Judge', model: 'fable' })
        .then(r => (r ? { run: r.run, rep, per_round: r.per_round, total: r.total } : null))
    )
  }
}
const results = (await parallel(jobs)).filter(Boolean)
log(`collected ${results.length}/${RUNS.length * REPLICATES} %(code)s judge results`)
return results
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", required=True, help="taxonomy code, e.g. B11, A17, A9")
    ap.add_argument("--input-dir", required=True, help="abs path where in_<RUN>.json live")
    ap.add_argument("--runs", nargs="+", help="run names; default: all in --input-dir")
    ap.add_argument("--replicates", type=int, default=4)
    args = ap.parse_args()

    cat = load_category(args.category)
    runs = args.runs or runs_from_inputs(args.input_dir)
    if not runs:
        raise SystemExit(f"no in_*.json found in {args.input_dir}")

    script = TEMPLATE % {
        "code": cat["code"], "codelc": cat["code"].lower(),
        "name": cat["name"].replace("`", "'"),
        "input_dir": os.path.abspath(args.input_dir),
        "runs_js": json.dumps(runs),
        "replicates": args.replicates,
        "definition": build_definition_block(cat),
        "excludes": build_excludes_block(cat),
        "discriminator": build_discriminator_block(cat),
    }
    print(script)


if __name__ == "__main__":
    main()
