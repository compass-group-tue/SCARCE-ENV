#!/usr/bin/env python3
"""Discover trajectory runs in a folder and build per-run judge inputs + submap.

A "run" is any directory containing a `negotiation_history_full.jsonl` (and,
ideally, a `subrun_reports/` sibling for the TRUE round->subrun map). This script
is source-agnostic: point it at a folder and it finds every run beneath it. No
run_to_raw.json or project-specific keys required.

For each run it writes:
    <out>/in_<RUN>.json   {"run","rounds":[{round_id,takes:[{worker,reasoning,
                          ballot,messages}]}]}   -- each take = one actor-round
    <out>/submap.json     {run:{round_id(str):subrun_index}}  -- true subruns
                          (from subrun_reports/ if present; else null -> the
                          aggregator falls back to a flagged first-quarter split)

Run name = the run directory's basename (so sibling dirs stay distinguishable).
Rounds are deduped by round_id keeping the max-updated_at_monotonic snapshot.

Usage:
    # scan a folder, auto-discover every run under it
    python build_judge_inputs.py --runs-dir /fast/.../runs --out ./ji

    # or name specific run directories explicitly
    python build_judge_inputs.py --run-dirs /path/RUNA /path/RUNB --out ./ji

    # limit how deep the scan goes (default 3 levels)
    python build_judge_inputs.py --runs-dir /path --out ./ji --max-depth 2
"""
from __future__ import annotations

import argparse
import json
import os

NEG_FILE = "negotiation_history_full.jsonl"


def discover_runs(root: str, max_depth: int) -> list:
    """Return [(run_name, neg_file_path)] for every run dir under root."""
    found = []
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        if NEG_FILE in filenames:
            found.append((os.path.basename(dirpath), os.path.join(dirpath, NEG_FILE)))
            dirnames[:] = []  # don't descend into a run dir
    return sorted(found)


def dedup_rounds(path: str) -> dict:
    best: dict = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = o.get("round_id")
            if rid is None:
                continue
            ts = o.get("updated_at_monotonic") or 0
            if rid not in best or ts >= (best[rid].get("updated_at_monotonic") or 0):
                best[rid] = o
    return best


def build_subrun_map(neg_path: str) -> dict:
    out: dict = {}
    srd = os.path.join(os.path.dirname(neg_path), "subrun_reports")
    if not os.path.isdir(srd):
        return out
    for fn in os.listdir(srd):
        with open(os.path.join(srd, fn), errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid, si = r.get("instance_id"), r.get("subrun_index")
                if iid and si:
                    out[iid] = si
    return out


def round_subrun(o: dict, inst2sub: dict):
    subs = [inst2sub.get(c.get("instance_id")) for c in (o.get("candidates") or [])]
    subs = [s for s in subs if s]
    return max(set(subs), key=subs.count) if subs else None


def clip(text, limit: int) -> str:
    return (text or "").replace("\n", " ").strip()[:limit]


def extract_run(neg_path: str):
    inst2sub = build_subrun_map(neg_path)
    best = dedup_rounds(neg_path)
    rounds_out, submap = [], {}
    for rid in sorted(best, key=lambda x: x if isinstance(x, int) else 1 << 62):
        o = best[rid]
        submap[str(rid)] = round_subrun(o, inst2sub)
        per_task = o.get("per_task", {}) or {}
        t2repo = {str(c.get("task_id")): c.get("repo_name")
                  for c in (o.get("candidates") or [])}
        msgs_by_from: dict = {}
        for m in (o.get("transcript", []) or []):
            if m.get("message"):
                msgs_by_from.setdefault(m.get("from_repo_name"), []).append(
                    clip(m.get("message"), 400))
        takes = []
        for tid, d in per_task.items():
            repo = t2repo.get(str(tid), "task" + str(tid))
            takes.append({
                "worker": repo,
                "reasoning": clip(d.get("reasoning"), 900),
                "ballot": clip(d.get("ballot"), 700),
                "messages": msgs_by_from.get(repo, [])[:3],
            })
        rounds_out.append({"round_id": rid, "takes": takes})
    return {"rounds": rounds_out}, submap, bool(inst2sub)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs-dir", help="folder to scan for run dirs")
    src.add_argument("--run-dirs", nargs="+", help="explicit run directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.runs_dir:
        runs = discover_runs(args.runs_dir, args.max_depth)
    else:
        runs = []
        for d in args.run_dirs:
            p = os.path.join(d, NEG_FILE)
            if os.path.exists(p):
                runs.append((os.path.basename(os.path.abspath(d)), p))
            else:
                print(f"  SKIP {d}: no {NEG_FILE}")
    if not runs:
        raise SystemExit("no runs found")

    submaps, no_submap = {}, []
    for name, neg_path in runs:
        data, submap, has_map = extract_run(neg_path)
        data["run"] = name
        with open(os.path.join(args.out, f"in_{name}.json"), "w") as fh:
            json.dump(data, fh, ensure_ascii=False)
        submaps[name] = submap
        if not has_map:
            no_submap.append(name)
        subs = sorted({v for v in submap.values() if v})
        n_takes = sum(len(r["takes"]) for r in data["rounds"])
        tag = "" if has_map else "  [no subrun_reports -> aggregator will first-quarter]"
        print(f"  {name}: {len(data['rounds'])} rounds, {n_takes} takes, subruns {subs}{tag}")

    with open(os.path.join(args.out, "submap.json"), "w") as fh:
        json.dump(submaps, fh)
    print(f"\nwrote {len(submaps)} input files + submap.json to {args.out}")
    if no_submap:
        print(f"WARNING: {len(no_submap)} run(s) lack subrun_reports: "
              f"{', '.join(no_submap)} -- their subruns will be approximated.")


if __name__ == "__main__":
    main()
