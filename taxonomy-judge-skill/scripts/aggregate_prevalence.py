#!/usr/bin/env python3
"""Aggregate Fable-judge per-round counts into per-subrun PREVALENCE (% of takes).

Consumes the Workflow result (list of {run, rep, per_round, total}) + the
judge-input dir (submap + take denominators). For each group of runs reports:

  * per-subrun prevalence trajectory [S1..S4]  (mean over replicates)
  * replicate min-max band + median-consensus trajectory
  * per-subrun run count n (thin cells like a lost S4 stay visible)
  * per-run S1..S4 breakdown

With exactly two groups, also a RUN-LEVEL S1 comparison (exact permutation +
Mann-Whitney U + Cohen's d). Run-level is the only honest unit.

Prevalence = (takes coded YES in subrun) / (takes in subrun). Denominator is
TAKES (actor-rounds), never total labels -> prevalence, not compositional share.

Subruns come from the true submap. For runs whose submap is null (no
subrun_reports), a first-quarter-of-rounds split is used and FLAGGED -- an
approximation, not the truth.

Usage:
    python aggregate_prevalence.py --results wf.json --inputs ./ji          # one group ALL
    python aggregate_prevalence.py --results wf.json --inputs ./ji \
        --group A:A1,A2,A3,A4,A5 --group B:B1,B2,B3,B4,B5
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics as S
from collections import defaultdict


def load_results(path: str) -> list:
    obj = json.load(open(path))
    if isinstance(obj, dict) and "result" in obj:
        obj = obj["result"]
    if not isinstance(obj, list):
        raise SystemExit('results must be a JSON array or {"result": [...]}')
    return [r for r in obj if isinstance(r, dict) and "run" in r and "per_round" in r]


def load_inputs(inputs_dir: str):
    submap = json.load(open(os.path.join(inputs_dir, "submap.json")))
    takes, order = {}, {}
    for run in submap:
        p = os.path.join(inputs_dir, f"in_{run}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            takes[run] = {str(r["round_id"]): len(r["takes"]) for r in d["rounds"]}
            order[run] = [str(r["round_id"]) for r in d["rounds"]]  # file order = chrono
    return submap, takes, order


def effective_submap(run, submap, order):
    """True submap, or a flagged first-quarter split if it's all null."""
    sm = submap.get(run, {})
    if any(v in (1, 2, 3, 4) for v in sm.values()):
        return {k: v for k, v in sm.items() if v in (1, 2, 3, 4)}, False
    ids = order.get(run, list(sm.keys()))
    n = len(ids) or 1
    return {rid: min(4, i * 4 // n + 1) for i, rid in enumerate(ids)}, True


def replicates_by_run(results):
    out = defaultdict(list)
    for r in results:
        out[r["run"]].append(r["per_round"])
    return out


def subrun_counts(per_round, sm, takes_run):
    agg = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}
    for rid_s, c in per_round.items():
        si = sm.get(str(rid_s))
        if si in (1, 2, 3, 4):
            agg[si][0] += c
            agg[si][1] += takes_run.get(str(rid_s), 0)
    return agg


def fmt(vals):
    return "[" + ", ".join(f"{v:.1f}" if v is not None else "—" for v in vals) + "]"


def group_report(runs, repls, submap, takes, order):
    sms = {r: effective_submap(r, submap, order)[0] for r in runs}
    approx = [r for r in runs if effective_submap(r, submap, order)[1]]
    max_reps = max((len(repls[r]) for r in runs if r in repls), default=0)

    per_rep = []
    for k in range(max_reps):
        a = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}
        for run in runs:
            if run in repls and k < len(repls[run]):
                g = subrun_counts(repls[run][k], sms[run], takes[run])
                for s in (1, 2, 3, 4):
                    a[s][0] += g[s][0]; a[s][1] += g[s][1]
        per_rep.append([100 * a[s][0] / a[s][1] if a[s][1] else None for s in (1, 2, 3, 4)])

    col = lambda s: [t[s] for t in per_rep if t[s] is not None]
    mean = [round(S.mean(col(s)), 1) if col(s) else None for s in range(4)]
    rng = [(round(min(col(s)), 1), round(max(col(s)), 1)) if col(s) else None for s in range(4)]

    cagg = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}
    for run in runs:
        if run not in repls:
            continue
        rounds = set().union(*[set(pr) for pr in repls[run]])
        for rid in rounds:
            c = S.median([pr.get(rid, 0) for pr in repls[run]])
            si = sms[run].get(str(rid))
            if si in (1, 2, 3, 4):
                cagg[si][0] += c; cagg[si][1] += takes[run].get(str(rid), 0)
    consensus = [round(100 * cagg[s][0] / cagg[s][1], 1) if cagg[s][1] else None for s in (1, 2, 3, 4)]

    per_run, n_runs = {}, {1: 0, 2: 0, 3: 0, 4: 0}
    for run in runs:
        if run not in repls:
            per_run[run] = [None] * 4; continue
        rv = {s: [] for s in (1, 2, 3, 4)}
        for pr in repls[run]:
            g = subrun_counts(pr, sms[run], takes[run])
            for s in (1, 2, 3, 4):
                if g[s][1]:
                    rv[s].append(100 * g[s][0] / g[s][1])
        per_run[run] = [round(S.mean(rv[s]), 0) if rv[s] else None for s in (1, 2, 3, 4)]
        for s in (1, 2, 3, 4):
            if rv[s]:
                n_runs[s] += 1
    return dict(mean=mean, range=rng, consensus=consensus, per_run=per_run,
                n_runs=n_runs, approx=approx)


def run_level_s1(runs, repls, submap, takes, order):
    sms = {r: effective_submap(r, submap, order)[0] for r in runs}
    out = []
    for run in runs:
        if run not in repls:
            continue
        vals = []
        for pr in repls[run]:
            g = subrun_counts(pr, sms[run], takes[run])
            if g[1][1]:
                vals.append(100 * g[1][0] / g[1][1])
        if vals:
            out.append(S.mean(vals))
    return out


def mwu(a, b):
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    rank = [0.0] * len(allv); i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            rank[k] = (i + j) / 2 + 1
        i = j + 1
    ra = sum(rank[k] for k in range(len(allv)) if allv[k][1] == 0)
    na, nb = len(a), len(b)
    return min(ra - na * (na + 1) / 2, na * nb - (ra - na * (na + 1) / 2))


def perm_p(a, b):
    pool, obs, n = a + b, abs(S.mean(a) - S.mean(b)), len(a)
    hits = tot = 0
    for combo in itertools.combinations(range(len(pool)), n):
        g1 = [pool[i] for i in combo]; g2 = [pool[i] for i in range(len(pool)) if i not in combo]
        hits += abs(S.mean(g1) - S.mean(g2)) >= obs - 1e-9; tot += 1
    return hits / tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--group", action="append", help="NAME:run1,run2,... (repeatable)")
    args = ap.parse_args()

    results = load_results(args.results)
    submap, takes, order = load_inputs(args.inputs)
    repls = replicates_by_run(results)

    if args.group:
        groups = {}
        for g in args.group:
            name, runs = g.split(":", 1)
            groups[name] = [r.strip() for r in runs.split(",") if r.strip()]
    else:
        groups = {"ALL": sorted(repls.keys())}

    n_reps = max((len(v) for v in repls.values()), default=0)
    print(f"loaded {len(results)} judge results ({len(repls)} runs, up to {n_reps} reps)\n")

    for name, runs in groups.items():
        rep = group_report(runs, repls, submap, takes, order)
        print(f"=== GROUP {name}  ({', '.join(runs)}) ===")
        print(f"  prevalence (mean over reps) : {fmt(rep['mean'])}")
        print(f"  consensus (median judge)    : {fmt(rep['consensus'])}")
        band = "  ".join(f"S{i+1}[{r[0]}-{r[1]}]" if r else f"S{i+1}[—]"
                         for i, r in enumerate(rep["range"]))
        print(f"  replicate range             : {band}")
        print(f"  runs per subrun (n)         : "
              + "  ".join(f"S{s}={rep['n_runs'][s]}" for s in (1, 2, 3, 4)))
        for run in runs:
            print(f"    {run}: {fmt(rep['per_run'][run])}")
        if rep["approx"]:
            print(f"  NOTE: first-quarter split (no subrun_reports) for: {', '.join(rep['approx'])}")
        print()

    if len(groups) == 2:
        (na, ra), (nb, rb) = list(groups.items())
        a = run_level_s1(ra, repls, submap, takes, order)
        b = run_level_s1(rb, repls, submap, takes, order)
        print(f"=== RUN-LEVEL S1: {na} vs {nb} (n={len(a)} vs {len(b)}) ===")
        print(f"  {na}: {[round(x,1) for x in a]}  mean={S.mean(a):.2f}")
        print(f"  {nb}: {[round(x,1) for x in b]}  mean={S.mean(b):.2f}")
        if len(a) >= 2 and len(b) >= 2:
            sp = math.sqrt((S.pvariance(a) + S.pvariance(b)) / 2) or float("nan")
            d = (S.mean(a) - S.mean(b)) / sp if sp else float("nan")
            print(f"  exact permutation p = {perm_p(a,b):.3f}")
            print(f"  Mann-Whitney U = {mwu(a,b):.1f}  (n=5,5: U<=2 => p<.05)")
            print(f"  Cohen's d = {d:.2f}   (large d + p>.05 => real but underpowered)")


if __name__ == "__main__":
    main()
