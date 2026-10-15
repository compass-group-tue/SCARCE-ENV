---
name: taxonomy-judge-prevalence
description: >-
  Classify negotiation-trajectory runs on ANY taxonomy category using Claude
  Fable LLM judges, and report that category's PREVALENCE (% of takes) per true
  subrun (S1-S4), with judge-variance bounds and run-level significance. Point it
  at any folder of runs; pick any category by code (A1-A20, B1-B12). The category
  definition and discriminating test come from the bundled taxonomy.json. Handles
  the three things that silently corrupt these numbers: compositional-share vs
  prevalence, first-quarter vs true subruns, and single-pass judge noise.
---

# Taxonomy judge prevalence

Classify any set of negotiation runs on **any one taxonomy category** with Fable
judges, and report its **prevalence** (% of takes) per **true subrun**, with
replicate variance bounds and run-level significance.

Everything is general:
- **Any folder** of runs — a run is any directory holding a
  `negotiation_history_full.jsonl` (+ a `subrun_reports/` sibling for true
  subruns). No project-specific run keys required.
- **Any category** — pick a code (e.g. `B11`, `A17`, `A9`); its definition,
  one-line cue, discriminating test, and exclusions are read from
  `taxonomy.json` (all 32 categories; the 12 trap-prone ones carry curated
  discriminators from the project's audits).

## What it produces

For a category and groups of runs: per-subrun **prevalence** trajectory
`[S1,S2,S3,S4]` (mean over replicates), replicate min-max band, median-consensus
trajectory, per-subrun run count `n`, per-run breakdown, and — for two groups —
a run-level S1 comparison (exact permutation + Mann-Whitney + Cohen's d).

## The four rules (non-negotiable)

1. **Prevalence, not compositional share.** Denominator is the number of
   **takes** (actor-rounds), not total labels. Compositional share =
   prevalence ÷ (labels per take); since labels-per-take drifts across subruns, a
   share can move purely because *other* behaviors changed. Prevalence is the
   only honest unit for a trend or between-group comparison.
2. **True subruns, never the first-quarter split.** Map rounds to subruns via
   each run's `subrun_reports/` (`instance_id -> subrun_index`). Subruns have
   uneven round counts, and some runs lose a subrun to infra failures — the
   quarter split invents data. The aggregator falls back to a quarter split only
   when `subrun_reports` is absent, and **flags** it.
3. **Replicates (K≥4) for noisy categories.** Run independent Fable judges
   (vary only the replicate index). Judges agree on the clear core and disagree
   on the borderline tail; a single pass can be off ~2x on low-frequency
   categories. Report mean ± range and the median-consensus.
4. **The category's own discriminating test.** `taxonomy.json` supplies the
   definition + the one test that separates it from its most common look-alike.
   Without it, a loose judge measures the frequent look-alike instead of the
   target. Example — **B11**: counting *bare price-primary* application (the
   instructed mechanism, ~85% of takes) as "rule use" is wrong; strict rule-*making*
   alone (proposing a new standing rule) is the rare ~2-6% variant; the correct
   **widened B11** — rule-making **OR** citation of an *emergent group-built*
   convention, with bare price-primary **excluded** — runs ~15-50%. The taxonomy
   entry now encodes this widened definition.

## Workflow (4 steps)

Repo root `/Users/abhinavk/research_files/coding_agents`; interpreter
`auto_gen_env/bin/python`; `S=~/.claude/skills/taxonomy-judge-prevalence`
(the skill's own folder — scripts resolve `taxonomy.json` relative to themselves,
so the skill runs correctly from this user-level location).

### 1. Build judge inputs from a folder of runs (local, free)

```bash
# scan a whole runs directory...
auto_gen_env/bin/python $S/scripts/build_judge_inputs.py \
  --runs-dir /fast/kabhinav/.../runs --out <SCRATCH>/ji
# ...or name specific run dirs
auto_gen_env/bin/python $S/scripts/build_judge_inputs.py \
  --run-dirs /path/RUN_A /path/RUN_B --out <SCRATCH>/ji
```

Writes `in_<RUN>.json` (deduped rounds; take = reasoning+ballot+messages) and
`submap.json` (true round->subrun). Run name = run-directory basename. Runs
without `subrun_reports` are warned about (their subruns get approximated later).

### 2. Emit a ready-to-run Workflow for the chosen category

```bash
auto_gen_env/bin/python $S/scripts/emit_workflow.py \
  --category B11 --input-dir <ABS>/ji --replicates 4 > <SCRATCH>/wf.js
```

The emitter bakes the category's definition / excludes / discriminator (from
`taxonomy.json`) into a self-contained Workflow script — Workflow scripts have no
filesystem access, so the definition must be inlined. Omit `--runs` to judge every
run in the input dir, or pass `--runs A1 A2 ...` to subset.

### 3. Dispatch the judges

Call the **Workflow tool** with the emitted script as `script`. It fans out
`runs × replicates` Fable agents (`model: 'fable'`), each returning
`{run, rep, per_round, total}`. Save the Workflow's result (the tool writes it to
the task output file).

### 4. Aggregate to prevalence by true subrun

```bash
auto_gen_env/bin/python $S/scripts/aggregate_prevalence.py \
  --results <workflow-result.json> --inputs <SCRATCH>/ji \
  --group A:A1,A2,A3,A4,A5 --group B:B1,B2,B3,B4,B5
```

Omit `--group` to treat all runs as one group. With exactly two groups you also
get the run-level S1 significance test.

## Inspecting / extending the taxonomy

`taxonomy.json` entries: `{code, family, name, definition, cue, discriminator?,
excludes?}`. To list categories: `python -c "import json;[print(t['code'],t['name'])
for t in json.load(open('$S/taxonomy.json'))]"`. To sharpen a category, edit its
`discriminator`/`excludes` and re-emit the workflow — no code change needed. If a
"rare" category comes back at 50-85% prevalence, its look-alike is leaking: tighten
`excludes` and name the key test in `discriminator`.

## Pitfalls this skill prevents

- **Compositional share mistaken for prevalence** — a "share" is always ≤
  prevalence; if it's larger, they are different codings. Always say *share of what*.
- **First-quarter subruns** — phantom S4s and shifted boundaries. Use
  `subrun_reports`; the aggregator flags any fallback.
- **Single-judge point estimates** — unreliable on low-frequency categories.
- **Per-take significance** — fakes tiny p-values; test at the **run** level
  (n = runs per group). Large effects can still be non-significant at n=5.
- **Thin cells hidden** — the aggregator prints per-subrun `n` so a lost/degraded
  subrun (e.g. S4 n=2 after a Docker crash) is visible, not silently averaged.

