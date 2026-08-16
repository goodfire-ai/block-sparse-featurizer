# Silico: informative log narration — findings and proposed prompt patch

Follow-up to the `#public-silico-feedback` thread (2026-08-14): *"we should
probably instruct Silico to give more informative logs"*, plus the reply that
interactive worker sessions (salloc/srun) are the least visible surface.

This directory contains a survey of the current prompting and an applyable
patch against `goodfire-ai/silico` (base commit
`d5d3b1418ecd95f253b925722ccde742e101c8aa`, main as of 2026-08-16). It lives in
this repo only because that is this session's designated push target — nothing
here touches block-sparse-featurizer itself.

## Current state (before the patch)

There is no instruction anywhere in Silico's prompt surface to produce
informative logs.

**Submitted (batch) GPU jobs** get log-adjacent guidance only indirectly:

- `SILICO_WORKER.md` ("Compute and dependencies"): every iterative submitted
  job ≳30 s emits phase-level **telemetry** via the `/slurm` patterns
  (`tqdm_telemetry`, `report_progress`, `bridge_wandb`) — but that feeds the
  Resource Usage progress bar, not the log stream.
- The only stdout-content prescription in the entire surface is the W&B
  fallback: *"Without the key, log the same useful metrics to stdout and a
  plot-ready file"* — training jobs only, and only when `WANDB_API_KEY` is
  missing. (`silico/prompts/planner.md` mirrors it.)
- Platform-side, `--output`/`--error` are server-owned and stdout spools to
  `slurm-<jobid>.out` — whatever the script happens to print is exactly what
  the Compute tab shows.

**Interactive sessions** (salloc/srun, on-demand sessions) have zero log
guidance. The injected Compute execution mode block
(`silico/agent_server/image_pull.py::format_compute_execution_mode`) and the
`/slurm` skill cover mechanics only (leases, timeouts, checkpointing, the
interactive-vs-batch split). Worse, the telemetry rule is scoped to
*submitted* jobs, and smoke tests are explicitly exempt — so the interactive
lane is doubly out of scope. Existing prompting actively pushes toward *quiet*
output ("Never relay large bytes through context", capped log tails).

A separate, complementary issue is UI/transport: an interactive `srun`'s
stdout returns as a bounded tool-result capture, so the salloc job's
cluster-side spool only carries command-controller output — that is why the
Compute tab log view for interactive jobs is bare. A prompt change cannot fix
that view; it can only make the content worth surfacing.

## The patch

`0001-informative-log-narration.patch` — 4 files, +21/−2, additive only.
Placement follows the repo's one-authoritative-statement convention (rule in
the owner fragment, mechanics in the skill, pointers elsewhere):

1. **`silico/prompts/fragments/compute_execution.md`** — new
   `#### Narrate work in its logs` subsection (the rule, both lanes): startup
   banner, bounded progress lines every few minutes, phase transitions, exit
   summary with artifact paths, errors with context, unbuffered output;
   interactive commands print what they check and end with a named result
   line; cadence stays bounded (no per-step spam, no tensor dumps).
2. **`plugins/worker/skills/slurm/SKILL.md`** — new `## Log narration`
   section (the line-level recipe): banner contents, print stride sized to
   the work, reuse of telemetry `phase=` labels so log and progress bar tell
   one story, greppable `RESULT metric=value` lines, rank-prefixed multi-rank
   logs, `PYTHONUNBUFFERED=1`/`flush=True`. Skill description gains
   "log narration" for discoverability.
3. **`silico/agent_server/image_pull.py`** — one reinforcing sentence in the
   live interactive-mode block's shared tail (rendered by both the Slurm and
   on-demand arms): make each command's output self-describing; a probe that
   succeeds silently proves nothing.
4. **`SILICO_WORKER.md`** — pointer line updated: `/slurm` owns "telemetry
   and log-narration recipes".

## Verification

The full pytest env couldn't be stood up in this sandbox (broken system
`cryptography` wheel breaks `tests/conftest.py`), so every assertion from the
CI tests that touches these four surfaces was replicated and run standalone —
all pass:

- `tests/ci/test_prompt_skill_hygiene.py`: assembled `SILICO_WORKER.md` word
  budget (9,527 / 13,825 after the patch), byte-deterministic assembly, all
  pinned substrings, all banned phrases (`final/unattended` outside the owner
  fragment, sbatch-invocation regex, storage-capacity figures, standalone
  `df` token, retired vocab).
- `tests/test_worker_job_image_prompt.py`,
  `tests/agent_server/test_image_pull.py`: both rendered arms of
  `format_compute_execution_mode` keep every pinned/banned substring; the
  Modal arm's retired-Slurm-vocabulary ban holds; the new sentence renders in
  both arms. `python -m py_compile` passes.
- `tests/test_worker_contract_fragments.py`,
  `tests/test_worker_compute_fabric_prompt.py`,
  `tests/test_artifact_store_read_routes.py`: all fragment pins hold.

Please run the real suite once applied.

## Apply

```bash
cd silico
git am path/to/0001-informative-log-narration.patch   # keeps message; or:
git apply path/to/0001-informative-log-narration.patch
```
