# Guidance for Claude working in this repository

## What this project is

A portfolio project owned by José Miguel Antunes (MSc AI student, targeting ML
engineering roles). It optimizes inference for a small language model one
technique at a time and measures each step. Read `PLAN.md` first: it has the
milestones, the fixed choices and the acceptance criteria. Specs for each rung
are in `docs/rungs/`. The measurement rules are in `docs/benchmark-protocol.md`.

## The rule that matters most

The owner has to be able to explain every part of this in an interview. That
shapes how to work here:

- Prefer the simple, visible implementation over the clever or the wrapped one.
  A hand-written greedy loop beats a library call that hides the mechanism.
- Every rung ships with a note in `docs/defend/` that explains the idea, the
  result, and the questions an interviewer would ask.
- When finishing a rung, tell the owner what changed and why in plain language,
  and point him at what to read. Do not just report that tests pass.
- If he asks why something is the way it is, explain it; do not just change it.

## Numbers

- Never write a performance or quality number that did not come from a run
  recorded under `results/`. No estimates presented as results, no numbers from
  blog posts in the results table.
- Benchmarks that get reported are run on the owner's machine, by him or with
  him watching. Do not commit benchmark results from a sandbox or CI runner as
  if they were the project's numbers.
- A rung that underperforms stays in the table with its real numbers.
- Calibration and tuning use WikiText-2 train. The test split is for final
  evaluation only.

## Workflow

- One milestone per branch and pull request. Do not start a milestone before the
  previous one meets its acceptance criteria.
- The owner wants a lead-and-sub-agents setup: the lead model plans, writes or
  updates the spec, reviews and integrates; sub-agents implement against the
  spec and the `Backend` interface. Give each sub-agent a disjoint set of files.
- Specs describe intent. Library APIs move: before implementing a rung, check
  the current documentation of the libraries involved (Optimum, ONNX Runtime,
  llama.cpp) and update the spec if it was wrong.
- Each rung is complete when it has: backend code, tests on the tiny model, a
  results file from a real run, an updated spec status, and a defend note.

## Code conventions

- Package code in `src/ladder/`, tests in `tests/`. Rungs are registered by name
  in `src/ladder/registry.py`.
- New rungs implement `ladder.backends.base.Backend`. Do not change that
  interface without updating every backend, the service and the protocol doc.
- Unit tests must not download anything. They use the tiny random model fixture
  in `tests/conftest.py`. Tests that need the real model are marked `slow`.
- Optional dependencies stay optional: importing `ladder` must not require
  ONNX, FastAPI or matplotlib.
- Generated models and exports go in `artifacts/` (git-ignored). Results go in
  `results/<machine>/` and are committed.

## Commands

    pip install -e ".[dev,serve]"        # development install
    pytest                               # unit tests, no downloads
    ruff check . && ruff format --check .
    ladder bench   --rung 01_torch_fp32 --machine <name>
    ladder quality --rung 01_torch_fp32 --machine <name>
    ladder report  --machine <name>

Run `ladder --help` for the full option list.
