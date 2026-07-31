# Mini CI/CD Pipeline — Final Report

## 1. Architecture

The system is a local, dependency-light replacement for a cloud CI service
(GitHub Actions / Jenkins), built from two cooperating Python components and a
single declarative configuration file.

**`watcher.py`** is the *Commit Watcher* stage of the rubric. It polls the
target git repository at a configurable interval, calling `git fetch` and
`git rev-parse <branch>` to resolve the tip commit. It persists the last-seen
SHA to `state/watcher_state.json`. When the resolved SHA differs from the
stored watermark, it shells out to `orchestrator.py run --sha <sha>` and only
advances the watermark after that subprocess returns — so a watcher crash
mid-build can never silently "skip" a commit. Polling (rather than a git
`post-commit` hook) was chosen because it requires no modification of the
target repository's `.git/hooks` and works identically for local-only and
remote-tracked branches, satisfying the "polling is acceptable" constraint.

**`orchestrator.py`** is the pipeline engine and is completely project
agnostic — it contains no reference to Flask, SQLite, PyTorch, or any
Elevator-MARL-specific path. It exposes three CLI subcommands (`run`,
`history`, `show`) and is driven entirely by `pipeline.yml`, which declares:
the target repo path/branch, the historical logging locations, and an ordered
list of *stages*. Each stage is either a generic shell command (`build`,
`unit_tests`, `lint`, `coverage`) or a specially-typed `docker` stage. Because
all commands are shell strings, Bash/PowerShell tooling (venv activation,
`&&` chaining, `pytest`/`flake8`/`coverage` CLIs) is invoked exactly as a
human operator would from a terminal — the Python engine is a thin,
generic *supervisor* around Bash-style command execution, not a reimplementation
of build tools.

For every triggered run, the engine:

1. Resolves the commit SHA (`git rev-parse`).
2. Generates a unique `run_id = <UTC timestamp>_<short sha>_<uuid6>`.
3. Creates an **isolated workspace** via `git worktree add --detach`, i.e. a
   fresh, independent checkout of that exact commit, separate from the
   developer's working tree.
4. Executes each configured stage sequentially inside that workspace,
   redirecting `stdout`/`stderr` to a per-stage log file
   (`logs/runs/<run_id>/<stage>.log`) with explicit `START`/`END` timestamp
   markers.
5. Persists structured results to SQLite (`history/build_history.db`).
6. Tears down the worktree (`git worktree remove --force`), keeping only the
   logs — so disk usage doesn't grow unbounded while isolation is preserved.

## 2. Security, Failure Handling, and Idempotent History

**Failure handling.** Every stage returns a `(status, exit_code)` pair. If a
stage's exit code is non-zero (or it times out, captured via
`subprocess.TimeoutExpired`), its status is recorded as `failed`. Unless the
stage explicitly sets `continue_on_failure: true` in `pipeline.yml` (used
only for the non-blocking `lint` stage, mirroring how linting warnings
typically shouldn't block a build), the orchestrator immediately `break`s out
of the stage loop — no downstream stage (e.g. `coverage`, `docker_build`)
executes after a hard failure. The overall run status is then set to
`failed` and recorded, and a clear message is printed to the console and
written into the stage's own log file.

**Idempotent build history.** The core risk with re-running a pipeline for
the same commit is silently overwriting or corrupting prior evidence of that
commit's history. This is avoided structurally: the primary key of a run is
never the commit SHA alone but `run_id = timestamp + short-sha + random
suffix`, guaranteeing every invocation — even for an identical commit —
inserts a brand-new row in the `runs` table (`INSERT OR IGNORE` is a safety
net against literal run_id collisions, not a substitute for uniqueness).
Stage rows use `INSERT OR REPLACE` keyed on `(run_id, stage_name)`, which
makes only *within-run* stage logging idempotent (e.g. if a stage's log
line were re-emitted), while cross-run data is strictly additive. This gives
a complete, queryable audit trail (`python orchestrator.py history`) of every
attempt against a commit without any data loss.

**Isolation & security posture.** Each run executes against a `git worktree`
checkout rather than the developer's live working directory, so pipeline
runs cannot mutate the repository a developer is actively editing, and stale
artifacts from a previous run cannot leak into a new one (the workspace
directory is deleted and recreated per run). The Docker stage explicitly
probes `docker info` before attempting a build, so a stopped Docker daemon
produces a clear, single-line diagnostic in the log rather than a cryptic
stack trace or a hung process; timeouts are enforced on every subprocess
(shell stages and `docker build`) to prevent a runaway/hanging command from
blocking the pipeline indefinitely. Commands are supplied only via the
trusted, locally-authored `pipeline.yml` (not user/network input), which
avoids the classic "arbitrary command injection from an external trigger"
pitfall of naive polling CI systems.

## 3. Integration With the Elevator MARL Simulator

The Elevator MARL project (`Flask` + `SQLite` + `PyTorch`, at
`../Elevator_MARL` relative to this pipeline) is wired in purely through
`pipeline.yml`, demonstrating the engine's project-agnosticism:

- **`build`** installs `requirements.txt` (`flask`, `numpy`, `torch`) plus the
  pipeline's own tooling (`pytest`, `flake8`, `coverage`) into the isolated
  checkout.
- **`unit_tests`** runs `pytest test_sim.py -v`, the project's existing
  scenario-based test module (traffic-rate sanity check + SCAN-controller
  simulation runs for peak/off-peak hours), emitting a JUnit XML report.
- **`lint`** runs `flake8` across the whole MARL codebase
  (`agents.py`, `environment.py`, `simulation.py`, `traffic.py`, `app.py`, …).
- **`coverage`** re-runs the same test module under `coverage.py`, producing
  both a terminal summary and an HTML report (`coverage_html/`).
- **`docker_build`** builds the image from a `Dockerfile` added to the
  Elevator MARL repo root (`python:3.11-slim` base, installs
  `requirements.txt`, copies the app, exposes port `5000`, runs
  `python app.py`), tagged `elevator-marl:<commit_sha[:8]>`.

Because the simulator is a normal Python package with a standard
`requirements.txt` and `pytest`-discoverable test file, no changes to
`orchestrator.py` were required to support it — only the 60-line
`pipeline.yml`.

## 4. Demonstrating a Successful and a Failed Execution

**Successful execution:** with the Elevator MARL repository in a known-good
state, run `python orchestrator.py run` from `mini_cicd/`. Expected output:
all five stages print `-> passed`, `python orchestrator.py show <run_id>`
reports `status: passed` for the run and every stage, and `docker images`
lists `elevator-marl:<sha8>`.

**Failed execution:** introduce a regression in the Elevator MARL repo (e.g.
break an assertion in `test_sim.py` or a syntax error in `agents.py`) and
commit it. Re-running `python orchestrator.py run` shows the `unit_tests`
(or `build`) stage reported as `failed` with its non-zero exit code, the
pipeline halting immediately (no `coverage`/`docker_build` log files are
created for that run), and the run's overall `status` recorded as `failed` in
both the console output and `python orchestrator.py show <run_id>`. The
corresponding `logs/runs/<run_id>/unit_tests.log` contains the full `pytest`
failure traceback with `START`/`END` timestamps, satisfying the rubric's
requirement for clear, inspectable failure logging.
