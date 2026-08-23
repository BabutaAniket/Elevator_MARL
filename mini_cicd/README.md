# Mini CI/CD Pipeline

A fully local, project-agnostic CI/CD pipeline (inspired by GitHub Actions)
implemented in Python + Bash. It watches a git repository for new commits and
runs **build → unit tests → lint → coverage → Docker image build**, persisting
results to a local SQLite build-history database. This pipeline lives inside
the **Elevator MARL Simulator** repository itself (`mini_cicd/` at the repo
root) and is configured to build/test/lint/containerize that same project
entirely through [`pipeline.yml`](pipeline.yml) — the orchestrator itself
contains no project-specific logic and could target any other git repo by
editing `pipeline.yml` alone.

## Directory Structure

```
mini_cicd/
├── orchestrator.py     # Core pipeline engine (stages, isolation, history, Docker)
├── watcher.py          # Polling-based commit watcher
├── pipeline.yml         # Declarative config for the Elevator MARL target project
├── requirements.txt     # pyyaml
├── run.sh / run.bat     # Convenience one-shot launchers
├── state/               # watcher_state.json (last-seen commit SHA) — created at runtime
├── logs/                # Per-run, per-stage timestamped logs — created at runtime
│   └── runs/<run_id>/
│       ├── workspace/   # Isolated git-worktree checkout used for that run
│       ├── build.log
│       ├── unit_tests.log
│       ├── lint.log
│       ├── coverage.log
│       └── docker_build.log
└── history/
    └── build_history.db # SQLite: runs + stage_results tables
```

## Requirements

- Python 3.9+
- Git (with `git worktree` support — any modern git)
- Docker Desktop / dockerd (only required for the `docker_build` stage)
- `pip install -r requirements.txt`

## Installation

```powershell
cd mini_cicd
pip install -r requirements.txt
```

The target repository is referenced by relative path in `pipeline.yml`
(`project.repo_path: ".."`, i.e. the Elevator MARL repo root, since
`mini_cicd/` lives inside it). Update this if you move the pipeline elsewhere.

## Usage

### Run the pipeline once (manual trigger, builds the current branch tip)

```powershell
python orchestrator.py run
# or a specific commit:
python orchestrator.py run --sha <commit_sha>
```

### Start the polling commit watcher (continuous local "CI server")

```powershell
python watcher.py
# single poll, useful for testing/cron:
python watcher.py --once
```

Whenever a new commit lands on the `branch` configured in `pipeline.yml`
(default `main`), the watcher automatically triggers `orchestrator.py run`.

### Inspect build history

```powershell
python orchestrator.py history
python orchestrator.py show <run_id>
```

### Start the local CI/CD dashboard

```powershell
python -m pip install -r requirements.txt
python dashboard.py
# or on Windows:
run_dashboard.bat
```

Open `http://127.0.0.1:5050`. The dashboard shows the latest 50 runs, commit
SHA, branch, timestamps, overall status, duration, per-stage status/exit
evidence, successful Docker image tags, and clickable stage logs. It reads the
existing SQLite build history and log directory; it does not start or modify
pipeline runs.

## Configuration (`pipeline.yml`)

| Section    | Purpose                                                             |
|------------|----------------------------------------------------------------------|
| `project`  | Target repo path + watched branch                                   |
| `watcher`  | Poll interval + state file location                                 |
| `history`  | SQLite DB path + logs directory                                     |
| `stages`   | Ordered list of stages: `name`, `command`, `working_dir`, `continue_on_failure`, `timeout_seconds`; Docker stage additionally uses `type: docker`, `dockerfile`, `image_name` |

To point the pipeline at a **different** project, copy `pipeline.yml`,
change `repo_path`/`branch`, and rewrite the `stages[].command` values — the
orchestrator code requires no changes.

## Non-Functional Guarantees

- **Isolation** — every run performs `git worktree add --detach <workspace> <sha>`,
  giving each build a brand-new, clean checkout; the worktree is removed after the run.
- **Idempotency** — each run gets a unique `run_id` (`timestamp_shortsha_uuid`), so
  re-running the same commit appends a new history row instead of overwriting prior data.
- **Failure Handling** — a failing stage (unless `continue_on_failure: true`) immediately
  stops all downstream stages; the failure and exit code are recorded in both the
  per-stage log file and the SQLite `stage_results` table.
- **Logging** — every stage writes a timestamped, self-contained log file under
  `logs/runs/<run_id>/<stage>.log`.
- **Dashboard** — `dashboard.py` provides a local Flask interface for build
  history, stage status, image tags, and stage-log inspection.

## Demonstrating Success / Failure

- **Successful run:** ensure `test_sim.py` passes on the current commit, run
  `python orchestrator.py run`, then `python orchestrator.py show <run_id>` — all
  stages show `passed` and the Docker image `elevator-marl:<sha8>` exists (`docker images`).
- **Failed run:** intentionally break a test (e.g. change an assertion in `test_sim.py`)
  or introduce a syntax error, commit it, then run the pipeline again. The `unit_tests`
  stage will show `status=failed`, downstream stages (`coverage`, `docker_build`) will
  be skipped, and the run's overall `status` will be `failed` — all visible via
  `python orchestrator.py show <run_id>` and the corresponding `.log` files.

See [`report.md`](report.md) for the full architectural write-up.
