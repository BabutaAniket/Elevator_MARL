#!/usr/bin/env python3
"""
orchestrator.py -- Mini CI/CD Pipeline Engine
================================================
A fully local, project-agnostic CI/CD orchestrator. All project-specific
knowledge lives in pipeline.yml; this engine only knows how to:

  1. Resolve the target commit in the target git repository.
  2. Check out that commit into an *isolated* workspace (git worktree).
  3. Run each configured stage as a subprocess, capturing timestamped logs.
  4. Stop downstream stages on failure (unless continue_on_failure: true).
  5. Build & tag a Docker image (if a docker-type stage is configured).
  6. Persist immutable, idempotent build-history records to SQLite.

Usage:
    python orchestrator.py run [--config pipeline.yml] [--sha <commit_sha>]
    python orchestrator.py history [--limit 20]
    python orchestrator.py show <run_id>
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class PipelineError(RuntimeError):
    """Raised for fatal, non-stage-related pipeline errors (bad config, git errors)."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

class Config:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.root = self.config_path.parent
        with open(self.config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        project = raw.get("project", {})
        self.project_name = project.get("name", "unnamed-project")
        self.repo_path = (self.root / project["repo_path"]).resolve()
        self.branch = project.get("branch", "main")

        watcher = raw.get("watcher", {})
        self.poll_interval = int(watcher.get("poll_interval_seconds", 10))
        self.state_file = self.root / watcher.get("state_file", "state/watcher_state.json")

        history = raw.get("history", {})
        self.db_path = self.root / history.get("db_path", "history/build_history.db")
        self.logs_dir = self.root / history.get("logs_dir", "logs")

        self.stages = raw.get("stages", [])
        if not self.stages:
            raise PipelineError("pipeline.yml defines no stages")

        if not self.repo_path.exists():
            raise PipelineError(f"repo_path does not exist: {self.repo_path}")


# --------------------------------------------------------------------------- #
# Build history persistence (SQLite)
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    commit_sha  TEXT NOT NULL,
    branch      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    duration_s  REAL
);

CREATE TABLE IF NOT EXISTS stage_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    stage_name  TEXT NOT NULL,
    status      TEXT NOT NULL,
    exit_code   INTEGER,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    duration_s  REAL,
    log_path    TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE(run_id, stage_name)
);
"""


class BuildHistory:
    """
    Idempotent, append-only build history store.

    Idempotency guarantee: every pipeline execution gets a unique run_id
    (derived from commit SHA + timestamp + short uuid), so re-running the
    pipeline for the *same* commit simply appends a new run row rather than
    overwriting or corrupting any prior run's history. Stage rows use
    INSERT OR REPLACE keyed on (run_id, stage_name), so re-processing the
    same run_id (e.g. a crash-recovery re-log) safely upserts instead of
    duplicating rows.
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def start_run(self, run_id, project, commit_sha, branch):
        self.conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, project, commit_sha, branch, started_at, status) "
            "VALUES (?, ?, ?, ?, ?, 'running')",
            (run_id, project, commit_sha, branch, utc_now_iso()),
        )
        self.conn.commit()

    def record_stage(self, run_id, stage_name, status, exit_code, started_at, finished_at, log_path):
        duration = None
        if started_at and finished_at:
            duration = (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds()
        self.conn.execute(
            "INSERT OR REPLACE INTO stage_results "
            "(run_id, stage_name, status, exit_code, started_at, finished_at, duration_s, log_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, stage_name, status, exit_code, started_at, finished_at, duration, str(log_path)),
        )
        self.conn.commit()

    def finish_run(self, run_id, status, started_at):
        finished_at = utc_now_iso()
        duration = (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds()
        self.conn.execute(
            "UPDATE runs SET status=?, finished_at=?, duration_s=? WHERE run_id=?",
            (status, finished_at, duration, run_id),
        )
        self.conn.commit()

    def list_runs(self, limit=20):
        cur = self.conn.execute(
            "SELECT run_id, project, commit_sha, branch, started_at, status, duration_s "
            "FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()

    def get_run(self, run_id):
        run = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        stages = self.conn.execute(
            "SELECT stage_name, status, exit_code, duration_s, log_path FROM stage_results "
            "WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return run, stages


# --------------------------------------------------------------------------- #
# Git helpers
# --------------------------------------------------------------------------- #

def git(*args, cwd):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_commit_sha(repo_path: Path, branch: str, requested_sha: str = None) -> str:
    if requested_sha:
        return git("rev-parse", requested_sha, cwd=repo_path)
    return git("rev-parse", branch, cwd=repo_path)


def create_isolated_worktree(repo_path: Path, commit_sha: str, dest: Path) -> Path:
    """Checks out `commit_sha` into a brand-new, isolated worktree directory."""
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "--detach", str(dest), commit_sha, cwd=repo_path)
    return dest


def remove_worktree(repo_path: Path, dest: Path):
    try:
        git("worktree", "remove", "--force", str(dest), cwd=repo_path)
    except PipelineError:
        shutil.rmtree(dest, ignore_errors=True)
        try:
            git("worktree", "prune", cwd=repo_path)
        except PipelineError:
            pass


# --------------------------------------------------------------------------- #
# Stage execution
# --------------------------------------------------------------------------- #

def run_shell_stage(stage, cwd: Path, log_path: Path, timeout: int):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now_iso()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"=== STAGE '{stage['name']}' START {started} ===\n")
        log.write(f"cwd: {cwd}\ncommand: {stage['command']}\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                stage["command"],
                shell=True,
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\n!!! STAGE TIMED OUT after {timeout}s !!!\n")
            exit_code = -1
        finished = utc_now_iso()
        status = "passed" if exit_code == 0 else "failed"
        log.write(f"\n=== STAGE '{stage['name']}' END {finished} status={status} exit_code={exit_code} ===\n")
    return status, exit_code, started, finished


def run_docker_stage(stage, cwd: Path, log_path: Path, commit_sha: str, timeout: int):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now_iso()
    image = stage.get("image_name", "app")
    tag = f"{image}:{commit_sha[:8]}"
    dockerfile = stage.get("dockerfile", "Dockerfile")

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"=== STAGE '{stage['name']}' (docker) START {started} ===\n")
        log.write(f"cwd: {cwd}\nimage: {tag}\ndockerfile: {dockerfile}\n\n")
        log.flush()

        # Basic daemon-availability check so failures are diagnosed cleanly.
        try:
            ping = subprocess.run(["docker", "info"], capture_output=True, text=True)
            daemon_ok = ping.returncode == 0
            daemon_error = ping.stderr
        except FileNotFoundError:
            daemon_ok = False
            daemon_error = "`docker` executable not found on PATH."

        if not daemon_ok:
            log.write("!!! Docker is unavailable (daemon not running or CLI not installed) !!!\n")
            log.write(daemon_error + "\n")
            finished = utc_now_iso()
            log.write(f"=== STAGE '{stage['name']}' END {finished} status=failed (daemon unavailable) ===\n")
            return "failed", -2, started, finished

        try:
            proc = subprocess.run(
                ["docker", "build", "-f", dockerfile, "-t", tag, "."],
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\n!!! DOCKER BUILD TIMED OUT after {timeout}s !!!\n")
            exit_code = -1
        except FileNotFoundError:
            log.write("!!! `docker` executable not found on PATH. !!!\n")
            exit_code = -3

        finished = utc_now_iso()
        status = "passed" if exit_code == 0 else "failed"
        if status == "passed":
            log.write(f"\nBuilt and tagged image: {tag}\n")
        log.write(f"\n=== STAGE '{stage['name']}' END {finished} status={status} exit_code={exit_code} ===\n")
    return status, exit_code, started, finished


# --------------------------------------------------------------------------- #
# Pipeline runner
# --------------------------------------------------------------------------- #

def run_pipeline(config: Config, requested_sha: str = None) -> int:
    commit_sha = resolve_commit_sha(config.repo_path, config.branch, requested_sha)
    run_id = f"{utc_now_stamp()}_{commit_sha[:8]}_{uuid.uuid4().hex[:6]}"
    run_dir = config.logs_dir / "runs" / run_id
    workspace = run_dir / "workspace"

    history = BuildHistory(config.db_path)
    started_at = utc_now_iso()
    history.start_run(run_id, config.project_name, commit_sha, config.branch)

    print(f"[orchestrator] run_id={run_id} commit={commit_sha[:8]} branch={config.branch}")

    overall_status = "passed"
    try:
        create_isolated_worktree(config.repo_path, commit_sha, workspace)

        for stage in config.stages:
            name = stage["name"]
            timeout = int(stage.get("timeout_seconds", 600))
            cwd = (workspace / stage.get("working_dir", ".")).resolve()
            log_path = run_dir / f"{name}.log"

            print(f"[orchestrator] -> stage '{name}' ...")
            if stage.get("type") == "docker":
                status, exit_code, s_at, f_at = run_docker_stage(stage, cwd, log_path, commit_sha, timeout)
            else:
                status, exit_code, s_at, f_at = run_shell_stage(stage, cwd, log_path, timeout)

            history.record_stage(run_id, name, status, exit_code, s_at, f_at, log_path)
            print(f"[orchestrator]    stage '{name}' -> {status} (exit={exit_code}) log={log_path}")

            if status != "passed":
                overall_status = "failed"
                if not stage.get("continue_on_failure", False):
                    print(f"[orchestrator] stage '{name}' failed; stopping downstream stages.")
                    break
    except PipelineError as exc:
        overall_status = "error"
        print(f"[orchestrator] FATAL: {exc}", file=sys.stderr)
    except Exception as exc:  # defense in depth: never leave a run's status unrecorded
        overall_status = "error"
        print(f"[orchestrator] UNEXPECTED ERROR: {exc}", file=sys.stderr)
    finally:
        remove_worktree(config.repo_path, workspace)
        history.finish_run(run_id, overall_status, started_at)

    print(f"[orchestrator] run {run_id} finished with status={overall_status}")
    return 0 if overall_status == "passed" else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def cmd_run(args):
    config = Config(Path(args.config))
    return run_pipeline(config, requested_sha=args.sha)


def cmd_history(args):
    config = Config(Path(args.config))
    history = BuildHistory(config.db_path)
    rows = history.list_runs(args.limit)
    if not rows:
        print("No runs recorded yet.")
        return 0
    print(f"{'run_id':40s} {'commit':10s} {'status':8s} {'duration':>9s}  started_at")
    for run_id, project, sha, branch, started, status, duration in rows:
        dur = f"{duration:.1f}s" if duration else "-"
        print(f"{run_id:40s} {sha[:8]:10s} {status:8s} {dur:>9s}  {started}")
    return 0


def cmd_show(args):
    config = Config(Path(args.config))
    history = BuildHistory(config.db_path)
    run, stages = history.get_run(args.run_id)
    if not run:
        print(f"No such run: {args.run_id}", file=sys.stderr)
        return 1
    cols = ["run_id", "project", "commit_sha", "branch", "started_at", "finished_at", "status", "duration_s"]
    print(json.dumps(dict(zip(cols, run)), indent=2))
    print("\nStages:")
    for stage_name, status, exit_code, duration, log_path in stages:
        print(f"  - {stage_name:15s} {status:8s} exit={exit_code} duration={duration} log={log_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Mini local CI/CD pipeline orchestrator")
    parser.add_argument("--config", default=str(Path(__file__).parent / "pipeline.yml"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Execute the pipeline for the latest (or a specific) commit")
    p_run.add_argument("--sha", default=None, help="Specific commit SHA to build (default: tip of branch)")
    p_run.set_defaults(func=cmd_run)

    p_hist = sub.add_parser("history", help="List recent pipeline runs")
    p_hist.add_argument("--limit", type=int, default=20)
    p_hist.set_defaults(func=cmd_history)

    p_show = sub.add_parser("show", help="Show details for a single run")
    p_show.add_argument("run_id")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
