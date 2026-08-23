#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from orchestrator import Config, resolve_commit_sha, git


def load_state(state_file: Path):
    if state_file.exists():
        return json.loads(state_file.read_text(encoding="utf-8"))
    return {"last_sha": None}


def save_state(state_file: Path, state: dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def poll_once(config: Config) -> bool:
    subprocess.run(["git", "fetch", "--all", "--quiet"], cwd=config.repo_path, capture_output=True)

    latest_sha = resolve_commit_sha(config.repo_path, config.branch)
    state = load_state(config.state_file)

    if state.get("last_sha") == latest_sha:
        print(f"[watcher] no new commits on '{config.branch}' (still {latest_sha[:8]})")
        return False

    print(f"[watcher] new commit detected on '{config.branch}': {latest_sha[:8]} -> triggering pipeline")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "orchestrator.py"),
         "--config", str(config.config_path), "run", "--sha", latest_sha],
    )
    state["last_sha"] = latest_sha
    save_state(config.state_file, state)
    print(f"[watcher] pipeline exited with code {result.returncode}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Poll a git repo for new commits and trigger the CI/CD pipeline")
    parser.add_argument("--config", default=str(Path(__file__).parent / "pipeline.yml"))
    parser.add_argument("--once", action="store_true", help="Poll a single time and exit (useful for cron/testing)")
    args = parser.parse_args()

    config = Config(Path(args.config))
    print(f"[watcher] watching '{config.repo_path}' branch '{config.branch}' "
          f"every {config.poll_interval}s (state: {config.state_file})")

    if args.once:
        poll_once(config)
        return

    while True:
        try:
            poll_once(config)
        except Exception as exc:
            print(f"[watcher] error during poll: {exc}", file=sys.stderr)
        time.sleep(config.poll_interval)


if __name__ == "__main__":
    main()
