#!/usr/bin/env python3
import argparse
from pathlib import Path

from flask import Flask, Response, abort, render_template

try:
    from .orchestrator import BuildHistory, Config
except ImportError:
    from orchestrator import BuildHistory, Config


RUN_COLUMNS = [
    "run_id",
    "project",
    "commit_sha",
    "branch",
    "started_at",
    "status",
    "duration_s",
]


def create_app(config_path: str | Path | None = None) -> Flask:
    config_file = Path(config_path) if config_path else Path(__file__).with_name("pipeline.yml")
    config = Config(config_file)
    app = Flask(__name__, template_folder=str(Path(__file__).with_name("dashboard_templates")))
    app.config["PIPELINE_CONFIG"] = config

    @app.get("/")
    def index():
        history = BuildHistory(config.db_path)
        try:
            runs = [_serialize_run(config, history, row) for row in history.list_runs(limit=50)]
        finally:
            history.conn.close()
        summary = {
            "total": len(runs),
            "passed": sum(run["status"] == "passed" for run in runs),
            "failed": sum(run["status"] in {"failed", "error"} for run in runs),
            "running": sum(run["status"] == "running" for run in runs),
        }
        return render_template(
            "dashboard.html",
            project_name=config.project_name,
            branch=config.branch,
            stages=[stage["name"] for stage in config.stages],
            runs=runs,
            summary=summary,
        )

    @app.get("/runs/<run_id>/logs/<stage_name>")
    def stage_log(run_id: str, stage_name: str):
        history = BuildHistory(config.db_path)
        try:
            _, stages = history.get_run(run_id)
        finally:
            history.conn.close()
        stage = next((item for item in stages if item[0] == stage_name), None)
        if stage is None:
            abort(404)
        log_path = _safe_log_path(config, run_id, stage_name, stage[4])
        if not log_path.is_file():
            abort(404)
        return Response(log_path.read_text(encoding="utf-8", errors="replace"), mimetype="text/plain")

    return app


def _serialize_run(config: Config, history: BuildHistory, row: tuple) -> dict:
    run = dict(zip(RUN_COLUMNS, row))
    _, stages = history.get_run(run["run_id"])
    stage_map = {
        name: {
            "status": status,
            "exit_code": exit_code,
            "duration_s": duration_s,
            "has_log": _safe_log_path(config, run["run_id"], name, log_path).is_file(),
        }
        for name, status, exit_code, duration_s, log_path in stages
    }
    docker_stage = stage_map.get("docker_build")
    run["stages"] = stage_map
    run["image_tag"] = (
        f"{_docker_image_name(config)}:{run['commit_sha'][:8]}"
        if docker_stage and docker_stage["status"] == "passed"
        else None
    )
    return run


def _docker_image_name(config: Config) -> str:
    docker_stage = next((stage for stage in config.stages if stage.get("type") == "docker"), {})
    return docker_stage.get("image_name", "app")


def _safe_log_path(config: Config, run_id: str, stage_name: str, stored_path: str | None) -> Path:
    expected = (config.logs_dir / "runs" / run_id / f"{stage_name}.log").resolve()
    logs_root = (config.logs_dir / "runs").resolve()
    if not expected.is_relative_to(logs_root):
        return Path()
    if stored_path and Path(stored_path).resolve() != expected:
        return Path()
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Mini CI/CD dashboard")
    parser.add_argument("--config", default=str(Path(__file__).with_name("pipeline.yml")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    create_app(args.config).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()