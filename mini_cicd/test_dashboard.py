import tempfile
import unittest
from pathlib import Path

from mini_cicd.dashboard import create_app
from mini_cicd.orchestrator import BuildHistory


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "target").mkdir()
        self.config_path = self.root / "pipeline.yml"
        self.config_path.write_text(
            """
project:
  name: demo-project
  repo_path: target
  branch: main
history:
  db_path: history/build_history.db
  logs_dir: logs
stages:
  - name: build
    command: echo build
  - name: unit_tests
    command: echo tests
  - name: docker_build
    type: docker
    image_name: demo-image
""".strip(),
            encoding="utf-8",
        )
        self.run_id = "run-001"
        self.log_path = self.root / "logs" / "runs" / self.run_id / "unit_tests.log"
        self.log_path.parent.mkdir(parents=True)
        self.log_path.write_text("test output\n", encoding="utf-8")
        history = BuildHistory(self.root / "history" / "build_history.db")
        started_at = "2026-08-23T10:00:00+00:00"
        finished_at = "2026-08-23T10:00:03+00:00"
        history.start_run(self.run_id, "demo-project", "abcdef1234567890", "main")
        history.record_stage(
            self.run_id,
            "unit_tests",
            "passed",
            0,
            started_at,
            finished_at,
            self.log_path,
        )
        history.record_stage(
            self.run_id,
            "docker_build",
            "passed",
            0,
            started_at,
            finished_at,
            self.root / "logs" / "runs" / self.run_id / "docker_build.log",
        )
        history.finish_run(self.run_id, "passed", started_at)
        history.conn.close()
        self.app = create_app(self.config_path)
        self.app.testing = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dashboard_renders_run_history_and_image_tag(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"abcdef12", response.data)
        self.assertIn(b"demo-image:abcdef12", response.data)
        self.assertIn(b"unit_tests", response.data)

    def test_stage_log_is_available_only_for_recorded_stage(self):
        response = self.client.get(f"/runs/{self.run_id}/logs/unit_tests")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"test output\n")
        self.assertEqual(self.client.get(f"/runs/{self.run_id}/logs/unknown").status_code, 404)