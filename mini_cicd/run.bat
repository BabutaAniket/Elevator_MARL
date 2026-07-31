@echo off
REM Convenience launcher for Windows: run the pipeline once against the
REM current tip of the configured branch.
setlocal
cd /d "%~dp0"
python orchestrator.py --config pipeline.yml run %*
