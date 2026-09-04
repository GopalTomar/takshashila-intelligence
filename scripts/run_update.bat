@echo off
REM ============================================================================
REM  run_update.bat — one incremental knowledge-base update (website + Commit KB)
REM
REM  Called by Windows Task Scheduler every Monday 18:00 (see
REM  setup_windows_task.ps1). Can also be double-clicked to run on demand.
REM
REM  It:
REM    1. moves to the project root (this file lives in \scripts),
REM    2. activates the .venv virtual environment if present,
REM    3. runs one incremental update and reindex,
REM    4. appends all output to data\logs\weekly_update.log.
REM ============================================================================

setlocal
REM Project root = parent folder of this script's folder.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."
set "PROJECT_ROOT=%CD%"

REM Ensure the log folder exists.
if not exist "%PROJECT_ROOT%\data\logs" mkdir "%PROJECT_ROOT%\data\logs"
set "LOG=%PROJECT_ROOT%\data\logs\weekly_update.log"

echo. >> "%LOG%"
echo ==================================================================== >> "%LOG%"
echo [%DATE% %TIME%] Starting weekly knowledge-base update >> "%LOG%"

REM Prefer the project virtualenv's python; fall back to system python.
set "PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" "%PROJECT_ROOT%\scripts\update_knowledge_base.py" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

echo [%DATE% %TIME%] Finished with exit code %RC% >> "%LOG%"
popd
endlocal & exit /b %RC%