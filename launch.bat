@echo off
echo ===================================================
echo      NeuroAI Workstation Launcher
echo ===================================================
echo.

set WORKSPACE_DIR=%~dp0
cd /d "%WORKSPACE_DIR%"

:: Check if .venv directory exists
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment in .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is installed and in your PATH.
        pause
        exit /b 1
    )
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Check if flask is installed in venv to determine if we need offline install
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required dependencies offline from local dependencies folder...
    python -m pip install --no-index --find-links=dependencies -r Configs/requirements.txt
    if errorlevel 1 (
        echo [ERROR] Offline dependency installation failed.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Dependencies are already satisfied.
)

echo.
echo [SUCCESS] Workstation environment is ready!
echo [INFO] Instant Preview Link: http://127.0.0.1:5000
echo [INFO] Launching default web browser...
start http://127.0.0.1:5000

echo.
echo Starting Flask Backend Server...
echo (Press Ctrl+C to stop the server at any time)
echo ---------------------------------------------------
python Backend/app.py
pause
