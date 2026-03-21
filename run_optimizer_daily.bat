@REM ========================================================================
@REM Windows Task Scheduler Script for Daily Optimizer
@REM 
@REM Installation:
@REM   1. Open Task Scheduler (taskschd.msc)
@REM   2. Create Basic Task
@REM   3. Set trigger to daily (e.g., 02:00 AM)
@REM   4. Set action: Start a program
@REM   5. Program: C:\Users\liagk\Project\scenario-planner\run_optimizer_daily.bat
@REM   6. Advanced: Check "Run whether user is logged in or not"
@REM
@REM ========================================================================

@echo off
setlocal enabledelayedexpansion

REM Set paths
set PROJECT_DIR=C:\Users\liagk\Project\scenario-planner
set PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe
set SCHEDULER_SCRIPT=%PROJECT_DIR%\src\scheduler.py
set LOG_DIR=%PROJECT_DIR%\logs
set USER_ID=default

REM Create log directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Timestamp for log file
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a-%%b)
set TIMESTAMP=%mydate%_%mytime%

REM Run the optimizer once (non-interactive, file-based approval)
echo [%TIMESTAMP%] Starting daily optimizer run >> "%LOG_DIR%\scheduler.log"
"%PYTHON_EXE%" "%SCHEDULER_SCRIPT%" --once --user-id "%USER_ID%" >> "%LOG_DIR%\scheduler.log" 2>&1

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo [%TIMESTAMP%] Optimizer run completed successfully >> "%LOG_DIR%\scheduler.log"
) else (
    echo [%TIMESTAMP%] Optimizer run FAILED with exit code %ERRORLEVEL% >> "%LOG_DIR%\scheduler.log"
)

endlocal
