@REM ========================================================================
@REM Automated Task Scheduler Setup Script for Daily Optimizer
@REM 
@REM This script automatically creates a Windows Task Scheduler entry
@REM that runs the optimizer every 2 days at 3:00 PM (15:00)
@REM
@REM USAGE: 
@REM   Double-click this file (no administrator needed)
@REM
@REM ========================================================================

@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo SCENARIO PLANNER - AUTOMATED SCHEDULER SETUP
echo ========================================================================
echo.

REM Define paths
set PROJECT_DIR=C:\Users\liagk\Project\scenario-planner
set SCRIPT_PATH=%PROJECT_DIR%\run_optimizer_daily.bat
set TASK_NAME=Scenario-Planner-Daily-Optimizer-User

echo [INFO] Creating scheduled task...
echo        Task: %TASK_NAME%
echo        Script: %SCRIPT_PATH%
echo        Time: 15:00 (3:00 PM, every 2 days)
echo.

REM Check if script exists
if not exist "%SCRIPT_PATH%" (
    echo [ERROR] Script not found: %SCRIPT_PATH%
    pause
    exit /b 1
)

REM Delete existing task if present
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

REM Create the scheduled task (every 2 days at 15:00) in current user context
schtasks /create /tn "%TASK_NAME%" /tr "%SCRIPT_PATH%" /sc daily /mo 2 /st 15:00:00 /rl LIMITED /f

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Scheduled task created!
    echo.
    echo ========================================================================
    echo NEXT STEPS:
    echo ========================================================================
    echo.
    echo 1. Task is now scheduled to run EVERY 2 DAYS at 15:00 (3:00 PM)
    echo.
    echo 2. When the optimizer finds serious problems, it creates:
    echo    output\default\.optimizer\approval_queue.json
    echo.
    echo 3. You need to APPROVE or REJECT the changes by editing that file:
    echo    - Change: "status": "pending"
    echo    - To: "status": "YES" or "status": "NO"
    echo    - Save the file
    echo    - Scheduler detects the change in 2 seconds and continues
    echo.
    echo 4. Monitor logs in: logs\scheduler.log
    echo.
    echo 5. View results in: output\default\optimizer_report.json
    echo.
    echo ========================================================================
    echo MANUAL EXECUTION (without waiting for 2:00 AM):
    echo ========================================================================
    echo.
    echo Run in PowerShell:
    echo   cd C:\Users\liagk\Project\scenario-planner
    echo   python src\scheduler.py --once
    echo.
    echo ========================================================================
    goto end
) else (
    echo.
    echo [ERROR] Failed to create scheduled task!
    echo If a task with the same name exists under another account,
    echo create it manually in Task Scheduler with this name:
    echo   %TASK_NAME%
    pause
    exit /b 1
)

:end
echo.
pause
