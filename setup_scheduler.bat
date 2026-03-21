@REM ========================================================================
@REM Automated Task Scheduler Setup Script for Daily Optimizer
@REM 
@REM This script automatically creates a Windows Task Scheduler entry
@REM that runs the optimizer daily at 2:00 AM
@REM
@REM USAGE: 
@REM   Right-click this file → "Run as administrator" → Press Enter
@REM
@REM ========================================================================

@echo off
setlocal enabledelayedexpansion

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] This script must be run as Administrator!
    echo.
    echo Please:
    echo   1. Right-click this file (setup_scheduler.bat)
    echo   2. Select "Run as administrator"
    echo   3. Press Enter when prompted
    pause
    exit /b 1
)

echo ========================================================================
echo SCENARIO PLANNER - AUTOMATED SCHEDULER SETUP
echo ========================================================================
echo.

REM Define paths
set PROJECT_DIR=C:\Users\liagk\Project\scenario-planner
set SCRIPT_PATH=%PROJECT_DIR%\run_optimizer_daily.bat
set TASK_NAME=Scenario-Planner-Daily-Optimizer

echo [INFO] Creating scheduled task...
echo        Task: %TASK_NAME%
echo        Script: %SCRIPT_PATH%
echo        Time: 02:00 AM (daily)
echo.

REM Check if script exists
if not exist "%SCRIPT_PATH%" (
    echo [ERROR] Script not found: %SCRIPT_PATH%
    pause
    exit /b 1
)

REM Delete existing task if present
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

REM Create the scheduled task
schtasks /create /tn "%TASK_NAME%" /tr "%SCRIPT_PATH%" /sc daily /st 02:00:00 /rl highest /f

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Scheduled task created!
    echo.
    echo ========================================================================
    echo NEXT STEPS:
    echo ========================================================================
    echo.
    echo 1. Task is now scheduled to run DAILY at 02:00 AM (2:00 AM)
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
    echo Please ensure you are running as Administrator.
    pause
    exit /b 1
)

:end
echo.
pause
