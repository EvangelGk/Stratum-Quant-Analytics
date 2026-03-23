@REM ========================================================================
@REM Automated Task Scheduler Setup Script for Daily Optimizer
@REM ========================================================================

@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo STRATUM QUANT ANALYTICS - AUTOMATED SCHEDULER SETUP
echo ========================================================================
echo.

for %%I in ("%~dp0.") do set "PROJECT_DIR=%%~fI"
set "SCRIPT_PATH=%PROJECT_DIR%\run_optimizer_daily.bat"
set "TASK_NAME=Stratum-Quant-Analytics-Daily-Optimizer-User"
set "LEGACY_TASK_NAME=Scenario-Planner-Daily-Optimizer-User"

echo [INFO] Creating scheduled task...
echo        Task: %TASK_NAME%
echo        Script: %SCRIPT_PATH%
echo        Time: 15:00 (3:00 PM, every 2 days)
echo.

if not exist "%SCRIPT_PATH%" (
    echo [ERROR] Script not found: %SCRIPT_PATH%
    pause
    exit /b 1
)

schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1
schtasks /delete /tn "%LEGACY_TASK_NAME%" /f >nul 2>&1
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
    echo 2. Queue file: output\default\.optimizer\approval_queue.json
    echo 3. You can approve from:
    echo    - the web owner panel,
    echo    - python respond_to_approval.py --user-id default --approve,
    echo    - or by editing the queue file to YES / NO.
    echo 4. Logs: logs\scheduler.log
    echo 5. Report: output\default\optimizer_report.json
    echo.
    echo Manual execution from this project folder:
    echo   python src\scheduler.py --once --user-id default
    echo.
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


