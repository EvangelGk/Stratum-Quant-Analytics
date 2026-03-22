@REM ========================================================================
@REM Windows Task Scheduler Script for Daily Optimizer
@REM 
@REM Installation:
@REM   1. Open Task Scheduler (taskschd.msc)
@REM   2. Create Basic Task
@REM   3. Set trigger to every 2 days at 15:00 (or your preferred time)
@REM   4. Set action: Start a program
@REM   5. Program: C:\Users\liagk\Project\Stratum-Quant-Analytics\run_optimizer_daily.bat
@REM   6. Advanced: Check "Run whether user is logged in or not"
@REM
@REM ========================================================================

@echo off
setlocal enabledelayedexpansion

REM Set paths
set PROJECT_DIR=C:\Users\liagk\Project\Stratum-Quant-Analytics
set PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe
set SCHEDULER_SCRIPT=%PROJECT_DIR%\src\scheduler.py
set LOG_DIR=%PROJECT_DIR%\logs
set USER_ID=default

REM =========================================================================
REM  Credentials are loaded automatically from .env (project root).
REM  Edit .env and set:
REM    TELEGRAM_BOT_TOKEN=<your bot token from @BotFather>
REM    TELEGRAM_CHAT_ID=<your chat id>
REM  The Python code calls load_dotenv() at startup so no manual set needed.
REM  See .env.example for the full list of available variables.
REM =========================================================================

REM Load .env into this batch session so env vars are also available
REM to any subprocesses that do NOT use python-dotenv.
if exist "%PROJECT_DIR%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /v "^#" "%PROJECT_DIR%\.env" ^| findstr /v "^$"`) do (
        set "%%A=%%B"
    )
)

REM Create log directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Timestamp for log file
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a-%%b)
set TIMESTAMP=%mydate%_%mytime%

REM Run the optimizer once (scheduled execution, file-based approval)
echo [%TIMESTAMP%] Starting scheduled optimizer run >> "%LOG_DIR%\scheduler.log"
"%PYTHON_EXE%" "%SCHEDULER_SCRIPT%" --once --user-id "%USER_ID%" >> "%LOG_DIR%\scheduler.log" 2>&1

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo [%TIMESTAMP%] Optimizer run completed successfully >> "%LOG_DIR%\scheduler.log"
) else (
    echo [%TIMESTAMP%] Optimizer run FAILED with exit code %ERRORLEVEL% >> "%LOG_DIR%\scheduler.log"
)

endlocal


