@REM ========================================================================
@REM Windows Task Scheduler Script for Daily Optimizer
@REM ========================================================================

@echo off
setlocal enabledelayedexpansion

REM Resolve project directory from the location of this .bat file.
for %%I in ("%~dp0.") do set "PROJECT_DIR=%%~fI"
set "PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCHEDULER_SCRIPT=%PROJECT_DIR%\src\scheduler.py"
set "LOG_DIR=%PROJECT_DIR%\logs"
set "USER_ID=default"

REM Load .env into this batch session so env vars are also available
REM to any subprocesses that do NOT use python-dotenv.
if exist "%PROJECT_DIR%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /v "^#" "%PROJECT_DIR%\.env" ^| findstr /v "^$"`) do (
        set "%%A=%%B"
    )
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found: "%PYTHON_EXE%" >> "%LOG_DIR%\scheduler.log"
    exit /b 1
)

for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a-%%b)
set "TIMESTAMP=%mydate%_%mytime%"

echo [%TIMESTAMP%] Starting scheduled optimizer run from "%PROJECT_DIR%" >> "%LOG_DIR%\scheduler.log"
"%PYTHON_EXE%" "%SCHEDULER_SCRIPT%" --once --user-id "%USER_ID%" >> "%LOG_DIR%\scheduler.log" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [%TIMESTAMP%] Optimizer run completed successfully >> "%LOG_DIR%\scheduler.log"
) else (
    echo [%TIMESTAMP%] Optimizer run FAILED with exit code %ERRORLEVEL% >> "%LOG_DIR%\scheduler.log"
)

endlocal


