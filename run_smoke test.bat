@echo off
setlocal ENABLEDELAYEDEXPANSION
chcp 65001 >nul

rem ========================================
rem Sound Realty Service - one-click runner
rem ========================================

set "REPO_DIR=%~dp0"
set "PORT=8080"
set "HOST=127.0.0.1"
set "URL=http://%HOST%:%PORT%"
set "LOGFILE=%REPO_DIR%\run_smoke_test.log"

rem Helper: log and echo
set "NL="
for /f "delims=" %%a in ('powershell -NoProfile -Command "[Environment]::NewLine"') do set "NL=%%a"
call :log "========== Run started %DATE% %TIME% =========="

rem Use repo root if the .bat is placed there; otherwise try to detect
cd /d "%REPO_DIR%" || (call :die "Cannot cd to %REPO_DIR%")

if exist "sound_realty_service" (
    cd /d "sound_realty_service" || (call :die "Cannot cd into sound_realty_service")
)

if not exist "requirements.txt" (
    call :die "requirements.txt not found in %CD%"
)

rem 1) Create & locate venv
if not exist ".venv" (
    call :log "Creating virtual environment..."
    py -m venv .venv || (call :die "Failed to create virtual environment")
)
set "VENV_DIR=%CD%\.venv"
set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"
if not exist "%PY_EXE%" (call :die "Virtual environment python not found at %PY_EXE%")

rem 2) Install deps
call :log "Upgrading pip..."
"%PY_EXE%" -m pip install --upgrade pip >>"%LOGFILE%" 2>&1 || call :log "WARN: pip upgrade failed (continuing)"
call :log "Installing dependencies from requirements.txt..."
"%PIP_EXE%" install -r requirements.txt >>"%LOGFILE%" 2>&1 || (call :die "Dependency install failed. See log.")

rem 3) Start API server in new window
call :log "Starting API server at %URL%"
set "UVICORN_CMD=%PY_EXE% -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%"
start "API Server - Sound Realty" cmd /k "%UVICORN_CMD%" || (call :die "Failed to start API server window")

rem 4) Wait for /healthz
call :log "Waiting for server readiness (/healthz)..."
set "MAX_TRIES=30"
set "TRIED=0"

:wait_loop
set /a TRIED+=1
rem Try curl; if missing, try PowerShell Invoke-WebRequest
curl --silent --fail "%URL%/healthz" >nul 2>nul
if not errorlevel 1 goto server_ready

powershell -NoProfile -Command "try{ iwr -UseBasicParsing -TimeoutSec 1 '%URL%/healthz' | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 goto server_ready

if %TRIED% GEQ %MAX_TRIES% (
    call :die "Server did not become ready after %MAX_TRIES% seconds. Check the other window for errors."
)

timeout /t 1 >nul
goto :wait_loop

:server_ready
call :log "Server is ready."

rem 5) Run client (k=3)
call :log "Running client: scripts\test_client.py --url %URL% -k 3"
"%PY_EXE%" scripts\test_client.py --url "%URL%" -k 3 >>"%LOGFILE%" 2>&1 || call :log "WARN: client request failed"

echo.
echo [INFO] Client run complete. Open docs at: %URL%/docs
echo [INFO] The API server is running in the window titled: "API Server - Sound Realty"
echo [INFO] Log file: "%LOGFILE%"

goto :final

:die
rem %* = message
echo [ERROR] %~1
echo [ERROR] See log: "%LOGFILE%"
call :log "ERROR: %~1"
goto :final

:log
rem %~1 = message
>>"%LOGFILE%" echo [%DATE% %TIME%] %~1
echo %~1
exit /b 0

:final
echo.
echo Press any key to close this window...
pause >nul
call :log "========== Run ended %DATE% %TIME% =========="
endlocal
exit /b