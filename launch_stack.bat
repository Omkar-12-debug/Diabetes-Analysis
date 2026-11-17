@echo off
echo =====================================================================
echo           DIABETES RISK MLOPS PLATFORM - 1-CLICK LAUNCHER
echo =====================================================================
echo.

:: Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/3] Starting Docker Desktop...
    start "" "C:\Users\parth\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe"
    echo [INFO] Waiting for Docker engine to initialize...
    :wait_loop
    timeout /t 3 /nobreak >nul
    docker info >nul 2>&1
    if %errorlevel% neq 0 (
        echo ...waiting for Docker daemon
        goto wait_loop
    )
    echo [OK] Docker Desktop is ready!
) else (
    echo [1/3] Docker daemon is already running!
)

echo.
echo [2/3] Launching multi-service containers in detached mode...
docker compose up -d

echo.
echo [3/3] Verifying container health and forwarded ports...
docker compose ps

echo.
echo =====================================================================
echo                  STACK READY FOR PRESENTATION
echo =====================================================================
echo   1. React Clinical Dashboard:  http://localhost:5173
echo   2. Grafana Telemetry Panels:  http://localhost:3000  (admin / admin)
echo   3. Prometheus Target Metrics: http://localhost:9090/targets
echo   4. FastAPI Swagger Docs:      http://localhost:8000/docs
echo   5. MLflow Model Registry:     http://localhost:5000
echo =====================================================================
echo.

:: Open presentation browser tabs
set /p open_browser="Open dashboards in browser now? (Y/N, default Y): "
if /i "%open_browser%"=="N" goto end

start http://localhost:5173
start http://localhost:3000
start http://localhost:5000
start http://localhost:8000/docs

:end
echo.
echo To shut down the stack later, run stop_stack.bat or 'docker compose stop'
echo.
pause
