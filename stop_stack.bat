@echo off
echo =====================================================================
echo           DIABETES RISK MLOPS PLATFORM - GRACEFUL SHUTDOWN
echo =====================================================================
echo.
echo Stopping all containers (preserves database records and model files)...
docker compose stop
echo.
docker compose ps
echo.
echo All containers stopped safely.
pause

