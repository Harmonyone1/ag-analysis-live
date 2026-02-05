@echo off
REM AG Analyzer - Stop Script for Windows

echo ===================================
echo   AG Analyzer - Stopping Services
echo ===================================

cd /d "%~dp0\.."

docker-compose down

echo.
echo Services stopped.
pause
