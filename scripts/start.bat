@echo off
REM AG Analyzer - Start Script for Windows

echo ===================================
echo   AG Analyzer - Starting Services
echo ===================================

REM Check if Docker is running
docker info > nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Navigate to project root
cd /d "%~dp0\.."

REM Check for .env file
if not exist .env (
    echo Warning: .env file not found. Copying from .env.example...
    copy .env.example .env
    echo Please edit .env with your TradeLocker credentials before running again.
    pause
    exit /b 1
)

REM Create models directory if it doesn't exist
if not exist models mkdir models

REM Build and start services
echo.
echo Building Docker images...
docker-compose build

echo.
echo Starting services...
docker-compose up -d

REM Wait for services to be ready
echo.
echo Waiting for services to start...
timeout /t 5 /nobreak > nul

REM Check service status
echo.
echo Service Status:
docker-compose ps

echo.
echo ===================================
echo   Services Started Successfully!
echo ===================================
echo.
echo Access points:
echo   - UI:       http://localhost:3000
echo   - API:      http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo.
echo To view logs: docker-compose logs -f
echo To stop:      docker-compose down
echo.
pause
