@echo off
REM AG Analyzer - Development Setup Script for Windows

echo ===================================
echo   AG Analyzer - Dev Environment
echo ===================================

cd /d "%~dp0\.."

REM Check Python version
python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed. Please install Python 3.11+
    pause
    exit /b 1
)

REM Check Node version
node --version > nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed. Please install Node.js 20+
    pause
    exit /b 1
)

echo.
echo Setting up Python virtual environment for Engine...
cd engine
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio
deactivate
cd ..

echo.
echo Setting up Python virtual environment for Backend...
cd backend
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt
deactivate
cd ..

echo.
echo Setting up Node.js dependencies for UI...
cd ui
call npm install
cd ..

echo.
echo ===================================
echo   Development Setup Complete!
echo ===================================
echo.
echo To start development:
echo.
echo 1. Start PostgreSQL (Docker or local):
echo    docker-compose up -d db
echo.
echo 2. Start Engine:
echo    cd engine ^&^& venv\Scripts\activate ^&^& python src\main.py
echo.
echo 3. Start Backend:
echo    cd backend ^&^& venv\Scripts\activate ^&^& uvicorn src.main:app --reload
echo.
echo 4. Start UI:
echo    cd ui ^&^& npm run dev
echo.
pause
