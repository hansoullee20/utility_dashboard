@echo off
setlocal
echo ============================================
echo   Utility Dashboard - Windows Build Script
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org and check "Add to PATH".
    pause
    exit /b 1
)
echo [1/4] Python found:
python --version

:: Create virtual environment
echo.
echo [2/4] Creating build environment...
if exist build_env (
    echo   build_env already exists, reusing.
) else (
    python -m venv build_env
)

:: Install dependencies
echo.
echo [3/4] Installing dependencies (this may take a few minutes)...
call build_env\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install --quiet ^
    streamlit ^
    pandas ^
    numpy ^
    plotly ^
    scipy ^
    reportlab ^
    matplotlib ^
    Pillow ^
    openpyxl ^
    pyarrow ^
    altair ^
    pyinstaller

:: Build
echo.
echo [4/4] Building UtilityDashboard.exe...
pyinstaller launcher.spec --clean --noconfirm

echo.
if exist dist\UtilityDashboard.exe (
    echo ============================================
    echo   SUCCESS! Your exe is ready:
    echo   dist\UtilityDashboard.exe
    echo ============================================
) else (
    echo   BUILD FAILED. Check errors above.
)

pause
