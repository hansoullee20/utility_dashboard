@echo off
title Utility Analysis Dashboard
cd /d "%~dp0"

echo ============================================
echo  Utility Analysis Dashboard
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python을 찾을 수 없습니다.
    echo Python을 설치한 후 다시 실행하세요: https://www.python.org
    pause
    exit /b 1
)

:: Install / update packages (only if needed)
echo 패키지 확인 중...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 패키지 설치 중 (최초 1회)...
    pip install -r requirements_client.txt --quiet
    if errorlevel 1 (
        echo [ERROR] 패키지 설치 실패. 인터넷 연결을 확인하세요.
        pause
        exit /b 1
    )
    echo 설치 완료.
    echo.
)

echo 대시보드 시작 중...
echo 브라우저가 자동으로 열립니다.
echo 종료하려면 이 창을 닫으세요.
echo.

streamlit run app.py --server.headless false --browser.gatherUsageStats false
