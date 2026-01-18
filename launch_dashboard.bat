@echo off
echo Starting European Market Dashboard...
echo.

cd /d "%~dp0"

:: Start Streamlit in background and open browser
start "" streamlit run market_dashboard.py --server.headless true

:: Wait a moment for server to start, then open browser
timeout /t 3 /nobreak >nul
start "" http://localhost:8501

echo Dashboard is running at http://localhost:8501
echo Press Ctrl+C in the Streamlit window to stop.
