@echo off
echo ==========================================
echo GeoGreen Revolution - Setup Script
echo ==========================================

set PYTHON_EXE=C:\Users\RAHUL\anaconda3\python.exe

:: Check if the specific Python exists
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found at %PYTHON_EXE%
    echo Please edit this script to point to your python.exe
    pause
    exit /b 1
)

echo [INFO] Found Python at: %PYTHON_EXE%
echo [INFO] Creating virtual environment...
"%PYTHON_EXE%" -m venv venv

echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Installing requirements...
pip install -r requirements.txt
pip install streamlit easyocr

echo [INFO] Setup complete!
echo ==========================================
echo To run the dashboard:
echo 1. Double-click setup.bat (you just did this!)
echo 2. Type: venv\Scripts\activate
echo 3. Type: streamlit run app.py
echo ==========================================
cmd /k "venv\Scripts\activate"
