@echo off
REM Starts the quick-predict research GUI on this computer only, then opens it in the default browser.
cd /d "%~dp0.."
start "" http://127.0.0.1:5090
python quick_predict\app.py
pause
