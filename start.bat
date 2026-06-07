@echo off
REM Inversa — double-click to set up (first run) and launch the GUI. No prior setup needed.
cd /d "%~dp0"
where py >nul 2>nul && (py run.py) || (python run.py)
echo.
echo (window kept open so you can read any messages; close it to stop the server)
pause
