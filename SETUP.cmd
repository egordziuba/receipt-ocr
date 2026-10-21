@echo off
setlocal
set "PYTHONPATH="
set "ROOT=%~dp0"
set "UV=uv"
"%UV%" --version
if errorlevel 1 goto fail
if not exist "%ROOT%.venv\Scripts\python.exe" (
  "%UV%" venv --python 3.11 "%ROOT%.venv"
  if errorlevel 1 goto fail
)
"%UV%" pip install --python "%ROOT%.venv\Scripts\python.exe" -r "%ROOT%requirements.txt"
if errorlevel 1 goto fail
echo Setup complete. Tesseract must also be installed separately.
pause
exit /b 0
:fail
echo Setup failed. Install uv or review the error above.
pause
exit /b 1
