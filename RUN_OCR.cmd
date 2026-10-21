@echo off
setlocal
set "PYTHONPATH="
set "ROOT=%~dp0"
if "%~1"=="" (
  echo Drag an image or an image folder onto this file.
  pause
  exit /b 2
)
if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo Run SETUP.cmd first.
  pause
  exit /b 2
)
"%ROOT%.venv\Scripts\python.exe" -E "%ROOT%ocr.py" --input "%~1" --output "%ROOT%results" --models "%ROOT%models" --mode accurate
set "CODE=%ERRORLEVEL%"
echo.
echo OCR finished with exit code %CODE%. See results folder.
echo OCR text is unverified. Do not post financial entries without review.
pause
exit /b %CODE%
