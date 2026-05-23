@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Run setup_windows_jupyter.bat first.
  pause
  exit /b 1
)

call ".venv\Scripts\activate"
python -m notebook
