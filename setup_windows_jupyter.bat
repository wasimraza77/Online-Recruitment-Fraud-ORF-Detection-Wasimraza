@echo off
cd /d "%~dp0"

py -3.9 -m venv .venv
call ".venv\Scripts\activate"
python -m pip install --upgrade pip
pip install -r requirements-windows-jupyter.txt
python -m ipykernel install --user --name fraudrecruitment --display-name "Python (.venv FraudRecruitment)"

echo.
echo Setup complete.
echo Next:
echo 1. Run run_jupyter_windows.bat
echo 2. Open FraudJobDetection.ipynb
echo 3. Select kernel: Python (.venv FraudRecruitment)
pause
