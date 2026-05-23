@echo off
cd /d "%~dp0"

echo Paste your Gemini API key from https://aistudio.google.com/apikey
echo The key will be saved to .env in this project folder.
echo.
set /p GEMINI_KEY=Gemini API key: 

if "%GEMINI_KEY%"=="" (
  echo No key entered. Nothing changed.
  pause
  exit /b 1
)

> ".env" echo # FraudRecruitment environment configuration
>> ".env" echo # Keep this file private. It is ignored by .gitignore.
>> ".env" echo GEMINI_API_KEY=%GEMINI_KEY%
>> ".env" echo GEMINI_MODEL_NAME=gemini-2.5-flash

echo.
echo Gemini key saved. Restart the Flask app for the change to take effect.
pause
