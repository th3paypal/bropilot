@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  echo [bropilot] Первый запуск: создаю окружение и ставлю зависимости, это займёт несколько минут...
  py -3 -m venv .venv 2>nul || python -m venv .venv
  if errorlevel 1 (
    echo Не найден Python 3.10+. Установите его с python.org, отметив "Add python.exe to PATH".
    pause
    exit /b 1
  )
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -e .
  if errorlevel 1 (
    echo Установка зависимостей не удалась, см. сообщения выше.
    pause
    exit /b 1
  )
) else (
  call .venv\Scripts\activate.bat
)
echo [bropilot] Открываю интерфейс в браузере. Чтобы остановить, закройте это окно.
rem `bropilot serve` сам поднимает Streamlit и открывает вкладку.
bropilot serve %*
pause
