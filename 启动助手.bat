@echo off
setlocal enableextensions enabledelayedexpansion
rem ==========================================================================
rem  竖屏视频自动上滑助手 - 启动器
rem
rem  重要:本文件必须保存为 ANSI/GBK 编码 + Windows(CRLF)换行。
rem        用 UTF-8 或只有 LF 换行时,cmd 会读错中文字符位置,
rem        报 "'xxx' 不是内部或外部命令" 之类莫名其妙的错误。
rem  用法:在本文件后面附加参数(如 启动助手.bat --dry-run),走控制台显示日志。
rem ==========================================================================
cd /d "%~dp0"
set "APP=%~dp0vertical_swiper.py"
set "PY="
set "PYW="
set "EXE="
set "VENV=%~dp0venv\Scripts"
set "CONDA_PY=C:\Users\Chen\.conda\envs\Scroller\python.exe"
set "CONDA_PYW=C:\Users\Chen\.conda\envs\Scroller\pythonw.exe"

if exist "%APP%" goto :check_venv
echo.
echo 找不到 vertical_swiper.py,请确认它和本文件在同一个文件夹里。
echo.
pause
exit /b 1

:check_venv
rem ---- 0) 首选 conda 环境 Scroller(本机运行环境) ----
if not exist "%CONDA_PY%" goto :try_venv
if not "%~1"=="" goto :conda_console
if exist "%CONDA_PYW%" (
    start "" "%CONDA_PYW%" "%APP%"
    exit /b 0
)
:conda_console
"%CONDA_PY%" "%APP%" %*
if errorlevel 1 goto :failed
exit /b 0

:try_venv
rem ---- 1) 再试项目自带的 venv:金币任务模式需要里面的 numpy + opencv ----
if not exist "%VENV%\python.exe" goto :find_python
if not "%~1"=="" goto :venv_console
if exist "%VENV%\pythonw.exe" (
    start "" "%VENV%\pythonw.exe" "%APP%"
    exit /b 0
)
:venv_console
"%VENV%\python.exe" "%APP%" %*
if errorlevel 1 goto :failed
exit /b 0

:find_python
rem ---- 1) 找一个能用的 Python 3(优先应用商店那个占位符) ----
call :probe py -3
if defined PY goto :got_python
call :probe python
if defined PY goto :got_python
echo.
echo 没有找到可用的 Python 3。
echo 请到 https://www.python.org/downloads/ 下载安装,安装时勾选 "Add python.exe to PATH"。
echo 提示:金币任务(图像识别)模式还需要先运行一次 tools\setup_env.py 建好运行环境。
echo.
pause
exit /b 1

:got_python
rem ---- 2) 用同一个 Python 找 pythonw.exe 启动,不带黑窗 ----
for /f "delims=" %%P in ('%PY% -c "import sys;print(sys.executable)" 2^>nul') do set "EXE=%%P"
if not defined EXE goto :console
set "PYW=!EXE:python.exe=pythonw.exe!"
if /i "!PYW!"=="!EXE!" goto :console
if not exist "!PYW!" goto :console
if not "%~1"=="" goto :console
start "" "!PYW!" "%APP%"
exit /b 0

:console
rem ---- 3) 带参数调试、或系统里没有 pythonw 时,用带窗口的 python 跑,方便看报错 ----
if not defined PY exit /b 1
%PY% "%APP%" %*
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo 程序异常退出(错误码 %errorlevel%),上面一行就是具体原因。
pause
exit /b 1

:probe
rem %1 = 命令(如 py),%2 = 版本参数(可为空)
%1 %2 -c "import sys" >nul 2>nul
if errorlevel 1 goto :eof
set "PY=%1 %2"
goto :eof