@echo off
setlocal enableextensions enabledelayedexpansion
rem ==========================================================================
rem  竖屏视频自动上滑助手 - 启动器
rem
rem  重要:本文件必须保存为 ANSI/GBK 编码 + Windows(CRLF)换行。
rem        用 UTF-8 或只有 LF 换行时,cmd 会读错中文字符位置,
rem        报 "'xxx' 不是内部或外部命令" 之类莫名其妙的错误。
rem  用法:在本文件后面附加参数(如 启动助手.bat --dry-run),走控制台显示日志。
rem
rem  环境怎么挑(2026-11-18 改):
rem    1) 先找带 CUDA 的环境 —— 项目自带的 env_gpu,然后是用户自己的 myTorch。
rem       光看目录里有 torch 不算数,要真跑一次 torch.cuda.is_available():
rem       显卡不支持 / 驱动太旧 / 装的是 CPU 版 torch 都会返回 False。
rem    2) 一个能用的显卡环境都没有 → 退回 CPU 环境(Scroller,判页走 ONNX)。
rem    3) 都没有 → 再退回项目里的 venv、最后退回系统 Python。
rem  换句话说:双击就走"有显卡用显卡,没显卡用 CPU",不用手动改配置。
rem ==========================================================================
cd /d "%~dp0"
set "APP=%~dp0vertical_swiper.py"
set "PY="
set "PYW="
set "EXE="
set "VENV=%~dp0venv\Scripts"

rem ---- 1) 优先:能用显卡的 conda 环境(GPU 推理) ----
set "CONDA_PY="
set "CONDA_PYW="
set "ENV_KIND="
for %%D in (
    "%~dp0env_gpu"
    "E:\CondaData\envs\myTorch"
) do (
    if not defined CONDA_PY if exist "%%~D\python.exe" if exist "%%~D\Lib\site-packages\torch\__init__.py" (
        "%%~D\python.exe" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
        if not errorlevel 1 (
            set "CONDA_PY=%%~D\python.exe"
            set "CONDA_PYW=%%~D\pythonw.exe"
            set "ENV_KIND=GPU(显卡加速)"
        )
    )
)
if defined CONDA_PY goto :env_ready

rem ---- 2) 没有可用显卡:退回 CPU 环境 ----
for %%D in (
    "E:\CondaData\envs\Scroller"
    "%USERPROFILE%\.conda\envs\Scroller"
    "%~dp0env_gpu"
) do (
    if not defined CONDA_PY if exist "%%~D\python.exe" (
        set "CONDA_PY=%%~D\python.exe"
        set "CONDA_PYW=%%~D\pythonw.exe"
        set "ENV_KIND=CPU(没有可用的显卡)"
    )
)

:env_ready
if exist "%APP%" goto :check_env
echo.
echo 找不到 vertical_swiper.py,请确认它和本文件在同一个文件夹里。
echo.
pause
exit /b 1

:check_env
if not defined CONDA_PY goto :try_venv
if not "%~1"=="" goto :conda_console
if exist "%CONDA_PYW%" (
    start "" "%CONDA_PYW%" "%APP%"
    exit /b 0
)
:conda_console
echo 运行环境: %CONDA_PY%
echo 推理方式: %ENV_KIND%
"%CONDA_PY%" "%APP%" %*
if errorlevel 1 goto :failed
exit /b 0

:try_venv
rem ---- 3) 再试项目自带的 venv ----
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
rem ---- 3) 找一个能用的 Python 3(优先应用商店那个占位符) ----
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
rem ---- 4) 用同一个 Python 找 pythonw.exe 启动,不带黑窗 ----
for /f "delims=" %%P in ('%PY% -c "import sys;print(sys.executable)" 2^>nul') do set "EXE=%%P"
if not defined EXE goto :console
set "PYW=!EXE:python.exe=pythonw.exe!"
if /i "!PYW!"=="!EXE!" goto :console
if not exist "!PYW!" goto :console
if not "%~1"=="" goto :console
start "" "!PYW!" "%APP%"
exit /b 0

:console
rem ---- 5) 带参数调试、或系统里没有 pythonw 时,用带窗口的 python 跑,方便看报错 ----
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