@echo off
chcp 65001 >nul
cd /d %~dp0
REM 优先用 D 盘虚拟环境(没有则提示按安装指南创建)
if exist D:\envs\shiguang\Scripts\python.exe (
    D:\envs\shiguang\Scripts\python.exe run.py
) else if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe run.py
) else (
    echo [提示] 未找到虚拟环境,请先按 docs\安装运行指南.md 创建
    python run.py
)
pause
