@echo off
REM MindForge 一键启动脚本（Windows）
REM 用法: 双击或 start.bat

echo ==================== MindForge 启动中 ====================

REM 1. 基础设施
echo [1/3] 启动 Qdrant + Redis ...
docker compose up -d 2>nul || echo   ⚠️ Docker 未运行，跳过基础设施

REM 2. 前端构建
echo [2/3] 构建前端 ...
cd /d "%~dp0mindforge-web"
call npm install --silent 2>nul
call npm run build 2>nul || echo   ⚠️ 前端构建失败，将以 API-only 模式运行
cd /d "%~dp0"

REM 3. 后端启动
echo [3/3] 启动后端 ...
cd /d "%~dp0src"
start "MindForge Server" cmd /k "uvicorn mindforge.api.server:app --host 0.0.0.0 --port 8000 --reload"
cd /d "%~dp0"

timeout /t 3 /nobreak >nul

echo.
echo ==================== 启动完成 ====================
echo   浏览器打开: http://localhost:8000
echo   API 文档  : http://localhost:8000/docs
echo ==================================================
pause
