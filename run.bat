@echo off
REM ══════════════════════════════════════════════════════════════════════════════
REM  DevMind Multi-Agent MCP — Windows Launcher
REM ══════════════════════════════════════════════════════════════════════════════

SET VENV=%~dp0venv\Scripts

echo.
echo  ██████╗ ███████╗██╗   ██╗███╗   ███╗██╗███╗   ██╗██████╗
echo  ██╔══██╗██╔════╝██║   ██║████╗ ████║██║████╗  ██║██╔══██╗
echo  ██║  ██║█████╗  ██║   ██║██╔████╔██║██║██╔██╗ ██║██║  ██║
echo  ██║  ██║██╔══╝  ╚██╗ ██╔╝██║╚██╔╝██║██║██║╚██╗██║██║  ██║
echo  ██████╔╝███████╗ ╚████╔╝ ██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
echo  ╚═════╝ ╚══════╝  ╚═══╝  ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝
echo.
echo  Developer Productivity Intelligence — Multi-Agent MCP System
echo  ─────────────────────────────────────────────────────────────
echo.

IF "%1"=="" GOTO MENU
IF "%1"=="setup"  GOTO SETUP
IF "%1"=="ingest" GOTO INGEST
IF "%1"=="server" GOTO SERVER
IF "%1"=="chat"   GOTO CHAT
IF "%1"=="stats"  GOTO STATS
GOTO MENU

:MENU
echo  Choose an option:
echo.
echo    1. setup   — Initialize database tables (run first time only)
echo    2. ingest  — Index your GitHub repo into DevMind
echo    3. server  — Start MCP server (for Claude Desktop)
echo    4. chat    — Start the original terminal chat agent
echo    5. stats   — Show knowledge base statistics
echo.
SET /P CHOICE="  Enter option (1-5): "

IF "%CHOICE%"=="1" GOTO SETUP
IF "%CHOICE%"=="2" GOTO INGEST
IF "%CHOICE%"=="3" GOTO SERVER
IF "%CHOICE%"=="4" GOTO CHAT
IF "%CHOICE%"=="5" GOTO STATS
echo Invalid choice. Exiting.
EXIT /B 1

:SETUP
echo.
echo [1/1] Creating all database tables...
"%VENV%\python.exe" setup_db.py
GOTO END

:INGEST
echo.
echo [1/1] Indexing GitHub repository...
echo       (This may take 2-5 minutes depending on repo size)
"%VENV%\python.exe" mcp_server.py --ingest
GOTO END

:SERVER
echo.
echo [MCP Server] Starting DevMind MCP server...
echo             Claude Desktop will connect automatically.
echo             Press Ctrl+C to stop.
echo.
"%VENV%\python.exe" mcp_server.py
GOTO END

:CHAT
echo.
echo [Terminal Agent] Starting original chat interface...
"%VENV%\python.exe" agent.py
GOTO END

:STATS
echo.
"%VENV%\python.exe" mcp_server.py --stats
GOTO END

:END
echo.
pause
