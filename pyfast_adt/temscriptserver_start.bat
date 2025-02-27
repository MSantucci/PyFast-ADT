REM general bat file to start socket communication with pyfast-adt for FEI/TFS machines
@echo off
set TEMSPY_SOCKET_PATH="C:\Documents and settings\supervisor\Desktop\temspy_bot\temspy_socket.py"
title temscript-server start
echo starting temscript-server ports 8080, 8081 and 8082 for Fast-ADT
start cmd.exe /k temscript-server --port 8080
start cmd.exe /k temscript-server --port 8081
start cmd.exe /k %TEMSPY_SOCKET_PATH%
cmd.exe /k temscript-server --port 8082
pause
REM eventually set the path to the temspy_socket.py file
REM C:\pyfast_adt\main\adaptor\microscope\temspy_socket.py