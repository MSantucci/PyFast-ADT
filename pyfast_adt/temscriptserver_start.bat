@echo off
title temscript-server start
echo starting temscript-server ports 8080, 8081 and 8082 for Fast-ADT
start cmd.exe /k temscript-server --port 8080
start cmd.exe /k temscript-server --port 8081
start cmd.exe /k temspy_socket.py
cmd.exe /k temscript-server --port 8082
pause
REM eventually set the path to the temspy_socket.py file
REM C:\pyfast_adt\main\adaptor\microscope\temspy_socket.py