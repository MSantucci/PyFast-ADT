REM specific bat file to start socket communication with pyfast-adt in the f30 @mainz
@echo off
:: Define the path to your script in a variable for winXP
set CRED_PATH="C:\Documents and settings\supervisor\Desktop\temspy_bot\cred_temspy_server.py"
set TEMSPY_SOCKET_PATH="C:\Documents and settings\supervisor\Desktop\temspy_bot\temspy_socket.py"
:: Define the path to your script in a variable for Win10
REM set TEMSPY_SOCKET_PATH="C:\Users\supervisor\Desktop\temspy_bot\temspy_socket.py"

echo ===== DETECTING WINDOWS VERSION =====
for /f "tokens=4 delims=[]. " %%A in ('ver') do set VERSION=%%A
echo Windows Version: %VERSION%

if "%VERSION%" == "5" (
    set USE_AT=1
    echo Using 'at' command (Windows XP)
) else (
    set USE_AT=0
    REM echo Using 'schtasks'
)
:: Get current time (HH:MM) and AM/PM
for /f "tokens=1,2,3 delims=:. " %%A in ('time /t') do (
    set HH=%%A
    set MM=%%B
    set AMPM=%%C
)

set HH=%HH: =%
set MM=%MM: =%
set AMPM=%AMPM: =%

:: Convert time to 24-hour format
if /i "%AMPM%" == "PM" (
    if %HH% lss 12 set /a HH+=12
) else if /i "%AMPM%" == "AM" (
    if %HH% == 12 set HH=0
)

:: Add 1 minute
set /a MM=%MM% + 1
if %MM% GEQ 60 (
    set /a MM=0
    set /a HH=%HH% + 1
)
if %HH% GEQ 24 set HH=0
if %MM% LSS 10 set MM=0%MM%

:: Ensure the format is correct
if %HH% LSS 10 set HH=0%HH%
if %MM% LSS 10 set MM=0%MM%

:: Run as SYSTEM
if %USE_AT%==1 (
    REM echo [DEBUG] Executing: at %HH%:%MM% /interactive cmd.exe /c %CRED_PATH%
    echo ===== started Cred_temspy_server =====
    echo Scheduled Time for Cred_tesmpy_server: %HH%:%MM%
    at %HH%:%MM% /interactive cmd.exe /c %CRED_PATH%
) else (
    echo skipping Cred_temspy_server
    REM echo [DEBUG] Executing: schtasks /create /tn "ElevatedCMD" /tr "cmd.exe" /sc once /st %HH%:%MM% /ru System
    REM schtasks /create /tn "ElevatedCMD" /tr "cmd.exe /c %CRED_PATH%" /sc once /st %HH%:%MM% /ru System /IT
)

REM original bat file here
title temscript-server start
echo starting temscript-server ports 8080, 8081 and 8082 for Fast-ADT
start cmd.exe /k temscript-server --port 8080
start cmd.exe /k temscript-server --port 8081
cmd.exe /k temscript-server --port 8082
if %USE_AT%==1 (
    timeout /t 65 /nobreak
) else (
)
start cmd.exe /k %TEMSPY_SOCKET_PATH%

pause
