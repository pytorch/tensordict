@echo off
REM Windows wrapper script that calls bash to execute the shell script
REM This is needed because conda run on Windows can't directly execute .sh files

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Call bash to execute the shell script with all passed arguments
bash "%SCRIPT_DIR%version_script_windows.sh" %*
