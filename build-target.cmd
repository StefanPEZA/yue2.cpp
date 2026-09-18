@echo off
rem Build one CMake target in the configured build tree, with the VS 2026 Build
rem Tools environment loaded. scripts\build-engine.cmd configures; this only
rem builds, so an edit-compile loop does not re-run vcvars and CMake configure.
rem Usage: build-target.cmd <target>

setlocal

set ENGINE=%~dp0
set TARGET=%1
if "%TARGET%"=="" set TARGET=ALL_BUILD

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto :fail

cmake --build "%ENGINE%build" --config Release --target %TARGET% -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 goto :fail

echo === BUILD OK: %TARGET% ===
exit /b 0

:fail
echo === BUILD FAILED: %TARGET% (errorlevel %errorlevel%) ===
exit /b 1
