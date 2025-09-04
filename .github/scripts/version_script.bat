@echo off

REM Only set static version for release branches and release candidate tags
if "%GITHUB_REF_TYPE%"=="branch" (
    echo %GITHUB_REF_NAME% | findstr /R "^release/" >nul
    if not errorlevel 1 (
        echo Setting static version for release branch: %GITHUB_REF_NAME%
        set TENSORDICT_BUILD_VERSION=0.10.0
        set SETUPTOOLS_SCM_PRETEND_VERSION=%TENSORDICT_BUILD_VERSION%
        goto setup_build
    )
)

if "%GITHUB_REF_TYPE%"=="tag" (
    echo %GITHUB_REF_NAME% | findstr /R "^v[0-9]*\.[0-9]*\.[0-9]*-rc[0-9]*$" >nul
    if not errorlevel 1 (
        echo Setting static version for release candidate tag: %GITHUB_REF_NAME%
        set TENSORDICT_BUILD_VERSION=0.10.0
        set SETUPTOOLS_SCM_PRETEND_VERSION=%TENSORDICT_BUILD_VERSION%
        goto setup_build
    )
)

echo Using dynamic versioning for development build: %GITHUB_REF_NAME%
REM Ensure the variable is unset for dynamic versioning
set SETUPTOOLS_SCM_PRETEND_VERSION=

:setup_build
echo TENSORDICT_BUILD_VERSION is set to %TENSORDICT_BUILD_VERSION%

if "%CONDA_RUN%"=="" (
    echo CONDA_RUN is not set. Please activate your conda environment or set CONDA_RUN.
    exit /b 1
)

@echo on

set VC_VERSION_LOWER=17
set VC_VERSION_UPPER=18
if "%VC_YEAR%" == "2019" (
    set VC_VERSION_LOWER=16
    set VC_VERSION_UPPER=17
)
if "%VC_YEAR%" == "2017" (
    set VC_VERSION_LOWER=15
    set VC_VERSION_UPPER=16
)

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [%VC_VERSION_LOWER%^,%VC_VERSION_UPPER%^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)

:vswhere
if "%VSDEVCMD_ARGS%" == "" (
    call "%VS15VCVARSALL%" x64 || exit /b 1
) else (
    call "%VS15VCVARSALL%" x64 %VSDEVCMD_ARGS% || exit /b 1
)

@echo on

if "%CU_VERSION%" == "xpu" call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

set DISTUTILS_USE_SDK=1

set args=%1
shift
:start
if [%1] == [] goto done
set args=%args% %1
shift
goto start

:done
if "%args%" == "" (
    echo Usage: vc_env_helper.bat [command] [args]
    echo e.g. vc_env_helper.bat cl /c test.cpp
)

%args% || exit /b 1
