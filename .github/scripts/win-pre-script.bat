@echo off
:: Check if CONDA_RUN is set, if not, set it to a default value
if "%CONDA_RUN%"=="" (
    echo CONDA_RUN is not set. Please activate your conda environment or set CONDA_RUN.
    exit /b 1
)

:: Run the pip install command for pybind11
%CONDA_RUN% conda install conda-forge::pybind11 -y

:: Check if the installation was successful
if errorlevel 1 (
    echo Failed to install pybind11.
    exit /b 1
) else (
    echo Successfully installed pybind11.
)

:: Install setuptools_scm which is required for building
%CONDA_RUN% pip install setuptools_scm

:: Check if the installation was successful
if errorlevel 1 (
    echo Failed to install setuptools_scm.
    exit /b 1
) else (
    echo Successfully installed setuptools_scm.
)
