@echo off
:: Check if CONDA_RUN is set, if not, set it to a default value
if "%CONDA_RUN%"=="" (
    echo CONDA_RUN is not set. Please activate your conda environment or set CONDA_RUN.
    exit /b 1
)

:: Run the pip install command
%CONDA_RUN% conda install conda-forge::pybind11 -y
%CONDA_RUN% conda install conda-forge::rust -y

:: Check if the installation was successful
if errorlevel 1 (
    echo Failed to install cmake and pybind11.
    exit /b 1
) else (
    echo Successfully installed cmake and pybind11.
)
