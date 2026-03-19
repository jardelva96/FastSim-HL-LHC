@echo off
title FastSim HL-LHC Dashboard
echo ============================================
echo   FastSim HL-LHC - Simulacao Rapida
echo ============================================
echo.

if not exist ".venv" (
    echo [1/3] Criando ambiente virtual...
    python -m venv .venv
) else (
    echo [1/3] Ambiente virtual encontrado.
)

echo [2/3] Instalando dependencias...
.venv\Scripts\pip.exe install -q -e ".[dev,ui]" >nul 2>&1

echo [3/3] Abrindo dashboard...
echo.
.venv\Scripts\python.exe -m fastsim_tt4a
