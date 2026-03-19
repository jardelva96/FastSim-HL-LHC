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

call .venv\Scripts\activate.bat

echo [2/3] Instalando dependencias...
pip install -q -e ".[dev,ui]" >nul 2>&1

echo [3/3] Abrindo dashboard...
echo.
python -m fastsim_tt4a
