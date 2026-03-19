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

if not exist ".streamlit" mkdir .streamlit
if not exist ".streamlit\credentials.toml" (
    echo [general] > .streamlit\credentials.toml
    echo email = "" >> .streamlit\credentials.toml
)
if not exist ".streamlit\config.toml" (
    echo [server] > .streamlit\config.toml
    echo headless = false >> .streamlit\config.toml
    echo. >> .streamlit\config.toml
    echo [browser] >> .streamlit\config.toml
    echo gatherUsageStats = false >> .streamlit\config.toml
)

echo [3/3] Abrindo dashboard...
echo.
.venv\Scripts\python.exe -m streamlit run src\fastsim_tt4a\dashboard.py
