#!/usr/bin/env bash
set -e

echo "============================================"
echo "  FastSim HL-LHC - Simulacao Rapida"
echo "============================================"
echo

if [ ! -d ".venv" ]; then
    echo "[1/3] Criando ambiente virtual..."
    python3 -m venv .venv
else
    echo "[1/3] Ambiente virtual encontrado."
fi

echo "[2/3] Instalando dependencias..."
.venv/bin/pip install -q -e ".[dev,ui]" > /dev/null 2>&1

mkdir -p .streamlit
if [ ! -f ".streamlit/credentials.toml" ]; then
    printf '[general]\nemail = ""\n' > .streamlit/credentials.toml
fi
if [ ! -f ".streamlit/config.toml" ]; then
    printf '[server]\nheadless = false\n\n[browser]\ngatherUsageStats = false\n' > .streamlit/config.toml
fi

echo "[3/3] Abrindo dashboard..."
echo
.venv/bin/python -m streamlit run src/fastsim_tt4a/dashboard.py
