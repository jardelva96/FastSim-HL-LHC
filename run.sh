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

source .venv/bin/activate

echo "[2/3] Instalando dependencias..."
pip install -q -e ".[dev,ui]" > /dev/null 2>&1

echo "[3/3] Abrindo dashboard..."
echo
python -m fastsim_tt4a
