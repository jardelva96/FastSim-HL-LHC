@echo off
title Build FastSim.exe
echo ============================================
echo   Gerando FastSim.exe
echo ============================================
echo.

pip install pyinstaller pillow -q >nul 2>&1

echo [1/2] Gerando icone...
python assets\generate_icon.py

echo [2/2] Compilando executavel...
pyinstaller --onefile --windowed --name FastSim --icon assets\icon.ico launcher.pyw --distpath .

echo.
echo ============================================
echo   FastSim.exe criado com sucesso!
echo ============================================
echo.
echo Basta dar dois cliques no FastSim.exe para
echo abrir o dashboard.
echo.
pause
