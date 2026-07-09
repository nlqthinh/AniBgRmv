@echo off
setlocal

echo [1/3] Installing runtime requirements...
python -m pip install -r requirements.txt

echo [2/3] Building AniBGRmv with PyInstaller...
pyinstaller AniBGRmv.spec --clean --noconfirm

echo [3/3] Done.
echo Output folder: dist\AniBGRmv
pause
