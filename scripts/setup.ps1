# Crea el entorno, instala dependencias, ejecuta el pipeline y las pruebas (Windows PowerShell).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python scripts/create_structure.py .
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python scripts/run_pipeline.py --source xm --force
python -m pytest -q
Write-Host "Listo. Ejecute: .\.venv\Scripts\Activate.ps1; streamlit run app.py"
