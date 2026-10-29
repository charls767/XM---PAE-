#!/usr/bin/env bash
# Crea el entorno, instala dependencias, ejecuta el pipeline y las pruebas (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/create_structure.py .
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
python scripts/run_pipeline.py --source xm --force
python -m pytest -q
echo "Listo. Ejecute: source .venv/bin/activate && streamlit run app.py"
