"""Crea la estructura de carpetas del proyecto guavio-xai.

Uso (desde la carpeta que contendrá el repositorio):
    python scripts/create_structure.py            # crea en el directorio actual
    python scripts/create_structure.py ruta/repo  # crea en otra ruta

Es idempotente: no sobrescribe archivos existentes.
"""
from __future__ import annotations

import sys
from pathlib import Path

DIRECTORIES = [
    "config",
    "data/raw",
    "data/processed",
    "data/external",
    "models",
    "reports/figures",
    "reports/metrics",
    "notebooks",
    "dashboards",
    "scripts",
    "src",
    "src/data",
    "src/models",
    "src/explainability",
    "src/utils",
    "tests",
]

PACKAGE_INITS = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/models/__init__.py",
    "src/explainability/__init__.py",
    "src/utils/__init__.py",
    "dashboards/__init__.py",
    "tests/__init__.py",
]

KEEP_FILES = [
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",
    "models/.gitkeep",
    "reports/figures/.gitkeep",
    "reports/metrics/.gitkeep",
]


def create(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for d in DIRECTORIES:
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in PACKAGE_INITS + KEEP_FILES:
        path = root / f
        if not path.exists():
            path.touch()
    print(f"Estructura creada en {root.resolve()}")
    for d in DIRECTORIES:
        print(f"  {d}/")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    create(target)
