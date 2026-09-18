"""Chemins du studio, surchargeables par variables d'environnement (isolation des tests).

- ``YUE2_STUDIO_OUTPUT`` : dossier des jobs (défaut ``outputs/studio`` à la racine du dépôt)
- ``YUE2_STUDIO_DATA``   : réglages persistants (défaut ``studio/data``)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("YUE2_STUDIO_OUTPUT") or ROOT / "outputs" / "studio")
DATA_DIR = Path(os.environ.get("YUE2_STUDIO_DATA") or Path(__file__).resolve().parent / "data")
PROJECTS_ROOT = OUTPUT_ROOT / "projects"


def load_env_file(path: Path | None = None, *, override: bool = False) -> list[str]:
    """Charge un fichier ``.env`` minimal (``CLÉ=valeur``, ``#`` commentaires, guillemets simples ou doubles tolérés)
    dans ``os.environ`` sans dépendance externe. Les variables déjà définies gagnent sauf ``override``.
    Retourne les noms chargés. Le même fichier sert à ``docker compose``, qui le lit de son côté."""
    path = path or ROOT / ".env"
    loaded: list[str] = []
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded
