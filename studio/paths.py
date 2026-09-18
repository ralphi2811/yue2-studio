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
