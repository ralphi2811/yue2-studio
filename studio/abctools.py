"""Pont vers les outils ABC du skill officiel (skills/yue2-music/scripts/abc_tools.py).

Le script est chargé par chemin, sans copie, pour rester aligné sur la version du dépôt.
Il implémente un vérificateur strict du dialecte ABC natif : il peut refuser une
partition que le modèle accepterait ; un refus signale « hors périmètre du vérificateur ».
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "yue2-music" / "scripts" / "abc_tools.py"


@lru_cache(maxsize=1)
def tools():
    spec = importlib.util.spec_from_file_location("yue2_abc_tools", SCRIPT)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("yue2_abc_tools", module)
    spec.loader.exec_module(module)
    return module


def inspect(text: str) -> dict:
    """Valide et résume une partition. Lève ValueError (AbcError) si hors dialecte."""
    t = tools()
    score = t.parse_abc(text)
    rep = t.report(score)
    voices = {}
    for name, v in rep["voices"].items():
        voices[name] = {"notes": v["sounding_notes"], "measures": v["measures"],
                        "chords": len(v["chords"]), "key_changes": max(0, len(v["keys"]) - 1)}
    return {"ok": True, "bpm": rep["bpm"], "unit_length": rep["unit_length"],
            "duration_quarters": rep["duration_quarters"],
            "nominal_duration_seconds": rep["nominal_duration_seconds"], "voices": voices}


def strip_chords(text: str, keep_voice: str = "both") -> str:
    return tools().strip_chords(text, keep_voice=keep_voice)


def compare(before: str, after: str, voices: str = "both", allow_tempo_change: bool = False) -> dict:
    t = tools()
    names = t.VOICES if voices == "both" else (voices,)
    return t.compare(t.parse_abc(before), t.parse_abc(after), names, allow_tempo_change=allow_tempo_change)
