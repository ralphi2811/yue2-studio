"""Projets : une conversation avec l'assistant et la suite de versions générées.

Un projet est un dossier ``outputs/studio/projects/<id>/project.json``. Il référence des
jobs (``outputs/studio/<job_id>``) sans jamais les déplacer : chaque version garde son
audio, sa partition et ses réglages. Supprimer un projet ne supprime pas les jobs, sauf
demande explicite.
"""
from __future__ import annotations

import difflib
import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import PROJECTS_ROOT

MAX_MESSAGES = 14          # historique brut envoyé au LLM (les plus récents)
TRACKED_SETTINGS = ("cot", "seed", "cfg_scale")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Version:
    number: int
    job_id: str
    created: str = field(default_factory=_now)
    label: str = ""
    source: str = "manual"          # assistant | manual | import
    outcome: dict = field(default_factory=dict)   # audio_seconds, truncated, planned…, rempli à la fin du job

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Project:
    id: str
    name: str
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    seed: int | None = None                       # seed stable partagée entre versions
    messages: list[dict] = field(default_factory=list)   # historique OpenAI (sans system)
    turns: list[dict] = field(default_factory=list)      # affichage : role, text, proposal, warnings, error, note
    proposal: dict | None = None                          # dernière proposition normalisée
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    versions: list[Version] = field(default_factory=list)

    # ---- dérivés -----------------------------------------------------
    @property
    def latest(self) -> Version | None:
        return self.versions[-1] if self.versions else None

    @property
    def next_number(self) -> int:
        return (self.versions[-1].number + 1) if self.versions else 1

    def version(self, number: int) -> Version | None:
        return next((v for v in self.versions if v.number == number), None)

    def version_for_job(self, job_id: str) -> Version | None:
        return next((v for v in self.versions if v.job_id == job_id), None)

    def recent_messages(self) -> list[dict]:
        return self.messages[-MAX_MESSAGES:]

    # ---- sérialisation ---------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["versions"] = [v.to_dict() for v in self.versions]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        versions = [Version(**{k: v for k, v in item.items() if k in Version.__dataclass_fields__})
                    for item in data.get("versions", [])]
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "versions"}
        project = cls(**fields)
        project.versions = versions
        return project


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip()).strip("._-")[:40]
    return s or "projet"


class ProjectStore:
    """Persistance JSON des projets, un dossier par projet, protégée par un verrou."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else PROJECTS_ROOT
        self.lock = threading.RLock()
        self._cache: dict[str, Project] = {}
        self._loaded = False

    # ---- chargement --------------------------------------------------
    def _path(self, project_id: str) -> Path:
        return self.root / project_id / "project.json"

    def load_all(self) -> None:
        with self.lock:
            self._cache.clear()
            if self.root.is_dir():
                for path in sorted(self.root.glob("*/project.json")):
                    try:
                        project = Project.from_dict(json.loads(path.read_text(encoding="utf-8")))
                        self._cache[project.id] = project
                    except Exception:
                        continue
            self._loaded = True

    def _ensure(self) -> None:
        if not self._loaded:
            self.load_all()

    def all(self) -> list[Project]:
        with self.lock:
            self._ensure()
            return sorted(self._cache.values(), key=lambda p: p.updated, reverse=True)

    def get(self, project_id: str | None) -> Project | None:
        if not project_id:
            return None
        with self.lock:
            self._ensure()
            return self._cache.get(project_id)

    def for_job(self, job_id: str) -> tuple[Project, Version] | None:
        with self.lock:
            self._ensure()
            for project in self._cache.values():
                version = project.version_for_job(job_id)
                if version is not None:
                    return project, version
        return None

    # ---- écriture ----------------------------------------------------
    def save(self, project: Project) -> Project:
        with self.lock:
            self._ensure()
            project.updated = _now()
            path = self._path(project.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(project.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            self._cache[project.id] = project
            return project

    def create(self, name: str = "Nouveau projet", *, seed: int | None = None) -> Project:
        project_id = f"{datetime.now():%Y%m%d-%H%M%S}-{slug(name)}-{uuid.uuid4().hex[:4]}"
        return self.save(Project(id=project_id, name=name.strip() or "Nouveau projet", seed=seed))

    def rename(self, project_id: str, name: str) -> Project | None:
        project = self.get(project_id)
        name = " ".join((name or "").split())[:120]
        if project is None or not name:
            return None
        project.name = name
        return self.save(project)

    def add_version(self, project: Project, job_id: str, *, source: str = "manual", label: str = "") -> Version:
        with self.lock:
            if project.version_for_job(job_id) is not None:
                return project.version_for_job(job_id)
            version = Version(number=project.next_number, job_id=job_id, source=source, label=label)
            project.versions.append(version)
            self.save(project)
            return version

    def record_outcome(self, job_id: str, outcome: dict) -> tuple[Project, Version] | None:
        found = self.for_job(job_id)
        if found is None:
            return None
        project, version = found
        version.outcome = dict(outcome)
        project.turns.append({"role": "note", "text": outcome_text(version), "version": version.number})
        self.save(project)
        return project, version

    def delete(self, project_id: str) -> Project | None:
        with self.lock:
            self._ensure()
            project = self._cache.pop(project_id, None)
            if project is not None:
                shutil.rmtree(self.root / project_id, ignore_errors=True)
            return project


# --------------------------------------------------------------------------
# Textes et diffs
# --------------------------------------------------------------------------
def _fmt(seconds) -> str:
    if seconds is None:
        return "?"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f} s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s"


def outcome_text(version: Version) -> str:
    o = version.outcome or {}
    if o.get("status") and o["status"] != "done":
        return f"v{version.number} : {o.get('status')}" + (f" — {o.get('error')}" if o.get("error") else "")
    parts = [f"v{version.number} générée"]
    if o.get("audio_seconds"):
        parts.append(f"{_fmt(o['audio_seconds'])} d'audio")
    if o.get("planned_bars") and o.get("planned_bpm"):
        parts.append(f"{o['planned_bars']} mesures à {o['planned_bpm']} BPM")
    tr = o.get("truncated") or {}
    if tr.get("abc") or tr.get("semantic"):
        parts.append("TRONQUÉE (max_tokens atteint)")
    if o.get("e2e_seconds"):
        parts.append(f"{_fmt(o['e2e_seconds'])} de calcul")
    return " · ".join(parts)


def diff_lines(before: str, after: str) -> list[dict]:
    """Diff ligne à ligne des paroles : [{op: equal|insert|delete|replace, a: [...], b: [...]}]."""
    a = (before or "").replace("\r\n", "\n").split("\n")
    b = (after or "").replace("\r\n", "\n").split("\n")
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        out.append({"op": op, "a": a[i1:i2], "b": b[j1:j2]})
    return out


def diff_requests(before: dict, after: dict, *, before_extra: dict | None = None, after_extra: dict | None = None) -> dict:
    """Résume ce qui change entre deux requêtes (style, paroles, réglages)."""
    before, after = before or {}, after or {}
    changes: dict[str, Any] = {"style": None, "lyrics": None, "settings": {}, "lyrics_stats": None}
    if (before.get("style") or "") != (after.get("style") or ""):
        changes["style"] = {"before": before.get("style") or "", "after": after.get("style") or ""}
    if (before.get("lyrics") or "") != (after.get("lyrics") or ""):
        d = diff_lines(before.get("lyrics") or "", after.get("lyrics") or "")
        added = sum(len(x["b"]) for x in d if x["op"] in ("insert", "replace"))
        removed = sum(len(x["a"]) for x in d if x["op"] in ("delete", "replace"))
        changes["lyrics"] = d
        changes["lyrics_stats"] = {"added": added, "removed": removed}
    for key in TRACKED_SETTINGS:
        if before.get(key) != after.get(key):
            changes["settings"][key] = {"before": before.get(key), "after": after.get(key)}
    if (before.get("abc") or "") != (after.get("abc") or ""):
        changes["settings"]["abc"] = {"before": "partition fournie" if before.get("abc") else "—",
                                      "after": "partition fournie" if after.get("abc") else "—"}
    for key, b_val, a_val in (("ode_steps", (before_extra or {}).get("ode_steps"), (after_extra or {}).get("ode_steps")),
                              ("semantic_sampling", (before_extra or {}).get("semantic_sampling"), (after_extra or {}).get("semantic_sampling")),
                              ("abc_sampling", (before_extra or {}).get("abc_sampling"), (after_extra or {}).get("abc_sampling"))):
        if (b_val or None) != (a_val or None) and not (not b_val and not a_val):
            changes["settings"][key] = {"before": b_val, "after": a_val}
    changes["empty"] = not (changes["style"] or changes["lyrics"] or changes["settings"])
    return changes


def summarize_changes(changes: dict) -> str:
    """Une ligne lisible pour la frise et pour le LLM."""
    if changes.get("empty"):
        return "aucun changement de requête (nouvelle graine ou même contenu)"
    parts = []
    if changes.get("style"):
        parts.append("style modifié")
    if changes.get("lyrics"):
        st = changes.get("lyrics_stats") or {}
        parts.append(f"paroles : +{st.get('added', 0)} / -{st.get('removed', 0)} lignes")
    for key, ch in (changes.get("settings") or {}).items():
        parts.append(f"{key} {ch['before']} → {ch['after']}")
    return ", ".join(parts)
