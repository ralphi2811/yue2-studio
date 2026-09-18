"""Isolation des tests : dossiers de sortie et de réglages temporaires, aucun GPU, aucun réseau.

Les variables d'environnement doivent être posées AVANT l'import des modules du studio
(``studio.paths`` les lit à l'import), d'où le ``pytest_configure``.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="yue2-studio-tests-"))
os.environ["YUE2_STUDIO_OUTPUT"] = str(_TMP / "outputs")
os.environ["YUE2_STUDIO_DATA"] = str(_TMP / "data")
for _k in ("YUE2_LLM_API_KEY", "YUE2_LLM_BASE_URL", "YUE2_LLM_MODEL"):
    os.environ.pop(_k, None)

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture
def engine():
    from studio.engine import get_engine
    return get_engine()


@pytest.fixture
def store(tmp_path):
    """Un ProjectStore isolé par test."""
    from studio.projects import ProjectStore
    return ProjectStore(tmp_path / "projects")


@pytest.fixture
def make_job(engine):
    """Fabrique un job terminé, persisté sur disque, avec des artefacts factices."""
    from studio.engine import Job

    def _make(name="chanson", *, status="done", audio=True, score=None, project_id=None, version=None,
              style="English, warm piano pop, female voice, 88 BPM",
              lyrics="[Verse]\nLine one here\nLine two here\n\n[Chorus]\nSing it loud\nSing it clear",
              cot="full", seed=831001, cfg_scale=None, ode_steps=32, semantic_sampling=None, abc_sampling=None,
              audio_seconds=46.0):
        request = {"id": name, "style": style, "lyrics": lyrics, "cot": cot, "seed": seed}
        if cfg_scale is not None:
            request["cfg_scale"] = cfg_scale
        job = Job(id=engine.new_job_id(name), name=name, kind="song", request=request, ode_steps=ode_steps,
                  semantic_sampling=semantic_sampling or {}, abc_sampling=abc_sampling or {},
                  project_id=project_id, version=version, status=status)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        if audio and status == "done":
            (job.output_dir / "audio.flac").write_bytes(b"fLaC")
            (job.output_dir / "latent.npy").write_bytes(b"\x93NUMPY")
            job.summary = {"audio_seconds": audio_seconds, "truncated": {"abc": False, "semantic": False},
                           "timing": {"e2e_seconds": 17.7}, "has_audio": True, "has_score": bool(score), "vae": "standard"}
        if score:
            (job.output_dir / "score.abc").write_text(score, encoding="utf-8")
        job.started_at, job.finished_at = 0.0, 17.7
        job.seq = engine.next_seq()
        job.persist()
        engine.jobs[job.id] = job
        return job

    return _make


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from studio.app import app
    with TestClient(app) as c:
        yield c


SCORE_ABC = """X:1
T:
M:4/4
L:1/32
Q:1/4=83
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% intro
V: Vocal
z24z4"C"z4|"C"z32|"Em"z32|"Fmaj7"z32|
V: Ins
z24z6G2|c4G4c4G4c4G4c2d2e4|B4G4B4G4B4G4B2c2d4|A8A8A8A8|
% verse
V: Vocal
"C"z8d2e6e6d2d6c2|"Em"d16z16|"F"z8d4e4e6d2d4c4-|"F"c4c12z16|
V: Ins
Z4|
% outro
V: Vocal
"C"z32|"Em"z32|"Fmaj7"z32|"Fmaj7"z32|
V: Ins
c8c8c8c2d2e4|B8B8B8B2c2d4|A8A8A8A8|A8z24|
"""


@pytest.fixture
def score_abc() -> str:
    return SCORE_ABC
