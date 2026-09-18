"""Moteur : persistance des jobs, file, renommage, suppression, réglages (sans GPU ni modèle)."""
import json

from studio.engine import EngineSettings, Job, Stage, slugify


def test_job_persist_and_reload(make_job):
    job = make_job("persist", project_id="proj-1", version=2)
    job.stages.append(Stage(label="Planning score", label_fr="Plan", completed=10, total=None, started=0.0, finished=2.0, status="completed"))
    job.live_abc = "X:1"
    job.persist()
    data = json.loads((job.output_dir / "studio.json").read_text())
    assert data["project_id"] == "proj-1" and data["version"] == 2 and "live_abc" not in data
    assert data["stages"][0]["elapsed"] == 2.0
    loaded = Job.from_disk(job.output_dir)
    assert loaded.id == job.id and loaded.project_id == "proj-1" and loaded.status == "done"
    assert loaded.stages[0].label == "Planning score" and loaded.elapsed == 17.7


def test_running_job_on_disk_becomes_failed(make_job):
    job = make_job("interrupted", status="running", audio=False)
    loaded = Job.from_disk(job.output_dir)
    assert loaded.status == "failed" and "redémarr" in loaded.error


def test_from_disk_ignores_garbage(tmp_path):
    (tmp_path / "studio.json").write_text("{not json")
    assert Job.from_disk(tmp_path) is None
    assert Job.from_disk(tmp_path / "missing") is None


def test_rename_persists_and_bumps(engine, make_job):
    job = make_job("avant")
    v = engine.version
    assert engine.rename(job.id, "  Après   coup ").name == "Après coup"
    assert engine.version > v
    assert json.loads((job.output_dir / "studio.json").read_text())["name"] == "Après coup"
    assert engine.rename(job.id, "  ") is None and engine.rename("ghost", "x") is None


def test_cancel_queued_and_delete(engine, make_job):
    job = make_job("q", status="queued", audio=False)
    engine.queue.append(job.id)
    engine.cancel(job.id)
    assert job.status == "cancelled" and job.id not in engine.queue
    assert engine.delete(job.id) is True and job.id not in engine.jobs and not job.output_dir.exists()
    assert engine.delete("ghost") is False


def test_delete_refuses_running(engine, make_job):
    job = make_job("run", status="running", audio=False)
    assert engine.delete(job.id) is False
    job.status = "done"


def test_snapshot_orders_history(engine, make_job):
    a = make_job("a")
    b = make_job("b")
    snap = engine.snapshot()
    ids = [j.id for j in snap["history"]]
    assert ids.index(b.id) < ids.index(a.id)
    assert snap["current"] is None and isinstance(snap["settings"], EngineSettings)


def test_settings_roundtrip(tmp_path, monkeypatch):
    import studio.engine as E
    monkeypatch.setattr(E, "SETTINGS_FILE", tmp_path / "settings.json")
    s = EngineSettings(budget=20, offload_ar=True, vae="custom", vae_custom="/x", vae_core_frames="512")
    s.save()
    loaded = EngineSettings.load()
    assert loaded == s
    kw = loaded.pipeline_kwargs()
    assert kw["vae"] == "/x" and kw["vae_core_frames"] == 512 and kw["memory_budget_gib"] == 20 and kw["offload_ar"] is True
    (tmp_path / "settings.json").write_text("garbage")
    assert EngineSettings.load() == EngineSettings()


def test_new_job_id_is_filesystem_safe(engine):
    jid = engine.new_job_id("Été & hiver / 2026")
    assert "/" not in jid and " " not in jid and slugify("Été & hiver") in jid
