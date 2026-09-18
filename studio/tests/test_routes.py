"""Routes HTTP de bout en bout, avec un exécuteur de jobs simulé (pas de GPU) et un LLM simulé."""
import json
import time

import pytest

from studio import assistant as A
from studio import llm


@pytest.fixture
def fake_runner(engine, monkeypatch):
    """Remplace l'exécution GPU : le job « réussit » instantanément avec des artefacts factices."""
    def _run(job):
        with engine.lock:
            engine.current = job
            job.status, job.started_at = "running", time.monotonic()
        job.output_dir.mkdir(parents=True, exist_ok=True)
        (job.output_dir / "audio.flac").write_bytes(b"fLaC")
        (job.output_dir / "latent.npy").write_bytes(b"\x93NUMPY")
        (job.output_dir / "score.abc").write_text("X:1\nM:4/4\nQ:1/4=90\nK:C\nV: Vocal\nC4|D4|E4|F4|\n", encoding="utf-8")
        job.summary = {"audio_seconds": 12.0, "truncated": {"abc": False, "semantic": False}, "timing": {"e2e_seconds": 1.0},
                       "has_audio": True, "has_score": True, "vae": "standard"}
        job.status = "done"
        with engine.lock:
            job.finished_at = time.monotonic()
            engine.current = None
            job.persist()
            engine._bump()
        for listener in list(engine.listeners):
            listener(job)

    monkeypatch.setattr(engine, "_run", _run)
    return _run


def wait_done(engine, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = engine.jobs.get(job_id)
        if job and job.status in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job never finished")


BASE = {"name": "route test", "style": "French, chanson, 80 BPM", "lyrics": "[Verse]\nla la\nli li", "cot": "full", "seed": "7", "kind": "song"}


# ---------------------------------------------------------------- pages
def test_pages_render(client):
    assert client.get("/").status_code == 200
    assert "Mes morceaux" in client.get("/").text
    r = client.get("/studio")
    assert r.status_code == 200 and 'id="job-form"' in r.text and "assistant-drawer" in r.text
    # deux modes : les quatre panneaux existent, le mode initial est « Composer »
    assert 'data-mode="compose"' in r.text and 'data-action="mode" data-mode="browse"' in r.text
    for cls in ("panel-form", "panel-queue", "panel-detail", "panel-library"):
        assert f'class="panel {cls}"' in r.text, cls
    assert client.get("/partials/settings").status_code == 200
    assert client.get("/partials/form?from_job=nope").status_code == 404
    assert client.get("/api/state").json()["model_state"] in {"unloaded", "loading", "ready", "error"}


def test_form_validation_error_returns_422_with_form(client):
    r = client.post("/jobs", data=dict(BASE, lyrics=""))
    assert r.status_code == 422 and "obligatoires" in r.text


# ---------------------------------------------------------------- jobs
def test_create_job_runs_and_appears_in_library(client, engine, fake_runner):
    r = client.post("/jobs", data=BASE)
    assert r.status_code == 200 and "ajoutée à la file" in r.text and r.headers.get("HX-Trigger") == "queue-changed"
    job_id = next(j.id for j in engine.jobs.values() if j.name == "route test")
    job = wait_done(engine, job_id)
    assert job.status == "done"
    assert "route test" in client.get("/partials/library").text
    detail = client.get(f"/jobs/{job_id}")
    assert detail.status_code == 200 and "Durée nominale" in detail.text
    assert client.get(f"/jobs/{job_id}/file/audio.flac").content == b"fLaC"
    assert client.get(f"/jobs/{job_id}/file/../etc").status_code == 404
    assert client.get(f"/jobs/{job_id}/file/nope.txt").status_code == 404


def test_form_title_reflects_origin_and_detail_has_close_bar(client, engine, fake_runner):
    blank = client.get("/partials/form").text
    assert "Nouveau morceau" in blank and "Repartir de zéro" not in blank
    client.post("/jobs", data=BASE)
    job = wait_done(engine, next(j.id for j in engine.jobs.values() if j.name == "route test"))
    r = client.get(f"/partials/form?from_job={job.id}&mode=variation").text
    assert "Variation de « route test »" in r and "Repartir de zéro" in r and 'name="origin_kind" value="variation"' in r
    assert "Réglages repris de « route test »" in client.get(f"/partials/form?from_job={job.id}&mode=reuse").text
    # l'origine survit à une erreur de validation (le formulaire est renvoyé avec son titre)
    r = client.post("/jobs", data=dict(BASE, lyrics="", origin_kind="variation", origin_name="route test"))
    assert r.status_code == 422 and "Variation de « route test »" in r.text
    detail = client.get(f"/jobs/{job.id}").text
    assert "Morceau sélectionné" in detail and 'data-action="close-detail"' in detail


SCORE = "X:1\nM:4/4\nQ:1/4=90\nK:C\nV: Vocal\nC4|D4|E4|F4|\n"


def test_assistant_panel_follows_the_open_track(client, make_job, store, monkeypatch):
    monkeypatch.setattr(A, "_STORE", store)
    project = store.create("mon_projet")
    linked = make_job("liee", score=SCORE)
    store.add_version(project, linked.id)
    loose = make_job("libre", score=SCORE)
    # morceau d'un projet → conversation du projet, sans sélecteur de projet
    r = client.get(f"/assistant/panel?job_id={linked.id}")
    assert r.status_code == 200 and f'data-project="{project.id}"' in r.text and "mon_projet" in r.text
    assert "choisir un projet" not in r.text and "project-select" not in r.text
    # morceau hors projet → contexte + bouton pour en faire la v1 d'un projet
    r = client.get(f"/assistant/panel?job_id={loose.id}")
    assert r.status_code == 200 and 'data-project=""' in r.text and f'data-job="{loose.id}"' in r.text
    assert "hors projet" in r.text and f"/projects/from-job/{loose.id}" in r.text
    assert client.get("/assistant/panel?job_id=nope").status_code == 404
    # la fiche annonce son projet pour que le tiroir puisse le suivre
    assert f'data-project="{project.id}"' in client.get(f"/jobs/{linked.id}").text
    assert 'data-project=""' in client.get(f"/jobs/{loose.id}").text


def test_compare_candidates_are_limited_to_related_tracks(client, make_job, store, monkeypatch):
    monkeypatch.setattr(A, "_STORE", store)
    project = store.create("proj")
    v1 = make_job("v_un", score=SCORE)
    v2 = make_job("v_deux", score=SCORE)
    store.add_version(project, v1.id)
    store.add_version(project, v2.id)
    stranger = make_job("etranger", score=SCORE)
    edited = make_job("v_un_edit", score=SCORE)
    edited.source_job = v1.id
    edited.persist()
    text = client.get(f"/jobs/{v1.id}").text
    assert f'value="{v2.id}"' in text and "v2 · v_deux" in text            # autre version du projet
    assert f'value="{edited.id}"' in text and "reprend cette partition" in text
    assert stranger.id not in text                                          # aucun lien : pas proposé
    text = client.get(f"/jobs/{edited.id}").text
    assert f'value="{v1.id}"' in text and "partition d&#39;origine" in text   # (échappement HTML de l'apostrophe)
    assert client.get(f"/jobs/{stranger.id}").text.count("compare-row") == 0   # rien de comparable


def test_form_origin_labels():
    from studio.app import form_origin

    class Proj:
        name = "mon_projet"

    assert form_origin({}, None) is None
    assert form_origin({"origin_kind": "reuse"}, None) is None            # pas de nom : pas de titre
    assert form_origin({"origin_kind": "edit-score", "origin_name": "x"}, None) == "Partition retouchée de « x »"
    assert form_origin({"origin_kind": "assistant", "origin_name": "x"}, None) == "Proposition de l'assistant"
    assert form_origin({"origin_kind": "reuse", "origin_name": "x"}, Proj()) == "Nouvelle version de « mon_projet »"
    assert form_origin({"origin_kind": "inconnu", "origin_name": "x"}, None) is None


def test_rename_and_delete_job(client, make_job):
    job = make_job("renommable")
    r = client.patch(f"/jobs/{job.id}/name", data={"name": "Nouveau nom é"})
    assert r.status_code == 200 and 'data-name="Nouveau nom é"' in r.text
    r = client.patch(f"/jobs/{job.id}/name?view=detail", data={"name": "Encore"})
    assert r.status_code == 200 and "<h2" in r.text
    assert client.patch(f"/jobs/{job.id}/name", data={"name": " "}).status_code == 400
    assert client.delete(f"/jobs/{job.id}").status_code == 200
    assert client.get(f"/jobs/{job.id}").status_code == 404


def test_batch_import(client, engine, fake_runner):
    rows = [{"name": "lot_a", "seed": 1}, {"name": "lot_b", "seed": 2, "cot": "off"}, {"name": "bad", "seed": -1}]
    payload = "\n".join(json.dumps(r) for r in rows)
    r = client.post("/jobs/batch", files={"file": ("lot.jsonl", payload, "application/x-ndjson")},
                    data={"style": "S 90 BPM", "lyrics": "[Verse]\nx", "cot": "full", "seed": "1", "kind": "song"})
    assert r.status_code == 200 and "2 jobs ajoutés" in r.text and "Ligne 3" in r.text
    for name in ("lot_a", "lot_b"):
        wait_done(engine, next(j.id for j in engine.jobs.values() if j.name == name))
    assert client.post("/jobs/batch", data={}).status_code == 422
    assert client.get("/batch/template.jsonl").status_code == 200


def test_gallery_filters(client, make_job):
    make_job("zeta", style="English, zouk, 100 BPM")
    make_job("alpha", style="English, ambient, 60 BPM")
    assert "zouk" in client.get("/partials/gallery?q=zouk").text
    assert "ambient" not in client.get("/partials/gallery?q=zouk").text
    html = client.get("/partials/gallery?sort=name").text
    assert html.index("alpha") < html.index("zeta")


# ---------------------------------------------------------------- outils ABC
def test_abc_tools_routes(client, score_abc):
    r = client.post("/abc/inspect", data={"abc": score_abc})
    assert r.status_code == 200 and "Partition valide" in r.text and "83 BPM" in r.text
    assert "refusée" in client.post("/abc/inspect", data={"abc": "X:1\nK:C\n(3CDE|"}).text
    r = client.post("/abc/strip", data={"abc": score_abc, "keep_voice": "Vocal"})
    assert r.status_code == 200 and r.json()["ok"] and '"' not in r.json()["abc"].split("K:C")[1]
    assert client.post("/abc/strip", data={"abc": score_abc, "keep_voice": "Nope"}).status_code == 400
    r = client.post("/abc/compare", data={"before": score_abc, "abc": score_abc})
    assert "identiques" in r.text
    assert "impossible" in client.post("/abc/compare", data={"before": "", "abc": score_abc}).text


# ---------------------------------------------------------------- assistant + projets
@pytest.fixture
def fake_llm(monkeypatch):
    async def chat(messages, **kw):
        users = [m for m in messages if m["role"] == "user"]
        last = users[-1]["content"] if users else ""
        if "Prompt actuel" in last:
            return json.dumps({"style": "French, pop, 90 BPM", "why": "w"}), {}
        return json.dumps({"type": "proposal", "message": "ok", "proposal": {
            "name": "prop_song", "style": "French, indie pop, 84 BPM", "lyrics": "[Verse]\nun\ndeux\n\n[Chorus]\ntrois\nquatre",
            "cot": "full", "cfg_scale": None, "seed": None, "target_duration_seconds": 40}}, ensure_ascii=False), {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "list_models", lambda s=None: _coro(["m1", "m2"]))
    settings = llm.LLMSettings(base_url="http://fake/v1", model="m1")
    monkeypatch.setattr(llm, "_SETTINGS", settings)
    return settings


async def _coro(value):
    return value


def test_assistant_settings_roundtrip(client, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "SETTINGS_FILE", tmp_path / "assistant.json")
    monkeypatch.setattr(llm, "_SETTINGS", None)
    r = client.post("/assistant/settings", data={"base_url": "http://localhost:1234/v1/", "model": "x", "api_key": "sk-secret", "temperature": "0.5", "max_tokens": "1000"})
    assert r.status_code == 200 and "enregistré" in r.text and "sk-secret" not in r.text
    saved = json.loads((tmp_path / "assistant.json").read_text())
    assert saved["base_url"] == "http://localhost:1234/v1" and saved["api_key"] == "sk-secret"
    r = client.post("/assistant/settings", data={"base_url": "http://localhost:1234/v1", "model": "x", "api_key": "", "temperature": "0.5", "max_tokens": "1000"})
    assert json.loads((tmp_path / "assistant.json").read_text())["api_key"] == "sk-secret"   # vide = conserver
    client.post("/assistant/settings", data={"base_url": "http://localhost:1234/v1", "model": "x", "clear_key": "1", "temperature": "0.5", "max_tokens": "1000"})
    assert json.loads((tmp_path / "assistant.json").read_text())["api_key"] == ""
    assert client.post("/assistant/settings", data={"base_url": "u", "model": "x", "temperature": "abc", "max_tokens": "1"}).status_code == 422


def test_assistant_settings_locked_by_environment(client, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "SETTINGS_FILE", tmp_path / "assistant.json")
    monkeypatch.setattr(llm, "_SETTINGS", llm.LLMSettings(base_url="http://saved/v1", model="saved-model"))
    monkeypatch.setenv("YUE2_LLM_BASE_URL", "http://env-host/v1")
    monkeypatch.setenv("YUE2_LLM_MODEL", "env-model")
    # le formulaire affiche les valeurs imposées, champs désactivés, préréglages masqués
    r = client.get("/partials/settings")
    assert r.status_code == 200
    assert 'value="http://env-host/v1"' in r.text and 'value="env-model"' in r.text
    assert "YUE2_LLM_BASE_URL" in r.text and 'data-action="llm-preset"' not in r.text
    assert r.text.count("disabled title=") == 2
    # un envoi (champs désactivés donc absents) ne doit pas effacer les valeurs enregistrées
    r = client.post("/assistant/settings", data={"temperature": "0.5", "max_tokens": "1000"})
    assert r.status_code == 200 and "Actif : env-model @ http://env-host/v1" in r.text
    saved = json.loads((tmp_path / "assistant.json").read_text())
    assert saved["base_url"] == "http://saved/v1" and saved["model"] == "saved-model" and saved["temperature"] == 0.5
    # une seule variable : seul ce champ est verrouillé, l'autre reste modifiable
    monkeypatch.delenv("YUE2_LLM_MODEL")
    r = client.post("/assistant/settings", data={"model": "chosen", "temperature": "0.5", "max_tokens": "1000"})
    assert r.status_code == 200 and "Actif : chosen @ http://env-host/v1" in r.text and 'data-action="llm-preset"' in r.text
    assert json.loads((tmp_path / "assistant.json").read_text())["model"] == "chosen"


def test_assistant_flow_creates_project_and_versions(client, engine, fake_runner, fake_llm, store, monkeypatch):
    monkeypatch.setattr(A, "_STORE", store)
    # 1. message → projet créé + proposition
    r = client.post("/assistant/message", data={"project_id": "", "text": "Une chanson"})
    assert r.status_code == 200 and "Appliquer au formulaire" in r.text
    project = store.all()[0]
    assert project.proposal["name"] == "prop_song" and project.name == "prop_song"
    # 2. appliquer → formulaire avec project_id
    r = client.post("/assistant/apply", data={"project_id": project.id})
    assert r.status_code == 200 and f'name="project_id" value="{project.id}"' in r.text and "v1 du projet" in r.text
    assert "Nouvelle version de « prop_song »" in r.text and "Repartir de zéro" in r.text
    # 3. générer telle quelle → version 1 « assistant »
    r = client.post("/jobs", data={"name": "prop_song", "style": "French, indie pop, 84 BPM", "lyrics": "[Verse]\nun\ndeux\n\n[Chorus]\ntrois\nquatre",
                                   "cot": "full", "seed": "5", "kind": "song", "project_id": project.id})
    assert "Version v1" in r.text
    job1 = wait_done(engine, next(j.id for j in engine.jobs.values() if j.project_id == project.id))
    project = store.get(project.id)
    assert project.versions[0].source == "assistant" and project.seed == 5 and job1.version == 1
    assert project.versions[0].outcome["audio_seconds"] == 12.0 and project.versions[0].outcome["planned_bars"] == 4
    assert project.turns[-1]["role"] == "note" and "v1 générée" in project.turns[-1]["text"]
    # 4. retouche manuelle → v2 « manual », seed du projet réutilisée
    r = client.post("/jobs", data={"name": "prop_song", "style": "French, indie pop, 84 BPM", "lyrics": "[Verse]\nun\nDEUX\n\n[Chorus]\ntrois\nquatre",
                                   "cot": "full", "seed": "5", "kind": "song", "project_id": project.id})
    job2 = wait_done(engine, next(j.id for j in engine.jobs.values() if j.project_id == project.id and j.version == 2))
    project = store.get(project.id)
    assert project.versions[1].source == "manual"
    # 5. frise, diff, galerie groupée, page projet
    page = client.get(f"/projects/{project.id}")
    assert page.status_code == 200 and "v2" in page.text and "Écoute A / B" in page.text
    diff = client.get(f"/projects/{project.id}/diff?a=1&b=2")
    assert diff.status_code == 200 and "+ DEUX" in diff.text and "- deux" in diff.text
    assert client.get(f"/projects/{project.id}/diff?a=1&b=9").status_code == 404
    gallery = client.get("/partials/gallery").text
    assert "2 versions" in gallery and gallery.count("prop_song") >= 1
    # 6. le résumé envoyé au LLM contient les versions
    r = client.post("/assistant/message", data={"project_id": project.id, "text": "plus court"})
    assert r.status_code == 200
    # 7. renommage et suppression du projet (les jobs restent)
    assert client.patch(f"/projects/{project.id}/name", data={"name": "Renommé"}).status_code == 200
    assert store.get(project.id).name == "Renommé"
    r = client.delete(f"/projects/{project.id}")
    assert r.status_code == 204 and r.headers["HX-Redirect"] == "/"
    assert store.get(project.id) is None and job1.id in engine.jobs and job2.id in engine.jobs


def test_project_from_job_and_delete_with_jobs(client, engine, make_job, fake_llm, store, monkeypatch):
    monkeypatch.setattr(A, "_STORE", store)
    job = make_job("existant", seed=99)
    r = client.post(f"/projects/from-job/{job.id}")
    assert r.status_code == 200 and "assistant-open" in r.headers.get("HX-Trigger", "")
    project = store.for_job(job.id)[0]
    assert project.seed == 99 and engine.jobs[job.id].project_id == project.id
    # idempotent
    client.post(f"/projects/from-job/{job.id}")
    assert len(store.all()) == 1
    r = client.post(f"/projects/from-job/{job.id}?target=page")
    assert r.status_code == 204 and r.headers["HX-Redirect"] == f"/projects/{project.id}"
    assert client.post("/projects/from-job/ghost").status_code == 404
    # La pochette du projet dans la galerie porte sa propre corbeille (projet + versions)
    gallery = client.get("/partials/gallery").text
    assert f'hx-delete="/projects/{project.id}?with_jobs=1&card=1"' in gallery
    r = client.delete(f"/projects/{project.id}?with_jobs=1&card=1")
    assert r.status_code == 200 and r.text == "" and "library-changed" in r.headers["HX-Trigger"]
    assert job.id not in engine.jobs and not job.output_dir.exists() and store.get(project.id) is None
    job2 = make_job("second")
    client.post(f"/projects/from-job/{job2.id}")
    project2 = store.for_job(job2.id)[0]
    r = client.delete(f"/projects/{project2.id}?with_jobs=1")
    assert r.status_code == 204 and r.headers["HX-Redirect"] == "/" and job2.id not in engine.jobs


def test_assistant_quick_and_apply_errors(client, fake_llm):
    r = client.post("/assistant/quick", data={"action": "style", "style": "pop"})
    assert r.status_code == 200 and r.json()["value"] == "French, pop, 90 BPM"
    assert client.post("/assistant/quick", data={"action": "nope"}).status_code == 400
    assert client.post("/assistant/apply", data={"project_id": "ghost"}).status_code == 404
