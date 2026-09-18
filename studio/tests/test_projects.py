"""Projets : persistance, numérotation des versions, diffs, résumés."""
import json

from studio import projects as PR


def test_create_save_load_roundtrip(store, tmp_path):
    p = store.create("Mon projet", seed=42)
    assert p.seed == 42 and p.versions == [] and p.next_number == 1
    p.messages.append({"role": "user", "content": "salut"})
    p.turns.append({"role": "user", "text": "salut"})
    store.save(p)
    fresh = PR.ProjectStore(tmp_path / "projects")
    loaded = fresh.get(p.id)
    assert loaded is not None and loaded.name == "Mon projet" and loaded.messages[0]["content"] == "salut"
    assert (tmp_path / "projects" / p.id / "project.json").is_file()
    assert json.loads((tmp_path / "projects" / p.id / "project.json").read_text())["seed"] == 42


def test_versions_are_numbered_and_idempotent(store):
    p = store.create("v")
    v1 = store.add_version(p, "job-a", source="import")
    v2 = store.add_version(p, "job-b", source="assistant")
    again = store.add_version(p, "job-b")
    assert (v1.number, v2.number) == (1, 2) and again is v2 and len(p.versions) == 2
    assert p.latest.job_id == "job-b" and p.next_number == 3
    assert store.for_job("job-a")[1].number == 1 and store.for_job("nope") is None
    assert p.version(2).source == "assistant" and p.version(9) is None


def test_record_outcome_adds_note(store):
    p = store.create("o")
    store.add_version(p, "job-a")
    res = store.record_outcome("job-a", {"status": "done", "audio_seconds": 66.0, "planned_bars": 24, "planned_bpm": 83,
                                          "truncated": {"abc": False, "semantic": True}, "e2e_seconds": 27.0})
    assert res is not None
    note = p.turns[-1]
    assert note["role"] == "note" and "v1 générée" in note["text"] and "TRONQUÉE" in note["text"] and "1 min 06 s" in note["text"]
    assert store.record_outcome("ghost", {}) is None


def test_rename_and_delete(store, tmp_path):
    p = store.create("a")
    assert store.rename(p.id, "  Nouveau   nom ").name == "Nouveau nom"
    assert store.rename(p.id, "   ") is None and store.rename("ghost", "x") is None
    assert store.delete(p.id) is not None and store.get(p.id) is None
    assert not (tmp_path / "projects" / p.id).exists()


def test_all_is_sorted_by_update(store):
    a = store.create("a")
    b = store.create("b")
    store.save(a)  # a devient le plus récent
    assert [p.id for p in store.all()][:2] == [a.id, b.id]


def test_diff_lines_and_requests():
    before = {"style": "S1", "lyrics": "[Verse]\nun\ndeux\n[Chorus]\ntrois", "cot": "full", "seed": 1}
    after = {"style": "S1", "lyrics": "[Verse]\nun\ndeux bis\n[Chorus]\ntrois\nquatre", "cot": "melody", "seed": 1, "cfg_scale": 1.2}
    ch = PR.diff_requests(before, after)
    assert ch["style"] is None and not ch["empty"]
    assert ch["lyrics_stats"] == {"added": 2, "removed": 1}
    assert set(ch["settings"]) == {"cot", "cfg_scale"}
    text = PR.summarize_changes(ch)
    assert "paroles : +2 / -1" in text and "cot full → melody" in text
    assert PR.summarize_changes(PR.diff_requests(before, dict(before))) .startswith("aucun changement")


def test_diff_requests_tracks_extras_and_abc():
    b = {"style": "S", "lyrics": "L", "cot": "full", "seed": 1}
    a = dict(b, abc="X:1")
    ch = PR.diff_requests(b, a, before_extra={"ode_steps": 32, "semantic_sampling": {}}, after_extra={"ode_steps": 16, "semantic_sampling": {"max_tokens": 500}})
    assert ch["settings"]["abc"]["after"] == "partition fournie"
    assert ch["settings"]["ode_steps"] == {"before": 32, "after": 16}
    assert ch["settings"]["semantic_sampling"]["after"] == {"max_tokens": 500}


def test_slug():
    assert PR.slug("Mon projet !") == "Mon_projet" and PR.slug("") == "projet"
