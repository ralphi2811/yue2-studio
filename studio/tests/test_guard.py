"""Garde-fous de durée : borne sémantique, contrôle de la partition planifiée, exécution par étapes du moteur."""
import numpy as np
import pytest

from studio import guard
from studio.engine import Job


# ---------------------------------------------------------------- fonctions pures
def test_semantic_cap_picks_the_smallest_bound():
    cap, info = guard.semantic_cap(None, None, None)
    assert cap == guard.DEFAULT_MAX_TOKENS and info["reason"] == "utilisateur"
    cap, info = guard.semantic_cap(9000, 60.0, None)            # partition d'1 min → 60×25×1.25+100 = 1975
    assert cap == 1975 and info["reason"] == "partition"
    cap, info = guard.semantic_cap(9000, 300.0, 90)              # durée max 90 s → 2475 < partition 9475
    assert cap == 2475 and info["reason"] == "durée maximale"
    cap, info = guard.semantic_cap(1200, 300.0, 90)              # l'utilisateur est le plus strict
    assert cap == 1200 and info["reason"] == "utilisateur"
    assert set(info["candidates"]) == {"utilisateur", "partition", "durée maximale"}


def test_check_plan_tolerance():
    assert guard.check_plan(None, 60) is None and guard.check_plan(100.0, None) is None
    assert guard.check_plan(77.0, 60) is None                    # 60 × 1.3 = 78 → toléré
    msg = guard.check_plan(483.0, 180)
    assert msg and "8 min 03 s" in msg and "3 min 00 s" in msg and "enregistrée" in msg


def test_sung_lines_and_truncation_notes():
    assert guard.sung_lines("[Intro]\n\n[Outro]") == 0
    assert guard.sung_lines("[Verse]\nla\nli\n[Chorus]\nlo") == 3
    assert guard.truncation_note(False, {"reason": "partition"}) is None
    assert "partition" in guard.truncation_note(True, {"reason": "partition"})
    assert "durée maximale" in guard.truncation_note(True, {"reason": "durée maximale"})
    assert "max_tokens" in guard.truncation_note(True, None)


def test_abc_duration_reexported_from_app(score_abc):
    from studio.app import abc_duration
    assert abc_duration(score_abc)["bars"] == 12


# ---------------------------------------------------------------- moteur : exécution par étapes avec un pipeline simulé
class _Plan:
    def __init__(self, abc, request=None):
        self.abc, self.abc_ids, self.prefix, self.timing, self.truncated = abc, [1], [0], {"seconds": 0.1}, False
        self.request = request

    def save(self, directory):
        (directory / "score.abc").write_text(self.abc or "", encoding="utf-8")
        (directory / "plan.json").write_text("{}")


class _Semantic:
    def __init__(self, plan, n):
        self.plan, self.tokens, self.timing, self.truncated = plan, list(range(n)), {"seconds": 0.2, "output_tokens": n}, False


class FakePipe:
    """Reproduit l'interface utilisée par Engine._run_song sans modèle ni GPU."""

    def __init__(self, planned_abc, seconds_of_audio=10.0):
        self.planned_abc, self.seconds = planned_abc, seconds_of_audio
        self.weights, self.load_timing, self.generation_config = {"mot": {}, "vae": {}}, {}, None
        self.calls = []

    def _request(self, **kw):
        from yue2.protocol import SongRequest
        return SongRequest(**kw)

    def plan(self, request=None, abc_sampling=None, cancelled=None, on_token=None):
        self.calls.append("plan")
        return _Plan(self.planned_abc, request)

    def effective_config(self, request, abc_sampling, semantic_sampling):
        self.calls.append(("config", semantic_sampling))
        return {"generation": {"semantic": semantic_sampling or {}}, "overrides": {}}

    def generate_semantic(self, plan, sampling=None, cancelled=None, on_token=None):
        self.calls.append(("semantic", sampling))
        return _Semantic(plan, 250)

    def synthesize(self, semantic, cancelled=None):
        self.calls.append("synth")
        return np.zeros((10, 64), dtype=np.float32)

    def decode(self, latents, **kw):
        self.calls.append("decode")
        return np.zeros((int(48000 * self.seconds), 2), dtype=np.float32)


SHORT_ABC = "X:1\nM:4/4\nQ:1/4=120\nK:C\nV: Vocal\nC4|D4|E4|F4|\n"          # 4 mesures → 8 s
LONG_ABC = "X:1\nM:4/4\nQ:1/4=60\nK:C\nV: Vocal\n" + "Z4|" * 30 + "\n"       # 120 mesures → 8 min


def _job(engine, make_job, **kw):
    job = make_job("guarded", status="queued", audio=False, **{k: v for k, v in kw.items() if k in ("semantic_sampling",)})
    job.max_seconds = kw.get("max_seconds")
    return job


def test_run_song_applies_plan_cap(engine, make_job, monkeypatch):
    monkeypatch.setattr(engine, "_configure", lambda pipe, job: None)
    monkeypatch.setattr(engine, "_abc_streamer", lambda pipe, job: None)
    pipe = FakePipe(SHORT_ABC)
    job = _job(engine, make_job)
    engine._run_song(pipe, job)
    semantic_call = next(c for c in pipe.calls if isinstance(c, tuple) and c[0] == "semantic")
    assert semantic_call[1] == {"max_tokens": 350}                # 8 s × 25 × 1.25 + 100
    assert job.summary["guard"]["reason"] == "partition" and job.summary["has_audio"]
    assert job.summary["planned"]["bars"] == 4 and job.summary["truncation_note"] is None
    assert (job.output_dir / "audio.flac").is_file() and (job.output_dir / "result.json").is_file()
    assert pipe.calls[-1] == "decode"


def test_run_song_keeps_stricter_user_cap(engine, make_job, monkeypatch):
    monkeypatch.setattr(engine, "_configure", lambda pipe, job: None)
    monkeypatch.setattr(engine, "_abc_streamer", lambda pipe, job: None)
    pipe = FakePipe(SHORT_ABC)
    job = _job(engine, make_job, semantic_sampling={"max_tokens": 200})
    engine._run_song(pipe, job)
    semantic_call = next(c for c in pipe.calls if isinstance(c, tuple) and c[0] == "semantic")
    assert semantic_call[1] == {"max_tokens": 200} and job.summary["guard"]["reason"] == "utilisateur"


def test_run_song_aborts_when_plan_exceeds_max_duration(engine, make_job, monkeypatch):
    monkeypatch.setattr(engine, "_configure", lambda pipe, job: None)
    monkeypatch.setattr(engine, "_abc_streamer", lambda pipe, job: None)
    pipe = FakePipe(LONG_ABC)
    job = _job(engine, make_job, max_seconds=120)
    with pytest.raises(guard.PlanTooLong, match="trop longue"):
        engine._run_song(pipe, job)
    assert "plan" in pipe.calls and not any(isinstance(c, tuple) and c[0] == "semantic" for c in pipe.calls)
    assert (job.output_dir / "score.abc").is_file()                # la partition est conservée
    assert job.summary["has_audio"] is False and job.summary["planned"]["bars"] == 120


def test_run_song_max_duration_within_tolerance_caps_tokens(engine, make_job, monkeypatch):
    monkeypatch.setattr(engine, "_configure", lambda pipe, job: None)
    monkeypatch.setattr(engine, "_abc_streamer", lambda pipe, job: None)
    pipe = FakePipe(SHORT_ABC)                                        # 8 s planifiées
    job = _job(engine, make_job, max_seconds=7)                       # 7 × 1.3 = 9.1 ≥ 8 → toléré
    engine._run_song(pipe, job)
    semantic_call = next(c for c in pipe.calls if isinstance(c, tuple) and c[0] == "semantic")
    assert semantic_call[1] == {"max_tokens": 193} and job.summary["guard"]["reason"] == "durée maximale"


def test_run_marks_plan_too_long_as_failed(engine, make_job, monkeypatch):
    """Le chemin complet Engine._run : statut failed, message lisible, écouteurs notifiés."""
    monkeypatch.setattr(engine, "_configure", lambda pipe, job: None)
    monkeypatch.setattr(engine, "_abc_streamer", lambda pipe, job: None)
    monkeypatch.setattr(engine, "_ensure_pipeline", lambda: FakePipe(LONG_ABC))
    seen = []
    engine.listeners.append(seen.append)
    try:
        job = _job(engine, make_job, max_seconds=60)
        engine._run(job)
    finally:
        engine.listeners.remove(seen.append)
    assert job.status == "failed" and "trop longue" in job.error and seen == [job]
    assert not (job.output_dir / "traceback.txt").exists()


# ---------------------------------------------------------------- raccourcissement de la partition
SECTIONED_ABC = ("X:1\nM:4/4\nL:1/32\nQ:1/4=120\nV: Vocal\nV: Ins\nK:C\n"
                 "% intro\nV: Vocal\nz32|z32|z32|z32|\nV: Ins\nC8G8e8d8|C8G8e8d8|C8G8e8d8|C8G8e8d8|\n"      # 4 mesures = 8 s
                 "% verse\nV: Vocal\n" + "z32|" * 8 + "\nV: Ins\n" + "C8G8e8d8|" * 8 + "\n"             # 8 mesures = 16 s
                 "% chorus\nV: Vocal\n" + "z32|" * 8 + "\nV: Ins\n" + "C8G8e8d8|" * 8 + "\n"            # 8 mesures = 16 s
                 "% outro\nV: Vocal\nz32|z32|\nV: Ins\nC8G8e8d8|C8G8e8d8|\n")                           # 2 mesures = 4 s


def test_split_sections_keeps_header_and_names():
    header, sections = guard.split_sections(SECTIONED_ABC)
    assert header.startswith("X:1") and "K:C" in header and "% intro" not in header
    assert [n for n, _ in sections] == ["intro", "verse", "chorus", "outro"]
    assert guard.split_sections("")[1] == []


def test_trim_abc_keeps_whole_sections_and_restores_outro():
    text, info = guard.trim_abc(SECTIONED_ABC, 30)                     # intro+verse = 24 s ; + outro 4 s = 28 s ≤ 30
    assert info["kept"] == ["intro", "verse", "outro"] and info["dropped"] == ["chorus"]
    assert info["bars_before"] == 22 and info["bars_after"] == 14
    assert text.startswith("X:1") and "% chorus" not in text and text.rstrip().endswith("C8G8e8d8|C8G8e8d8|")
    assert guard.abc_duration(text)["seconds"] == pytest.approx(28.0)
    note = guard.trim_note(info)
    assert "22 mesures" in note and "14 mesures" in note and "score_full.abc" in note


def test_trim_abc_swaps_last_section_for_outro_when_needed():
    text, info = guard.trim_abc(SECTIONED_ABC, 26)                     # intro+verse = 24 s, outro ne tient plus → remplace verse
    assert info["kept"] == ["intro", "outro"] and info["dropped"] == ["verse", "chorus"]
    assert guard.abc_duration(text)["seconds"] == pytest.approx(12.0)


def test_trim_abc_refuses_when_nothing_to_gain():
    assert guard.trim_abc(SECTIONED_ABC, 600) is None                   # tout tient déjà
    assert guard.trim_abc(SHORT_ABC, 2) is None                         # pas de sections
    assert guard.trim_abc(SECTIONED_ABC, None) is None
    text, info = guard.trim_abc(SECTIONED_ABC, 3)                       # la première section est toujours gardée
    assert info["kept"] == ["intro"]


def test_should_trim_modes():
    assert guard.should_trim("auto", "[Intro]\n\n[Outro]") is True
    assert guard.should_trim(None, "[Verse]\nla la") is False
    assert guard.should_trim("trim", "[Verse]\nla la") is True
    assert guard.should_trim("stop", "[Intro]") is False


class _Tok:
    def encode(self, text):
        return [(ord(c) % 100) + 1 for c in text]


class TrimmingPipe(FakePipe):
    tokenizer = _Tok()


def test_run_song_trims_instrumental_plan_and_continues(engine, make_job, monkeypatch):
    monkeypatch.setattr(engine, "_configure", lambda pipe, job: None)
    monkeypatch.setattr(engine, "_abc_streamer", lambda pipe, job: None)
    pipe = TrimmingPipe(SECTIONED_ABC)                                  # 44 s planifiées
    job = _job(engine, make_job, max_seconds=30)                        # 30 × 1.3 = 39 < 44 → trop longue
    job.request["lyrics"] = "[Intro]\n\n[Outro]"                        # instrumental → auto = raccourcir
    engine._run_song(pipe, job)
    assert pipe.calls[-1] == "decode" and job.summary["has_audio"]
    assert job.summary["trim"]["kept"] == ["intro", "verse", "outro"] and "raccourcie" in job.summary["trim_note"]
    assert job.summary["planned"]["bars"] == 14 and job.summary["guard"]["reason"] == "durée maximale"   # 825 < 975
    assert (job.output_dir / "score_full.abc").read_text() == SECTIONED_ABC
    assert "% chorus" not in (job.output_dir / "score.abc").read_text()
    assert job.summary["timing"]["abc"]["trimmed_from_tokens"] == 1


def test_run_song_stops_for_sung_lyrics_in_auto_mode(engine, make_job, monkeypatch):
    monkeypatch.setattr(engine, "_configure", lambda pipe, job: None)
    monkeypatch.setattr(engine, "_abc_streamer", lambda pipe, job: None)
    pipe = TrimmingPipe(SECTIONED_ABC)
    job = _job(engine, make_job, max_seconds=30)                        # paroles chantées par défaut
    with pytest.raises(guard.PlanTooLong, match="raccourcir la partition"):
        engine._run_song(pipe, job)
    assert not (job.output_dir / "score_full.abc").exists()
    job.plan_overflow = "trim"
    engine._run_song(pipe, job)
    assert job.summary["has_audio"] and job.summary["trim"]["dropped"] == ["chorus"]
