"""Parsing et validation des formulaires, import par lot, durée ABC, renommage sûr."""
import pytest

from studio import app as A
from studio import params as P
from studio.engine import slugify


# ---------------------------------------------------------------- _coerce
def test_coerce_numbers_and_bounds():
    seed = P.ALL_PARAMS["seed"]
    assert A._coerce(seed, "42") == 42
    assert A._coerce(seed, "") == seed.default
    with pytest.raises(A.FormError):
        A._coerce(seed, "-1")
    with pytest.raises(A.FormError):
        A._coerce(seed, "abc")
    temp = P.ALL_PARAMS["semantic.temperature"]
    assert A._coerce(temp, "0,5") == 0.5          # virgule décimale acceptée
    with pytest.raises(A.FormError):
        A._coerce(temp, "9")


def test_coerce_nullable_select_bool_text():
    assert A._coerce(P.ALL_PARAMS["cfg_scale"], "") is None
    assert A._coerce(P.ALL_PARAMS["cfg_scale"], "1.2") == 1.2
    assert A._coerce(P.ALL_PARAMS["cot"], "melody") == "melody"
    with pytest.raises(A.FormError):
        A._coerce(P.ALL_PARAMS["cot"], "banana")
    assert A._coerce(P.ALL_PARAMS["offload_ar"], None) is False
    assert A._coerce(P.ALL_PARAMS["offload_ar"], "on") is True
    assert A._coerce(P.ALL_PARAMS["style"], "") == ""     # texte vide ≠ défaut


# ---------------------------------------------------------------- parse_job_form
BASE = {"name": "Mon titre", "style": "French, chanson, 80 BPM", "lyrics": "[Verse]\nla la\nli li", "cot": "full",
        "seed": "7", "kind": "song"}


def test_parse_job_form_minimal():
    job = A.parse_job_form(dict(BASE))
    assert job.kind == "song"
    assert job.request["id"] == "Mon_titre"
    assert job.request["seed"] == 7
    assert "cfg_scale" not in job.request and "abc" not in job.request
    assert job.abc_sampling == {} and job.semantic_sampling == {}
    assert job.ode_steps == 32
    assert job.project_id is None


def test_parse_job_form_sampling_overrides_only_non_default():
    form = dict(BASE, **{"semantic.max_tokens": "1200", "semantic.temperature": "1.0", "abc.top_k": "30", "ode_steps": "16"})
    job = A.parse_job_form(form)
    assert job.semantic_sampling == {"max_tokens": 1200}
    assert job.abc_sampling == {}
    assert job.ode_steps == 16


def test_parse_job_form_rejections():
    with pytest.raises(A.FormError, match="obligatoires"):
        A.parse_job_form(dict(BASE, lyrics=""))
    with pytest.raises(A.FormError, match="full ou melody"):
        A.parse_job_form(dict(BASE, cot="off", abc="X:1\nK:C\nC4|"))
    with pytest.raises(A.FormError, match="Planifier"):
        A.parse_job_form(dict(BASE, kind="plan", cot="off"))
    with pytest.raises(A.FormError, match="runtime"):
        A.parse_job_form(dict(BASE, **{"semantic.min_tokens": "5000", "semantic.max_tokens": "100"}))


def test_parse_job_form_abc_and_cfg():
    job = A.parse_job_form(dict(BASE, abc="X:1\r\nK:C\r\nC4|", cfg_scale="1.2"))
    assert job.request["abc"] == "X:1\nK:C\nC4|\n"
    assert job.request["cfg_scale"] == 1.2


def test_parse_job_form_max_duration():
    assert A.parse_job_form(dict(BASE)).max_seconds is None
    assert A.parse_job_form(dict(BASE, max_duration="90")).max_seconds == 90
    with pytest.raises(A.FormError):
        A.parse_job_form(dict(BASE, max_duration="2"))


def test_parse_job_form_unknown_project_is_dropped():
    job = A.parse_job_form(dict(BASE, project_id="nope"))
    assert job.project_id is None


# ---------------------------------------------------------------- settings
def test_parse_settings_form_custom_vae_requires_value():
    base = {p.key: (p.default if p.kind != "bool" else "") for p in P.ENGINE}
    base = {k: ("" if v is None else str(v)) for k, v in base.items()}
    s = A.parse_settings_form(dict(base, vae="legacy"))
    assert s.vae_repo == "m-a-p/YuE2-Vae-legacy"
    with pytest.raises(A.FormError, match="personnalisé"):
        A.parse_settings_form(dict(base, vae="custom", vae_custom=""))
    s = A.parse_settings_form(dict(base, vae="custom", vae_custom="/tmp/vae"))
    assert s.vae_repo == "/tmp/vae" and s.resolve_vae("standard") == "m-a-p/YuE2-Vae"


# ---------------------------------------------------------------- slugify / abc_duration
@pytest.mark.parametrize("name, expected", [("Mon titre é", "Mon_titre"), ("", "song"), ("...", "song"), ("ok-1.2_x", "ok-1.2_x")])
def test_slugify(name, expected):
    assert slugify(name) == expected


def test_abc_duration_counts_vocal_bars_and_multibar_rests(score_abc):
    d = A.abc_duration(score_abc)
    assert d["bars"] == 12 and d["bpm"] == 83 and d["meter"] == "4/4"
    assert d["seconds"] == pytest.approx(12 * 4 * 60 / 83)


def test_abc_duration_other_meter_and_beat_unit():
    d = A.abc_duration("X:1\nM:3/4\nQ:1/8=120\nK:C\nV: Vocal\nC2D2E2|Z2|F6|")
    assert d["bars"] == 4 and d["seconds"] == pytest.approx(12.0)
    assert A.abc_duration("") is None


# ---------------------------------------------------------------- import par lot
def test_rows_from_upload_formats():
    assert A._rows_from_upload(b'{"name":"a"}\n{"name":"b"}\n', "x.jsonl") == [{"name": "a"}, {"name": "b"}]
    assert A._rows_from_upload(b'[{"name":"a"}]', "x.json") == [{"name": "a"}]
    assert A._rows_from_upload(b'{"requests":[{"name":"a"}]}', "x.json") == [{"name": "a"}]
    rows = A._rows_from_upload("name,lyrics\nc,\"[Verse]\\nla\\nli\"\n".encode("utf-8-sig"), "x.csv")
    assert rows == [{"name": "c", "lyrics": "[Verse]\nla\nli"}]
    assert A._rows_from_upload(b"   ", "x.jsonl") == []


def test_row_to_form_inherits_defaults_and_flattens_sampling():
    d = A._row_to_form({"id": "x", "tags": "S", "semantic_sampling": {"max_tokens": 500}}, {"style": "def", "lyrics": "L", "kind": "song"})
    assert d["name"] == "x" and d["style"] == "S" and d["lyrics"] == "L" and d["semantic.max_tokens"] == "500"
    with pytest.raises(A.FormError, match="inconnus"):
        A._row_to_form({"bpm": 90}, {})
    with pytest.raises(A.FormError):
        A._row_to_form({"abc_sampling": "oops"}, {})


def test_parse_job_form_plan_overflow():
    assert A.parse_job_form(dict(BASE)).plan_overflow == "auto"
    assert A.parse_job_form(dict(BASE, plan_overflow="trim")).plan_overflow == "trim"
    with pytest.raises(A.FormError):
        A.parse_job_form(dict(BASE, plan_overflow="maybe"))
    assert A._row_to_form({"plan_overflow": "stop"}, {})["plan_overflow"] == "stop"


# ---------------------------------------------------------------- .env
def test_load_env_file_sets_missing_variables_only(tmp_path, monkeypatch):
    from studio.paths import load_env_file
    env = tmp_path / ".env"
    env.write_text('# commentaire\nYUE2_TEST_A=un\nexport YUE2_TEST_B="deux mots"\nYUE2_TEST_C=\'trois\'\nSANS_EGAL\n\nYUE2_TEST_D=\n', encoding="utf-8")
    for k in ("YUE2_TEST_A", "YUE2_TEST_B", "YUE2_TEST_C", "YUE2_TEST_D"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("YUE2_TEST_A", "déjà là")
    loaded = load_env_file(env)
    assert loaded == ["YUE2_TEST_B", "YUE2_TEST_C", "YUE2_TEST_D"]
    import os
    assert os.environ["YUE2_TEST_A"] == "déjà là" and os.environ["YUE2_TEST_B"] == "deux mots"
    assert os.environ["YUE2_TEST_C"] == "trois" and os.environ["YUE2_TEST_D"] == ""
    assert load_env_file(env, override=True) == ["YUE2_TEST_A", "YUE2_TEST_B", "YUE2_TEST_C", "YUE2_TEST_D"]
    assert os.environ["YUE2_TEST_A"] == "un"
    assert load_env_file(tmp_path / "absent.env") == []
