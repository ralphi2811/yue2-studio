"""Assistant : heuristiques, normalisation, résumé de projet, tours de conversation (LLM simulé)."""
import json

import pytest

from studio import assistant as A
from studio import llm


# ---------------------------------------------------------------- durée / bpm
@pytest.mark.parametrize("style, bpm", [("French, pop, 88 BPM", 88), ("rock 120bpm", 120), ("no tempo", None), ("999 BPM", None)])
def test_parse_bpm(style, bpm):
    assert A.parse_bpm(style) == bpm


def test_estimate_duration_calibration():
    lyrics = "[Verse]\n" + "\n".join(f"line {i}" for i in range(4)) + "\n\n[Chorus]\n" + "\n".join(f"c {i}" for i in range(4))
    est = A.estimate_duration(lyrics, "pop, 83 BPM")
    assert est["sung_lines"] == 8 and est["bars"] == 9 + 16 and est["bpm"] == 83
    assert est["seconds"] == pytest.approx(25 * 4 * 60 / 83)
    est2 = A.estimate_duration("[Intro]\n" + lyrics + "\n[Outro]", "pop, 83 BPM")
    assert est2["bars"] == 25 + 8


# ---------------------------------------------------------------- instrumental
def test_normalize_proposal_instrumental_forces_full_and_reaches_the_form():
    prop, warnings = A.normalize_proposal({"name": "instru", "style": "dark ambient, 70 BPM", "lyrics": "[intro]\n[verse]\n[outro]",
                                           "cot": "melody", "instrumental": "true", "target_duration_seconds": 60})
    assert prop["instrumental"] is True and prop["cot"] == "full"
    assert any("remplacé par full" in w for w in warnings) and any("expérimental" in w for w in warnings)
    assert A.proposal_to_form(prop)["instrumental"] is True
    prop2, warnings2 = A.normalize_proposal({"name": "x", "style": "pop, 90 BPM", "lyrics": "[Intro]\n[Outro]", "cot": "full"})
    assert prop2["instrumental"] is False and any("cochez « Instrumental »" in w for w in warnings2)
    assert A.proposal_to_form(prop2)["instrumental"] is False


# ---------------------------------------------------------------- réglages LLM et environnement
def test_llm_settings_env_overrides_saved_values(monkeypatch):
    s = llm.LLMSettings(base_url="http://saved/v1/", model="saved-model", api_key="saved-key")
    assert s.effective_base_url == "http://saved/v1" and s.effective_model == "saved-model" and s.ready
    assert not (s.base_url_from_env or s.model_from_env or s.key_from_env)
    monkeypatch.setenv("YUE2_LLM_BASE_URL", "http://host.docker.internal:11434/v1/ ")
    monkeypatch.setenv("YUE2_LLM_MODEL", " qwen3:32b ")
    monkeypatch.setenv("YUE2_LLM_API_KEY", "env-key")
    assert s.effective_base_url == "http://host.docker.internal:11434/v1" and s.base_url_from_env
    assert s.effective_model == "qwen3:32b" and s.model_from_env
    assert s.effective_key == "env-key" and s.key_from_env
    pub = s.public()
    assert pub["base_url"] == "http://host.docker.internal:11434/v1" and pub["model"] == "qwen3:32b" and pub["ready"]
    assert pub["base_url_from_env"] and pub["model_from_env"] and pub["api_key"] == "" and pub["has_key"]
    # une variable vide ne verrouille rien et n'écrase pas la valeur enregistrée
    monkeypatch.setenv("YUE2_LLM_MODEL", "")
    assert s.effective_model == "saved-model" and not s.model_from_env


def test_llm_settings_ready_from_env_alone(monkeypatch):
    s = llm.LLMSettings(base_url="", model="")
    assert not s.ready
    monkeypatch.setenv("YUE2_LLM_BASE_URL", "http://localhost:1234/v1")
    assert not s.ready
    monkeypatch.setenv("YUE2_LLM_MODEL", "local")
    assert s.ready


@pytest.mark.anyio
async def test_chat_retries_once_when_cut_by_max_tokens(monkeypatch):
    """Une réponse coupée (paroles longues, modèle qui raisonne) est relancée une fois avec plus de budget,
    au lieu de remonter un « JSON incomplet » incompréhensible."""
    import httpx
    budgets = []

    async def post(self, url, headers=None, json=None):
        budgets.append(json["max_tokens"])
        if len(budgets) == 1:
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": '{"type": "prop'}}],
                                             "usage": {"prompt_tokens": 100, "completion_tokens": 900}})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 50}})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    s = llm.LLMSettings(base_url="http://h/v1", model="m", max_tokens=2000)
    content, usage = await llm.chat([{"role": "user", "content": "x"}], s=s)
    assert content == '{"ok": true}' and budgets == [2000, 8000]
    assert usage == {"prompt_tokens": 200, "completion_tokens": 950}     # les deux appels sont comptés


@pytest.mark.anyio
async def test_chat_says_what_to_change_when_truncation_persists(monkeypatch):
    import httpx
    calls = []

    async def post(self, url, headers=None, json=None):
        calls.append(json["max_tokens"])
        return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "{"}}], "usage": {}})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    s = llm.LLMSettings(base_url="http://h/v1", model="m", max_tokens=8000)
    with pytest.raises(llm.LLMError, match="max_tokens"):
        await llm.chat([{"role": "user", "content": "x"}], s=s)
    assert calls == [8000, llm.MAX_TOKENS_CEILING]      # un seul nouvel essai, puis on explique


@pytest.mark.anyio
async def test_chat_uses_env_base_url_and_model(monkeypatch):
    import httpx
    seen = {}

    async def post(self, url, headers=None, json=None):
        seen["url"] = url
        seen["model"] = json["model"]
        seen["auth"] = headers.get("Authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"ok\": true}"}}], "usage": {}})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setenv("YUE2_LLM_BASE_URL", "http://env-host/v1")
    monkeypatch.setenv("YUE2_LLM_MODEL", "env-model")
    s = llm.LLMSettings(base_url="http://saved/v1", model="saved", api_key="k")
    content, _ = await llm.chat([{"role": "user", "content": "x"}], s=s)
    assert content == '{"ok": true}'
    assert seen == {"url": "http://env-host/v1/chat/completions", "model": "env-model", "auth": "Bearer k"}


# ---------------------------------------------------------------- extraction JSON
@pytest.mark.parametrize("text", [
    '{"a": 1}',
    'Voici :\n```json\n{"a": 1}\n```\nmerci',
    'blabla {"a": 1, "s": "avec } accolade"} suite',
])
def test_extract_json_variants(text):
    assert llm.extract_json(text)["a"] == 1


def test_extract_json_errors():
    with pytest.raises(llm.LLMError):
        llm.extract_json("pas de json")
    with pytest.raises(llm.LLMError):
        llm.extract_json('{"a": ')


# ---------------------------------------------------------------- normalisation
def test_normalize_proposal_clamps_and_warns():
    raw = {"name": "Nuit néon !", "style": "French, pop, 400 BPM  ", "lyrics": "[Verse]\r\nun\r\ndeux", "cot": "weird",
           "cfg_scale": 99, "seed": -5, "ode_steps": 1000, "semantic_sampling": {"max_tokens": 50000, "temperature": 1.0},
           "abc_sampling": None, "target_duration_seconds": 300, "rationale": {"style": 1}}
    prop, warnings = A.normalize_proposal(raw)
    assert prop["name"] == "Nuit_n_on" and prop["cot"] == "full" and prop["cfg_scale"] == 20 and prop["seed"] == 0
    assert prop["ode_steps"] == 128 and prop["semantic_sampling"] == {"max_tokens": 20000} and prop["abc_sampling"] == {}
    assert prop["rationale"] == {"style": "1"}
    assert any("cot inconnu" in w for w in warnings) and any("cible" in w for w in warnings)
    assert any("BPM" in w for w in warnings)      # 400 BPM hors plage → pas de BPM détecté
    assert not any("runtime" in w for w in warnings)


def test_normalize_proposal_max_duration_and_instrumental():
    prop, warnings = A.normalize_proposal({"name": "i", "style": "Instrumental, trip-hop, 74 BPM", "lyrics": "[Intro]\n\n[Outro]",
                                           "cot": "full", "target_duration_seconds": 120})
    assert prop["max_duration_seconds"] == 168                     # cible × 1.4 par défaut
    assert any("aucune ligne chantée" in w for w in warnings)
    prop, _ = A.normalize_proposal({"name": "i", "style": "S 90 BPM", "lyrics": "[Verse]\nx", "cot": "full", "max_duration_seconds": 45})
    assert prop["max_duration_seconds"] == 45 and A.proposal_to_form(prop)["max_duration"] == 45


def test_normalize_proposal_clean_has_no_warnings():
    prop, warnings = A.normalize_proposal({"name": "ok", "style": "French, pop, 90 BPM", "lyrics": "[Verse]\nun\ndeux\ntrois\nquatre",
                                           "cot": "melody", "cfg_scale": None, "target_duration_seconds": 45})
    assert warnings == [] and prop["estimated_bars"] == 17 and prop["cfg_scale"] is None


def test_proposal_to_form_keeps_project_seed(store):
    p = store.create("p", seed=4242)
    prop, _ = A.normalize_proposal({"name": "n", "style": "S 90 BPM", "lyrics": "l", "cot": "full", "seed": None,
                                    "semantic_sampling": {"max_tokens": 700}})
    values = A.proposal_to_form(prop, p)
    assert values["seed"] == 4242 and values["project_id"] == p.id and values["semantic.max_tokens"] == 700
    prop2, _ = A.normalize_proposal({"name": "n", "style": "S", "lyrics": "l", "cot": "full", "seed": 7})
    assert A.proposal_to_form(prop2, p)["seed"] == 7
    assert "project_id" not in A.proposal_to_form(prop2)


def test_proposal_matches_request():
    prop = {"style": "A  B", "lyrics": "x\ny", "cot": "full"}
    assert A.proposal_matches_request(prop, {"style": "A B", "lyrics": "x\r\ny", "cot": "full"})
    assert not A.proposal_matches_request(prop, {"style": "A B", "lyrics": "x\nz", "cot": "full"})
    assert not A.proposal_matches_request(None, {})


# ---------------------------------------------------------------- résumé de projet
class _J:
    def __init__(self, request, ode_steps=32, semantic_sampling=None, abc_sampling=None):
        self.request, self.ode_steps = request, ode_steps
        self.semantic_sampling, self.abc_sampling = semantic_sampling or {}, abc_sampling or {}


def test_project_summary_lists_versions_changes_and_reference(store):
    p = store.create("Ville", seed=1)
    store.add_version(p, "j1", source="import")
    store.add_version(p, "j2", source="assistant")
    p.version(1).outcome = {"status": "done", "audio_seconds": 64.2, "planned_bars": 25, "planned_bpm": 93, "truncated": {}}
    jobs = {"j1": _J({"style": "S1 90 BPM", "lyrics": "[Verse]\na\nb", "cot": "full", "seed": 1}),
            "j2": _J({"style": "S2 90 BPM", "lyrics": "[Verse]\na\nc", "cot": "full", "seed": 1})}
    text = A.project_summary(p, jobs)
    assert "Graine du projet : 1" in text
    assert "v1 (import" in text and "durée réelle 64 s" in text and "25 mesures planifiées à 93 BPM" in text
    assert "v2 (assistant" in text and "style modifié" in text and "paroles : +1 / -1" in text and "non terminée" in text
    assert "Dernière version générée (v2)" in text and "style : S2 90 BPM" in text and text.rstrip().endswith("c")
    assert A.project_summary(store.create("vide"), jobs) == ""


# ---------------------------------------------------------------- tours de conversation avec LLM simulé
PROPOSAL = {"type": "proposal", "message": "Voici.", "proposal": {"name": "nuit_neon", "style": "French, indie pop, 84 BPM",
            "lyrics": "[Verse]\na\nb\n\n[Chorus]\nc\nd", "cot": "full", "cfg_scale": None, "seed": None, "target_duration_seconds": 60}}


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    async def chat(messages, **kw):
        calls.append(messages)
        users = [m for m in messages if m["role"] == "user"]
        if len(users) == 1:
            return json.dumps({"type": "questions", "message": "1. Langue ?\n2. Thème ?", "proposal": None}), {"prompt_tokens": 10, "completion_tokens": 5}
        return json.dumps(PROPOSAL, ensure_ascii=False), {"prompt_tokens": 20, "completion_tokens": 8}

    monkeypatch.setattr(llm, "chat", chat)
    return calls


@pytest.mark.anyio
async def test_step_questions_then_proposal(store, fake_llm, monkeypatch):
    monkeypatch.setattr(A, "_STORE", store)
    p = store.create()
    await A.step(p, "Une chanson triste", context={"style": "ctx style", "lyrics": ""}, jobs={})
    assert p.turns[-1]["role"] == "assistant" and "Langue" in p.turns[-1]["text"] and p.proposal is None
    assert "ctx style" in p.messages[0]["content"]          # contexte du formulaire injecté au premier tour
    assert fake_llm[0][0]["role"] == "system" and "Paramètres du runtime" in fake_llm[0][0]["content"]
    await A.step(p, "Français, la nuit", jobs={})
    assert p.proposal["name"] == "nuit_neon" and p.name == "nuit_neon"
    assert p.turns[-1]["proposal"]["estimated_bars"] == 9 + 8
    assert p.usage == {"prompt_tokens": 30, "completion_tokens": 13}
    assert store.get(p.id).proposal["name"] == "nuit_neon"   # persisté


@pytest.mark.anyio
async def test_step_includes_project_summary_as_second_system_message(store, fake_llm, monkeypatch):
    monkeypatch.setattr(A, "_STORE", store)
    p = store.create("Ville")
    store.add_version(p, "j1", source="import")
    jobs = {"j1": _J({"style": "S1 90 BPM", "lyrics": "[Verse]\na", "cot": "full", "seed": 1})}
    await A.step(p, "Plus court", jobs=jobs)
    systems = [m for m in fake_llm[-1] if m["role"] == "system"]
    assert len(systems) == 2 and "État du projet" in systems[1]["content"]


@pytest.mark.anyio
async def test_step_records_llm_error(store, monkeypatch):
    monkeypatch.setattr(A, "_STORE", store)

    async def boom(messages, **kw):
        raise llm.LLMError("HTTP 401 : bad key")

    monkeypatch.setattr(llm, "chat", boom)
    p = store.create()
    await A.step(p, "hello", jobs={})
    assert p.turns[-1]["error"].startswith("HTTP 401") and p.proposal is None
    assert len(p.messages) == 1              # le message utilisateur reste, pas de réponse fantôme


@pytest.mark.anyio
async def test_quick_actions(monkeypatch):
    async def chat(messages, **kw):
        prompt = messages[-1]["content"]
        if "Prompt actuel" in prompt:
            return json.dumps({"style": "  French,  pop, 90 BPM ", "why": "ok"}), {}
        return json.dumps({"lyrics": "[Verse]\r\nx", "why": "ok"}), {}

    monkeypatch.setattr(llm, "chat", chat)
    assert (await A.quick("style", {"style": "pop"}))["value"] == "French, pop, 90 BPM"
    assert (await A.quick("lyrics", {"lyrics": "l", "instruction": "court"}))["value"] == "[Verse]\nx"
    with pytest.raises(llm.LLMError):
        await A.quick("nope", {})


def test_create_from_job_seeds_project(store, make_job, monkeypatch):
    monkeypatch.setattr(A, "_STORE", store)
    job = make_job("origine", seed=777)
    p = A.create_from_job(job)
    assert p.seed == 777 and p.versions[0].job_id == job.id and p.versions[0].source == "import"
    assert p.versions[0].outcome["audio_seconds"] == 46.0
    assert p.turns[0]["role"] == "note" and p.turns[1]["role"] == "assistant"
    assert store.for_job(job.id)[0].id == p.id
