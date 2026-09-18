"""Connecteur LLM OpenAI-compatible (OpenRouter, OpenAI, Mistral, Ollama, LM Studio, vLLM…).

Un seul protocole : ``POST {base_url}/chat/completions``. La clé, l'URL de base et le modèle
viennent des variables d'environnement ``YUE2_LLM_API_KEY``, ``YUE2_LLM_BASE_URL`` et
``YUE2_LLM_MODEL`` si elles sont définies (typiquement via le ``.env``), sinon du fichier de
réglages local ``studio/data/assistant.json`` (hors git). Une valeur venue de l'environnement
verrouille le champ correspondant dans l'interface.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx

from .paths import DATA_DIR

SETTINGS_FILE = DATA_DIR / "assistant.json"
ENV_KEY = "YUE2_LLM_API_KEY"
ENV_BASE_URL = "YUE2_LLM_BASE_URL"
ENV_MODEL = "YUE2_LLM_MODEL"

PRESETS = {
    "openrouter": ("https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.5"),
    "openai": ("https://api.openai.com/v1", "gpt-5"),
    "mistral": ("https://api.mistral.ai/v1", "mistral-large-latest"),
    "ollama": ("http://localhost:11434/v1", "qwen3:32b"),
    "lmstudio": ("http://localhost:1234/v1", ""),
}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMSettings:
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = ""
    api_key: str = ""
    temperature: float = 0.8
    max_tokens: int = 8000    # une proposition complète (paroles + justifications) tient rarement en moins
    timeout: float = 120.0

    @property
    def effective_key(self) -> str:
        return os.environ.get(ENV_KEY) or self.api_key or ""

    @property
    def key_from_env(self) -> bool:
        return bool(os.environ.get(ENV_KEY))

    @property
    def effective_base_url(self) -> str:
        return (os.environ.get(ENV_BASE_URL) or self.base_url or "").strip().rstrip("/")

    @property
    def base_url_from_env(self) -> bool:
        return bool(os.environ.get(ENV_BASE_URL, "").strip())

    @property
    def effective_model(self) -> str:
        return (os.environ.get(ENV_MODEL) or self.model or "").strip()

    @property
    def model_from_env(self) -> bool:
        return bool(os.environ.get(ENV_MODEL, "").strip())

    @property
    def ready(self) -> bool:
        return bool(self.effective_base_url and self.effective_model)

    @classmethod
    def load(cls) -> "LLMSettings":
        if SETTINGS_FILE.is_file():
            try:
                data = json.loads(SETTINGS_FILE.read_text())
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()

    def save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
        try:
            SETTINGS_FILE.chmod(0o600)
        except OSError:
            pass

    def public(self) -> dict:
        d = asdict(self)
        d["api_key"] = ""
        d["has_key"] = bool(self.effective_key)
        d["key_from_env"] = self.key_from_env
        d["base_url"] = self.effective_base_url
        d["base_url_from_env"] = self.base_url_from_env
        d["model"] = self.effective_model
        d["model_from_env"] = self.model_from_env
        d["ready"] = self.ready
        return d


_SETTINGS: LLMSettings | None = None


def get_settings() -> LLMSettings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = LLMSettings.load()
    return _SETTINGS


def set_settings(settings: LLMSettings):
    global _SETTINGS
    _SETTINGS = settings
    settings.save()


def _headers(s: LLMSettings) -> dict:
    headers = {"Content-Type": "application/json", "X-Title": "YuE2 Studio", "HTTP-Referer": "http://localhost:8420"}
    if s.effective_key:
        headers["Authorization"] = f"Bearer {s.effective_key}"
    return headers


async def list_models(s: LLMSettings | None = None) -> list[str]:
    s = s or get_settings()
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(s.effective_base_url + "/models", headers=_headers(s))
    if r.status_code >= 400:
        raise LLMError(f"HTTP {r.status_code} sur /models : {r.text[:300]}")
    data = r.json()
    items = data.get("data", data if isinstance(data, list) else [])
    return [m.get("id") or m.get("name") for m in items if isinstance(m, dict)][:400]


def extract_json(text: str) -> dict:
    """Récupère le premier objet JSON d'une réponse (tolère les blocs ``` et le texte autour)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise LLMError("La réponse ne contient pas de JSON")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise LLMError("JSON incomplet dans la réponse")


MAX_TOKENS_CEILING = 24000


async def _post(s: LLMSettings, messages: list[dict], json_mode: bool, temperature: float | None, budget: int) -> dict:
    payload = {"model": s.effective_model, "messages": messages,
               "temperature": s.temperature if temperature is None else temperature,
               "max_tokens": budget}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    url = s.effective_base_url + "/chat/completions"
    async with httpx.AsyncClient(timeout=s.timeout) as client:
        r = await client.post(url, headers=_headers(s), json=payload)
        if r.status_code == 400 and json_mode:
            payload.pop("response_format", None)
            r = await client.post(url, headers=_headers(s), json=payload)
    if r.status_code >= 400:
        detail = r.text[:400]
        try:
            detail = r.json().get("error", {}).get("message", detail)
        except Exception:
            pass
        raise LLMError(f"HTTP {r.status_code} : {detail}")
    return r.json()


def _content(data: dict) -> str:
    try:
        content = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Réponse inattendue du fournisseur : {str(data)[:300]}") from exc
    if isinstance(content, list):   # certains fournisseurs renvoient des blocs
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


def _truncated(data: dict) -> bool:
    """Le modèle a-t-il été coupé par max_tokens ? (``length`` chez OpenAI, ``MAX_TOKENS`` chez d'autres)"""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return False
    reason = (choice.get("finish_reason") or choice.get("native_finish_reason") or "")
    return str(reason).lower() in {"length", "max_tokens"}


def _add_usage(total: dict, more: dict) -> dict:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if more.get(key):
            total[key] = total.get(key, 0) + int(more[key])
    return total


async def chat(messages: list[dict], *, json_mode: bool = True, s: LLMSettings | None = None,
               temperature: float | None = None, max_tokens: int | None = None) -> tuple[str, dict]:
    """Appelle /chat/completions. Retourne (texte, usage cumulé). Repli automatique si json_object est refusé.

    Une réponse coupée par ``max_tokens`` (paroles longues, modèle qui raisonne avant d'écrire) est relancée
    UNE fois avec le double du budget, plafonné. Sinon l'erreur dit quoi régler plutôt que « JSON incomplet ».
    """
    s = s or get_settings()
    if not s.ready:
        raise LLMError("Assistant non configuré : renseignez l'URL et le modèle dans ⚙︎ Moteur → Assistant LLM.")
    budget = int(s.max_tokens if max_tokens is None else max_tokens)
    usage: dict = {}
    for attempt in (1, 2):      # un seul nouvel essai : chaque appel coûte des jetons
        data = await _post(s, messages, json_mode, temperature, budget)
        _add_usage(usage, data.get("usage") or {})
        content = _content(data)
        if not _truncated(data):
            if not content.strip():
                raise LLMError("Réponse vide du modèle (modèle de raisonnement sans sortie, ou filtre du fournisseur ?)")
            return content, usage
        if attempt == 2 or budget >= MAX_TOKENS_CEILING:
            raise LLMError(f"Réponse coupée à {budget} jetons de sortie. Augmentez « max_tokens » dans "
                           f"⚙︎ Moteur → Assistant LLM, ou demandez un morceau plus court.")
        budget = min(max(budget * 4, 8000), MAX_TOKENS_CEILING)
    raise LLMError("Réponse coupée par le fournisseur.")   # inatteignable
