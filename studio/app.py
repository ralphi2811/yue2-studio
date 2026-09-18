"""YuE2 Studio — interface FastAPI + HTMX pour piloter le pipeline YuE2 en local.

Lancement : ``.venv/bin/uvicorn studio.app:app --port 8420`` depuis la racine du dépôt,
ou ``.venv/bin/python -m studio``.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import abctools
from . import assistant as A
from . import llm
from . import params as P
from . import projects as PR
from .engine import EngineSettings, Job, get_engine, slugify

HERE = Path(__file__).resolve().parent
app = FastAPI(title="YuE2 Studio", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


# --------------------------------------------------------------------------
# Filtres / helpers Jinja
# --------------------------------------------------------------------------
def fmt_duration(seconds) -> str:
    if seconds is None:
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s"


def fmt_rate(rate, unit) -> str:
    if not rate:
        return ""
    return f"{rate:.1f} {unit or 'it'}/s"


templates.env.filters["duration"] = fmt_duration
templates.env.filters["rate"] = fmt_rate
templates.env.filters["hash_hue"] = lambda value: int(__import__("hashlib").md5(str(value).encode()).hexdigest()[:4], 16) % 360
def static_version() -> str:
    """Empreinte des statiques (mtime max) : toute modification change l'URL et contourne le cache."""
    return str(int(max(p.stat().st_mtime for p in (HERE / "static").rglob("*") if p.is_file())))


templates.env.globals.update(P=P, LYRIC_TAGS=P.LYRIC_TAGS, STYLE_PRESETS=P.STYLE_PRESETS, v=static_version,
                             llm_ready=lambda: llm.get_settings().ready,
                             project_of=lambda pid: A.store().get(pid) if pid else None)


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    elif "text/html" in response.headers.get("content-type", ""):
        response.headers["Cache-Control"] = "no-store"
    return response


def render(name: str, request: Request | None = None, **ctx) -> str:
    ctx.setdefault("request", request)
    return templates.get_template(name).render(**ctx)


def engine_ctx() -> dict:
    engine = get_engine()
    snap = engine.snapshot()
    snap["gpu"] = engine.gpu_status()
    snap["engine"] = engine
    return snap


# --------------------------------------------------------------------------
# Parsing du formulaire
# --------------------------------------------------------------------------
class FormError(Exception):
    pass


def _coerce(param: P.Param, raw: str | None):
    if raw is None or raw.strip() == "":
        if param.kind == "bool":
            return False
        if param.nullable:
            return None
        return "" if param.kind in {"text", "textarea"} else param.default
    raw = raw.strip()
    if param.kind == "int":
        try:
            value = int(raw)
        except ValueError:
            raise FormError(f"« {param.label} » doit être un entier.")
    elif param.kind == "float":
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            raise FormError(f"« {param.label} » doit être un nombre.")
        if not math.isfinite(value):
            raise FormError(f"« {param.label} » doit être fini.")
    elif param.kind == "bool":
        return raw.lower() in {"1", "true", "on", "yes", "oui"}
    elif param.kind == "select":
        allowed = {c[0] for c in param.choices}
        if raw not in allowed:
            raise FormError(f"Valeur inconnue pour « {param.label} » : {raw}")
        return raw
    else:
        return raw
    if param.min is not None and value < param.min:
        raise FormError(f"« {param.label} » doit être ≥ {param.min}.")
    if param.max is not None and value > param.max:
        raise FormError(f"« {param.label} » doit être ≤ {param.max}.")
    return value


def form_defaults() -> dict[str, Any]:
    return {p.key: p.default for p in P.ALL_PARAMS.values()}


def parse_job_form(form) -> Job:
    values = {key: _coerce(param, form.get(key)) for key, param in P.ALL_PARAMS.items()
              if param not in P.ENGINE}
    values["abc"] = (form.get("abc") or "").strip()
    kind = form.get("kind", "song")
    if kind not in {"song", "plan"}:
        raise FormError("Type de job inconnu.")
    if not values["style"] or not values["lyrics"].strip():
        raise FormError("Le style et les paroles sont obligatoires.")
    cot = values["cot"]
    if values["abc"] and cot == "off":
        raise FormError("Une partition fournie exige le mode full ou melody.")
    if kind == "plan" and (cot == "off" or values["abc"]):
        raise FormError("« Planifier seulement » n'a de sens qu'en mode full ou melody sans partition fournie.")
    name = values["name"] or "song"
    request = {"id": slugify(name), "style": values["style"].strip(), "lyrics": values["lyrics"].replace("\r\n", "\n"),
               "cot": cot, "seed": int(values["seed"])}
    if values["abc"]:
        request["abc"] = values["abc"].replace("\r\n", "\n") + "\n"
    if values["cfg_scale"] is not None:
        request["cfg_scale"] = float(values["cfg_scale"])

    def sampling(prefix, catalog):
        out = {}
        for p in catalog:
            v = values[p.key]
            if v != p.default:
                out[p.key.split(".", 1)[1]] = v
        return out

    # Validation stricte par les dataclasses du runtime (mêmes règles que la CLI).
    from yue2.protocol import GenerationConfig, SongRequest, resolve_sampling
    defaults = GenerationConfig()
    abc_s, sem_s = sampling("abc", P.ABC_SAMPLING), sampling("semantic", P.SEMANTIC_SAMPLING)
    try:
        SongRequest(**request)
        resolve_sampling(abc_s or None, defaults.abc)
        resolve_sampling(sem_s or None, defaults.semantic)
        GenerationConfig(ode_steps=int(values["ode_steps"]))
    except (ValueError, TypeError) as exc:
        raise FormError(f"Requête refusée par le runtime : {exc}")

    engine = get_engine()
    project_id = (form.get("project_id") or "").strip() or None
    if project_id and A.store().get(project_id) is None:
        project_id = None
    return Job(id=engine.new_job_id(name), name=name, kind=kind, request=request,
               abc_sampling=abc_s, semantic_sampling=sem_s, ode_steps=int(values["ode_steps"]), project_id=project_id,
               max_seconds=int(values["max_duration"]) if values.get("max_duration") is not None else None,
               plan_overflow=values.get("plan_overflow") or "auto")


def parse_settings_form(form) -> EngineSettings:
    values = {p.key: _coerce(p, form.get(p.key)) for p in P.ENGINE}
    for k in ("revision", "vae_revision"):
        values[k] = values[k] or None
    if not values["model"]:
        raise FormError("Le modèle est obligatoire.")
    values["vae_custom"] = (values.get("vae_custom") or "").strip()
    if values["vae"] == "custom" and not values["vae_custom"]:
        raise FormError("Indiquez un dépôt ou un chemin pour le VAE personnalisé.")
    return EngineSettings(**values)


def job_to_form(job: Job, mode: str) -> dict[str, Any]:
    """Pré-remplit le formulaire depuis un job (réutiliser / éditer la partition)."""
    values = form_defaults()
    r = job.request
    values.update(name=job.name, style=r.get("style", ""), lyrics=r.get("lyrics", ""), cot=r.get("cot", "full"),
                  seed=r.get("seed", 831001), cfg_scale=r.get("cfg_scale"), abc=r.get("abc") or "",
                  ode_steps=job.ode_steps, project_id=job.project_id or "", max_duration=job.max_seconds,
                  plan_overflow=job.plan_overflow or "auto")
    for prefix, sampling in (("abc", job.abc_sampling), ("semantic", job.semantic_sampling)):
        for k, v in (sampling or {}).items():
            values[f"{prefix}.{k}"] = v
    if mode == "edit-score":
        score = job.output_dir / "score.abc"
        if score.is_file():
            values["abc"] = score.read_text(encoding="utf-8")
            if values["cot"] == "off":
                values["cot"] = "full"
            values["name"] = f"{job.name}_edit"
    elif mode == "variation":
        values["seed"] = random.randrange(0, 2**31)
        values["name"] = f"{job.name}_v{values['seed'] % 1000:03d}"
    return values


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def landing(request: Request, q: str = "", sort: str = "recent", all: str = ""):
    ctx = engine_ctx()
    ctx.update(gallery_ctx(q, sort, bool(all)))
    return HTMLResponse(render("landing.html", request, **ctx))


@app.get("/studio", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse(render("index.html", request, values=form_defaults(), error=None, **engine_ctx()))


def gallery_ctx(q: str = "", sort: str = "recent", show_all: bool = False) -> dict:
    jobs = [j for j in get_engine().jobs.values() if not j.is_active]
    if not show_all:
        jobs = [j for j in jobs if j.status == "done" and (j.output_dir / "audio.flac").is_file()]
    if q:
        needle = q.lower()
        jobs = [j for j in jobs if needle in j.name.lower() or needle in j.request.get("style", "").lower()
                or needle in j.request.get("lyrics", "").lower()]
    keys = {"recent": lambda j: j.created, "name": lambda j: j.name.lower(),
            "duration": lambda j: j.summary.get("audio_seconds") or 0}
    jobs.sort(key=keys.get(sort, keys["recent"]), reverse=sort in {"recent", "duration"})
    # Regroupement : un projet = une carte portant sa dernière version audible ; le reste en cartes simples.
    store = A.store()
    by_project: dict[str, list] = {}
    loose = []
    for j in jobs:
        found = store.for_job(j.id)
        if found:
            by_project.setdefault(found[0].id, []).append(j)
        else:
            loose.append(j)
    items = [{"kind": "job", "job": j, "sort_key": keys.get(sort, keys["recent"])(j)} for j in loose]
    for pid, members in by_project.items():
        project = store.get(pid)
        latest = max(members, key=lambda j: store.for_job(j.id)[1].number)
        items.append({"kind": "project", "project": project, "job": latest, "count": len(project.versions),
                      "sort_key": keys.get(sort, keys["recent"])(latest)})
    items.sort(key=lambda it: it["sort_key"], reverse=sort in {"recent", "duration"})
    return {"items": items, "tracks": jobs, "q": q, "sort": sort, "show_all": show_all,
            "total_done": sum(1 for j in get_engine().jobs.values() if j.status == "done"),
            "total_projects": len(store.all())}


@app.get("/partials/gallery", response_class=HTMLResponse)
async def partial_gallery(request: Request, q: str = "", sort: str = "recent", all: str = ""):
    return HTMLResponse(render("partials/gallery.html", request, **gallery_ctx(q, sort, bool(all))))


@app.get("/partials/card/{job_id}", response_class=HTMLResponse)
async def partial_card(request: Request, job_id: str):
    job = get_engine().jobs.get(job_id)
    if job is None:
        raise HTTPException(404)
    return HTMLResponse(render("partials/card.html", request, job=job))


@app.patch("/jobs/{job_id}/name", response_class=HTMLResponse)
async def rename_job(request: Request, job_id: str, view: str = "card"):
    form = await request.form()
    job = get_engine().rename(job_id, form.get("name") or "")
    if job is None:
        raise HTTPException(400, "Nom vide ou job introuvable.")
    if view == "detail":
        return HTMLResponse(render("partials/job_detail.html", request, job=job, flash=None, **detail_ctx(job)),
                            headers={"HX-Trigger": "library-changed"})
    return HTMLResponse(render("partials/card.html", request, job=job), headers={"HX-Trigger": "library-changed"})


@app.get("/partials/form", response_class=HTMLResponse)
async def partial_form(request: Request, from_job: str | None = None, mode: str = "reuse"):
    values = form_defaults()
    if from_job:
        job = get_engine().jobs.get(from_job)
        if job is None:
            raise HTTPException(404)
        values = job_to_form(job, mode)
        if (job.output_dir / "score.abc").is_file():
            values["source_job"] = job.id
    return HTMLResponse(render("partials/form.html", request, values=values, error=None, flash=None))


@app.get("/partials/queue", response_class=HTMLResponse)
async def partial_queue(request: Request):
    return HTMLResponse(render("partials/queue.html", request, **engine_ctx()))


@app.get("/partials/library", response_class=HTMLResponse)
async def partial_library(request: Request, q: str = ""):
    ctx = engine_ctx()
    if q:
        needle = q.lower()
        ctx["history"] = [j for j in ctx["history"] if needle in j.name.lower() or needle in j.request.get("style", "").lower()]
    ctx["q"] = q
    return HTMLResponse(render("partials/library.html", request, **ctx))


@app.get("/partials/status", response_class=HTMLResponse)
async def partial_status(request: Request):
    return HTMLResponse(render("partials/status.html", request, **engine_ctx()))


def llm_ctx(**extra) -> dict:
    return {"llm": llm.get_settings().public(), "llm_flash": None, "llm_error": None, "llm_models": None, **extra}


@app.get("/partials/settings", response_class=HTMLResponse)
async def partial_settings(request: Request):
    ctx = engine_ctx()
    return HTMLResponse(render("partials/settings.html", request, values=asdict(ctx["settings"]), error=None, flash=None,
                               **llm_ctx(), **ctx))


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
@app.post("/jobs", response_class=HTMLResponse)
async def create_job(request: Request):
    form = await request.form()
    values = {k: form.get(k) for k in P.ALL_PARAMS}
    values["source_job"] = form.get("source_job")
    try:
        job = parse_job_form(form)
    except FormError as exc:
        return HTMLResponse(render("partials/form.html", request, values=values, error=str(exc), flash=None), status_code=422)
    label = "Planification" if job.kind == "plan" else "Génération"
    flash = f"{label} « {job.name} » ajoutée à la file."
    project = A.store().get(job.project_id)
    if project is not None and job.kind == "song":
        # Lier la version avant la soumission : le job peut se terminer (et notifier) très vite.
        source = "assistant" if A.proposal_matches_request(project.proposal, job.request) else "manual"
        version = A.store().add_version(project, job.id, source=source, label=job.name)
        job.version = version.number
        if project.seed is None:
            project.seed = job.request.get("seed")
            A.store().save(project)
        flash += f" Version v{version.number} du projet « {project.name} »."
    else:
        job.project_id = None
    get_engine().submit(job)
    values["project_id"] = job.project_id or ""
    return HTMLResponse(render("partials/form.html", request, values=values, error=None, flash=flash),
                        headers={"HX-Trigger": "queue-changed"})


@app.post("/jobs/{job_id}/cancel", response_class=HTMLResponse)
async def cancel_job(request: Request, job_id: str):
    get_engine().cancel(job_id)
    return HTMLResponse(render("partials/queue.html", request, **engine_ctx()))


@app.delete("/jobs/{job_id}", response_class=HTMLResponse)
async def delete_job(request: Request, job_id: str):
    engine = get_engine()
    if not engine.delete(job_id):
        raise HTTPException(409, "Job en cours : annulez-le d'abord.")
    return HTMLResponse("", headers={"HX-Trigger": "library-changed"})


@app.post("/jobs/{job_id}/decode", response_class=HTMLResponse)
async def decode_job(request: Request, job_id: str, vae: str = "legacy"):
    engine = get_engine()
    source = engine.jobs.get(job_id)
    if source is None or not (source.output_dir / "latent.npy").is_file():
        raise HTTPException(404, "Latents introuvables pour ce job.")
    if vae not in {"standard", "legacy", "custom"}:
        raise HTTPException(400)
    if vae == "custom" and not (engine.settings.vae_custom or "").strip():
        raise HTTPException(400, "Aucun VAE personnalisé configuré dans le moteur.")
    job = Job(id=engine.new_job_id(f"{source.name}_vae_{vae}"), name=f"{source.name} · VAE {vae}", kind="decode",
              request=dict(source.request), source_job=source.id, decode_vae=vae, ode_steps=source.ode_steps)
    engine.submit(job)
    return HTMLResponse(render("partials/job_detail.html", request, job=source,
                               flash=f"Re-décodage avec le VAE {vae} ajouté à la file.", **detail_ctx(source)))


from .guard import abc_duration  # noqa: E402  (réexport : utilisé par les routes et les tests)


def detail_ctx(job: Job) -> dict:
    d = job.output_dir
    files = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
    score = (d / "score.abc").read_text(encoding="utf-8") if (d / "score.abc").is_file() else None
    result = json.loads((d / "result.json").read_text()) if (d / "result.json").is_file() else None
    config = json.loads((d / "config.json").read_text()) if (d / "config.json").is_file() else None
    trace = (d / "traceback.txt").read_text() if (d / "traceback.txt").is_file() else None
    engine = get_engine()
    score_jobs = [j for j in engine.jobs.values() if j.id != job.id and (j.output_dir / "score.abc").is_file()]
    score_jobs.sort(key=lambda j: j.created, reverse=True)
    found = A.store().for_job(job.id)
    return {"files": files, "score": score, "result": result, "config": config, "trace": trace,
            "planned": abc_duration(score) if score else None,
            "project": found[0] if found else None, "project_version": found[1] if found else None,
            "score_jobs": score_jobs, "custom_vae": (engine.settings.vae_custom or "").strip(),
            "has_audio": (d / "audio.flac").is_file(), "has_latents": (d / "latent.npy").is_file()}


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str):
    job = get_engine().jobs.get(job_id)
    if job is None:
        raise HTTPException(404)
    return HTMLResponse(render("partials/job_detail.html", request, job=job, flash=None, **detail_ctx(job)))


_SAFE_FILE = re.compile(r"^[A-Za-z0-9_.-]+$")


@app.get("/jobs/{job_id}/file/{name}")
async def job_file(job_id: str, name: str):
    job = get_engine().jobs.get(job_id)
    if job is None or not _SAFE_FILE.match(name):
        raise HTTPException(404)
    path = job.output_dir / name
    if not path.is_file():
        raise HTTPException(404)
    media = {"flac": "audio/flac", "abc": "text/plain; charset=utf-8", "json": "application/json",
             "txt": "text/plain; charset=utf-8"}.get(path.suffix.lstrip("."), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=f"{job.request.get('id', job.id)}_{name}")


@app.get("/jobs/{job_id}/audio.wav")
async def job_wav(job_id: str):
    job = get_engine().jobs.get(job_id)
    if job is None or not (job.output_dir / "audio.flac").is_file():
        raise HTTPException(404)

    def convert():
        import soundfile as sf
        data, sr = sf.read(job.output_dir / "audio.flac", dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_24")
        return buf.getvalue()

    payload = await asyncio.to_thread(convert)
    return Response(payload, media_type="audio/wav",
                    headers={"Content-Disposition": f'attachment; filename="{job.request.get("id", job.id)}.wav"'})


# --------------------------------------------------------------------------
# Moteur
# --------------------------------------------------------------------------
@app.post("/settings", response_class=HTMLResponse)
async def save_settings(request: Request):
    form = await request.form()
    engine = get_engine()
    try:
        settings = parse_settings_form(form)
    except FormError as exc:
        ctx = engine_ctx()
        return HTMLResponse(render("partials/settings.html", request, values={p.key: form.get(p.key) for p in P.ENGINE},
                                   error=str(exc), flash=None, **llm_ctx(), **ctx), status_code=422)
    engine.update_settings(settings)
    if form.get("reload") == "1":
        engine.request_load()
        flash = "Réglages enregistrés, rechargement du pipeline demandé."
    else:
        flash = "Réglages enregistrés. Ils s'appliqueront au prochain chargement du pipeline."
    ctx = engine_ctx()
    return HTMLResponse(render("partials/settings.html", request, values=asdict(settings), error=None, flash=flash, **llm_ctx(), **ctx),
                        headers={"HX-Trigger": "status-changed"})


@app.post("/engine/load", response_class=HTMLResponse)
async def engine_load(request: Request):
    get_engine().request_load()
    return HTMLResponse(render("partials/status.html", request, **engine_ctx()))


@app.post("/engine/unload", response_class=HTMLResponse)
async def engine_unload(request: Request):
    get_engine().request_unload()
    return HTMLResponse(render("partials/status.html", request, **engine_ctx()))


@app.get("/api/state")
async def api_state():
    ctx = engine_ctx()
    return JSONResponse({"model_state": ctx["model_state"], "gpu": ctx["gpu"], "version": ctx["version"],
                         "current": ctx["current"].id if ctx["current"] else None,
                         "queued": [j.id for j in ctx["queued"]]})


# --------------------------------------------------------------------------
# Server-Sent Events
# --------------------------------------------------------------------------
def _sse(event: str, html: str) -> str:
    data = "".join(f"data: {line}\n" for line in html.splitlines()) or "data: \n"
    return f"event: {event}\n{data}\n"


@app.get("/events")
async def events(request: Request):
    engine = get_engine()

    async def stream():
        last_version, last_status, last_history = -1, 0.0, None
        yield ": connected\n\n"
        while True:
            if await request.is_disconnected():
                break
            now = asyncio.get_event_loop().time()
            if engine.version != last_version:
                last_version = engine.version
                ctx = engine_ctx()
                yield _sse("queue", render("partials/queue.html", None, **ctx))
                yield _sse("status", render("partials/status.html", None, **ctx))
                last_status = now
                history_ids = tuple(j.id for j in ctx["history"])
                if history_ids != last_history:
                    last_history = history_ids
                    yield _sse("library", render("partials/library.html", None, q="", **ctx))
            elif now - last_status > 3.0:
                last_status = now
                yield _sse("status", render("partials/status.html", None, **engine_ctx()))
            else:
                # Les étapes en cours mettent à jour elapsed/rate même sans nouveau token.
                if engine.current is not None:
                    yield _sse("queue", render("partials/queue.html", None, **engine_ctx()))
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------
# Import par lot (JSONL / JSON / CSV)
# --------------------------------------------------------------------------
_ROW_KEYS = {"name", "id", "style", "tags", "lyrics", "cot", "seed", "cfg_scale", "abc", "ode_steps", "kind",
             "abc_sampling", "semantic_sampling", "max_duration", "plan_overflow"}


def _rows_from_upload(data: bytes, filename: str) -> list[dict]:
    text = data.decode("utf-8-sig")
    name = (filename or "").lower()
    if name.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
        for row in rows:
            if row.get("lyrics"):
                row["lyrics"] = row["lyrics"].replace("\\n", "\n")
        return rows
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        return json.loads(stripped)
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            return obj.get("requests", obj.get("songs", [obj])) if isinstance(obj, dict) else obj
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in stripped.splitlines() if line.strip()]


def _row_to_form(row: dict, defaults: dict) -> dict[str, str]:
    """Convertit une ligne d'import en champs de formulaire (chaînes), en héritant du formulaire courant."""
    unknown = set(row) - _ROW_KEYS - {"lang", "eval_index", "clip_id", "prompt"}
    if unknown:
        raise FormError(f"Champs inconnus : {sorted(unknown)}")
    out = dict(defaults)
    out["name"] = str(row.get("name") or row.get("id") or defaults.get("name") or "song")
    for key in ("style", "lyrics", "cot", "seed", "cfg_scale", "abc", "ode_steps", "kind", "max_duration", "plan_overflow"):
        src = "tags" if key == "style" and "style" not in row and "tags" in row else key
        if src in row and row[src] is not None:
            out[key] = str(row[src])
    for prefix in ("abc", "semantic"):
        sampling = row.get(f"{prefix}_sampling") or {}
        if not isinstance(sampling, dict):
            raise FormError(f"{prefix}_sampling doit être un objet")
        for k, v in sampling.items():
            out[f"{prefix}.{k}"] = str(v)
    return out


@app.get("/batch/template.jsonl")
async def batch_template():
    rows = [
        {"name": "exemple_pop", "style": P.STYLE_PRESETS[0][1], "lyrics": P.ALL_PARAMS["lyrics"].default,
         "cot": "full", "seed": 831001},
        {"name": "exemple_jazz_cfg", "style": P.STYLE_PRESETS[2][1], "lyrics": P.ALL_PARAMS["lyrics"].default,
         "cot": "melody", "seed": 42, "cfg_scale": 1.2, "semantic_sampling": {"max_tokens": 6000}, "ode_steps": 32},
    ]
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    return Response(body, media_type="application/x-ndjson",
                    headers={"Content-Disposition": 'attachment; filename="yue2-lot-exemple.jsonl"'})


@app.post("/jobs/batch", response_class=HTMLResponse)
async def create_batch(request: Request):
    form = await request.form()
    upload = form.get("file")
    if upload is None or not getattr(upload, "filename", ""):
        return HTMLResponse(render("partials/batch_result.html", request, error="Choisissez un fichier .jsonl, .json ou .csv.", created=[], failures=[]), status_code=422)
    data = await upload.read()
    try:
        rows = _rows_from_upload(data, upload.filename)
    except Exception as exc:
        return HTMLResponse(render("partials/batch_result.html", request, error=f"Fichier illisible : {exc}", created=[], failures=[]), status_code=422)
    # Les champs non renseignés dans le fichier héritent du formulaire courant (style, échantillonnage…).
    defaults = {k: form.get(k) for k in P.ALL_PARAMS if form.get(k) is not None}
    defaults.setdefault("kind", "song")
    engine = get_engine()
    created, failures = [], []
    for index, row in enumerate(rows, 1):
        try:
            if not isinstance(row, dict):
                raise FormError("chaque entrée doit être un objet")
            job = parse_job_form(_row_to_form(row, defaults))
            engine.submit(job)
            created.append(job)
        except FormError as exc:
            failures.append((index, row.get("name") or row.get("id") if isinstance(row, dict) else "?", str(exc)))
    return HTMLResponse(render("partials/batch_result.html", request, error=None, created=created, failures=failures),
                        headers={"HX-Trigger": "queue-changed"})


# --------------------------------------------------------------------------
# Outils de partition (skill yue2-music)
# --------------------------------------------------------------------------
def _score_text(form, key: str) -> str:
    """Texte ABC depuis un champ texte ou un identifiant de job (<key>_job)."""
    job_id = form.get(f"{key}_job")
    if job_id:
        job = get_engine().jobs.get(job_id)
        if job is None or not (job.output_dir / "score.abc").is_file():
            raise FormError("Partition source introuvable.")
        return (job.output_dir / "score.abc").read_text(encoding="utf-8")
    text = form.get(key)
    if text is None and key == "after":
        text = form.get("abc")          # le formulaire principal envoie la partition sous le nom « abc »
    return (text or "").replace("\r\n", "\n")


@app.post("/abc/inspect", response_class=HTMLResponse)
async def abc_inspect(request: Request):
    form = await request.form()
    text = (form.get("abc") or "").replace("\r\n", "\n")
    if not text.strip():
        return HTMLResponse(render("partials/abc_check.html", request, kind="inspect", error="Aucune partition à valider.", data=None))
    try:
        data = await asyncio.to_thread(abctools.inspect, text + ("" if text.endswith("\n") else "\n"))
        return HTMLResponse(render("partials/abc_check.html", request, kind="inspect", error=None, data=data))
    except Exception as exc:
        return HTMLResponse(render("partials/abc_check.html", request, kind="inspect", error=str(exc), data=None))


@app.post("/abc/strip")
async def abc_strip(request: Request):
    form = await request.form()
    text = (form.get("abc") or "").replace("\r\n", "\n")
    keep = form.get("keep_voice") or "both"
    if keep not in {"both", "Vocal", "Ins"}:
        raise HTTPException(400)
    try:
        out = await asyncio.to_thread(abctools.strip_chords, text + ("" if text.endswith("\n") else "\n"), keep)
        return JSONResponse({"ok": True, "abc": out})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)


@app.post("/abc/compare", response_class=HTMLResponse)
async def abc_compare(request: Request):
    form = await request.form()
    voices = form.get("voices") or "both"
    allow_tempo = form.get("allow_tempo_change") in {"1", "on", "true"}
    try:
        before, after = _score_text(form, "before"), _score_text(form, "after")
        if not before.strip() or not after.strip():
            raise FormError("Il faut deux partitions à comparer.")
        data = await asyncio.to_thread(abctools.compare, before, after, voices, allow_tempo)
        return HTMLResponse(render("partials/abc_check.html", request, kind="compare", error=None, data=data))
    except Exception as exc:
        return HTMLResponse(render("partials/abc_check.html", request, kind="compare", error=str(exc), data=None))


# --------------------------------------------------------------------------
# Assistant de composition (LLM OpenAI-compatible)
# --------------------------------------------------------------------------
def _assistant_html(request: Request, project: PR.Project | None) -> str:
    return render("partials/assistant.html", request, project=project, settings=llm.get_settings(),
                  projects=A.store().all()[:20])


@app.get("/assistant/panel", response_class=HTMLResponse)
async def assistant_panel(request: Request, project_id: str | None = None):
    return HTMLResponse(_assistant_html(request, A.store().get(project_id)))


@app.post("/assistant/reset", response_class=HTMLResponse)
async def assistant_reset(request: Request):
    """Nouveau projet vierge (l'ancien reste dans la liste, rien n'est perdu)."""
    return HTMLResponse(_assistant_html(request, A.store().create()))


@app.post("/assistant/message", response_class=HTMLResponse)
async def assistant_message(request: Request):
    form = await request.form()
    text = (form.get("text") or "").strip()
    project = A.store().get(form.get("project_id")) or A.store().create()
    if text:
        context = {"style": form.get("style"), "lyrics": form.get("lyrics"), "cot": form.get("cot")}
        await A.step(project, text, context=context, jobs=get_engine().jobs)
    return HTMLResponse(_assistant_html(request, project))


@app.post("/assistant/apply", response_class=HTMLResponse)
async def assistant_apply(request: Request):
    form = await request.form()
    project = A.store().get(form.get("project_id"))
    if project is None or not project.proposal:
        raise HTTPException(404, "Aucune proposition à appliquer.")
    values = A.proposal_to_form(project.proposal, project)
    flash = (f"Proposition « {project.proposal['name']} » appliquée. La prochaine génération deviendra la "
             f"v{project.next_number} du projet « {project.name} ».")
    return HTMLResponse(render("partials/form.html", request, values=values, error=None, flash=flash, open_tab="song"))


@app.post("/assistant/quick")
async def assistant_quick(request: Request):
    form = await request.form()
    action = form.get("action") or ""
    payload = {"style": (form.get("style") or "").strip(), "lyrics": (form.get("lyrics") or "").replace("\r\n", "\n"),
               "instruction": (form.get("instruction") or "").strip()}
    try:
        result = await A.quick(action, payload)
        return JSONResponse({"ok": True, **result})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/assistant/settings", response_class=HTMLResponse)
async def assistant_settings(request: Request):
    form = await request.form()
    current = llm.get_settings()
    try:
        temperature = float((form.get("temperature") or "0.8").replace(",", "."))
        max_tokens = int(form.get("max_tokens") or 4000)
    except ValueError:
        return HTMLResponse(render("partials/assistant_settings.html", request, **llm_ctx(llm_error="Température ou max_tokens invalide.")), status_code=422)
    key = current.api_key
    if form.get("clear_key") == "1":
        key = ""
    if (form.get("api_key") or "").strip():
        key = form.get("api_key").strip()
    # Un champ imposé par l'environnement n'est pas soumis (désactivé dans l'UI) : on conserve la valeur enregistrée.
    base_url = current.base_url if current.base_url_from_env else (form.get("base_url") or "").strip().rstrip("/")
    model = current.model if current.model_from_env else (form.get("model") or "").strip()
    settings = llm.LLMSettings(base_url=base_url, model=model, api_key=key,
                               temperature=min(max(temperature, 0.0), 2.0), max_tokens=min(max(max_tokens, 256), 32000))
    llm.set_settings(settings)
    ctx = llm_ctx(llm_flash="Connecteur enregistré.")
    if form.get("test") == "1":
        try:
            models = await llm.list_models(settings)
            ctx["llm_models"] = models
            ctx["llm_flash"] = f"Connexion OK : {len(models)} modèle(s) annoncé(s)."
            if settings.effective_model:
                content, _ = await llm.chat([{"role": "user", "content": 'Réponds exactement {"ok": true}'}], s=settings, max_tokens=50, temperature=0)
                ctx["llm_flash"] += f" Appel de test réussi avec {settings.effective_model}."
        except Exception as exc:
            ctx["llm_flash"] = None
            ctx["llm_error"] = f"Test échoué : {exc}"
    return HTMLResponse(render("partials/assistant_settings.html", request, **ctx), headers={"HX-Trigger": "assistant-config-changed"})


# --------------------------------------------------------------------------
# Projets : conversation + versions
# --------------------------------------------------------------------------
def project_ctx(project: PR.Project) -> dict:
    """Versions résolues (job, diff avec la précédente) pour la frise."""
    jobs = get_engine().jobs
    rows, prev = [], None
    for v in project.versions:
        job = jobs.get(v.job_id)
        req = dict(job.request) if job else {}
        extra = {"ode_steps": job.ode_steps, "semantic_sampling": job.semantic_sampling, "abc_sampling": job.abc_sampling} if job else {}
        changes = PR.diff_requests(prev["req"], req, before_extra=prev["extra"], after_extra=extra) if prev else None
        rows.append({"version": v, "job": job, "changes": changes,
                     "summary": PR.summarize_changes(changes) if changes else "version initiale",
                     "has_audio": bool(job and (job.output_dir / "audio.flac").is_file())})
        prev = {"req": req, "extra": extra}
    return {"project": project, "rows": rows, "audible": [r for r in rows if r["has_audio"]]}


@app.get("/projects/{project_id}", response_class=HTMLResponse)
async def project_page(request: Request, project_id: str):
    project = A.store().get(project_id)
    if project is None:
        raise HTTPException(404)
    return HTMLResponse(render("project.html", request, **project_ctx(project), **engine_ctx()))


@app.get("/partials/project/{project_id}", response_class=HTMLResponse)
async def project_partial(request: Request, project_id: str):
    project = A.store().get(project_id)
    if project is None:
        raise HTTPException(404)
    return HTMLResponse(render("partials/project_timeline.html", request, **project_ctx(project)))


@app.patch("/projects/{project_id}/name", response_class=HTMLResponse)
async def project_rename(request: Request, project_id: str):
    form = await request.form()
    project = A.store().rename(project_id, form.get("name") or "")
    if project is None:
        raise HTTPException(400, "Nom vide ou projet introuvable.")
    return HTMLResponse(render("partials/project_timeline.html", request, **project_ctx(project)),
                        headers={"HX-Trigger": "library-changed"})


@app.delete("/projects/{project_id}")
async def project_delete(project_id: str, with_jobs: str = "", card: str = ""):
    """Supprime un projet ; ``card=1`` (pochette de la galerie) renvoie une réponse vide à échanger sur place."""
    store = A.store()
    project = store.get(project_id)
    if project is None:
        raise HTTPException(404)
    engine = get_engine()
    if with_jobs == "1":
        for v in project.versions:
            engine.delete(v.job_id)
    store.delete(project_id)
    if card == "1":
        return HTMLResponse("", headers={"HX-Trigger": "library-changed"})
    return Response(status_code=204, headers={"HX-Redirect": "/", "HX-Trigger": "library-changed"})


@app.post("/projects/from-job/{job_id}", response_class=HTMLResponse)
async def project_from_job(request: Request, job_id: str, target: str = "drawer"):
    job = get_engine().jobs.get(job_id)
    if job is None:
        raise HTTPException(404)
    found = A.store().for_job(job_id)
    project = found[0] if found else A.create_from_job(job)
    if not found:
        job.project_id, job.version = project.id, 1
        job.persist()
    if target == "page":
        return Response(status_code=204, headers={"HX-Redirect": f"/projects/{project.id}"})
    return HTMLResponse(_assistant_html(request, project), headers={"HX-Trigger": json.dumps({"assistant-open": True, "library-changed": True})})


@app.get("/projects/{project_id}/diff", response_class=HTMLResponse)
async def project_diff(request: Request, project_id: str, a: int, b: int):
    project = A.store().get(project_id)
    if project is None:
        raise HTTPException(404)
    jobs = get_engine().jobs
    va, vb = project.version(a), project.version(b)
    if va is None or vb is None:
        raise HTTPException(404, "Version inconnue")
    ja, jb = jobs.get(va.job_id), jobs.get(vb.job_id)
    if ja is None or jb is None:
        raise HTTPException(404, "Job de version introuvable")
    changes = PR.diff_requests(ja.request, jb.request,
                               before_extra={"ode_steps": ja.ode_steps, "semantic_sampling": ja.semantic_sampling, "abc_sampling": ja.abc_sampling},
                               after_extra={"ode_steps": jb.ode_steps, "semantic_sampling": jb.semantic_sampling, "abc_sampling": jb.abc_sampling})
    return HTMLResponse(render("partials/version_diff.html", request, project=project, a=va, b=vb, ja=ja, jb=jb,
                               changes=changes, summary=PR.summarize_changes(changes)))


def _on_job_finished(job: Job) -> None:
    """Consigne le résultat objectif d'une version dans son projet (durée, troncature, mesures planifiées)."""
    if not job.project_id:
        return
    score = job.output_dir / "score.abc"
    planned = abc_duration(score.read_text(encoding="utf-8")) if score.is_file() else None
    A.store().record_outcome(job.id, A.outcome_from_job(job, planned))


get_engine().listeners.append(_on_job_finished)
