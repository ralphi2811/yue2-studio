"""Moteur du studio : pipeline YuE2 résident, file de jobs séquentielle, progression.

Un seul thread de travail exécute les jobs les uns après les autres (le pipeline
n'est pas conçu pour un usage concurrent). L'état des jobs est protégé par un
verrou et versionné pour que la couche SSE ne pousse que les changements.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ROOT, OUTPUT_ROOT, DATA_DIR
from . import guard

SETTINGS_FILE = DATA_DIR / "settings.json"

STAGE_LABELS_FR = {
    "Resolving model files": "Résolution des fichiers modèle",
    "Verifying model files": "Vérification des empreintes",
    "Loading model": "Chargement du modèle",
    "Using provided score": "Partition fournie",
    "Planning score": "Planification de la partition",
    "Generating song": "Génération des tokens sémantiques",
    "Synthesizing audio": "Synthèse acoustique (flow matching)",
    "Loading audio decoder": "Chargement du décodeur VAE",
    "Decoding audio": "Décodage audio",
}


# --------------------------------------------------------------------------
# Réglages moteur
# --------------------------------------------------------------------------
@dataclass
class EngineSettings:
    model: str = "m-a-p/YuE2-3B"
    vae: str = "standard"
    vae_custom: str = ""
    revision: str | None = None
    vae_revision: str | None = None
    device: str = "auto"
    budget: float = 24.0
    backend: str = "torch"
    quantization: str = "none"
    offload_ar: bool = False
    vae_core_frames: str = "auto"
    offline: bool = False
    verify_hashes: bool = True

    @property
    def vae_repo(self) -> str:
        if self.vae == "custom":
            return (self.vae_custom or "").strip()
        return {"standard": "m-a-p/YuE2-Vae", "legacy": "m-a-p/YuE2-Vae-legacy"}.get(self.vae, self.vae)

    def resolve_vae(self, choice: str) -> str:
        """standard | legacy | custom → dépôt HF ou chemin local."""
        if choice == "custom":
            return (self.vae_custom or "").strip()
        return {"standard": "m-a-p/YuE2-Vae", "legacy": "m-a-p/YuE2-Vae-legacy"}[choice]

    def pipeline_kwargs(self) -> dict:
        frames = None if self.vae_core_frames == "auto" else int(self.vae_core_frames)
        return dict(model=self.model, vae=self.vae_repo, revision=self.revision or None,
                    vae_revision=self.vae_revision or None, local_files_only=self.offline,
                    device=self.device, memory_budget_gib=self.budget, backend=self.backend,
                    quantization=self.quantization, offload_ar=self.offload_ar,
                    vae_core_frames=frames, verify_hashes=self.verify_hashes)

    @classmethod
    def load(cls) -> "EngineSettings":
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


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
@dataclass
class Stage:
    label: str = ""
    label_fr: str = ""
    completed: int = 0
    total: int | None = None
    unit: str | None = None
    started: float = 0.0
    finished: float | None = None
    status: str = "running"

    @property
    def elapsed(self) -> float:
        end = self.finished if self.finished is not None else time.monotonic()
        return max(0.0, end - self.started)

    @property
    def rate(self) -> float | None:
        e = self.elapsed
        return self.completed / e if e > 0.2 and self.completed else None

    @property
    def percent(self) -> float | None:
        if self.total:
            return min(100.0, 100.0 * self.completed / self.total)
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(elapsed=self.elapsed, rate=self.rate, percent=self.percent)
        return d


@dataclass
class Job:
    id: str
    name: str
    kind: str                                   # song | plan | decode
    request: dict = field(default_factory=dict)  # style, lyrics, cot, seed, abc, cfg_scale, id
    abc_sampling: dict = field(default_factory=dict)
    semantic_sampling: dict = field(default_factory=dict)
    ode_steps: int = 32
    source_job: str | None = None               # pour decode : job d'origine
    decode_vae: str | None = None               # pour decode : "standard" | "legacy"
    status: str = "queued"                      # queued | running | done | failed | cancelled
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: float | None = None
    finished_at: float | None = None
    stages: list[Stage] = field(default_factory=list)
    error: str | None = None
    summary: dict = field(default_factory=dict)  # audio_seconds, truncated, timing, output files
    engine: dict = field(default_factory=dict)   # réglages moteur utilisés
    cancel_requested: bool = False
    live_abc: str = ""                          # partition en cours de planification (streaming)
    live_tokens: int = 0
    project_id: str | None = None               # projet (conversation + versions) auquel ce job appartient
    version: int | None = None
    max_seconds: int | None = None              # durée maximale demandée (garde-fou), None = aucune
    plan_overflow: str = "auto"                 # partition trop longue : auto | trim | stop (voir guard.should_trim)
    seq: int = 0                                # ordre d'arrivée dans la session (départage des créations dans la même seconde)

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.id

    @property
    def current_stage(self) -> Stage | None:
        return self.stages[-1] if self.stages else None

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at

    @property
    def is_active(self) -> bool:
        return self.status in {"queued", "running"}

    def persist(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(self).items()
                if k not in {"stages", "cancel_requested", "started_at", "finished_at", "live_abc", "live_tokens", "seq"}}
        data["elapsed_seconds"] = self.elapsed
        data["stages"] = [s.to_dict() for s in self.stages]
        (self.output_dir / "studio.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))

    @classmethod
    def from_disk(cls, directory: Path) -> "Job | None":
        meta = directory / "studio.json"
        if not meta.is_file():
            return None
        try:
            data = json.loads(meta.read_text())
        except Exception:
            return None
        stages = [Stage(**{k: v for k, v in s.items() if k in Stage.__dataclass_fields__}) for s in data.pop("stages", [])]
        elapsed = data.pop("elapsed_seconds", None)
        job = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        job.stages = stages
        if job.status in {"queued", "running"}:
            job.status, job.error = "failed", "Interrompu (le serveur a redémarré pendant le job)"
        if elapsed is not None:
            job.started_at, job.finished_at = 0.0, float(elapsed)
        return job


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-")[:60]
    return slug if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", slug or "") else "song"


# --------------------------------------------------------------------------
# Pont de progression : remplace l'affichage stderr du runtime
# --------------------------------------------------------------------------
class _StageProxy:
    """Imite l'API de yue2.progress._Stage et alimente le job courant."""

    def __init__(self, engine: "Engine", job: Job, label: str, total, unit):
        self.engine, self.job = engine, job
        self.stage = Stage(label=label, label_fr=STAGE_LABELS_FR.get(label, label), total=total, unit=unit,
                           started=time.monotonic())
        self._last = 0.0

    def start(self):
        with self.engine.lock:
            self.job.stages.append(self.stage)
            self.engine._bump()

    def update(self, completed, total=None):
        self.stage.completed = int(completed)
        if total is not None:
            self.stage.total = int(total)
        now = time.monotonic()
        if now - self._last > 0.15:
            self._last = now
            self.engine._bump()

    def set_total(self, total):
        self.stage.total = None if total is None else int(total)

    def advance(self, count=1):
        self.update(self.stage.completed + int(count))

    def token(self, phase, token):
        self.advance()

    def finish(self, status="completed"):
        if self.stage.finished is None:
            self.stage.finished = time.monotonic()
            self.stage.status = status
            self.engine._bump()


def _make_pipeline_class():
    from yue2.pipeline import YuE2Pipeline

    class StudioPipeline(YuE2Pipeline):
        """Pipeline YuE2 dont les étapes remontent dans le studio au lieu de stderr."""

        def __init__(self, *args, engine=None, **kwargs):
            self._engine = engine
            kwargs["progress"] = True   # nécessaire pour que le runtime câble les callbacks
            super().__init__(*args, **kwargs)

        @contextmanager
        def _status(self, label, *, total=None, unit=None):
            engine = self._engine
            job = engine.current if engine is not None else None
            if job is None:
                from yue2.progress import Progress
                with Progress(enabled=False) as reporter, reporter.stage(label, total=total, unit=unit) as stage:
                    yield stage
                return
            proxy = _StageProxy(engine, job, label, total, unit)
            proxy.start()
            try:
                yield proxy
            except BaseException as exc:
                proxy.finish("cancelled" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed")
                raise
            else:
                proxy.finish("completed")

    return StudioPipeline


# --------------------------------------------------------------------------
# Moteur
# --------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.version = 0
        self.settings = EngineSettings.load()
        self.loaded_settings: EngineSettings | None = None
        self.pipe = None
        self.model_state = "unloaded"          # unloaded | loading | ready | error
        self.model_error: str | None = None
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.current: Job | None = None
        self._wake = threading.Event()
        self._load_requested = False
        self._unload_requested = False
        self.listeners: list = []               # callbacks(job) appelés à la fin de chaque job
        self._seq = 0
        self._load_jobs_from_disk()
        self._thread = threading.Thread(target=self._worker, name="yue2-studio-worker", daemon=True)
        self._thread.start()

    # ---- état -----------------------------------------------------------
    def _bump(self):
        self.version += 1

    def _load_jobs_from_disk(self):
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        for directory in sorted(OUTPUT_ROOT.iterdir()):
            if directory.is_dir():
                job = Job.from_disk(directory)
                if job is not None:
                    self.jobs[job.id] = job

    def snapshot(self) -> dict:
        with self.lock:
            active = [self.jobs[j] for j in self.queue]
            history = sorted((j for j in self.jobs.values() if not j.is_active), key=lambda j: (j.created, j.seq), reverse=True)
            return {"current": self.current, "queued": active, "history": history,
                    "model_state": self.model_state, "model_error": self.model_error,
                    "settings": self.settings, "loaded_settings": self.loaded_settings,
                    "settings_dirty": self.loaded_settings is not None and self.loaded_settings != self.settings,
                    "version": self.version}

    def gpu_status(self) -> dict | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2).stdout
            name, used, total, util, temp = [x.strip() for x in out.strip().splitlines()[0].split(",")]
            return {"name": name, "used_gib": int(used) / 1024, "total_gib": int(total) / 1024,
                    "percent": 100 * int(used) / int(total), "util": int(util), "temp": int(temp)}
        except Exception:
            return None

    # ---- réglages -------------------------------------------------------
    def update_settings(self, settings: EngineSettings):
        with self.lock:
            self.settings = settings
            settings.save()
            self._bump()

    def request_load(self):
        with self.lock:
            self._load_requested = True
            self._wake.set()

    def request_unload(self):
        with self.lock:
            self._unload_requested = True
            self._wake.set()

    # ---- jobs -----------------------------------------------------------
    def new_job_id(self, name: str) -> str:
        return f"{datetime.now():%Y%m%d-%H%M%S}-{slugify(name)}-{uuid.uuid4().hex[:4]}"

    def next_seq(self) -> int:
        with self.lock:
            self._seq += 1
            return self._seq

    def submit(self, job: Job) -> Job:
        with self.lock:
            job.seq = self.next_seq()
            job.engine = asdict(self.settings)
            self.jobs[job.id] = job
            self.queue.append(job.id)
            job.persist()
            self._bump()
            self._wake.set()
        return job

    def cancel(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            if job.status == "queued":
                try:
                    self.queue.remove(job_id)
                except ValueError:
                    pass
                job.status = "cancelled"
                job.error = "Annulé avant démarrage"
                job.persist()
            elif job.status == "running":
                job.cancel_requested = True
            self._bump()

    def rename(self, job_id: str, name: str) -> Job | None:
        name = " ".join(name.split())[:120]
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or not name:
                return None
            job.name = name
            if job.status != "running":
                job.persist()
            self._bump()
            return job

    def delete(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.status == "running":
                return False
            if job_id in self.queue:
                self.queue.remove(job_id)
            del self.jobs[job_id]
            shutil.rmtree(job.output_dir, ignore_errors=True)
            self._bump()
            return True

    # ---- worker ---------------------------------------------------------
    def _ensure_pipeline(self):
        with self.lock:
            wanted = dataclasses.replace(self.settings)
            fresh = self.pipe is not None and self.loaded_settings == wanted
        if fresh:
            return self.pipe
        self._close_pipeline()
        with self.lock:
            self.model_state, self.model_error = "loading", None
            self._bump()
        try:
            cls = _make_pipeline_class()
            pipe = cls.from_pretrained(engine=self, **wanted.pipeline_kwargs())
            # Précharge le modèle principal pour que le premier job ne paie pas le chargement.
            pipe._load_model()
            with self.lock:
                self.pipe, self.loaded_settings, self.model_state = pipe, wanted, "ready"
                self._bump()
            return pipe
        except BaseException as exc:
            with self.lock:
                self.model_state, self.model_error = "error", f"{type(exc).__name__}: {exc}"
                self._bump()
            raise

    def _close_pipeline(self):
        with self.lock:
            pipe, self.pipe, self.loaded_settings = self.pipe, None, None
            self.model_state = "unloaded"
            self._bump()
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass
            try:
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _worker(self):
        while True:
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            with self.lock:
                unload, load = self._unload_requested, self._load_requested
                self._unload_requested = self._load_requested = False
                job_id = self.queue.popleft() if self.queue else None
                job = self.jobs.get(job_id) if job_id else None
            if unload and job is None:
                self._close_pipeline()
                continue
            if load and job is None:
                try:
                    self._ensure_pipeline()
                except BaseException:
                    traceback.print_exc()
                continue
            if job is None:
                continue
            self._run(job)

    def _run(self, job: Job):
        with self.lock:
            self.current = job
            job.status, job.started_at = "running", time.monotonic()
            job.stages = []
            job.persist()
            self._bump()
        try:
            pipe = self._ensure_pipeline()
            if job.cancel_requested:
                raise InterruptedError("Annulé")
            {"song": self._run_song, "plan": self._run_plan, "decode": self._run_decode}[job.kind](pipe, job)
            job.status = "done"
        except InterruptedError:
            job.status, job.error = "cancelled", "Annulé par l'utilisateur"
        except guard.PlanTooLong as exc:
            job.status, job.error = "failed", str(exc)
        except BaseException as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            (job.output_dir / "traceback.txt").write_text(traceback.format_exc())
            traceback.print_exc()
            if "out of memory" in str(exc).lower():
                job.error += " — Mémoire GPU insuffisante : baissez le budget, activez « décharger l'AR » ou réduisez max_tokens."
        finally:
            with self.lock:
                job.finished_at = time.monotonic()
                self.current = None
                job.persist()
                self._bump()
            for listener in list(self.listeners):
                try:
                    listener(job)
                except Exception:
                    traceback.print_exc()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _configure(self, pipe, job: Job):
        from yue2.protocol import GenerationConfig
        pipe.generation_config = GenerationConfig(ode_steps=int(job.ode_steps))

    def _abc_streamer(self, pipe, job: Job):
        """Callback on_token : décode la partition au fil de l'eau pendant la planification."""
        from yue2.protocol import EOD
        ids: list[int] = []
        last = [0.0]

        def on_token(phase, token):
            if phase != "abc":
                return
            value = int(token)
            if value < EOD:
                ids.append(value)
            now = time.monotonic()
            if now - last[0] > 0.25:
                last[0] = now
                job.live_abc = pipe.tokenizer.decode(ids)
                job.live_tokens = len(ids)
                self._bump()
        return on_token

    def _run_song(self, pipe, job: Job):
        """Même séquence que ``YuE2Pipeline.__call__`` avec, entre la planification et la phase sémantique,
        un contrôle de la durée planifiée et une borne automatique des tokens sémantiques."""
        import time as _time
        from yue2.pipeline import SongResult
        from yue2.storage import identity
        self._configure(pipe, job)
        cancelled = lambda: job.cancel_requested
        request = pipe._request(**job.request)
        start = _time.perf_counter()
        plan = pipe.plan(request=request, abc_sampling=job.abc_sampling or None, cancelled=cancelled,
                         on_token=self._abc_streamer(pipe, job))
        job.live_abc = plan.abc or ""
        planned = guard.abc_duration(plan.abc) if plan.abc else None
        problem = guard.check_plan(planned["seconds"] if planned else None, job.max_seconds)
        trim_info = None
        if problem:
            trimmed = guard.trim_abc(plan.abc, job.max_seconds) if guard.should_trim(job.plan_overflow, request.lyrics) else None
            if trimmed is None:
                plan.save(job.output_dir)          # la partition reste consultable et éditable
                job.summary = {"has_audio": False, "has_score": plan.abc is not None, "planned": planned,
                               "truncated": {"abc": plan.truncated, "semantic": False}, "timing": {"abc": plan.timing}}
                raise guard.PlanTooLong(problem)
            job.output_dir.mkdir(parents=True, exist_ok=True)
            (job.output_dir / "score_full.abc").write_text(plan.abc, encoding="utf-8")
            plan = self._replan(pipe, plan, trimmed[0])
            planned, trim_info = guard.abc_duration(plan.abc), trimmed[1]
            job.live_abc = plan.abc
        user_max = (job.semantic_sampling or {}).get("max_tokens")
        cap, cap_info = guard.semantic_cap(user_max, planned["seconds"] if planned else None, job.max_seconds)
        semantic_sampling = dict(job.semantic_sampling or {})
        if cap < int(user_max or guard.DEFAULT_MAX_TOKENS):
            semantic_sampling["max_tokens"] = cap
        config = pipe.effective_config(request, job.abc_sampling or None, semantic_sampling or None)
        request_id = identity({"request": request.to_dict(), "config": config, "weights": pipe.weights})
        semantic = pipe.generate_semantic(plan, sampling=semantic_sampling or None, cancelled=cancelled)
        nar_start = _time.perf_counter()
        latents = pipe.synthesize(semantic, cancelled=cancelled)
        nar_seconds = _time.perf_counter() - nar_start
        if cancelled():
            raise InterruptedError("Annulé avant le décodage")
        vae_start = _time.perf_counter()
        audio = pipe.decode(latents)
        timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
                  "vae_seconds": _time.perf_counter() - vae_start, "load": dict(pipe.load_timing),
                  "e2e_seconds": _time.perf_counter() - start}
        result = SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, request_id)
        receipt = result.save_artifacts(job.output_dir)
        job.summary = {"audio_seconds": receipt["audio_seconds"], "truncated": receipt["truncated"],
                       "timing": receipt["timing"], "has_audio": True, "has_score": result.abc is not None,
                       "identity": receipt["identity"], "vae": job.engine.get("vae", "standard"),
                       "planned": planned, "guard": cap_info, "trim": trim_info, "trim_note": guard.trim_note(trim_info),
                       "truncation_note": guard.truncation_note(receipt["truncated"].get("semantic", False), cap_info)}

    @staticmethod
    def _replan(pipe, plan, abc: str):
        """Nouveau plan sur une partition raccourcie, avec le préfixe exact attendu par la phase sémantique."""
        from yue2.pipeline import SymbolicPlan
        from yue2.protocol import token_prefixes
        ids = pipe.tokenizer.encode(abc)
        return SymbolicPlan(plan.request, abc, ids, token_prefixes(plan.request, pipe.tokenizer, ids),
                            dict(plan.timing, trimmed_from_tokens=len(plan.abc_ids)), plan.truncated)

    def _run_plan(self, pipe, job: Job):
        self._configure(pipe, job)
        request = {k: v for k, v in job.request.items()}
        plan = pipe.plan(**request, abc_sampling=job.abc_sampling or None, cancelled=lambda: job.cancel_requested,
                         on_token=self._abc_streamer(pipe, job))
        plan.save(job.output_dir)
        job.live_abc = plan.abc or ""
        job.summary = {"truncated": {"abc": plan.truncated}, "timing": {"abc": plan.timing},
                       "has_audio": False, "has_score": plan.abc is not None}

    def _run_decode(self, pipe, job: Job):
        import numpy as np
        import soundfile as sf
        from yue2.storage import resolve_model
        source = self.jobs.get(job.source_job or "")
        if source is None or not (source.output_dir / "latent.npy").is_file():
            raise FileNotFoundError("Latents du job source introuvables")
        latents = np.load(source.output_dir / "latent.npy", allow_pickle=False)
        repo = EngineSettings(**{k: v for k, v in job.engine.items() if k in EngineSettings.__dataclass_fields__}).resolve_vae(job.decode_vae or "standard")
        if not repo:
            raise ValueError("Aucun VAE personnalisé n'est configuré dans le moteur")
        vae_path = resolve_model(repo, local_files_only=self.settings.offline)
        audio = pipe.decode(latents, vae=vae_path)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(job.output_dir / "audio.flac", audio, 48000, subtype="PCM_24")
        for name in ("score.abc", "request.json", "plan.json"):
            if (source.output_dir / name).is_file():
                shutil.copy(source.output_dir / name, job.output_dir / name)
        job.summary = {"audio_seconds": len(audio) / 48000, "truncated": source.summary.get("truncated", {}),
                       "has_audio": True, "has_score": (job.output_dir / "score.abc").is_file(),
                       "vae": job.decode_vae, "decoded_from": source.id}


ENGINE: Engine | None = None


def get_engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        ENGINE = Engine()
    return ENGINE
