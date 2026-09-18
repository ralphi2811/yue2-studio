"""Assistant de composition : questionnaire court puis proposition complète de paramètres YuE2.

Le prompt système est construit à partir du catalogue ``params.py`` (bornes, défauts,
explications) et des règles d'usage du skill officiel, pour que le LLM raisonne sur les
vrais paramètres du runtime. Les propositions sont validées par les dataclasses YuE2
avant d'être versées dans le formulaire.

Depuis l'introduction des projets, la conversation vit dans un ``Project`` persistant
(voir ``projects.py``) et l'assistant reçoit à chaque tour un résumé des versions
générées et de leur résultat objectif (durée réelle, troncature, mesures planifiées).
"""
from __future__ import annotations

import json
import re

from . import params as P
from . import llm
from .projects import Project, ProjectStore, Version, diff_requests, summarize_changes
from .guard import sung_lines

BARS_PER_LINE, OVERHEAD_BARS, EMPTY_SECTION_BARS, DEFAULT_BPM = 2, 9, 4, 90

_STORE: ProjectStore | None = None


def store() -> ProjectStore:
    global _STORE
    if _STORE is None:
        _STORE = ProjectStore()
    return _STORE


# --------------------------------------------------------------------------
# Heuristique de durée (même règle que le JavaScript du formulaire)
# --------------------------------------------------------------------------
def parse_bpm(style: str) -> int | None:
    m = re.search(r"(\d{2,3})\s*bpm", style or "", re.I)
    if m and 30 <= int(m.group(1)) <= 300:
        return int(m.group(1))
    return None


def estimate_duration(lyrics: str, style: str) -> dict:
    lines = [l.strip() for l in (lyrics or "").splitlines() if l.strip()]
    tags = [i for i, l in enumerate(lines) if re.fullmatch(r"\[.+\]", l)]
    sung = len(lines) - len(tags)
    empty = sum(1 for i in tags if i + 1 >= len(lines) or re.fullmatch(r"\[.+\]", lines[i + 1]))
    bpm = parse_bpm(style)
    bars = OVERHEAD_BARS + BARS_PER_LINE * sung + EMPTY_SECTION_BARS * empty
    return {"seconds": bars * 4 * 60 / (bpm or DEFAULT_BPM), "bars": bars, "bpm": bpm, "sung_lines": sung}


# --------------------------------------------------------------------------
# Prompt système
# --------------------------------------------------------------------------
def _param_lines(catalog) -> str:
    out = []
    for p in catalog:
        bounds = ""
        if p.kind in ("int", "float"):
            bounds = f" [{p.min}..{p.max}], défaut {p.default}"
        elif p.kind == "select":
            bounds = " ∈ {" + ", ".join(c[0] for c in p.choices) + "}" + (f", défaut {p.default}" if p.default is not None else "")
        out.append(f"- {p.key}{bounds} : {p.help}")
    return "\n".join(out)


PROPOSAL_SCHEMA = {
    "type": "questions | proposal",
    "message": "texte en français affiché à l'utilisateur (questions numérotées, ou résumé de la proposition / des changements)",
    "proposal": {
        "name": "identifiant court du morceau (lettres, chiffres, _ -)",
        "style": "prompt de style EN ANGLAIS, séparé par des virgules, avec la langue chantée et le BPM",
        "lyrics": "paroles dans la langue demandée, balises [Intro]/[Verse]/[Pre-Chorus]/[Chorus]/[Bridge]/[Outro], une phrase par ligne",
        "instrumental": "true UNIQUEMENT si l'utilisateur veut un morceau sans voix (le studio applique alors une LoRA instrumentale, impose cot=full et réduit les paroles à leurs balises), sinon false",
        "cot": "full | melody | off",
        "cfg_scale": "nombre 0..20 ou null (défaut du modèle)",
        "seed": "entier ou null (null = conserver la graine du projet)",
        "ode_steps": "entier ou null (32 = référence)",
        "semantic_sampling": {"temperature": "…", "top_p": "…", "top_k": "…", "repetition_penalty": "…", "penalty_window": "…", "min_tokens": "…", "max_tokens": "…"},
        "abc_sampling": {"…": "mêmes clés, uniquement si justifié"},
        "target_duration_seconds": "durée visée en secondes (entier)",
        "max_duration_seconds": "durée maximale en secondes (entier) : garde-fou, typiquement cible × 1.4 ; OBLIGATOIRE pour un instrumental",
        "rationale": {"style": "pourquoi", "lyrics": "pourquoi", "cot": "pourquoi", "cfg_scale": "pourquoi", "sampling": "pourquoi", "duration": "comment la longueur des paroles a été calibrée"}
    }
}


def system_prompt() -> str:
    return f"""Tu es l'assistant de composition de YuE2 Studio. YuE2 est un modèle open source de génération de chansons
(voix + accompagnement, 48 kHz) piloté par un prompt de style, des paroles et des paramètres d'échantillonnage.
Ton rôle : comprendre ce que veut l'utilisateur en peu de questions, puis produire UNE proposition complète,
directement exploitable, en justifiant chaque choix. Tu écris en français avec l'utilisateur.

## Déroulé
1. Si l'utilisateur n'a pas encore donné assez d'éléments, pose AU PLUS 5 questions courtes et numérotées en un seul
   message (type = "questions"). Sujets utiles : langue chantée et thème/histoire, genre et références, type de voix,
   énergie et tempo, durée visée, contraintes (mots à placer ou éviter, structure, refrain existant).
   Ne repose jamais une question déjà répondue. Si tout est clair dès le départ, passe directement à la proposition.
2. Dès que tu as assez d'éléments, réponds avec type = "proposal" et un objet "proposal" complet.
3. Si l'utilisateur demande une retouche, renvoie une NOUVELLE proposition complète (type = "proposal") en ne changeant
   que ce qui est demandé, et résume le changement dans "message".
4. Quand un projet existe déjà (voir « État du projet » plus bas), pars TOUJOURS de la dernière version générée :
   c'est le texte de référence. Ne réinvente pas ce que l'utilisateur n'a pas demandé de changer. Tu ne peux pas
   écouter l'audio : appuie-toi sur le retour d'écoute de l'utilisateur et sur les mesures objectives fournies
   (durée réelle, troncature, mesures et tempo planifiés).

## Règles YuE2 à respecter absolument
- Le champ style est TOUJOURS en anglais, virgules entre les idées : langue chantée en premier (ex. "French"), genre,
  instruments, caractère de la voix (female/male, warm, raspy…), tempo en BPM explicite, ambiance. Pas de prompt négatif :
  décrire ce qu'on veut, jamais ce qu'on évite. Pas de consignes techniques dans les paroles : tout ce qui est dans
  lyrics sera chanté.
- Paroles : dans la langue demandée, structurées avec des balises de section entre crochets sur leur propre ligne,
  une phrase musicale par ligne, lignes de longueur régulière au sein d'une section (compte les syllabes), rimes
  naturelles, refrain mémorisable et répété tel quel. Une balise sans texte ([Intro], [Interlude], [Outro]) produit
  un passage instrumental.
- Durée : YuE2 n'a pas de paramètre de durée. Repère calibré : environ 2 mesures par ligne chantée + 9 mesures
  d'intro/outro + 4 mesures par section instrumentale vide, en 4/4 au BPM du style. Durée ≈ mesures × 4 × 60 / BPM.
  Exemples à 90 BPM : 8 lignes ≈ 1 min 07 s, 16 lignes ≈ 1 min 49 s, 28 lignes ≈ 2 min 53 s, 36 lignes ≈ 3 min 36 s.
  Calibre le nombre de lignes sur la durée visée et explique le calcul dans rationale.duration. Si une version
  générée a une durée réelle connue, sers-t'en pour recalibrer.
- Mode cot : "full" par défaut (partition mélodie + accords éditable, le plus contrôlable) ; "melody" si l'utilisateur
  veut un accompagnement libre ou fera une reprise ; "off" seulement s'il veut aller vite sans partition.
- cfg_scale : laisse null (défaut validé) sauf demande explicite d'adhérence plus forte au style/paroles ; alors 1.2,
  jamais plus de 2 sans raison. Coût : génération plus lente.
- seed : laisse null pour conserver la graine du projet (les différences entre versions viennent alors des changements
  voulus, pas du hasard). Propose un entier seulement si l'utilisateur veut « une autre variation ».
- Échantillonnage : les défauts sont le protocole de référence des auteurs. Ne renvoie semantic_sampling / abc_sampling
  QUE si un réglage précis est justifié (ex. max_tokens sémantique ≈ 25 tokens par seconde d'audio pour borner une durée
  très courte ; température légèrement plus haute pour plus de surprise). Sinon null.
- ode_steps : null (32) sauf demande.
- max_duration_seconds : toujours renseigné (≈ cible × 1.4). C'est un garde-fou : le studio borne les tokens sémantiques et
  annule avant la phase coûteuse si la partition planifiée dépasse cette durée de plus de 30 %.
- Instrumental (morceau sans voix) : mets "instrumental": true et cot "full". Le studio fusionne alors une LoRA
  instrumentale (expérimentale) dans le modèle et réduit les paroles à leurs balises : écris les paroles comme une simple
  structure, une balise par ligne en minuscules parmi [intro] [verse] [pre-chorus] [chorus] [bridge] [outro], avec si
  utile un horodatage « [verse 0:15-0:45] » qui guide les proportions ; ou « [instrumental] » seul pour laisser le modèle
  choisir. Aucun texte chanté, aucune note de production dans les crochets. Décris le style sans mots de voix. Fixe
  max_duration_seconds (cible × 1.3) : la durée d'un instrumental est moins prévisible. Préviens l'utilisateur que la
  voix peut encore apparaître (fonction expérimentale) et que la LoRA est sous licence non commerciale (CC BY-NC).
- name : court, sans espaces ni accents.

## Paramètres du runtime (bornes, défauts, explications)
### Requête
{_param_lines(P.REQUEST)}
### Échantillonnage de la partition (abc_sampling)
{_param_lines(P.ABC_SAMPLING)}
### Échantillonnage sémantique (semantic_sampling)
{_param_lines(P.SEMANTIC_SAMPLING)}
### Synthèse
{_param_lines(P.GENERATION)}

## Format de réponse
Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, de la forme :
{json.dumps(PROPOSAL_SCHEMA, ensure_ascii=False, indent=1)}
"proposal" vaut null quand type = "questions". Les valeurs "…" ci-dessus sont des descriptions, pas des valeurs.
"""


QUICK_PROMPTS = {
    "style": (
        "Améliore ce prompt de style YuE2. Garde l'intention, la langue chantée et le BPM s'ils sont présents (sinon "
        "ajoute un BPM cohérent), rends-le plus précis et évocateur : instruments concrets, caractère de la voix, "
        "ambiance, production. Anglais, virgules, 15 à 35 mots, pas de négation. "
        'Réponds en JSON : {"style": "...", "why": "explication courte en français"}.'
    ),
    "lyrics": (
        "Retravaille ces paroles pour YuE2 en conservant la langue, le thème, la structure de sections et le nombre "
        "approximatif de lignes. Améliore la prosodie chantée (syllabes régulières par ligne au sein d'une section, "
        "accents naturels), les rimes, la force du refrain. Garde les balises de section. Si une instruction "
        "est donnée, applique-la en priorité. "
        'Réponds en JSON : {"lyrics": "...", "why": "explication courte en français"}.'
    ),
}


# --------------------------------------------------------------------------
# Résumé du projet injecté dans le contexte du LLM
# --------------------------------------------------------------------------
def _job_request(jobs, job_id: str) -> tuple[dict, dict]:
    job = (jobs or {}).get(job_id)
    if job is None:
        return {}, {}
    return dict(job.request), {"ode_steps": job.ode_steps, "semantic_sampling": job.semantic_sampling, "abc_sampling": job.abc_sampling}


def project_summary(project: Project, jobs) -> str:
    """Texte compact : versions, changements entre elles, résultat, et contenu complet de la dernière version."""
    if not project.versions:
        return ""
    lines = [f"## État du projet « {project.name} »", f"Graine du projet : {project.seed if project.seed is not None else 'non fixée'}."]
    prev_req, prev_extra = None, None
    for v in project.versions:
        req, extra = _job_request(jobs, v.job_id)
        change = ""
        if prev_req is not None:
            change = " · " + summarize_changes(diff_requests(prev_req, req, before_extra=prev_extra, after_extra=extra))
        outcome = v.outcome or {}
        result = []
        if outcome.get("status") and outcome["status"] != "done":
            result.append(f"statut {outcome['status']}")
        if outcome.get("audio_seconds"):
            result.append(f"durée réelle {outcome['audio_seconds']:.0f} s")
        tr = outcome.get("truncated") or {}
        if tr.get("abc") or tr.get("semantic"):
            result.append("TRONQUÉE")
        if outcome.get("planned_bars"):
            result.append(f"{outcome['planned_bars']} mesures planifiées à {outcome.get('planned_bpm')} BPM")
        if not outcome:
            result.append("génération en cours ou non terminée")
        lines.append(f"- v{v.number} ({v.source}, {v.created[:16]}){change} · " + ", ".join(result))
        prev_req, prev_extra = req, extra
    latest_req, latest_extra = _job_request(jobs, project.latest.job_id)
    if latest_req:
        lines.append("")
        lines.append(f"### Dernière version générée (v{project.latest.number}) — texte de référence")
        lines.append(f"cot : {latest_req.get('cot')} · seed : {latest_req.get('seed')} · cfg_scale : {latest_req.get('cfg_scale')}"
                     + (f" · ode_steps : {latest_extra.get('ode_steps')}" if latest_extra.get("ode_steps") not in (None, 32) else "")
                     + (f" · semantic_sampling : {latest_extra.get('semantic_sampling')}" if latest_extra.get("semantic_sampling") else ""))
        lines.append(f"style : {latest_req.get('style', '')}")
        lines.append("lyrics :\n" + (latest_req.get("lyrics") or ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Normalisation / validation d'une proposition
# --------------------------------------------------------------------------
def _num(value, kind, lo, hi, default=None):
    if value is None or value == "":
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    v = min(max(v, lo), hi)
    return int(round(v)) if kind == "int" else v


def normalize_proposal(raw: dict) -> tuple[dict, list[str]]:
    """Ramène la proposition dans les bornes du runtime ; retourne (proposition, avertissements)."""
    warnings: list[str] = []
    prop = dict(raw or {})
    prop["name"] = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prop.get("name") or "proposition")).strip("._-")[:60] or "proposition"
    prop["style"] = " ".join(str(prop.get("style") or "").split())
    prop["lyrics"] = str(prop.get("lyrics") or "").replace("\r\n", "\n").strip()
    if prop.get("cot") not in {"full", "melody", "off"}:
        if prop.get("cot"):
            warnings.append(f"mode cot inconnu « {prop.get('cot')} », remplacé par full")
        prop["cot"] = "full"
    instrumental = prop.get("instrumental")
    prop["instrumental"] = instrumental is True or str(instrumental).strip().lower() in {"true", "1", "yes", "oui"}
    if prop["instrumental"] and prop["cot"] != "full":
        warnings.append(f"instrumental : mode {prop['cot']} remplacé par full (exigé par la LoRA)")
        prop["cot"] = "full"
    cfg = prop.get("cfg_scale")
    prop["cfg_scale"] = None if cfg in (None, "", "null", "auto") else _num(cfg, "float", 0, 20)
    prop["seed"] = _num(prop.get("seed"), "int", 0, 2**31 - 1)
    prop["ode_steps"] = _num(prop.get("ode_steps"), "int", 1, 128)
    for key, catalog in (("semantic_sampling", P.SEMANTIC_SAMPLING), ("abc_sampling", P.ABC_SAMPLING)):
        sampling = prop.get(key)
        clean = {}
        if isinstance(sampling, dict):
            for p in catalog:
                field_name = p.key.split(".", 1)[1]
                if field_name in sampling and sampling[field_name] not in (None, ""):
                    v = _num(sampling[field_name], p.kind, p.min, p.max)
                    if v is not None and v != p.default:
                        clean[field_name] = v
        prop[key] = clean
    if not prop["style"]:
        warnings.append("style vide")
    if not prop["lyrics"]:
        warnings.append("paroles vides")
    if not parse_bpm(prop["style"]):
        warnings.append("le style ne précise pas de BPM (le modèle en choisira un)")
    est = estimate_duration(prop["lyrics"], prop["style"])
    prop["estimated_seconds"] = est["seconds"]
    prop["estimated_bars"] = est["bars"]
    target = _num(prop.get("target_duration_seconds"), "int", 5, 3600)
    prop["target_duration_seconds"] = target
    max_d = _num(prop.get("max_duration_seconds"), "int", 5, 3600)
    if max_d is None and target:
        max_d = min(3600, int(round(target * 1.4)))
    prop["max_duration_seconds"] = max_d
    if prop["instrumental"]:
        warnings.append("instrumental (expérimental) : LoRA appliquée, paroles réduites aux balises de section ; la voix peut encore apparaître")
    elif sung_lines(prop["lyrics"]) == 0 and prop["lyrics"]:
        warnings.append("aucune ligne chantée : durée non ancrée par les paroles, la durée maximale fera office de borne ; "
                        "pour un vrai instrumental, cochez « Instrumental »")
    if target and abs(est["seconds"] - target) > max(20, 0.25 * target):
        warnings.append(f"durée estimée {est['seconds']:.0f} s pour une cible de {target} s : ajustez le nombre de lignes")
    rationale = prop.get("rationale")
    prop["rationale"] = {k: str(v) for k, v in rationale.items()} if isinstance(rationale, dict) else {}
    # Validation finale par le runtime
    try:
        from yue2.protocol import GenerationConfig, SongRequest, resolve_sampling
        req = {"id": prop["name"], "style": prop["style"] or "x", "lyrics": prop["lyrics"] or "x", "cot": prop["cot"],
               "seed": prop["seed"] if prop["seed"] is not None else 831001}
        if prop["cfg_scale"] is not None:
            req["cfg_scale"] = prop["cfg_scale"]
        SongRequest(**req)
        defaults = GenerationConfig()
        resolve_sampling(prop["semantic_sampling"] or None, defaults.semantic)
        resolve_sampling(prop["abc_sampling"] or None, defaults.abc)
    except Exception as exc:
        warnings.append(f"refusé par le runtime : {exc}")
    return prop, warnings


def proposal_to_form(prop: dict, project: Project | None = None) -> dict:
    """Valeurs de formulaire (clés du catalogue) à partir d'une proposition normalisée.

    La graine du projet est conservée si la proposition n'en impose pas.
    """
    values = {p.key: p.default for p in P.ALL_PARAMS.values()}
    values.update(name=prop["name"], style=prop["style"], lyrics=prop["lyrics"], cot=prop["cot"],
                  cfg_scale=prop["cfg_scale"], abc="")
    if prop.get("seed") is not None:
        values["seed"] = prop["seed"]
    elif project is not None and project.seed is not None:
        values["seed"] = project.seed
    if prop.get("ode_steps"):
        values["ode_steps"] = prop["ode_steps"]
    values["instrumental"] = bool(prop.get("instrumental"))
    if prop.get("max_duration_seconds"):
        values["max_duration"] = prop["max_duration_seconds"]
    for prefix in ("semantic", "abc"):
        for k, v in (prop.get(f"{prefix}_sampling") or {}).items():
            values[f"{prefix}.{k}"] = v
    if project is not None:
        values["project_id"] = project.id
    values["origin_kind"] = "assistant"
    values["origin_name"] = prop["name"]
    return values


def proposal_matches_request(prop: dict | None, request: dict) -> bool:
    """Vrai si la requête lancée reprend telle quelle la dernière proposition (style + paroles + mode)."""
    if not prop:
        return False
    norm = lambda s: " ".join((s or "").split())
    return (norm(prop.get("style")) == norm(request.get("style")) and norm(prop.get("lyrics")) == norm(request.get("lyrics"))
            and prop.get("cot") == request.get("cot"))


# --------------------------------------------------------------------------
# Cycle de vie des projets
# --------------------------------------------------------------------------
def create_from_job(job, *, name: str | None = None) -> Project:
    """Nouveau projet dont la v1 est un morceau existant ; l'assistant partira de son contenu."""
    project = store().create(name or job.name, seed=job.request.get("seed"))
    store().add_version(project, job.id, source="import", label=job.name)
    project.turns.append({"role": "note", "text": f"Projet créé depuis le morceau « {job.name} » (v1)."})
    project.turns.append({"role": "assistant", "text": "Voici le morceau tel qu'il existe : je pars de son style, de ses "
                          "paroles et de ses réglages. Dites-moi ce que vous voulez changer (ambiance, refrain, durée, voix…) "
                          "ou ce qui vous a gêné à l'écoute, et je vous proposerai une nouvelle version.", "proposal": None, "warnings": []})
    if job.summary:
        outcome = outcome_from_job(job)
        v = project.versions[0]
        v.outcome = outcome
    return store().save(project)


def outcome_from_job(job, planned: dict | None = None) -> dict:
    s = job.summary or {}
    out = {"status": job.status, "error": job.error, "audio_seconds": s.get("audio_seconds"),
           "truncated": s.get("truncated") or {}, "e2e_seconds": (s.get("timing") or {}).get("e2e_seconds")}
    if planned:
        out["planned_bars"], out["planned_bpm"], out["planned_seconds"] = planned.get("bars"), planned.get("bpm"), planned.get("seconds")
    return out


# --------------------------------------------------------------------------
# Tours de conversation
# --------------------------------------------------------------------------
async def step(project: Project, user_text: str, *, context: dict | None = None, jobs=None) -> None:
    """Ajoute le message utilisateur, interroge le LLM, range la réponse dans le projet et le sauvegarde."""
    text = user_text.strip()
    if context and not project.messages and not project.versions:
        ctx = {k: v for k, v in context.items() if v}
        if ctx:
            text = f"{text}\n\n[Contexte du formulaire actuel, à réutiliser si pertinent : {json.dumps(ctx, ensure_ascii=False)[:2500]}]"
    project.messages.append({"role": "user", "content": text})
    project.turns.append({"role": "user", "text": user_text.strip()})
    system = [{"role": "system", "content": system_prompt()}]
    summary = project_summary(project, jobs)
    if summary:
        system.append({"role": "system", "content": summary})
    try:
        content, usage = await llm.chat([*system, *project.recent_messages()])
        for k in ("prompt_tokens", "completion_tokens"):
            project.usage[k] += int(usage.get(k) or 0)
        data = llm.extract_json(content)
    except Exception as exc:
        project.turns.append({"role": "assistant", "text": None, "error": str(exc)})
        store().save(project)
        return
    project.messages.append({"role": "assistant", "content": content})
    turn = {"role": "assistant", "text": str(data.get("message") or "").strip(), "proposal": None, "warnings": []}
    if data.get("type") == "proposal" and isinstance(data.get("proposal"), dict):
        prop, warnings = normalize_proposal(data["proposal"])
        project.proposal = prop
        turn["proposal"], turn["warnings"] = prop, warnings
        if project.name in ("Nouveau projet", "") and prop.get("name"):
            project.name = prop["name"]
    project.turns.append(turn)
    store().save(project)


async def quick(action: str, payload: dict) -> dict:
    """Actions rapides sur un champ : améliorer le style, retravailler les paroles."""
    if action not in QUICK_PROMPTS:
        raise llm.LLMError("Action inconnue")
    parts = [QUICK_PROMPTS[action]]
    if action == "style":
        parts.append(f"Prompt actuel : {payload.get('style') or '(vide)'}")
        if payload.get("lyrics"):
            parts.append(f"Extrait des paroles (pour la cohérence) : {payload['lyrics'][:600]}")
    else:
        parts.append(f"Style du morceau : {payload.get('style') or '(non précisé)'}")
        parts.append(f"Paroles actuelles :\n{payload.get('lyrics') or '(vides)'}")
    if payload.get("instruction"):
        parts.append(f"Instruction de l'utilisateur : {payload['instruction']}")
    content, _ = await llm.chat([{"role": "system", "content": "Tu es un parolier et directeur artistique expert de YuE2. Réponds uniquement en JSON."},
                                 {"role": "user", "content": "\n\n".join(parts)}], max_tokens=2500)
    data = llm.extract_json(content)
    if action == "style":
        return {"value": " ".join(str(data.get("style") or "").split()), "why": str(data.get("why") or "")}
    return {"value": str(data.get("lyrics") or "").replace("\r\n", "\n").strip(), "why": str(data.get("why") or "")}
