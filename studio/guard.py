"""Garde-fous de durée : borne automatique dérivée de la partition planifiée et durée maximale.

YuE2 n'a pas de paramètre de durée. Deux dérives coûteuses existent :
- une requête sans texte chanté (instrumental, balises vides) n'ancre pas la longueur : le planificateur
  peut écrire une partition de plusieurs minutes et la phase sémantique court jusqu'à ``max_tokens`` ;
- une partition planifiée bien plus longue que voulu consomme la phase sémantique (la plus chère) en pure perte.

Ce module contient des fonctions pures, testées unitairement, utilisées par le moteur entre la
planification et la génération sémantique.
"""
from __future__ import annotations

import math
import re

TOKENS_PER_SECOND = 25       # tokens sémantiques par seconde d'audio (observé : 1652 → 66 s, 3532 → 141 s)
CAP_MARGIN = 1.25            # marge au-dessus de la durée de la partition planifiée
CAP_EXTRA_TOKENS = 100       # tokens de fin, respiration
PLAN_TOLERANCE = 1.3         # la partition peut dépasser la durée maximale de 30 % avant d'agir
DEFAULT_MAX_TOKENS = 9000    # défaut du runtime pour la phase sémantique
OVERFLOW_MODES = ("auto", "trim", "stop")   # partition trop longue : raccourcir si instrumental / toujours / jamais


class PlanTooLong(RuntimeError):
    """La partition planifiée dépasse largement la durée maximale demandée : on arrête avant la phase coûteuse."""


def abc_duration(text: str) -> dict | None:
    """Durée nominale d'une partition ABC native : Σ mesures × (M ÷ unité de Q) × 60 ÷ BPM, voix Vocal."""
    meter, beat, bpm, voice = (4, 4), (1, 4), None, None
    bars: dict[str, list] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        field = re.match(r"^([A-Za-z]):\s*(.*)$", line)
        if field:
            key, value = field.groups()
            if key == "M":
                m = re.match(r"(\d+)\s*/\s*(\d+)", value)
                if m:
                    meter = (int(m[1]), int(m[2]))
            elif key == "Q":
                q = re.search(r"(?:(\d+)\s*/\s*(\d+)\s*=\s*)?(\d+)", value)
                if q:
                    bpm = int(q[3])
                    beat = (int(q[1]), int(q[2])) if q[1] else (1, 4)
            elif key == "V":
                voice = value.split()[0] if value.split() else voice
            continue
        sec_per_bar = (meter[0] / meter[1]) / (beat[0] / beat[1]) * 60 / (bpm or 90)
        for seg in line.split("|"):
            seg = re.sub(r'"[^"]*"', "", seg).strip()
            if not seg:
                continue
            z = re.fullmatch(r"Z(\d*)", seg)
            n = int(z[1] or 1) if z else 1
            b = bars.setdefault(voice or "_", [0, 0.0])
            b[0] += n
            b[1] += n * sec_per_bar
    key = "Vocal" if "Vocal" in bars else next(iter(bars), None)
    if key is None:
        return None
    return {"bars": bars[key][0], "seconds": bars[key][1], "bpm": bpm, "meter": f"{meter[0]}/{meter[1]}"}


def sung_lines(lyrics: str) -> int:
    return sum(1 for l in (lyrics or "").splitlines() if l.strip() and not re.fullmatch(r"\[.+\]", l.strip()))


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f} s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s"


def check_plan(planned_seconds: float | None, max_seconds: int | None) -> str | None:
    """Message d'erreur si la partition planifiée dépasse la durée maximale au-delà de la tolérance, sinon None."""
    if not planned_seconds or not max_seconds:
        return None
    if planned_seconds > max_seconds * PLAN_TOLERANCE:
        return (f"Partition planifiée trop longue : {fmt(planned_seconds)} pour une durée maximale de {fmt(max_seconds)} "
                f"(champ « Durée maximale » de ce job, modifiable). Génération arrêtée avant la phase sémantique (la plus "
                f"coûteuse). La partition est enregistrée : relancez avec une autre graine, une durée maximale plus grande, "
                f"moins de sections ou moins de paroles, ou choisissez « raccourcir la partition » dans le formulaire.")
    return None


def should_trim(mode: str | None, lyrics: str) -> bool:
    """Faut-il raccourcir une partition trop longue plutôt qu'arrêter ? auto = seulement sans texte chanté."""
    mode = mode or "auto"
    if mode == "trim":
        return True
    if mode == "stop":
        return False
    return sung_lines(lyrics) == 0


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Sépare l'en-tête d'une partition native (jusqu'au premier commentaire ``% section``) de ses sections."""
    header, sections, current = [], [], None
    for raw in (text or "").splitlines():
        m = re.match(r"^%\s*(.*)$", raw.strip())
        if m:
            current = [m[1].strip().lower() or f"section {len(sections) + 1}", []]
            sections.append(current)
        elif current is None:
            header.append(raw)
        else:
            current[1].append(raw)
    return "\n".join(header) + "\n", [(name, "\n".join(lines) + "\n") for name, lines in sections]


def trim_abc(text: str, max_seconds: int | float) -> tuple[str, dict] | None:
    """Raccourcit une partition trop longue en gardant des sections entières jusqu'à la durée maximale.

    Les sections sont gardées dans l'ordre tant qu'elles tiennent dans la limite (la première toujours). Si la
    partition finit par une section « outro » qui a été écartée, on essaie de la remettre à la place de la dernière
    section gardée pour conserver une vraie fin. Retourne None si la partition n'a pas au moins deux sections.
    """
    header, sections = split_sections(text)
    if len(sections) < 2 or not max_seconds:
        return None
    durations = []
    for name, body in sections:
        d = abc_duration(header + "%" + name + "\n" + body)
        durations.append((d or {}).get("seconds", 0.0))
    kept, total = [0], durations[0]
    for i in range(1, len(sections)):
        if total + durations[i] <= max_seconds:
            kept.append(i)
            total += durations[i]
        else:
            break
    last = len(sections) - 1
    if last not in kept and "outro" in sections[last][0]:
        if total + durations[last] <= max_seconds:
            kept.append(last)
            total += durations[last]
        elif len(kept) > 1 and total - durations[kept[-1]] + durations[last] <= max_seconds:
            total = total - durations[kept.pop()] + durations[last]
            kept.append(last)
    if len(kept) == len(sections):
        return None
    new_text = header + "".join(f"% {sections[i][0]}\n{sections[i][1]}" for i in kept)
    before, after = abc_duration(text) or {}, abc_duration(new_text) or {}
    info = {"bars_before": before.get("bars"), "seconds_before": before.get("seconds"),
            "bars_after": after.get("bars"), "seconds_after": after.get("seconds"),
            "kept": [sections[i][0] for i in kept], "dropped": [s[0] for i, s in enumerate(sections) if i not in kept]}
    return new_text, info


def trim_note(info: dict | None) -> str | None:
    if not info:
        return None
    return (f"Partition raccourcie de {info['bars_before']} mesures ({fmt(info['seconds_before'])}) à {info['bars_after']} mesures "
            f"({fmt(info['seconds_after'])}) pour respecter la durée maximale : sections gardées {', '.join(info['kept'])} ; "
            f"écartées {', '.join(info['dropped'])}. La partition complète est conservée dans score_full.abc.")


def semantic_cap(user_max_tokens: int | None, planned_seconds: float | None, max_seconds: int | None) -> tuple[int, dict]:
    """Plafond effectif de tokens sémantiques et la raison retenue.

    Le plafond est le minimum entre : la valeur de l'utilisateur (ou le défaut du runtime), la durée de la
    partition planifiée avec marge, et la durée maximale demandée avec une petite marge.
    """
    user = int(user_max_tokens or DEFAULT_MAX_TOKENS)
    candidates = {"utilisateur": user}
    if planned_seconds:
        candidates["partition"] = math.ceil(planned_seconds * TOKENS_PER_SECOND * CAP_MARGIN) + CAP_EXTRA_TOKENS
    if max_seconds:
        candidates["durée maximale"] = math.ceil(max_seconds * TOKENS_PER_SECOND * 1.1)
    reason, value = min(candidates.items(), key=lambda kv: kv[1])
    value = max(1, value)
    info = {"max_tokens": value, "reason": reason, "candidates": candidates,
            "planned_seconds": planned_seconds, "max_seconds": max_seconds}
    return value, info


def truncation_note(truncated_semantic: bool, cap_info: dict | None) -> str | None:
    """Explication humaine d'une troncature sémantique, selon la borne qui a joué."""
    if not truncated_semantic:
        return None
    reason = (cap_info or {}).get("reason")
    if reason == "partition":
        return ("Coupé à la borne automatique dérivée de la partition planifiée (durée de la partition + 25 %) : "
                "le modèle a continué au-delà de sa propre partition. L'audio couvre normalement toute la partition.")
    if reason == "durée maximale":
        return "Coupé à la durée maximale demandée. Augmentez-la ou raccourcissez les paroles pour une fin plus naturelle."
    return "Le plafond max_tokens sémantique a été atteint. Augmentez la limite ou raccourcissez les paroles."
