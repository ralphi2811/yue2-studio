"""LoRA instrumentale (expérimental).

YuE2 n'a pas de mode instrumental : entraîné sur des chansons, il ajoute une voix même quand la voix
« Vocal » de la partition planifiée est vide. Une LoRA communautaire de rang 64 sur la branche
autorégressive (Mothersuperior/YuE2-instrumental-cot-full-loras, CC BY-NC 4.0) lui apprend à écrire
de la musique instrumentale avec un plan de sections. Ce module :

- résout et charge le fichier ``safetensors`` (cache Hugging Face ou chemin local) ;
- fusionne les deltas ``W += scale · B @ A`` dans les 7 projections linéaires de chaque couche
  (même formule que le script de référence de l'auteur, sans dépendance PEFT) ;
- réécrit les paroles au format attendu par la LoRA : balises nues en minuscules, une par ligne,
  parmi intro / verse / pre-chorus / chorus / bridge / outro, éventuellement horodatées
  (``[verse 0:15-0:45]``), ou ``[instrumental]`` seul.

La fusion modifie les poids en place : pour revenir au modèle de base, le moteur recharge le modèle.
"""
from __future__ import annotations

import re
from pathlib import Path

DEFAULT_REPO = "Mothersuperior/YuE2-instrumental-cot-full-loras"
DEFAULT_FILE = "ar_lora_inst_v3abc.bf16.safetensors"
LICENSE_NOTE = "Poids dérivés de YuE2-3B, licence CC BY-NC 4.0 : usage non commercial uniquement."

# Ordre du script de référence : self_attn q,k,v,o puis mlp gate,up,down, pour chaque couche.
PROJECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ("mlp", ("gate_proj", "up_proj", "down_proj")),
)

TAGS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "outro")
TAG_ALIASES = {
    "prechorus": "pre-chorus", "pre_chorus": "pre-chorus", "pre chorus": "pre-chorus",
    "refrain": "chorus", "hook": "chorus", "drop": "chorus",
    "interlude": "bridge", "solo": "bridge", "break": "bridge", "instrumental break": "bridge", "instrumental": "bridge",
    "end": "outro", "ending": "outro", "coda": "outro",
    "couplet": "verse", "pont": "bridge", "final": "outro",
}
_TIME = r"\d{1,2}:\d{2}"
_TAG_LINE = re.compile(rf"^\s*\[\s*([^\]]+?)\s*(?:({_TIME})\s*[-–]\s*({_TIME}))?\s*\]\s*$")


class LoRAError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Fichier
# --------------------------------------------------------------------------
def resolve_file(repo: str, filename: str = DEFAULT_FILE, *, offline: bool = False) -> Path:
    """Chemin local du fichier LoRA : un fichier ou dossier local est utilisé tel quel,
    sinon le fichier est téléchargé dans le cache Hugging Face (une seule fois)."""
    repo = (repo or "").strip()
    filename = (filename or DEFAULT_FILE).strip()
    if not repo:
        raise LoRAError("Aucun dépôt ou chemin de LoRA instrumentale n'est configuré (⚙︎ Moteur → Instrumental).")
    local = Path(repo).expanduser()
    if local.is_file():
        return local
    if local.is_dir():
        candidate = local / filename
        if not candidate.is_file():
            raise LoRAError(f"Fichier LoRA introuvable : {candidate}")
        return candidate
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo, filename, local_files_only=offline))
    except Exception as exc:   # réseau, dépôt inconnu, mode hors-ligne sans cache…
        raise LoRAError(f"Impossible de récupérer la LoRA « {repo} / {filename} » : {exc}") from exc


def load_tensors(path: Path) -> dict:
    """Tenseurs ``layers.{i}.{bloc}.{proj}.lora_{A,B}`` sur CPU."""
    from safetensors.torch import load_file
    try:
        tensors = load_file(str(path), device="cpu")
    except Exception as exc:
        raise LoRAError(f"Fichier LoRA illisible ({path.name}) : {exc}") from exc
    if not any(k.endswith(".lora_A") for k in tensors):
        raise LoRAError(f"{path.name} ne contient pas de tenseurs lora_A / lora_B : mauvais fichier ou format ComfyUI fusionné.")
    return tensors


def describe(tensors: dict) -> dict:
    """Résumé (couches, rang, nombre de tenseurs) pour les manifestes."""
    layers = sorted({int(k.split(".")[1]) for k in tensors if k.startswith("layers.")})
    first = next((v for k, v in tensors.items() if k.endswith(".lora_A")), None)
    return {"layers": len(layers), "rank": int(first.shape[0]) if first is not None else None, "tensors": len(tensors)}


# --------------------------------------------------------------------------
# Fusion dans le modèle
# --------------------------------------------------------------------------
def merge(model, tensors: dict, scale: float = 1.0) -> int:
    """Ajoute ``scale · B @ A`` au poids de chaque projection ciblée. Retourne le nombre de matrices modifiées.

    Le calcul se fait en float32 sur le périphérique du poids, puis est reconverti dans son dtype.
    Refuse un modèle FP8 (poids non additionnables) et une LoRA dont les formes ne correspondent pas.
    """
    import torch
    if getattr(model, "_yue2_fp8_originals", None):
        raise LoRAError("La LoRA ne peut pas être appliquée à un modèle quantifié FP8 : choisissez la quantification « none ».")
    backbone = getattr(model, "model", model)
    layers = list(getattr(backbone, "layers", []))
    if not layers:
        raise LoRAError("Modèle inattendu : pas de couches « model.layers ».")
    count = 0
    with torch.no_grad():
        for index, layer in enumerate(layers):
            for block, names in PROJECTIONS:
                module = getattr(layer, block)
                for name in names:
                    key = f"layers.{index}.{block}.{name}"
                    try:
                        a, b = tensors[f"{key}.lora_A"], tensors[f"{key}.lora_B"]
                    except KeyError as exc:
                        raise LoRAError(f"Tenseur manquant dans la LoRA : {exc.args[0]}") from exc
                    linear = getattr(module, name)
                    weight = linear.weight
                    if tuple(b.shape[:1] + a.shape[1:]) != tuple(weight.shape) or a.shape[0] != b.shape[1]:
                        raise LoRAError(f"Formes incompatibles pour {key} : W {tuple(weight.shape)}, A {tuple(a.shape)}, B {tuple(b.shape)}")
                    delta = float(scale) * (b.to(weight.device, torch.float32) @ a.to(weight.device, torch.float32))
                    weight.copy_((weight.to(torch.float32) + delta).to(weight.dtype))   # un seul arrondi
                    count += 1
    return count


# --------------------------------------------------------------------------
# Paroles au format de la LoRA
# --------------------------------------------------------------------------
def _canonical_tag(raw: str) -> str | None:
    name = re.sub(r"[\s_]+", " ", raw.strip().lower()).strip(" :.-")
    name = name.replace(" - ", "-")
    if name in TAGS:
        return name
    if name in TAG_ALIASES:
        return TAG_ALIASES[name]
    base = re.sub(r"\s*\d+\s*$", "", name)          # « verse 2 » → verse ; « chorus 1 » → chorus
    if base in TAGS:
        return base
    if base in TAG_ALIASES:
        return TAG_ALIASES[base]
    first = base.split(" ")[0] if base else ""
    if first in TAGS:
        return first
    return TAG_ALIASES.get(first)


def instrumental_lyrics(lyrics: str) -> str:
    """Réduit des paroles YuE2 à la structure attendue par la LoRA.

    Les lignes de texte chanté sont supprimées ; les balises de section sont ramenées aux six noms
    connus (en minuscules), les horodatages ``m:ss-m:ss`` sont conservés, les balises inconnues sont
    ignorées. Sans aucune balise reconnue : ``[instrumental]`` (le modèle choisit la structure).
    """
    out: list[str] = []
    for line in (lyrics or "").replace("\r\n", "\n").splitlines():
        m = _TAG_LINE.match(line)
        if not m:
            continue
        tag = _canonical_tag(m.group(1))
        if tag is None:
            continue
        if m.group(2) and m.group(3):
            out.append(f"[{tag} {m.group(2)}-{m.group(3)}]")
        else:
            out.append(f"[{tag}]")
    if out and all(l == "[bridge]" for l in out) and "instrumental" in (lyrics or "").lower():
        return "[instrumental]"
    return "\n".join(out) if out else "[instrumental]"
