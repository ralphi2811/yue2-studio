"""Catalogue unique des paramètres exposés par le studio.

Chaque entrée alimente à la fois le formulaire (libellé, aide, bornes) et
le parsing/validation côté serveur. Les valeurs par défaut sont celles du
runtime YuE2 (``yue2.protocol``), pas des choix du studio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from yue2.protocol import GenerationConfig

_DEFAULTS = GenerationConfig()


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str                      # text | textarea | int | float | bool | select
    default: Any = None
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()   # (valeur, libellé)
    placeholder: str = ""
    rows: int = 3
    advanced: bool = False
    nullable: bool = False         # champ vide => None (défaut du runtime)


# --------------------------------------------------------------------------
# Requête de chanson
# --------------------------------------------------------------------------
REQUEST: tuple[Param, ...] = (
    Param("name", "Nom du morceau", "text", "mon_morceau",
          "Sert de nom de dossier et d'identifiant de la requête. Lettres, chiffres, tiret, "
          "point et underscore uniquement (180 caractères max). Le studio le nettoie si besoin.",
          placeholder="ex. nuit_neon"),
    Param("style", "Style (prompt)", "textarea",
          "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, "
          "lyrical memorable melody, unhurried phrasing, 88 BPM",
          "Description libre en anglais du morceau : langue chantée, genre, instruments, caractère de la voix "
          "(female/male, warm, raspy…), tempo en BPM, ambiance. Une virgule entre chaque idée. "
          "Ce texte conditionne à la fois la partition planifiée et le rendu audio. "
          "Il n'y a pas de prompt négatif : décrivez ce que vous voulez, pas ce que vous évitez.",
          rows=3),
    Param("lyrics", "Paroles", "textarea",
          "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\nFold the night and leave it here\n"
          "Morning has a sky to clear\n\n[Chorus]\nLet the day come into view\nEvery road begins with you\n"
          "Hold a little room for light\nWe will sing beyond the night",
          "Les paroles chantées, structurées par des balises de section : [Intro], [Verse], [Pre-Chorus], "
          "[Chorus], [Bridge], [Outro], [Interlude]. Une ligne par phrase musicale. La durée du morceau "
          "dépend surtout de la quantité de paroles : comptez environ 2 mesures par ligne chantée, plus une "
          "intro et un outro instrumentaux (≈ 9 mesures), soit 5 à 6 s par ligne à 88 BPM. L'estimation en bas "
          "du formulaire se met à jour en direct. Gardez les consignes techniques hors des paroles : "
          "tout ce qui est écrit ici est destiné à être chanté.",
          rows=12),
    Param("instrumental", "Instrumental (expérimental)", "bool", False,
          "Morceau sans voix. YuE2 n'a pas de mode instrumental natif : entraîné sur des chansons, il ajoute une voix "
          "même quand la mélodie vocale planifiée est vide. Cette option fusionne dans le modèle une LoRA communautaire "
          "entraînée sur 2 700 morceaux instrumentaux (Mothersuperior/YuE2-instrumental-cot-full-loras, licence CC BY-NC 4.0, "
          "usage non commercial). Conséquences : le mode de composition « full » est imposé, les paroles sont réduites à "
          "leurs balises de section en minuscules ([intro], [verse], [pre-chorus], [chorus], [bridge], [outro], horodatage "
          "optionnel « [verse 0:15-0:45] »), ou « [instrumental] » seul si aucune balise n'est reconnue. Le modèle est "
          "rechargé (≈ 10 s) à chaque passage entre chanson et instrumental. Résultat non garanti : la voix peut "
          "encore apparaître, et l'auteur signale des fins précoces. Backend vllm et quantification FP8 non supportés."),
    Param("cot", "Mode de composition (CoT)", "select", "full",
          "« full » : YuE2 écrit d'abord une partition ABC complète (mélodie + accords) puis la réalise en audio. "
          "C'est le mode par défaut, le plus contrôlable : vous récupérez une partition éditable. "
          "« melody » : planifie la mélodie seule, l'accompagnement reste libre ; recommandé pour les reprises. "
          "« off » : génère l'audio directement depuis le texte, sans partition. Plus rapide, aucun score éditable.",
          choices=(("full", "full — mélodie + accords (défaut)"),
                   ("melody", "melody — mélodie seule, accompagnement libre"),
                   ("off", "off — audio direct, sans partition"))),
    Param("seed", "Graine (seed)", "int", 831001,
          "Initialise le générateur aléatoire. Même graine + mêmes réglages + même machine = même résultat. "
          "Changez-la pour obtenir une autre variation du même prompt. Entier entre 0 et 2⁶³.",
          min=0, max=2**63 - 1, step=1),
    Param("cfg_scale", "Guidage CFG (cfg_scale)", "float", None,
          "Intensité du classifier-free guidance pendant la génération des tokens sémantiques. "
          "Vide = défaut du modèle (1.0 en full/melody, 1.01 en off), c'est le réglage validé par les auteurs. "
          "Une valeur > 1 (ex. 1.2) force davantage l'adhérence au style et aux paroles au prix d'une "
          "génération plus lente (deux passes) et sans garantie de qualité. Plage 0 à 20. "
          "La planification de la partition n'utilise pas de CFG.",
          min=0, max=20, step=0.05, nullable=True, placeholder="auto"),
    Param("max_duration", "Durée maximale (secondes)", "int", None,
          "Garde-fou optionnel. YuE2 n'a pas de paramètre de durée : cette limite borne le nombre de tokens sémantiques "
          "(≈ 25 par seconde) et, en mode full ou melody, arrête le job avant la phase coûteuse si la partition planifiée "
          "dépasse la limite de plus de 30 %. Indispensable pour un morceau instrumental (paroles sans texte chanté), "
          "dont la longueur n'est ancrée par rien. Indépendamment de ce champ, la phase sémantique est toujours bornée à la "
          "durée de la partition planifiée + 25 %. Vide = aucune limite explicite.",
          min=5, max=3600, step=5, nullable=True, placeholder="aucune"),
    Param("plan_overflow", "Partition trop longue", "select", "auto",
          "Que faire quand la partition planifiée dépasse la durée maximale de plus de 30 % (mode full ou melody). "
          "« auto » : raccourcir la partition si le morceau est instrumental (aucune ligne chantée), sinon arrêter. "
          "« raccourcir » : garder des sections entières (intro, couplets…) jusqu'à la durée maximale, en réintégrant "
          "l'outro si elle tient, puis générer l'audio sur cette partition ; la partition complète est conservée dans "
          "score_full.abc. Sur un morceau chanté, les paroles des sections écartées restent dans la requête : le résultat "
          "peut être moins cohérent, préférez raccourcir les paroles. « arrêter » : annuler avant la phase sémantique et "
          "garder la partition pour la retoucher.",
          choices=[("auto", "auto : raccourcir si instrumental, sinon arrêter"), ("trim", "raccourcir la partition"),
                   ("stop", "arrêter le job")]),
    Param("abc", "Partition ABC fournie", "textarea", "",
          "Optionnel. Collez ici une partition au format ABC natif YuE2 (deux voix Vocal et Ins, accords entre "
          "guillemets). Elle remplace la planification : le modèle réalise exactement cette partition. "
          "Requiert le mode full (partition avec accords) ou melody (partition sans accords, pour une reprise). "
          "Incompatible avec le mode off. Le rendu ci-dessous est indicatif (abcjs) et peut différer des "
          "conventions natives sur les altérations.",
          rows=10, placeholder="X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=88\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
                               "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:C\n…"),
)

# --------------------------------------------------------------------------
# Échantillonnage : partition (abc) et tokens sémantiques (semantic)
# --------------------------------------------------------------------------
_SAMPLING_HELP = {
    "temperature": "Aplatit (>1) ou accentue (<1) la distribution avant tirage. 0 = choix déterministe du "
                   "token le plus probable (greedy). Plus haut = plus de surprise et plus de risque d'incohérence.",
    "top_p": "Nucleus sampling : on ne tire que parmi les tokens dont la probabilité cumulée atteint cette "
             "valeur. 1.0 désactive le filtre. Plus bas = plus conservateur.",
    "top_k": "Ne conserve que les k tokens les plus probables avant tirage. Se combine avec top_p "
             "(le plus restrictif des deux l'emporte).",
    "repetition_penalty": "Pénalise les tokens déjà émis dans la fenêtre récente. 1.0 = aucune pénalité. "
                          "Trop haut casse les structures répétitives légitimes (refrains, motifs).",
    "penalty_window": "Nombre de tokens récents pris en compte par la pénalité de répétition (1 à 100).",
    "min_tokens": "Le token de fin est interdit avant ce nombre de tokens émis. Empêche les sorties trop courtes.",
    "max_tokens": "Plafond dur de tokens. Si atteint, la sortie est marquée « tronquée » (truncated) "
                  "mais reste exploitable.",
}


def _sampling_params(prefix: str, defaults, extra: dict[str, str]) -> tuple[Param, ...]:
    return (
        Param(f"{prefix}.temperature", "Température", "float", defaults.temperature,
              _SAMPLING_HELP["temperature"], min=0, max=5, step=0.05, advanced=True),
        Param(f"{prefix}.top_p", "Top-p", "float", defaults.top_p,
              _SAMPLING_HELP["top_p"], min=0.01, max=1, step=0.01, advanced=True),
        Param(f"{prefix}.top_k", "Top-k", "int", defaults.top_k,
              _SAMPLING_HELP["top_k"], min=1, max=5000, step=1, advanced=True),
        Param(f"{prefix}.repetition_penalty", "Pénalité de répétition", "float", defaults.repetition_penalty,
              _SAMPLING_HELP["repetition_penalty"], min=0.5, max=3, step=0.005, advanced=True),
        Param(f"{prefix}.penalty_window", "Fenêtre de pénalité", "int", defaults.penalty_window,
              _SAMPLING_HELP["penalty_window"], min=1, max=100, step=1, advanced=True),
        Param(f"{prefix}.min_tokens", "Tokens minimum", "int", defaults.min_tokens,
              _SAMPLING_HELP["min_tokens"] + " " + extra.get("min_tokens", ""), min=0, max=20000, step=1, advanced=True),
        Param(f"{prefix}.max_tokens", "Tokens maximum", "int", defaults.max_tokens,
              _SAMPLING_HELP["max_tokens"] + " " + extra.get("max_tokens", ""), min=1, max=20000, step=1, advanced=True),
    )


ABC_SAMPLING = _sampling_params("abc", _DEFAULTS.abc, {
    "max_tokens": "Pour la partition, 4096 tokens couvrent largement une chanson de 3 à 4 minutes.",
})
SEMANTIC_SAMPLING = _sampling_params("semantic", _DEFAULTS.semantic, {
    "max_tokens": "Repère : environ 25 tokens sémantiques par seconde d'audio. 9000 ≈ 6 minutes. "
                  "Baissez cette valeur pour borner la durée et le temps de calcul.",
    "min_tokens": "200 tokens ≈ 8 secondes d'audio.",
})

GENERATION: tuple[Param, ...] = (
    Param("ode_steps", "Pas de synthèse (ODE steps)", "int", _DEFAULTS.ode_steps,
          "Nombre de pas de l'intégrateur (méthode midpoint) du flow matching qui transforme les tokens "
          "sémantiques en latents acoustiques. 32 est le protocole de référence. Moins de pas = plus rapide "
          "mais rendu potentiellement plus terne ; davantage n'améliore pas forcément. Non quantifié par les auteurs.",
          min=1, max=128, step=1, advanced=True),
)

# --------------------------------------------------------------------------
# Moteur (nécessite un rechargement du pipeline)
# --------------------------------------------------------------------------
ENGINE: tuple[Param, ...] = (
    Param("model", "Modèle génératif", "text", "m-a-p/YuE2-3B",
          "Identifiant Hugging Face ou chemin local du modèle Mixture-of-Transformers 3B. "
          "Téléchargé automatiquement dans le cache HF au premier usage (7,3 Go)."),
    Param("vae", "Décodeur audio (VAE)", "select", "standard",
          "« standard » (YuE2-Vae) : décodeur d'écoute par défaut, recommandé. « legacy » (YuE2-Vae-legacy) : "
          "décodeur utilisé pour les benchmarks publiés, à réserver aux comparaisons. Vous pouvez aussi "
          "re-décoder un morceau existant avec l'autre VAE depuis la bibliothèque sans régénérer la musique.",
          choices=(("standard", "standard — YuE2-Vae (écoute)"), ("legacy", "legacy — YuE2-Vae-legacy (benchmark)"),
                   ("custom", "personnalisé — dépôt HF ou chemin local ci-dessous"))),
    Param("vae_custom", "VAE personnalisé", "text", "",
          "Utilisé seulement si « personnalisé » est sélectionné : identifiant Hugging Face (ex. m-a-p/YuE2-Vae) ou "
          "chemin local d'un export YuE2VAE (dossier contenant config.json et model.safetensors). Permet de tester un "
          "décodeur alternatif ou une révision locale sans toucher aux deux décodeurs officiels.",
          nullable=True, placeholder="m-a-p/YuE2-Vae ou /chemin/vers/vae"),
    Param("revision", "Révision du modèle", "text", "",
          "Commit ou tag Hugging Face à épingler pour des comparaisons reproductibles. Vide = dernière version.",
          nullable=True, placeholder="main"),
    Param("vae_revision", "Révision du VAE", "text", "",
          "Commit ou tag Hugging Face du décodeur audio, à épingler pour reproduire exactement un rendu. Vide = dernière version.",
          nullable=True, placeholder="main"),
    Param("device", "Périphérique", "select", "auto",
          "« auto » choisit CUDA si disponible. Le CPU n'est pas un mode supporté (BF16 CUDA requis pour le "
          "préréglage non quantifié).",
          choices=(("auto", "auto"), ("cuda", "cuda"), ("cuda:0", "cuda:0"), ("cuda:1", "cuda:1"))),
    Param("budget", "Budget VRAM (GiB)", "float", 24,
          "Plafond mémoire imposé au processus (moins une réserve de 2 GiB). Le studio en déduit aussi la taille "
          "des fenêtres du décodeur VAE : 512 frames si ≤ 12, sinon 1024. Baissez-le si votre bureau occupe déjà "
          "de la VRAM ou si vous rencontrez un OOM.",
          min=4, max=96, step=0.5),
    Param("backend", "Backend d'inférence", "select", "torch",
          "« torch » : PyTorch avec CUDA graphs (rapide, défaut). « torch-eager » : sans CUDA graphs, plus lent "
          "mais utile pour déboguer ou en FP8. « vllm » : requiert l'extra optionnel `pip install .[fast]`, "
          "non installé par défaut.",
          choices=(("torch", "torch (CUDA graphs)"), ("torch-eager", "torch-eager"), ("vllm", "vllm (extra fast)"))),
    Param("quantization", "Quantification", "select", "none",
          "« fp8 » (expérimental) : quantifie les couches linéaires de la phase autorégressive en FP8 pour "
          "économiser de la VRAM. Nécessite une compute capability ≥ 8.9 (RTX 40xx / Ada et plus récent). "
          "Les poids BF16 restent en RAM pour être restaurés avant la synthèse. Désactive les CUDA graphs. "
          "Aucune garantie de qualité selon les auteurs.",
          choices=(("none", "none — BF16 (référence)"), ("fp8", "fp8 — expérimental"))),
    Param("offload_ar", "Décharger l'AR pendant la synthèse", "bool", False,
          "Pendant la phase acoustique (flow matching), déplace les couches autorégressives inutilisées vers la RAM "
          "système puis les restaure. Réduit le pic VRAM au prix de transferts PCIe. Indolore avec beaucoup de RAM."),
    Param("vae_core_frames", "Fenêtre du décodeur VAE", "select", "auto",
          "Nombre de frames latentes décodées par tuile. « auto » suit le budget VRAM. Plus petit = moins de VRAM, "
          "plus de tuiles (le rendu reste exact, sans crossfade, grâce au halo de 16 frames).",
          choices=(("auto", "auto (selon budget)"), ("256", "256"), ("512", "512"), ("1024", "1024"), ("2048", "2048"))),
    Param("offline", "Mode hors-ligne", "bool", False,
          "N'interroge jamais Hugging Face : n'utilise que les fichiers déjà présents dans le cache local."),
    Param("verify_hashes", "Vérifier les empreintes des poids", "bool", True,
          "Calcule le SHA-256 des fichiers modèle au chargement (quelques secondes). Garantit l'identité exacte "
          "des poids dans les manifestes de résultats."),
    Param("instrumental_lora", "LoRA instrumentale", "text", "Mothersuperior/YuE2-instrumental-cot-full-loras",
          "Dépôt Hugging Face, dossier ou fichier local de la LoRA appliquée quand la case « Instrumental » d'un job est "
          "cochée. Téléchargée dans le cache HF au premier usage (≈ 140 Mo en bf16). Ne nécessite pas de rechargement du "
          "pipeline : la fusion se fait job par job. Licence de la LoRA par défaut : CC BY-NC 4.0 (non commercial).",
          placeholder="Mothersuperior/YuE2-instrumental-cot-full-loras"),
    Param("instrumental_lora_file", "Fichier dans le dépôt", "text", "ar_lora_inst_v3abc.bf16.safetensors",
          "Nom du fichier safetensors (format lora_A / lora_B par projection, pas la variante ComfyUI fusionnée). "
          "La version fp32 « ar_lora_inst_v3abc.safetensors » donne le même résultat pour deux fois plus d'octets.",
          placeholder="ar_lora_inst_v3abc.bf16.safetensors"),
    Param("instrumental_lora_scale", "Intensité de la LoRA", "float", 1.0,
          "Facteur appliqué au delta de poids (1.0 = tel qu'entraîné). En dessous de 1, le modèle de base reprend la main : "
          "plus de variété, plus de risque de voix.",
          min=0.0, max=2.0, step=0.05),
)

ALL_PARAMS: dict[str, Param] = {p.key: p for p in (*REQUEST, *ABC_SAMPLING, *SEMANTIC_SAMPLING, *GENERATION, *ENGINE)}

LYRIC_TAGS = ("[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Interlude]", "[Outro]")

STYLE_PRESETS: tuple[tuple[str, str], ...] = (
    ("Pop piano chaleureuse", "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"),
    ("Chanson française intimiste", "French, intimate chanson, soft male voice, nylon guitar, upright bass, brushed drums, gentle accordion touches, 76 BPM"),
    ("Jazz-funk", "English, jazz-funk, warm lead vocal, Rhodes electric piano, slap bass, tight drums, horn stabs, 104 BPM"),
    ("Rock alternatif", "English, alternative rock, raspy male vocal, distorted electric guitars, driving bass, powerful drums, anthemic chorus, 128 BPM"),
    ("Électro pop", "English, synth pop, bright female vocal, analog synthesizers, punchy electronic drums, sidechained pads, catchy hook, 118 BPM"),
    ("Ballade acoustique", "English, acoustic folk ballad, tender female vocal, fingerpicked acoustic guitar, light strings, soft harmonies, 72 BPM"),
    ("Hip-hop lo-fi", "English, lo-fi hip hop, relaxed male vocal, dusty drum loop, warm bass, vinyl crackle, mellow keys, 84 BPM"),
    ("R&B contemporain", "English, contemporary R&B, smooth female vocal with runs, electric piano, deep 808 bass, crisp trap hi-hats, 92 BPM"),
)

# Aide contextuelle affichée dans les panneaux (pas liée à un champ)
PANEL_HELP = {
    "workflow": (
        "Le pipeline enchaîne quatre étapes : 1) planification de la partition ABC (sauf mode off), "
        "2) génération des tokens sémantiques (la « musique » sous forme discrète), 3) synthèse des latents "
        "acoustiques par flow matching, 4) décodage en audio 48 kHz stéréo par le VAE. Chaque étape est "
        "sauvegardée : vous pouvez rejouer le décodage avec un autre VAE sans régénérer."
    ),
    "plan_only": (
        "« Planifier seulement » exécute uniquement l'étape 1 et enregistre la partition. Vous pouvez ensuite "
        "l'ouvrir, la modifier (accords, tempo, structure) et la soumettre comme partition fournie pour un rendu."
    ),
}


def group(params: tuple[Param, ...]) -> list[dict]:
    return [p.__dict__ for p in params]
