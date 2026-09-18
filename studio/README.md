# YuE2 Studio

Interface web locale pour piloter le pipeline YuE2 sans ComfyUI : FastAPI + HTMX + un peu de JavaScript.
Tout tourne sur ta machine, un seul processus, le modèle reste chargé en mémoire entre les jobs.

## Lancer

Depuis la racine du dépôt (le venv doit contenir `yue2-infer`, `fastapi`, `uvicorn`, `jinja2`, `python-multipart`) :

```bash
.venv/bin/python -m studio --port 8420
```

Puis ouvrir <http://127.0.0.1:8420>. Ajoute `--host 0.0.0.0` pour y accéder depuis une autre machine du réseau local.
`--reload` recharge le code à chaud pendant le développement. Au démarrage, le studio charge un éventuel `.env` à la racine
du dépôt (voir `.env.example`, même fichier que pour Docker Compose) : pratique pour `YUE2_LLM_API_KEY`, `YUE2_LLM_BASE_URL`
et `YUE2_LLM_MODEL` sans les exporter dans le shell.

## Ce que fait l'interface

- **Page Morceaux** (`/`) : grille de pochettes générées (couleur dérivée de l'identifiant), lecture d'un clic avec un lecteur
  global fixé en bas de page (forme d'onde, précédent / suivant, espace = lecture / pause), recherche sur nom, style et paroles,
  tri, renommage en ligne (clic sur le nom), téléchargement, suppression, lien « Ouvrir » vers le détail dans le studio.
  Les jobs en cours s'affichent au-dessus de la grille. L'atelier complet est sur `/studio`.

- **Formulaire complet** : style, paroles (avec insertion de balises de section), mode CoT (`full` / `melody` / `off`), seed, CFG,
  partition ABC fournie (import de fichier, aperçu abcjs), et tous les paramètres d'échantillonnage des deux phases
  autorégressives (température, top-p, top-k, pénalité de répétition, fenêtre, min/max tokens) ainsi que les pas ODE de synthèse.
  Chaque champ a un bouton « ? » ; l'interrupteur « Explications » en haut affiche toutes les aides d'un coup.
- **Estimation de durée** en direct dans le pied du formulaire : depuis les paroles et le BPM du style
  (≈ 2 mesures par ligne chantée + 9 mesures d'intro/outro, ±25 %), ou calcul nominal exact quand une partition ABC est fournie
  (mesures × temps par mesure ÷ BPM). Le badge passe en orange si `max_tokens` coupera avant la fin. Le détail d'un morceau
  compare la durée nominale de sa partition à l'audio réel.
- **Assistant de composition** (bouton ✨ Assistant dans le studio) : conversation avec un LLM via n'importe quel service
  OpenAI-compatible (OpenRouter, OpenAI, Mistral, ou en local Ollama, LM Studio, vLLM). Il pose au plus cinq questions, puis
  propose nom, style, paroles, mode, CFG et réglages d'échantillonnage avec une justification par choix et une estimation de
  durée ; « Appliquer au formulaire » verse le tout dans le studio, et une retouche en langage naturel régénère la proposition.
  Le prompt système est construit depuis `params.py` (bornes, défauts, explications) et les règles du skill. Les propositions
  sont bornées et validées par les dataclasses YuE2 avant application. Deux actions rapides utilisent le même connecteur :
  « ✨ Améliorer » sur le style et « ✨ Retravailler » sur les paroles (avec instruction facultative et annulation).
  Réglages dans ⚙︎ Moteur → Assistant LLM, ou par variables d'environnement `YUE2_LLM_API_KEY`, `YUE2_LLM_BASE_URL` et
  `YUE2_LLM_MODEL` (prioritaires : le champ correspondant est alors verrouillé dans l'interface) ; sinon les valeurs saisies
  vont dans `studio/data/assistant.json` (hors git, permissions 600). Bouton « Enregistrer et tester » qui interroge `/models`
  et fait un mini appel.
- **Projets et versions** : chaque conversation avec l'assistant est un projet persistant (`outputs/studio/projects/<id>/project.json`).
  Toute génération lancée depuis un projet devient une version (v1, v2…) qui référence le job sans le déplacer : audio, partition et
  réglages sont conservés. Page projet (`/projects/<id>`) : frise des versions avec lecteur, origine (proposition de l'assistant,
  retouche manuelle, morceau importé), résultat objectif (durée réelle, mesures planifiées, troncature), différences ligne à ligne
  entre versions, écoute A/B synchronisée, conversation complète, renommage, suppression avec ou sans les audios. Un morceau existant
  peut devenir la v1 d'un projet (« Retravailler avec l'assistant »). L'assistant reçoit à chaque tour un résumé des versions et le
  texte de référence de la dernière ; la graine du projet est conservée entre versions sauf demande de variation. Sur la page Morceaux,
  un projet occupe une seule pochette avec son compteur de versions.
- **Garde-fous de durée** (`guard.py`) : entre la planification et la phase sémantique, le moteur borne automatiquement les tokens
  sémantiques à la durée de la partition planifiée + 25 %, et, si une **durée maximale** est renseignée dans le formulaire, borne
  aussi à cette durée. Quand la partition planifiée dépasse la durée maximale de plus de 30 %, le champ **Partition trop longue**
  décide : `auto` (défaut) raccourcit la partition à des sections entières si le morceau est instrumental et arrête sinon,
  `trim` raccourcit toujours (partition complète conservée dans `score_full.abc`), `stop` annule avant la phase sémantique. Le formulaire
  avertit quand les paroles ne contiennent aucune ligne chantée (instrumental), cas où rien n'ancre la longueur. Le détail d'un
  morceau indique quelle borne a joué et explique une éventuelle troncature.
- **Moteur** (⚙︎) : modèle, VAE standard/legacy, révisions HF, device, budget VRAM, backend (`torch`, `torch-eager`, `vllm`),
  quantification FP8 expérimentale, déchargement de l'AR pendant la synthèse, fenêtre du décodeur, mode hors-ligne, vérification des empreintes.
  Les réglages sont persistés dans `studio/data/settings.json`.
- **File de jobs** : exécution séquentielle (le pipeline n'est pas concurrent), progression par étape en temps réel (SSE),
  annulation à chaud, préchargement / libération du modèle.
- **Deux pages, une seule navigation** : **Morceaux** (`/`) est la galerie ; ouvrir un morceau mène à sa fiche
  (`/morceaux/<id>`, liste des morceaux à gauche, lecteur / partition / réglages à droite, toujours sous l'onglet Morceaux).
  **Composer** (`/studio`) affiche le formulaire, la file d'attente et l'assistant, et rien d'autre. Depuis une fiche,
  « Réutiliser », « Variation » et « Éditer la partition » ouvrent Composer pré-rempli (`/studio?from_job=<id>&mode=…`) et le
  titre du formulaire indique l'origine des valeurs (« Nouvelle version de… », « Variation de… », etc.). La file d'attente de
  Composer liste les derniers morceaux terminés : le ▶ lance la lecture sur place (barre de lecture commune à toutes les
  pages, ⏮/⏭ entre les morceaux visibles), le nom ouvre la fiche. En fin de génération, le morceau qui vient d'apparaître
  est chargé tout seul dans la barre, en pause : il ne reste qu'à appuyer sur lecture (jamais pendant une autre écoute).
  L'assistant suit le morceau ouvert : il affiche la conversation de son projet, ou propose d'en faire la v1 d'un projet s'il
  n'en a pas ; depuis une fiche, sa proposition s'ouvre dans Composer (`/studio?apply=<projet>`).
  **Appliquer et générer** (case cochée par défaut dans le composeur de l'assistant, sur la page Composer) : chaque nouvelle
  proposition remplit le formulaire (swap « hors bande ») et part en file sans confirmation. Une modification faite à la main
  dans le formulaire suspend l'automatisme pour ce tour, et une proposition refusée par la validation remplit quand même le
  formulaire sans rien lancer. À la fin de la génération, la note de résultat (durée réelle, troncature, mesures planifiées)
  retourne dans la conversation avec un bouton ▶ pour écouter la version sur place ; elle fait partie du contexte envoyé au
  LLM au tour suivant, sans appel supplémentaire. La comparaison de mélodies ne
  propose que les versions du même projet et les morceaux partageant la partition.
- **Bibliothèque** : lecteur avec forme d'onde, téléchargement FLAC / WAV, partition rendue en portée + source ABC,
  requête, réglages effectifs (`config.json`), temps par étape, liste des fichiers.
- **Outils de partition** (onglet Partition, et dans le détail d'un morceau) : validation stricte par le vérificateur du skill
  `yue2-music` (structure, ties, altérations, vocabulaire d'accords, durée nominale), retrait des accords avec choix de la voix
  conservée (bascule automatiquement en mode `melody`), comparaison mélodique entre la partition éditée et son original ou entre
  deux morceaux de la bibliothèque (invariant de réharmonisation : mêmes notes, seuls les accords changent).
- **Import par lot** (onglet Lot) : `.jsonl` au format de `yue2 batch`, tableau `.json` ou `.csv`. Les champs absents héritent du
  formulaire courant. Un modèle `.jsonl` est téléchargeable. Les lignes refusées sont listées avec la raison.
- **Partition en direct** : pendant la planification, le texte ABC s'affiche au fil des tokens dans la carte du job en cours.
- **VAE personnalisé** : dans le panneau Moteur, choisir « personnalisé » et indiquer un dépôt HF ou un chemin local ; un bouton
  de re-décodage avec ce VAE apparaît alors dans le détail des morceaux.
- **Actions** : renommer (clic sur le titre), réutiliser les réglages, variation (nouvelle seed), éditer la partition (recharge `score.abc` dans le formulaire pour
  une réharmonisation puis re-rendu en `full`), re-décoder les mêmes latents avec l'autre VAE, planifier seulement, supprimer.

## Où vont les fichiers

Chaque job écrit dans `outputs/studio/<horodatage>-<nom>-<id>/` : les artefacts natifs YuE2 (`audio.flac`, `score.abc`, `plan.json`,
`semantic.npy`, `latent.npy`, `config.json`, `result.json`…) plus `studio.json` (métadonnées du job, étapes, erreurs).
La bibliothèque est reconstruite depuis ce dossier au démarrage : tu peux supprimer un dossier à la main.

## Tests

Suite pytest isolée (dossiers temporaires, aucun GPU, LLM et exécuteur de jobs simulés) :

```bash
.venv/bin/python -m pytest studio/tests -q
```

Couverture : catalogue de paramètres, parsing et validation des formulaires, import par lot, durée ABC, moteur (persistance, file,
renommage, réglages), projets (persistance, versions, diffs), assistant (normalisation, résumé de projet, tours de conversation,
actions rapides) et routes HTTP de bout en bout (génération, bibliothèque, outils ABC, assistant, projets).

## Instrumental (expérimental)

YuE2 n'a pas de mode instrumental : entraîné sur des chansons, il ajoute une voix même quand la mélodie vocale de la partition
planifiée est vide (constat partagé par la communauté, sans réponse des mainteneurs). La case **Instrumental** du formulaire
fusionne dans le modèle autorégressif une LoRA communautaire de rang 64 entraînée sur ≈ 2 700 morceaux instrumentaux
([Mothersuperior/YuE2-instrumental-cot-full-loras](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras),
licence **CC BY-NC 4.0**, usage non commercial). Concrètement :

- le mode de composition « full » est imposé (la LoRA écrit d'abord sa partition) ;
- les paroles sont réduites à leur structure au format de la LoRA : balises nues en minuscules parmi `[intro]`, `[verse]`,
  `[pre-chorus]`, `[chorus]`, `[bridge]`, `[outro]`, horodatage optionnel `[verse 0:15-0:45]`, ou `[instrumental]` seul.
  Les autres balises YuE2 sont converties (`[Interlude]` → `[bridge]`, `[Refrain]` → `[chorus]`…), le texte chanté est supprimé ;
- la fusion (`W += scale · B @ A`, formule du script de référence de l'auteur, sans PEFT) se fait job par job : passer d'une
  chanson à un instrumental ou l'inverse recharge le modèle depuis le cache (≈ 10 s), ce qui garantit des poids de base exacts ;
- réglages dans ⚙︎ Moteur → Instrumental : dépôt ou chemin local, nom du fichier, intensité (1.0 = tel qu'entraîné).

Limites : backend `vllm` et quantification FP8 non supportés ; résultat non garanti (la voix peut encore apparaître, l'auteur
signale des fins précoces) ; l'assistant coche la case quand on lui demande un instrumental. Le module est `lora.py`, testé par
`tests/test_lora.py` sur un modèle jouet.

## Déploiement en conteneur

Le dépôt fournit un `Dockerfile` (base officielle PyTorch 2.10 / CUDA 12.8, utilisateur non root) et un `docker-compose.yml`.
L'image est publiée sur GHCR à chaque tag `v*` : `ghcr.io/ralphi2811/yue2-studio`.

**Prérequis hôte** : Linux, pilote NVIDIA récent, [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
un GPU avec 24 Go de VRAM (BF16). Les modèles ne sont pas dans l'image : ils se téléchargent depuis Hugging Face au premier
chargement du moteur (≈ 7,3 Go) dans le volume `models`, puis restent en cache.

```bash
cp .env.example .env          # renseignez YUE2_LLM_API_KEY / _BASE_URL / _MODEL si vous voulez l'assistant (fichier ignoré par git)
docker compose up -d          # image publiée ; Compose lit le .env automatiquement
docker compose up -d --build  # ou construction locale
```

Volumes : `models` (cache Hugging Face), `outputs` (morceaux et projets), `data` (réglages du moteur et de l'assistant).
Variables du `.env` (voir `.env.example`) : `YUE2_LLM_API_KEY`, `YUE2_LLM_BASE_URL`, `YUE2_LLM_MODEL`, `YUE2_STUDIO_PORT`
(défaut 8420), `HF_TOKEN` (inutile pour les modèles publics). Les réglages LLM peuvent aussi être saisis dans l'interface
(⚙︎ Moteur → Assistant LLM) ; ils sont alors stockés dans le volume `data`. Un LLM local sur l'hôte (Ollama) est joignable depuis
le conteneur via `http://host.docker.internal:11434/v1`.

> **Aucune authentification n'est intégrée.** Le studio est pensé pour un poste local ou un réseau privé : toute personne qui
> atteint le port peut lancer des générations sur le GPU et consommer la clé LLM. Pour l'exposer, placez une identification en
> amont : Tailscale, reverse proxy avec authentification (Caddy, nginx, Traefik), Cloudflare Access, etc.

Hébergement : une machine personnelle avec GPU derrière Tailscale, ou un GPU loué à l'heure (RunPod, Vast.ai, Lambda…) avec
une carte 24 Go ; comptez le téléchargement des modèles au premier démarrage, d'où l'intérêt d'un volume persistant.

## Architecture

| Fichier | Rôle |
|---|---|
| `params.py` | Catalogue unique des paramètres : libellés, bornes, valeurs par défaut du runtime, textes d'aide. Alimente le formulaire et la validation. |
| `engine.py` | Pipeline résident (sous-classe de `YuE2Pipeline` qui capte les étapes de progression), thread de travail, file, persistance des jobs. |
| `app.py` | Routes FastAPI : pages, partials HTMX, SSE `/events`, fichiers, conversion WAV, import par lot, outils ABC. |
| `llm.py` | Connecteur OpenAI-compatible (httpx) : réglages, `/models`, `/chat/completions`, extraction JSON tolérante. |
| `assistant.py` | Prompt système généré depuis le catalogue, sessions de conversation, normalisation et validation des propositions, actions rapides. |
| `lora.py` | LoRA instrumentale (expérimental) : résolution du fichier, fusion des deltas dans le modèle AR, réécriture des paroles en structure. |
| `guard.py` | Garde-fous de durée : durée d'une partition ABC, borne sémantique, contrôle du plan, notes de troncature. |
| `projects.py` | Projets : persistance JSON, versions, diffs de requêtes et résumés de changements. |
| `paths.py` | Chemins des sorties et des réglages, surchargeables par `YUE2_STUDIO_OUTPUT` / `YUE2_STUDIO_DATA` (tests). |
| `abctools.py` | Pont vers `skills/yue2-music/scripts/abc_tools.py` (validation, retrait d'accords, comparaison). |
| `templates/` | Jinja2 : `base.html`, `landing.html` (Morceaux), `project.html` (frise), `index.html` (Studio) + partials (`form`, `queue`, `library`, `gallery`, `card`, `job_detail`, `settings`, `status`, `abc_check`, `batch_result`). |
| `static/` | `style.css`, `app.js` (onglets, aide, lecteur, aperçu ABC), bibliothèques vendorisées (htmx, extension SSE, wavesurfer, abcjs). |

Pour ajouter un paramètre : le déclarer dans `params.py`, le placer dans le template concerné, et le transmettre au pipeline dans `engine.py`.
