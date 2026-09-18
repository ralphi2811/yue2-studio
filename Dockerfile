# YuE2 Studio — image GPU (CUDA 12.8, torch 2.10). Les modèles ne sont PAS embarqués : ils se téléchargent
# depuis Hugging Face au premier chargement dans le volume /models (licence MODEL_LICENSE, ~7,3 Go).
#
#   docker build -t yue2-studio .
#   docker compose up -d          # voir docker-compose.yml (GPU, volumes, variables)
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ARG UID=1000
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/models \
    YUE2_STUDIO_OUTPUT=/outputs \
    YUE2_STUDIO_DATA=/data \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# L'image de base a déjà un utilisateur avec l'UID 1000 (ubuntu) : on le renomme plutôt que d'en créer un second.
RUN if getent passwd "${UID}" >/dev/null; then usermod -l studio -d /home/studio -m "$(getent passwd "${UID}" | cut -d: -f1)"; \
    else useradd --create-home --uid "${UID}" studio; fi \
    && usermod -G "" studio \
    && mkdir -p /models /outputs /data \
    && chown studio /models /outputs /data

WORKDIR /app
# Dépendances d'abord (cache de build), code ensuite.
COPY pyproject.toml MANIFEST.in README.md LICENSE MODEL_LICENSE THIRD_PARTY_NOTICES.md ./
COPY licenses ./licenses
COPY src ./src
COPY studio/requirements.txt ./studio/requirements.txt
RUN pip install . -r studio/requirements.txt pytest

COPY skills ./skills
COPY studio ./studio
RUN chown -R studio /app

USER studio
VOLUME ["/models", "/outputs", "/data"]
EXPOSE 8420
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8420/api/state', timeout=4)"

CMD ["python", "-m", "studio", "--host", "0.0.0.0", "--port", "8420"]
