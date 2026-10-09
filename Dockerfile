# Image du dashboard SequoIA : API de validation + site Quarto rendu.
# Les données (SQLite, modèles) ne sont jamais dans l'image : elles sont montées
# depuis la VM sur dashboard/data et mpnet_sgd_artifacts.
#
# BASE_IMAGE peut désigner une image qui contient déjà Quarto et les dépendances
# Python (cas de la VM, peu de disque) : les étapes d'installation deviennent
# alors quasi instantanées.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG QUARTO_VERSION=1.10.18
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN command -v quarto >/dev/null || ( \
    apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL -o /tmp/quarto.deb \
       "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb" \
    && apt-get install -y --no-install-recommends /tmp/quarto.deb \
    && rm -rf /tmp/quarto.deb /var/lib/apt/lists/* )

WORKDIR /sequoia
COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-optional.txt

COPY . .
RUN quarto render dashboard

EXPOSE 10000
CMD ["uvicorn", "dashboard.scripts.validation_api:app", "--host", "0.0.0.0", "--port", "10000"]
