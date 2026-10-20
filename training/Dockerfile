# Environnement d'entraînement (TensorFlow, scikit-learn). Les scripts et les données
# ne sont pas copiés dans l'image : le dossier training/ est monté à l'exécution.
#
#   docker build -t skindetector-training training
#   docker run --rm -e HF_TOKEN -v "${PWD}/training:/training" -v skindetector-hf:/hf_cache \
#       skindetector-training python <script>.py
#
# Le cache des modèles Hugging Face (/hf_cache) doit être un volume Docker : il ne
# fonctionne pas dans un dossier partagé avec Windows.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /bin/uv

WORKDIR /training
# L'environnement est hors de /training, que le montage du dossier masquerait.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/hf_cache \
    PATH="/opt/venv/bin:$PATH"

# Secret optionnel `extra_ca` : voir le Dockerfile de l'application.
COPY pyproject.toml uv.lock ./
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then \
        cat /etc/ssl/certs/ca-certificates.crt /run/secrets/extra_ca > /tmp/ca.pem; \
        export SSL_CERT_FILE=/tmp/ca.pem; \
    fi; \
    uv sync --frozen --no-install-project && rm -f /tmp/ca.pem
