FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/hf \
    PATH="/app/.venv/bin:$PATH" \
    TF_CPP_MIN_LOG_LEVEL=2

# EXTRAS=ham10000 ajoute l'ancien modèle (PyTorch et ses poids, environ 3 Go de plus).
ARG EXTRAS=""

# Secret optionnel `extra_ca` : autorité de certification supplémentaire pour les postes
# dont le trafic HTTPS est inspecté (antivirus, proxy). Utilisé pendant le build
# uniquement, jamais copié dans l'image ; la CI s'en passe.
COPY pyproject.toml uv.lock ./
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then \
        cat /etc/ssl/certs/ca-certificates.crt /run/secrets/extra_ca > /tmp/ca.pem; \
        export SSL_CERT_FILE=/tmp/ca.pem REQUESTS_CA_BUNDLE=/tmp/ca.pem; \
    fi; \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS:+--extra $EXTRAS} \
    && if [ "$EXTRAS" = ham10000 ]; then \
        python -c "from huggingface_hub import hf_hub_download; hf_hub_download('pthennakoon25/skinvision-ai-model', 'best_model.pth', revision='9429e5a4ef81bb90064d8cdacc5895c4fc442cf7')"; \
    fi \
    && rm -f /tmp/ca.pem

# Derm Foundation n'est pas embarqué : son accès est soumis aux conditions de Google, il
# est téléchargé au premier démarrage avec HF_TOKEN (voir app/derm_foundation.py).
COPY app ./app

# OpenShift lance le conteneur avec un UID arbitraire appartenant au groupe 0.
RUN mkdir -p /opt/hf && chgrp -R 0 /app /opt/hf && chmod -R g=u /app /opt/hf
USER 1001

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
