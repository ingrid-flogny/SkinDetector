#!/usr/bin/env bash
# Calcule les embeddings Derm Foundation d'ISIC 2020 sur une VM Google Cloud, avec ou
# sans GPU.
#
# Déposer embed.py, ../data/manifest.csv et ce script dans le dossier personnel de la
# VM, puis lancer :
#     bash run_on_vm.sh
# Le script est relançable : chaque étape déjà faite est sautée, et le calcul reprend
# là où il s'est arrêté. Résultat : ~/training/data/embeddings/derm_foundation/isic2020.npz
set -euo pipefail

WORK=~/training
ISIC_URL=https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip

if command -v nvidia-smi >/dev/null; then
    echo "== GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    TENSORFLOW="tensorflow[and-cuda]" BATCH_SIZE=32
else
    echo "== Pas de GPU : calcul sur les $(nproc) processeurs"
    TENSORFLOW="tensorflow" BATCH_SIZE=16
fi

echo "== Fichiers"
mkdir -p "$WORK/data/isic2020"
[ -f ~/embed.py ] && mv ~/embed.py "$WORK/"
[ -f ~/manifest.csv ] && mv ~/manifest.csv "$WORK/data/"

echo "== Python et TensorFlow"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$WORK"
if [ ! -d .venv ]; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv "$TENSORFLOW" huggingface-hub numpy pillow
fi
if [ "$TENSORFLOW" != tensorflow ]; then
    .venv/bin/python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(gpus); assert gpus, 'TensorFlow ne voit pas le GPU'"
fi

echo "== Images ISIC 2020"
if [ ! -d data/isic2020/train ] || [ "$(ls data/isic2020/train | wc -l)" -lt 33126 ]; then
    command -v unzip >/dev/null || sudo apt-get install -y unzip
    curl -fL -C - -o data/isic2020/isic2020.zip "$ISIC_URL"
    unzip -q -n data/isic2020/isic2020.zip -d data/isic2020
    rm data/isic2020/isic2020.zip
fi
echo "$(ls data/isic2020/train | wc -l) images"

echo "== Embeddings"
if [ -z "${HF_TOKEN:-}" ]; then
    read -rsp "Jeton Hugging Face (ne s'affiche pas) : " HF_TOKEN
    echo
fi
# Le collage dans un terminal web peut ajouter des caractères invisibles, que Hugging
# Face rejette (« Bad request ») : on ne garde que ceux d'un jeton.
HF_TOKEN=$(printf %s "$HF_TOKEN" | tr -cd '[:alnum:]_')
# nohup : le calcul continue si la fenêtre SSH se ferme.
HF_TOKEN="$HF_TOKEN" nohup .venv/bin/python embed.py isic2020 --batch-size "$BATCH_SIZE" > embed.log 2>&1 &
echo "Calcul lancé. Suivre l'avancement avec : tail -f ~/training/embed.log"
