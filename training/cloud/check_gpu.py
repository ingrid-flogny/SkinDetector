"""Vérifie que Derm Foundation donne les mêmes embeddings sur GPU que sur CPU.

À lancer sur la VM avant le calcul complet, depuis ~/training, deux fois :
    .venv/bin/python check_gpu.py                          # sur le GPU
    CUDA_VISIBLE_DEVICES= .venv/bin/python check_gpu.py    # sur le CPU, puis comparaison

Les deux calculs se font dans deux processus séparés : dans un même processus, les
poids du modèle restent sur le GPU et le CPU ne peut pas s'en servir.
"""

import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download

from embed import DATA, load_model, to_example

GPU_RESULT = Path("/tmp/check_gpu.npy")
IMAGES = 16

rows = [r for r in csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")) if r["dataset"] == "isic2020"]
examples = tf.constant([to_example(DATA / r["path"]) for r in rows[:IMAGES]])
infer = load_model(Path(
    snapshot_download("google/derm-foundation", allow_patterns=["*.pb", "variables/*"])
)).signatures["serving_default"]

device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
infer(inputs=examples[:2])  # première exécution : compilation
start = time.time()
embeddings = infer(inputs=examples)["embedding"].numpy()
print(f"{device} : {IMAGES / (time.time() - start):.1f} images/s")

if device == "GPU":
    np.save(GPU_RESULT, embeddings)
    print("Relancer avec CUDA_VISIBLE_DEVICES= devant la commande pour comparer au CPU.")
else:
    gpu = np.load(GPU_RESULT)
    cosine = (gpu * embeddings).sum(1) / np.linalg.norm(gpu, axis=1) / np.linalg.norm(embeddings, axis=1)
    print(f"écart max : {np.abs(gpu - embeddings).max():.4f}, similarité cosinus min : {cosine.min():.5f}")
    print("OK" if cosine.min() > 0.999 else "PROBLÈME : les embeddings GPU diffèrent")
