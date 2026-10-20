"""Derm Foundation (Google) + classifieur linéaire entraîné dans training/.

Résultats sur les parties test, jamais vues à l'entraînement (training/results/) :
- ISIC 2020 (dermatoscope, 6 162 images dont 106 mélanomes) : AUC 0,86 (0,83 à 0,89) ;
- PAD-UFES-20 (smartphone, 453 images) : AUC 0,96 (0,94 à 0,98).

Derm Foundation n'est pas embarqué dans l'image : son accès est soumis aux conditions de
Google, il est téléchargé au premier démarrage avec le jeton HF_TOKEN.
"""

import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
from PIL import Image

from app.skin import SkinAnalysis, Source

# Révision figée : celle qui a servi à calculer les embeddings d'entraînement. Une autre
# version de Derm Foundation donnerait des embeddings que le classifieur ne connaît pas.
REVISION = "a16a6ab4f87888948fe248136e697ed28146a1c6"
IMAGE_SIZE = 448
# Classifieur exporté par training/export_probe.py : uniquement des nombres, aucun code.
PROBE = Path(__file__).parent / "probe.npz"


class DermFoundationLogReg:
    def __init__(self) -> None:
        path = snapshot_download(
            "google/derm-foundation", revision=REVISION, allow_patterns=["*.pb", "variables/*"]
        )
        self._infer = tf.saved_model.load(path).signatures["serving_default"]
        probe = np.load(PROBE)
        self._mean, self._scale = probe["mean"], probe["scale"]
        self._coef, self._intercept = probe["coef"], float(probe["intercept"])
        self._thresholds = {
            "smartphone": float(probe["threshold_smartphone"]),
            "dermatoscope": float(probe["threshold_dermatoscope"]),
        }

    def analyze(self, image: Image.Image, source: Source) -> SkinAnalysis:
        # Préparation identique à celle de l'entraînement (training/embed.py).
        resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        resized.save(buffer, "PNG")
        feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[buffer.getvalue()]))
        example = tf.train.Example(features=tf.train.Features(feature={"image/encoded": feature}))
        embedding = self._infer(inputs=tf.constant([example.SerializeToString()]))["embedding"].numpy()[0]
        logit = ((embedding - self._mean) / self._scale) @ self._coef + self._intercept
        return SkinAnalysis(
            model="Derm Foundation + régression logistique (PAD-UFES-20 + ISIC 2020)",
            malignancy_score=float(1 / (1 + np.exp(-logit))),
            alert_threshold=self._thresholds[source],
        )
