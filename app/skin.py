"""Analyse d'une photo de lésion cutanée.

Le modèle est un composant remplaçable : pour en brancher un autre, écrire une classe
qui respecte `SkinClassifier`, l'ajouter à `CLASSIFIERS`, puis la sélectionner avec la
variable d'environnement SKIN_MODEL. Le reste de l'application ne change pas.

Modèles disponibles :
- derm-foundation-logreg (par défaut) : Derm Foundation de Google + classifieur entraîné
  sur PAD-UFES-20 et ISIC 2020 (app/derm_foundation.py) ;
- ham10000-efficientnet-b0 : le premier modèle, conservé pour pouvoir y revenir
  (app/ham10000.py). Demande l'image construite avec --build-arg EXTRAS=ham10000.
"""

import os
from functools import lru_cache
from typing import Literal, Protocol

from PIL import Image
from pydantic import BaseModel

SKIN_MODEL = os.getenv("SKIN_MODEL", "derm-foundation-logreg")

# Type d'appareil qui a pris la photo : les seuils d'alerte en dépendent.
Source = Literal["smartphone", "dermatoscope"]


class ClassScore(BaseModel):
    code: str
    label: str
    probability: float
    malignant: bool


class SkinAnalysis(BaseModel):
    model: str
    # Score de 0 à 1, croissant avec le risque de malignité. Ce n'est pas une
    # probabilité calibrée : il se lit par rapport au seuil d'alerte.
    malignancy_score: float
    # Seuil qui détectait 95 % des lésions malignes en validation, pour ce type de photo.
    # Absent si le modèle n'en fournit pas.
    alert_threshold: float | None = None
    # Détail par type de lésion, pour les modèles qui le fournissent.
    classes: list[ClassScore] = []


class SkinClassifier(Protocol):
    def analyze(self, image: Image.Image, source: Source) -> SkinAnalysis: ...


def _derm_foundation() -> SkinClassifier:
    from app.derm_foundation import DermFoundationLogReg

    return DermFoundationLogReg()


def _ham10000() -> SkinClassifier:
    from app.ham10000 import Ham10000EfficientNet

    return Ham10000EfficientNet()


# Imports différés : chaque modèle n'importe ses dépendances que s'il est choisi.
CLASSIFIERS = {
    "derm-foundation-logreg": _derm_foundation,
    "ham10000-efficientnet-b0": _ham10000,
}


@lru_cache(maxsize=1)
def get_classifier() -> SkinClassifier:
    return CLASSIFIERS[SKIN_MODEL]()
