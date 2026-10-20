"""Exporte le classifieur PAD-UFES-20 + ISIC 2020 pour l'application (app/probe.npz).

Le pickle de scikit-learn est remplacé par de simples tableaux de nombres (aucun code
exécuté au chargement), accompagnés d'un seuil d'alerte par type de photo : le seuil
qui détecte 95 % des lésions malignes de la partie validation de chaque jeu (smartphone :
PAD-UFES-20, dermatoscope : ISIC 2020). La partie test n'est pas utilisée ici.

    python export_probe.py
"""

import pickle
from pathlib import Path

import numpy as np

import metrics
from train_probe import DATA, MODEL_NAME, load

DATASETS = ["pad_ufes_20", "isic2020"]
SOURCES = {"smartphone": "pad_ufes_20", "dermatoscope": "isic2020"}
OUT = Path(__file__).parent.parent / "app" / "probe.npz"


def main() -> None:
    name = f"{MODEL_NAME}_{'+'.join(DATASETS)}"
    with open(DATA / "models" / f"{name}.pkl", "rb") as f:
        pipeline = pickle.load(f)
    scaler, logreg = pipeline[0], pipeline[-1]

    val = load(DATASETS)["val"]
    scores = pipeline.predict_proba(val["x"])[:, 1]
    thresholds = {}
    for source, dataset in SOURCES.items():
        mask = val["dataset"] == dataset
        thresholds[source] = metrics.threshold_at_sensitivity(
            val["y"][mask], scores[mask], metrics.TARGET_SENSITIVITY
        )

    weights = {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
        "coef": logreg.coef_[0].astype(np.float32),
        "intercept": np.float32(logreg.intercept_[0]),
        **{f"threshold_{source}": np.float32(t) for source, t in thresholds.items()},
    }
    # Le calcul refait à la main doit donner les mêmes scores que scikit-learn.
    logits = ((val["x"] - weights["mean"]) / weights["scale"]) @ weights["coef"] + weights["intercept"]
    gap = np.abs(1 / (1 + np.exp(-logits)) - scores).max()
    assert gap < 1e-4, gap

    np.savez(OUT, **weights)
    print(f"{OUT} : seuils {thresholds}, écart avec scikit-learn {gap:.1e}")


if __name__ == "__main__":
    main()
