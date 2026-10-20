"""Validation croisée du classifieur sur un petit jeu de données.

Le jeu est coupé en 5 groupes de patients ; chaque groupe est noté par un classifieur
entraîné sur les 4 autres. Toutes les images reçoivent ainsi un score d'un modèle qui
ne les a jamais vues, ce qui donne une mesure plus stable que la seule partie test
quand les cas sont rares (52 mélanomes dans PAD-UFES-20, dont 7 dans la partie test).

    python cross_validate_probe.py pad_ufes_20
"""

import csv
import hashlib
import sys

import numpy as np

import metrics
from train_probe import DATA, EMBEDDINGS, MODEL_NAME, LogisticRegression, StandardScaler, make_pipeline

FOLDS = 5
C = 0.1  # valeur retenue par train_probe.py sur la partie validation


def main() -> None:
    dataset = sys.argv[1]
    saved = np.load(EMBEDDINGS / f"{dataset}.npz")
    vectors = dict(zip(saved["image_id"], saved["embedding"]))
    with open(DATA / "manifest.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == dataset and r["image_id"] in vectors]

    x = np.stack([vectors[r["image_id"]] for r in rows])
    y = np.array([int(r["malignant"]) for r in rows])
    melanoma = np.array([int(r["melanoma"]) for r in rows])
    fold = np.array([
        int(hashlib.sha256(f"cv:{r['patient_id']}".encode()).hexdigest(), 16) % FOLDS for r in rows
    ])

    scores = np.zeros(len(rows))
    for k in range(FOLDS):
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        ).fit(x[fold != k], y[fold != k])
        scores[fold == k] = model.predict_proba(x[fold == k])[:, 1]

    report = {"folds": FOLDS, "C": C, **metrics.evaluate(y, scores, melanoma)}
    print(metrics.save(MODEL_NAME, dataset, f"cv{FOLDS}", report).read_text())


if __name__ == "__main__":
    main()
