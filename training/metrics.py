"""Mesures communes à tous les modèles, pour que leurs résultats soient comparables.

Entrées : pour chaque image, l'étiquette (1 = malin), le score du modèle (plus il est
haut, plus la lésion est jugée maligne) et un indicateur mélanome.
"""

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
TARGET_SENSITIVITY = 0.95


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Probabilité qu'une lésion maligne reçoive un score plus haut qu'une bénigne."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    # Rang moyen de chaque score, les ex æquo partageant le même rang.
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    positives = labels == 1
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_at_sensitivity(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    """Seuil le plus haut qui détecte encore au moins `target` des lésions malignes."""
    positives = np.sort(scores[labels == 1])
    return float(positives[int(np.floor((1 - target) * len(positives)))])


def evaluate(labels, scores, melanoma, seed: int = 0) -> dict:
    labels, scores, melanoma = (np.asarray(x) for x in (labels, scores, melanoma))
    rng = np.random.default_rng(seed)
    resampled = []
    for _ in range(1000):
        index = rng.integers(0, len(labels), len(labels))
        if 0 < labels[index].sum() < len(index):
            resampled.append(auc(labels[index], scores[index]))
    low, high = np.percentile(resampled, [2.5, 97.5])

    threshold = threshold_at_sensitivity(labels, scores, TARGET_SENSITIVITY)
    flagged = scores >= threshold
    is_melanoma = melanoma == 1
    return {
        "images": int(len(labels)),
        "malignant": int(labels.sum()),
        "melanoma": int(is_melanoma.sum()),
        "auc": round(auc(labels, scores), 3),
        "auc_ci95": [round(float(low), 3), round(float(high), 3)],
        "threshold_at_95_sensitivity": round(threshold, 4),
        "specificity_at_95_sensitivity": round(float((~flagged[labels == 0]).mean()), 3),
        # Le modèle distingue-t-il un mélanome d'une lésion bénigne ? (les autres
        # lésions malignes sont laissées de côté)
        "melanoma_vs_benign_auc": (
            round(auc(labels[is_melanoma | (labels == 0)], scores[is_melanoma | (labels == 0)]), 3)
            if is_melanoma.any() else None
        ),
        "melanoma_sensitivity_at_that_threshold": (
            round(float(flagged[is_melanoma].mean()), 3) if is_melanoma.any() else None
        ),
        "melanoma_sensitivity_at_0.5": (
            round(float((scores[is_melanoma] >= 0.5).mean()), 3) if is_melanoma.any() else None
        ),
    }


def save(model: str, dataset: str, split: str, report: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{model}__{dataset}__{split}.json"
    path.write_text(json.dumps({"model": model, "dataset": dataset, "split": split, **report}, indent=2))
    return path
