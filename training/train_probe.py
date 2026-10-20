"""Entraîne un classifieur malin / bénin sur les embeddings Derm Foundation.

Le classifieur est une régression logistique : rapide, et suffisante pour savoir si
les embeddings portent l'information. Sa régularisation est choisie sur la partie
validation ; la partie test ne sert qu'à la mesure finale.

    python train_probe.py pad_ufes_20 [isic2020 ...]
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import metrics

DATA = Path(__file__).parent / "data"
EMBEDDINGS = DATA / "embeddings" / "derm_foundation"
MODEL_NAME = "derm-foundation-logreg"


def load(datasets: list[str]) -> dict[str, dict]:
    """Par partie (train / val / test) : embeddings, étiquettes et jeu d'origine."""
    vectors = {}
    for dataset in datasets:
        saved = np.load(EMBEDDINGS / f"{dataset}.npz")
        vectors.update(zip(saved["image_id"], saved["embedding"]))
    with open(DATA / "manifest.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] in datasets and r["image_id"] in vectors]
    return {
        split: {
            "x": np.stack([vectors[r["image_id"]] for r in rows if r["split"] == split]),
            "y": np.array([int(r["malignant"]) for r in rows if r["split"] == split]),
            "melanoma": np.array([int(r["melanoma"]) for r in rows if r["split"] == split]),
            "dataset": np.array([r["dataset"] for r in rows if r["split"] == split]),
        }
        for split in ("train", "val", "test")
    }


def fit(parts: dict, c: float):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=2000),
    )
    return model.fit(parts["train"]["x"], parts["train"]["y"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    parts = load(args.datasets)
    print({split: len(part["y"]) for split, part in parts.items()})

    # Moyenne des AUC par jeu : une AUC calculée sur les jeux mélangés récompenserait le
    # simple fait de reconnaître la source (78 % de lésions malignes dans PAD-UFES-20,
    # 2 % dans ISIC 2020) au lieu de la lésion.
    val = parts["val"]
    candidates = {}
    for c in (0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0):
        scores = fit(parts, c).predict_proba(val["x"])[:, 1]
        per_dataset = [
            metrics.auc(val["y"][val["dataset"] == d], scores[val["dataset"] == d]) for d in args.datasets
        ]
        candidates[c] = float(np.mean(per_dataset))
        detail = ", ".join(f"{d} {a:.3f}" for d, a in zip(args.datasets, per_dataset))
        print(f"C={c:<7} AUC validation = {candidates[c]:.3f} ({detail})", flush=True)
    best = max(candidates, key=candidates.get)
    model = fit(parts, best)

    # Un nom par combinaison de jeux d'entraînement, pour que les résultats de l'une
    # n'écrasent pas ceux d'une autre.
    name = f"{MODEL_NAME}_{'+'.join(args.datasets)}"
    test = parts["test"]
    scores = model.predict_proba(test["x"])[:, 1]
    for dataset in args.datasets:
        mask = test["dataset"] == dataset
        report = metrics.evaluate(test["y"][mask], scores[mask], test["melanoma"][mask])
        report = {"trained_on": args.datasets, "C": best, **report}
        print(metrics.save(name, dataset, "test", report).read_text())

    out = DATA / "models"
    out.mkdir(exist_ok=True)
    with open(out / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
