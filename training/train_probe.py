"""Entraîne un classifieur malin / bénin sur les embeddings Derm Foundation.

Le classifieur de l'application est une régression logistique : rapide, et suffisante
pour savoir si les embeddings portent l'information. XGBoost et un petit réseau de
neurones (MLP) peuvent être entraînés à l'identique pour comparaison : mêmes données,
même choix des hyperparamètres sur la partie validation, mêmes mesures. La partie test
ne sert qu'à la mesure finale.

    python train_probe.py pad_ufes_20 [isic2020 ...] [--model logreg|xgboost|mlp]
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import metrics

DATA = Path(__file__).parent / "data"
EMBEDDINGS = DATA / "embeddings" / "derm_foundation"
MODEL_NAME = "derm-foundation-logreg"

# Préfixe des fichiers de résultats et de modèles, par classifieur.
NAMES = {
    "logreg": MODEL_NAME,
    "xgboost": "derm-foundation-xgboost",
    "mlp": "derm-foundation-mlp",
}

# Hyperparamètres essayés sur la partie validation.
CANDIDATES = {
    "logreg": [{"C": c} for c in (0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0)],
    "xgboost": [
        {"max_depth": depth, "n_estimators": trees} for depth in (3, 6) for trees in (300, 1000)
    ],
    "mlp": [
        {"hidden_layer_sizes": layers, "alpha": alpha}
        for layers in ((128,), (256, 64))
        for alpha in (0.01, 0.1, 1.0)
    ],
}


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


def fit(parts: dict, classifier: str, **params):
    x, y = parts["train"]["x"], parts["train"]["y"]
    if classifier == "logreg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(**params, class_weight="balanced", max_iter=2000),
        )
        return model.fit(x, y)

    # XGBoost et le MLP n'ont pas de class_weight : des poids par image donnent le même
    # rééquilibrage que class_weight="balanced".
    weights = compute_sample_weight("balanced", y)
    if classifier == "xgboost":
        # Les arbres n'ont pas besoin de normalisation. Chaque arbre ne voit qu'un tiers
        # des 6 144 dimensions : plus rapide, et moins de surapprentissage.
        model = XGBClassifier(
            **params, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
            tree_method="hist", n_jobs=-1, random_state=0,
        )
        return model.fit(x, y, sample_weight=weights)

    # L'arrêt précoce met de côté 10 % de la partie entraînement, tirés au hasard.
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(**params, early_stopping=True, max_iter=300, random_state=0),
    )
    return model.fit(x, y, mlpclassifier__sample_weight=weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--model", choices=NAMES, default="logreg")
    args = parser.parse_args()
    parts = load(args.datasets)
    print({split: len(part["y"]) for split, part in parts.items()})

    # Moyenne des AUC par jeu : une AUC calculée sur les jeux mélangés récompenserait le
    # simple fait de reconnaître la source (78 % de lésions malignes dans PAD-UFES-20,
    # 2 % dans ISIC 2020) au lieu de la lésion.
    val = parts["val"]
    candidates = []
    for params in CANDIDATES[args.model]:
        scores = fit(parts, classifier=args.model, **params).predict_proba(val["x"])[:, 1]
        per_dataset = [
            metrics.auc(val["y"][val["dataset"] == d], scores[val["dataset"] == d]) for d in args.datasets
        ]
        candidates.append((float(np.mean(per_dataset)), params))
        detail = ", ".join(f"{d} {a:.3f}" for d, a in zip(args.datasets, per_dataset))
        print(f"{params} AUC validation = {candidates[-1][0]:.3f} ({detail})", flush=True)
    best = max(candidates, key=lambda candidate: candidate[0])[1]
    model = fit(parts, classifier=args.model, **best)

    # Un nom par combinaison de jeux d'entraînement, pour que les résultats de l'une
    # n'écrasent pas ceux d'une autre.
    name = f"{NAMES[args.model]}_{'+'.join(args.datasets)}"
    test = parts["test"]
    scores = model.predict_proba(test["x"])[:, 1]
    for dataset in args.datasets:
        mask = test["dataset"] == dataset
        report = metrics.evaluate(test["y"][mask], scores[mask], test["melanoma"][mask])
        report = {"trained_on": args.datasets, **best, **report}
        print(metrics.save(name, dataset, "test", report).read_text())

    out = DATA / "models"
    out.mkdir(exist_ok=True)
    with open(out / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
