"""Mesure un modèle de l'application (variable SKIN_MODEL) sur un jeu de données.

Se lance dans l'image de l'application ; pour l'ancien modèle, la construire avec
--build-arg EXTRAS=ham10000 :

    docker run --rm -e PYTHONPATH=/app -e SKIN_MODEL=ham10000-efficientnet-b0 -v "${PWD}/training:/app/training" \
        skindetector:dev python training/evaluate_baseline.py pad_ufes_20
"""

import argparse
import csv
from pathlib import Path

from PIL import Image

import metrics
from app.skin import SKIN_MODEL, get_classifier

DATA = Path(__file__).parent / "data"
SOURCES = {"pad_ufes_20": "smartphone", "isic2020": "dermatoscope"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--split", default="test", help="train, val, test ou all")
    args = parser.parse_args()

    with open(DATA / "manifest.csv", encoding="utf-8") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row["dataset"] == args.dataset and args.split in ("all", row["split"])
        ]
    missing = [row["path"] for row in rows if not (DATA / row["path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images absentes, par exemple {missing[0]}")

    classifier = get_classifier()
    scores = []
    for row in rows:
        with Image.open(DATA / row["path"]) as image:
            scores.append(classifier.analyze(image, SOURCES[args.dataset]).malignancy_score)

    report = metrics.evaluate(
        [int(row["malignant"]) for row in rows],
        scores,
        [int(row["melanoma"]) for row in rows],
    )
    path = metrics.save(SKIN_MODEL, args.dataset, args.split, report)
    print(path.read_text())


if __name__ == "__main__":
    main()
