"""Cherche un raccourci dans un jeu de données : peut-on deviner l'étiquette sans
regarder la lésion, à partir de la seule taille de l'image ou du poids du fichier ?

Une AUC proche de 0,5 est rassurante. Une AUC élevée signifie que les images malignes
et bénignes n'ont pas été produites de la même façon, et qu'un modèle peut tricher.

    python check_shortcuts.py pad_ufes_20
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import metrics

DATA = Path(__file__).parent / "data"


def main() -> None:
    dataset = sys.argv[1]
    with open(DATA / "manifest.csv", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["dataset"] == dataset]

    labels, pixels, file_sizes = [], [], []
    by_diagnosis = defaultdict(list)
    for row in rows:
        path = DATA / row["path"]
        with Image.open(path) as image:
            width, height = image.size
        labels.append(int(row["malignant"]))
        pixels.append(width * height)
        file_sizes.append(path.stat().st_size)
        by_diagnosis[row["diagnosis"]].append(max(width, height))

    labels = np.array(labels)
    for name, values in (("nombre de pixels", pixels), ("poids du fichier", file_sizes)):
        score = metrics.auc(labels, np.array(values, dtype=float))
        print(f"AUC en ne regardant que le {name} : {max(score, 1 - score):.3f}")
    print(f"\n{'diagnostic':<32} {'images':>6} {'côté médian':>12} {'≤ 300 px':>9}")
    for diagnosis, sides in sorted(by_diagnosis.items()):
        sides = np.array(sides)
        print(f"{diagnosis:<32} {len(sides):>6} {int(np.median(sides)):>12} {(sides <= 300).mean():>9.0%}")


if __name__ == "__main__":
    main()
