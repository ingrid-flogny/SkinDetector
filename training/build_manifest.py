"""Réunit les jeux de données dans training/data/manifest.csv, avec une étiquette
binaire et un découpage entraînement / validation / test.

Étiquette `malignant` : 1 pour les lésions malignes et, par choix du projet, pour les
kératoses actiniques (précancéreuses, classées « Indeterminate » dans l'archive ISIC).

Le découpage se fait par patient : toutes les images d'un patient vont du même côté,
sinon le test serait faussé. Il dépend uniquement de l'identifiant du patient, donc il
ne change pas quand on relance le script ou qu'on ajoute des données.
"""

import csv
import hashlib
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"
FIELDS = ["dataset", "image_id", "path", "patient_id", "diagnosis", "malignant", "melanoma", "split"]


def split_of(patient_id: str) -> str:
    bucket = int(hashlib.sha256(patient_id.encode()).hexdigest(), 16) % 100
    return "test" if bucket < 20 else "val" if bucket < 30 else "train"


def pad_ufes_20() -> list[dict]:
    with open(DATA / "pad_ufes_20" / "metadata.csv", encoding="utf-8") as f:
        return [
            {
                "dataset": "pad_ufes_20",
                "image_id": row["isic_id"],
                "path": f"pad_ufes_20/images/{row['isic_id']}.jpg",
                "patient_id": row["patient_id"],
                "diagnosis": row["diagnosis_3"],
                "malignant": int(row["diagnosis_1"] in ("Malignant", "Indeterminate")),
                "melanoma": int(row["diagnosis_3"].startswith("Melanoma")),
            }
            for row in csv.DictReader(f)
        ]


def isic2020() -> list[dict]:
    with open(DATA / "isic2020" / "ISIC_2020_Training_GroundTruth_v2.csv", encoding="utf-8") as f:
        return [
            {
                "dataset": "isic2020",
                "image_id": row["image_name"],
                "path": f"isic2020/train/{row['image_name']}.jpg",
                "patient_id": row["patient_id"],
                "diagnosis": row["diagnosis"],
                "malignant": int(row["target"]),
                # Dans ISIC 2020, tous les cas malins sont des mélanomes.
                "melanoma": int(row["target"]),
            }
            for row in csv.DictReader(f)
        ]


def main() -> None:
    rows = pad_ufes_20() + isic2020()
    for row in rows:
        row["split"] = split_of(row["patient_id"])
    with open(DATA / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((r["dataset"], r["split"]) for r in rows)
    malignant = Counter((r["dataset"], r["split"]) for r in rows if r["malignant"])
    melanoma = Counter((r["dataset"], r["split"]) for r in rows if r["melanoma"])
    print(f"{'jeu':<12} {'partie':<6} {'images':>7} {'malin':>6} {'mélanome':>9}")
    for key in sorted(counts):
        print(f"{key[0]:<12} {key[1]:<6} {counts[key]:>7} {malignant[key]:>6} {melanoma[key]:>9}")


if __name__ == "__main__":
    main()
